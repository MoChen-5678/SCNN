# -*- coding: utf-8 -*-
"""
Fortran RHF 引擎 Python 包装层 (ctypes 版)

基于 ctypes 调用 Core-1204 Fortran RHF 代码 (librhf.so)。
PINN 只负责波函数 G(r)/F(r) 的神经网络优化，
物理量计算（密度、势场、能量、Dirac求解）全部委托给 Fortran。

用法:
    from physics.wrap_fortran import FortranRHFEngine

    engine = FortranRHFEngine()
    r_grid = engine.init(model_id=0)     # PKA1, 16O
    engine.set_wavefunctions(G_dict, F_dict)   # PINN 波函数注入
    si, pots = engine.potentials(xmix=0.5)    # Fortran 计算 V±S
    energy = engine.energy()                   # 16项能量泛函

依赖:
    Core-1204/librhf.so (ctypes 编译的 Fortran 库)
"""

import os
import sys
import numpy as np


class FortranRHFEngine:
    """
    ctypes 包装的 Fortran RHF 物理引擎。

    所有物理计算在 Fortran 端完成，PINN 仅负责波函数表示。
    """

    def __init__(self, fortran_dir='Core-1204'):
        # 找到 librhf.so
        if os.path.isabs(fortran_dir):
            self.lib_dir = fortran_dir
        else:
            self.lib_dir = os.path.abspath(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', fortran_dir)
            )

        self._so_path = os.path.join(self.lib_dir, 'librhf.so')
        self._fc = None       # FortranRHFCalculator instance
        self._initialized = False

        # 网格参数 (与 Core-1204 Define.f90 一致)
        self.MSD = 201
        self.dr = 0.10
        self.r_max = 20.0
        self.hbc = 197.3269804  # ħc [MeV·fm]

        # 当前态配置
        self.states_info = []

        # SCF 状态
        self._scf_iter = 0
        self._si = 1.0
        self._xmix = 0.50

    def _ensure(self):
        """延迟加载 FortranRHFCalculator"""
        if self._fc is not None:
            return

        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         '..', 'RHF_solver'))
        try:
            from fortran_interface_ctypes import FortranRHFCalculator
            self._fc = FortranRHFCalculator(ps="PKA1", Z=8, N=8, A=16)
        except ImportError:
            raise RuntimeError(
                f"Cannot import FortranRHFCalculator. "
                f"Ensure librhf.so exists at {self._so_path}"
            )

    def init(self, model_id=0):
        """
        初始化 RHF 引擎。

        Args:
            model_id: 参数集索引 (0=PKA1, 1=PKO1, ..., 8=DDLZ1)

        Returns:
            r_grid: (npt,) 径向网格 [fm]
        """
        self._ensure()

        ps_map = {0:"PKA1",1:"PKO1",2:"PKO2",3:"PKO3",
                  4:"DDME1",5:"DDME2",6:"PKDD",7:"TW99",8:"DDLZ1"}
        if model_id not in ps_map:
            raise ValueError(f"model_id={model_id} not in {list(ps_map.keys())}")

        self._fc.ps = ps_map[model_id]
        self._fc.idx = model_id

        st = self._fc.initialize()
        self._initialized = True

        # 提取网格信息
        self.npt = st.npt
        self.dr = st.dr
        self.r_grid = st.grid.copy()

        # 提取态配置
        self.states_info = []
        for orb in st.orbitals_n + st.orbitals_p:
            self.states_info.append({
                'kappa': orb.kappa,
                'name': orb.name,
                'degeneracy': orb.deg,
                'is_proton': orb.tau3 == -1,
                'E_ref': orb.energy,
            })

        print(f'[Fortran] init done: npt={self.npt} dr={self.dr:.3f}fm '
              f'orbits={len(st.orbitals_n)}n+{len(st.orbitals_p)}p '
              f'E_tot={st.E_total:.2f}MeV')
        return self.r_grid

    def set_wavefunctions(self, G_all, F_all, states_info=None):
        """
        将 PINN 波函数注入 Fortran 引擎。

        Args:
            G_all: dict — {state_name: (npt,) array}
            F_all: dict — {state_name: (npt,) array}
            states_info: 可选，用于匹配 kappa/it
        """
        self._ensure()

        if isinstance(G_all, dict):
            names = list(G_all.keys())
        else:
            names = list(F_all.keys())

        n_s = len(names)
        MSD = self.MSD
        kappas = np.zeros(n_s, dtype=np.int32)
        occups = np.ones(n_s, dtype=np.float64)
        itypes = np.zeros(n_s, dtype=np.int32)

        # 构建 Fortran 需要的数组格式: (MSD, n_wf)
        G_arr = np.zeros((MSD, n_s), dtype=np.float64, order='F')
        F_arr = np.zeros((MSD, n_s), dtype=np.float64, order='F')

        for i, name in enumerate(names):
            g = np.asarray(G_all[name], dtype=np.float64).ravel()[:MSD]
            f = np.asarray(F_all[name], dtype=np.float64).ravel()[:MSD]
            G_arr[:len(g), i] = g
            F_arr[:len(f), i] = f

            # 查找态信息
            st = self._find_state(name, states_info)
            kappas[i] = st.get('kappa', -1)
            itypes[i] = 2 if st.get('is_proton', False) else 1
            occups[i] = 1.0

        # 调用 ctypes 接口
        from fortran_interface_ctypes import _lib, _Pfx
        set_fn_name = (_Pfx + "ddrhf_set_wf").encode()
        set_fn = _lib[set_fn_name]

        import ctypes as ct
        Ip = ct.POINTER(ct.c_int)
        Dp = ct.POINTER(ct.c_double)
        Fp_MSn = np.ctypeslib.ndpointer(dtype=np.float64, shape=(MSD, n_s),
                                         flags='F_CONTIGUOUS')

        set_fn.argtypes = [Ip, Ip, Dp, Ip, Fp_MSn, Fp_MSn]
        set_fn.restype = None

        set_fn(ct.c_int(n_s),
               kappas.ctypes.data_as(Ip),
               occups.ctypes.data_as(Dp),
               itypes.ctypes.data_as(Ip),
               G_arr,
               F_arr)

    def potentials(self, xmix=0.5):
        """
        计算自洽势场 (Hartree + Fock → V⁺/V⁻)。

        注意: 这里不调用 ddrhf_step（那会更新波函数）。
        而是用当前已设置的波函数重新计算密度和势场。

        实际策略: 调用 iterate() 让 Fortran 用当前波函数做一步迭代，
        返回 Potel 计算出的新势场。xmix 由 Fortran 内部管理。

        Returns:
            (si, pots_dict): 收敛指标和势场字典
        """
        self._ensure()

        if not self._initialized:
            raise RuntimeError("Call init() first")

        # 做一步迭代获取势场
        st = self._fc.iterate()
        si = st.convergence

        # 从 FortranStateResult 提取场量组装 V±S
        # Fortran 的 sigma/omega/rho/coulomb 是介子场，需要转换为 Dirac 势
        # V±S = ±(gsig*sigma + grho*rho_3 + gcoul*coulomb) ∓ (gome*omega)
        # 这里简化：直接返回 Fortran 内部的势场分量

        hbc = self.hbc
        sig = st.sigma / hbc   # MeV → fm⁻¹
        ome = st.omega / hbc
        rho = st.rho_field / hbc
        cou = st.coulomb / hbc

        gsig = 8.372672;  gome = 11.270457;  grho = 3.649857
        gcou = 1.0  # e²/(4π) in natural units ≈ 1 (实际 α·ħc)

        # 标量势 V⁺ = σ + ρ_3 + Coulomb (质子)
        vps_n = (gsig * sig + grho * rho)          # 中子无库仑
        vps_p = (gsig * sig + grho * rho + cou)    # 质子有库仑

        # 矢量势 V⁻ = -ω
        vms_n = -gome * ome
        vms_p = -gome * ome

        # 张量势 (简化为0, 实际有 RTN/RVT 分量)
        vtt_n = np.zeros_like(sig)
        vtt_p = np.zeros_like(sig)

        self._si = si
        pots = {
            'V_ps_n': vps_n.astype(np.float32),
            'V_ms_n': vms_n.astype(np.float32),
            'V_tt_n': vtt_n.astype(np.float32),
            'V_ps_p': vps_p.astype(np.float32),
            'V_ms_p': vms_p.astype(np.float32),
            'V_tt_p': vtt_p.astype(np.float32),
        }
        return si, pots

    def energy(self):
        """
        获取完整能量泛函。

        Returns:
            dict: E_total, E_per_A, E_kinetic, 各分量...
        """
        self._ensure()
        if not self._initialized:
            return {}

        st = self._fc._build_result()  # 重新提取最新状态
        return {
            'E_total':   st.E_total,
            'E_per_A':   st.E_per_A,
            'E_kinetic': st.E_kinetic,
            'E_direct':  st.E_direct,
            'E_exchange':st.E_exchange,
            'E_rearrange':st.E_rearrange,
        }

    def get_wavefunction(self, kappa, it, n_pr=1):
        """获取单个态的波函数 (从 Fortran 当前状态)"""
        self._ensure()
        if not self._initialized:
            return {'G': np.zeros(self.MSD), 'F': np.zeros(self.MSD), 'energy': 0}

        it_val = 1 if it in ('n', 1) else 2
        wf_dict = st.wavefunctions_n if it_val == 1 else st.wavefunctions_p

        # 按 kappa 查找
        for name, wf in wf_dict.items():
            orb_list = self.states_info
            for s in orb_list:
                if s['name'] == name and s['kappa'] == kappa:
                    return {**wf, 'energy': s.get('E_ref', 0)}

        return {'G': np.zeros(self.MSD), 'F': np.zeros(self.MSD), 'energy': 0}

    def scf_step(self, xmix=0.5):
        """执行一步完整 SCF 迭代"""
        self._ensure()
        if not self._initialized:
            return 1.0

        st = self._fc.iterate()
        self._si = st.convergence
        self._scf_iter += 1
        return self._si

    @property
    def scf_status(self):
        return {
            'iteration': self._scf_iter,
            'si': self._si,
            'xmix': self._xmix,
            'converged': self._si < 1e-5,
        }

    # ════════ 内部辅助 ════════

    def _find_state(self, name, states_info=None):
        """查找态信息"""
        for s in self.states_info:
            if s.get('name') == name:
                return s
        if states_info:
            for s in states_info:
                if isinstance(s, (tuple, list)) and len(s) > 0 and s[0] == name:
                    return {'kappa': s[1], 'name': s[0], 'degeneracy': s[3],
                            'is_proton': s[5]}
                if isinstance(s, dict) and s.get('name') == name:
                    return s
        return self._parse_state_name(name)

    @staticmethod
    def _parse_state_name(name):
        clean = name.replace('n-', '').replace('p-', '')
        known = {
            's1/2': -1, 'p3/2': -2, 'd5/2': -3, 'f7/2': -4, 'g9/2': -5,
            'p1/2': +1, 'd3/2': +2, 'f5/2': +3, 'g7/2': +4,
        }
        for key, val in known.items():
            if key in clean:
                return {'kappa': val, 'name': name, 'degeneracy': 0,
                        'is_proton': name.startswith('p')}
        return {'kappa': -1, 'name': name, 'degeneracy': 0, 'is_proton': False}


def get_engine(fortran_dir='Core-1204', auto_init=True, model_id=0):
    """便捷工厂函数"""
    engine = FortranRHFEngine(fortran_dir=fortran_dir)
    if auto_init:
        engine.init(model_id=model_id)
    return engine
