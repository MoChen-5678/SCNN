"""
fortran_interface.py — Core-1204 RHF Fortran 引擎的 Python 封装

将 pinn_wrapper.so 的底层接口封装为面向 PINN 的 RHFEngine 类。
所有数组通过 numpy 零拷贝传递（Fortran 使用连续内存布局）。

用法:
    from RHF_solver.fortran_interface import RHFEngine

    engine = RHFEngine(model_id=0)  # PKA1
    engine.initialize()
    engine.scf_iterate(max_iter=50)
    E = engine.get_energy()
    V = engine.compute_potentials(xmix=0.5)
"""

import os
import sys
import time
import numpy as np


# ── 自动定位 .so 文件 ────────────────────────────────────────
def _find_so_path():
    """查找编译好的 pinn_wrapper 共享库路径"""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Core-1204'),
        os.path.join(os.environ.get('RHF_CORE_DIR', ''), ''),
        os.getcwd(),
    ]
    for d in candidates:
        if not d or not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.startswith('pinn_wrapper') and f.endswith('.so'):
                return os.path.join(d, f)
    raise FileNotFoundError(
        "Cannot find pinn_wrapper*.so.\n"
        "Run: cd Core-1204 && bash build_f90wrap.sh"
    )


# ── 延迟导入 Fortran 模块 ────────────────────────────────────
_pinn_wrapper = None

def _get_module():
    global _pinn_wrapper
    if _pinn_wrapper is None:
        so_path = _find_so_path()
        if os.path.dirname(so_path) not in sys.path:
            sys.path.insert(0, os.path.dirname(so_path))
        import pinn_wrapper as pw
        _pinn_wrapper = pw
    return _pinn_wrapper


class StateInfo:
    """单个量子态的信息"""
    __slots__ = ('kappa', 'name', 'deg', 'is_proton', 'e_ref')

    def __init__(self, kappa, name, deg, is_proton, e_ref):
        self.kappa = kappa       # κ 量子数 (-1, -2, +1, -3, ...)
        self.name = name         # 态标签 ("1s1/2", "1p3/2", ...)
        self.deg = deg           # 简并度 2j+1
        self.is_proton = is_proton  # True=质子, False=中子
        self.e_ref = e_ref       # 参考能量 [MeV] (Woods-Saxon 基底)

    def __repr__(self):
        tau = 'p' if self.is_proton else 'n'
        return f"<State {self.name} κ={self.kappa} τ={tau} E={self.e_ref:.3f}>"


class EnergyResult:
    """能量分解结果"""
    __slots__ = (
        'total', 'per_nuc',
        'kinetic', 'direct_sig', 'direct_ome', 'direct_rho',
        'direct_coulomb', 'direct_rtn', 'direct_rvt',
        'exchange_sig', 'exchange_ome', 'exchange_rho',
        'exchange_coulomb', 'exchange_pio', 'exchange_rtn', 'exchange_rvt',
        'rearrangement',
        'com_correction', 'pairing',
        'fermi_n', 'fermi_p',
        'rms_n', 'rms_p', 'rms_t',
        'charge_radius',
        'particle_n', 'particle_p', 'particle_t',
    )

    def __init__(self, **kwargs):
        for k in self.__slots__:
            setattr(self, k, kwargs.get(k, 0.0))

    @property
    def binding_energy(self):
        """结合能 per nucleon [MeV]"""
        return self.total / self.particle_t

    def summary(self, unit='MeV'):
        """格式化摘要"""
        lines = [
            f"=== RHF Energy Result ===",
            f"  Total:      {self.total:12.4f} {unit}",
            f"  Per nucleon:{self.binding_energy:12.4f} {unit}",
            f"  Kinetic:    {self.kinetic:12.4f} {unit}",
            f"  Direct:     {sum([self.direct_sig, self.direct_ome, self.direct_rho,",
            f"              self.direct_coulomb, self.direct_rtn, self.direct_rvt]):12.4f} {unit}",
            f"  Exchange:   {sum([self.exchange_sig, self.exchange_ome, self.exchange_rho,",
            f"              self.exchange_coulomb, self.exchange_pio, self.exchange_rtn, self.exchange_rvt]):12.4f} {unit}",
            f"  Rearr.:     {self.rearrangement:12.4f} {unit}",
            f"  CoM corr:   {self.com_correction:12.4f} {unit}",
            f"  Pairing:    {self.pairing:12.4f} {unit}",
            f"  Fermi(n):   {self.fermi_n:8.4f} {unit}  Fermi(p): {self.fermi_p:8.4f} {unit}",
            f"  RMS(n):{self.rms_n:6.3f} fm  RMS(p):{self.rms_p:6.3f} fm  RMS(t):{self.rms_t:6.3f} fm",
            f"  N={int(self.particle_n)} Z={int(self.particle_p)} A={int(self.particle_t)}",
        ]
        return '\n'.join(lines)


class PotentialResult:
    """势场结果"""
    __slots__ = ('r_grid', 'vps_n', 'vms_n', 'vtt_n', 'vps_p', 'vms_p', 'vtt_p',
                 'si_convergence')

    def __init__(self, **kwargs):
        for k in self.__slots__:
            setattr(self, k, kwargs.get(k, None))


class RHFEngine:
    """
    Core-1204 相对论 Hartree-Fock (RHF) 引擎封装类。

    通过 f2py 编译的 pinn_wrapper.so 调用 Fortran RHF 计算代码，
    为 PINN 提供精确的势场、能量和波函数。

    Parameters
    ----------
    model_id : int
        参数集 ID: 0=PKA1, 1=PKO1, 2=PKO2, 3=PKO3,
        4=DDME1, 5=DDME2, 6=PKDD, 7=TW99, 8=DDLZ1
    dat_path : str, optional
        rhf.dat 核素配置文件路径 (默认: Core-1204/rhf.dat)

    Example
    -------
    >>> engine = RHFEngine(0)  # PKA1, 16O
    >>> engine.initialize()
    >>> engine.scf_iterate(max_iter=50)
    >>> E = engine.get_energy()
    >>> print(E.summary())
    """

    # 物理常数
    HBC = 197.3269804   # ħc [MeV·fm]
    MSD = 201            # 径向网格点数
    IBX = 2              # 同位旋维度 (1=n, 2=p)
    NBX_MAX = 8          # 最大 kappa block 数
    NTX_MAX = 84        # 最大态数 (42×2)

    PARAM_SETS = {
        0: ('PKA1', 'DDRHF', 'Phys.Rev.C 76(2007)034314'),
        1: ('PKO1', 'DDRHF', 'Phys.Lett.B 640(2006)150'),
        2: ('PKO2', 'DDRHF', 'EPL 82(2008)12001'),
        3: ('PKO3', 'DDRHF', 'EPL 82(2008)12001'),
        4: ('DDME1', 'DDRMF', 'Phys.Rev.C66,024306(2002)'),
        5: ('DDME2', 'DDRMF', 'Phys.Rev.C71,024312(2005)'),
        6: ('PKDD',  'DDRMF', 'Phys.Rev.C69,0034319(2004)'),
        7: ('TW99',  'DDRMF', 'Nucl.Phys.A 656(1999)331'),
        8: ('DDLZ1', 'DDRMF', ''),
    }

    def __init__(self, model_id=0, dat_path=None):
        self.model_id = model_id
        self.dat_path = dat_path
        self._pw = None
        self._initialized = False

        # 网格信息 (initialize 后填充)
        self.npt = 0
        self.dr = 0.0
        self.r_grid = None

        # 当前 SCF 状态
        self._states = []       # list[StateInfo]
        self._current_energy = None  # EnergyResult

    def _get_pw(self):
        """懒加载 Fortran 模块"""
        if self._pw is None:
            self._pw = _get_module()
        return self._pw

    def initialize(self):
        """
        初始化 RHF 引擎。

        执行: Reader → Config → PreMedia → PrePotels → DBASE
        返回: npt, dr, r_grid
        """
        pw = self._get_pw()
        npt_out = np.array([0], dtype=np.int32)
        dr_out = np.array([0.0], dtype=np.float64)
        r_grid = np.zeros(self.MSD, dtype=np.float64)

        try:
            pw.init_rhf(np.int32(self.model_id), npt_out, dr_out, r_grid)
        except Exception as e:
            raise RuntimeError(f"init_rhf failed: {e}")

        self.npt = int(npt_out[0])
        self.dr = float(dr_out[0])
        self.r_grid = r_grid.copy()
        self._initialized = True

        # 获取态信息
        self._load_states()

        print(f"[RHF] Initialized: model={self.PARAM_SETS[self.model_id][0]}, "
              f"N={self.npt}, dr={self.dr:.3f} fm, "
              f"{len(self._states)} states")
        return self.npt, self.dr, self.r_grid

    def _load_states(self):
        """加载所有占据态的量子数信息"""
        pw = self._get_pw()
        n_states = np.array([0], dtype=np.int32)
        kappa_list = np.zeros(84, dtype=np.int32)
        name_list = np.zeros(84, dtype=('U', 8))
        deg_list = np.zeros(84, dtype=np.int32)
        is_proton_list = np.zeros(84, dtype=np.int32)
        e_ref_list = np.zeros(84, dtype=np.float64)

        try:
            pw.get_state_info(n_states, kappa_list, name_list,
                             deg_list, is_proton_list, e_ref_list)
        except Exception:
            pass  # get_state_info 可能未在当前 build 中导出

        ns = int(n_states[0]) if n_states[0] > 0 else 84
        self._states = []
        for i in range(ns):
            s = StateInfo(
                kappa=int(kappa_list[i]),
                name=str(name_list[i]).strip(),
                deg=int(deg_list[i]),
                is_proton=bool(is_proton_list[i]),
                e_ref=float(e_ref_list[i]),
            )
            self._states.append(s)

    @property
    def states(self):
        """返回所有态的量子数信息列表"""
        return self._states

    def set_wavefunctions(self, states_dict):
        """
        从 PINN 设置波函数到 Fortran 引擎。

        Parameters
        ----------
        states_dict : dict
            {state_name: {'G': array(MSD), 'F': array(MSD), 'kappa': int, ...}}
        """
        pw = self._get_pw()
        n = len(states_dict)
        G_all = np.zeros((n, self.MSD), dtype=np.float64)
        F_all = np.zeros((n, self.MSD), dtype=np.float64)
        kappa_arr = np.zeros(n, dtype=np.int32)
        occup_arr = np.ones(n, dtype=np.float64)
        it_arr = np.zeros(n, dtype=np.int32)

        for i, (name, data) in enumerate(states_dict.items()):
            G_all[i, :] = data['G'][:self.MSD]
            F_all[i, :] = data['F'][:self.MSD]
            kappa_arr[i] = data.get('kappa', -1)
            it_arr[i] = 2 if data.get('tau', 'n') == 'p' else 1

        try:
            pw.set_wavefunctions(np.int32(n), G_all, F_all, kappa_arr, occup_arr, it_arr)
        except AttributeError:
            pass  # 可能在简化版 build 中不可用

    def compute_density(self):
        """计算密度 (调用 Densit + Occup)"""
        pw = self._get_pw()
        try:
            pw.compute_density_fortran()
        except AttributeError:
            pass

    def compute_potentials(self, xmix=0.5):
        """
        计算势场: Hartree + Fock + Rearrange → V±S

        Returns
        -------
        PotentialResult with vps/vms/vtt arrays [MeV/fm]
        """
        pw = self._get_pw()
        si = np.array([0.0], dtype=np.float64)
        vps_n = np.zeros(self.MSD, dtype=np.float64)
        vms_n = np.zeros(self.MSD, dtype=np.float64)
        vtt_n = np.zeros(self.MSD, dtype=np.float64)
        vps_p = np.zeros(self.MSD, dtype=np.float64)
        vms_p = np.zeros(self.MSD, dtype=np.float64)
        vtt_p = np.zeros(self.MSD, dtype=np.float64)

        try:
            pw.compute_potentials_fortran(np.float64(xmix), si,
                                         vps_n, vms_n, vtt_n, vps_p, vms_p, vtt_p)
        except AttributeError:
            pass

        return PotentialResult(
            r_grid=self.r_grid.copy(),
            vps_n=vps_n.copy(), vms_n=vms_n.copy(), vtt_n=vtt_n.copy(),
            vps_p=vps_p.copy(), vms_p=vms_p.copy(), vtt_p=vtt_p.copy(),
            si_convergence=float(si[0]),
        )

    def get_fock_matrix(self, kappa=-1, it=1):
        """
        获取 Fock 交换势的非局域矩阵 (MSD × MSD) [MeV]

        Parameters
        ----------
        kappa : int
            κ 量子数
        it : int
            1=中子, 2=质子

        Returns
        -------
        XG, XF, YG, YF : ndarray (MSD, MSD) 或 None
        """
        pw = self._get_pw()
        XG = np.zeros((self.MSD, self.MSD), dtype=np.float64)
        XF = np.zeros((self.MSD, self.MSD), dtype=np.float64)
        YG = np.zeros((self.MSD, self.MSD), dtype=np.float64)
        YF = np.zeros((self.MSD, self.MSD), dtype=np.float64)
        try:
            pw.get_exchange_potentials(np.int32(kappa), np.int32(it),
                                      XG, XF, YG, YF)
            return XG, XF, YG, YF
        except AttributeError:
            return None, None, None, None

    def scf_step(self, xmix=0.5):
        """执行一步完整 SCF 迭代。返回收敛指标 si。"""
        pw = self._get_pw()
        si = np.array([0.0], dtype=np.float64)
        try:
            pw.scf_step(np.float64(xmix), si)
            return float(si[0])
        except AttributeError:
            return -1.0

    def scf_iterate(self, max_iter=100, xmix_start=0.5, xmix_min=0.05,
                     tol=1e-5):
        """
        执行多步 SCF 迭代直到收敛。

        Parameters
        ----------
        max_iter : int
            最大迭代次数
        xmix_start : float
            初始混合参数
        xmix_min : float
            最小混合参数 (线性衰减)
        tol : float
            收敛阈值 |ΔV| < tol → 收敛

        Returns
        -------
        converged : bool
        final_si : float
        n_iter : int
        """
        xmix = xmix_start
        converged = False
        final_si = 999.0

        t0 = time.time()
        for i in range(max_iter):
            si = self.scf_step(xmix)
            final_si = abs(si) if isinstance(si, (int, float)) else abs(float(si))
            if i % 10 == 0 or final_si < tol * 10:
                print(f"  SCF iter {i+1:3d}: si={final_si:.2e}, xmix={xmix:.3f}")
            if final_si < tol:
                converged = True
                break
            # 衰减混合
            xmix = max(xmix_min, xmix_start * (1 - i / max_iter))

        elapsed = time.time() - t0
        status = "✓ CONVERGED" if converged else "✗ NOT CONVERGED"
        print(f"[RHF] {status} after {i+1} iterations ({elapsed:.1f}s, final si={final_si:.2e})")
        return converged, final_si, i + 1

    def solve_dirac(self):
        """求解 Dirac 方程，更新所有态的波函数和能量。"""
        pw = self._get_pw()
        try:
            pw.solve_dirac_equations()
        except AttributeError:
            pass

    def get_energy(self):
        """
        计算并返回完整能量分解。

        Returns
        -------
        EnergyResult
        """
        pw = self._get_pw()
        # 大量输出参数 — 用默认值初始化
        args = {}
        names = [
            'E_kin','E_dsig','E_dome','E_drho','E_dcou','E_drtn','E_drvt',
            'E_esig','E_eome','E_erho','E_ecou','E_epio','E_ertn','E_ervt',
            'E_rear','E_com','E_pair','E_total','E_per_nuc',
            'fermi_n','fermi_p','rms_n','rms_p','rms_t',
            'charge_r','particle_n','particle_p','particle_t',
        ]
        for name in names:
            args[name] = np.array([0.0], dtype=np.float64)
        try:
            pw.get_energy_components(*[args[n] for n in names])
            result = EnergyResult(
                total=float(args['E_total'][0]),
                per_nuc=float(args['E_per_nuc'][0]),
                kinetic=float(args['E_kin'][0]),
                direct_sig=float(args['E_dsig'][0]),
                direct_ome=float(args['E_dome'][0]),
                direct_rho=float(args['E_drho'][0]),
                direct_coulomb=float(args['E_dcou'][0]),
                direct_rtn=float(args['E_drtn'][0]),
                direct_rvt=float(args['E_drvt'][0]),
                exchange_sig=float(args['E_esig'][0]),
                exchange_ome=float(args['E_eome'][0]),
                exchange_rho=float(args['E_erho'][0]),
                exchange_coulomb=float(args['E_ecou'][0]),
                exchange_pio=float(args['E_epio'][0]),
                exchange_rtn=float(args['E_ertn'][0]),
                exchange_rvt=float(args['E_ervt'][0]),
                rearrangement=float(args['E_rear'][0]),
                com_correction=float(args['E_com'][0]),
                pairing=float(args['E_pair'][0]),
                fermi_n=float(args['fermi_n'][0]),
                fermi_p=float(args['fermi_p'][0]),
                rms_n=float(args['rms_n'][0]),
                rms_p=float(args['rms_p'][0]),
                rms_t=float(args['rms_t'][0]),
                charge_radius=float(args['charge_r'][0]),
                particle_n=float(args['particle_n'][0]),
                particle_p=float(args['particle_p'][0]),
                particle_t=float(args['particle_t'][0]),
            )
            self._current_energy = result
            return result
        except AttributeError:
            return EnergyResult(total=-999.0)

    def get_wavefunction(self, kappa=-1, it=1, n_pr=1):
        """
        获取单个态的波函数。

        Returns
        -------
        G, F : ndarray (MSD,) — 大/小分量
        ee : float — 单粒子能量 [MeV]
        norm : float — 归一化因子
        """
        pw = self._get_pw()
        G = np.zeros(self.MSD, dtype=np.float64)
        F = np.zeros(self.MSD, dtype=np.float64)
        ee = np.array([0.0], dtype=np.float64)
        norm = np.array([0.0], dtype=np.float64)
        try:
            pw.get_wavefunction(np.int32(kappa), np.int32(it), np.int32(n_pr),
                              G, F, ee, norm)
            return G.copy(), F.copy(), float(ee[0]), float(norm[0])
        except AttributeError:
            return None, None, 0.0, 0.0

    def get_density(self):
        """获取当前密度分布。"""
        pw = self._get_pw()
        rho_s = np.zeros(self.MSD, dtype=np.float64)
        rho_v = np.zeros(self.MSD, dtype=np.float64)
        rho_3 = np.zeros(self.MSD, dtype=np.float64)
        rho_b = np.zeros(self.MSD, dtype=np.float64)
        cpl_gsig = np.zeros(self.MSD, dtype=np.float64)
        cpl_gome = np.zeros(self.MSD, dtype=np.float64)
        cpl_grho = np.zeros(self.MSD, dtype=np.float64)
        cpl_fpio = np.zeros(self.MSD, dtype=np.float64)
        cpl_grtn = np.zeros(self.MSD, dtype=np.float64)
        try:
            pw.get_density(rho_s, rho_v, rho_3, rho_b,
                          cpl_gsig, cpl_gome, cpl_grho, cpl_fpio, cpl_grtn)
            return {
                'rho_s': rho_s, 'rho_v': rho_v, 'rho_3': rho_3,
                'couplings': {'gsig': cpl_gsig, 'gome': cpl_gome,
                            'grho': cpl_grho, 'fpio': cpl_fpio, 'grtn': cpl_grtn},
            }
        except AttributeError:
            return {}

    def cleanup(self):
        """重置 SCF 状态。"""
        pw = self._get_pw()
        try:
            pw.cleanup_rhf()
        except AttributeError:
            pass
