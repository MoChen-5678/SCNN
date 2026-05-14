#!/usr/bin/env python3
"""
Dirac方程: 矩阵对角化 vs PINN求解
====================================
PINN 使用 Shooting POT 文件提供完整势场 (含Fock交换势).
边界条件嵌入网络结构 (r^{l+1} * Re[e^{i k c r}], c ∈ [0.95, 1.05]).
Rayleigh商从波函数计算物理能量.

数据来源:
  1. Shooting POT文件 (--pot-file, 必需)
  2. WAV CSV文件 (plusPINN/wav_{G|F}_{p|n}.csv, 用于参考波函数对比)
  3. LEV文件 (--wav-dir/../LEV, 用于参考能量)

支持激发态自动求解 (迁移学习 + 正交归一化):
  - 求解2s1/2时自动先解基态
  - 基态模型作为迁移学习起点

用法:
    python dirac_matrix_vs_pinn.py --state 1s1/2 --tau n \
        --pot-file results/16O/POT/O16_state001_POT.it001.final000

    # 激发态 (全自动)
    python dirac_matrix_vs_pinn.py --state 2s1/2 --tau n \
        --pot-file results/16O/POT/O16_state002_POT.it001.final000
"""

import os, sys, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# PINN 项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import HBAR_C, DR, NPT, R_GRID, M_NUCLEON
from model import DiracNet
from pde_residuals import compute_dirac_residual, clear_fd_cache, apply_nonlocal_kernels, has_nonlocal_kernels
from boundary_conditions import compute_total_boundary_loss, count_nodes, get_angular_momenta, compute_orthonormal_loss

# -- 势场加载 ------------------------------------------------

def load_shooting_potentials(pot_file, target_r_grid):
    """
    从 Shooting 方法的 POT 文件读取势场 (直接Hartree + 二维Fock交换核).
    
    新 POT 文件先给局部列 r, vps, vms, vtt，然后以字段三元组
    j, i, value 写出 XG/XF/YG/YF 四个二维非局域核。二维核已经
    按 Simpson 数值求积权重配置好，后续只执行 K @ ψ，不再额外
    乘积分权重。局部势和核均按 fm^-1 读取并转为 MeV。
    """
    import re

    E_shooting = None
    npt_file = None
    local_rows = []
    kernel_rows = {'XG': [], 'XF': [], 'YG': [], 'YF': []}
    current_kernel = None

    with open(pot_file) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith('#'):
                m = re.search(r'Energy[\s=]+([-\d.]+)', stripped)
                if m:
                    E_shooting = float(m.group(1))
                m = re.search(r'npt=\s*(\d+)', stripped)
                if m:
                    npt_file = int(m.group(1))
                m = re.search(r'kernel:\s*(XG|XF|YG|YF)\(j,i\)', stripped)
                if m:
                    current_kernel = m.group(1)
                continue

            parts = stripped.split()
            if current_kernel is None:
                if len(parts) >= 4:
                    local_rows.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])
            elif len(parts) >= 3:
                j = int(parts[0]) - 1
                i = int(parts[1]) - 1
                value = float(parts[2])
                kernel_rows[current_kernel].append((j, i, value))

    if not local_rows:
        raise ValueError(f"No local potential rows found in {pot_file}")

    data = np.array(local_rows, dtype=np.float64)
    r_file = data[:, 0]
    vps_fm1 = data[:, 1]
    vms_fm1 = data[:, 2]
    vtt_fm1 = data[:, 3]

    # 插值到 PINN 网格 (所有列都是 fm⁻¹, 统一乘 HBAR_C → MeV)
    def interp_and_convert(r_src, vals_fm1):
        f_interp = interp1d(r_src, vals_fm1, kind='cubic', fill_value='extrapolate')
        return torch.tensor(f_interp(target_r_grid) * HBAR_C, dtype=torch.float32)

    result = {
        'vps': interp_and_convert(r_file, vps_fm1),
        'vms': interp_and_convert(r_file, vms_fm1),
        'vtt': interp_and_convert(r_file, vtt_fm1),
        'XG': torch.zeros(len(target_r_grid), dtype=torch.float32),
        'XF': torch.zeros(len(target_r_grid), dtype=torch.float32),
        'YG': torch.zeros(len(target_r_grid), dtype=torch.float32),
        'YF': torch.zeros(len(target_r_grid), dtype=torch.float32),
        'r_grid': torch.tensor(target_r_grid, dtype=torch.float32),
        'E_shooting': E_shooting,
    }

    n_kernel = npt_file or len(r_file)
    if any(kernel_rows[name] for name in kernel_rows):
        if len(target_r_grid) != n_kernel:
            raise ValueError(
                f"Non-local kernel grid has n={n_kernel}, but target grid has n={len(target_r_grid)}"
            )
        for name, rows in kernel_rows.items():
            mat = np.zeros((n_kernel, n_kernel), dtype=np.float32)
            for j, i, value in rows:
                if 0 <= j < n_kernel and 0 <= i < n_kernel:
                    mat[j, i] = value * HBAR_C
            result[f'{name}_kernel'] = torch.tensor(mat, dtype=torch.float32)

    return result


def load_shooting_wavefunction(final_file, state_name):
    """
    从 Shooting 方法的 FINAL 文件读取指定态的波函数.
    FINAL 文件: 每行一个r, 42列对应42个态 (仅大分量G).
    
    参数:
        final_file: FINAL 文件路径
        state_name: 如 'P.1s.1/2', 'N.1p.3/2' 等
    
    返回: dict {'r': ndarray, 'G': ndarray, 'E': float} 或 None
    """
    # 读取态名列表
    states_line = None
    eigen_line = None
    with open(final_file) as f:
        for line in f:
            if line.startswith('# States:'):
                states_line = line
            elif line.startswith('# Eigenvalues'):
                eigen_line = line
    
    if states_line is None:
        raise ValueError(f"Cannot find '# States:' in {final_file}")
    
    # 解析态名
    state_names = states_line.split(':')[1].split()
    eigen_strs = eigen_line.split(':')[1].split()
    eigenvalues = [float(x) for x in eigen_strs]
    
    # 查找目标态
    target_idx = None
    for i, name in enumerate(state_names):
        if name.strip() == state_name.strip():
            target_idx = i
            break
    
    if target_idx is None:
        print(f"  Warning: '{state_name}' not found in FINAL file. "
              f"Available: {state_names[:5]}...")
        return None
    
    # 读取数据
    data = np.loadtxt(final_file)
    r_file = data[:, 0]
    G = data[:, 1 + target_idx]
    
    return {
        'r': r_file, 'G': G,
        'E': eigenvalues[target_idx],
        'state_name': state_names[target_idx],
    }


def load_wav_wavefunction(wav_dir, tau, state_name):
    """
    从 PKA1 WAV 文件直接读取指定态的完整波函数 (G和F分量).

    PKA1文件来源: {wav_dir}/Ca48.{G|F}-{P|N}.PKA1
      - G-P: 质子 G(大)分量, 正能量态
      - G-N: 中子 G(大)分量, 正能量态  
      - F-P: 质子 F(小)分量, 正能量态
      - F-N: 中子 F(小)分量, 正能量态

    文件格式:
      第1行: 占据数 (1.0=占据, 0.0=空)
      第2行: 列标题 (r + 各轨道名)
      第3行起: r值 + 各轨道波函数值 (Fortran E格式)

    参数:
        wav_dir: WAV目录路径 (如 results/48Ca/WAV)
        tau: 'p' 或 'n'
        state_name: 如 'P.1s.1/2', 'N.1p.3/2' 等

    返回: dict {'r': ndarray, 'G': ndarray, 'F': ndarray, 'E': float} 或 None
    """
    import numpy as np
    import re

    tau_upper = tau.upper()  # 'P' 或 'N'

    # 从 wav_dir 提取核素名和参数集
    # wav_dir = .../results/{PSET}/{NUCLEUS}/WAV/
    # WAV文件命名规则: {Symbol}{Mass}.G-N.{PSET}
    _nuc_dir = os.path.basename(os.path.dirname(wav_dir))  # e.g., '16O'
    _pset_dir = os.path.basename(os.path.dirname(os.path.dirname(wav_dir)))  # e.g., 'PKA1'

    try:
        from batch_solve_all_nuclei import NUCLEI_INFO
        if _nuc_dir in NUCLEI_INFO:
            _A, _Z, _sym = NUCLEI_INFO[_nuc_dir]
            nuc_prefix = f'{_sym}{_A}'   # 16O -> O16
        else:
            nuc_prefix = _nuc_dir
    except ImportError:
        nuc_prefix = _nuc_dir

    g_pka1 = os.path.join(wav_dir, f'{nuc_prefix}.G-{tau_upper}.{_pset_dir}')
    f_pka1 = os.path.join(wav_dir, f'{nuc_prefix}.F-{tau_upper}.{_pset_dir}')

    if not os.path.exists(g_pka1) or not os.path.exists(f_pka1):
        print(f"  Warning: PKA1 WAV files not found: {g_pka1} or {f_pka1}")
        return None

    def _parse_pka1(filepath):
        """解析单个PKA1 WAV文件, 返回 (r_array, data_dict)."""
        r_list, col_names, data_arrays = None, None, {}
        with open(filepath) as f:
            lines = f.readlines()

        # 第1行: 占据数(跳过)
        # 第2列: 列标题
        header = lines[1].split()
        col_names = header[1:]  # 去掉 'r'

        # 第3行起: 数据
        n_cols = len(header)
        raw_data = []
        r_list = []
        for line in lines[2:]:
            parts = line.split()
            if len(parts) < n_cols:
                continue
            r_list.append(float(parts[0]))
            raw_data.append([float(x) for x in parts[1:n_cols]])

        arr = np.array(raw_data, dtype=np.float64)
        r_arr = np.array(r_list, dtype=np.float64)

        for idx, name in enumerate(col_names):
            data_arrays[name] = arr[:, idx]

        return r_arr, col_names, data_arrays

    r_g, names_g, data_g = _parse_pka1(g_pka1)
    _,     names_f, data_f = _parse_pka1(f_pka1)

    # 查找目标态列名
    if state_name not in data_g:
        print(f"  Warning: '{state_name}' not found in PKA1. "
              f"Available: {list(names_g[:6])}...")
        return None

    G = data_g[state_name]
    F = data_f[state_name]

    # 从LEV文件读取能量 (48Ca用 Ca48.psp-{P|N}.PKA1)
    # wav_dir = results/48Ca/WAV → LEV在 results/48Ca/LEV/
    E_shooting = None
    lev_base = os.path.dirname(wav_dir)  # results/48Ca
    for lev_pattern in [f'Ca48.psp-{tau_upper}.PKA1']:
        lev_file = os.path.join(lev_base, 'LEV', lev_pattern)
        if not os.path.exists(lev_file):
            lev_file = os.path.join(wav_dir, lev_pattern)
        if os.path.exists(lev_file):
            with open(lev_file) as fl:
                for line in fl:
                    parts = line.split()
                    if len(parts) >= 5 and parts[1] == state_name:
                        E_shooting = float(parts[4])
                        break
            if E_shooting is not None:
                break

    return {
        'r': r_g, 'G': G, 'F': F,
        'E': E_shooting,
        'state_name': state_name,
    }


def scan_pot_files(pot_dir, iterations=('it001', 'it002')):
    """
    扫描 POT 目录, 返回所有 final000 文件对应的态信息.
    it001=中子, it002=质子
    
    返回: list of dict {'pot_file', 'state_name', 'label', 'tau', 'E', 'occ'}
    """
    import re
    states = []
    
    for iteration in iterations:
        for fname in sorted(os.listdir(pot_dir)):
            if f'{iteration}.final000' not in fname:
                continue
            fpath = os.path.join(pot_dir, fname)
            state_name = None
            E = None
            occ = None
            with open(fpath) as f:
                for line in f:
                    m = re.search(r'State:\s+(\S+),\s+Energy=\s+([-\d.]+)', line)
                    if m:
                        state_name = m.group(1)  # e.g. P.1s.1/2 or N.1s.1/2
                        E = float(m.group(2))
                    m2 = re.search(r'probability:\s+([-\d.]+)', line)
                    if m2:
                        occ = float(m2.group(1))
                    if state_name and occ is not None:
                        break
            
            if state_name:
                tau = 'p' if state_name.startswith('P.') else 'n'
                # 从 P.1s.1/2 提取 label: 1s1/2 (去掉P.前缀和点)
                label = state_name[2:]  # 1s.1/2
                label = re.sub(r'^(\d+)([a-z])\.(\d+/\d+)$', r'\1\2\3', label)  # 1s1/2
                states.append({
                    'pot_file': fpath,
                    'state_name': state_name,
                    'label': label,
                    'tau': tau,
                    'E': E,
                    'occ': occ,
                })
    
    # 按中子/质子分组, 每组按能量排序
    states.sort(key=lambda s: (s['tau'], s['E']))
    return states


def find_pot_for_state(pot_dir, state_label, tau):
    """在 POT 目录中查找指定 tau/state 的 POT 文件。"""
    if not pot_dir or not os.path.isdir(pot_dir):
        return None
    for state in scan_pot_files(pot_dir):
        if state['tau'] == tau and state['label'] == state_label:
            return state
    return None


# -- 物理计算能量 (Rayleigh 商) ----------------------------

def compute_energy_rayleigh(g, f, kappa, potentials, dr=DR, npt=NPT, device=None):
    """
    从 Dirac 方程直接提取能量 (积分形式, 避免逐点除法).
    
    Dirac 方程 (含Fock交换势):
      dG/dr = -(κ/r+Vtt+XG)G + (ε-Vms-XF)F
      dF/dr = +(κ/r+Vtt+YF)F - (ε-Vps-YG)G
    
    Rayleigh商推导:
      G方程×F积分: ε·∫F²dr = ∫F·[dG/dr + (κ/r+Vtt+XG)G + (Vms+XF)·F]dr
      F方程×G积分: ε·∫G²dr = ∫G·[(κ/r+Vtt+YF)F - dF/dr + (Vps+YG)·G]dr
    
    相加: ε·∫(G²+F²)dr = ∫[F·dG/dr - G·dF/dr + 2(κ/r+Vtt)GF 
                              + (Vps+YG)·G² + (Vms+XF)·F²
                              + XG·G·F + YF·F·G]dr
                         = ∫[F·dG/dr - G·dF/dr + 2(κ/r+Vtt)GF
                              + (Vps+YG)·G² + (Vms+XF)·F²
                              + (XG+YF)·GF]dr
    
    其中 ε = E/ħc (fm⁻¹), 所有势场 fm⁻¹.
    """
    if device is None:
        device = g.device
    
    has_batch = g.dim() == 2
    if not has_batch:
        g = g.unsqueeze(0)
        f = f.unsqueeze(0)
    
    B, N = g.shape
    hbc = HBAR_C
    
    # 势场 MeV → fm⁻¹
    vps = potentials.get('vps', torch.zeros_like(g)).to(device) / hbc
    vms = potentials.get('vms', torch.zeros_like(g)).to(device) / hbc
    vtt = potentials.get('vtt', torch.zeros_like(g)).to(device) / hbc
    use_nonlocal_kernels = has_nonlocal_kernels(potentials)
    if not use_nonlocal_kernels:
        XG = potentials.get('XG', torch.zeros_like(g)).to(device) / hbc
        XF = potentials.get('XF', torch.zeros_like(g)).to(device) / hbc
        YG = potentials.get('YG', torch.zeros_like(g)).to(device) / hbc
        YF = potentials.get('YF', torch.zeros_like(g)).to(device) / hbc
    
    # 导数 (5PADF)
    from pde_residuals import build_5padf_matrix, apply_fd_matrix, get_fd_directions
    g_dir, f_dir = get_fd_directions(kappa)
    D_g = build_5padf_matrix(N, dr, g_dir, device=device, dtype=g.dtype)
    D_f = build_5padf_matrix(N, dr, f_dir, device=device, dtype=g.dtype)
    dg_dr = apply_fd_matrix(g, D_g)
    df_dr = apply_fd_matrix(f, D_f)
    
    # r 和 κ/r
    r = torch.arange(N, device=device, dtype=g.dtype) * dr
    from config import R_SAFE_OFFSET
    r_safe = torch.clamp(r, min=R_SAFE_OFFSET)
    kap_vtt = float(kappa) / r_safe.unsqueeze(0) + vtt  # (κ/r + Vtt) fm⁻¹
    
    # 归一化
    norm_int = torch.trapz(g**2 + f**2, dim=-1, dx=dr).clamp(min=1e-30)
    
    numerator = (torch.trapz(f * dg_dr, dim=-1, dx=dr)
                 - torch.trapz(g * df_dr, dim=-1, dx=dr)
                 + torch.trapz(2.0 * kap_vtt * g * f, dim=-1, dx=dr)
                 + torch.trapz(vps * g**2, dim=-1, dx=dr)
                 + torch.trapz(vms * f**2, dim=-1, dx=dr))

    if use_nonlocal_kernels:
        x_int, y_int = apply_nonlocal_kernels(g, f, potentials, dr=dr, hbc=hbc)
        numerator = numerator + torch.trapz(f * x_int, dim=-1, dx=dr)
        numerator = numerator + torch.trapz(g * y_int, dim=-1, dx=dr)
    else:
        numerator = (numerator
                     + torch.trapz(YG * g**2, dim=-1, dx=dr)
                     + torch.trapz(XF * f**2, dim=-1, dx=dr)
                     + torch.trapz((XG + YF) * g * f, dim=-1, dx=dr))
    
    E_hc = numerator / norm_int  # fm⁻¹
    E_MeV = E_hc * hbc           # MeV (单粒子能量, 与矩阵解同约定)
    
    if not has_batch:
        return E_MeV.squeeze(0)
    return E_MeV


# -- PINN 求解器 --------------------------------------------

class DiracPINNSolver:
    """固定 kappa + nodes, 用 PINN 项目的 5PADF + DiracNet 求解.
    势场来自 RHF_solver, 边界条件来自 boundary_conditions.py.
    能量不拟合, 用物理方法 (Rayleigh商) 从波函数计算.
    支持正交归一化损失 (用于激发态求解).
    """

    def __init__(self, A, Z, tau, kappa, potentials=None, device=None, 
                 ref_wavefunctions=None, lambda_ortho=10.0,
                 boundary_options=None):
        self.A, self.Z, self.kappa = A, Z, kappa
        self.tau = tau
        self.is_proton = (tau == 'p')
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ref_wavefunctions = ref_wavefunctions or []  # 参考波函数列表
        self.lambda_ortho = lambda_ortho  # 正交损失权重
        self.boundary_options = boundary_options or {}

        if potentials is not None:
            # 外部提供的势场 (如 shooting POT 文件)
            self.potentials = {}
            for key in ['vps', 'vms', 'vtt', 'XG', 'XF', 'YG', 'YF']:
                val = potentials.get(key, torch.zeros(len(R_GRID), dtype=torch.float32))
                if isinstance(val, torch.Tensor):
                    self.potentials[key] = val.to(self.device)
                else:
                    self.potentials[key] = torch.tensor(val, dtype=torch.float32, device=self.device)
            for key in ['XG_kernel', 'XF_kernel', 'YG_kernel', 'YF_kernel']:
                if key not in potentials:
                    continue
                val = potentials[key]
                if isinstance(val, torch.Tensor):
                    self.potentials[key] = val.to(self.device)
                else:
                    self.potentials[key] = torch.tensor(val, dtype=torch.float32, device=self.device)
            if 'r_grid' in potentials:
                self.potentials['r_grid'] = potentials['r_grid']
            self.E_shooting = potentials.get('E_shooting', None)

        # 径向网格 (与 config 一致: dr=0.10, NPT=201)
        self.r_grid = torch.tensor(R_GRID, dtype=torch.float32, device=self.device)

        # numpy 版 (用于绘图对比)
        self.r_np = R_GRID.copy()
        self.VPS_np = self.potentials['vps'].cpu().numpy()
        self.VMS_np = self.potentials['vms'].cpu().numpy()

    def train(self, E_init_MeV=-60.0, target_nodes=0,
              max_epochs=50000, lr=1e-3, print_every=500,
              E_tol=1e-10, patience=500, lambda_ortho=None,
              load_model=None, w_pde=20.0, w_bc=20.0, w_ortho=10.0,
              w_phase=0.0,
              live_plot=True):
        """训练 PINN, 每步用Rayleigh商计算物理能量(不拟合E参数).
        节点数不对施加极大惩罚.
        支持正交归一化损失 (用于激发态求解).
        支持加载预训练模型作为起点 (迁移学习).

        ★ 核心改动: E 不再是可训练参数!
           - 前向传播得到 g,f → Rayleigh商算出 E_ray
           - E_ray 代入PDE残差 (detach, 不传梯度回E)
           - 只优化网络权重, 能量完全由波函数形状决定
        """
        if lambda_ortho is None:
            lambda_ortho = self.lambda_ortho
        
        clear_fd_cache()

        # 加载预训练模型作为起点
        if load_model and os.path.exists(load_model):
            print(f'\nLoading pre-trained model: {load_model}')
            net = DiracNet(
                n_hidden=128, n_layers=6, activation='swish',
                hard_normalize=True, init_energy=E_init_MeV,
                **self.boundary_options,
            )
            checkpoint = torch.load(load_model, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                net.load_state_dict(checkpoint['state_dict'], strict=False)  # strict=False: 兼容旧模型缺边界参数
                loaded_E = checkpoint.get('E')
                if loaded_E is not None:
                    print(f'  Loaded model E_Rayleigh={loaded_E:.4f} MeV')
            else:
                net.load_state_dict(checkpoint, strict=False)  # 同上
            net.to(self.device)
            print(f'  Pre-trained model loaded successfully')
        else:
            net = DiracNet(
                n_hidden=128, n_layers=6, activation='swish',
                hard_normalize=True, init_energy=E_init_MeV,
                **self.boundary_options,
            ).to(self.device)

        # ★ 内嵌正交化: 把参考波函数注入网络 (forward中自动Gram-Schmidt投影)
        if self.ref_wavefunctions:
            net.set_ref_wavefunctions(self.ref_wavefunctions)
            print(f'  Embedded ortho: {len(self.ref_wavefunctions)} ref state(s) in DiracNet.forward()')

        # ★ Rayleigh商参与梯度优化 (E由波函数形状决定)
        optimizer = torch.optim.Adam(
            list(net.parameters()),
            lr=lr, betas=(0.9, 0.999),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=lr * 0.01,
        )

        best_loss, best_state, best_boundary_energy = float('inf'), None, None
        history = []
        E_prev = float('inf')
        converge_count = 0

        # -- 实时动态绘图初始化 --
        if live_plot:
            import matplotlib.pyplot as plt
            plt.style.use('default')
            plt.ion()
            fig, axes = plt.subplots(3, 2, figsize=(12, 11))
            fig.suptitle('PINN Training Monitor', fontsize=13)
            ax_loss, ax_e = axes[0][0], axes[0][1]
            ax_comp, ax_param = axes[1][0], axes[1][1]
            ax_wf_g, ax_wf_f = axes[2][0], axes[2][1]

            # Loss
            ax_loss.set_title('Total Loss'); ax_loss.set_yscale('log'); ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('Loss')
            ln_loss, = ax_loss.plot([], [], 'b-', lw=1.5, label='Total Loss')
            ax_loss.legend(loc='upper right')

            # Energy
            ax_e.set_title('Rayleigh Energy'); ax_e.set_xlabel('Epoch')
            ax_e.set_ylabel('E (MeV)')
            ln_e, = ax_e.plot([], [], 'r-', lw=1.5)

            # Components
            ax_comp.set_title('Loss Components'); ax_comp.set_yscale('log'); ax_comp.set_xlabel('Epoch')
            ln_pde, = ax_comp.plot([], [], 'r-', lw=1.2, label='PDE')
            ln_bc,  = ax_comp.plot([], [], 'g-', lw=1.2, label='BC')
            ln_ort, = ax_comp.plot([], [], 'b-', lw=1.2, label='Ortho')
            ax_comp.legend(loc='upper right', fontsize=8)

            # Params
            ax_param.set_title('Boundary Params'); ax_param.set_xlabel('Epoch')
            ln_c, = ax_param.plot([], [], 'm-', lw=1.5, label='c')
            ax_param.legend(loc='upper right', fontsize=8)

            # Wavefunctions
            r_np = self.r_grid.cpu().numpy()
            ax_wf_g.set_title('Wavefunction G(r)'); ax_wf_g.set_xlabel('r (fm)')
            ax_wf_f.set_title('Wavefunction F(r)'); ax_wf_f.set_xlabel('r (fm)')
            ln_wfg, = ax_wf_g.plot([], [], 'b-', lw=1.5)
            ln_wff, = ax_wf_f.plot([], [], 'r-', lw=1.5)

            fig.tight_layout(rect=[0, 0, 1, 0.96])
            plot_update_interval = max(print_every // 3, 50)

            # 定期保存PNG (无GUI也能看)
            plot_save_path = os.path.join('outputs', 'training_live.png')
            os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)

        r_input = self.r_grid.unsqueeze(0)  # (1, N)
        boundary_energy = torch.tensor(E_init_MeV, dtype=torch.float32, device=self.device)

        for epoch in range(max_epochs):
            optimizer.zero_grad()

            g, f = net(r_input, kappa=self.kappa, dr=DR, boundary_energy=boundary_energy)

            # ★ Rayleigh商: 从当前波函数+势场计算物理能量 (全程可微!)
            #    梯度流: g,f → E_ray → PDE残差 → ∂L/∂(g,f) → 网络权重
            E_ray_hc = compute_energy_rayleigh(
                g, f, self.kappa, self.potentials,
                dr=DR, npt=NPT, device=self.device,
            )

            # 用 Rayleigh 能量代入 PDE 残差 (E_ray 参与梯度传播)
            loss_pde = compute_dirac_residual(
                g, f, E_ray_hc, self.kappa, self.potentials,
                dr=DR, npt=NPT, f_weight=3.0,
                return_components=False, device=self.device,
            )

            # 边界条件损失: 远端因子已嵌入网络结构 r^{l+1}·Re[e^{ikcr}]
            if w_bc > 0:
                bc_losses = compute_total_boundary_loss(
                    g, f, self.kappa, dr=DR,
                    w_norm=10.0,
                    w_R=5.0,
                    w_kin=0.1, w_node=5.0,
                    n_expected_nodes=target_nodes,
                )
                loss_bc = bc_losses['total']
            else:
                loss_bc = torch.tensor(0.0, device=self.device)

            # 正交归一化损失 (用于激发态求解)
            loss_ortho = torch.tensor(0.0, device=self.device)
            if lambda_ortho > 0 and self.ref_wavefunctions:
                loss_ortho = compute_orthonormal_loss(
                    g, f, self.ref_wavefunctions, dr=DR, weight=lambda_ortho
                )

            loss_phase = torch.tensor(0.0, device=self.device)
            if w_phase > 0:
                loss_phase = net.boundary_phase_loss(r_input, boundary_energy)

            # ★ 归一化约束 (对标RHF论文 L_conz = [G/max(G)-1]^2)
            G_max = g.abs().max().clamp(min=1e-10)
            loss_norm = ((g.abs() / G_max) - 1.0).pow(2).mean()

            loss_total = (w_pde * loss_pde + w_bc * loss_bc + w_ortho * loss_ortho
                          + w_phase * loss_phase + 0*loss_norm)
            loss_total.backward()

            torch.nn.utils.clip_grad_norm_(list(net.parameters()), 1.0)
            optimizer.step()
            scheduler.step()
            boundary_energy = E_ray_hc.detach()

            if loss_total.item() < best_loss:
                best_loss = loss_total.item()
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
                best_boundary_energy = boundary_energy.detach().clone()

            c_val = net.get_boundary_c().item()
            phi_val = net.get_boundary_phi().item()
            if epoch % print_every == 0 or epoch == max_epochs - 1:
                print(f"  Ep {epoch:5d}: L={loss_total.item():.2e}  "
                      f"pde={loss_pde.item():.2e} bc={loss_bc.item():.2e} "
                      f"ortho={loss_ortho.item():.2e} phase={loss_phase.item():.2e} "
                      f"norm={loss_norm.item():.2e}  "
                      f"E_ray={E_ray_hc.item():+.4f} c={c_val:.4f} phi={phi_val:+.4f}")

            # -- 动态绘图更新 --
            if live_plot and (epoch % plot_update_interval == 0 or epoch == max_epochs - 1):
                eps_list = [h['epoch'] for h in history]
                loss_list = [h['loss'] for h in history]
                e_list = [h['E_MeV'] for h in history]

                ln_loss.set_data(eps_list, loss_list)
                ax_loss.relim(); ax_loss.autoscale_view()

                ln_e.set_data(eps_list, e_list)
                ax_e.relim(); ax_e.autoscale_view()

                # components
                if not hasattr(self, '_comp_history'):
                    self._comp_history = {'pde': [], 'bc': [], 'ort': []}
                self._comp_history['pde'].append(loss_pde.item())
                self._comp_history['bc'].append(loss_bc.item())
                self._comp_history['ort'].append(loss_ortho.item())
                n_comp = len(self._comp_history['pde'])
                c_eps = list(range(n_comp))
                ln_pde.set_data(c_eps, self._comp_history['pde'])
                ln_bc.set_data(c_eps, self._comp_history['bc'])
                ln_ort.set_data(c_eps, self._comp_history['ort'])
                ax_comp.relim(); ax_comp.autoscale_view()

                # boundary c
                if not hasattr(self, '_param_history'):
                    self._param_history = {'c': []}
                self._param_history['c'].append(c_val)
                n_p = len(self._param_history['c'])
                p_eps = list(range(n_p))
                ln_c.set_data(p_eps, self._param_history['c'])
                ax_param.relim(); ax_param.autoscale_view()

                # wavefunctions
                g_np = g.detach().cpu().numpy().flatten()
                f_np = f.detach().cpu().numpy().flatten()
                ln_wfg.set_data(r_np, g_np)
                ln_wff.set_data(r_np, f_np)
                ax_wf_g.relim(); ax_wf_g.autoscale_view()
                ax_wf_f.relim(); ax_wf_f.autoscale_view()

                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.05)
                # 保存PNG供远程查看
                try:
                    fig.savefig(plot_save_path, dpi=100, bbox_inches='tight')
                except Exception:
                    pass

            history.append({
                'epoch': epoch, 'loss': loss_total.item(), 'E_MeV': E_ray_hc.item(),
                'loss_pde': loss_pde.item(), 'loss_bc': loss_bc.item(),
                'loss_ortho': loss_ortho.item(), 'loss_phase': loss_phase.item(),
                'boundary_c': c_val, 'boundary_phi': phi_val,
            })

            dE = abs(E_ray_hc.item() - E_prev)
            if dE < E_tol:
                converge_count += 1
                if converge_count >= patience:
                    print(f"  Converged at Ep {epoch}: |dE|={dE:.2e}")
                    break
            else:
                converge_count = 0
            E_prev = E_ray_hc.item()

        # 恢复最佳
        if best_state is not None:
            net.load_state_dict(best_state)
            if best_boundary_energy is not None:
                boundary_energy = best_boundary_energy

        self.net = net
        self.boundary_energy = boundary_energy.detach()

        # 最终用 Rayleigh 商算物理能量
        with torch.no_grad():
            g_final, f_final = net(
                r_input, kappa=self.kappa, dr=DR, boundary_energy=self.boundary_energy
            )
            E_rayleigh = compute_energy_rayleigh(
                g_final, f_final, self.kappa, self.potentials,
                dr=DR, npt=NPT, device=self.device,
            )
            E_final = E_rayleigh.item() if E_rayleigh.dim() == 0 else E_rayleigh[0].item()

        print(f"  E_Rayleigh={E_final:+.4f} MeV")

        # -- 关闭动态绘图 --
        if live_plot:
            plt.ioff()
            # 保存最终图
            save_path = 'outputs/training_curves.png'
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            fig.savefig(save_path, dpi=120, bbox_inches='tight')
            print(f'  Training curves saved to: {save_path}')
            plt.close(fig)

        return E_final, history

    @torch.no_grad()
    def get_wavefunction(self, r_eval=None):
        if r_eval is None:
            r_eval = self.r_np
        rt = torch.tensor(r_eval, dtype=torch.float32, device=self.device).unsqueeze(0)
        boundary_energy = getattr(self, 'boundary_energy', None)
        g, f = self.net(rt, kappa=self.kappa, dr=DR, boundary_energy=boundary_energy)
        g = g.squeeze(0).cpu().numpy()
        f = f.squeeze(0).cpu().numpy()
        return g, f

    @torch.no_grad()
    def save_wavefunction(self, filepath, r_eval=None, state_name=None):
        """
        保存当前求解的波函数到JSON文件。
        
        参数:
            filepath: 保存路径 (JSON格式)
            r_eval: 径向网格, 如果为None则使用默认网格
            state_name: 态名称, 用于标识
        """
        import json
        
        g, f = self.get_wavefunction(r_eval)
        r_grid = self.r_np if r_eval is None else r_eval
        
        data = {
            'state_name': state_name or f'{self.tau}_{self.kappa}',
            'kappa': self.kappa,
            'tau': self.tau,
            'A': self.A,
            'Z': self.Z,
            'r': r_grid.tolist(),
            'G': g.tolist(),
            'F': f.tolist(),
        }
        
        # 自动创建目录
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f'  Wavefunction saved to: {filepath}')

    def save_model(self, filepath, E_MeV=None):
        """保存训练好的模型参数（含能量值）."""
        if self.net is not None:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            E_val = float(E_MeV) if E_MeV is not None else None
            save_data = {
                'state_dict': self.net.state_dict(),
                'E': E_val,
            }
            torch.save(save_data, filepath)
            print(f'  Model saved to: {filepath}' + (f'  (E_Rayleigh={E_val:.4f} MeV)' if E_val else ''))
        else:
            print(f'  Warning: No trained model to save')


# -- 参考波函数加载 -----------------------------------------

def load_ref_wavefunctions(filepath, device=None):
    """
    加载参考波函数文件 (JSON格式), 用于正交归一化损失.
    
    支持三种格式:
      1. 单态 save_result 格式: {'G': [...], 'F': [...], ...}
      2. 多态列表格式: {'states': [{'G': [...], 'F': [...], ...}, ...]}
      3. 直接解析: 尝试解析为波函数列表
    
    返回:
        ref_wavefunctions: list of dict, 每个dict含 {'g': tensor, 'f': tensor, 'name': str}
    """
    import json
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Reference wavefunction file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    ref_wavefunctions = []
    
    # 格式1: 单态 save_result 格式
    if 'G' in data and 'F' in data:
        G = np.array(data['G'], dtype=np.float32)
        F = np.array(data['F'], dtype=np.float32)
        name = data.get('state_name', os.path.basename(filepath))
        
        g_tensor = torch.tensor(G, dtype=torch.float32)
        f_tensor = torch.tensor(F, dtype=torch.float32)
        
        if device is not None:
            g_tensor = g_tensor.to(device)
            f_tensor = f_tensor.to(device)
        
        ref_wavefunctions.append({
            'g': g_tensor,
            'f': f_tensor,
            'name': name,
        })
        print(f"  Loaded reference wavefunction: {name}")
    
    # 格式2: 多态列表格式
    elif 'states' in data:
        for i, state in enumerate(data['states']):
            if 'G' in state and 'F' in state:
                G = np.array(state['G'], dtype=np.float32)
                F = np.array(state['F'], dtype=np.float32)
                name = state.get('state_name', f'state_{i}')
                
                g_tensor = torch.tensor(G, dtype=torch.float32)
                f_tensor = torch.tensor(F, dtype=torch.float32)
                
                if device is not None:
                    g_tensor = g_tensor.to(device)
                    f_tensor = f_tensor.to(device)
                
                ref_wavefunctions.append({
                    'g': g_tensor,
                    'f': f_tensor,
                    'name': name,
                })
                print(f"  Loaded reference wavefunction: {name}")
    
    # 格式3: 直接是列表
    elif isinstance(data, list):
        for i, state in enumerate(data):
            if 'G' in state and 'F' in state:
                G = np.array(state['G'], dtype=np.float32)
                F = np.array(state['F'], dtype=np.float32)
                name = state.get('state_name', f'state_{i}')
                
                g_tensor = torch.tensor(G, dtype=torch.float32)
                f_tensor = torch.tensor(F, dtype=torch.float32)
                
                if device is not None:
                    g_tensor = g_tensor.to(device)
                    f_tensor = f_tensor.to(device)
                
                ref_wavefunctions.append({
                    'g': g_tensor,
                    'f': f_tensor,
                    'name': name,
                })
                print(f"  Loaded reference wavefunction: {name}")
    
    else:
        raise ValueError(f"Unknown reference wavefunction format in {filepath}")
    
    print(f"  Total loaded: {len(ref_wavefunctions)} reference wavefunction(s)")
    return ref_wavefunctions


# -- 绘图 -----------------------------------------------------

def plot_comparison(r_mat, state_info, G_mat, F_mat,
                    r_pinn, G_pinn, F_pinn,
                    E_mat, E_pinn, history, output_dir='outputs',
                    shooting_data=None, tau='p', state_name=None):
    os.makedirs(output_dir, exist_ok=True)
    label = state_info['label']
    tau_prefix = 'P' if tau == 'p' else 'N'
    # 文件名用Fortran命名 (如 N.1p.1/2 → N_1p1_2)
    if state_name is not None:
        safe_name = state_name.replace('.', '_').replace('/', '_')
    else:
        safe_name = f'{tau_prefix}_{label.replace("/", "_")}'

    # -- 插值到同一网格 --
    if len(r_mat) != len(r_pinn):
        G_pinn_i = np.interp(r_mat, r_pinn, G_pinn)
        F_pinn_i = np.interp(r_mat, r_pinn, F_pinn)
    else:
        G_pinn_i, F_pinn_i = G_pinn, F_pinn

    # -- Shooting 参考解 --
    G_shoot_i, F_shoot_i = None, None
    E_shoot = None
    if shooting_data is not None:
        E_shoot = shooting_data.get('E', None)
        if 'G' in shooting_data:
            r_shoot = shooting_data['r']
            G_shoot_i = np.interp(r_mat, r_shoot, shooting_data['G'])
            F_shoot_raw = shooting_data.get('F', None)
            if F_shoot_raw is not None:
                F_shoot_i = np.interp(r_mat, r_shoot, F_shoot_raw)

    # -- 决定参考基准 --
    has_shooting_wf = (G_shoot_i is not None)

    if has_shooting_wf:
        overlap = np.trapezoid(G_shoot_i * G_pinn_i, r_mat)
        if overlap < 0:
            G_pinn_i, F_pinn_i = -G_pinn_i, -F_pinn_i
        is_rayleigh = (shooting_data.get('source') == 'rayleigh')
        ref_label = 'WAV(Rayleigh)' if is_rayleigh else 'Shooting'
        title_E = f'E_{ref_label}={E_shoot:.5f}  E_PINN={E_pinn:.5f}  dE={abs(E_pinn-E_shoot):.4f} MeV'
        ref_G, ref_F = G_shoot_i, F_shoot_i
        ref_E = E_shoot
    else:
        if G_mat is not None:
            overlap = np.trapezoid(G_mat * G_pinn_i, r_mat)
            if overlap < 0:
                G_pinn_i, F_pinn_i = -G_pinn_i, -F_pinn_i
            title_E = f'E_Ref={E_mat:.5f}  E_PINN={E_pinn:.5f}  dE={abs(E_pinn-E_mat):.4f} MeV'
            ref_G, ref_F, ref_label = G_mat, F_mat, 'Ref'
            ref_E = E_mat
        else:
            title_E = f'E_Ref={E_mat:.5f}  E_PINN={E_pinn:.5f}  dE={abs(E_pinn-E_mat):.4f} MeV'
            ref_G, ref_F, ref_label = None, None, 'None'
            ref_E = E_mat

    # =====================================================================
    # 新布局: 2x2
    #   [0,0] G(r)+F(r) 合并图   [0,1] Loss 曲线
    #   [1,0] ΔG+ΔF 差异合并     [1,1] 能量曲线 + 局部放大inset
    # =====================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    kappa_str = f'(k={state_info.get("kappa", "?")})' if 'kappa' in state_info else ''
    fig.suptitle(
        f'{tau_prefix}.{label} {kappa_str}  |  {title_E}',
        fontweight='bold', fontsize=12,
    )

    # ====== [0,0] G + F 合并 ======
    ax_gf = axes[0, 0]
    ax_gf.plot(r_mat, ref_G, 'b-', lw=2, alpha=0.85, label=f'G {ref_label}')
    ax_gf.plot(r_mat, G_pinn_i, 'r--', lw=1.5, alpha=0.8, label='G PINN')
    if ref_F is not None:
        ax_gf.plot(r_mat, ref_F, 'g-', lw=2, alpha=0.85, label=f'F {ref_label}')
    ax_gf.plot(r_mat, F_pinn_i, color='magenta', ls='--', lw=1.5, alpha=0.8, label='F PINN')
    ax_gf.set_xlabel('r (fm)')
    ax_gf.set_ylabel('Wavefunction')
    ax_gf.legend(loc='upper right', fontsize=9, ncol=2)
    ax_gf.grid(alpha=0.3)
    ax_gf.set_xlim(0, 20)
    ax_gf.set_title('G(r) & F(r)', fontweight='bold')

    # ====== [0,1] Loss 曲线 ======
    ax_loss = axes[0, 1]
    if history:
        ep = [h['epoch'] for h in history]
        ls = [h['loss'] for h in history]
        ax_loss.semilogy(ep, ls, '#6a0dad', lw=1.0, alpha=0.85)
        ax_loss.fill_between(ep, ls, alpha=0.15, color='#6a0dad')
        # ★ 避免 semilogy 自动生成的 $\mathdefault{10^{-6}}$ 格式在多线程下崩溃
        from matplotlib.ticker import ScalarFormatter
        ax_loss.yaxis.set_major_formatter(ScalarFormatter())
        ax_loss.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 4))
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss (log scale)')
    ax_loss.set_title('Training Loss', fontweight='bold')
    ax_loss.grid(alpha=0.3, which='both')
    if history:
        final_l = ls[-1]
        min_l = min(ls)
        ax_loss.annotate(f'min={min_l:.1e}\nfinal={final_l:.1e}',
                         xy=(0.97, 0.95), xycoords='axes fraction',
                         ha='right', va='top', fontsize=9,
                         bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.8))

    # ====== [1,0] ΔG + ΔF 差异合并 ======
    ax_diff = axes[1, 0]
    dG = G_pinn_i - ref_G
    dF = (F_pinn_i - ref_F) if ref_F is not None else None
    ax_diff.plot(r_mat, dG, 'r-', lw=1.2, alpha=0.85, label=r'$\Delta$G')
    if dF is not None:
        ax_diff.plot(r_mat, dF, color='magenta', lw=1.2, alpha=0.85, label=r'$\Delta$F')
    ax_diff.axhline(0, color='k', lw=0.6, ls='-')
    ax_diff.set_xlabel('r (fm)')
    ax_diff.set_ylabel(r'Difference $\Delta$ = PINN $-$ Ref')
    ax_diff.legend(loc='upper right', fontsize=9)
    ax_diff.grid(alpha=0.3)
    ax_diff.set_xlim(0, 20)
    rms_dG = np.sqrt(np.mean(dG**2))
    rms_dF = np.sqrt(np.mean(dF**2)) if dF is not None else 0.0
    ax_diff.set_title(f'Wavefunction Difference  RMS(G)={rms_dG:.2e}  RMS(F)={rms_dF:.2e}',
                      fontweight='bold')

    # ====== [1,1] 能量曲线 + 局部放大inset ======
    ax_e = axes[1, 1]
    if history:
        ep = [h['epoch'] for h in history]
        Es = [h['E_MeV'] for h in history]

        # 主图: 全程能量
        ax_e.plot(ep, Es, '#ff6600', lw=1.2, alpha=0.9, label='PINN Energy')
        ax_e.axhline(ref_E, color='#0033cc', ls='--', lw=1.2, alpha=0.75,
                     label=f'{ref_label} = {ref_E:.4f}')
        ax_e.set_xlabel('Epoch')
        ax_e.set_ylabel('Energy (MeV)')
        ax_e.set_title('Energy Convergence', fontweight='bold')
        ax_e.legend(loc='upper right', fontsize=9)
        ax_e.grid(alpha=0.3)

        # --- 局部放大 inset (后35% epochs) ---
        n_total = len(ep)
        n_inlet_start = max(int(n_total * 0.65), 10)

        ax_ins = ax_e.inset_axes([0.52, 0.48, 0.44, 0.46])
        ep_s = ep[n_inlet_start:]
        Es_s = Es[n_inlet_start:]

        ax_ins.plot(ep_s, Es_s, '#ff6600', lw=1.5, alpha=0.95, label='PINN')
        ax_ins.axhline(ref_E, color='#0033cc', ls='--', lw=1.0, alpha=0.7)
        ax_ins.axhline(E_pinn, color='#cc0000', ls=':', lw=1.0, alpha=0.6,
                       label=f'Final={E_pinn:.5f}')

        final_dE = abs(Es[-1] - ref_E)
        y_min_ins = min(min(Es_s), ref_E, E_pinn)
        y_max_ins = max(max(Es_s), ref_E, E_pinn)
        y_margin = (y_max_ins - y_min_ins) * 0.25
        ax_ins.set_ylim(y_min_ins - y_margin, y_max_ins + y_margin)
        ax_ins.set_xlim(ep_s[0], ep_s[-1])

        ax_ins.set_title(f'Zoom (last {len(ep_s)} eps)\ndE={final_dE:.5f} MeV',
                         fontsize=8.5, fontweight='bold')
        ax_ins.tick_params(labelsize=8)
        ax_ins.grid(alpha=0.35, which='both')

        # 主图→inset 的连接线
        from mpl_toolkits.axes_grid1.inset_locator import mark_inset
        mark_inset(ax_e, ax_ins, loc1=2, loc2=4, fc="none", ec="gray", lw=0.8, ls='--')

    try:
        plt.tight_layout(rect=[0, 0, 1, 0.94])
    except Exception:
        pass  # tight_layout 在多线程/mathtext解析失败时跳过
    fname = os.path.join(output_dir, f'matrix_vs_pinn_{safe_name}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {fname}')

    rms_G = np.sqrt(np.mean((G_pinn_i - ref_G)**2))
    rms_F = np.sqrt(np.mean((F_pinn_i - ref_F)**2)) if ref_F is not None else 0.0
    print(f'  dE={abs(E_pinn-ref_E):.5f} MeV  RMS(G)={rms_G:.6f}  RMS(F)={rms_F:.6f}')


def compute_wav_rayleigh_energy(wav_data, potentials, kappa):
    """
    用Rayleigh商从WAV波函数数据计算能量.
    
    参数:
        wav_data: load_wav_wavefunction返回的dict (含r, G, F)
        potentials: load_shooting_potentials返回的dict (含vps,vms,vtt,...)
        kappa: 量子数κ
    
    返回: float 能量(MeV), 或None(无法计算时)
    """
    import torch
    if wav_data is None or potentials is None:
        return None
    
    try:
        r_wav = wav_data['r']
        G_wav = wav_data['G']
        F_wav = wav_data['F']
        
        # 插值到R网格上（与potentials一致）
        r_pot = potentials.get('r_grid', None)
        if r_pot is not None:
            r_arr = r_pot.cpu().numpy() if isinstance(r_pot, torch.Tensor) else np.array(r_pot)
            # 确保G/F插值到与potentials相同的r网格
            G_interp = np.interp(r_arr, r_wav, G_wav)
            F_interp = np.interp(r_arr, r_wav, F_wav)
        else:
            r_arr = r_wav
            G_interp = G_wav
            F_interp = F_wav
        
        g_t = torch.tensor(G_interp, dtype=torch.float32).unsqueeze(0)  # (1,N)
        f_t = torch.tensor(F_interp, dtype=torch.float32).unsqueeze(0)
        
        # 调用已有的Rayleigh商函数
        E_hc = compute_energy_rayleigh(g_t, f_t, kappa, potentials,
                                        dr=DR, npt=NPT, device='cpu')
        E_MeV = float(E_hc.item())
        return E_MeV
    except Exception as e:
        print(f'  Warning: Rayleigh energy computation failed: {e}')
        return None


# -- Main -----------------------------------------------------

def parse_kappa_from_label(label):
    l_map = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5}
    body = label.split('/')[0]
    l_char = next(ch for ch in reversed(body) if ch.isalpha())
    two_j_str = ''
    for c in reversed(body.split(l_char)[-1]):
        if c.isdigit():
            two_j_str = c + two_j_str
    two_j = int(two_j_str) if two_j_str else 1
    l = l_map.get(l_char, 0)
    j = two_j / 2.0
    return -(l + 1) if abs(j - l - 0.5) < 0.1 else l



def auto_solve_base_state(args, target_state):
    """
    自动求解基态（如果需要的话），收集所有更低主量子数的同(l,j)态。
    
    对于 3s1/2 → 收集 [1s1/2, 2s1/2] 全部波函数做正交约束
    
    参数:
        args: 命令行参数
        target_state: 目标态 (如 '3s1/2')
    
    返回:
        base_model_path: 紧邻下一级基态模型路径 (迁移学习用, 已禁用)
        all_wf_paths: list[str] 所有可能更低n态的波函数路径 (正交惩罚用)
    """
    import re
    match = re.match(r'(\d+)([a-z])(\d+/\d+)', target_state)
    if not match:
        return None, []
    
    n_target = int(match.group(1))
    l_char = match.group(2)
    j_str = match.group(3)
    
    # 如果目标就是基态 (n=1)，不需要求解任何基态
    if n_target <= 1:
        return None, []
    
    nucleus = f'{args.A}{args.Z}'
    
    # ★ 收集从 n=1 到 n=n_target-1 所有同(l,j)态的波函数路径
    all_wf_paths = []
    for n_lower in range(1, n_target):
        lower_state = f'{n_lower}{l_char}{j_str}'
        lower_wf_path = os.path.join(
            args.output_dir,
            f'{nucleus}_{args.tau}_{lower_state.replace("/", "_")}_wavefunction.json',
        )
        
        if os.path.exists(lower_wf_path):
            print(f'  Found existing: {lower_state} -> {lower_wf_path}')
            all_wf_paths.append(lower_wf_path)
        else:
            # 该级不存在，自动求解
            print(f'\n{"="*60}')
            print(f'Auto-solving lower state: {lower_state} (for {target_state} orthogonality)')
            print(f'{"="*60}')
            
            # 先递归确保更低的态都已求解（虽然循环是从n=1往上，但保险起见）
            auto_solve_base_state(args, lower_state)
            
            lower_pot = find_pot_for_state(args.pot_dir, lower_state, args.tau)
            lower_pot_file = lower_pot['pot_file'] if lower_pot else args.pot_file
            lower_E_init = lower_pot['E'] if lower_pot else -60.0
            if lower_pot:
                print(f'  Using lower-state POT: {lower_pot_file}')
            else:
                print(f'  Warning: lower-state POT not found in --pot-dir; fallback to target POT')

            lower_args = argparse.Namespace(
                state=lower_state,
                tau=args.tau,
                A=args.A,
                Z=args.Z,
                pot_file=lower_pot_file,
                final_file=args.final_file,
                wav_dir=args.wav_dir,
                pot_dir=args.pot_dir,
                batch=False,
                bound_only=False,
                epochs=args.epochs,
                lr=args.lr,
                E_init=lower_E_init,
                list_states=False,
                output_dir=args.output_dir,
                ref_wavefunctions=None,
                lambda_ortho=0.0,
                save_wavefunction=lower_wf_path,
                save_model=os.path.join(
                    args.output_dir,
                    f'{nucleus}_{args.tau}_{lower_state.replace("/", "_")}_model.pth',
                ),
                load_model=None,
                no_boundary_factor=getattr(args, 'no_boundary_factor', False),
                no_hard_g_endpoint=getattr(args, 'no_hard_g_endpoint', False),
                fixed_c=getattr(args, 'fixed_c', False),
                boundary_phase=getattr(args, 'boundary_phase', False),
                phase_constraint_weight=getattr(args, 'phase_constraint_weight', 0.0),
                w_pde=getattr(args, 'w_pde', 20.0),
                w_bc=getattr(args, 'w_bc', 0.0),
                no_live_plot=getattr(args, 'no_live_plot', False),
            )
            
            solve_state(lower_args)
            print(f'\nLower state {lower_state} solved successfully')
            
            if os.path.exists(lower_wf_path):
                all_wf_paths.append(lower_wf_path)
    
    # 紧邻的下一级态模型路径，用于迁移学习热启动
    adjacent_state = f'{n_target-1}{l_char}{j_str}'
    base_model_path = os.path.join(
        args.output_dir,
        f'{nucleus}_{args.tau}_{adjacent_state.replace("/", "_")}_model.pth',
    )
    
    print(f'\nCollected {len(all_wf_paths)} lower-state wavefunction(s) for orthogonality:')
    for p in all_wf_paths:
        print(f'    - {os.path.basename(p)}')
    
    return base_model_path, all_wf_paths


def solve_state(args):
    """
    求解单个态（从main()中提取的核心逻辑）。
    """
    kappa = parse_kappa_from_label(args.state)
    nucleus = f'{args.A}{args.Z}'
    
    # -- Shooting 参考解 (主要参考来源) --
    shooting_wf = None
    E_shooting = None
    
    # 优先从WAV文件读取 (有G和F)
    if args.wav_dir:
        import re
        tau_prefix = 'P' if args.tau == 'p' else 'N'
        state_base = re.sub(r'([a-z])(\d)', r'\1.\2', args.state)
        state_name = f'{tau_prefix}.{state_base}'
        print(f'Loading WAV: {args.wav_dir}  state={state_name}')
        shooting_wf = load_wav_wavefunction(args.wav_dir, args.tau, state_name)
        if shooting_wf is not None:
            E_shooting = shooting_wf['E']
            print(f'  Shooting E = {E_shooting:.4f} MeV')
    
    # 从FINAL文件读取 (只有G)
    if shooting_wf is None and args.final_file:
        import re
        tau_prefix = 'P' if args.tau == 'p' else 'N'
        state_base = re.sub(r'([a-z])(\d)', r'\1.\2', args.state)
        state_name = f'{tau_prefix}.{state_base}'
        print(f'Loading FINAL file: {args.final_file}  state={state_name}')
        shooting_wf = load_shooting_wavefunction(args.final_file, state_name)
        if shooting_wf is not None:
            E_shooting = shooting_wf['E']
            print(f'  Shooting E = {E_shooting:.4f} MeV')
    
    # -- 势场 (Shooting POT, 必需) --
    if args.pot_file is None:
        raise ValueError("--pot-file is required (Shooting POT file)")
    print(f'Loading POT file: {args.pot_file}')
    potentials = load_shooting_potentials(args.pot_file, R_GRID)
    if potentials.get('E_shooting') is not None:
        E_shooting = potentials['E_shooting']
        print(f'  E_shooting from POT = {E_shooting:.4f} MeV')
    
    # -- 用Rayleigh商从WAV波函数计算真实能量 --
    E_rayleigh = None
    if shooting_wf is not None and potentials is not None:
        E_rayleigh = compute_wav_rayleigh_energy(shooting_wf, potentials, kappa)
        if E_rayleigh is not None:
            print(f'  E_Rayleigh(WAV+POT) = {E_rayleigh:+.4f} MeV')
            old_E = E_shooting
            shooting_wf['E'] = E_rayleigh
            shooting_wf['source'] = 'rayleigh'
            E_shooting = E_rayleigh
            if old_E is not None:
                print(f'     (replaced old E={old_E:.4f} from LEV/POT)')

    if args.list_states:
        return

    E_ref = E_shooting or 0.0
    print(f'Reference: E={E_ref:+.4f} MeV  source={"Rayleigh" if E_rayleigh else "POT/FINAL"}')

    # -- PINN 求解 --
    #   1. 用户显式指定 args.E_init
    #   2. 有 load_model -> 继承模型能量
    #   3. 有 Shooting E -> 用作初始猜测
    #   4. 否则 -> 固定 -60.0
    E_init = args.E_init
    if E_init is not None:
        print(f'  E_init (user) = {E_init:.4f} MeV')
    elif args.load_model and os.path.exists(args.load_model):
        try:
            ckpt = torch.load(args.load_model, map_location='cpu', weights_only=False)
            if isinstance(ckpt, dict) and 'E' in ckpt and ckpt['E'] is not None:
                E_init = float(ckpt['E'])
                print(f'  E_init (from base model) = {E_init:.4f} MeV')
            else:
                E_init = E_ref if E_ref != 0 else -60.0
                print(f'  E_init (from Shooting E, fallback) = {E_init:.4f} MeV')
        except Exception:
            E_init = E_ref if E_ref != 0 else -60.0
            print(f'  E_init (load failed, fallback) = {E_init:.4f} MeV')
    elif E_ref != 0:
        E_init = E_ref
        print(f'  E_init (from Shooting E) = {E_init:.4f} MeV')
    else:
        E_init = -60.0
        print(f'  E_init (fixed) = -60.0 MeV')
    
    # 加载参考波函数 (用于正交归一化损失)
    ref_wavefunctions = []
    if args.ref_wavefunctions:
        # 支持逗号分隔的多个文件
        ref_files = [f.strip() for f in args.ref_wavefunctions.split(',')]
        for ref_file in ref_files:
            if os.path.exists(ref_file):
                print(f"\nLoading reference wavefunctions: {ref_file}")
                ref_wavefunctions.extend(load_ref_wavefunctions(ref_file, device=None))
            else:
                print(f"\nWarning: Reference file not found: {ref_file}")
    
    target_nodes = count_nodes(shooting_wf['G']) if shooting_wf else None
    print(f'\nPINN solve: k={kappa} nodes={target_nodes}  E_init={E_init:.2f}')
    solver = DiracPINNSolver(
        A=args.A, Z=args.Z, tau=args.tau, kappa=kappa,
        potentials=potentials,
        ref_wavefunctions=ref_wavefunctions,
        lambda_ortho=args.lambda_ortho,
        boundary_options={
            'use_boundary_factor': not getattr(args, 'no_boundary_factor', False),
            'hard_g_endpoint_zeros': not getattr(args, 'no_hard_g_endpoint', False),
            'learn_boundary_c': not getattr(args, 'fixed_c', False),
            'use_boundary_phase': getattr(args, 'boundary_phase', False),
        },
    )
    # 激发态：相邻低态模型迁移学习 + 内嵌正交(硬约束) + 外部正交(安全网)
    is_excited = len(ref_wavefunctions) > 0
    if is_excited:
        w_ortho_val = 1.0
        load_label = args.load_model if args.load_model else 'random init'
        print(f'  [Excited state] transfer={load_label}, embedded Gram-Schmidt ON, w_ortho(safety)={w_ortho_val}')
    else:
        w_ortho_val = 0.0

    E_pinn, history = solver.train(
        E_init_MeV=E_init,
        target_nodes=target_nodes,
        max_epochs=args.epochs,
        lr=args.lr,
        print_every=max(args.epochs // 20, 400),
        lambda_ortho=args.lambda_ortho,
        load_model=args.load_model,
        w_pde=getattr(args, 'w_pde', 20.0),
        w_bc=getattr(args, 'w_bc', 0.0),
        w_ortho=w_ortho_val,
        w_phase=getattr(args, 'phase_constraint_weight', 0.0),
        live_plot=not getattr(args, 'no_live_plot', False),
    )
    G_pinn, F_pinn = solver.get_wavefunction()
    
    src_label = 'Rayleigh' if (shooting_wf and shooting_wf.get('source') == 'rayleigh') else 'Shooting'
    print(f'\nResult: E_{src_label}={E_ref:+.4f}  E_PINN={E_pinn:+.4f}  dE={abs(E_pinn-E_ref):.4f} MeV')
    
    # -- 保存模型 --
    if args.save_model:
        print(f'\nSaving model to: {args.save_model}')
        solver.save_model(args.save_model, E_MeV=E_pinn)
    
    # -- 保存波函数 --
    if args.save_wavefunction:
        print(f'\nSaving wavefunction to: {args.save_wavefunction}')
        solver.save_wavefunction(
            args.save_wavefunction,
            state_name=f'{nucleus}_{args.tau}_{args.state}',
        )
    
    # -- 绘图 (用Shooting数据作为主参考) --
    r_pinn = solver.r_np
    ref_G = shooting_wf['G'] if shooting_wf else None
    ref_F = shooting_wf['F'] if shooting_wf else None

    if ref_G is not None or ref_F is not None:
        plot_comparison(
            r_pinn, {'label': args.state, 'E': E_ref},
            ref_G, ref_F,
            r_pinn, G_pinn, F_pinn,
            E_ref, E_pinn, history,
            output_dir=args.output_dir,
            shooting_data=shooting_wf,
            tau=args.tau,
        )
    else:
        print('  Plot skipped: no WAV/FINAL reference wavefunction available')


def run_batch(args):
    """
    批量求解 POT 目录中的所有态。
    
    输出文件命名: {A}{Z}_{tau}_{state}_model.pth / _wavefunction.json
    """
    import copy

    pot_dir = args.pot_dir
    if not pot_dir or not os.path.isdir(pot_dir):
        print(f"[ERROR] --pot-dir '{pot_dir}' not found or not a directory")
        return

    nucleus = f'{args.A}{args.Z}'
    print(f'\n{"═"*60}')
    print(f' BATCH MODE: {nucleus} (A={args.A}, Z={args.Z})')
    print(f' POT directory: {pot_dir}')
    print(f'{"═"*60}\n')

    # 扫描所有 POT 态
    states = scan_pot_files(pot_dir)
    if not states:
        print('No POT files found.')
        return

    # 过滤束缚态
    if args.bound_only:
        states = [s for s in states if s['E'] < 0]
        print(f'Bound states only: {len(states)} states\n')

    print(f'Total states to solve: {len(states)}')
    for i, s in enumerate(states):
        print(f'  [{i+1}] {s["state_name"]:12s}  tau={s["tau"]}'
              f'  E={s["E"]:+.2f} MeV  occ={s.get("occ", "?")}')

    results = []
    for i, state_info in enumerate(states):
        label = state_info['label']
        tau = state_info['tau']
        pot_file = state_info['pot_file']
        E_shooting = state_info['E']

        safe_label = label.replace('/', '_')
        state_name = f'{nucleus}_{tau}_{safe_label}'

        model_path = os.path.join(args.output_dir,
                                  f'{state_name}_model.pth')
        wf_path = os.path.join(args.output_dir,
                               f'{state_name}_wavefunction.json')

        sep = "-" * 50
        print(f"\n{sep}")
        print(f' [{i+1}/{len(states)}] Solving: {state_info["state_name"]} '
              f'(tau={tau}, label={label})')
        print(f'   POT: {os.path.basename(pot_file)}')
        print(f'   Output model:     {model_path}')
        print(f'   Output wavefunc:  {wf_path}')
        print(sep)

        # 复制 args 并覆盖单个态的参数
        state_args = copy.deepcopy(args)
        state_args.state = label
        state_args.tau = tau
        state_args.pot_file = pot_file
        state_args.E_init = E_shooting
        state_args.save_model = model_path
        state_args.save_wavefunction = wf_path
        # 不嵌套 batch
        state_args.batch = False

        try:
            solve_state(state_args)
            results.append({'state': state_name, 'status': 'OK'})
        except Exception as e:
            print(f'[ERROR] Failed to solve {state_name}: {e}')
            import traceback
            traceback.print_exc()
            results.append({'state': state_name, 'status': f'FAIL: {e}'})

    # 汇总
    print(f'\n\n{"═"*60}')
    print(f' BATCH SUMMARY: {nucleus}')
    print(f'{"═"*60}')
    ok = sum(1 for r in results if r['status'] == 'OK')
    fail = len(results) - ok
    print(f'  Total: {len(results)}  |  OK: {ok}  |  FAIL: {fail}')
    for r in results:
        mark = '✓' if r['status'] == 'OK' else '✗'
        print(f'    {mark} {r["state"]:30s}  {r["status"]}')


def main():
    parser = argparse.ArgumentParser(description='Dirac: Matrix vs PINN')
    parser.add_argument('--state', type=str, default='1s1/2')
    parser.add_argument('--tau', type=str, default='p', choices=['p', 'n'])
    parser.add_argument('--A', type=int, default=16)
    parser.add_argument('--Z', type=int, default=8)
    parser.add_argument('--pot-file', type=str, default=None,
                        help='Shooting方法的POT文件 (必需, 含vps/vms/vtt和三元组非局域核)')
    parser.add_argument('--final-file', type=str, default=None,
                        help='Shooting方法的FINAL文件 (含参考波函数)')
    parser.add_argument('--wav-dir', type=str, default=None,
                        help='WAV文件目录 (含完整G/F波函数)')
    parser.add_argument('--pot-dir', type=str, default=None,
                        help='POT文件目录 (batch模式, 自动扫描所有final000)')
    parser.add_argument('--batch', action='store_true',
                        help='批量对比所有有POT文件的态')
    parser.add_argument('--bound-only', action='store_true',
                        help='batch模式下只对比束缚态 (E<0)')
    parser.add_argument('--epochs', type=int, default=8000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--E-init', type=float, default=None,
                        help='Initial E guess (MeV), 默认从POT文件读取或-55')
    parser.add_argument('--list-states', action='store_true')
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--ref-wavefunctions', type=str, default=None,
                        help='参考波函数JSON文件路径 (用于正交归一化损失, 求解激发态时使用)')
    parser.add_argument('--lambda-ortho', type=float, default=10.0,
                        help='正交归一化损失权重 (默认10.0, 0表示不使用)')
    parser.add_argument('--save-wavefunction', type=str, default=None,
                        help='保存求解得到的波函数到JSON文件')
    parser.add_argument('--save-model', type=str, default=None,
                        help='保存训练好的模型参数到文件 (.pth)')
    parser.add_argument('--load-model', type=str, default=None,
                        help='加载预训练模型参数作为起点 (迁移学习)')
    parser.add_argument('--w-pde', type=float, default=20.0,
                        help='PDE残差权重')
    parser.add_argument('--w-bc', type=float, default=0.0,
                        help='软边界损失权重')
    parser.add_argument('--no-boundary-factor', action='store_true',
                        help='关闭 r^(l+1)*渐近因子硬嵌入')
    parser.add_argument('--no-hard-g-endpoint', action='store_true',
                        help='关闭 G(0)=G(R)=0 端点硬约束')
    parser.add_argument('--fixed-c', action='store_true',
                        help='固定渐近相位参数 c=1')
    parser.add_argument('--boundary-phase', action='store_true',
                        help='启用 cos(k*c*r+phi) 中的可学习相位 phi')
    parser.add_argument('--phase-constraint-weight', type=float, default=0.0,
                        help='相位约束 cos(k*c*R+phi)=0 的损失权重')
    parser.add_argument('--no-live-plot', action='store_true',
                        help='训练时关闭实时绘图')
    args = parser.parse_args()
    
    # -- 路径解析：相对路径优先相对于项目根目录 --
    _PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    
    def _resolve_path(p):
        if p is None:
            return None
        if os.path.isabs(p):
            return p
        cwd_path = os.path.abspath(p)
        if os.path.exists(cwd_path):
            return cwd_path
        root_path = os.path.join(_PROJECT_ROOT, p)
        if os.path.exists(root_path):
            print(f'  [path] Resolved "{p}" → "{root_path}"')
            return root_path
        return cwd_path
    
    args.pot_file = _resolve_path(args.pot_file)
    args.final_file = _resolve_path(args.final_file)
    args.wav_dir = _resolve_path(args.wav_dir)
    args.pot_dir = _resolve_path(args.pot_dir)
    args.output_dir = _resolve_path(args.output_dir) or args.output_dir
    args.ref_wavefunctions = _resolve_path(args.ref_wavefunctions)
    args.save_wavefunction = _resolve_path(args.save_wavefunction)
    args.save_model = _resolve_path(args.save_model)
    args.load_model = _resolve_path(args.load_model)
    
    # -- 自动化解配置 --
    # 自动生成输出文件路径 (含核素名 + 质子/中子标识)
    nucleus = f'{args.A}{args.Z}'
    state_safe = args.state.replace("/", "_")
    default_model_path = os.path.join(args.output_dir, f'{nucleus}_{args.tau}_{state_safe}_model.pth')
    default_wavefunction_path = os.path.join(args.output_dir, f'{nucleus}_{args.tau}_{state_safe}_wavefunction.json')
    
    # 自动设置save-model（如果未指定）
    if args.save_model is None:
        args.save_model = default_model_path
        print(f'Auto-setting --save-model: {args.save_model}')
    
    # 自动设置save-wavefunction（如果未指定）
    if args.save_wavefunction is None:
        args.save_wavefunction = default_wavefunction_path
        print(f'Auto-setting --save-wavefunction: {args.save_wavefunction}')
    
    # -- 批量模式 --
    if args.batch:
        run_batch(args)
        return
    
    # -- 自动求解基态（如果需要）--
    base_result = auto_solve_base_state(args, args.state)
    if base_result and base_result[1]:  # base_result[1] = all_wf_paths (list)
        base_model_path, all_wf_paths = base_result
        # 导入所有更低n态的波函数作为正交参考 (3s→[1s,2s], 4s→[1s,2s,3s], ...)
        if args.ref_wavefunctions is None:
            args.ref_wavefunctions = ','.join(all_wf_paths)
            print(f'Auto-setting --ref-wavefunctions: {args.ref_wavefunctions}')
            print(f'  → Orthogonalizing against {len(all_wf_paths)} lower state(s)')
        # 激发态正交惩罚权重调到极大值
        if args.lambda_ortho < 50.0:
            args.lambda_ortho = 100.0
            print(f'Auto-setting --lambda-ortho: {args.lambda_ortho} (enforced orthogonality)')
        if args.load_model is None and base_model_path and os.path.exists(base_model_path):
            args.load_model = base_model_path
            print(f'Auto-setting --load-model: {args.load_model} (transfer learning)')
        print(f'\n{"="*60}')
        print(f'Now solving target state: {args.state}  [transfer learning]')
        print(f'{"="*60}')
    
    # 求解目标态
    solve_state(args)
    


if __name__ == '__main__':
    main()
