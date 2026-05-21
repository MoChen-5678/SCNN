"""
Dirac PDE 残差计算模块

核心功能:
  1. 构建 5PADF 有限差分矩阵 (复用 SCNN 已验证方案)
  2. 根据 κ 量子数确定差分方向 (宇称规则)
  3. 计算 Dirac 耦合方程的 PDE 残差
  4. Rayleigh 商能量估计
  5. 严格量纲统一 (全部转换为 fm⁻¹)

参考文献:
  - Wang et al., Chin. Phys. C 49, 014106 (2025) — Eq.(13)-(14): 5PADF 公式
  - Szafran et al., Phys. Rev. B 99, 195406 (2019) — 宇称→差分方向映射
"""

import torch
import torch.nn as nn
from config import HBAR_C, DR, NPT, R_SAFE_OFFSET, M_NUCLEON


# ═══════════════════════════════════════════════════════════════
#   全局有限差分矩阵缓存
# ═══════════════════════════════════════════════════════════════

_fd_cache = {}


def clear_fd_cache():
    """清空差分矩阵缓存（当网格参数改变时调用）"""
    global _fd_cache
    _fd_cache = {}


# ═══════════════════════════════════════════════════════════════
#   宇称 → 差分方向映射
# ═══════════════════════════════════════════════════════════════

def get_fd_directions(kappa):
    """
    根据κ量子数确定 G 和 F 各自应使用的差分方向。

    ★ 物理依据: Wang et al. (2025) Eq.4 + Ref.[50] Szafran et al. (2019)
      - 波函数 ψ = (1/r)[iG·Ω^l, F·Ω^{l'}], l+l'=2j
      - G 的宇称 π_G = (-1)^l, F 的宇称 π_F = (-1)^{l'}
      - 偶宇称(even parity) → forward ADF
      - 奇宇称(odd parity) → backward ADF
      - κ<0: l=|κ|-1; κ>0: l=κ → G方向 = forward if l%2==0 else backward

    示例:
      - 1s₁/₂ (κ=-1): l=0(even→forward), l'=1(odd→backward) ✓
      - 1p₃/₂ (κ=-2): l=1(odd→backward), l'=2(even→forward) ✓
      - 1p₁/₂ (κ=+1): l=1(odd→backward), l'=0(even→forward) ✓

    参数:
        kappa: int, 角量子数

    返回:
        g_dir: str, G 的差分方向 ('forward' 或 'backward')
        f_dir: str, F 的差分方向 (始终与 G 相反)
    """
    kappa_val = int(kappa)
    l_G = abs(kappa_val) - 1 if kappa_val < 0 else kappa_val
    g_dir = 'forward' if l_G % 2 == 0 else 'backward'
    f_dir = 'backward' if g_dir == 'forward' else 'forward'
    return g_dir, f_dir


# ═══════════════════════════════════════════════════════════════
#   5 点非对称差分 (5PADF) 矩阵构建
# ═══════════════════════════════════════════════════════════════

def build_5padf_matrix(n, dr, direction='forward', device=None, dtype=torch.float32):
    """
    构建 N×N 一阶导数有限差分矩阵（基于 Wang et al. 2025 的 5PADF 方案）。

    ★ 关键性质: D_backward = -D_forward.flip([0,1]) (精确伴随关系)
       这保证离散化的哈密顿量保持厄米性!

    5PADF 公式 (精度 O(h⁴)):
      前向(Forward): df/dr|₀ = (-25f₀ + 48f₁ - 36f₂ + 16f₃ - 3f₄) / 12h   Eq.(13)
      后向(Backward): df/dr|₀ = (+25f₀ - 48f₋₁ + 36f₋₂ - 16f₋₃ + 3f₋₄)/ 12h   Eq.(14)

    边界降级处理:
      - ≥5点可用: 完整 5PADF (O(h⁴))
      - 4点可用: 3点中心差分 (O(h²))
      - 3点可用: 3点中心差分 (O(h²))
      - 2点可用: 2点前/后向 (O(h))
      - 1点可用: 0 (边界)

    参数:
        n: int, 网格点数
        dr: float, 径向步长 (fm)
        direction: str, 'forward' 或 'backward'
                   由 G/F 的宇称决定! 用 get_fd_directions(kappa) 获取正确方向
        device: torch.device, 目标设备
        dtype: 数据类型

    返回:
        D: (n, n) 一阶导数差分矩阵 (稠密张量)
    """
    cache_key = (n, dr, direction)
    if cache_key in _fd_cache:
        cached = _fd_cache[cache_key]
        if device is not None and cached.device != device:
            cached = cached.to(device=device, dtype=dtype)
        return cached

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inv_12dr = 1.0 / (12.0 * dr)
    inv_2dr = 1.0 / (2.0 * dr)
    inv_dr = 1.0 / dr

    if direction == 'forward':
        D = _build_forward_matrix(n, inv_12dr, inv_2dr, inv_dr, device, dtype)
    elif direction == 'backward':
        D_fwd = _build_forward_matrix(n, inv_12dr, inv_2dr, inv_dr, device, dtype)
        # ★ 反角镜像伴随变换: D_bw[i,j] = -D_fw[n-1-i, n-1-j]
        D = (-D_fwd).flip([0, 1]).contiguous()
    else:
        raise ValueError(f"direction 必须是 'forward' 或 'backward', 得到: {direction}")

    _fd_cache[cache_key] = D
    return D


def _build_forward_matrix(n, inv_12dr, inv_2dr, inv_dr, device, dtype):
    """构建 forward 差分矩阵的内部实现"""
    D = torch.zeros(n, n, device=device, dtype=dtype)

    for i in range(n):
        avail_right = n - i
        if avail_right >= 5:
            # 完整 5PADF (O(h⁴))
            D[i, i]   = (-25.0) * inv_12dr
            D[i, i+1] = (+48.0) * inv_12dr
            D[i, i+2] = (-36.0) * inv_12dr
            D[i, i+3] = (+16.0) * inv_12dr
            D[i, i+4] = (-3.0)  * inv_12dr
        elif avail_right == 4:
            D[i, i]   = (-3.0) * inv_2dr
            D[i, i+1] = (+4.0) * inv_2dr
            D[i, i+2] = (-1.0) * inv_2dr
        elif avail_right == 3:
            D[i, i]   = (-3.0) * inv_2dr
            D[i, i+1] = (+4.0) * inv_2dr
            D[i, i+2] = (-1.0) * inv_2dr
        elif avail_right == 2:
            D[i, i]   = (-1.0) * inv_dr
            D[i, i+1] = (+1.0) * inv_dr
        # avail_right == 1: D[i,i] = 0 (已初始化)

    return D


def apply_fd_matrix(signal, D_matrix):
    """
    对信号应用有限差分矩阵，计算一阶导数。

    参数:
        signal: (..., N) 信号张量
        D_matrix: (N, N) 差分矩阵
    返回:
        derivative: (...,) 与 signal 同形的导数张量
    """
    # 矩阵乘法: d(signal)/dr = signal @ D^T
    # 因为 D[i,j] = coeff for df/dx at point i from point j
    return signal @ D_matrix.T


def _as_device_tensor(value, device, dtype):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(value, device=device, dtype=dtype)


def has_nonlocal_kernels(potentials):
    """判断势场字典是否包含真正的二维非局域核。"""
    return any(key in potentials for key in ('XG_kernel', 'XF_kernel', 'YG_kernel', 'YF_kernel'))


def apply_nonlocal_kernels(g, f, potentials, dr=DR, hbc=HBAR_C):
    """
    在网络预测的 G/F 上直接计算非局域势积分。

    输入矩阵采用字段三元组 j, i, value 还原为 K[j,i]。输出的二维核
    已经按 Simpson 数值求积权重配置好，因此这里只做矩阵乘法：
    out[j] = ∑_i K[j,i] ψ[i]，不再额外乘 dr 或积分权重。

    势场文件中的 value 为 fm^-1；加载器为保持 potentials 字典单位一致
    会转为 MeV，因此这里统一除以 ħc 还原为 fm^-1。
    """
    B, N = g.shape
    device, dtype = g.device, g.dtype

    def kernel(name):
        value = potentials.get(name)
        if value is None:
            return torch.zeros((N, N), device=device, dtype=dtype)
        tensor = _as_device_tensor(value, device, dtype) / hbc
        if tensor.shape != (N, N):
            raise ValueError(f"{name} shape must be {(N, N)}, got {tuple(tensor.shape)}")
        return tensor

    xg = kernel('XG_kernel')
    xf = kernel('XF_kernel')
    yg = kernel('YG_kernel')
    yf = kernel('YF_kernel')

    x_int = g @ xg.T + f @ xf.T
    y_int = g @ yg.T + f @ yf.T
    return x_int, y_int


# ═══════════════════════════════════════════════════════════════
#   Dirac PDE 残差计算 (核心!)
# ═══════════════════════════════════════════════════════════════

def compute_dirac_residual(g, f, E, kappa, potentials, dr=DR, npt=NPT,
                           f_weight=3.0, return_components=False,
                           device=None):
    """
    计算 Dirac 耦合方程的 PDE 残差。

    ★ Dirac 方程组 (核物理教材 Eq.3.57, 与 Fortran Detgff.f90 一致):
    
      dG/dr = -u1g·G + u1f·F     ... (Eq.G)
      dF/dr = +u2f·F - u2g·G     ... (Eq.F)
    
      其中:
        u1g = κ/r + V_tt + XG        (G 方程中 G 分量的自耦合系数)
        u1f = E/ħc - V_ms - XF       (G 方程中 F 分量的交叉耦合系数)
        u2f = κ/r + V_tt + YF        (F 方程中 F 分量的自耦合系数)
        u2g = E/ħc - V_ps - YG       (F 方程中 G 分量的交叉耦合系数)

    ★ PDF Eq.3.57 展开 (无张量势/Fock项时退化为标准形式):
      dG/dr = -(κ/r)·G + (ε - Σ_- + M)·F
      dF/dr = +(κ/r)·F - (ε - Σ_+ - M)·G
      
      Fortran 中 vms = Σ_- - M, vps = Σ_+ (不含M)
      所以 ε - vms = ε - Σ_- + M,  ε - vps - M = ε - Σ_+ - M ← 一致

    ★ 量纲统一: 所有势场必须除以 ħc 转换为 fm⁻¹ 才能与 E/ħc 相加!

    参数:
        g: (B, N) 或 (N,), 大分量波函数
        f: (B, N) 或 (N,), 小分量波函数
        E: 标量或 (B,), 本征值能量 (MeV)
        kappa: int 或 (B,), 角量子数
        potentials: dict 包含以下键:
            - 'vps': Σ_+ 势 (标量势+, MeV), 不含 M
            - 'vms': Σ_- 势 (标量势-, MeV), 已含 -M
            - 'vtt': 张量势 T(r) (MeV)
            - 'XG_kernel'/'XF_kernel'/'YG_kernel'/'YF_kernel':
              二维非局域交换核 K[j,i] (MeV), 优先使用
            - 'XG'/'XF'/'YG'/'YF': 旧等效局域化项, 仅在没有 kernel 时兼容
        dr: float, 径向步长 (fm)
        npt: int, 网格点数
        f_weight: float, F 分量残差的额外权重 (默认3.0, 因为 F≪G)
        return_components: bool, True 返回详细分量, False 返回标量损失
        device: torch.device, 计算设备
    """
    if device is None:
        device = g.device

    # ─── 0. 形状标准化 ───
    has_batch = g.dim() == 2
    if not has_batch:
        g = g.unsqueeze(0)
        f = f.unsqueeze(0)

    B, N = g.shape
    hbc = HBAR_C

    # ─── 1. 提取势场 (带默认零值) ───
    vps = potentials.get('vps', torch.zeros_like(g))
    vms = potentials.get('vms', torch.zeros_like(g))
    vtt = potentials.get('vtt', torch.zeros_like(g))
    use_nonlocal_kernels = has_nonlocal_kernels(potentials)
    if not use_nonlocal_kernels:
        XG = potentials.get('XG', torch.zeros_like(g))
        XF = potentials.get('XF', torch.zeros_like(g))
        YG = potentials.get('YG', torch.zeros_like(g))
        YF = potentials.get('YF', torch.zeros_like(g))
    else:
        XG = XF = YG = YF = torch.zeros_like(g)

    # ─── 2. 量纲统一: 全部转为 fm⁻¹ ───
    # 这是v18修复的核心: MeV → fm⁻¹ 通过除以 ħc
    # E/hbc 的单位是 fm⁻¹, 所有势也必须 /hbc
    E_hc = E / hbc                          # (B,) fm⁻¹
    vps_hc = vps / hbc                     # (B,N) fm⁻¹
    vms_hc = vms / hbc                     # (B,N) fm⁻¹
    vtt_hc = vtt / hbc                      # (B,N) fm⁻¹
    XG_hc = XG / hbc                        # (B,N) fm⁻¹
    XF_hc = XF / hbc                        # (B,N) fm⁻¹
    YG_hc = YG / hbc                        # (B,N) fm⁻¹
    YF_hc = YF / hbc                        # (B,N) fm⁻¹

    # ─── 3. 准备径向坐标 ───
    r = torch.arange(N, device=device, dtype=g.dtype) * dr
    r_safe = torch.clamp(r, min=R_SAFE_OFFSET)  # 避免 r=0 除零
    r_expanded = r_safe.unsqueeze(0).expand(B, -1)  # (B, N)

    # ─── 4. 计算系数 ───
    kappa_tensor = torch.full((B,), float(kappa), device=device, dtype=g.dtype)
    kappa_over_r = kappa_tensor / r_expanded           # (B, N) κ/r

    # ★ Dirac 方程系数 (与 Fortran Detgff.f90 完全一致):
    #   dG/dr = -u1g·G + u1f·F
    #   dF/dr = +u2f·F - u2g·G
    #
    #   u1g = κ/r + vtt + XG    (G自耦合: 包含κ/r、张量势、Fock G分量)
    #   u1f = ε/ħc - vms - XF   (F交叉耦合: 能量-标量势-Fock F分量)
    #   u2f = κ/r + vtt + YF    (F自耦合: 包含κ/r、张量势、Fock F分量)
    #   u2g = ε/ħc - vps - YG   (G交叉耦合: 能量-标量势-Fock G分量)
    
    u1g = kappa_over_r + vtt_hc + XG_hc              # (B,N)
    u1f = E_hc.unsqueeze(1) - vms_hc - XF_hc        # (B,N)
    u2f = kappa_over_r + vtt_hc + YF_hc              # (B,N)
    u2g = E_hc.unsqueeze(1) - vps_hc - YG_hc        # (B,N)

    # ─── 5. 构建差分矩阵并计算数值导数 ───
    g_dir, f_dir = get_fd_directions(kappa)

    D_g = build_5padf_matrix(N, dr, g_dir, device=device, dtype=g.dtype)
    D_f = build_5padf_matrix(N, dr, f_dir, device=device, dtype=g.dtype)

    # 数值导数: dg/dr 和 df/dr
    dg_dr = apply_fd_matrix(g, D_g)   # (B, N)
    df_dr = apply_fd_matrix(f, D_f)   # (B, N)

    # ─── 6. 计算 PDE 残差 ───
    # Eq.G: dg/dr - [-u1g·G + u1f·F] = 0
    #       即: dg/dr + u1g·G - u1f·F = 0
    target_dg = -u1g * g + u1f * f                     # (B, N)
    if use_nonlocal_kernels:
        x_int, y_int = apply_nonlocal_kernels(g, f, potentials, dr=dr, hbc=hbc)
        target_dg = target_dg - x_int
    R_g = dg_dr - target_dg                            # (B, N) G方程残差

    # Eq.F: df/dr - [u2f·F - u2g·G] = 0
    #       即: df_dr - u2f·F + u2g·G = 0
    target_df = u2f * f - u2g * g                      # (B, N)
    if use_nonlocal_kernels:
        target_df = target_df + y_int
    R_f = df_dr - target_df                            # (B, N) F方程残差

    # ─── 7. 加权损失 ───
    # F 分量加权: 因为 |F| << |G|, 残差自然更小, 需要加权平衡
    loss_g = torch.mean(R_g ** 2)                    # ()
    loss_f = torch.mean(R_f ** 2)                    # ()
    loss_pde = loss_g + f_weight * loss_f            # ()

    # ─── 8. Rayleigh 商能量估计 (可选诊断) ───
    # <H> = <ψ|H|ψ> / <ψ|ψ>, 用于验证能量自洽性
    energy_rayleigh = compute_rayleigh_quotient(
        g, f, E, kappa, potentials, dr, npt, device
    )

    if not return_components:
        if not has_batch:
            return loss_pde.squeeze()
        return loss_pde

    result = {
        'loss_pde': loss_pde,
        'loss_g': loss_g,
        'loss_f': loss_f,
        'R_g': R_g,
        'R_f': R_f,
        'energy_rayleigh': energy_rayleigh,
    }

    # 恢复无 batch 的形状
    if not has_batch:
        result['R_g'] = result['R_g'].squeeze(0)
        result['R_f'] = result['R_f'].squeeze(0)

    return result


def compute_rayleigh_quotient(g, f, E, kappa, potentials, dr=DR, npt=NPT,
                              device=None):
    """
    通过 Rayleigh 商估计 Dirac Hamiltonian 的期望能量。

    ε_Rayleigh = <ψ|H_Dirac|ψ> / <ψ|ψ>

    用于诊断: 如果 PINN 收敛良好, E (可学习参数) 应接近 ε_Rayleigh。

    参数: 与 compute_dirac_residual 相同
    返回:
        E_rayleigh: (B,) Rayleigh 商能量估计 (MeV)
    """
    if device is None:
        device = g.device

    has_batch = g.dim() == 2
    if not has_batch:
        g = g.unsqueeze(0)
        f = f.unsqueeze(0)

    B, N = g.shape
    hbc = HBAR_C

    # 势场提取
    vps = potentials.get('vps', torch.zeros_like(g))
    vms = potentials.get('vms', torch.zeros_like(g))
    vtt = potentials.get('vtt', torch.zeros_like(g))

    # 量纲转换
    vps_hc = vps / hbc
    vms_hc = vms / hbc
    vtt_hc = vtt / hbc

    # 径向坐标
    r = torch.arange(N, device=device, dtype=g.dtype) * dr
    r_safe = torch.clamp(r, min=R_SAFE_OFFSET)

    kappa_t = torch.full((B,), float(kappa), device=device, dtype=g.dtype)
    k_r = kappa_t / r_safe.unsqueeze(0)

    norm_int = torch.trapz(g**2 + f**2, dim=-1, dx=dr).clamp(min=1e-30)

    g_dir, f_dir = get_fd_directions(kappa)
    D_g = build_5padf_matrix(N, dr, g_dir, device=device, dtype=g.dtype)
    D_f = build_5padf_matrix(N, dr, f_dir, device=device, dtype=g.dtype)
    dg_dr = apply_fd_matrix(g, D_g)
    df_dr = apply_fd_matrix(f, D_f)

    local_num = (torch.trapz(f * dg_dr, dim=-1, dx=dr)
                 - torch.trapz(g * df_dr, dim=-1, dx=dr)
                 + torch.trapz(2.0 * (k_r + vtt_hc) * g * f, dim=-1, dx=dr)
                 + torch.trapz(vps_hc * g**2, dim=-1, dx=dr)
                 + torch.trapz(vms_hc * f**2, dim=-1, dx=dr))

    if has_nonlocal_kernels(potentials):
        x_int, y_int = apply_nonlocal_kernels(g, f, potentials, dr=dr, hbc=hbc)
        nonlocal_num = (torch.trapz(f * x_int, dim=-1, dx=dr)
                        + torch.trapz(g * y_int, dim=-1, dx=dr))
    else:
        XG = potentials.get('XG', torch.zeros_like(g)) / hbc
        XF = potentials.get('XF', torch.zeros_like(g)) / hbc
        YG = potentials.get('YG', torch.zeros_like(g)) / hbc
        YF = potentials.get('YF', torch.zeros_like(g)) / hbc
        nonlocal_num = (torch.trapz(YG * g**2, dim=-1, dx=dr)
                        + torch.trapz(XF * f**2, dim=-1, dx=dr)
                        + torch.trapz((XG + YF) * g * f, dim=-1, dx=dr))

    E_rayleigh = (local_num + nonlocal_num) / norm_int * hbc

    if not has_batch:
        return E_rayleigh.squeeze(0)
    return E_rayleigh


# ═══════════════════════════════════════════════════════════════
#   便捷函数: 从势场字典创建完整 potentials 字典
# ═══════════════════════════════════════════════════════════════

def make_local_potentials(vps, vms, vtt=None):
    """
    创建只含局部势场的 potentials 字典 (Hartree近似, 无Fock交换项)。
    用于 MVP 阶段测试。

    参数:
        vps: (N,) or (B,N) 标量+势
        vms: (N,) or (B,N) 标量-势
        vts: (N,) or (B,N) 张量势 (可选)
    返回:
        potentials: dict
    """
    pots = {'vps': vps, 'vms': vms}
    if vtt is not None:
        pots['vtt'] = vtt
    else:
        pots['vtt'] = torch.zeros_like(vps)
    # 非局域项设为零
    pots['XG'] = torch.zeros_like(vps)
    pots['XF'] = torch.zeros_like(vps)
    pots['YG'] = torch.zeros_like(vps)
    pots['YF'] = torch.zeros_like(vps)
    return pots
