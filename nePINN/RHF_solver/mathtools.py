import numpy as np
import scipy.linalg as la
from scipy.special import iv,kv,spherical_in, spherical_kn
from scipy.special import ive, kve
import scipy as sp
import re
from typing import Dict, Optional
HBARC = 197.3269804
L_MAP = {"s": 0,"p": 1,"d": 2,"f": 3,"g": 4,"h": 5,"i": 6,"j": 7,"k": 8,"l": 9,"m": 10,"n": 11,"o": 12,"q": 13,"r": 14,"t": 15,"u": 16,"v": 17,"w": 18,"x": 19,"y": 20,"z": 21,}

def create_D(r_min,r_max,h,kappa):
    r = np.linspace(r_min,r_max,h)
    dr = r[1]-r[0]
    factor = 1.0 / (60.0 * dr) 
    coeffs_fwd = np.array([-147.0, 360.0, -450.0, 400.0, -225.0, 72.0, -10.0]) * factor
    coeffs_bwd = np.array([147.0, -360.0, 450.0, -400.0, 225.0, -72.0, 10.0]) * factor
    D_plus = np.zeros((h, h))  # Parity +1 (Forward dominant)
    D_minus = np.zeros((h, h)) # Parity -1 (Backward dominant)
    for i in range(h - 6):
        for k in range(7):
            D_plus[i, i + k] = coeffs_fwd[k]
    for i in range(h - 6, h):
        for k in range(7):
            D_plus[i, i - k] = coeffs_bwd[k]
    for i in range(6):
        for k in range(7):
            D_minus[i, i + k] = coeffs_fwd[k]
    for i in range(6, h):
        for k in range(7):
            D_minus[i, i - k] = coeffs_bwd[k]
    if kappa < 0:
        D_G = D_plus
        D_F = D_minus
    else: # kappa > 0
        D_G = D_minus
        D_F = D_plus

    return D_G, D_F, r
def kappa_R(kappa,r):
    inr = 1.0/r
    kpaR = kappa * inr
    kapR =np.diag(kpaR)
    return kapR
def R_nucleon(A):
    return 1.2 * (A ** (1.0/ 3.0)) 
def parse_shell_label(label: str):
    """
    输入壳层标签，如 '1s1/2'，'2p3/2'，'1d5/2'
    返回:
        n  : 主量子数 (int)
        l  : 轨道角动量 (int)
        j  : 总角动量 (float)
        kappa : Dirac κ (int)
        g    : 简并度 = 2j+1 (int)
    """
    # 用正则解析 n, l_letter, j_num
    # e.g.  '1s1/2' → n=1, letter='s', j_str='1/2'
    m = re.fullmatch(r"(\d+)([spdfghi])(\d+)/(\d+)", label)
    if m is None:
        raise ValueError(f"Invalid shell label format: {label}")

    n = int(m.group(1))
    l_letter = m.group(2)
    j_num = int(m.group(3))
    j_den = int(m.group(4))
    j = j_num / j_den

    # l 映射
    if l_letter not in L_MAP:
        raise ValueError(f"Unknown orbital letter '{l_letter}' in {label}")
    l = L_MAP[l_letter]

    # 简并度 2j + 1
    g = int(2 * j + 1)

    # Dirac 量子数 κ 计算（非常重要）
    # j = |l ± 1/2|
    # 如果 j = l + 1/2 → κ = -(j + 1/2)
    # 如果 j = l - 1/2 → κ = +(j + 1/2)
    if abs(j - (l + 0.5)) < 1e-8:
        kappa = -int(j + 0.5)
    elif abs(j - (l - 0.5)) < 1e-8:
        kappa = +int(j + 0.5)
    else:
        raise ValueError(f"Inconsistent l and j in label {label}")

    return g
def density_pm(rho_s_n,rho_v_n,rho_T_n,rho_s_p,rho_v_p,rho_T_p):
    rho_s = rho_s_n + rho_s_p
    rho_v = rho_v_n + rho_v_p
    rho_T = rho_T_n + rho_T_p
    rs_3 = rho_s_n - rho_s_p
    rv_3 = rho_v_n - rho_v_p
    rt_3 = rho_T_n - rho_T_p
    return rho_s,rho_v,rho_T,rs_3,rv_3,rt_3
def simpson_weights(r):
    N = len(r)
    dr = r[1] - r[0]
    w = np.ones(N)
    w[1:N-1:2] = 4
    w[2:N-1:2] = 2
    w *= dr / 3
    return w
def build_local_hamiltonian(
    kapR,
    mass: float,
    r: np.ndarray,
    Sigma_plus: np.ndarray,
    Sigma_minus: np.ndarray,
    DG: np.ndarray,
    DF: np.ndarray,
    Sigma_T: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    组装单个 (τ, κ) 通道的局域 Dirac 哈密顿矩阵（不含 Fock）。
    结构对应：
        [ Σ_+(r) + M       ,  -d/dr + κ/r + Σ_T(r) ]
        [  d/dr + κ/r + Σ_T(r) ,  Σ_-(r) - M       ]

    目前阶段：没有 Fock → Σ_T 可以先取 0 数组，将来加 Fock 时再传入实际的 Σ_T(r)。

    参数
    ----
    kappa       : Dirac κ 量子数（int）
    mass        : 核子质量 M_τ (MeV)，例如 938.9 等
    r           : 径向网格 (N,)
    Sigma_plus  : Σ_+(r) 数组 (N,) —— 已经包含 Hartree (S,V) 的贡献
    Sigma_minus : Σ_-(r) 数组 (N,)
    D           : 一阶导数矩阵 d/dr (N, N)，
                  你之前构造好的有限差分算子（边界条件也包含在里面）
    Sigma_T     : Σ_T(r) 张量势数组 (N,)，默认 None → 全 0

    返回
    ----
    H : 形状 (2N, 2N) 的实对称哈密顿矩阵
    """
    r = np.asarray(r)
    N = len(r)

    Sigma_plus  = np.asarray(Sigma_plus)
    Sigma_minus = np.asarray(Sigma_minus)

    if Sigma_T is None:
        Sigma_T = np.zeros_like(r)
    else:
        Sigma_T = np.asarray(Sigma_T)


    # 对角块 A, C
    # A = diag( Σ_+ + M )
    # C = diag( Σ_- - M )
    A = np.diag(Sigma_plus + mass)
    C = np.diag(Sigma_minus - mass)

    # Σ_T(r) 对角矩阵
    Tmat = np.diag(Sigma_T)

    # κ / r 项
    K_over_r = kapR

    # 上下耦合块：
    # 上： -d/dr + κ/r + Σ_T
    B1 = -DG + K_over_r + Tmat

    # 下：  d/dr + κ/r + Σ_T
    B2 = B1.T

    # 组装成 2N×2N
    H = np.zeros((2 * N, 2 * N), dtype=float)

    H[:N, :N] = A
    H[:N, N:] = B1
    H[N:, :N] = B2
    H[N:, N:] = C

    return H
def _get_rho0(param: Dict[str, float]) -> float:
    """
    从参数表里取饱和密度 rho0，如果没有就用 PKA1 常用值 0.159996 fm^-3
    """
    return float(param.get("rho0", 0.159996))
def g_rho_tensor_rho(rho_b: np.ndarray, param: Dict[str, float]) -> np.ndarray:
    """
    ρ-tensor 密度依赖耦合 g_{ρT}(ρ):
    这里给一个占位形式：
        g_{ρT}(ρ) = grtn * exp[-a_{ρT}(x - 1)], x = ρ/ρ0
    你可以根据 PKA1 原始定义微调。
    """
    rho_b = np.asarray(rho_b)
    rho0 = float(param.get("rho0", 0.159996))
    x = rho_b / rho0
    a = param["artn"]
    base = param["grtn"]
    return base * np.exp(-a * (x - 1.0))
def f_pi_rho(rho_b: np.ndarray, param: Dict[str, float]) -> np.ndarray:
    """
    π-PV 的密度依赖耦合 f_pi(ρ):
        f_pi(ρ) = fpi * exp[-apio (x - 1)], x = ρ/ρ0
    这是 PKA1 常用的形式，你可以按论文/Fortran 调整。
    """
    rho_b = np.asarray(rho_b)
    rho0 = float(param.get("rho0", 0.159996))
    x = rho_b / rho0
    a = param["apio"]
    f0 = param["fpi"]
    return f0 * np.exp(-a * (x - 1.0))
def g_sigma_rho(rho_b: np.ndarray, param: Dict[str, float]) -> np.ndarray:
    """
    PKA1: g_sigma(rho) = gsig * f_sigma(rho/rho0)
    f_sigma(x) = a * [1 + b (x + d)^2] / [1 + c (x + d)^2]
    """
    rho_b = np.asarray(rho_b)
    rho0 = _get_rho0(param)
    x = rho_b / rho0

    a = param["asig"]
    b = param["bsig"]
    c = param["csig"]
    d = param["dsig"]

    f = a * (1.0 + b * (x + d) ** 2) / (1.0 + c * (x + d) ** 2)
    return param["gsig"] * f
def g_omega_rho(rho_b: np.ndarray, param: Dict[str, float]) -> np.ndarray:
    """
    PKA1: g_omega(rho) = gome * f_omega(rho/rho0)
    f_omega(x) = a * [1 + b (x + d)^2] / [1 + c (x + d)^2]
    """
    rho_b = np.asarray(rho_b)
    rho0 = _get_rho0(param)
    x = rho_b / rho0

    a = param["aome"]
    b = param["bome"]
    c = param["come"]
    d = param["dome"]

    f = a * (1.0 + b * (x + d) ** 2) / (1.0 + c * (x + d) ** 2)
    return param["gome"] * f
def g_rho_rho(rho_b: np.ndarray, param: Dict[str, float]) -> np.ndarray:
    """
    PKA1: g_rho(rho) = grho * exp[- a_rho (x - 1)], x = rho/rho0
    """
    rho_b = np.asarray(rho_b)
    rho0 = _get_rho0(param)
    x = rho_b / rho0

    a = param["arho"]
    f = np.exp(-a * (x - 1.0))
    return param["grho"] * f
def build_yukawa_RL(r: np.ndarray, mass: float, L: int) -> np.ndarray:
    """
    构造数值稳定的 R_L 径向格林函数。
    公式: R_L = 4*pi*m * I_{L+1/2}(z_<) * K_{L+1/2}(z_>)
    使用 ive/kve 消除 exp(z) 的溢出风险。
    """
    r = np.asarray(r)
    # 避免 r=0 导致发散，物理上 r=0 处波函数为 0
    r_safe = np.where(r < 1e-20, 1e-20, r)
    z = mass * r_safe
    
    nu = L + 0.5
    
    # 计算缩放 Bessel: ive(z) = I(z)*e^-z, kve(z) = K(z)*e^z
    Ie = ive(nu, z)
    Ke = kve(nu, z)
    
    R_i, R_j = np.meshgrid(r_safe, r_safe, indexing='ij')
    z_i, z_j = np.meshgrid(z, z, indexing='ij')
    
    # 指数修正因子: exp( -|z_i - z_j| )
    # 原理: I(z_<)K(z_>) = Ie(z_<)Ke(z_>) * exp(z_< - z_>)
    # z_> - z_< 恒为正，因此指数项 <= 1，非常安全
    exp_factor = np.exp(-np.abs(z_i - z_j))
    
    # 构造矩阵
    Ie_mat_i, Ke_mat_j = np.meshgrid(Ie, Ke, indexing='ij')
    Ke_mat_i, Ie_mat_j = np.meshgrid(Ke, Ie, indexing='ij')
    
    mat = np.zeros_like(R_i)
    mask_upper = R_i <= R_j
    mask_lower = R_i > R_j
    
    mat[mask_upper] = (Ie_mat_i * Ke_mat_j)[mask_upper]
    mat[mask_lower] = (Ke_mat_i * Ie_mat_j)[mask_lower]
    
    # 组合所有系数: 4*pi * m
    # 注意: 论文公式可能包含 1/sqrt(z z') ? 
    # 标准 Green 函数展开: 4mk/pi * j(ikr_<)h(ikr_>) -> 4m * i(mr_<)k(mr_>)
    # 修正贝塞尔定义: i_l(x) = sqrt(pi/2x) I_{l+1/2}(x)
    # 代入后: 4*pi*m * (pi/2 * 1/sqrt(z z') * I * K) 
    # = 2 * pi^2 * m / sqrt(z z') * I * K
    # 请根据您的具体公式约定确认系数。这里按您之前的 build_yukawa_RL 逻辑（无sqrt因子）
    # 如果您之前的逻辑是 4*pi*m * i_L * k_L:
    # i_L(x) = sqrt(pi/2x) I_{L+1/2}(x)
    # 那么系数应该是: 4*pi*m * (pi/2) * (1/sqrt(z_i z_j))
    
    # 假设 build_yukawa_RL 原意是直接用 scipy.special.spherical_in (即 i_L)
    # spherical_in(n, z) = sqrt(pi/2z) * I_{n+1/2}(z)
    # 我们这里手动构造了 sqrt(pi/2z) 部分
    
    prefactor = (4.0 * np.pi * mass) * (0.5 * np.pi) / np.sqrt(z_i * z_j)
    
    return prefactor * mat * exp_factor
def yukawa_convolution(
    r: np.ndarray,
    density: np.ndarray,
    m_phi: float,
    weights: np.ndarray,
    L: int = 0  # 必须指定 L
) -> np.ndarray:
    """
    计算 Yukawa 势的卷积积分 V(r) = Integral [ G_L(r,r') * density(r') * r'^2 ] dr'
    """
    # 1. 获取对应 L 的格林函数矩阵 kernel[i, j]
    kernel = build_yukawa_RL(r, m_phi, L)
    
    # 2. 准备被积函数
    # 通常积分测度包含 r'^2 (球坐标体积元)
    # 假设 density 尚未包含 r'^2，需要乘进去；如果 density 已经是 u^2 + v^2 形式且定义包含 r^2 则不需要。
    # 标准 RMF 习惯：Integral V(r,r') rho(r') r'^2 dr'
    integrand = density * (r**2) * weights *(r[1]-r[0])/6.0
    
    # 3. 执行矩阵向量乘法进行积分
    # V[i] = Sum_j ( Kernel[i, j] * integrand[j] )
    potential = kernel @ integrand
    
    return potential
def build_coulomb_RL(r: np.ndarray, L: int) -> np.ndarray:
    """
    构造库仑势在特定角动量 L 下的径向格林函数.
    
    数学形式:
        G_L(r, r') = (r_< ^ L) / (r_> ^ (L+1))
        
    参数:
    r : 径向网格点 (N,)
    L : 角动量阶数
    """
    r = np.asarray(r)
    # 避免除以 0 的风险，对 r=0 做微小偏移或掩码处理
    # 物理上 r_> 必定 >= r_min，只要网格不全为0即可
    r_safe = np.where(r == 0, 1e-20, r)
    
    R_i, R_j = np.meshgrid(r, r, indexing="ij")
    
    # 确定 r_min 和 r_max
    R_less = np.minimum(R_i, R_j)
    R_greater = np.maximum(R_i, R_j)
    
    # 再次保护分母，防止 r_i=r_j=0 的情况
    R_greater = np.where(R_greater == 0, 1e-20, R_greater)
    
    # 计算格林函数
    # 注意：如果 L=0 且 r=0，0^0=1，逻辑正确
    # 如果 L>0 且 r=0，0^L=0，逻辑也正确
    kernel = (R_less ** L) / (R_greater ** (L + 1))
    
    return kernel
def coulomb_convolution(
    r: np.ndarray,
    density: np.ndarray,
    weights: np.ndarray,
    L: int = 0
) -> np.ndarray:
    """
    计算库仑多极展开积分.
    Potential(r) = Integral [ G_L(r,r') * density(r') * r'^2 ] dr'
    """
    # 1. 构建核
    kernel = build_coulomb_RL(r, L)
    
    # 2. 准备被积函数
    # 统一约定：传入的 density 是纯密度 rho
    # 积分元为 rho(r') * r'^2 * weights
    integrand = density * (r**2) * weights *(r[1]-r[0])/6.0
    
    # 3. 积分
    potential = kernel @ integrand
    
    return potential
def build_yukawa_SL_matrix(r: np.ndarray, mass: float, L1: int, L2: int) -> np.ndarray:
    """
    构造数值稳定的 S_{L1 L2} (导数型) 核函数。
    S ~ I(z_<)K(z_>) [r<r'] - K(z_<)I(z_>) [r>r']
    """
    r = np.asarray(r)
    r_safe = np.where(r < 1e-20, 1e-20, r)
    z = mass * r_safe
    
    nu1 = L1 + 0.5
    nu2 = L2 + 0.5
    
    Ie1 = ive(nu1, z)
    Ke1 = kve(nu1, z)
    Ie2 = ive(nu2, z)
    Ke2 = kve(nu2, z)
    
    R_i, R_j = np.meshgrid(r_safe, r_safe, indexing='ij')
    z_i, z_j = np.meshgrid(z, z, indexing='ij')
    
    exp_factor = np.exp(-np.abs(z_i - z_j))
    
    # 同样加上球贝塞尔的转换系数 pi/2 * 1/sqrt(zz')
    # 系数 m (来自 S 定义) * 4*pi? (通常 S 也来源于 Yukawa 展开)
    # 假设 S 是导数，系数与 R 类似
    prefactor = (mass * 4.0 * np.pi) * (0.5 * np.pi) / np.sqrt(z_i * z_j)
    
    Ie1_i, Ie1_j = np.meshgrid(Ie1, Ie1, indexing='ij')
    Ke1_i, Ke1_j = np.meshgrid(Ke1, Ke1, indexing='ij')
    Ie2_i, Ie2_j = np.meshgrid(Ie2, Ie2, indexing='ij')
    Ke2_i, Ke2_j = np.meshgrid(Ke2, Ke2, indexing='ij')
    
    mat = np.zeros_like(R_i)
    
    # r < r'
    mask_upper = R_i < R_j
    mat[mask_upper] = (Ie1_i * Ke2_j)[mask_upper]
    
    # r > r' (注意负号)
    mask_lower = R_i > R_j
    mat[mask_lower] = -1.0 * (Ke1_i * Ie2_j)[mask_lower]
    
    return prefactor * mat * exp_factor
def calculate_R_L1(r, m_rho):
    return build_yukawa_RL(r, m_rho, 1)

def calculate_S10(r, m_rho):
    return build_yukawa_SL_matrix(r, m_rho, 1, 0)
def get_kappa_from_label(label):
    # 简单的解析逻辑，假设 label 格式为 "1s1/2"
    # 这里是一个简化的示例，你需要根据你的 state_name 格式调整
    import re
    match = re.match(r'(\d+)([a-z])(\d+)/2', label)
    if not match:
        raise ValueError(f"Unknown shell format: {label}")
    
    n, l_char, j_num = match.groups()
    j = float(j_num) / 2.0
    
    l_map = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4}
    l = l_map[l_char]
    
    # kappa = -(l + 1) if j = l + 1/2
    # kappa = +l       if j = l - 1/2
    if abs(j - (l + 0.5)) < 0.01:
        return -(l + 1)
    else:
        return l
def parse_kappa_from_label(label: str) -> int:
    """辅助工具：从壳层标签解析 kappa"""
    match = re.match(r'(\d+)([a-zA-Z])(\d+)/2', label)
    if not match:
        raise ValueError(f"无法解析壳层标签: {label}")
    n_str, l_str, j2_str = match.groups()
    j = float(j2_str) / 2.0
    l_map = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5}
    l = l_map.get(l_str.lower(), 0)
    if abs(j - (l + 0.5)) < 1e-3: return -(l + 1)
    else: return l
