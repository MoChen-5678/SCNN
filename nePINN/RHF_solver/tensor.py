import numpy as np
from fock import compute_fock_pi_channel,compute_fock_rhoT_channel
import numpy as np
from mathtools import (
    g_rho_tensor_rho, 
    g_rho_rho, 
    calculate_R_L1, 
    calculate_S10
)

def get_tensor_self_energy(
    r: np.ndarray,
    rho_v_total: np.ndarray, # [修正] 总矢量密度 (Total Vector Density, rho_n + rho_p)，用于计算密度依赖耦合常数
    rho_v_iso: np.ndarray,   # [修正] 同位旋矢量密度 (Isovector Vector Density, rho_v_n - rho_v_p)
    rho_t_iso: np.ndarray,   # 同位旋张量密度 (Isovector Tensor Density, rho_t_n - rho_t_p)
    param: dict,
    weights: np.ndarray,
    tau: str
) -> np.ndarray:
    """
    计算局域张量自能 Sigma_T(r)。
    
    理论依据 [RHF-FC.pdf]:
    Sigma_T^tau(r) = (Sigma_{rho-T}(r) + Sigma_{rho-TV}(r)) * tau  [Eq. 3.388]
    
    参数:
    rho_v_total : 用于计算 g_rho(rho_v) 和 f_rho(rho_v) 的密度依赖性 [cite: 251]
    rho_v_iso   : 用于 rho-TV 耦合的源项 (Vector Density Source) 
    rho_t_iso   : 用于 rho-T 耦合的源项 (Tensor Density Source) 
    
    返回:
    Sigma_T_iso3 : 同位旋矢量部分。
    """
    # 1. 准备物理常数
    m_rho = param['m_rho']
    M_nuc = 938.272 if tau == 'p' else 939.565
    
    # 2. 计算耦合常数 (依赖于总矢量密度 rho_v_total)
    # f_rho: 张量耦合常数
    # g_rho: 矢量耦合常数
    f_rho_r = g_rho_tensor_rho(rho_v_total, param) 
    g_rho_r = g_rho_rho(rho_v_total, param)
    
    # 有效张量耦合系数: f_eff = f(r) / 2M [cite: 42, 1955]
    f_eff = f_rho_r / (2.0 * M_nuc)
    
    # =========================================================================
    # 第一部分: rho-Tensor Direct Term (Sigma_{rho-T})
    # 物理意义: 张量场与张量密度的自相互作用
    # 公式: Eq. (3.319) 
    # Sigma = - m^2 * f_eff * Integral[ f_eff * rho_t * R_11 ] + 接触项
    # =========================================================================
    
    # 1.1 构造积分核 R_11 (L=1)
    # 对应 R_11(m_rho; r, r')
    R11_kernel = calculate_R_L1(r, m_rho)
    
    # 1.2 构造被积源项 (Source Term)
    # 源是: 张量密度 (rho_t_iso)
    # source = (f/2M) * rho_t * r^2 * w
    source_T = f_eff * rho_t_iso * (r**2) * weights*(r[1]-r[0])/6.0
    
    # 1.3 卷积积分
    integral_T = R11_kernel @ source_T
    
    # 1.4 组合积分项与接触项
    # Integral term: - m^2 * (f/2M) * I
    term_integral = -1.0 * (m_rho**2) * f_eff * integral_T
    
    # Contact term (从 delta 函数中扣除): + (2/3) * (f/2M)^2 * rho_t
    term_contact = (2.0 / 3.0) * (f_eff**2) * rho_t_iso
    
    Sigma_rho_T = term_integral + term_contact
    
    # =========================================================================
    # 第二部分: rho-Tensor-Vector Direct Term (Sigma_{rho-TV})
    # 物理意义: 张量场与矢量密度的交叉耦合
    # 公式: Eq. (3.321) 
    # Sigma = - f_eff * m * Integral[ g_rho * rho_v * S_10 ]
    # =========================================================================
    
    # 2.1 构造积分核 S_10 (L1=1, L2=0)
    # 对应 S_10(m_rho; r, r')，即传播子的一阶导数
    S10_kernel = calculate_S10(r, m_rho)
    
    # 2.2 构造被积源项 (Source Term)
    # [修正点]: 源是 矢量密度 (rho_v_iso)，而非张量密度
    # source = g_rho * rho_v * r^2 * w
    source_TV = g_rho_r * rho_v_iso * (r**2) * weights *(r[1]-r[0])/6.0
    
    # 2.3 卷积积分
    integral_TV = S10_kernel @ source_TV
    
    # 2.4 组合
    # Sigma = - (f/2M) * m_rho * Integral
    Sigma_rho_TV = -1.0 * f_eff * m_rho * integral_TV
    
    # =========================================================================
    # 总局域张量自能
    # =========================================================================
    Sigma_T_total = Sigma_rho_T + Sigma_rho_TV
    
    return Sigma_T_total if tau == 'n' else -Sigma_T_total

