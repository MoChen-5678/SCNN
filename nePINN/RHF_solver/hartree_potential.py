# hartree_potential.py

from typing import Dict
import numpy as np
from mathtools import g_sigma_rho,g_omega_rho,g_rho_rho,yukawa_convolution,_get_rho0,coulomb_convolution

# =======================
#  主函数：Hartree 势
# =======================

def compute_hartree_potentials(
    r: np.ndarray,
    rho_s_p: np.ndarray,
    rho_s_n: np.ndarray,
    rho_v_p: np.ndarray,
    rho_v_n: np.ndarray,
    param: Dict[str, float],
    weights: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    由局域密度 (ρ_s^p, ρ_s^n, ρ_v^p, ρ_v^n) 计算 Hartree 势：
        S_p(r), S_n(r), V_p(r), V_n(r),
        以及 Σ_±^H,τ(r) = V_τ ± S_τ

    参数
    ----
    r        : 径向网格 (N,)
    rho_s_p  : 质子标量密度 ρ_s^(p)(r)
    rho_s_n  : 中子标量密度 ρ_s^(n)(r)
    rho_v_p  : 质子向量密度 ρ_v^(p)(r) = ρ_p
    rho_v_n  : 中子向量密度 ρ_v^(n)(r) = ρ_n
    param    : PKA1 参数字典（get_param_set("PKA1") 的输出）
    weights  : 径向积分权重 (N,)，对应 ∫ f(r') dr'

    返回
    ----
    dict:
        {
          "S_p": S_p(r),
          "S_n": S_n(r),
          "V_p": V_p(r),
          "V_n": V_n(r),
          "Sigma_plus_p":  Σ_+^H,p(r),
          "Sigma_minus_p": Σ_-^H,p(r),
          "Sigma_plus_n":  Σ_+^H,n(r),
          "Sigma_minus_n": Σ_-^H,n(r),
        }
    """
    r = np.asarray(r)
    rho_s_p = np.asarray(rho_s_p)
    rho_s_n = np.asarray(rho_s_n)
    rho_v_p = np.asarray(rho_v_p)
    rho_v_n = np.asarray(rho_v_n)
    w = np.asarray(weights)

    # ---- 1) 组合同位旋：isoscalar / isovector ----
    rho_s = rho_s_p + rho_s_n        # isoscalar 标量
    rho_s_3 = rho_s_n - rho_s_p      # isovector 标量（这里 Hartree 中通常不用）

    rho_v = rho_v_p + rho_v_n        # isoscalar 向量 (总重子密度)
    rho_v_3 = rho_v_n - rho_v_p      # isovector 向量

    rho_b = rho_v                    # 重子密度 = ρ_v^(p) + ρ_v^(n)

    # ---- 2) 密度依赖耦合常数 g_phi(r) ----
    gsig_r = g_sigma_rho(rho_b, param)   # g_σ(r)
    gome_r = g_omega_rho(rho_b, param)   # g_ω(r)
    grho_r = g_rho_rho(rho_b, param)     # g_ρ(r)

    # ---- 3) Yukawa 卷积 I_σ[rho_s], I_ω[rho_v], I_ρ[rho_v^3] ----
    m_sigma = param["m_sigma"]
    m_omega = param["m_omega"]
    m_rho   = param["m_rho"]

    I_sigma = yukawa_convolution(r, rho_s,   m_sigma, w)
    I_omega = yukawa_convolution(r, rho_v,   m_omega, w)
    I_rho   = yukawa_convolution(r, rho_v_3, m_rho,   w)

    # ---- 4) Hartree 势（注意 g 是数组）----
    # S_sigma = - g_sigma(r)^2 * I_sigma
    S_sigma = - (gsig_r ** 2) * I_sigma

    # V_omega = + g_omega(r)^2 * I_omega
    V_omega = (gome_r ** 2) * I_omega

    # V_rho3 = + g_rho(r)^2 * I_rho
    V_rho3 = (grho_r ** 2) * I_rho

    # ---- 5) Coulomb 势（只对质子）----
    # 这里 Coulomb 是 Hartree 电势，使用 ρ_p = ρ_v_p
    e2 = 4.0 * np.pi / 137.035999084  # 自然单位 α = e^2 / 4π
    I_coul = coulomb_convolution(r, rho_v_p, w)
    V_coul = e2 * I_coul

    # ---- 6) 对 p / n 组合总的 S, V ----
    # 标量 σ 势对 p、n 相同
    S_p = S_sigma.copy()
    S_n = S_sigma.copy()

    # 向量势：ω + ρ_3 τ3 + Coulomb(仅 p)
    V_p = V_omega + V_rho3 + V_coul   # τ3(p) = +1
    V_n = V_omega - V_rho3            # τ3(n) = -1

    # ---- 7) Σ_±^H,τ = V_τ ± S_τ ----
    Sigma_plus_p  = V_p + S_p
    Sigma_minus_p = V_p - S_p

    Sigma_plus_n  = V_n + S_n
    Sigma_minus_n = V_n - S_n

    return S_p,S_n,V_p,V_n,Sigma_plus_p,Sigma_minus_p,Sigma_plus_n,Sigma_minus_n
