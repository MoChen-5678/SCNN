from typing import Dict
import numpy as np

r_min = 0.1
r_max = 20.1
h = 201
r = np.linspace(r_min,r_max,h)

def compute_local_densities(r, pre_density):
    """
    计算单一同位旋 τ (p 或 n) 的局域密度：
        ρ_s^(τ)(r), ρ_v^(τ)(r), ρ_T^(τ)(r)

    参数
    ----
    r : np.ndarray, 形状 (N,)
        径向网格
    pre_density : dict
        形如 pre_density[state_name] = {'v2','g','G','F'}

    返回
    ----
    dict:
        {
          "r"     : r,
          "rho_s" : rho_s_tau,
          "rho_v" : rho_v_tau,
          "rho_T" : rho_T_tau,
        }
    """
    r = np.asarray(r)
    N = len(r)
    pi4 = 4.0 * np.pi

    rho_s = np.zeros(N)
    rho_v = np.zeros(N)
    rho_T = np.zeros(N)

    rr2 = r**2

    for name, st in pre_density.items():
        v2 = st["v2"]
        g  = st["g"]
        G  = st["G"]
        F  = st["F"]

        # 公共因子 (2j+1) v^2 / (4π r^2)
        weight = g * v2 / (pi4 * rr2)
        norm = np.sqrt(np.trapz(G**2 + F**2, r))
        G /= norm
        F /= norm
        rho_s += weight * (G**2 - F**2)
        rho_v += weight * (G**2 + F**2)
        rho_T += weight * (2.0 * G * F)

    return rho_s,rho_v,rho_T


def compute_nonlocal_densities(pre_density,N):
    """
    计算单一同位旋 τ 的非局域密度：
        Rpp, Rmm, Rpm, Rmp  (分别对应 R^{++}, R^{--}, R^{+-}, R^{-+})

    参数
    ----
    pre_density : dict
        同上
    N : int
        网格点数 (len(r))

    返回
    ----
    dict:
        {
          "Rpp": Rpp,  # shape (N,N)
          "Rmm": Rmm,
          "Rpm": Rpm,
          "Rmp": Rmp,
        }
    """
    Rpp = np.zeros((N, N))
    Rmm = np.zeros((N, N))
    Rpm = np.zeros((N, N))
    Rmp = np.zeros((N, N))

    for name, st in pre_density.items():
        v2 = st["v2"]
        g  = st["g"]
        G  = st["G"]    # (N,)
        F  = st["F"]    # (N,)
        norm = np.sqrt(np.trapz(G**2 + F**2, r))
        G /= norm
        F /= norm
        w = g * v2      # 这里只是 (2j+1) v^2，没有 1/(4π r^2)

        # 外积：G(r)G(r') 等
        Rpp += w * np.outer(G, G)   # G_i(r) G_i(r')
        Rmm += w * np.outer(F, F)   # F_i(r) F_i(r')
        Rpm += w * np.outer(G, F)   # G_i(r) F_i(r')
        Rmp += w * np.outer(F, G)   # F_i(r) G_i(r')

    return Rpp,Rpm,Rmp,Rmm