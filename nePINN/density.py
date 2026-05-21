"""
密度计算模块

从占据态的 Dirac 波函数 (G, F) 计算各种密度分布，
这些密度是 Hartree 势场更新的输入。

核心公式 (来自 Density.f90 第48-80行):
  ρ_s(r) = Σ_α [v_α · μ_α · (G_α² - F_α²)] / (4πr²)
  ρ_v(r) = Σ_α [v_α · μ_α · (G_α² + F_α²)] / (4πr²)
  ρ_t(r) = Σ_α [v_α · μ_α · 2·G_α·F_α] / (4πr²)

其中 μ = 2j+1 为简并度, v_α 为占据概率 (闭壳=1)
"""

import torch
import numpy as np
from config import DR, NPT, R_GRID, R_SAFE_OFFSET


class DensityCalculator:
    """
    密度计算器：管理所有占据态并累加各态贡献。
    """

    def __init__(self, dr=DR, npt=NPT):
        self.dr = dr
        self.npt = npt
        self.states = []

    def add_state(self, name, g, f, occupancy=2.0, kappa=None, is_proton=False):
        """
        添加一个占据态的贡献。

        参数:
            name: str, 态标识符
            g: (npt,), 大分量波函数
            f: (npt,), 小分量波函数
            occupancy: float, 占据数 = vv × μ (考虑简并)
            kappa: int or None
            is_proton: bool, 是否质子态
        """
        self.states.append({
            'name': name,
            'g': g.detach().cpu() if isinstance(g, torch.Tensor) else torch.tensor(g),
            'f': f.detach().cpu() if isinstance(f, torch.Tensor) else torch.tensor(f),
            'occupancy': occupancy,
            'kappa': kappa,
            'is_proton': is_proton,
        })

    def clear(self):
        self.states = []

    def compute_single_density(self, g, f):
        """计算单态的密度分量。"""
        rho_s = g ** 2 - f ** 2
        rho_v = g ** 2 + f ** 2
        return {'rho_s': rho_s, 'rho_v': rho_v}

    def compute_all(self):
        """
        计算所有占据态的总密度 (累加各态贡献 × 占据数)。
        不除以 4πr² — 留给调用者按需处理。
        """
        rho_s_total = torch.zeros(self.npt)
        rho_v_total = torch.zeros(self.npt)
        per_state_info = []

        for state in self.states:
            g = state['g'].float()
            f = state['f'].float()
            occ = state['occupancy']

            single = self.compute_single_density(g, f)
            rho_s_total += occ * single['rho_s']
            rho_v_total += occ * single['rho_v']

            per_state_info.append({
                'name': state['name'],
                'rho_s': single['rho_s'] * occ,
                'rho_v': single['rho_v'] * occ,
                'norm': torch.trapz(single['rho_v'], dx=self.dr).item() * occ,
            })

        total_particles = torch.trapz(rho_v_total, dx=self.dr).item()

        return {
            'rho_s': rho_s_total,
            'rho_v': rho_v_total,
            'total_particles': total_particles,
            'per_state': per_state_info,
        }


def compute_scf_densities(state_results, r_grid=None, dr=DR):
    """
    从 SCF 求解结果计算完整的核密度分布。

    严格按照 Fortran Density.f90 的约定:
      - 区分中子(it=1)和质子(it=2)密度
      - 除以 4πr²
      - r=0 点用外推: ρ(1) = 3ρ(2) - 3ρ(3) + ρ(4)

    参数:
        state_results: dict {state_name: {'g': tensor, 'f': tensor, ...}}
                      每个态还需包含 'degeneracy', 'is_proton' 信息
        r_grid: (N,) 径向网格 (默认用 R_GRID)
        dr: float, 步长
    返回:
        densities: dict {
            'rho_s_n', 'rho_v_n',  — 中子标量/矢量密度
            'rho_s_p', 'rho_v_p',  — 质子标量/矢量密度
            'rho_s', 'rho_v',      — 总密度
            'rho_v3',              — 同位旋矢量 ρ_n - ρ_p
        }
    """
    if r_grid is None:
        r_grid = torch.tensor(R_GRID, dtype=torch.float32)

    N = len(r_grid)
    r_safe = torch.clamp(r_grid, min=R_SAFE_OFFSET)

    # 初始化
    rho_s_n = torch.zeros(N)
    rho_v_n = torch.zeros(N)
    rho_s_p = torch.zeros(N)
    rho_v_p = torch.zeros(N)

    for name, res in state_results.items():
        g = res['g'] if isinstance(res['g'], torch.Tensor) else torch.tensor(res['g'], dtype=torch.float32)
        f = res['f'] if isinstance(res['f'], torch.Tensor) else torch.tensor(res['f'], dtype=torch.float32)
        deg = res.get('degeneracy', 2)  # μ = 2j+1
        is_proton = res.get('is_proton', False)

        # xvv = vv × μ (闭壳 vv=1)
        xvv = float(deg)

        # 密度累加 (不含 4πr² 因子)
        contrib_s = (g ** 2 - f ** 2) * xvv
        contrib_v = (g ** 2 + f ** 2) * xvv

        if is_proton:
            rho_s_p += contrib_s
            rho_v_p += contrib_v
        else:
            rho_s_n += contrib_s
            rho_v_n += contrib_v

    # 除以 4πr²
    inv_4pi_r2 = 1.0 / (4.0 * np.pi * r_safe ** 2)
    rho_s_n = rho_s_n * inv_4pi_r2
    rho_v_n = rho_v_n * inv_4pi_r2
    rho_s_p = rho_s_p * inv_4pi_r2
    rho_v_p = rho_v_p * inv_4pi_r2

    # r=0 点外推: ρ(1) = 3ρ(2) - 3ρ(3) + ρ(4)
    if N >= 4:
        for rho in [rho_s_n, rho_v_n, rho_s_p, rho_v_p]:
            rho[0] = 3.0 * rho[1] - 3.0 * rho[2] + rho[3]

    # 总密度和同位旋矢量
    rho_s = rho_s_n + rho_s_p
    rho_v = rho_v_n + rho_v_p
    rho_v3 = rho_v_n - rho_v_p  # ρ_n - ρ_p

    return {
        'rho_s_n': rho_s_n, 'rho_v_n': rho_v_n,
        'rho_s_p': rho_s_p, 'rho_v_p': rho_v_p,
        'rho_s': rho_s, 'rho_v': rho_v,
        'rho_v3': rho_v3,
    }


def compute_point_nucleon_density(Z, A, r_grid=R_GRID):
    """构建均匀球形核子密度分布 (Fermi 分布)。"""
    r = r_grid if isinstance(r_grid, torch.Tensor) else torch.tensor(r_grid, dtype=torch.float32)
    R0 = 1.12 * (A ** (1.0 / 3.0))
    a = 0.54
    rho_0 = 0.17
    arg = (r - R0) / a
    arg = torch.clamp(arg, min=-70, max=70)
    rho = rho_0 / (1.0 + torch.exp(arg))
    return rho


def compute_fermi_momentum_from_density(rho):
    """从密度计算局域 Fermi 动量 k_F(r) = (3π²ρ)^{1/3}"""
    return (3.0 * np.pi ** 2 * rho) ** (1.0 / 3.0)
