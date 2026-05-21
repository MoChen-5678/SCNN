"""
势场构造模块

功能:
  1. Hartree 直接势场（Yukawa Green 函数积分）
  2. 密度依赖耦合常数计算
  3. Dirac 自能组装 (V±S)
  4. 势场混合与迭代更新

势场来源: Shooting POT 文件 (含Fock交换项)
"""

import torch
import numpy as np
from config import (
    HBAR_C, DR, NPT, R_GRID, R_SAFE,
    M_NUCLEON, PKA1_PARAMS,
    TAU_Z, TAU_C,
)


def make_local_potentials(vps, vms, vtt=None):
    """
    创建只含局部势场的 potentials 字典 (Hartree近似, 无Fock交换项)。
    """
    pots = {'vps': vps, 'vms': vms}
    if vtt is not None:
        pots['vtt'] = vtt
    else:
        pots['vtt'] = torch.zeros_like(vps)
    pots['XG'] = torch.zeros_like(vps)
    pots['XF'] = torch.zeros_like(vps)
    pots['YG'] = torch.zeros_like(vps)
    pots['YF'] = torch.zeros_like(vps)
    return pots


# ═══════════════════════════════════════════════════════════════
#   密度依赖耦合常数
# ═══════════════════════════════════════════════════════════════

def compute_density_dependent_couplings(rho_v, pset=None):
    """
    计算密度依赖的耦合常数 g(ρ) 及其导数 dg/dρ。

    来自 Density.f90 第118-149行。

    参数:
        rho_v: (N,) 矢量密度 ρ_v (fm⁻³)
        pset: dict, 参数集 (默认 PKA1_PARAMS)
    返回:
        couplings: dict {
            'gsig', 'gome', 'grho', 'fpio', 'grtn': (N,) 耦合常数,
            'dsig', 'dome', 'drho', 'dpio', 'drtn': (N,) dg/dρ,
        }
    """
    if pset is None:
        pset = PKA1_PARAMS

    rho = rho_v if isinstance(rho_v, torch.Tensor) else torch.tensor(rho_v, dtype=torch.float32)
    rvs = pset['rvs']
    zeta = rho / rvs  # ρ/ρ_0

    def _rational_form(g0, a, b, c, d, zeta):
        """g(ρ) = g₀ · a · (1 + b·ξ²) / (1 + c·ξ²), ξ = ζ + δ"""
        xi = zeta + d
        xi2 = xi ** 2
        numer = 1.0 + b * xi2
        denom = 1.0 + c * xi2
        g = g0 * a * numer / denom
        # dg/dζ = g₀·a·(b-c)·2ξ / (1+c·ξ²)²  →  dg/dρ = dg/dζ / rvs
        dg = g0 * a * (b - c) * 2.0 * xi / (denom ** 2) / rvs
        return g, dg

    def _exponential_form(g0, a_exp, zeta):
        """g(ρ) = g₀ / exp(a·ζ),  dg/dρ = -a·g/rvs"""
        exp_term = torch.exp(a_exp * zeta)
        g = g0 / exp_term
        dg = -a_exp * g / rvs
        return g, dg

    gsig, dsig = _rational_form(pset['gsig'], pset['asig'], pset['bsig'], pset['csig'], pset['dsig'], zeta)
    gome, dome = _rational_form(pset['gome'], pset['aome'], pset['bome'], pset['come'], pset['dome'], zeta)
    grho, drho = _exponential_form(pset['grho'], pset['arho'], zeta)
    grtn, drtn = _exponential_form(pset['grtn'], pset['artn'], zeta)
    fpio, dpio = _exponential_form(pset['fpio'], pset['apio'], zeta)

    return {
        'gsig': gsig, 'gome': gome, 'grho': grho, 'fpio': fpio, 'grtn': grtn,
        'dsig': dsig, 'dome': dome, 'drho': drho, 'dpio': dpio, 'drtn': drtn,
    }


# ═══════════════════════════════════════════════════════════════
#   Yukawa Green 函数
# ═══════════════════════════════════════════════════════════════

def yukawa_green_function(r_grid, m_meson_MeV, hbc=HBAR_C):
    """
    计算 Yukawa Green 函数矩阵 G(r, r')。

    G(r,r';m) = m · [exp(-m|r-r'|) - exp(-m(r+r'))] / (4π|r-r'|)
              ≈ m/(4π) · [I_{1/2}(z<)·K_{1/2}(z>)] / √(rr')

    其中 z = m·r/ħc, m 为介子质量/ħc (fm⁻¹)

    简化: 对 I_{1/2} 和 K_{1/2} 有解析公式:
      I_{1/2}(z) = √(2/(πz)) · sinh(z)
      K_{1/2}(z) = √(π/(2z)) · exp(-z)

    所以: G(r,r') = m/(4π) · [exp(-m|r-r'|) - exp(-m(r+r'))] / (rr') · √(rr') · 1/(m/ħc)

    实际更直接的公式 (核物理常用):
      G(r,r') = exp(-m|r-r'|) / (4π|r-r'|)  (纯 Yukawa)

    参数:
        r_grid: (N,) 径向网格 (fm)
        m_meson_MeV: float, 介子质量 (MeV)
        hbc: float, ħc (MeV·fm)
    返回:
        W: (N, N) Green 函数矩阵 W[i,j] = G(r_i, r_j) · r_j²
    """
    m_fm = m_meson_MeV / hbc  # 质量转为 fm⁻¹
    N = len(r_grid)
    r = r_grid if isinstance(r_grid, torch.Tensor) else torch.tensor(r_grid, dtype=torch.float32)
    device = r.device

    ri = r.unsqueeze(1)  # (N, 1)
    rj = r.unsqueeze(0)  # (1, N)

    # |r - r'| 和 r + r'
    dr_abs = torch.abs(ri - rj).clamp(min=1e-10)
    r_sum = ri + rj

    # Yukawa: exp(-m|r-r'|)/(4π|r-r'|)
    # 减去 exp(-m(r+r'))/(4π|r-r'|) 以满足有限核边界条件
    W = (torch.exp(-m_fm * dr_abs) - torch.exp(-m_fm * r_sum)) / (4.0 * np.pi * dr_abs)

    # 乘以 r'² 用于积分 (Simpson/梯形法则的 r'² dr' 因子)
    W = W * rj ** 2

    return W


# ═══════════════════════════════════════════════════════════════
#   Hartree 势场计算
# ═══════════════════════════════════════════════════════════════

def compute_hartree_fields(rho_s, rho_v, rho_v3, r_grid, pset=None, dr=DR):
    """
    从密度计算 Hartree 介子场 (σ, ω, ρ, Coulomb)。

    来自 Meanfield.f90 第31-69行。

    参数:
        rho_s: (N,) 总标量密度 (fm⁻³)
        rho_v: (N,) 总矢量密度 (fm⁻³)
        rho_v3: (N,) 同位旋矢量密度 ρ_n - ρ_p (fm⁻³)
        r_grid: (N,) 径向网格 (fm)
        pset: dict, 参数集
        dr: float, 步长
    返回:
        fields: dict {'sig', 'ome', 'rho', 'cou'} — 介子场
    """
    if pset is None:
        pset = PKA1_PARAMS

    r = r_grid if isinstance(r_grid, torch.Tensor) else torch.tensor(r_grid, dtype=torch.float32)
    N = len(r)
    h6 = dr / 6.0  # Simpson 1/6 因子

    # 计算密度依赖耦合常数
    cct = compute_density_dependent_couplings(rho_v, pset)

    # 构建 Green 函数矩阵
    W_sig = yukawa_green_function(r, pset['amsig'])
    W_ome = yukawa_green_function(r, pset['amome'])
    W_rho = yukawa_green_function(r, pset['amrho'])

    # Coulomb: m=0 极限 → 1/(4π|r-r'|)
    # 用小质量近似 (1 MeV)
    W_cou = yukawa_green_function(r, 1.0)  # 近似

    # 积分: σ(n) = -Σ_i W_sig(n,i) · ρ_s(i) · g_σ(i) · h/6
    sig = -W_sig @ (rho_s * cct['gsig']) * h6
    ome =  W_ome @ (rho_v * cct['gome']) * h6
    rho =  W_rho @ (rho_v3 * cct['grho']) * h6

    # Coulomb: 只用质子密度 (rho_v_proton = (rho_v + rho_v3)/2)
    rho_p = (rho_v + rho_v3) / 2.0
    alphi = 137.03602  # 精细结构常数倒数
    cou = W_cou @ (rho_p * r ** 2 / alphi * 4.0 * np.pi) * h6

    return {
        'sig': sig,
        'ome': ome,
        'rho': rho,
        'cou': cou,
    }


def compute_rearrangement(rho_s, rho_v, rho_v3, fields, cct, pset=None):
    """
    计算重排项 Σ_R (密度依赖耦合常数导致)。

    来自 Meanfield.f90 第214-293行。
    """
    if pset is None:
        pset = PKA1_PARAMS
    rvs = pset['rvs']

    SigR = (rho_s * cct['dsig'] * fields['sig'] +
            rho_v * cct['dome'] * fields['ome'] +
            rho_v3 * cct['drho'] * fields['rho'])

    SigR = SigR / rvs  # Fortran: SigR = SigR/pset%rvs
    return SigR


def assemble_dirac_potentials(fields, SigR, cct, it, pset=None):
    """
    组装 Dirac 方程的势场 V±S。

    来自 PotelHF.f90 第57-76行。

    参数:
        fields: dict from compute_hartree_fields()
        SigR: (N,) 重排项
        cct: dict from compute_density_dependent_couplings()
        it: int, 1=中子, 2=质子
        pset: dict, 参数集
    返回:
        dict: {'vps', 'vms', 'vtt'}
    """
    if pset is None:
        pset = PKA1_PARAMS

    SS = cct['gsig'] * fields['sig']
    VS = cct['gome'] * fields['ome'] + SigR
    VT = cct['grho'] * fields['rho']
    TT = torch.zeros_like(SS)  # ρ-张量暂不实现
    emcc = pset['amu'][it - 1] / HBAR_C * 2.0

    tau_z = TAU_Z[it - 1]
    tau_c = TAU_C[it - 1]

    vps = VS + SS + tau_z * VT + tau_c * fields['cou']
    vms = VS - SS + tau_z * VT + tau_c * fields['cou'] - emcc
    vtt = TT * tau_z

    return {'vps': vps, 'vms': vms, 'vtt': vtt}


# ═══════════════════════════════════════════════════════════════
#   SCF 势场更新 (Phase 2)
# ═══════════════════════════════════════════════════════════════

def update_potentials_from_density(densities, isotope='16O', pset=None):
    """
    从密度计算完整的 Hartree 势场 (用于 SCF 迭代)。

    参数:
        densities: dict from density.compute_scf_densities()
        isotope: str
        pset: dict, 参数集
    返回:
        potentials_neutron: dict {'vps', 'vms', 'vtt'}  — 中子势场
        potentials_proton: dict {'vps', 'vms', 'vtt'}  — 质子势场
    """
    from config import ISOTOPE_CONFIG

    if pset is None:
        pset = PKA1_PARAMS

    cfg = ISOTOPE_CONFIG[isotope]
    Z = cfg['Z']

    r = torch.tensor(R_GRID, dtype=torch.float32)

    # 提取密度
    rho_s = densities['rho_s']  # 总标量密度
    rho_v = densities['rho_v']  # 总矢量密度
    rho_v3 = densities['rho_v3']  # 同位旋矢量密度 ρ_n - ρ_p

    # 计算介子场
    fields = compute_hartree_fields(rho_s, rho_v, rho_v3, r, pset)

    # 密度依赖耦合
    cct = compute_density_dependent_couplings(rho_v, pset)

    # 重排项
    SigR = compute_rearrangement(rho_s, rho_v, rho_v3, fields, cct, pset)

    # 分别为中子和质子组装势场
    pots_n = assemble_dirac_potentials(fields, SigR, cct, it=1, pset=pset)
    pots_p = assemble_dirac_potentials(fields, SigR, cct, it=2, pset=pset)

    return pots_n, pots_p


def mix_potentials(V_old, V_new, xmix):
    """
    势场混合 (SCF稳定化)。

    V_mix = (1-xmix)·V_old + xmix·V_new

    来自 PotelHF.f90 第84-87行。
    """
    V_mix = {}
    for key in ['vps', 'vms', 'vtt']:
        V_mix[key] = (1.0 - xmix) * V_old[key] + xmix * V_new[key]
    return V_mix


def potential_difference(V1, V2):
    """计算两组势场的最大差异 (用于收敛判断)。"""
    diffs = []
    for key in ['vps', 'vms', 'vtt']:
        diff = torch.max(torch.abs(V1[key] - V2[key])).item()
        diffs.append(diff)
    return max(diffs)
