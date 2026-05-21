"""
边界条件和物理约束模块

包含:
  1. 波函数归一化约束: ∫(G²+F²)dr = 1
  2. r→0 精确渐近行为约束: G ~ r^{l_u+1}, F ~ r^{l_d+1}  (核物理教材 Eq.3.61)
  3. r=R 截断边界条件: G(R)=0, F(R)=C_R  (核物理教材 Eq.3.60)
  4. 正动能约束: F 不能主导 (避免连续谱解)
  5. 节点数约束 (可选, 需要参考节点数)

参考文献: 核物理(1) §3.3, Eq.(3.60)-(3.61)
  - r→0: G(r) = C₀·r^{l_u+1},  F(r) = C₀'·r^{l_d+1}
  - r=R: G(R) = 0,              F(R) = C_R (非零常数)
  - l_u, l_d 为 Dirac 旋量上/下分量对应的轨道角动量
"""

import torch
import numpy as np
from config import DR, NPT, R_GRID, R_SAFE_OFFSET


# ═══════════════════════════════════════════════════════════════
#   1. 归一化约束
# ═══════════════════════════════════════════════════════════════

def loss_normalization(g, f, dr=DR, target_norm=1.0):
    """归一化损失: ∫(G²+F²)dr = target_norm"""
    integrand = g**2 + f**2
    norm_sq = torch.trapz(integrand, dim=-1, dx=dr)
    loss = (norm_sq - target_norm) ** 2
    return loss


def normalize_wavefunction(g, f, dr=DR):
    """硬归一化"""
    integrand = g**2 + f**2
    norm = torch.sqrt(torch.trapz(integrand, dim=-1, dx=dr).clamp(min=1e-30))
    if norm.dim() == 0:
        return g / norm, f / norm
    elif g.dim() == 1:
        return g / norm, f / norm
    else:
        scale = norm.unsqueeze(-1)
        return g * scale, f * scale


# ═══════════════════════════════════════════════════════════════
#   2. 角动量辅助函数
# ═══════════════════════════════════════════════════════════════

def get_angular_momenta(kappa):
    """
    从 κ 计算 G 和 F 对应的轨道角动量 l_u 和 l_d。

    ★ 核物理教材定义:
      - κ < 0: l_u = |κ| - 1,  l_d = |κ|
      - κ > 0: l_u = κ,         l_d = κ - 1

    示例:
      - 1s₁/₂ (κ=-1): l_u=0(s), l_d=1(p)  → G~r^1, F~r^2
      - 1p₃/₂ (κ=-2): l_u=1(p), l_d=2(d)  → G~r^2, F~r^3
      - 1p₁/₂ (κ=+1): l_u=1(p), l_d=0(s)  → G~r^2, F~r^1
      - 1d₅/₂ (κ=-3): l_u=2(d), l_d=3(f)  → G~r^3, F~r^4
      - 1d₃/₂ (κ=+2): l_u=2(d), l_d=1(p)  → G~r^3, F~r^2
    """
    k = int(kappa)
    if k < 0:
        l_u = abs(k) - 1   # G 的轨道角动量
    else:
        l_u = k
    # κ>0 时 l_d = κ-1
    if k > 0:
        l_d = k - 1
    else:
        l_d = abs(k)
    return l_u, l_d


# ═══════════════════════════════════════════════════════════════
#   2b. r=R 截断边界条件 (核物理教材 Eq.3.60)
# ═══════════════════════════════════════════════════════════════

def loss_boundary_R(g, f, weight=1.0):
    """
    r=R 截断边界条件损失 (核物理教材 Eq.3.60)。

    ★ 精确边界条件:
      G(R) = 0 (束缚态大分量在截断处必须为零)
      F(R) = C_R (小分量可以非零)

    实现方式: 对 G 的最后几个网格点施加趋零约束。
    """
    has_batch = g.dim() == 2
    if not has_batch:
        g = g.unsqueeze(0)
        f = f.unsqueeze(0)

    B, N = g.shape

    # G 在最后几个点应该趋于零
    n_tail = min(2, N)
    g_tail = g[:, -n_tail:]

    # 用绝对值损失: |G_tail|² 应很小
    loss_g = torch.mean(g_tail ** 2)

    # F 不需要约束为零 (F(R) = C_R 非零)

    loss = weight * loss_g
    if not has_batch:
        return loss.squeeze(0)
    return loss


# ═══════════════════════════════════════════════════════════════
#   4. 正动能约束
# ═══════════════════════════════════════════════════════════════

def loss_kinetic_positive(g, f, kappa=None, threshold=0.8, weight=0.5):
    """F分量不应超过G分量的阈值倍（只在G有足够振幅的区域检查）
    
    避免在 G→0 的区域（节点、r→0、r→R）计算 F/G 导致除零爆炸。
    """
    has_batch = g.dim() == 2
    if not has_batch:
        g = g.unsqueeze(0)
        f = f.unsqueeze(0)

    B, N = g.shape
    
    # 只在 G 有足够振幅的区域检查 (G > 5% of max |G|)
    g_max = torch.max(torch.abs(g), dim=-1, keepdim=True)[0].clamp(min=1e-10)
    mask = torch.abs(g) > 0.05 * g_max  # (B, N) bool
    
    g_abs = torch.abs(g) + 1e-10
    f_over_g = torch.abs(f) / g_abs
    violation = torch.clamp(f_over_g - threshold, min=0)
    # 只计算有效区域的违规
    loss = weight * (violation * mask.float()).sum() / (mask.float().sum() + 1e-10)

    if not has_batch:
        return loss.squeeze(0)
    return loss


# ═══════════════════════════════════════════════════════════════
#   5. 节点数约束
# ═══════════════════════════════════════════════════════════════

def count_nodes(wavefunc, dr=DR):
    """计算波函数节点数（过零次数/2）"""
    signs = torch.sign(wavefunc)
    sign_changes = (signs[1:] != signs[:-1]).float()
    amplitude_threshold = torch.max(torch.abs(wavefunc)) * 0.01
    valid_region = torch.abs(wavefunc[1:]) > amplitude_threshold
    nodes = (sign_changes * valid_region).sum().int().item()
    return nodes // 2


def expected_nodes(n_principal, kappa):
    """预期G分量节点数"""
    nr = n_principal - 1
    l, _ = get_angular_momenta(kappa)
    if l == 0:
        return 0
    return nr


def loss_node_count(g, n_expected, dr=DR, weight=0.1):
    """节点数匹配损失 — 可微版本.
    用 G·G_shifted 的正部分之和作为节点数的可微近似:
    每个节点处 G 在零交叉，G[i]*G[i+1] < 0，所以 -min(G[i]*G[i+1], 0) 反映节点强度.
    """
    # 可微节点数估计: 负叉积的总和
    cross = g[..., :-1] * g[..., 1:]  # 相邻点乘积
    # 节点处 cross < 0, 取 -min(cross, 0) 的积分
    node_indicator = torch.clamp(-cross, min=0.0)
    n_approx = torch.trapz(node_indicator, dx=dr)  # 近似节点"强度"

    # 目标: 对 n_expected=0 的态, node_indicator 应为 0
    #        对 n_expected>0 的态, node_indicator 应与参考态匹配
    # 简化: 直接惩罚 node_indicator 偏离目标
    # 对于 n_expected=0: 直接惩罚所有节点
    # 对于 n_expected>0: 用硬约束辅助
    
    # 软约束: 惩罚节点数差异 (不可微部分作辅助)
    n_actual = count_nodes(g, dr)
    diff = float(n_actual - n_expected)
    
    # 可微部分: n_expected=0 时完全禁止节点, n_expected>0 时需要至少一定节点强度
    if n_expected == 0:
        # 禁止节点: 惩罚所有叉积<0的区域
        loss_diff = weight * 100.0 * n_approx
    else:
        # 需要节点: 如果实际节点不够, 加大惩罚
        loss_diff = weight * (diff ** 2)
    
    return loss_diff


# ═══════════════════════════════════════════════════════════════
#   总边界损失组合
# ═══════════════════════════════════════════════════════════════

def compute_total_boundary_loss(g, f, kappa, dr=DR,
                                 w_norm=10.0,
                                 w_R=5.0,
                                 w_kin=0.5, w_node=0.1,
                                 n_expected_nodes=None):
    """计算所有边界条件的加权总和
    
    包含核物理教材精确边界条件:
      - Eq.3.61: r→0 渐近行为 G~r^{l_u+1}, F~r^{l_d+1} (已嵌入网络结构)
      - Eq.3.60: r=R 截断处 G(R)=0
    """
    L_norm = w_norm * loss_normalization(g, f, dr)
    L_R = w_R * loss_boundary_R(g, f)
    L_kin = w_kin * loss_kinetic_positive(g, f, kappa)

    L_node = torch.tensor(0.0, device=g.device)
    if n_expected_nodes is not None:
        L_node = w_node * loss_node_count(g, n_expected_nodes, dr)

    total = L_norm + L_R + L_kin + L_node

    return {
        'total': total,
        'norm': L_norm,
        'boundary_R': L_R,
        'kin': L_kin,
        'node': L_node,
    }


# ══════════════════════════════════════════════════════════════
#   6. 波函数正交归一化约束 (用于激发态求解)
# ══════════════════════════════════════════════════════════════

def loss_orthonormal(g, f, ref_wavefunctions, dr=DR, weight=10.0):
    """
    波函数正交归一化损失。
    
    在求解激发态时使用，确保激发态波函数与所有已求解的基态波函数正交:
        ∫(G_i·G_ref + F_i·F_ref) dr = 0   (对于 i ≠ ref)
    
    参数:
        g: (B, N) 或 (N,), 当前态的大分量波函数
        f: (B, N) 或 (N,), 当前态的小分量波函数
        ref_wavefunctions: list of dict, 参考波函数列表, 每个dict含 {'g': tensor, 'f': tensor}
        dr: float, 径向步长
        weight: float, 正交性损失的权重
    
    返回:
        loss_ortho: 标量, 所有参考波函数的正交性损失之和
    """
    if not ref_wavefunctions:
        return torch.tensor(0.0, device=g.device)
    
    has_batch = g.dim() == 2
    if not has_batch:
        g = g.unsqueeze(0)
        f = f.unsqueeze(0)
    
    B, N = g.shape
    device = g.device
    
    loss_ortho = torch.tensor(0.0, device=device)
    
    for ref in ref_wavefunctions:
        g_ref = ref['g'].to(device)
        f_ref = ref['f'].to(device)
        
        # 确保形状匹配
        if g_ref.dim() == 1:
            g_ref = g_ref.unsqueeze(0).expand(B, -1)
            f_ref = f_ref.unsqueeze(0).expand(B, -1)
        
        # 计算重叠积分: ∫(G·G_ref + F·F_ref) dr
        overlap = torch.trapz(g * g_ref + f * f_ref, dim=-1, dx=dr)
        
        # 正交性约束: overlap 应该为 0
        loss_ortho = loss_ortho + weight * (overlap ** 2).mean()
    
    if not has_batch:
        return loss_ortho.squeeze(0)
    return loss_ortho


def compute_orthonormal_loss(g, f, ref_wavefunctions, dr=DR, weight=10.0):
    """
    便捷函数: 计算正交归一化损失。
    
    参数:
        g, f: 当前态的波函数
        ref_wavefunctions: 参考波函数列表 [{'g': tensor, 'f': tensor, 'name': str}, ...]
        dr: 径向步长
        weight: 损失权重
    
    返回:
        loss: 标量损失值
    """
    return loss_orthonormal(g, f, ref_wavefunctions, dr, weight)
