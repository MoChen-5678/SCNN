"""
PINN-RHF 主训练程序

支持两种模式:
  1. MVP 模式: 单态固定 Woods-Saxon 势场下的 Dirac 方程求解
  2. SCF 模式: 完整自洽循环 (密度↔势场迭代)

用法:
    # MVP: 单态求解
    python train.py --mode mvp --isotope 16O --state 1s1/2 --epochs 2000
    
    # SCF: 自洽循环
    python train.py --mode scf --isotope 16O --max-scf 30

MVP 验证标准:
    - PINN 能量 vs Fortran 参考解误差 < 1%
    - 波形 L² 误差 < 5%
"""

import os
import sys
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DEVICE, SEED, DR, NPT, R_GRID, R_SAFE, R_SAFE_OFFSET,
    TRAIN_CONFIG, LOSS_WEIGHTS, SCF_CONFIG, ISOTOPE_CONFIG,
    MODEL_CONFIG, HBAR_C,
    set_seed, get_device, get_r_grid, get_r_safe,
)
from model import DiracNet, MultiStateDiracNet
from pde_residuals import (
    compute_dirac_residual, build_5padf_matrix, get_fd_directions,
    make_local_potentials, clear_fd_cache, apply_nonlocal_kernels, has_nonlocal_kernels,
)
from density import DensityCalculator


def compute_energy_rayleigh(g, f, kappa, potentials, dr=DR, npt=NPT, device=None):
    """
    从 Dirac 方程直接提取能量（Rayleigh 商积分形式，避免逐点除法）。

    Dirac 方程（含Fock交换势）：
      dG/dr = -(κ/r+Vtt)G + (ε-Vms)F - ∫dr'[XG G + XF F]
      dF/dr = +(κ/r+Vtt)F - (ε-Vps)G + ∫dr'[YG G + YF F]

    Rayleigh商推导：
      ε·∫(G²+F²)dr = ∫[F·dG/dr - G·dF/dr + 2(κ/r+Vtt)GF
                        + (Vps+YG)·G² + (Vms+XF)·F² + (XG+YF)·GF]dr
    其中 ε = E/ħc (fm⁻¹)，所有势场 fm⁻¹。
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
        XG  = potentials.get('XG', torch.zeros_like(g)).to(device) / hbc
        XF  = potentials.get('XF', torch.zeros_like(g)).to(device) / hbc
        YG  = potentials.get('YG', torch.zeros_like(g)).to(device) / hbc
        YF  = potentials.get('YF', torch.zeros_like(g)).to(device) / hbc

    # 导数 (5PADF)
    from pde_residuals import build_5padf_matrix, apply_fd_matrix, get_fd_directions
    g_dir, f_dir = get_fd_directions(kappa)
    D_g = build_5padf_matrix(N, dr, g_dir, device=device, dtype=g.dtype)
    D_f = build_5padf_matrix(N, dr, f_dir, device=device, dtype=g.dtype)
    dg_dr = apply_fd_matrix(g, D_g)
    df_dr = apply_fd_matrix(f, D_f)

    # r 和 κ/r
    r = torch.arange(N, device=device, dtype=g.dtype) * dr
    r_safe = torch.clamp(r, min=R_SAFE_OFFSET)
    kap_vtt = float(kappa) / r_safe.unsqueeze(0) + vtt  # (1, N)

    # 归一化 ∫(G²+F²)dr
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

    E_hc   = numerator / norm_int          # fm⁻¹
    E_MeV  = E_hc * hbc                     # MeV
    return E_MeV.squeeze() if E_MeV.dim() > 0 else E_MeV


# ═══════════════════════════════════════════════════════════════
#   态定义辅助
# ═══════════════════════════════════════════════════════════════

# 常见态的 (name, kappa, n_principal, expected_energy_MeV) 定义
STATE_DEFS = {
    # s 态 (κ=-1, j=1/2, deg=2)
    '1s1/2':  {'kappa': -1, 'n_pr': 1, 'E_ref': -36.7, 'nodes_g': 0},
    '2s1/2':  {'kappa': -1, 'n_pr': 2, 'E_ref':  -3.3, 'nodes_g': 1},
    '3s1/2':  {'kappa': -1, 'n_pr': 3, 'E_ref':   1.0, 'nodes_g': 2},
    '4s1/2':  {'kappa': -1, 'n_pr': 4, 'E_ref':   3.6, 'nodes_g': 3},
    '5s1/2':  {'kappa': -1, 'n_pr': 5, 'E_ref':   7.7, 'nodes_g': 4},
    '6s1/2':  {'kappa': -1, 'n_pr': 6, 'E_ref':  13.1, 'nodes_g': 5},
    # p3/2 态 (κ=-2, j=3/2, deg=4)
    '1p3/2':  {'kappa': -2, 'n_pr': 2, 'E_ref': -20.1, 'nodes_g': 1},
    '2p3/2':  {'kappa': -2, 'n_pr': 3, 'E_ref':   1.0, 'nodes_g': 2},
    '3p3/2':  {'kappa': -2, 'n_pr': 4, 'E_ref':   2.9, 'nodes_g': 3},
    '4p3/2':  {'kappa': -2, 'n_pr': 5, 'E_ref':   6.0, 'nodes_g': 4},
    '5p3/2':  {'kappa': -2, 'n_pr': 6, 'E_ref':  10.4, 'nodes_g': 5},
    '6p3/2':  {'kappa': -2, 'n_pr': 7, 'E_ref':  16.1, 'nodes_g': 6},
    # d5/2 态 (κ=-3, j=5/2, deg=6)
    '1d5/2':  {'kappa': -3, 'n_pr': 3, 'E_ref':  -5.3, 'nodes_g': 2},
    '2d5/2':  {'kappa': -3, 'n_pr': 4, 'E_ref':   1.7, 'nodes_g': 3},
    '3d5/2':  {'kappa': -3, 'n_pr': 5, 'E_ref':   4.5, 'nodes_g': 4},
    '4d5/2':  {'kappa': -3, 'n_pr': 6, 'E_ref':   8.4, 'nodes_g': 5},
    '5d5/2':  {'kappa': -3, 'n_pr': 7, 'E_ref':  13.5, 'nodes_g': 6},
    # f7/2 态 (κ=-4, j=7/2, deg=8)
    '1f7/2':  {'kappa': -4, 'n_pr': 4, 'E_ref':   2.5, 'nodes_g': 3},
    '2f7/2':  {'kappa': -4, 'n_pr': 5, 'E_ref':   5.2, 'nodes_g': 4},
    '3f7/2':  {'kappa': -4, 'n_pr': 6, 'E_ref':   7.3, 'nodes_g': 5},
    '4f7/2':  {'kappa': -4, 'n_pr': 7, 'E_ref':  11.1, 'nodes_g': 6},
    '5f7/2':  {'kappa': -4, 'n_pr': 8, 'E_ref':  16.6, 'nodes_g': 7},
    # p1/2 态 (κ=+1, j=1/2, deg=2)
    '1p1/2':  {'kappa': +1, 'n_pr': 2, 'E_ref': -14.1, 'nodes_g': 0},
    '2p1/2':  {'kappa': +1, 'n_pr': 3, 'E_ref':   1.1, 'nodes_g': 1},
    '3p1/2':  {'kappa': +1, 'n_pr': 4, 'E_ref':   3.2, 'nodes_g': 2},
    '4p1/2':  {'kappa': +1, 'n_pr': 5, 'E_ref':   6.5, 'nodes_g': 3},
    '5p1/2':  {'kappa': +1, 'n_pr': 6, 'E_ref':  10.9, 'nodes_g': 4},
    '6p1/2':  {'kappa': +1, 'n_pr': 7, 'E_ref':  16.6, 'nodes_g': 5},
    # d3/2 态 (κ=+2, j=3/2, deg=4)
    '1d3/2':  {'kappa': +2, 'n_pr': 3, 'E_ref':   0.6, 'nodes_g': 1},
    '2d3/2':  {'kappa': +2, 'n_pr': 4, 'E_ref':   1.9, 'nodes_g': 2},
    '3d3/2':  {'kappa': +2, 'n_pr': 5, 'E_ref':   4.9, 'nodes_g': 3},
    '4d3/2':  {'kappa': +2, 'n_pr': 6, 'E_ref':   9.1, 'nodes_g': 4},
    '5d3/2':  {'kappa': +2, 'n_pr': 7, 'E_ref':  14.4, 'nodes_g': 5},
    # f5/2 态 (κ=+3, j=5/2, deg=6)
    '1f5/2':  {'kappa': +3, 'n_pr': 4, 'E_ref':   2.5, 'nodes_g': 2},
    '2f5/2':  {'kappa': +3, 'n_pr': 5, 'E_ref':   5.5, 'nodes_g': 3},
    '3f5/2':  {'kappa': +3, 'n_pr': 6, 'E_ref':   8.9, 'nodes_g': 4},
    '4f5/2':  {'kappa': +3, 'n_pr': 7, 'E_ref':  12.8, 'nodes_g': 5},
    '5f5/2':  {'kappa': +3, 'n_pr': 8, 'E_ref':  18.1, 'nodes_g': 6},
    # g7/2 态 (κ=+4, j=7/2, deg=8)
    '1g7/2':  {'kappa': +4, 'n_pr': 5, 'E_ref':   3.5, 'nodes_g': 3},
    '2g7/2':  {'kappa': +4, 'n_pr': 6, 'E_ref':   7.1, 'nodes_g': 4},
    '3g7/2':  {'kappa': +4, 'n_pr': 7, 'E_ref':  11.5, 'nodes_g': 5},
    '4g7/2':  {'kappa': +4, 'n_pr': 8, 'E_ref':  16.7, 'nodes_g': 6},
}


def parse_state_name(state_str):
    """
    解析态名称字符串为物理参数。
    
    支持: "1s1/2", "1p3/2", "1d5/2", "n-1s1/2", "p-1p3/2" 等格式
    """
    # 从 STATE_DEFS 查找
    if state_str in STATE_DEFS:
        return STATE_DEFS[state_str]
    
    # 去掉 n-/p- 前缀 (中子/质子标识, 物理参数相同)
    clean = state_str
    if clean.startswith('n-') or clean.startswith('p-'):
        clean = clean[2:]
    
    if clean in STATE_DEFS:
        return STATE_DEFS[clean]
    
    # 尝试解析: nlj 格式
    # n=主量子数, l={s,p,d,f,g}, j=l±1/2
    import re
    m = re.match(r'(\d)([spdfg])(\d)\/(\d)', clean)
    if m:
        n = int(m.group(1))
        l_map = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4}
        l = l_map[m.group(2)]
        j_up = int(m.group(3)) * 2  # 分子
        j_dn = int(m.group(4))       # 分母
        j = j_up / j_dn
        
        # κ 的确定:
        #   j = l + 1/2 → κ = -(l+1) = -(2j-1)/2 ... 不对
        #   正确: κ = ±(j+1/2), sign = (-1)^{j-l-1/2}
        if abs(j - l - 0.5) < 0.01:  # j = l+1/2 → aligned → κ < 0
            kappa = -(l + 1)
        else:                          # j = l-1/2 or |l-1/2| → unaligned → κ > 0
            kappa = l
        
        return {
            'kappa': kappa,
            'n_pr': n,
            'E_ref': None,
            'nodes_g': max(0, n-1) if l > 0 else 0,
        }
    
    raise ValueError(f"无法解析态名称: {state_str}, 可用: {list(STATE_DEFS.keys())}")


# ═══════════════════════════════════════════════════════════════
#   MVP 训练器
# ═══════════════════════════════════════════════════════════════

class MVPSolver:
    """
    MVP 求解器: 在固定势场下用 PINN 求解单个 Dirac 态。
    
    这是验证 PINN-RHF 可行性的最小可行产品。
    支持激发态求解：通过 ref_wavefunctions 参数传入基态波函数，
    计算正交性损失确保激发态与基态正交。
    """

    def __init__(self, isotope='16O', state='1s1/2', device=None,
                 n_hidden=128, n_layers=6, activation='swish',
                 lr=1e-3, max_epochs=2000,
                 lambda_pde=1.0, lambda_norm=10.0, lambda_r0=1.0, 
                 lambda_inf=1.0, lambda_R=1.0, lambda_kin=0.1, f_weight=3.0,
                 adam_betas=(0.9, 0.999), grad_clip=1.0,
                 early_stop_patience=300,
                 lambda_ortho=10.0, ref_wavefunctions=None):
        """
        参数:
            lambda_ortho: 正交性损失权重 (默认10.0)
            ref_wavefunctions: 参考波函数列表, 每个元素为 dict:
                {'g': tensor, 'f': tensor, 'name': str}
                用于激发态求解时确保与基态正交
        """
        self.isotope = isotope
        self.state_str = state
        self.device = device or DEVICE
        # 超参数
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.activation = activation
        self.lr = lr
        self.max_epochs = max_epochs
        self.f_weight = f_weight
        self.adam_betas = adam_betas
        self.grad_clip = grad_clip
        self.early_stop_patience = early_stop_patience
        # BC权重 (与DiracPINNSolver一致)
        self.w_norm = lambda_norm
        self.w_r0 = lambda_r0
        self.w_inf = lambda_inf
        self.w_R = lambda_R if lambda_R != 1.0 else 5.0
        self.w_kin = lambda_kin
        self.w_node = 5.0
        # 正交性损失权重
        self.w_ortho = lambda_ortho
        self.ref_wavefunctions = ref_wavefunctions or []

        # 解析态参数
        self.state_info = parse_state_name(state)
        self.kappa = self.state_info['kappa']
        self.n_pr = self.state_info['n_pr']
        
        # 构建势场
        self.potentials = build_mvp_potentials(isotope)
        for key in self.potentials:
            if isinstance(self.potentials[key], torch.Tensor):
                self.potentials[key] = self.potentials[key].to(self.device)
        
        # 径向网格
        self.r_grid = get_r_grid(self.device)
        self.r_safe = get_r_safe(self.device)

        # 初始化网络 (E_init=-50，与DiracPINNSolver一致)
        self.net = DiracNet(
            n_hidden=self.n_hidden,
            n_layers=self.n_layers,
            activation=self.activation,
            hard_normalize=True,
            init_energy=-50.0,
        ).to(self.device)

        # 训练历史
        self.history = {
            'loss_total': [], 'loss_pde': [], 'loss_norm': [], 
            'loss_bc': [], 'loss_ortho': [], 'energy': [],
        }

    def compute_loss(self, r_input, target_nodes=0):
        """计算完整 PINN 损失（与DiracPINNSolver一致）
        
        包含:
            - PDE残差损失
            - 边界条件损失 (归一化、r->0、r->R、r->inf、动能正定性、节点数)
            - 节点数硬约束
            - 正交归一化损失 (如果提供了参考波函数)
        """
        # 1. 网络前向
        g, f = self.net(r_input, kappa=self.kappa, dr=DR)

        # 2. PDE残差 (无额外lambda_pde系数)
        loss_pde = compute_dirac_residual(
            g, f, self.net.E, self.kappa, self.potentials,
            dr=DR, npt=NPT, f_weight=3.0,
            return_components=False, device=self.device,
        )

        # 3. 边界条件 (固定权重，与对比程序一致)
        bc_losses = compute_total_boundary_loss(
            g, f, self.kappa, dr=DR,
            w_norm=self.w_norm,
            w_R=self.w_R,
            w_kin=self.w_kin, w_node=self.w_node,
            n_expected_nodes=target_nodes,
        )
        loss_bc = bc_losses['total']

        # 4. 节点数硬约束: 不对就1000倍惩罚 (用boundary_conditions的count_nodes)
        with torch.no_grad():
            g_np = g.squeeze(0).cpu().numpy()
            n_actual = count_nodes(g_np)
        node_diff = abs(n_actual - target_nodes)
        loss_node_hard = 1000.0 * float(node_diff)

        # 5. 正交归一化损失 (用于激发态求解)
        loss_ortho = torch.tensor(0.0, device=self.device)
        if self.ref_wavefunctions:
            loss_ortho = compute_orthonormal_loss(
                g, f, self.ref_wavefunctions, dr=DR, weight=self.w_ortho
            )

        # 总损失
        loss_total = loss_pde + loss_bc + loss_node_hard + loss_ortho

        info = {
            'loss_total': loss_total.item(),
            'loss_pde': loss_pde.item(),
            'loss_ortho': loss_ortho.item() if isinstance(loss_ortho, torch.Tensor) else loss_ortho,
            'energy': self.net.E.item(),
            'n_actual': n_actual,
            'node_diff': node_diff,
        }
        return loss_total, info

    def train(self, epochs=None, lr=None, verbose=True, print_every=500,
              E_tol=1e-10, patience=500, target_nodes=None):
        """训练 PINN：保留最终波函数，Rayleigh商作为收敛基准"""
        clear_fd_cache()

        epochs = epochs or self.max_epochs
        lr = lr or self.lr
        if target_nodes is None:
            target_nodes = self.state_info.get('nodes_g', 0)

        optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=lr, betas=(0.9, 0.999),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )

        start_time = time.time()
        E_ray_prev = None   # Rayleigh 商作为能量基准
        converge_count = 0
        ray_interval = max(print_every, 100)  # Rayleigh 计算间隔

        r_input = self.r_grid.unsqueeze(0)  # (1, N)

        for epoch in range(epochs):
            optimizer.zero_grad()

            loss, info = self.compute_loss(r_input, target_nodes=target_nodes)

            loss.backward()

            # 梯度裁剪
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.grad_clip
                )

            optimizer.step()
            scheduler.step()

            E_net = info['energy']
            n_actual = info['n_actual']
            node_diff = info['node_diff']

            # 记录历史（网络参数E用于参考）
            self.history['loss_total'].append(info['loss_total'])
            self.history['loss_pde'].append(info['loss_pde'])
            self.history['energy'].append(E_net)

            # ★ 定期计算 Rayleigh 商（作为真实能量基准）
            do_print = verbose and (epoch % print_every == 0 or epoch == epochs - 1 or epoch == 0)
            do_rayleigh = do_print or (epoch % ray_interval == 0)
            if do_rayleigh:
                with torch.no_grad():
                    g_tmp, f_tmp = self.net(r_input, kappa=self.kappa, dr=DR)
                    E_ray = compute_energy_rayleigh(
                        g_tmp, f_tmp, self.kappa, self.potentials,
                        dr=DR, npt=NPT, device=self.device,
                    )
                E_ray_val = E_ray.item() if E_ray.dim() == 0 else E_ray[0].item()
                self.history.setdefault('energy_rayleigh', []).append((epoch, E_ray_val))
            else:
                E_ray_val = E_ray_prev  # 上次的值

            # 打印
            if do_print:
                print(f"  Ep{epoch:5d}/{epochs} L={info['loss_total']:.2e} "
                      f"E_net={E_net:+.6f}MeV E_Ray={E_ray_val:+.4f}MeV "
                      f"nodes={n_actual}/{target_nodes}")

            # ★ 收敛判断：用 Rayleigh 能量变化
            if E_ray_prev is not None:
                dE_ray = abs(E_ray_val - E_ray_prev)
                if dE_ray < E_tol and node_diff == 0:
                    converge_count += 1
                    if converge_count >= patience:
                        if verbose:
                            print(f"  Converged at Ep {epoch}: |dE_Ray|={dE_ray:.2e} "
                                  f"E_Ray={E_ray_val:+.4f}MeV nodes={n_actual}/{target_nodes}")
                        break
                else:
                    converge_count = 0
            E_ray_prev = E_ray_val

        total_time = time.time() - start_time

        # ★ 不恢复 best_state —— 保留迭代最终的波函数和能量
        with torch.no_grad():
            g_final, f_final = self.net(r_input, kappa=self.kappa, dr=DR)
            E_rayleigh = compute_energy_rayleigh(
                g_final, f_final, self.kappa, self.potentials,
                dr=DR, npt=NPT, device=self.device,
            )
        self.E_rayleigh = E_rayleigh.item() if E_rayleigh.dim() == 0 else E_rayleigh[0].item()

        if verbose:
            print(f"  Done {total_time:.1f}s E_net={self.net.E.item():+.4f}MeV "
                  f"E_Rayleigh={self.E_rayleigh:+.4f}MeV nodes={n_actual}/{target_nodes}")

        return self.history

    def evaluate(self):
        """评估训练结果，返回波函数和诊断信息（能量为Rayleigh商）"""
        with torch.no_grad():
            r = self.r_grid.unsqueeze(0)
            g, f = self.net(r, kappa=self.kappa, dr=DR)

            # ★ Rayleigh 商：最终物理能量
            E_rayleigh = compute_energy_rayleigh(
                g, f, self.kappa, self.potentials,
                dr=DR, npt=NPT, device=self.device,
            )
            if hasattr(self, 'E_rayleigh'):
                E_final = self.E_rayleigh
            else:
                E_final = E_rayleigh.item() if E_rayleigh.dim() == 0 else E_rayleigh[0].item()

            # PDE残差分析
            pde_res = compute_dirac_residual(
                g, f, self.net.E, self.kappa, self.potentials,
                dr=DR, npt=NPT, f_weight=self.f_weight,
                return_components=True, device=self.device,
            )

            # 归一化检验
            norm_int = torch.trapz(g**2 + f**2, dim=-1, dx=DR).item()

            # 节点数
            n_nodes = count_nodes(g.squeeze(0))

        result = {
            'r': R_GRID.copy(),
            'g': g.cpu().numpy().flatten(),
            'f': f.cpu().numpy().flatten(),
            'energy': E_final,              # ← Rayleigh 商能量
            'E_net': self.net.E.item(),     # 网络参数E（仅供参考）
            'kappa': self.kappa,
            'norm_integral': norm_int,
            'n_nodes': n_nodes,
            'expected_nodes': self.state_info.get('nodes_g'),
            'pde_residual_mean': pde_res['R_g'].mean().item() if 'R_g' in pde_res else None,
            'history': self.history,
        }
        return result


# ═══════════════════════════════════════════════════════════════
#   SCF 训练器 (Phase 2)
# ═══════════════════════════════════════════════════════════════

def _scf_worker_fn(task, epochs, log_path=None):
    """模块级工作函数（spawn多进程必须能pickle，不能是方法）"""
    import io, contextlib
    from config import DEVICE

    name = task['name']
    device = torch.device(task['device'])

    # 重建solver
    solver = MVPSolver(
        isotope=task['isotope'], state=name,
        device=device,
        n_hidden=task['n_hidden'], n_layers=task['n_layers'],
        activation=task['activation'], lr=task['lr'], max_epochs=epochs,
        lambda_pde=task['lambda_pde'], lambda_norm=task['lambda_norm'],
        lambda_r0=task['lambda_r0'], lambda_inf=task['lambda_inf'],
        lambda_R=task['lambda_R'], lambda_kin=task['lambda_kin'],
        f_weight=task['f_weight'], adam_betas=task['adam_betas'],
        grad_clip=task['grad_clip'], early_stop_patience=epochs // 3,
    )
    solver.kappa = task['kappa']
    solver.state_info = {'kappa': task['kappa'], 'n_pr': task['n_pr'],
                         'E_ref': task['E_init'], 'nodes_g': max(0, task['n_pr'] - 1)}
    solver.potentials = {k: torch.tensor(v, dtype=torch.float64).to(device)
                        for k, v in task['pots_cpu'].items()}
    with torch.no_grad():
        solver.net.E.fill_(-50.0)

    # 训练 + 重定向日志
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        solver.train(epochs=epochs, lr=task['lr'], verbose=True)
    result = solver.evaluate()
    result['degeneracy'] = task['deg']
    result['is_proton'] = task['is_proton']

    # 写日志
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w') as f:
            f.write(buf.getvalue())
            f.write(f'\n=== Final Result (Rayleigh) ===\n')
            f.write(f'  E_Rayleigh = {result["energy"]:+.4f} MeV\n')
            f.write(f'  E_net      = {result.get("E_net", "N/A")} MeV\n')
            f.write(f'  norm = {result["norm_integral"]:.6f}\n')
            f.write(f'  nodes = {result["n_nodes"]}/{result.get("expected_nodes", "?")}\n')

    return name, result


class SCFSolver:
    """
    SCF 自洽求解器: 多态同时训练 + 势场自洽更新。

    模式 A (纯Python): 物理计算用 Python 实现
    模式 B (Fortran引擎): 物理计算委托给 Core-1204 Fortran (f2py)

    外层循环 (来自 DDRHF.f90 第78-136行):
      1. 在当前势场下训练所有占据态 PINN
      2. 从波函数计算密度 (Fortran: Densit)
      3. 从密度计算新势场 (Fortran: Meanfield + PotelHF)
      4. 组装 Dirac 自能 V±S
      5. 混合新旧势场 (xmix 策略)
      6. 收敛检查 (si = max|ΔV|, epsi = 10⁻⁵)
    """

    def __init__(self, isotope='16O', device=None,
                 n_hidden=128, n_layers=6, activation='swish',
                 pinn_epochs_per_scf=500, pinn_lr=5e-4,
                 xmix_initial=0.50, max_scf=50, epsi=1e-5,
                 lambda_pde=10.0, lambda_norm=10.0, lambda_r0=1.0,
                 lambda_inf=1.0, lambda_R=1.0, lambda_kin=0.1, f_weight=3.0,
                 adam_betas=(0.9, 0.999), grad_clip=1.0,
                 use_fortran=False, fortran_dir='Core-1204'):
        from config import NUCLEUS_STATES, ISOTOPE_CONFIG, SCF_CONFIG

        self.isotope = isotope
        self.device = device or DEVICE
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.activation = activation
        self.pinn_epochs = pinn_epochs_per_scf
        self.pinn_lr = pinn_lr
        self.xmix = xmix_initial
        self.xmix0 = xmix_initial
        self.max_scf = max_scf
        self.epsi = epsi
        self.lambda_pde = lambda_pde
        self.lambda_norm = lambda_norm
        self.lambda_r0 = lambda_r0
        self.lambda_inf = lambda_inf
        self.lambda_R = lambda_R
        self.lambda_kin = lambda_kin
        self.f_weight = f_weight
        self.adam_betas = adam_betas
        self.grad_clip = grad_clip

        # ★ 是否使用 Fortran 引擎
        self.use_fortran = use_fortran
        self._fortran_engine = None
        self._fc = None  # FortranRHFCalculator (ctypes) 实例

        # 核素信息
        if isotope not in ISOTOPE_CONFIG:
            raise ValueError(f"未知核素: {isotope}")
        self.nuc_cfg = ISOTOPE_CONFIG[isotope]
        self.Z = self.nuc_cfg['Z']
        self.A = self.nuc_cfg['A']

        # 态配置
        if isotope not in NUCLEUS_STATES:
            raise ValueError(f"核素 {isotope} 的态配置未定义")
        self.state_defs = NUCLEUS_STATES[isotope]

        # 径向网格
        self.r_grid = get_r_grid(self.device)
        self.r_safe = get_r_safe(self.device)

        # ════════════════════════════════════════
        #  ★ Fortran 引擎初始化 (模式 B)
        # ════════════════════════════════════════
        if self.use_fortran:
            try:
                from physics.wrap_fortran import FortranRHFEngine
                self._fortran_engine = FortranRHFEngine(fortran_dir=fortran_dir)
                r_fortran = self._fortran_engine.init(model_id=0)  # PKA1
                print(f'  [Fortran] Engine ready: {len(self._fortran_engine.states_info)} states')
                # 用 Fortran 网格覆盖 Python 网格（确保一致）
                if len(r_fortran) == NPT:
                    pass  # 网格一致
                else:
                    print(f'  WARNING: Fortran grid ({len(r_fortran)}) != Python grid ({NPT})')
            except Exception as e:
                print(f'  WARNING: Fortran engine init failed: {e}, falling back to Python mode')
                self.use_fortran = False
                self._fortran_engine = None

        # 初始化势场 (从 POT 文件读取, 非 WS)
        self.pots_n = None  # 中子势场
        self.pots_p = None  # 质子势场
        if not self.use_fortran:
            self._init_from_pot_file()
        else:
            # 从 Fortran 引擎获取初始势场 (DBASE 后的初始猜测)
            self._init_from_fortran()

        # 为每个态初始化网络
        self.solvers = {}
        self._init_state_solvers()

        # SCF 历史
        self.scf_history = {
            'si': [], 'xmix': [], 'energies': {},
        }

    def _init_from_pot_file(self):
        """从 POT 文件初始化势场 (纯 Python 模式, 无WS)"""
        # POT文件模式: 势场在外部加载后通过 set_potentials() 设置
        # 这里创建零势占位
        r = self.r_grid.cpu()
        zeros = torch.zeros_like(r)
        self.pots_n = make_local_potentials(zeros, zeros, zeros)
        self.pots_p = make_local_potentials(zeros.clone(), zeros.clone(), zeros.clone())
        print("  [WARNING] No POT file provided — using zero potentials. Use --pot-file or SCF mode.")

    def _init_from_fortran(self):
        """从 Fortran 引擎获取初始势场"""
        fe = self._fortran_engine
        # 先做一步 SCF 获取初始势场 (DBASE 后的 Woods-Saxon 初始解)
        si, pots = fe.potentials(xmix=1.0)  # xmix=1 表示全接受新势场

        import numpy as np
        dev = self.device
        self.pots_n = {
            'vps': torch.tensor(pots['V_ps_n'], dtype=torch.float32).to(dev),
            'vms': torch.tensor(pots['V_ms_n'], dtype=torch.float32).to(dev),
            'vtt': torch.tensor(pots['V_tt_n'], dtype=torch.float32).to(dev),
        }
        self.pots_p = {
            'vps': torch.tensor(pots['V_ps_p'], dtype=torch.float32).to(dev),
            'vms': torch.tensor(pots['V_ms_p'], dtype=torch.float32).to(dev),
            'vtt': torch.tensor(pots['V_tt_p'], dtype=torch.float32).to(dev),
        }

    def _init_state_solvers(self):
        """为每个占据态创建 MVPSolver"""
        for state_def in self.state_defs:
            name, kappa, n_pr, deg, E_init, is_proton = state_def

            # 选择势场
            pots = self.pots_p if is_proton else self.pots_n

            solver = MVPSolver(
                isotope=self.isotope, state=name, device=self.device,
                n_hidden=self.n_hidden, n_layers=self.n_layers,
                activation=self.activation,
                lr=self.pinn_lr, max_epochs=self.pinn_epochs,
                lambda_pde=self.lambda_pde, lambda_norm=self.lambda_norm,
                lambda_r0=self.lambda_r0, lambda_inf=self.lambda_inf,
                lambda_R=self.lambda_R,
                lambda_kin=self.lambda_kin, f_weight=self.f_weight,
                adam_betas=self.adam_betas, grad_clip=self.grad_clip,
                early_stop_patience=self.pinn_epochs // 3,
            )
            # 覆盖默认的 kappa 和 E_init
            solver.kappa = kappa
            solver.state_info = {
                'kappa': kappa, 'n_pr': n_pr,
                'E_ref': E_init, 'nodes_g': max(0, n_pr - 1),
            }
            # 重新初始化能量
            with torch.no_grad():
                solver.net.E.fill_(E_init)

            # 手动设置势场
            solver.potentials = {}
            for key in pots:
                if isinstance(pots[key], torch.Tensor):
                    solver.potentials[key] = pots[key].clone().to(self.device)

            self.solvers[name] = solver

    def _prepare_state_task(self, name):
        """准备单个态的参数（主进程中调用，返回可pickle的数据）"""
        state_def = None
        for s in self.state_defs:
            if s[0] == name:
                state_def = s; break
        if not state_def:
            raise ValueError(f"Unknown state: {name}")

        _, kappa, n_pr, deg, E_init, is_proton = state_def
        pots = self.pots_p if is_proton else self.pots_n
        # 全部转numpy（可pickle）
        pots_cpu = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                    for k, v in pots.items()}

        return {
            'name': name, 'isotope': self.isotope, 'device': str(self.device),
            'kappa': kappa, 'n_pr': n_pr, 'deg': deg,
            'E_init': E_init, 'is_proton': is_proton,
            'pots_cpu': pots_cpu,
            'n_hidden': self.n_hidden, 'n_layers': self.n_layers,
            'activation': self.activation, 'lr': self.pinn_lr,
            'lambda_pde': self.lambda_pde, 'lambda_norm': self.lambda_norm,
            'lambda_r0': self.lambda_r0, 'lambda_inf': self.lambda_inf,
            'lambda_R': self.lambda_R, 'lambda_kin': self.lambda_kin,
            'f_weight': self.f_weight, 'adam_betas': self.adam_betas,
            'grad_clip': self.grad_clip,
        }

    def _solve_all_states(self, epochs=None, verbose=True, log_dir=None):
        """
        分批并行训练：先并跑所有中子态，再并跑所有质子态。
        每批最多 ~42 个态（GPU 最大承载量），与对比程序 dirac_matrix_vs_pinn.py 的 --tau 一致。
        """
        import multiprocessing as mp

        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        epochs = epochs or self.pinn_epochs
        results = {}

        if log_dir is None:
            log_dir = './outputs/scf_logs'

        # ── 按 nucleon_type 分组 ──
        neut_names = [n for n in self.solvers.keys() if n.startswith('n-')]
        prot_names = [n for n in self.solvers.keys() if n.startswith('p-')]

        batches = [
            ('neutron', 'n', neut_names),
            ('proton',  'p', prot_names),
        ]

        total_t0 = time.time()
        summary_log = os.path.join(log_dir, 'summary.log')
        summary_lines = []

        for batch_label, tau_tag, name_list in batches:
            if not name_list:
                continue
            n_batch = len(name_list)
            MAX_SAFE_WORKERS = 21  # 每进程366MiB，10个约3.6GB，系统稳定
            n_workers = min(n_batch, MAX_SAFE_WORKERS)

            step_log = os.path.join(log_dir, f'batch_{batch_label}')
            print(f"\n  ══ Batch {batch_label}: {n_batch} states × {n_workers} workers ══")
            start_t = time.time()

            # 准备本批任务
            tasks = {name: self._prepare_state_task(name) for name in name_list}

            with ProcessPoolExecutor(max_workers=n_workers,
                                     mp_context=mp.get_context('spawn')) as executor:
                futures = {
                    executor.submit(
                        _scf_worker_fn, task, epochs,
                        log_path=os.path.join(step_log, f'{name}.log'),
                    ): name
                    for name, task in tasks.items()
                }
                for future in as_completed(futures):
                    try:
                        name, result = future.result()
                        results[name] = result
                        elapse = time.time() - start_t
                        done_now = sum(1 for n in name_list if n in results)
                        print(f'  [{tau_tag.upper()}: {done_now}/{n_batch}] '
                              f'{name} E_Ray={result["energy"]:+.4f}MeV  ({elapse:.1f}s)')
                    except Exception as e:
                        # 单个进程崩溃不影响其他进程
                        # 从 future 映射中找 name（通过遍历）
                        failed_name = '?'
                        for fn, fname in list(futures.items()):
                            if fn is future:
                                failed_name = fname; break
                        print(f'  [{tau_tag.upper()}] {failed_name} FAILED: {type(e).__name__}: {e}')
                        results[failed_name] = {
                            'r': [], 'g': [], 'f': [],
                            'energy': 0.0, 'E_net': 0.0,
                            'kappa': 0, 'norm_integral': 0,
                            'n_nodes': -1, 'expected_nodes': '?',
                            'pde_residual_mean': None, 'history': {},
                            '_error': str(e),
                        }

            batch_time = time.time() - start_t
            done_ok = sum(1 for n in name_list if n in results and '_error' not in results[n])
            print(f"  Batch {batch_label} done: {done_ok}/{n_batch} OK | {batch_time:.1f}s")

            # 汇总本批结果
            occ_n = sum(1 for n in name_list if n in results
                        and '_error' not in results[n] and results[n]['energy'] < 0)
            summary_lines.append(f"Batch_{batch_label}: {n_batch} states, "
                                 f"{occ_n} bound, {batch_time:.1f}s → {step_log}")

        total_time = time.time() - total_t0

        # 写汇总日志
        os.makedirs(log_dir, exist_ok=True)
        with open(summary_log, 'w') as f:
            f.write(f'SCF Parallel Training Summary\n')
            f.write(f'{"="*60}\n')
            for line in summary_lines:
                f.write(line + '\n')
            f.write(f'\nTotal wall time: {total_time:.1f}s\n\n')
            f.write(f'{"State":>12s}  {"E_Ray(MeV)":>12s}  {"E_net(MeV)":>10s}  {"norm":>8s}  {"nodes":>6s}\n')
            f.write(f'{"-"*12}  {"-"*12}  {"-"*10}  {"-"*8}  {"-"*6}\n')
            for name in sorted(results.keys()):
                r = results[name]
                f.write(f'{name:>12s}  {r["energy"]:+12.4f}  '
                        f'{r.get("E_net", 0):+10.4f}  '
                        f'{r["norm_integral"]:8.4f}  '
                        f'{r["n_nodes"]}/{r.get("expected_nodes","?")}\n')

        # 终端精简输出
        occ = {n: r for n, r in results.items() if r['energy'] < 0}
        unocc = {n: r for n, r in results.items() if r['energy'] >= 0}
        print(f'\n  All done: {len(occ)} bound + {len(unocc)} continuum | '
              f'{total_time:.1f}s total | log→{summary_log}')

        return results

    def _update_potentials(self, state_results, verbose=True):
        """从波函数计算密度 → 更新势场 (支持 Fortran/Python 双模式)"""

        if self.use_fortran and self._fortran_engine is not None:
            return self._update_potentials_fortran(state_results, verbose)
        else:
            return self._update_potentials_python(state_results, verbose)

    def _update_potentials_fortran(self, state_results, verbose=True):
        """★ Fortran 模式: 将 PINN 波函数注入 Fortran → 计算势场"""
        import numpy as np
        fe = self._fortran_engine

        # 1. 从 PINN 结果收集波函数
        G_dict = {}
        F_dict = {}
        for name, res in state_results.items():
            if '_error' in res or len(res.get('g', [])) == 0:
                continue
            G_dict[name] = np.asarray(res['g'], dtype=np.float64)
            F_dict[name] = np.asarray(res['f'], dtype=np.float64)

        # 2. 注入到 Fortran 引擎
        fe.set_wavefunctions(G_dict, F_dict)

        # 3. 计算密度 + 势场 (一步完成)
        si, pots = fe.potentials(xmix=self.xmix)

        # 4. 转为 torch tensor 并更新
        dev = self.device
        old_pots_n = self.pots_n
        old_pots_p = self.pots_p

        self.pots_n = {
            'vps': torch.tensor(pots['V_ps_n'], dtype=torch.float32).to(dev),
            'vms': torch.tensor(pots['V_ms_n'], dtype=torch.float32).to(dev),
            'vtt': torch.tensor(pots['V_tt_n'], dtype=torch.float32).to(dev),
        }
        self.pots_p = {
            'vps': torch.tensor(pots['V_ps_p'], dtype=torch.float32).to(dev),
            'vms': torch.tensor(pots['V_ms_p'], dtype=torch.float32).to(dev),
            'vtt': torch.tensor(pots['V_tt_p'], dtype=torch.float32).to(dev), }

        # 5. 更新各 solver 的势场
        for name, solver in self.solvers.items():
            is_proton = any(s[0] == name and s[5] for s in self.state_defs)
            pots = self.pots_p if is_proton else self.pots_n
            for key in solver.potentials:
                if key in pots:
                    solver.potentials[key] = pots[key].clone().to(self.device)

        return si

    def _update_potentials_python(self, state_results, verbose=True):
        """纯 Python 模式: 从波函数计算密度 → 更新势场"""
        from potentials import update_potentials_from_density, mix_potentials, potential_difference
        from density import compute_scf_densities

        # 1. 计算密度
        densities = compute_scf_densities(state_results, r_grid=self.r_grid.cpu(), dr=DR)

        # 2. 计算新 Hartree 势场
        new_pots_n, new_pots_p = update_potentials_from_density(
            densities, isotope=self.isotope
        )

        # 确保新势场与旧势场在同一设备上
        device = next(iter(self.pots_n.values())).device
        for pots in [new_pots_n, new_pots_p]:
            for key in pots:
                if isinstance(pots[key], torch.Tensor):
                    pots[key] = pots[key].to(device)

        # 3. 势场混合
        old_pots_n, old_pots_p = self.pots_n, self.pots_p
        self.pots_n = mix_potentials(old_pots_n, new_pots_n, self.xmix)
        self.pots_p = mix_potentials(old_pots_p, new_pots_p, self.xmix)

        # 4. 收敛指标
        si_n = potential_difference(old_pots_n, self.pots_n)
        si_p = potential_difference(old_pots_p, self.pots_p)
        si = max(si_n, si_p)

        # 5. 更新各 solver 的势场
        for name, solver in self.solvers.items():
            is_proton = any(s[0] == name and s[5] for s in self.state_defs)
            pots = self.pots_p if is_proton else self.pots_n
            for key in solver.potentials:
                if key in pots:
                    solver.potentials[key] = pots[key].clone().to(self.device)

        return si

    def _adjust_xmix(self, si, si0):
        """调整混合参数 (DDRHF.f90 第126-132行)"""
        if abs(si) < abs(si0) and self.xmix < 1.0:
            self.xmix = self.xmix * 1.0618
            if self.xmix > 1.0:
                self.xmix = 1.0
        elif abs(si) > abs(si0):
            self.xmix = self.xmix0

    def run(self, verbose=True, log_dir=None):
        """
        执行 SCF 自洽循环。

        返回:
            scf_history: dict, 自洽迭代历史
        """
        if log_dir is None:
            log_dir = './outputs/scf_logs'
        mode_tag = " [Fortran]" if self.use_fortran else " [Python]"
        print(f"SCF{mode_tag}: {self.isotope} Z={self.Z} N={self.A-self.Z} "
              f"states={len(self.solvers)} max_scf={self.max_scf}")

        si = 1.0  # 收敛指标
        si0 = si
        start_time = time.time()

        for scf_step in range(1, self.max_scf + 1):
            si0 = si

            # 每步SCF独立的日志目录
            step_log = os.path.join(log_dir, f'scf{scf_step:03d}')

            # 1. 并行训练所有态（带日志）
            state_results = self._solve_all_states(verbose=verbose, log_dir=step_log)

            # 2. 更新势场
            si = self._update_potentials(state_results, verbose=verbose)

            # 3. 记录历史
            energies = {name: res['energy'] for name, res in state_results.items()}
            self.scf_history['si'].append(si)
            self.scf_history['xmix'].append(self.xmix)
            for name, E in energies.items():
                if name not in self.scf_history['energies']:
                    self.scf_history['energies'][name] = []
                self.scf_history['energies'][name].append(E)

            # ★ Fortran 模式: 获取完整能量泛函
            fort_ene_str = ""
            if self.use_fortran and self._fortran_engine is not None and scf_step % 5 == 0:
                try:
                    ene = self._fortran_engine.energy()
                    fort_ene_str = (f" E_tot/A={ene.get('E_per_A',0):+.3f}MeV "
                                   f"E_kin={ene.get('E_kinetic',0):+.1f}MeV")
                except Exception as e:
                    fort_ene_str = f" [Fortran energy error: {e}]"

            # 4. 精简输出：只打印关键占据态能量
            if verbose:
                occ_energies = {n: e for n, e in energies.items() if e < 0}
                E_str = " ".join(f"{n}:{e:+.2f}" for n, e in sorted(occ_energies.items(), key=lambda x: x[1]))
                print(f"SCF{scf_step:3d}/{self.max_scf} si={si:.2e} "
                      f"xmix={self.xmix:.3f} | {E_str}{fort_ene_str}")

            # 5. 调整 xmix
            self._adjust_xmix(si, si0)

            # 6. 收敛检查
            if si < self.epsi:
                print(f"  SCF converged at step {scf_step}!")
                break
        else:
            if verbose:
                print(f"  SCF NOT converged, final si={si:.2e}")

        total_time = time.time() - start_time
        print(f"SCF done: {scf_step} steps, {total_time:.1f}s")

        return self.scf_history


# ═══════════════════════════════════════════════════════════════
#   主入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='PINN-RHF: 物理信息神经网络求解Dirac方程')
    parser.add_argument('--mode', type=str, default='mvp',
                       choices=['mvp', 'scf'],
                       help='运行模式: mvp=单态求解, scf=自洽循环')
    parser.add_argument('--isotope', type=str, default='16O',
                       help=f'核素名称: {list(ISOTOPE_CONFIG.keys())}')
    parser.add_argument('--state', type=str, default='1s1/2',
                       help='目标态名称 (MVP模式): 如 1s1/2, 1p3/2')
    parser.add_argument('--epochs', type=int, default=None,
                       help='最大训练轮次 (覆盖配置)')
    parser.add_argument('--lr', type=float, default=None,
                       help='学习率 (覆盖配置)')
    parser.add_argument('--seed', type=int, default=SEED,
                       help='随机种子')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cpu', 'cuda'], help='强制设备')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='输出目录')
    parser.add_argument('--max-scf', type=int, default=None,
                       help='SCF最大迭代步数 (覆盖配置)')
    parser.add_argument('--pinn-epochs', type=int, default=None,
                       help='每SCF步的PINN训练轮次 (覆盖配置)')
    parser.add_argument('--fortran', action='store_true',
                       help='使用Fortran RHF引擎 (f2py) 进行物理计算')

    # 激发态求解参数
    parser.add_argument('--lambda-ortho', type=float, default=10.0,
                       help='正交性损失权重 (用于激发态求解, 默认10.0)')
    parser.add_argument('--ref-wavefunctions', type=str, default=None,
                       help='参考波函数文件路径 (JSON格式, 用于激发态求解时的正交性约束)')

    args = parser.parse_args()

    # ─── 加载参考波函数 (用于激发态求解) ───
    ref_wavefunctions = None
    if args.ref_wavefunctions:
        ref_wavefunctions = load_ref_wavefunctions(args.ref_wavefunctions, device)
        if ref_wavefunctions:
            print(f"  已加载 {len(ref_wavefunctions)} 个参考波函数用于正交性约束")

    # ─── 配置: 直接从config导入, 不再用全局变量 ───
    from config import MODEL_CONFIG, TRAIN_CONFIG, LOSS_WEIGHTS

    model_nh = MODEL_CONFIG['n_hidden']
    model_nl = MODEL_CONFIG['n_layers']
    model_act = MODEL_CONFIG['activation']
    max_epochs = args.epochs or TRAIN_CONFIG['max_epochs']
    lr = args.lr or TRAIN_CONFIG['lr']
    adam_betas = TRAIN_CONFIG['adam_betas']
    grad_clip = TRAIN_CONFIG.get('grad_clip', 1.0)
    early_stop_patience = TRAIN_CONFIG.get('early_stop_patience', 300)
    lambda_pde = LOSS_WEIGHTS['pde']
    lambda_norm = LOSS_WEIGHTS['norm']
    lambda_r0 = LOSS_WEIGHTS['boundary_r0']
    lambda_inf = LOSS_WEIGHTS['boundary_inf']
    lambda_R = LOSS_WEIGHTS['boundary_R']
    lambda_kin = LOSS_WEIGHTS['kinetic_positive']
    f_weight = LOSS_WEIGHTS['f_weight']

    set_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = DEVICE

    print(f"PINN-RHF mode={args.mode.upper()} isotope={args.isotope} device={device} "
          f"grid={NPT}×{DR}fm MLP({model_nh})×{model_nl} epochs={max_epochs} lr={lr}")

    # ─── 创建输出目录 ───
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_state = args.state.replace('/', '_')
    output_dir = os.path.join(args.output_dir, f'{args.isotope}_{safe_state}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)

    # ─── 运行选定模式 ───
    if args.mode == 'mvp':
        run_mvp(args, output_dir, device,
                 n_hidden=model_nh, n_layers=model_nl, activation=model_act,
                 max_epochs=max_epochs, lr=lr,
                 adam_betas=adam_betas, grad_clip=grad_clip,
                 early_stop_patience=early_stop_patience,
                 lambda_pde=lambda_pde, lambda_norm=lambda_norm,
                 lambda_r0=lambda_r0, lambda_inf=lambda_inf,
                 lambda_R=lambda_R,
                 lambda_kin=lambda_kin, f_weight=f_weight,
                 lambda_ortho=args.lambda_ortho, ref_wavefunctions=ref_wavefunctions)
    elif args.mode == 'scf':
        run_scf(args, output_dir, device)


def load_ref_wavefunctions(filepath, device=None):
    """
    从 JSON 文件加载参考波函数 (基态波函数)。
    
    JSON 文件格式:
        {
            "states": [
                {"name": "1s1/2", "g": [..], "f": [..]},
                ...
            ]
        }
    或直接使用 train.py 的 save_result 格式。
    
    返回:
        ref_wavefunctions: list of dict, 每个dict含 {'g': tensor, 'f': tensor, 'name': str}
    """
    if not filepath or not os.path.exists(filepath):
        return None
    
    if device is None:
        device = DEVICE
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    ref_wavefunctions = []
    
    # 尝试多种格式
    if 'states' in data:
        # 多态格式
        for state_data in data['states']:
            g = torch.tensor(state_data['g'], dtype=torch.float32, device=device)
            f = torch.tensor(state_data['f'], dtype=torch.float32, device=device)
            ref_wavefunctions.append({
                'name': state_data.get('name', 'unknown'),
                'g': g,
                'f': f,
            })
    elif 'wavefunction' in data:
        # save_result 单态格式
        r = data['wavefunction']['r_fm']
        g = torch.tensor(data['wavefunction']['G'], dtype=torch.float32, device=device)
        f = torch.tensor(data['wavefunction']['F'], dtype=torch.float32, device=device)
        ref_wavefunctions.append({
            'name': data['meta'].get('state', 'unknown'),
            'g': g,
            'f': f,
        })
    else:
        # 尝试直接解析
        for key in data:
            if isinstance(data[key], dict) and 'g' in data[key]:
                g = torch.tensor(data[key]['g'], dtype=torch.float32, device=device)
                f = torch.tensor(data[key]['f'], dtype=torch.float32, device=device)
                ref_wavefunctions.append({
                    'name': key,
                    'g': g,
                    'f': f,
                })
    
    print(f"  已加载 {len(ref_wavefunctions)} 个参考波函数: {[ref['name'] for ref in ref_wavefunctions]}")
    return ref_wavefunctions


def run_mvp(args, output_dir, device,
            n_hidden=128, n_layers=6, activation='swish',
            max_epochs=2000, lr=1e-3,
            adam_betas=(0.9, 0.999), grad_clip=1.0,
            early_stop_patience=300,
            lambda_pde=10.0, lambda_norm=10.0, lambda_r0=1.0, 
            lambda_inf=1.0, lambda_R=1.0, lambda_kin=0.1, f_weight=3.0,
            lambda_ortho=10.0, ref_wavefunctions=None):
    """执行 MVP 单态求解
    
    参数:
        lambda_ortho: 正交性损失权重 (默认10.0)
        ref_wavefunctions: 参考波函数列表, 用于激发态求解
    """
    print(f"MVP: {args.isotope} {args.state} κ={parse_state_name(args.state)['kappa']}")
    if ref_wavefunctions:
        ref_names = [ref.get('name', 'unknown') for ref in ref_wavefunctions]
        print(f"  Ortho loss: weight={lambda_ortho}, ref_states={ref_names}")

    solver = MVPSolver(
        isotope=args.isotope, state=args.state, device=device,
        n_hidden=n_hidden, n_layers=n_layers, activation=activation,
        lr=lr, max_epochs=max_epochs,
        lambda_pde=lambda_pde, lambda_norm=lambda_norm, lambda_r0=lambda_r0, 
        lambda_inf=lambda_inf, lambda_R=lambda_R,
        lambda_kin=lambda_kin, f_weight=f_weight,
        adam_betas=adam_betas, grad_clip=grad_clip,
        early_stop_patience=early_stop_patience,
        lambda_ortho=lambda_ortho, ref_wavefunctions=ref_wavefunctions,
    )

    # 训练
    history = solver.train(
        epochs=max_epochs, lr=lr, 
        verbose=True, print_every=max(1, max_epochs // 20),
    )

    # 评估
    result = solver.evaluate()
    
    # 精简输出：只打关键结果
    print(f"  E_Rayleigh={result['energy']:+.4f}MeV E_net={result.get('E_net', 'N/A'):.4f}MeV "
          f"norm={result['norm_integral']:.6f} nodes={result['n_nodes']}/{result['expected_nodes']}")

    # 保存结果
    safe_state = args.state.replace('/', '_')
    save_path = os.path.join(output_dir, f'result_{safe_state}.json')
    save_result(result, history, save_path)
    print(f"\n  结果已保存至: {save_path}")

    # 绘图
    safe_state = args.state.replace('/', '_')
    plot_results(result, history, output_dir, safe_state)
    print(f"  图表已保存至: {output_dir}/plots/")


def run_scf(args, output_dir, device):
    """执行 SCF 自洽循环"""
    from config import SCF_CONFIG

    log_dir = os.path.join(output_dir, 'scf_logs')

    solver = SCFSolver(
        isotope=args.isotope, device=device,
        n_hidden=MODEL_CONFIG['n_hidden'],
        n_layers=MODEL_CONFIG['n_layers'],
        activation=MODEL_CONFIG['activation'],
        pinn_epochs_per_scf=args.pinn_epochs or SCF_CONFIG.get('pinn_epochs_per_scf', 500),
        pinn_lr=SCF_CONFIG.get('pinn_lr_per_scf', 5e-4),
        xmix_initial=SCF_CONFIG.get('xmix_initial', 0.50),
        max_scf=args.max_scf or SCF_CONFIG.get('max_iterations', 50),
        epsi=SCF_CONFIG.get('convergence_eps', 1e-5),
        use_fortran=args.fortran,
        fortran_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Core-1204')),
    )

    history = solver.run(verbose=True, log_dir=log_dir)

    # 保存结果
    save_path = os.path.join(output_dir, f'scf_result.json')
    scf_data = {
        'isotope': args.isotope,
        'mode': 'fortran' if args.fortran else 'python',
        'scf_steps': len(history['si']),
        'final_si': history['si'][-1] if history['si'] else None,
        'energies': history['energies'],
        'xmix_history': history['xmix'],
    }
    # ★ Fortran 模式: 附上完整能量泛函
    if args.fortran and solver._fortran_engine is not None:
        try:
            ene = solver._fortran_engine.energy()
            scf_data['energy_components'] = ene
        except Exception:
            pass

    with open(save_path, 'w') as f:
        json.dump(scf_data, f, indent=2, default=str)
    print(f"\n  结果已保存至: {save_path}")

    # 绘制收敛曲线
    plot_scf_convergence(history, output_dir)

def plot_scf_convergence(history, output_dir):
    """绘制 SCF 收敛曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # si 收敛
    ax = axes[0]
    ax.semilogy(history['si'], 'ko-', markersize=3)
    ax.set_xlabel('SCF Step')
    ax.set_ylabel('si (max |ΔV|)')
    ax.set_title('SCF Convergence')
    ax.grid(True, alpha=0.3)

    # 能量演化
    ax = axes[1]
    for name, energies in history['energies'].items():
        ax.plot(energies, '-o', markersize=3, label=name)
    ax.set_xlabel('SCF Step')
    ax.set_ylabel('Energy ε (MeV)')
    ax.set_title('Energy Eigenvalue Evolution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'plots', 'scf_convergence.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {plot_path}")


def save_result(result, history, filepath):
    """将结果保存为 JSON 文件"""
    data = {
        'meta': {
            'isotope': result.get('isotope', ''),
            'state': '',
            'kappa': result['kappa'],
            'energy_MeV': result['energy'],
            'norm_integral': result['norm_integral'],
            'n_nodes_g': result['n_nodes'],
            'expected_nodes': result.get('expected_nodes'),
        },
        'wavefunction': {
            'r_fm': result['r'].tolist(),
            'G': result['g'].tolist(),
            'F': result['f'].tolist(),
        },
        'training_history': {
            k: [float(v) for v in vals] if len(vals) > 0 and isinstance(vals[0], (int, float, np.floating)) else vals
            for k, vals in history.items()
        },
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def plot_results(result, history, output_dir, state_name):
    """绘制训练结果的对比图表"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    r = result['r']
    g = result['g']
    f = result['f']

    # 1. 波函数形状
    ax = axes[0, 0]
    ax.plot(r, g, 'b-', linewidth=2, label=r'$G(r)$ 大分量')
    ax.plot(r, f, 'r-', linewidth=2, label=r'$F(r)$ 小分量')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('r (fm)')
    ax.set_ylabel('Wavefunction amplitude')
    ax.set_title(f'Dirac Wavefunction: {state_name}  (ε = {result["energy"]:+.2f} MeV)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, min(20, r[-1])])

    # 2. 损失曲线
    ax = axes[0, 1]
    ax.semilogy(history['loss_total'], 'k-', label='Total Loss', alpha=0.8)
    ax.semilogy(history['loss_pde'], 'b-', label='PDE Loss', alpha=0.7)
    ax.semilogy(history['loss_norm'], 'r-', label='Norm Loss', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Training Loss Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 能量演化
    ax = axes[1, 0]
    energies = history['energy']
    ax.plot(energies, 'g-', linewidth=1.5)
    ax.axhline(y=result['energy'], color='red', linestyle='--', alpha=0.5,
               label=f'Final: {result["energy"]:+.3f} MeV')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Energy ε (MeV)')
    ax.set_title('Energy Eigenvalue Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. PDE 残差分布 (如果可用)
    ax = axes[1, 1]
    # 近似: 显示 G/F 的相对幅度比
    gf_ratio = np.abs(f) / (np.abs(g) + 1e-10)
    ax.plot(r, g**2, 'b-', label=r'$G^2(r)$', alpha=0.8)
    ax.plot(r, f**2 * 10, 'r-', label=r'$10 \times F^2(r)$', alpha=0.8)  # 放大10倍显示
    ax.fill_between(r, 0, g**2 + f**2, alpha=0.2, color='purple', label=r'$G^2+F^2$')
    ax.set_xlabel('r (fm)')
    ax.set_ylabel('Density')
    ax.set_title('Probability Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, min(20, r[-1])])

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'plots', f'pinn_{state_name}_result.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {plot_path}")


if __name__ == '__main__':
    main()
