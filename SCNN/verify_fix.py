#!/usr/bin/env python3
"""快速验证修复效果: 训练30个epoch, 每10个epoch画一次波函数"""
import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 确保可以导入本地模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Train import (
    train_model, build_datasets, normalize, calculate_normalization_stats,
    calculate_y_stats, plot_wavefunctions,
)
from Model_Architecture import RHF_FNO_GRU
from Physics_Informed_Loss import calc_simplified_residual

def quick_verify(num_epochs=30, device_str='cpu'):
    """快速验证: 训练num_epochs个epoch并检查波函数是否退化成delta"""
    
    device = torch.device(device_str)
    print(f"\n{'='*60}")
    print(f"  快速验证模式: {num_epochs} epochs on {device}")
    print(f"{'='*60}")
    
    # === 超参数（与Train.py保持一致，但缩小规模）===
    hidden_dim = 96
    gru_hidden = 1536
    modes = 64  # ★ 与Train.py同步: 从40提升到64减少Gibbs
    weight_decay = 1e-4
    use_self_attention = True
    
    # 修复后的损失权重（与Train.py同步）
    lambda_data = 1.0
    lambda_pde = 2.0
    lambda_norm = 2.0
    lambda_node = 3.0
    lambda_physical = 3.0
    lambda_smooth = 10.0       # ★ 与Train.py同步: 从2.0提升到10.0
    lambda_rayleigh = 3.0
    f_mse_weight = 5.0
    
    learning_rate = 1e-4
    batch_size = 16
    clip_grad_norm = 1.5
    dr = 0.10
    max_seq_len = 15
    min_seq_len = 2
    traj_usage_ratio = 0.95
    phy_loss_every_n_batches = 1
    
    # 仅Phase 1的7个核心态 + 2个核素
    phase1_isotopes = ['16O', '40Ca']
    PHASE1_STATES = [
        '1s1/2', '1p3/2', '1p1/2', '1d5/2', '1d3/2', '1f7/2', '1f5/2',
    ]
    
    data_dir = '/home/ubuntu/rhf/results'
    save_dir = '/home/ubuntu/rhf/SCNN/plots'
    os.makedirs(save_dir, exist_ok=True)
    
    # === 构建数据集 ===
    print("\n构建数据集...")
    train_dataset = build_datasets(data_dir, phase1_isotopes,
                                   max_seq_len=max_seq_len, min_seq_len=min_seq_len,
                                   traj_usage_ratio=traj_usage_ratio,
                                   mode='train', target_states=PHASE1_STATES)
    if train_dataset is None:
        print("ERROR: 训练数据集为空!")
        return False
    
    def make_quick_loader(dataset, batch_sz, shuffle=True):
        from Data_Loader import IsotopeGroupedBatchSampler
        actual_batch = min(batch_sz, max(8, len(dataset)))
        batch_sampler = IsotopeGroupedBatchSampler(dataset, actual_batch, shuffle=shuffle)
        return torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler,
            num_workers=2, pin_memory=False
        )
    
    train_loader = make_quick_loader(train_dataset, batch_size, shuffle=True)
    print(f"训练集: {len(train_dataset)} 样本, {len(train_loader)} batches")
    
    mean, std = calculate_normalization_stats(train_loader, 0, use_ddp=False)
    y_mean, y_std = calculate_y_stats(train_loader, 0, use_ddp=False)
    
    # === 实例化模型 ===
    print("\n实例化模型...")
    model = RHF_FNO_GRU(in_channels=12, hidden_dim=hidden_dim, npt=201,
                        gru_hidden=gru_hidden, modes=modes,
                        use_self_attention=use_self_attention).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {total_p:,}")
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = GradScaler()
    
    base_r_grid = torch.arange(0, 201, device=device, dtype=torch.float32) * dr
    base_r_grid[0] = 0.0010
    
    # === Warm-up参数 ===
    phy_warmup_epochs_p1 = 30  # 验证期间全程用warm-up phase 1
    
    # === 训练循环 ===
    print(f"\n开始训练 ({num_epochs} epochs)...")
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        
        # Warm-up 权重调度
        if epoch <= phy_warmup_epochs_p1:
            eff_lambda_pde = lambda_pde * 0.5
            eff_lambda_norm = lambda_norm * 0.5
            eff_lambda_node = 0.0
            eff_lambda_physical = 0.0
            eff_lambda_smooth = 1.0
            eff_lambda_rayleigh = 0.0
        else:
            eff_lambda_pde = lambda_pde
            eff_lambda_norm = lambda_norm
            eff_lambda_node = lambda_node
            eff_lambda_physical = lambda_physical
            eff_lambda_smooth = lambda_smooth
            eff_lambda_rayleigh = lambda_rayleigh
        
        total_loss = 0.0
        num_batches = 0
        
        # LR warm-up (前5 epoch)
        if epoch <= 5:
            for pg in optimizer.param_groups:
                pg['lr'] = learning_rate * epoch / 5
        
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, batch_data in enumerate(train_loader):
            x_seq = batch_data[0].to(device)
            y_true = batch_data[1].to(device)
            kappa = batch_data[2].to(device)
            is_proton = batch_data[3].to(device) if len(batch_data) > 3 else None
            z_num = batch_data[5].to(device) if len(batch_data) > 5 else None
            n_num = batch_data[6].to(device) if len(batch_data) > 6 else None
            n_principal = batch_data[7].to(device) if len(batch_data) > 7 else None
            
            B = x_seq.size(0)
            batch_r_grid = base_r_grid.unsqueeze(0).expand(B, -1)
            x_seq_norm = normalize(x_seq, mean, std)
            
            with autocast('cuda') if device_str == 'cuda' else _no_autocast():
                y_pred = model(x_seq_norm, kappa, batch_r_grid,
                               is_proton=is_proton, z_num=z_num, n_num=n_num,
                               n_principal=n_principal)
                
                loss_g = nn.functional.mse_loss(y_pred[:, 0:1, :], y_true[:, 0:1, :])
                loss_f = nn.functional.mse_loss(y_pred[:, 1:2, :], y_true[:, 1:2, :])
                loss_gf = loss_g + f_mse_weight * loss_f
                
                if True:  # 始终计算物理损失
                    phy_components = calc_simplified_residual(
                        pred_tensor=y_pred, kappa=kappa, dr=dr,
                        n_principal=n_principal, y_true=y_true
                    )
                    loss_phy = (eff_lambda_pde * phy_components['loss_pde'] +
                                eff_lambda_norm * phy_components['loss_norm'] +
                                eff_lambda_node * phy_components['loss_node'] +
                                eff_lambda_physical * phy_components['loss_physical_state'] +
                                eff_lambda_smooth * phy_components['loss_smoothness'] +
                                eff_lambda_rayleigh * phy_components['loss_energy_rayleigh'])
                else:
                    loss_phy = torch.tensor(0.0, device=device)
                
                loss = (lambda_data * loss_gf + loss_phy) / 1
            
            if device_str == 'cuda':
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # 每10个epoch画一次图
        if epoch % 10 == 0 or epoch == 1 or epoch == num_epochs:
            model.eval()
            with torch.no_grad():
                # 用最后一个batch的样本绘图
                sample_y_pred = model(x_seq_norm, kappa, batch_r_grid,
                                       is_proton=is_proton, z_num=z_num, n_num=n_num,
                                       n_principal=n_principal)
                
                # 绘制波函数对比
                _plot_quick_verify(sample_y_pred, y_true, base_r_grid, kappa, 
                                    epoch, save_dir, is_proton, n_principal)
                
                # 检测是否退化成delta
                g_pred = sample_y_pred[:, 0, :]
                f_pred = sample_y_pred[:, 1, :]
                prob_density = (g_pred**2 + f_pred**2) * dr
                prob_sum = prob_density.sum(dim=-1)
                max_prob = prob_density.max(dim=-1)[0]
                is_delta = (max_prob > 2.0).any().item()  # 归一化后单点概率>2 → delta!
                
                status = "DELTA!" if is_delta else "OK"
                print(f"Epoch [{epoch:3d}/{num_epochs}] Loss={avg_loss:.4f} | "
                      f"max_ρ={max_prob.max().item():.2f} | [{status}]")
    
    print(f"\n✅ 验证完成! 波函数图保存在 {save_dir}/verify_fix_*.png")
    return True


def _no_autocast():
    """CPU模式下禁用autocast的context manager"""
    from contextlib import nullcontext
    return nullcontext()


def _plot_quick_verify(pred_tensor, true_tensor, r_grid, kappa, epoch, 
                       save_dir, is_proton=None, n_principal=None):
    """快速绘制波函数验证图"""
    import torch.nn.functional as F
    
    os.makedirs(save_dir, exist_ok=True)
    B = pred_tensor.shape[0]
    device = pred_tensor.device
    dr_val = 0.10
    
    # 归一化
    def _normalize_wf(g, f):
        prob = g**2 + f**2
        integral = torch.sum(prob, dim=-1) * dr_val
        integral = integral.clamp(min=1e-12)
        nf = (1.0 / torch.sqrt(integral)).unsqueeze(-1)
        return g * nf, f * nf
    
    pred_g_all = pred_tensor[:, 0, :]
    pred_f_all = pred_tensor[:, 1, :]
    true_g_all = true_tensor[:, 0, :]
    true_f_all = true_tensor[:, 1, :]
    pred_g_all_norm, pred_f_all_norm = _normalize_wf(pred_g_all, pred_f_all)
    true_g_all_norm, true_f_all_norm = _normalize_wf(true_g_all, true_f_all)
    
    # 找1s1/2样本
    target_idx = 0
    if n_principal is not None:
        for i in range(B):
            if n_principal[i].item() == 1.0:
                k_i = kappa[i].item() if kappa.numel() > 1 else kappa.item()
                if k_i == -1.0 or k_i == -1:
                    target_idx = i
                    break
    
    pred_g = pred_g_all_norm[target_idx].detach().cpu().numpy()
    pred_f = pred_f_all_norm[target_idx].detach().cpu().numpy()
    true_g = true_g_all_norm[target_idx].detach().cpu().numpy()
    true_f = true_f_all_norm[target_idx].detach().cpu().numpy()
    
    r = r_grid[0].detach().cpu().numpy() if r_grid.dim() == 2 else r_grid.detach().cpu().numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'[FIX VERIFICATION] Epoch {epoch} | 1s1/2', fontsize=14, fontweight='bold')
    
    axes[0, 0].plot(r, true_g, 'b-', lw=2, label='True g(r)')
    axes[0, 0].plot(r, pred_g, 'r--', lw=2, label='Pred g(r)')
    axes[0, 0].set_xlabel('r (fm)')
    axes[0, 0].set_ylabel('g(r)')
    axes[0, 0].set_title('Large Component g(r)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(r, true_f, 'b-', lw=2, label='True f(r)')
    axes[0, 1].plot(r, pred_f, 'r--', lw=2, label='Pred f(r)')
    axes[0, 1].set_xlabel('r (fm)')
    axes[0, 1].set_ylabel('f(r)')
    axes[0, 1].set_title('Small Component f(r)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    prob_true = true_g**2 + true_f**2
    prob_pred = pred_g**2 + pred_f**2
    axes[1, 0].plot(r, prob_true, 'b-', lw=2, label=r'True $|g|^2+|f|^2$')
    axes[1, 0].plot(r, prob_pred, 'r--', lw=2, label=r'Pred $|g|^2+|f|^2$')
    axes[1, 0].set_xlabel('r (fm)')
    axes[1, 0].set_ylabel(r'$\rho(r)$')
    axes[1, 0].set_title('Probability Density (归一化后)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    err_g = np.abs(pred_g - true_g)
    err_f = np.abs(pred_f - true_f)
    axes[1, 1].semilogy(r, np.maximum(err_g, 1e-15), 'g-', lw=1.5, label='|g_pred - g_true|')
    axes[1, 1].semilogy(r, np.maximum(err_f, 1e-15), 'm-', lw=1.5, label='|f_pred - f_true|')
    axes[1, 1].set_xlabel('r (fm)')
    axes[1, 1].set_ylabel('Absolute Error')
    axes[1, 1].set_title('Pointwise Error (log scale)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(save_dir, f'verify_fix_epoch{epoch:04d}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


if __name__ == "__main__":
    quick_verify(num_epochs=31, device_str='cpu')
