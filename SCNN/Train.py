import os
import re
import csv
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Model_Architecture import RHF_FNO_GRU
from Physics_Informed_Loss import calc_physics_residual
from Data_Loader import build_datasets, get_zn, ISOTOPE_ZN

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()


def save_checkpoint(model, optimizer, scheduler, epoch, loss, save_dir, save_name):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'loss': loss,
    }
    save_path = os.path.join(save_dir, save_name)
    torch.save(checkpoint, save_path)
    return save_path


def plot_wavefunctions(pred_tensor_norm, true_tensor, r_grid, kappa, epoch, save_dir,
                        stats_mean, stats_std, is_proton=None, n_principal=None):
    """
    绘制预测波函数与真实波函数的对比图

    ★ 关键改进：统一绘制归一化后的波函数 (∫(g²+f²)dr = 1)
    这样不同核素/轨道可以直接比较波形形状
    """
    os.makedirs(save_dir, exist_ok=True)

    B = pred_tensor_norm.shape[0]
    device = pred_tensor_norm.device
    dr_val = 0.10

    # ══════════════════════════════════════
    # ★ 统一归一化：将 True 和 Pred 都归一化到 ∫(g²+f²)dr = 1
    # 注意：模型输出 (B,2,L) 仅含 g/f 两个通道，
    #       stats_mean/std 是输入 X 的 11 通道统计量，不能用于反归一化输出！
    #       Y_true 和模型输出均已是物理单位，直接做概率归一化即可。
    # ══════════════════════════════════════
    pred_phys = pred_tensor_norm   # 输出已是物理单位
    true_phys = true_tensor        # Y 已经是物理量纲

    def _normalize_wf(g, f):
        """将 (g,f) 归一化: ∫(g²+f²)dr = 1"""
        prob = g**2 + f**2
        integral = torch.sum(prob, dim=-1) * dr_val if g.dim() > 1 else torch.sum(prob) * dr_val
        integral = integral.clamp(min=1e-12)
        nf = (1.0 / torch.sqrt(integral)).unsqueeze(-1)
        return g * nf, f * nf

    # 对所有样本做归一化
    pred_g_all = pred_phys[:, 0, :]
    pred_f_all = pred_phys[:, 1, :]
    true_g_all = true_phys[:, 0, :]
    true_f_all = true_phys[:, 1, :]
    pred_g_all_norm, pred_f_all_norm = _normalize_wf(pred_g_all, pred_f_all)
    true_g_all_norm, true_f_all_norm = _normalize_wf(true_g_all, true_f_all)

    # ══════════════════════════════════════
    # ★ 精确查找 1s1/2 样本 (n=1, κ=-1)
    # 1s1/2 是任何原子核的基态束缚轨道，一定存在！
    # ══════════════════════════════════════
    target_batch_idx = -1
    if n_principal is not None:
        # 方法1: 直接用 n_principal == 1 精确匹配（最优）
        for i in range(B):
            if n_principal[i].item() == 1.0:
                k_i = kappa[i].item() if kappa.numel() > 1 else kappa.item()
                if k_i == -1.0 or k_i == -1:  # s₁/₂ 的 κ=-1
                    target_batch_idx = i
                    break

    if target_batch_idx < 0:
        # 兜底方法2: 遍历所有 κ=-1 选最少节点的
        candidates = []
        for i in range(B):
            k_i = kappa[i].item() if kappa.numel() > 1 else kappa.item()
            if k_i == -1.0:
                tg = true_tensor[i, 0, :].cpu().numpy() if true_tensor.dim() == 3 else true_tensor[0, :].cpu().numpy()
                signs = np.sign(tg); signs[signs == 0] = 1
                crossings = np.sum(signs[1:] != signs[:-1]) / 2
                candidates.append((i, crossings))
        if candidates:
            target_batch_idx = min(candidates, key=lambda x: x[1])[0]

    if target_batch_idx < 0:
        # 最终兜底：取 batch 第一个样本（不应发生）
        target_batch_idx = 0

    pred_g = pred_g_all_norm[target_batch_idx].cpu().detach().numpy()
    pred_f = pred_f_all_norm[target_batch_idx].cpu().detach().numpy()
    true_g = true_g_all_norm[target_batch_idx].cpu().detach().numpy()
    true_f = true_f_all_norm[target_batch_idx].cpu().detach().numpy()

    if len(r_grid.shape) == 2:
        r = r_grid[target_batch_idx].cpu().numpy()
    else:
        r = r_grid.cpu().numpy()

    particle_label = "Proton(P)" if (is_proton is not None and is_proton[target_batch_idx].item() > 0.5) else "Neutron(N)"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Wavefunction Validation @ Epoch {epoch} | 1s1/2 | {particle_label}', fontsize=14, fontweight='bold')

    axes[0, 0].plot(r, true_g, 'b-', linewidth=2, label='True g(r)')
    axes[0, 0].plot(r, pred_g, 'r--', linewidth=2, label='Pred g(r)')
    axes[0, 0].set_xlabel('r (fm)')
    axes[0, 0].set_ylabel('g(r)')
    axes[0, 0].set_title('Large Component g(r) — 1s1/2')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(r, true_f, 'b-', linewidth=2, label='True f(r)')
    axes[0, 1].plot(r, pred_f, 'r--', linewidth=2, label='Pred f(r)')
    axes[0, 1].set_xlabel('r (fm)')
    axes[0, 1].set_ylabel('f(r)')
    axes[0, 1].set_title('Small Component f(r) — 1s1/2')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    prob_true = true_g**2 + true_f**2
    prob_pred = pred_g**2 + pred_f**2
    axes[1, 0].plot(r, prob_true, 'b-', linewidth=2, label=r'True $|g|^2+|f|^2$')
    axes[1, 0].plot(r, prob_pred, 'r--', linewidth=2, label=r'Pred $|g|^2+|f|^2$')
    axes[1, 0].set_xlabel('r (fm)')
    axes[1, 0].set_ylabel(r'$\rho(r)$')
    axes[1, 0].set_title('Radial Probability Density — 1s1/2')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    err_g = np.abs(pred_g - true_g)
    err_f = np.abs(pred_f - true_f)
    axes[1, 1].semilogy(r, np.maximum(err_g, 1e-15), 'g-', linewidth=1.5, label='|g_pred - g_true|')
    axes[1, 1].semilogy(r, np.maximum(err_f, 1e-15), 'm-', linewidth=1.5, label='|f_pred - f_true|')
    axes[1, 1].set_xlabel('r (fm)')
    axes[1, 1].set_ylabel('Absolute Error')
    axes[1, 1].set_title('Pointwise Absolute Error (log scale) — 1s1/2')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'wavefunction_1s12_epoch{epoch:04d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def verify_normalization(wave_tensor, r_grid, kappa, dr, stats_mean, stats_std, tol=1e-3):
    """验证波函数的归一化条件（直接用物理单位张量计算）"""
    # wave_tensor 形状为 (B,2,L) — g/f 两通道，已是物理单位
    # stats_mean/std 是输入X的11通道统计量，不适用于输出
    device = wave_tensor.device

    g = wave_tensor[:, 0, :].cpu().detach()
    f = wave_tensor[:, 1, :].cpu().detach()
    prob = g**2 + f**2

    norm_integral = torch.trapz(prob, dx=dr, dim=1)
    is_normalized = torch.allclose(norm_integral, torch.ones_like(norm_integral), atol=tol)

    return norm_integral, is_normalized


def calculate_normalization_stats(dataloader, local_rank, use_ddp=False):
    """计算全局物理通道的 Mean & Std。"""
    if local_rank == 0 or not use_ddp:
        print("正在计算相空间全局物理通道统计信息...")

    sum_x = torch.zeros(11)
    sum_sq_x = torch.zeros(11)
    num_samples = 0

    for batch_data in dataloader:
        x_seq = batch_data[0]  # (B, L, 12, N)
        batch_size, seq_len, channels, npt = x_seq.shape
        x_physics = x_seq[:, :, :11, :]
        x_flat = x_physics.view(-1, 11, npt).permute(1, 0, 2).reshape(11, -1)

        sum_x += x_flat.sum(dim=1)
        sum_sq_x += (x_flat ** 2).sum(dim=1)
        num_samples += batch_size * seq_len * npt

    mean = sum_x / num_samples
    std = torch.sqrt(sum_sq_x / num_samples - mean ** 2)
    std = torch.clamp(std, min=1e-8)

    if local_rank == 0 or not use_ddp:
        print("通道均值:", mean.numpy())
        print("通道标准差:", std.numpy())
    return mean, std


def calculate_y_stats(dataloader, local_rank, use_ddp=False):
    """计算Y目标（最终收敛态）的11通道统计量，与X输入统计量分离。"""
    if local_rank == 0 or not use_ddp:
        print("正在计算Y目标通道统计信息...")

    sum_y = torch.zeros(11)
    sum_sq_y = torch.zeros(11)
    num_y = 0

    for batch_data in dataloader:
        y_true = batch_data[1]  # (B, 11, N)
        B, C, N = y_true.shape
        y_flat = y_true.permute(1, 0, 2).reshape(11, -1)
        sum_y += y_flat.sum(dim=1)
        sum_sq_y += (y_flat ** 2).sum(dim=1)
        num_y += B * N

    y_mean = sum_y / num_y
    y_std = torch.sqrt(sum_sq_y / num_y - y_mean ** 2)
    y_std = torch.clamp(y_std, min=1e-8)

    if local_rank == 0 or not use_ddp:
        print("Y通道均值:", y_mean.numpy())
        print("Y通道标准差:", y_std.numpy())
    return y_mean, y_std


def normalize(tensor, mean, std):
    """归一化张量，根据 mean/std 的实际通道数自适应。"""
    C = len(mean)
    mean_view = mean.view(1, C, 1).to(tensor.device)
    std_view = std.view(1, C, 1).to(tensor.device)
    if len(tensor.shape) == 4:
        mean_view = mean_view.unsqueeze(1)
        std_view = std_view.unsqueeze(1)

    n_channels = tensor.shape[-2]
    if n_channels >= 12:
        result = tensor.clone()
        result[..., :11, :] = (tensor[..., :11, :] - mean_view) / std_view
        return result
    elif n_channels == 11:
        return (tensor - mean_view) / std_view
    else:
        return tensor


def _evaluate(model, dataloader, device, stats_mean, stats_std, y_mean, y_std,
              base_r_grid, criterion, use_ddp):
    """在验证集或测试集上评估模型"""
    model.eval()
    total_loss = 0.0
    total_data_loss = 0.0
    n_samples = 0
    norm_integrals = []
    loss_pde_val = 0.0
    loss_norm_val = 0.0

    for batch_data in dataloader:
        x_batch = batch_data[0].to(device)       # (B, L, 12, N)
        y_true = batch_data[1].to(device)         # (B, 11, N) — Y现在是11ch物理值
        kappa_true = batch_data[2].to(device)
        is_proton = batch_data[3].to(device) if len(batch_data) > 3 else None
        z_num = batch_data[5].to(device) if len(batch_data) > 5 else None
        n_num = batch_data[6].to(device) if len(batch_data) > 6 else None
        n_principal = batch_data[7].to(device) if len(batch_data) > 7 else None

        B = x_batch.size(0)
        batch_r_grid = base_r_grid.unsqueeze(0).expand(B, -1)

        x_norm = normalize(x_batch, stats_mean, stats_std)

        with torch.no_grad():
            y_pred = model(x_norm, kappa_true, batch_r_grid,
                            is_proton=is_proton, z_num=z_num, n_num=n_num,
                            n_principal=n_principal)

            # ★ 分通道MSE: g/f(0,1)在物理空间, 其余通道用y_stats归一化后
            loss_gf = nn.functional.mse_loss(y_pred[:, :2, :], y_true[:, :2, :])
            y_true_others_norm = normalize(y_true[:, 2:, :], y_mean[2:], y_std[2:])
            y_pred_others_norm = normalize(y_pred[:, 2:, :], y_mean[2:], y_std[2:])
            loss_others = nn.functional.mse_loss(y_pred_others_norm, y_true_others_norm)
            loss_data = loss_gf + loss_others

            try:
                phy_comp = calc_physics_residual(
                    y_pred, kappa_true, y_mean, y_std,
                    dr=0.10, return_components=True,
                    n_principal=n_principal
                )
                loss_pde_val = phy_comp['loss_pde'].item()
                loss_norm_val = phy_comp['loss_norm'].item()
                norm_integrals.append(phy_comp['norm_integral'].item())
            except Exception:
                loss_pde_val = 0.0
                loss_norm_val = 0.0

        total_loss += loss_data.item()
        total_data_loss += loss_data.item()
        n_samples += B

    n_batches = max(len(dataloader), 1)

    metrics = {
        'total': total_loss / n_batches,
        'loss_data': total_data_loss / n_batches,
        'pde': loss_pde_val,
        'norm': loss_norm_val,
        'norm_integral_mean': np.mean(norm_integrals) if norm_integrals else 0.0,
        'n_samples': n_samples
    }

    model.train()
    return metrics


def train_model():
    # 1. 检测环境
    use_ddp = int(os.environ.get("RANK", -1)) >= 0

    if use_ddp:
        local_rank = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
        is_main = (local_rank == 0)
        if is_main:
            print(f"🔥 DDP 分布式拓扑已建立，{torch.cuda.device_count()} 张 GPU")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        local_rank = 0
        is_main = True
        print(f"⚡ 单卡模式，设备: {device}")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

    # ================================================================
    #   全局超参数面板
    # ================================================================

    # --- 模型结构超参数 ---
    hidden_dim = 96
    gru_hidden = 1536
    modes = 40
    weight_decay = 1e-4

    # --- ★ 物理主导损失权重 ---
    # 核心策略: PDE残差是优化主驱动力，MSE仅作为辅助引导
    # lambda_data: 控制MSE数据拟合的权重（小→弱数据约束）
    # lambda_pde:  控制狄拉克方程残差的权重（大→强物理驱动）
    # lambda_norm: 归一化约束权重
    # lambda_node: 节点数精确约束权重
    lambda_data = 1.0         # ★ MSE数据拟合权重（恢复数据引导力）
    lambda_pde = 2.0          # ★ PDE残差权重（降低，避免过度压制数据拟合）
    lambda_norm = 5.0         # 归一化强约束
    lambda_node = 8.0         # 节点数精确硬约束
    lambda_sign = 5.0        # ★ 已删除，替代为能量约束
    lambda_boundary = 3.0    # 边界衰减约束
    lambda_energy = 10.0     # ★ 正能量约束（标量密度 > 0）
    lambda_energy_range = 5.0 # ★ 能量范围惩罚 (-80 ~ +50 MeV)
    lambda_kinetic = 15.0    # ★ 动能正定性（防负能量海）
    lambda_shape = 8.0       # ★ 波形形态惩罚（高斯波包相似度，防偷懒）
    lambda_bsmooth = 5.0     # ★ 边界平滑性保护（反震荡）

    # --- 课程学习超参数（速度优化版）---
    # 阶段1 (Epoch 1-30):   仅双幻核, λ_physics=0
    # 阶段2 (Epoch 31-100): 全核素解锁, λ_physics Sigmoid→0.1
    # 阶段3 (Epoch 101-120): Cosine Annealing LR 收尾
    curriculum_phase1_epochs = 30
    curriculum_phase2_epochs = 100  # Phase3从101到300
    num_epochs = 300          # ★ 减少总epoch（足够收敛）
    learning_rate = 5e-4     # ★ 提高学习率补偿小batch
    batch_size = 32          # ★ 匹配小数据量，避免梯度估计不稳定
    grad_accum_steps = 1
    # ★ 物理主导损失权重（已在上方定义）
    clip_grad_norm = 1.5

    # --- 物理损失频率：每 N 个 batch 算一次（跳过大部分batch）---
    phy_loss_every_n_batches = 1  # 数据少，每batch都算物理损失

    # --- 序列长度参数（放宽以增加样本数）---
    max_seq_len = 15
    min_seq_len = 2        # 原3，降低→更多滑动窗口
    traj_usage_ratio = 0.95 # 原0.8，提高→更长轨迹利用

    # --- 各阶段核素列表 ---
    phase1_isotopes = ['16O', '40Ca']  # 双幻核
    all_isotopes = [
        '16O', '18O', '20O', '22O',
        '36Ca', '38Ca', '40Ca', '42Ca', '44Ca', '46Ca', '48Ca',
        '56Ni', '58Ni', '60Ni', '62Ni', '64Ni', '68Ni', '72Ni', '78Ni',
        '100Sn', '112Sn', '116Sn', '120Sn', '124Sn', '132Sn',
        '206Pb', '208Pb', '210Pb',
        '86Kr', '88Sr', '90Zr', '92Mo',
    ]

    # ══════════════════════════════════════════════════════════
    #   完整的核子态列表 — 涵盖狄拉克基的所有nlj轨道
    #   数据格式: 每个核素含 it001(中子N) + it002(质子P), 各42个态
    # ══════════════════════════════════════════════════════════
    target_states = [
        # --- s 轨道 (l=0, κ=-1, j=1/2) ---
        '1s1/2', '2s1/2', '3s1/2', '4s1/2', '5s1/2', '6s1/2',
        # --- p3/2 轨道 (l=1, κ=-2, j=3/2) ---
        '1p3/2', '2p3/2', '3p3/2', '4p3/2', '5p3/2', '6p3/2',
        # --- d5/2 轨道 (l=2, κ=-3, j=5/2) ---
        '1d5/2', '2d5/2', '3d5/2', '4d5/2', '5d5/2',
        # --- f7/2 轨道 (l=3, κ=-4, j=7/2) ---
        '1f7/2', '2f7/2', '3f7/2', '4f7/2', '5f7/2',
        # --- p1/2 轨道 (l=1, κ=+1, j=1/2) ---
        '1p1/2', '2p1/2', '3p1/2', '4p1/2', '5p1/2', '6p1/2',
        # --- d3/2 轨道 (l=2, κ=+2, j=3/2) ---
        '1d3/2', '2d3/2', '3d3/2', '4d3/2', '5d3/2',
        # --- f5/2 轨道 (l=3, κ=+3, j=5/2) ---
        '1f5/2', '2f5/2', '3f5/2', '4f5/2', '5f5/2',
        # --- g7/2 轨道 (l=4, κ=-4, j=7/2) ---
        '1g7/2', '2g7/2', '3g7/2', '4g7/2',
    ]  # 共 6+6+5+5+6+5+5+4 = 42 个态 × 2粒子(N+P) = 84 个轨道

    # --- 固定参数 ---
    data_dir = '/home/ubuntu/rhf/results'
    dr = 0.10
    save_interval = 50
    plot_interval = 25
    checkpoint_dir = '/home/ubuntu/rhf/SCNN/checkpoints'
    plot_dir = '/home/ubuntu/rhf/SCNN/plots'
    log_dir = '/home/ubuntu/rhf/SCNN/training_logs'
    os.makedirs(log_dir, exist_ok=True)

    log_csv_path = os.path.join(log_dir, 'training_loss_log.csv')

    if is_main:
        effective_batch = batch_size * grad_accum_steps
        print(f"\n   🚀 AMP混合精度: ON")
        print(f"   📦 batch_size={batch_size}, grad_accum={grad_accum_steps} → 等效batch={effective_batch}")
        print(f"   🧠 模型规模: hidden_dim={hidden_dim}, gru_hidden={gru_hidden}, modes={modes}")
        print(f"   ⏱️  物理损失频率: 每 {phy_loss_every_n_batches} 个batch算一次")
        print(f"   📚 课程学习: Phase1(Ep1-{curriculum_phase1_epochs}) → Phase2(Ep{curriculum_phase1_epochs+1}-{curriculum_phase2_epochs}) → Phase3(Ep{curriculum_phase2_epochs+1}-{num_epochs})")

    # ================================================================
    #   课程学习：阶段1 初始化
    # ================================================================
    if is_main:
        print(f"\n{'='*60}")
        print(f"  📚 课程学习阶段1: 双幻核预热 (Epoch 1-{curriculum_phase1_epochs})")
        print(f"{'='*60}")

    current_isotopes = phase1_isotopes
    current_phase = 1

    # 构建阶段1数据集
    train_dataset = build_datasets(data_dir, current_isotopes,
                                   max_seq_len=max_seq_len, min_seq_len=min_seq_len,
                                   traj_usage_ratio=traj_usage_ratio,
                                   mode='train', target_states=target_states)

    if train_dataset is None:
        raise ValueError("训练数据集为空！请检查数据和目标态设置。")

    val_dataset = build_datasets(data_dir, all_isotopes,
                                 max_seq_len=max_seq_len, min_seq_len=min_seq_len,
                                 traj_usage_ratio=traj_usage_ratio,
                                 mode='val', target_states=target_states)

    test_dataset = build_datasets(data_dir, all_isotopes,
                                  max_seq_len=max_seq_len, min_seq_len=min_seq_len,
                                  traj_usage_ratio=traj_usage_ratio,
                                  mode='test', target_states=target_states)

    # DataLoader
    def make_loader(dataset, batch_sz, shuffle=True):
        if dataset is None or len(dataset) == 0:
            return None
        actual_batch = min(batch_sz, max(16, len(dataset)))
        if use_ddp:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            return DataLoader(dataset, batch_size=actual_batch, sampler=sampler,
                              num_workers=4, pin_memory=True, persistent_workers=True,
                              drop_last=len(dataset) >= actual_batch * 2)
        return DataLoader(dataset, batch_size=actual_batch, shuffle=shuffle,
                          num_workers=4, pin_memory=True, persistent_workers=True,
                          drop_last=len(dataset) >= actual_batch * 2)

    train_loader = make_loader(train_dataset, batch_size, shuffle=True)
    val_loader = make_loader(val_dataset, batch_size, shuffle=False)
    test_loader = make_loader(test_dataset, batch_size, shuffle=False)

    if is_main:
        print(f"✅ 数据集构建完成:")
        print(f"   - 训练集: {len(train_dataset)} 样本 ({len(train_loader)} batches)")
        print(f"   - 验证集: {len(val_dataset) if val_dataset else 0} 样本")
        print(f"   - 测试集: {len(test_dataset) if test_dataset else 0} 样本")

    if train_loader is None or len(train_loader) == 0:
        raise RuntimeError("训练数据为空！")

    # 统计量
    mean, std = calculate_normalization_stats(train_loader, local_rank, use_ddp=use_ddp)
    y_mean, y_std = calculate_y_stats(train_loader, local_rank, use_ddp=use_ddp)

    # 3. 实例化模型
    if is_main:
        print(f"\n🏗️  正在实例化条件化FNO模型...")
        import time; t0 = time.time()

    model = RHF_FNO_GRU(in_channels=12, hidden_dim=hidden_dim, npt=201,
                        gru_hidden=gru_hidden, modes=modes).to(device)

    if is_main:
        total_p = sum(p.numel() for p in model.parameters())
        print(f"   模型构建完成: {total_p:,} 参数, 耗时 {time.time()-t0:.1f}s")

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = GradScaler()
    mse_criterion = nn.MSELoss()

    # 径向网格
    base_r_grid = torch.arange(0, 201, device=device, dtype=torch.float32) * dr
    base_r_grid[0] = 0.0010

    # 初始化 CSV 日志（★ 物理主导模式）
    if is_main:
        with open(log_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'phase', 'total_loss', 'loss_data', 'loss_pde', 'loss_norm',
                           'loss_node', 'loss_boundary', 'loss_energy', 'loss_kinetic',
                           'loss_energy_range', 'loss_shape', 'loss_bsmooth', 'energy_pred', 'vps_core',
                           'learning_rate', 'best_epoch', 'active_isotopes'])

    best_data_loss = float('inf')
    patience = 9999
    patience_counter = 0
    best_epoch = 0
    early_stopped = False

    if is_main:
        print("\n🚀 开始课程学习训练...")
        import time as _t
        _train_start = _t.time()

    # 4. 训练迭代大循环
    for epoch in range(1, num_epochs + 1):

        # ══════════════════════════════════════
        # 课程学习阶段切换逻辑
        # ══════════════════════════════════════
        new_phase = 1
        new_isotopes = phase1_isotopes

        if epoch > curriculum_phase1_epochs and epoch <= curriculum_phase2_epochs:
            new_phase = 2
            new_isotopes = all_isotopes
        elif epoch > curriculum_phase2_epochs:
            new_phase = 3
            new_isotopes = all_isotopes

        # 阶段切换时重建数据集
        if new_phase != current_phase:
            current_phase = new_phase
            current_isotopes = new_isotopes

            if is_main:
                phase_names = {1: "双幻核预热", 2: "多体耦合", 3: "极小值寻优"}
                print(f"\n{'='*60}")
                print(f"  📚 课程学习阶段{new_phase}: {phase_names[new_phase]} (Epoch {epoch}+)")
                print(f"  📦 激活核素: {current_isotopes}")
                print(f"{'='*60}")

            # 重建训练集
            train_dataset = build_datasets(data_dir, current_isotopes,
                                           max_seq_len=max_seq_len, min_seq_len=min_seq_len,
                                           traj_usage_ratio=traj_usage_ratio,
                                           mode='train', target_states=target_states)
            if train_dataset is not None and len(train_dataset) > 0:
                train_loader = make_loader(train_dataset, batch_size, shuffle=True)
                # 重新计算统计量
                mean, std = calculate_normalization_stats(train_loader, local_rank, use_ddp=use_ddp)
                y_mean, y_std = calculate_y_stats(train_loader, local_rank, use_ddp=use_ddp)

        if use_ddp and hasattr(train_loader, 'sampler') and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        # ══════════════════════════════════════
        # 物理损失权重调度
        # 从第1轮就启用物理约束（无预热）
        # ★ 物理主导模式：始终全量启用物理约束（无需调度）
        # 阶段3: Cosine Annealing LR 收尾时保持 λ
        # ══════════════════════════════════════
        # （compute_physics 在每个 batch 内部计算）

        # 学习率调度
        if current_phase == 3:
            # Cosine Annealing（论文 §4 极小值寻优期）
            if not hasattr(model, '_cosine_scheduler_initialized'):
                # 在阶段3开始时创建 cosine scheduler
                remaining_epochs = num_epochs - epoch + 1
                _cosine_scheduler = CosineAnnealingLR(optimizer, T_max=remaining_epochs, eta_min=1e-6)
                model._cosine_scheduler_initialized = True

        model.train()
        total_loss = 0.0
        total_loss_data = 0.0
        total_loss_pde = 0.0
        total_loss_norm = 0.0
        total_loss_node = 0.0
        total_loss_boundary = 0.0
        total_loss_energy = 0.0
        total_loss_kinetic = 0.0
        total_loss_shape = 0.0
        total_loss_bsmooth = 0.0
        total_loss_energy_range = 0.0
        num_batches = 0

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch_data in enumerate(train_loader):
            # 解包八元组: (X, Y_11ch, kappa, is_proton, actual_len, z_num, n_num, n_principal)
            x_seq = batch_data[0].to(device)       # (B, L, 12, N)
            y_true = batch_data[1].to(device)       # (B, 11, N) — 最终收敛态
            kappa = batch_data[2].to(device)
            is_proton = batch_data[3].to(device) if len(batch_data) > 3 else None
            # actual_len = batch_data[4]  # 不需要放到GPU
            z_num = batch_data[5].to(device) if len(batch_data) > 5 else None
            n_num = batch_data[6].to(device) if len(batch_data) > 6 else None
            n_principal = batch_data[7].to(device) if len(batch_data) > 7 else None  # ★ 主量子数

            B = x_seq.size(0)

            # 物理损失频率控制（物理主导策略：每batch都算）
            compute_physics = (batch_idx % phy_loss_every_n_batches == 0)
            batch_r_grid = base_r_grid.unsqueeze(0).expand(B, -1)

            x_seq_norm = normalize(x_seq, mean, std)
            # ★ Y不再用X的stats归一化！g/f通道在物理空间计算MSE，其余通道用Y自己的stats

            with autocast('cuda'):
                y_pred = model(x_seq_norm, kappa, batch_r_grid,
                                    is_proton=is_proton, z_num=z_num, n_num=n_num,
                                    n_principal=n_principal)  # ★ 传入主量子数

                # ★ 分通道MSE计算：
                # g/f通道(0,1): 模型输出已是物理空间值(硬归一化后), 直接与Y_true物理值比较
                loss_gf = nn.functional.mse_loss(y_pred[:, :2, :], y_true[:, :2, :])
                # 其余通道(2-10): 用Y的stats归一化后做MSE
                y_true_others_norm = normalize(y_true[:, 2:, :], y_mean[2:], y_std[2:])
                y_pred_others_norm = normalize(y_pred[:, 2:, :], y_mean[2:], y_std[2:])
                loss_others = nn.functional.mse_loss(y_pred_others_norm, y_true_others_norm)
                loss_data = loss_gf + loss_others

                # 物理损失（频率控制加速）
                if compute_physics:
                    if not hasattr(model, '_phy_ref_scale'):
                        phy_components = calc_physics_residual(
                            pred_tensor=y_pred,
                            kappa=kappa,
                            stats_mean=y_mean,
                            stats_std=y_std,
                            dr=dr,
                            ref_scale=None,
                            return_components=True,
                            n_principal=n_principal  # ★ 传入主量子数
                        )
                        ref_total = phy_components['loss_total'].detach().clone()
                        if use_ddp:
                            dist.all_reduce(ref_total, op=dist.ReduceOp.MAX)
                        estimated_ref = max(ref_total.item(), 100.0)
                        model._phy_ref_scale = estimated_ref
                        if is_main:
                            print(f"  📐 物理损失参考尺度: {estimated_ref:.2f}")
                        # ★ PDE/Norm 已在 calc_physics_residual 内部用 sqrt(ref) 缩放
                        # 这里不再额外除 ref，避免双重压制！
                        loss_pde = phy_components['loss_pde']
                        loss_norm = phy_components['loss_norm']  # 不再 / estimated_ref！
                        loss_node = phy_components['loss_node']
                        loss_boundary = phy_components.get('loss_boundary', torch.tensor(0.0, device=device))
                        loss_energy = phy_components.get('loss_positive_energy', torch.tensor(0.0, device=device))
                        loss_kinetic = phy_components.get('loss_kinetic_positive', torch.tensor(0.0, device=device))
                        loss_shape = phy_components.get('loss_shape', torch.tensor(0.0, device=device))       # ★
                        loss_bsmooth = phy_components.get('loss_boundary_smooth', torch.tensor(0.0, device=device))  # ★
                        loss_energy_range = phy_components.get('loss_energy_range', torch.tensor(0.0, device=device))  # ★ 能量范围
                    else:
                        phy_components = calc_physics_residual(
                            pred_tensor=y_pred,
                            kappa=kappa,
                            stats_mean=y_mean,
                            stats_std=y_std,
                            dr=dr,
                            ref_scale=model._phy_ref_scale,
                            return_components=True,
                            n_principal=n_principal  # ★ 传入主量子数
                        )
                        loss_pde = phy_components['loss_pde']
                        loss_norm = phy_components['loss_norm']
                        loss_node = phy_components.get('loss_node', torch.tensor(0.0, device=device))
                        loss_boundary = phy_components.get('loss_boundary', torch.tensor(0.0, device=device))
                        loss_energy = phy_components.get('loss_positive_energy', torch.tensor(0.0, device=device))
                        loss_kinetic = phy_components.get('loss_kinetic_positive', torch.tensor(0.0, device=device))
                        loss_shape = phy_components.get('loss_shape', torch.tensor(0.0, device=device))       # ★
                        loss_bsmooth = phy_components.get('loss_boundary_smooth', torch.tensor(0.0, device=device))  # ★
                        loss_energy_range = phy_components.get('loss_energy_range', torch.tensor(0.0, device=device))  # ★ 能量范围

                    # ★ 物理主导损失组合: PDE > Norm > Node > Energy/Kinetic > Shape/Boundary >> Data(MSE辅助)
                    loss_phy = (lambda_pde * loss_pde +          # ★ PDE 主导！狄拉克方程残差
                                lambda_norm * loss_norm +        # 归一化约束
                                lambda_node * loss_node +        # 节点数精确约束
                                lambda_energy * loss_energy +    # ★ 正能量约束（标量密度>0）
                                lambda_kinetic * loss_kinetic +  # ★ 动能正定性（防负能量海）
                                lambda_energy_range * loss_energy_range +  # ★ 能量范围 (-80~+50 MeV)
                                lambda_boundary * loss_boundary + # 边界端点约束
                                lambda_shape * loss_shape +       # ★ 波形形态惩罚（防偷懒）
                                lambda_bsmooth * loss_bsmooth)   # ★ 边界平滑性（反震荡）
                else:
                    loss_pde = torch.tensor(0.0, device=device)
                    loss_norm = torch.tensor(0.0, device=device)
                    loss_node = torch.tensor(0.0, device=device)
                    loss_boundary = torch.tensor(0.0, device=device)
                    loss_energy = torch.tensor(0.0, device=device)
                    loss_kinetic = torch.tensor(0.0, device=device)
                    loss_shape = torch.tensor(0.0, device=device)       # ★
                    loss_bsmooth = torch.tensor(0.0, device=device)    # ★
                    loss_energy_range = torch.tensor(0.0, device=device)  # ★ 能量范围
                    loss_phy = torch.tensor(0.0, device=device)

                # ★ 总损失 = 弱MSE辅助 + 强物理主导
                loss = (lambda_data * loss_data + loss_phy) / grad_accum_steps

            scaler.scale(loss).backward()

            # 梯度累积
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx == len(train_loader) - 1):
                saved = [(p, p.grad.clone()) for p in model.parameters()
                         if p.grad is not None and p.grad.is_complex()]
                for p, _ in saved:
                    p.grad = None

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

                for p, g in saved:
                    p.grad = g
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.data.item() * grad_accum_steps
            total_loss_data += loss_data.item()
            total_loss_pde += (loss_pde.item() * grad_accum_steps) if isinstance(loss_pde, torch.Tensor) and loss_pde.requires_grad else 0.0
            total_loss_norm += (loss_norm.item() * grad_accum_steps) if isinstance(loss_norm, torch.Tensor) and loss_norm.requires_grad else 0.0
            total_loss_node += (loss_node.item() * grad_accum_steps) if isinstance(loss_node, torch.Tensor) and loss_node.requires_grad else 0.0
            total_loss_boundary += (loss_boundary.item() * grad_accum_steps) if isinstance(loss_boundary, torch.Tensor) and loss_boundary.requires_grad else 0.0
            total_loss_energy += (loss_energy.item() * grad_accum_steps) if isinstance(loss_energy, torch.Tensor) and loss_energy.requires_grad else 0.0
            total_loss_kinetic += (loss_kinetic.item() * grad_accum_steps) if isinstance(loss_kinetic, torch.Tensor) and loss_kinetic.requires_grad else 0.0
            total_loss_shape += (loss_shape.item() * grad_accum_steps) if isinstance(loss_shape, torch.Tensor) and loss_shape.requires_grad else 0.0
            total_loss_bsmooth += (loss_bsmooth.item() * grad_accum_steps) if isinstance(loss_bsmooth, torch.Tensor) and loss_bsmooth.requires_grad else 0.0
            total_loss_energy_range += (loss_energy_range.item() * grad_accum_steps) if isinstance(loss_energy_range, torch.Tensor) and loss_energy_range.requires_grad else 0.0
            num_batches += 1

        # LR调度
        if current_phase < 3:
            # 阶段1-2: 使用简单的预热+常数LR
            if epoch <= 5:
                for pg in optimizer.param_groups:
                    pg['lr'] = learning_rate * epoch / 5
            # 阶段2后半段开始微降
            elif current_phase == 2 and epoch > curriculum_phase1_epochs + 50:
                decay = 1.0 - 0.5 * (epoch - curriculum_phase1_epochs - 50) / (curriculum_phase2_epochs - curriculum_phase1_epochs - 50)
                for pg in optimizer.param_groups:
                    pg['lr'] = learning_rate * max(decay, 0.5)
        else:
            # 阶段3: Cosine Annealing
            if hasattr(model, '_cosine_scheduler_initialized'):
                _cosine_scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # 日志聚合
        n_batches = max(num_batches, 1)
        loss_total_avg = total_loss / n_batches
        loss_data_avg = total_loss_data / n_batches
        loss_pde_avg = total_loss_pde / n_batches
        loss_norm_avg = total_loss_norm / n_batches
        loss_node_avg = total_loss_node / n_batches
        loss_boundary_avg = total_loss_boundary / n_batches
        loss_energy_avg = total_loss_energy / n_batches
        loss_kinetic_avg = total_loss_kinetic / n_batches
        loss_shape_avg = total_loss_shape / n_batches       # ★
        loss_bsmooth_avg = total_loss_bsmooth / n_batches   # ★
        loss_energy_range_avg = total_loss_energy_range / n_batches  # ★ 能量范围

        # Early Stopping
        current_data_loss = loss_data_avg
        if current_data_loss < best_data_loss:
            best_data_loss = current_data_loss
            patience_counter = 0
            best_epoch = epoch
            model_to_save = model.module if use_ddp else model
            save_checkpoint(model_to_save, optimizer, None, epoch,
                          best_data_loss, checkpoint_dir, 'rhf_fno_gru_best.pt')
        else:
            patience_counter += 1

        # Epoch 日志（★ 物理主导模式 + 能量诊断）
        if is_main:
            # ★ 从物理残差中提取能量和势场诊断值
            if hasattr(model, '_phy_ref_scale'):
                with torch.no_grad():
                    sample_y_ph = model(x_seq_norm, kappa, batch_r_grid,
                                        is_proton=is_proton, z_num=z_num, n_num=n_num,
                                        n_principal=n_principal)
                    phy_diag = calc_physics_residual(
                        sample_y_ph, kappa, y_mean, y_std,
                        dr=dr, return_components=True, n_principal=n_principal
                    )
                    energy_pred_val = phy_diag.get('energy_E', 0.0).item() if 'energy_E' in phy_diag else 0.0
                    vps_core_val = phy_diag.get('vps_core', 0.0).item() if 'vps_core' in phy_diag else 0.0
            else:
                energy_pred_val, vps_core_val = 0.0, 0.0

            with open(log_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, current_phase,
                    f"{loss_total_avg:.6f}", f"{loss_data_avg:.6f}",
                    f"{loss_pde_avg:.6f}", f"{loss_norm_avg:.6f}",
                    f"{loss_node_avg:.6f}", f"{loss_boundary_avg:.6f}",
                    f"{loss_energy_avg:.8f}", f"{loss_kinetic_avg:.8f}",
                    f"{loss_energy_range_avg:.8f}",
                    f"{loss_shape_avg:.6f}", f"{loss_bsmooth_avg:.6f}",
                    f"{energy_pred_val:.4f}", f"{vps_core_val:.4f}",
                    f"{current_lr:.2e}",
                    best_epoch, len(current_isotopes)
                ])

            if epoch % 5 == 0 or epoch == 1:
                phase_names = {1: "双幻核预热", 2: "多体耦合", 3: "极小值寻优"}
                print(f"Epoch [{epoch:3d}/{num_epochs}] | Phase{current_phase}({phase_names[current_phase]}) | LR: {current_lr:.2e}", flush=True)
                print(f"  [物理主导] Loss: {loss_total_avg:.4f} | Data: {loss_data_avg:.4f}(×{lambda_data}) | PDE: {loss_pde_avg:.4f}(×{lambda_pde}) | Norm: {loss_norm_avg:.4f}(×{lambda_norm}) | Node: {loss_node_avg:.4f}(×{lambda_node})", flush=True)
                print(f"             Energy: {loss_energy_avg:.6f}(×{lambda_energy}) | Kinetic: {loss_kinetic_avg:.6f}(×{lambda_kinetic}) | E_Range: {loss_energy_range_avg:.6f}(×{lambda_energy_range}) | Boundary: {loss_boundary_avg:.6f}(×{lambda_boundary})", flush=True)
                print(f"             Shape: {loss_shape_avg:.6f}(×{lambda_shape}) | BSmooth: {loss_bsmooth_avg:.6f}(×{lambda_bsmooth})", flush=True)
                print(f"             ★ E_pred={energy_pred_val:.2f} MeV | vps_core={vps_core_val:.2f}", flush=True)

            # 物理验证
            if epoch % plot_interval == 0 or epoch == 1:
                with torch.no_grad():
                    sample_y_pred = model(x_seq_norm, kappa, batch_r_grid,
                                         is_proton=is_proton, z_num=z_num, n_num=n_num,
                                         n_principal=n_principal)  # ★ 传入主量子数
                    norm_vals, is_norm = verify_normalization(
                        sample_y_pred, batch_r_grid, kappa, dr=dr,
                        stats_mean=y_mean, stats_std=y_std
                    )
                    print(f"  [Physics] Norm integral: {norm_vals.mean().item():.6f} | Valid: {is_norm}", flush=True)

                    # 绘图（传入真实物理值的Y_true和统计量以做反归一化）
                    plot_path = plot_wavefunctions(
                        sample_y_pred, y_true, batch_r_grid, kappa,
                        epoch, plot_dir, stats_mean=y_mean, stats_std=y_std,
                        is_proton=is_proton, n_principal=n_principal  # ★ 传入主量子数
                    )
                    if plot_path:
                        print(f"  [Plot] {plot_path}", flush=True)

            # 定期保存
            if epoch % save_interval == 0:
                model_to_save_cp = model.module if use_ddp else model
                save_path = save_checkpoint(model_to_save_cp, optimizer, None, epoch,
                                          loss_total_avg, checkpoint_dir,
                                          f'rhf_fno_gru_epoch{epoch:04d}.pt')
                print(f"  [Checkpoint] {save_path}", flush=True)

            # 验证集
            if val_loader is not None and (epoch % 10 == 0 or epoch == 1):
                with torch.no_grad():
                    val_metrics = _evaluate(model, val_loader, device, mean, std, y_mean, y_std,
                                            base_r_grid, mse_criterion, use_ddp)
                    print(f"  [Val] Loss={val_metrics['total']:.6f} | Data={val_metrics['loss_data']:.6f} | "
                          f"PDE={val_metrics.get('pde',0):.4f} | Norm={val_metrics.get('norm',0):.4f}", flush=True)

    # ===== 训练结束 =====
    if is_main:
        if test_loader is not None:
            print("\n" + "=" * 60)
            print("  🧪 测试集最终评估")
            print("=" * 60)
            model_to_eval = model.module if use_ddp else model
            model_to_eval.eval()

            with torch.no_grad():
                test_metrics = _evaluate(model_to_eval, test_loader, device, mean, std, y_mean, y_std,
                                         base_r_grid, mse_criterion, use_ddp=False)
                print(f"\n  📊 测试结果:")
                print(f"     Total Loss:     {test_metrics['total']:.6f}")
                print(f"     Data MSE:       {test_metrics['loss_data']:.6f}")
                print(f"     PDE Residual:   {test_metrics.get('pde', 0):.4e}")
                print(f"     Norm Error:     {test_metrics.get('norm', 0):.4e}")
                print(f"     Norm Integral:  {test_metrics.get('norm_integral_mean', 0):.6f}")

        # 保存最终模型
        model_to_save = model.module if use_ddp else model
        final_save_path = save_checkpoint(
            model_to_save, optimizer, None, num_epochs,
            loss_total_avg, checkpoint_dir, 'rhf_fno_gru_final.pt'
        )
        print(f"\n✅ 训练完成！最终模型: {final_save_path}")

        # 绘制 Loss 曲线
        try:
            import pandas as pd
            df = pd.read_csv(log_csv_path)

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Training Loss Analysis (Curriculum Learning)', fontsize=14, fontweight='bold')

            axes[0, 0].plot(df['epoch'], df['total_loss'], 'b-', linewidth=2)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Total Loss')
            axes[0, 0].set_title('Total Loss')
            axes[0, 0].grid(True, alpha=0.3)

            axes[0, 1].plot(df['epoch'], df['loss_data'], 'g-', label='Data Loss', linewidth=2)
            axes[0, 1].plot(df['epoch'], df['loss_pde'], 'r-', label='PDE Loss', linewidth=2)
            axes[0, 1].plot(df['epoch'], df['loss_norm'], 'm-', label='Norm Loss', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].set_title('Loss Components')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            axes[1, 0].plot(df['epoch'], df['learning_rate'], 'orange', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].grid(True, alpha=0.3)

            axes[1, 1].plot(df['epoch'], df['loss_pde'], 'purple', linewidth=2, label='PDE Loss')
            axes[1, 1].plot(df['epoch'], df['loss_norm'], 'green', linewidth=2, label='Norm Loss')
            # 标注课程学习阶段
            axes[1, 1].axvline(x=curriculum_phase1_epochs, color='r', linestyle='--', alpha=0.5, label='Phase1→2')
            axes[1, 1].axvline(x=curriculum_phase2_epochs, color='b', linestyle='--', alpha=0.5, label='Phase2→3')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('λ_physics')
            axes[1, 1].set_title('Physics Loss Weight (Sigmoid Ramp)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            loss_plot_path = os.path.join(log_dir, 'loss_curves.png')
            plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"📈 Loss 曲线: {loss_plot_path}")

        except ImportError:
            print("⚠️ pandas 未安装，跳过 Loss 分析")

    if use_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    train_model()
