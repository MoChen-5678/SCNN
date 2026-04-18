#!/usr/bin/env python3
"""
evaluate_final.py — 对比网络输出与真实收敛态（FINAL数据）

功能：
  1. 加载 RHF 计算的 FINAL 文件（真实收敛波函数 + 能量本征值）
  2. 加载训练好的 RHF_FNO_GRU 模型
  3. 对每个核素/粒子类型/量子态进行逐一对比
  4. 输出关键指标：能量误差、波函数MSE、归一化积分、峰值位置偏差
  5. 生成对比图和CSV报告

数据格式（FINAL文件）：
  头部注释行含:
    # Eigenvalues (MeV): E1 E2 ... E42
    # States: state1 state2 ... state42
  数据行: r  g1 g2 ... g42  (201行，dr=0.10fm)
  it001=中子(N), it002=质子(P)
"""

import os
import re
import sys
import csv
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Model_Architecture import RHF_FNO_GRU
from Data_Loader import get_zn, ISOTOPE_ZN, _fix_fortran_float


# ══════════════════════════════════════════════════════════════
#   FINAL 文件解析器
# ══════════════════════════════════════════════════════════════

# 42个态的标准顺序（与Data_Loader和Train.py一致）
ALL_42_STATES = [
    '1s1/2', '2s1/2', '3s1/2', '4s1/2', '5s1/2', '6s1/2',
    '1p3/2', '2p3/2', '3p3/2', '4p3/2', '5p3/2', '6p3/2',
    '1d5/2', '2d5/2', '3d5/2', '4d5/2', '5d5/2',
    '1f7/2', '2f7/2', '3f7/2', '4f7/2', '5f7/2',
    '1p1/2', '2p1/2', '3p1/2', '4p1/2', '5p1/2', '6p1/2',
    '1d3/2', '2d3/2', '3d3/2', '4d3/2', '5d3/2',
    '1f5/2', '2f5/2', '3f5/2', '4f5/2', '5f5/2',
    '1g7/2', '2g7/2', '3g7/2', '4g7/2',
]


def parse_state_label(label_str):
    """解析态标签 'N.1s.1/2' → (is_proton, n, l, j, kappa, state_name)"""
    m = re.match(r'([NP])\.(\d+)([a-z])\.(\d+)/2', label_str.strip())
    if not m:
        return None
    particle, n_str, l_char, j_half = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
    is_proton = 1.0 if particle == 'P' else 0.0

    l_map = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5, 'i': 6}
    l_val = l_map.get(l_char, 0)
    j_val = j_half / 2.0

    # kappa: j=l+1/2 -> kappa=-(l+1); j=l-1/2 -> kappa=l
    kappa = -(l_val + 1) if j_val > l_val else l_val

    state_name = f"{n_str}{l_char}{j_half}/2"
    return (is_proton, n_str, l_val, j_val, kappa, state_name)


def load_final_file(final_path):
    """
    加载 FINAL 文件，返回:
      states_info: [(is_proton, n, l, j, kappa, state_name, E_true), ...]
      r_grid: (npt,) 径向网格
      wavefunctions: (42, npt) 归一化波函数（g分量）
    """
    with open(final_path, 'r') as f:
        lines = f.readlines()

    # 解析能量本征值
    energies = None
    state_labels = None
    data_lines = []

    for line in lines:
        line_stripped = line.strip()
        if line_stripped.startswith('# Eigenvalues'):
            m = re.search(r'Eigenvalues \(MeV\):\s+(.*)', line_stripped)
            if m:
                energies = [float(x) for x in m.group(1).split()]
        elif line_stripped.startswith('# States:'):
            m = re.search(r'States:\s+(.*)', line_stripped)
            if m:
                state_labels = m.group(1).split()
        elif not line_stripped.startswith('#') and line_stripped:
            data_lines.append(line_stripped)

    if energies is None or state_labels is None:
        raise ValueError(f"无法从 {final_path} 解析头部信息")

    # 解析数据
    data = []
    for dl in data_lines:
        vals = dl.split()
        r = float(vals[0])
        g_vals = [float(v) for v in vals[1:]]
        data.append([r] + g_vals)

    data = np.array(data)  # (npt, 43) — r + 42个g
    r_grid = data[:, 0]
    wavefunctions = data[:, 1:].T  # (42, npt)
    npt = wavefunctions.shape[1]

    # 构建态信息
    states_info = []
    for i, label in enumerate(state_labels):
        info = parse_state_label(label)
        if info is not None:
            is_proton, n, l, j, kappa, state_name = info
            E_true = energies[i] if i < len(energies) else 0.0
            # 归一化波函数
            g_raw = wavefunctions[i]
            norm_int = np.trapz(g_raw**2, x=r_grid)
            if norm_int > 1e-12:
                g_norm = g_raw / np.sqrt(norm_int)
            else:
                g_norm = g_raw
            wavefunctions[i] = g_norm
            states_info.append((is_proton, n, l, j, kappa, state_name, E_true))

    return states_info, r_grid, wavefunctions


# ══════════════════════════════════════════════════════════════
#   网络推理
# ══════════════════════════════════════════════════════════════

def run_model_for_state(model, x_seq, kappa, r_grid, is_proton, z_num, n_num, n_principal,
                        stats_mean, stats_std, device, dr=0.10):
    """
    对单个态运行模型推理，返回预测的 (g, f, E_pred) 和完整 y_pred。
    """
    model.eval()
    B = x_seq.size(0)
    batch_r_grid = r_grid.unsqueeze(0).expand(B, -1) if x_seq.dim() == 4 else r_grid.unsqueeze(0)

    with torch.no_grad():
        # 归一化输入
        C = len(stats_mean)
        mean_view = stats_mean.view(1, 1, C, 1).to(device)
        std_view = stats_std.view(1, 1, C, 1).to(device)
        x_norm = x_seq.clone()
        x_norm[:, :, :11, :] = (x_seq[:, :, :11, :] - mean_view) / std_view

        y_pred = model(x_norm, kappa, batch_r_grid,
                       is_proton=is_proton, z_num=z_num, n_num=n_num,
                       n_principal=n_principal)

    g_pred = y_pred[0, 0, :].cpu().numpy()  # (N,)
    f_pred = y_pred[0, 1, :].cpu().numpy()  # (N,)
    E_pred = y_pred[0, 9, :].mean().item()   # 标量能量

    return g_pred, f_pred, E_pred, y_pred


# ══════════════════════════════════════════════════════════════
#   评估指标计算
# ══════════════════════════════════════════════════════════════

def compute_metrics(g_pred, f_pred, E_pred, g_true, f_true, E_true, r_grid, dr=0.10):
    """计算所有评估指标"""
    npt = len(r_grid)

    # 1. 波函数MSE
    mse_g = np.mean((g_pred - g_true) ** 2)
    mse_f = np.mean((f_pred - f_true) ** 2)

    # 2. 归一化积分
    norm_pred = np.trapz(g_pred**2 + f_pred**2, x=r_grid)
    norm_true = np.trapz(g_true**2 + f_true**2, x=r_grid)

    # 3. 峰值位置
    peak_pred = r_grid[np.argmax(np.abs(g_pred))]
    peak_true = r_grid[np.argmax(np.abs(g_true))]
    peak_error = abs(peak_pred - peak_true)

    # 4. 能量误差
    energy_error = E_pred - E_true

    # 5. 相对误差
    rel_energy = abs(energy_error) / max(abs(E_true), 1e-6) * 100  # 百分比
    rel_g_mse = mse_g / max(np.mean(g_true**2), 1e-12) * 100

    # 6. 波函数相关系数
    corr_g = np.corrcoef(g_pred, g_true)[0, 1] if np.std(g_pred) > 1e-10 and np.std(g_true) > 1e-10 else 0.0

    return {
        'mse_g': mse_g,
        'mse_f': mse_f,
        'norm_pred': norm_pred,
        'norm_true': norm_true,
        'peak_pred': peak_pred,
        'peak_true': peak_true,
        'peak_error_fm': peak_error,
        'E_pred': E_pred,
        'E_true': E_true,
        'E_error': energy_error,
        'E_rel_error_pct': rel_energy,
        'g_rel_mse_pct': rel_g_mse,
        'g_correlation': corr_g,
    }


# ══════════════════════════════════════════════════════════════
#   绘图
# ══════════════════════════════════════════════════════════════

def plot_comparison(g_pred, f_pred, E_pred, g_true, f_true, E_true,
                    r_grid, state_name, isotope, particle, save_dir):
    """绘制单态对比图"""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    particle_label = "Proton" if particle == 'P' else "Neutron"
    fig.suptitle(f'{isotope} | {particle_label} {state_name} | E_true={E_true:.2f} MeV | E_pred={E_pred:.2f} MeV',
                 fontsize=13, fontweight='bold')

    r = r_grid

    # g(r) 对比
    axes[0, 0].plot(r, g_true, 'b-', linewidth=2, label='True g(r)')
    axes[0, 0].plot(r, g_pred, 'r--', linewidth=2, label='Pred g(r)')
    axes[0, 0].set_xlabel('r (fm)')
    axes[0, 0].set_ylabel('g(r)')
    axes[0, 0].set_title('Large Component g(r)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # f(r) 对比
    axes[0, 1].plot(r, f_true, 'b-', linewidth=2, label='True f(r)')
    axes[0, 1].plot(r, f_pred, 'r--', linewidth=2, label='Pred f(r)')
    axes[0, 1].set_xlabel('r (fm)')
    axes[0, 1].set_ylabel('f(r)')
    axes[0, 1].set_title('Small Component f(r)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 概率密度
    prob_true = g_true**2 + f_true**2
    prob_pred = g_pred**2 + f_pred**2
    axes[1, 0].plot(r, prob_true, 'b-', linewidth=2, label=r'True $|g|^2+|f|^2$')
    axes[1, 0].plot(r, prob_pred, 'r--', linewidth=2, label=r'Pred $|g|^2+|f|^2$')
    axes[1, 0].set_xlabel('r (fm)')
    axes[1, 0].set_ylabel(r'$\rho(r)$')
    axes[1, 0].set_title('Radial Probability Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 误差图
    err_g = np.abs(g_pred - g_true)
    err_f = np.abs(f_pred - f_true)
    axes[1, 1].semilogy(r, np.maximum(err_g, 1e-15), 'g-', linewidth=1.5, label='|g_pred - g_true|')
    axes[1, 1].semilogy(r, np.maximum(err_f, 1e-15), 'm-', linewidth=1.5, label='|f_pred - f_true|')
    axes[1, 1].set_xlabel('r (fm)')
    axes[1, 1].set_ylabel('Absolute Error')
    axes[1, 1].set_title('Pointwise Error (log)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    safe_name = state_name.replace('/', '_')
    save_path = os.path.join(save_dir, f'{isotope}_{particle}_{safe_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def plot_energy_comparison(all_results, save_dir):
    """绘制所有态的能量对比汇总图"""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 提取数据
    E_trues = [r['E_true'] for r in all_results]
    E_preds = [r['E_pred'] for r in all_results]
    labels = [f"{r['isotope']}_{r['particle']}_{r['state']}" for r in all_results]

    # 左图：E_pred vs E_true 散点图
    axes[0].scatter(E_trues, E_preds, alpha=0.7, s=40, c='steelblue', edgecolors='navy', linewidth=0.5)
    e_min = min(min(E_trues), min(E_preds)) - 5
    e_max = max(max(E_trues), max(E_preds)) + 5
    axes[0].plot([e_min, e_max], [e_min, e_max], 'r--', linewidth=1.5, label='Perfect')
    axes[0].set_xlabel('E_true (MeV)')
    axes[0].set_ylabel('E_pred (MeV)')
    axes[0].set_title('Energy: Predicted vs True')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 右图：能量误差柱状图
    E_errors = [r['E_error'] for r in all_results]
    colors = ['red' if abs(e) > 10 else 'orange' if abs(e) > 5 else 'green' for e in E_errors]
    y_pos = np.arange(len(E_errors))
    axes[1].barh(y_pos, E_errors, color=colors, alpha=0.8)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(labels, fontsize=6)
    axes[1].set_xlabel('Energy Error (MeV)')
    axes[1].set_title('Energy Error by State')
    axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'energy_comparison_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


# ══════════════════════════════════════════════════════════════
#   主评估流程
# ══════════════════════════════════════════════════════════════

def evaluate_model(checkpoint_path, data_dir, isotopes, target_states,
                   stats_mean_path=None, device_str='cuda:0'):
    """
    主评估入口：加载模型和数据，逐态对比。

    参数:
      checkpoint_path: 模型检查点路径
      data_dir: RHF结果数据目录 (含FINAL子目录)
      isotopes: 评估的核素列表, e.g. ['16O', '40Ca']
      target_states: 评估的态列表, e.g. ['1s1/2', '1p3/2', ...]
      stats_mean_path: 训练时保存的统计量路径(可选)
      device_str: 计算设备
    """
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print(f"🔍 评估设备: {device}")

    # 1. 加载模型
    print(f"\n📦 加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 模型超参数（需与训练时一致）
    hidden_dim = 96
    gru_hidden = 1536
    modes = 40

    model = RHF_FNO_GRU(in_channels=12, hidden_dim=hidden_dim, npt=201,
                        gru_hidden=gru_hidden, modes=modes).to(device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"   模型加载完成 (epoch={checkpoint.get('epoch', '?')})")

    # 2. 构造默认统计量（如果未提供）
    # 训练时的统计量来自Data_Loader计算，这里用零均值/单位标准差作为fallback
    stats_mean = torch.zeros(11)
    stats_std = torch.ones(11)

    # 尝试从训练日志目录加载统计量
    if stats_mean_path and os.path.exists(stats_mean_path):
        stats_data = torch.load(stats_mean_path, map_location='cpu')
        stats_mean = stats_data.get('mean', stats_mean)
        stats_std = stats_data.get('std', stats_std)
        print(f"   统计量已加载: mean={stats_mean[:3].numpy()}, std={stats_std[:3].numpy()}")

    dr = 0.10
    base_r_grid = torch.arange(0, 201, device=device, dtype=torch.float32) * dr
    base_r_grid[0] = 0.0010

    # 3. 遍历核素和粒子类型
    all_results = []
    npt = 201

    for isotope in isotopes:
        Z, N = get_zn(isotope)
        z_num_t = torch.tensor([float(Z)], device=device)
        n_num_t = torch.tensor([float(N)], device=device)

        final_dir = os.path.join(data_dir, isotope, 'FINAL')
        if not os.path.exists(final_dir):
            print(f"  ⚠️ {isotope}: FINAL目录不存在，跳过")
            continue

        # 中子 (it001) 和 质子 (it002)
        particle_files = {
            'N': None, 'P': None
        }
        for f in os.listdir(final_dir):
            if f.endswith('.final000'):
                if '.it001.' in f:
                    particle_files['N'] = os.path.join(final_dir, f)
                elif '.it002.' in f:
                    particle_files['P'] = os.path.join(final_dir, f)

        for particle, final_path in particle_files.items():
            if final_path is None:
                continue

            print(f"\n{'='*60}")
            print(f"  📊 {isotope} | {particle} | {final_path}")
            print(f"{'='*60}")

            is_proton_t = torch.tensor([1.0 if particle == 'P' else 0.0], device=device)

            # 加载FINAL文件
            try:
                states_info, r_grid_np, wavefunctions = load_final_file(final_path)
            except Exception as e:
                print(f"  ⚠️ 解析失败: {e}")
                continue

            # 逐态评估
            for idx, (is_proton_val, n, l, j, kappa, state_name, E_true) in enumerate(states_info):
                # 检查是否在目标态列表中
                if state_name not in target_states:
                    continue

                kappa_t = torch.tensor([float(kappa)], device=device)
                n_principal_t = torch.tensor([float(int(n))], device=device)

                # ★ 构造模型输入 X: (1, max_seq_len, 12, npt)
                # 使用最终收敛态作为输入序列（单步）
                g_true = wavefunctions[idx]  # (npt,) 归一化后的g

                # 从FINAL文件无法直接获取完整11通道数据
                # 这里用WAV/POT目录的最终loop数据来构造
                wav_dir = os.path.join(data_dir, isotope, 'WAV')
                pot_dir = os.path.join(data_dir, isotope, 'POT')

                # 尝试找到对应的state文件
                # 命名规则: {prefix}_state{k}_{particle_letter}.it{it_num}.loop{loop_num}
                prefix_map = {'16O': 'O16_', '40Ca': 'Ca40_'}
                iso_prefix = prefix_map.get(isotope, isotope)

                # 查找匹配的WAV文件
                best_g_pred = None
                best_f_pred = None
                best_E_pred = None

                if os.path.exists(wav_dir):
                    # 找该态的最后一个loop文件
                    all_wav = sorted([f for f in os.listdir(wav_dir) if '.loop' in f])
                    # 匹配态标签
                    state_wav_files = []
                    for wf in all_wav:
                        # 读取文件头检查是否匹配当前态
                        wp = os.path.join(wav_dir, wf)
                        try:
                            with open(wp, 'r') as fh:
                                header = fh.read(2000)
                            m = re.search(r'State:\s*([NP])\.(\d+)([a-z])\.(\d+)/2', header)
                            if m:
                                p_label = m.group(1)
                                n_label = int(m.group(2))
                                l_label = m.group(3)
                                j_label = int(m.group(4))
                                if (p_label == particle and
                                    n_label == int(n) and
                                    l_label == l and
                                    j_label == j * 2):
                                    state_wav_files.append(wf)
                        except Exception:
                            continue

                    if state_wav_files:
                        # 取最后一个loop（最收敛的）
                        from Data_Loader import _extract_it_loop, _parse_single_step
                        state_wav_files.sort(key=lambda x: _extract_it_loop(x))
                        last_wav = state_wav_files[-1]
                        last_pot = last_wav.replace('.it', '_POT.it')

                        wav_path = os.path.join(wav_dir, last_wav)
                        pot_path = os.path.join(pot_dir, last_pot)

                        if os.path.exists(pot_path):
                            try:
                                res = _parse_single_step(wav_path, pot_path)
                                if res is not None:
                                    y_tensor, _, _ = res  # (11, npt)

                                    # 构造输入X: (1, 1, 12, npt)
                                    progress = torch.ones(1, 1, npt)  # progress=1.0 (最终步)
                                    X_11ch = y_tensor.unsqueeze(0)  # (1, 11, npt)
                                    X_12ch = torch.cat([X_11ch, progress.unsqueeze(1)], dim=1)  # (1, 12, npt)
                                    X_seq = X_12ch.unsqueeze(0)  # (1, 1, 12, npt) — seq_len=1

                                    X_seq = X_seq.to(device)

                                    g_pred, f_pred, E_pred, y_pred = run_model_for_state(
                                        model, X_seq, kappa_t, base_r_grid,
                                        is_proton_t, z_num_t, n_num_t, n_principal_t,
                                        stats_mean, stats_std, device, dr
                                    )

                                    best_g_pred = g_pred
                                    best_f_pred = f_pred
                                    best_E_pred = E_pred
                            except Exception as e:
                                print(f"    ⚠️ 态 {state_name} 数据解析失败: {e}")

                # 如果无法从WAV/POT获取模型输入，跳过该态
                if best_g_pred is None:
                    print(f"    ⏭️ 态 {state_name}: 无可用WAV/POT数据，跳过")
                    continue

                # f_true: FINAL文件中仅含g，需从WAV获取f
                # 简化处理：直接用_parse_single_step的f
                f_true = y_tensor[1, :].numpy() if 'y_tensor' in dir() else np.zeros(npt)
                g_true_full = y_tensor[0, :].numpy() if 'y_tensor' in dir() else g_true

                # 计算指标
                metrics = compute_metrics(
                    best_g_pred, best_f_pred, best_E_pred,
                    g_true_full, f_true, E_true,
                    r_grid_np, dr
                )
                metrics['isotope'] = isotope
                metrics['particle'] = particle
                metrics['state'] = state_name
                metrics['n'] = int(n)
                metrics['kappa'] = kappa

                all_results.append(metrics)

                # 打印结果
                status = "✅" if abs(metrics['E_error']) < 5 else "⚠️" if abs(metrics['E_error']) < 15 else "❌"
                print(f"    {status} {state_name:8s} | E_true={E_true:+8.2f} | E_pred={metrics['E_pred']:+8.2f} | "
                      f"ΔE={metrics['E_error']:+7.2f} MeV ({metrics['E_rel_error_pct']:+5.1f}%) | "
                      f"peak_err={metrics['peak_error_fm']:.2f}fm | norm={metrics['norm_pred']:.4f} | "
                      f"corr_g={metrics['g_correlation']:.4f}")

                # 绘图
                plot_dir = os.path.join(os.path.dirname(checkpoint_path), '..', 'eval_plots')
                plot_comparison(
                    best_g_pred, best_f_pred, best_E_pred,
                    g_true_full, f_true, E_true,
                    r_grid_np, state_name, isotope, particle, plot_dir
                )

    # 4. 汇总报告
    if not all_results:
        print("\n⚠️ 无有效评估结果！")
        return

    # 能量对比汇总图
    plot_dir = os.path.join(os.path.dirname(checkpoint_path), '..', 'eval_plots')
    plot_energy_comparison(all_results, plot_dir)

    # CSV报告
    csv_path = os.path.join(os.path.dirname(checkpoint_path), 'evaluation_report.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['isotope', 'particle', 'state', 'n', 'kappa',
                        'E_true', 'E_pred', 'E_error', 'E_rel_error_pct',
                        'mse_g', 'mse_f', 'norm_pred', 'norm_true',
                        'peak_pred', 'peak_true', 'peak_error_fm',
                        'g_rel_mse_pct', 'g_correlation'])
        for r in all_results:
            writer.writerow([
                r['isotope'], r['particle'], r['state'], r['n'], r['kappa'],
                f"{r['E_true']:.4f}", f"{r['E_pred']:.4f}",
                f"{r['E_error']:.4f}", f"{r['E_rel_error_pct']:.2f}",
                f"{r['mse_g']:.6e}", f"{r['mse_f']:.6e}",
                f"{r['norm_pred']:.6f}", f"{r['norm_true']:.6f}",
                f"{r['peak_pred']:.3f}", f"{r['peak_true']:.3f}",
                f"{r['peak_error_fm']:.3f}",
                f"{r['g_rel_mse_pct']:.2f}", f"{r['g_correlation']:.4f}",
            ])
    print(f"\n📄 CSV报告: {csv_path}")

    # 汇总统计
    E_errors = [abs(r['E_error']) for r in all_results]
    E_rel_errors = [abs(r['E_rel_error_pct']) for r in all_results]
    peak_errors = [r['peak_error_fm'] for r in all_results]
    corr_gs = [r['g_correlation'] for r in all_results]

    print(f"\n{'='*60}")
    print(f"  📊 评估汇总 ({len(all_results)} 个态)")
    print(f"{'='*60}")
    print(f"  能量绝对误差: mean={np.mean(E_errors):.2f} MeV, median={np.median(E_errors):.2f} MeV, max={np.max(E_errors):.2f} MeV")
    print(f"  能量相对误差: mean={np.mean(E_rel_errors):.1f}%, median={np.median(E_rel_errors):.1f}%")
    print(f"  峰值位置偏差: mean={np.mean(peak_errors):.2f} fm, median={np.median(peak_errors):.2f} fm")
    print(f"  g相关系数:    mean={np.mean(corr_gs):.4f}, median={np.median(corr_gs):.4f}")

    # 按粒子类型分组统计
    for p in ['N', 'P']:
        p_results = [r for r in all_results if r['particle'] == p]
        if p_results:
            p_ee = [abs(r['E_error']) for r in p_results]
            print(f"  {p}粒子 ({len(p_results)}态): E_error mean={np.mean(p_ee):.2f} MeV")


# ══════════════════════════════════════════════════════════════
#   命令行入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='评估RHF神经网络模型 vs FINAL真实数据')
    parser.add_argument('--checkpoint', type=str,
                       default='/home/ubuntu/rhf/SCNN/checkpoints/rhf_fno_gru_best.pt',
                       help='模型检查点路径')
    parser.add_argument('--data-dir', type=str, default='/home/ubuntu/rhf/results',
                       help='RHF结果数据目录')
    parser.add_argument('--isotopes', type=str, nargs='+', default=['16O', '40Ca'],
                       help='评估的核素列表')
    parser.add_argument('--states', type=str, nargs='+', default=None,
                       help='评估的态列表 (默认=Phase1核心态)')
    parser.add_argument('--device', type=str, default='cuda:0', help='计算设备')

    args = parser.parse_args()

    # 默认评估Phase1核心态
    if args.states is None:
        args.states = [
            '1s1/2',
            '1p3/2', '1p1/2',
            '1d5/2', '1d3/2',
            '1f7/2', '1f5/2',
        ]

    evaluate_model(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        isotopes=args.isotopes,
        target_states=args.states,
        device_str=args.device,
    )
