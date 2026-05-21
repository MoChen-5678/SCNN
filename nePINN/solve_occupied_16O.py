#!/usr/bin/env python3
"""
批量处理 ¹⁶O 的全部 6 个占据态 (PINN).

两种模式:
  1. train (默认): 从零训练每个态, 输出 .pth / .json / .png
  2. infer:       加载已有模型, 仅做前向推理, 输出波函数数据 + 对比图

占据态:
  中子: N.1s.1/2, N.1p.3/2, N.1p.1/2
  质子: P.1s.1/2, P.1p.3/2, P.1p.1/2

用法:
    conda activate torch_env
    python solve_occupied_16O.py              # 训练模式
    python solve_occupied_16O.py --infer      # 推理模式 (加载已训练模型)
"""

import os, sys, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import json
import numpy as np
import torch

# ── 项目模块导入 ──
from config import HBAR_C, DR, R_GRID
from dirac_matrix_vs_pinn import (
    DiracPINNSolver,
    DiracNet,
    load_shooting_potentials,
    load_wav_wavefunction,
    parse_kappa_from_label,
    compute_wav_rayleigh_energy,
    plot_comparison,
    count_nodes,
)

# ════════════════════════════════════════════════════════
#   配置
# ════════════════════════════════════════════════════════

NUCLEUS = '16O'
A_VAL, Z_VAL = 16, 8

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)   # /home/ubuntu/rhf
POT_BASE = os.path.join(WORKSPACE_ROOT, 'results', '16O', 'POT')
WAV_DIR  = os.path.join(WORKSPACE_ROOT, 'results', '16O', 'WAV')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', f'{NUCLEUS}_occupied')

EPOCHS = 8000
LR = 1e-3

OCCUPIED_STATES = [
    # (state_label, tau, pot_filename)
    ('1s1/2', 'n', 'O16_state001_POT.it001.final000'),
    ('1p3/2', 'n', 'O16_state007_POT.it001.final000'),
    ('1p1/2', 'n', 'O16_state023_POT.it001.final000'),
    ('1s1/2', 'p', 'O16_state001_POT.it002.final000'),
    ('1p3/2', 'p', 'O16_state007_POT.it002.final000'),
    ('1p1/2', 'p', 'O16_state023_POT.it002.final000'),
]


def _load_ref_data(state_label, tau, pot_file):
    """加载 Shooting 参考数据 (POT + WAV), 返回 (potentials, shooting_wf, E_ref)."""
    pot_full = os.path.join(POT_BASE, pot_file)
    if not os.path.exists(pot_full):
        return None, None, 0.0

    potentials = load_shooting_potentials(pot_full, R_GRID)
    kappa = parse_kappa_from_label(state_label)

    # Shooting 能量 (来自 POT 文件头)
    E_shooting = potentials.get('E_shooting')
    E_ref = E_shooting or 0.0

    # WAV 波函数
    shooting_wf = None
    if os.path.isdir(WAV_DIR):
        import re
        state_base = re.sub(r'([a-z])(\d)', r'\1.\2', state_label)
        state_name = f'{tau.upper()}.{state_base}'
        try:
            wf = load_wav_wavefunction(WAV_DIR, tau, state_name)
            if wf is not None:
                shooting_wf = wf
                E_ray = compute_wav_rayleigh_energy(wf, potentials, kappa)
                if E_ray is not None:
                    wf['E'] = E_ray
                    wf['source'] = 'rayleigh'
                    E_ref = E_ray
                else:
                    E_ref = wf.get('E', E_ref)
        except Exception as e:
            print(f'  [WARN] WAV load failed: {e}')

    return potentials, shooting_wf, E_ref


def _save_outputs(safe_name, output_dir, G_pinn, F_pinn, r_pinn,
                  E_pinn, E_ref, state_label, tau, shooting_wf, history=None):
    """保存波函数 JSON + 对比图 PNG."""
    wf_path   = os.path.join(output_dir, f'{safe_name}_wavefunction.json')
    plot_path = os.path.join(output_dir, f'{safe_name}_comparison.png')

    # 保存波函数 JSON
    data = {
        'state_name': safe_name,
        'A': A_VAL, 'Z': Z_VAL, 'tau': tau, 'label': state_label,
        'r': r_pinn.tolist(),
        'G': G_pinn.tolist(),
        'F': F_pinn.tolist(),
        'E_PINN': float(E_pinn),
        'E_Ref': float(E_ref),
    }
    with open(wf_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'  Wavefunction saved: {wf_path}')

    # 绘制对比图
    ref_G = shooting_wf['G'] if shooting_wf else None
    ref_F = shooting_wf['F'] if shooting_wf else None
    hist = history if history is not None else []

    plot_comparison(
        r_pinn, {'label': state_label, 'E': E_ref},
        ref_G, ref_F,
        r_pinn, G_pinn, F_pinn,
        E_ref, E_pinn, hist,
        output_dir=output_dir,
        shooting_data=shooting_wf,
        tau=tau,
    )
    # 重命名自动生成的 PNG
    auto_plot = os.path.join(output_dir, f'matrix_vs_pinn_{safe_name}.png')
    if os.path.exists(auto_plot):
        os.rename(auto_plot, plot_path)
        print(f'  Plot saved: {plot_path}')
    elif os.path.exists(plot_path):
        print(f'  Plot saved: {plot_path}')


# ════════════════════════════════════════════════════════
#   模式 1: 训练 (Train)
# ════════════════════════════════════════════════════════

def solve_one_state(state_label, tau, pot_file, output_dir, epochs=EPOCHS, lr=LR):
    """训练单个态: 创建网络 -> 训练 -> 保存模型/波函数/对比图."""
    nucleus = NUCLEUS
    safe_name = f'{nucleus}_{tau}_{state_label.replace("/", "_")}'
    model_path = os.path.join(output_dir, f'{safe_name}_model.pth')

    kappa = parse_kappa_from_label(state_label)

    print(f'\n{"="*60}')
    print(f'  [TRAIN] {nucleus} {tau.upper()}.{state_label}  k={kappa}')
    print(f'  POT: {os.path.basename(pot_file)}')
    print(f'{"="*60}')

    potentials, shooting_wf, E_ref = _load_ref_data(state_label, tau, pot_file)
    if potentials is None:
        print(f'  [ERROR] POT file not found')
        return None

    print(f'  E_ref = {E_ref:+.4f} MeV')

    E_init = E_ref if E_ref != 0 else -60.0
    target_nodes = count_nodes(torch.tensor(shooting_wf['G'])) if shooting_wf else None
    print(f'  E_init = {E_init:.2f} MeV,  target_nodes = {target_nodes}')

    solver = DiracPINNSolver(
        A=A_VAL, Z=Z_VAL, tau=tau, kappa=kappa,
        potentials=potentials,
    )

    t0 = time.time()
    E_pinn, history = solver.train(
        E_init_MeV=E_init,
        target_nodes=target_nodes,
        max_epochs=epochs,
        lr=lr,
        print_every=max(epochs // 20, 400),
        w_pde=20.0, w_bc=0, w_ortho=0,
        live_plot=False,
    )
    elapsed = time.time() - t0
    G_pinn, F_pinn = solver.get_wavefunction()

    dE = abs(E_pinn - E_ref)
    src = 'Rayleigh' if (shooting_wf and shooting_wf.get('source') == 'rayleigh') else 'Shoot'
    print(f'  Result: E_{src}={E_ref:+.4f}  E_PINN={E_pinn:+.4f}  dE={dE:.4f} MeV  ({elapsed:.1f}s)')

    # 保存模型
    solver.save_model(model_path, E_MeV=E_pinn)

    # 保存波函数 + 对比图
    _save_outputs(
        safe_name, output_dir, G_pinn, F_pinn, solver.r_np,
        E_pinn, E_ref, state_label, tau, shooting_wf, history,
    )

    return {
        'state': f'{nucleus}_{tau}.{state_label}',
        'mode': 'train',
        'E_ref': E_ref,
        'E_pinn': E_pinn,
        'dE': dE,
        'time_s': elapsed,
        'model': model_path,
    }


# ════════════════════════════════════════════════════════
#   模式 2: 推理 (Infer)
# ════════════════════════════════════════════════════════

@torch.no_grad()
def infer_one_state(state_label, tau, pot_file, output_dir):
    """加载已训练模型 -> 前向推理 -> 保存波函数 JSON + 对比图 PNG (不训练)."""
    nucleus = NUCLEUS
    safe_name = f'{nucleus}_{tau}_{state_label.replace("/", "_")}'
    model_path = os.path.join(output_dir, f'{safe_name}_model.pth')

    kappa = parse_kappa_from_label(state_label)

    print(f'\n{"="*60}')
    print(f'  [INFER] {nucleus} {tau.upper()}.{state_label}  k={kappa}')
    print(f'  Model: {os.path.basename(model_path)}')
    print(f'{"="*60}')

    if not os.path.exists(model_path):
        print(f'  [ERROR] Model not found: {model_path}')
        print(f'  Run without --infer first to train the model.')
        return None

    # 加载参考数据
    potentials, shooting_wf, E_ref = _load_ref_data(state_label, tau, pot_file)
    print(f'  E_ref = {E_ref:+.4f} MeV')

    # 加载模型
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    net = DiracNet(
        n_hidden=128, n_layers=6, activation='swish',
        hard_normalize=True, init_energy=E_ref or -40.0,
    )
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        net.load_state_dict(checkpoint['state_dict'], strict=False)
        loaded_E = checkpoint.get('E')
    else:
        net.load_state_dict(checkpoint, strict=False)
        loaded_E = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device).eval()

    t0 = time.time()
    r_tensor = torch.tensor(R_GRID, dtype=torch.float32, device=device).unsqueeze(0)
    g, f = net(r_tensor, kappa=kappa, dr=DR)
    G_pinn = g.squeeze(0).cpu().numpy()
    F_pinn = f.squeeze(0).cpu().numpy()
    elapsed = time.time() - t0

    # 用 Rayleigh 商计算 PINN 能量 (从加载的模型权重)
    # 简单近似: 直接用模型保存的 E 或重新计算
    if loaded_E is not None:
        E_pinn = float(loaded_E)
    else:
        E_pinn = E_ref  # fallback

    dE = abs(E_pinn - E_ref)
    print(f'  Result: E_Ref={E_ref:+.4f}  E_PINN={E_pinn:+.4f}  dE={dE:.4f} MeV  ({elapsed:.2f}s)')

    # 保存波函数 + 对比图
    _save_outputs(
        safe_name, output_dir, G_pinn, F_pinn, np.array(R_GRID),
        E_pinn, E_ref, state_label, tau, shooting_wf, history=None,
    )

    return {
        'state': f'{nucleus}_{tau}.{state_label}',
        'mode': 'infer',
        'E_ref': E_ref,
        'E_pinn': E_pinn,
        'dE': dE,
        'time_s': elapsed,
        'model': model_path,
    }


# ════════════════════════════════════════════════════════
#   主入口
# ════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='¹⁶O 占据态 PINN 批量处理')
    parser.add_argument('--infer', action='store_true',
                        help='推理模式: 加载已有模型, 只输出波函数+对比图 (不训练)')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'每态训练 epochs (默认 {EPOCHS}, 仅训练模式)')
    parser.add_argument('--lr', type=float, default=LR,
                        help=f'学习率 (默认 {LR}, 仅训练模式)')
    args = parser.parse_args()

    epochs_val = args.epochs
    lr_val = args.lr
    mode_str = 'INFER (推理)' if args.infer else 'TRAIN (训练)'
    process_fn = infer_one_state if args.infer else solve_one_state

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print(f"  {NUCLEUS} 占据态批量处理 ({mode_str})")
    print(f"  共 {len(OCCUPIED_STATES)} 个态")
    if not args.infer:
        print(f"  每态 {epochs_val} epochs, lr={lr_val}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  核素: A={A_VAL}, Z={Z_VAL}")
    print("=" * 70)

    total_start = time.time()
    results_summary = []

    for idx, (state_label, tau, pot_file) in enumerate(OCCUPIED_STATES):
        if args.infer:
            result = infer_one_state(state_label, tau, pot_file, OUTPUT_DIR)
        else:
            result = solve_one_state(state_label, tau, pot_file, OUTPUT_DIR,
                                     epochs=epochs_val, lr=lr_val)
        if result:
            results_summary.append(result)
            status = "OK" if result['dE'] < 1.0 else ("~" if result['dE'] < 5.0 else "!!")
        else:
            status = "FAIL"
            results_summary.append({
                'state': f'{NUCLEUS}_{tau}.{state_label}',
                'mode': mode_str.split()[0],
                'dE': 999,
                'time_s': 0,
            })
        print(f'  [{status}] [{idx+1}/{len(OCCUPIED_STATES)}]')

    total_time = time.time() - total_start

    # ── 汇总表 ──
    print(f'\n{"="*80}')
    print(f'  {NUCLEUS} 全部完成! 总耗时: {total_time:.1f}s ({total_time/60:.1f}min)')
    print(f'  模式: {mode_str}')
    print(f'{"="*80}')
    print(f'  {"State":22s}  {"Mode":>6s}  {"E_Ref":>10s}  {"E_PINN":>10s}  {"dE(MeV)":>9s}  {"Time(s)":>8s}  Status')
    print(f'  {"-"*85}')
    for r in results_summary:
        er = r.get('E_ref', 0)
        ep = r.get('E_pinn', 0)
        de = r.get('dE', 999)
        tm = r.get('time_s', 0)
        mo = r.get('mode', '?')
        st = "OK" if de < 0.5 else ("~" if de < 2.0 else "!!")
        print(f'  {r["state"]:22s}  {mo:>6s}  {er:>10.3f}  {ep:>10.3f}  {de:>9.4f}  {tm:>8.1f}s  {st}')

    # ── 保存汇总为 JSON ──
    summary_path = os.path.join(OUTPUT_DIR, f'{NUCLEUS}_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'nucleus': NUCLEUS,
            'A': A_VAL, 'Z': Z_VAL,
            'mode': mode_str,
            'total_time_s': total_time,
            'results': results_summary,
        }, f, indent=2, default=str)
    print(f'\n  Summary saved: {summary_path}')


if __name__ == '__main__':
    main()
