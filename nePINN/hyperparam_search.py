"""
超参数并行搜索: 找到2s1/2激发态的最佳loss权重组合.
使用 ProcessPoolExecutor 并行, 最多15个worker.

用法:
    python hyperparam_search.py
    python hyperparam_search.py --quick --epochs 2000   # 快速模式
    python hyperparam_search.py --workers 15 --epochs 5000

输出:
    - 控制台实时打印每组结果 (按完成顺序)
    - best_params.json (最佳参数)
    - search_results.json (完整结果)
"""

import os
import sys
import json
import itertools
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, '/home/ubuntu/rhf/plusPINN')


# ══════════════════════════════════════════════════════════
#   固定配置
# ══════════════════════════════════════════════════════════
STATE = "2s1/2"
TAU = "p"
POT_FILE = "/home/ubuntu/rhf/results/16O/POT/O16_state001_POT.it002.final000"
WAV_DIR = "/home/ubuntu/rhf/results/16O/WAV"
BASE_MODEL = "/home/ubuntu/rhf/outputs/1s1_2_model.pth"
BASE_WF = "/home/ubuntu/rhf/outputs/1s1_2_wavefunction.json"
OUTPUT_DIR = "/home/ubuntu/rhf/outputs/hyperparam_search"
TARGET_E_RAYLEIGH = -2.5627


# ══════════════════════════════════════════════════════════
#   单次运行函数 (必须在顶层, 供多进程pickle)
# ══════════════════════════════════════════════════════════
def run_single_config(cfg):
    """cfg: dict {w_pde, w_bc, w_ortho, lr, E_init, epochs} → result dict"""
    try:
        import torch
        import numpy as np
        from dirac_matrix_vs_pinn import (
            DiracPINNSolver,
            load_wav_wavefunction,
            load_shooting_potentials,
            load_ref_wavefunctions,
            parse_kappa_from_label,
            compute_energy_rayleigh,
        )
        from config import DR, NPT, R_GRID
        import re

        w_pde, w_bc, w_ortho, lr, E_init, epochs = (
            cfg['w_pde'], cfg['w_bc'], cfg['w_ortho'],
            cfg['lr'], cfg['E_init'], cfg['epochs']
        )

        kappa = parse_kappa_from_label(STATE)

        # 加载参考波函数 (正交约束用)
        ref_wfs = []
        if os.path.exists(BASE_WF):
            ref_wfs = load_ref_wavefunctions(BASE_WF, device='cpu')

        # 势场
        potentials = load_shooting_potentials(POT_FILE, R_GRID)

        # WAV参考
        shooting_wf = None
        if WAV_DIR:
            tau_prefix = 'P' if TAU == 'p' else 'N'
            state_base = re.sub(r'([a-z])(\d)', r'\1.\2', STATE)
            state_name = f'{tau_prefix}.{state_base}'
            shooting_wf = load_wav_wavefunction(WAV_DIR, TAU, state_name)

        solver = DiracPINNSolver(
            A=16, Z=8, tau=TAU, kappa=kappa,
            potentials=potentials,
            ref_wavefunctions=ref_wfs,
            lambda_ortho=w_ortho if ref_wfs else 0.0,
        )

        E_pin, history = solver.train(
            E_init_MeV=E_init,
            target_nodes=1,
            max_epochs=epochs,
            lr=lr,
            print_every=max(epochs, 999999),
            patience=200,
            load_model=BASE_MODEL if os.path.exists(BASE_MODEL) else None,
            w_pde=w_pde, w_bc=w_bc, w_ortho=w_ortho,
        )

        G_pin, F_pin = solver.get_wavefunction()

        status = 'ok'
        if E_pin < -100:
            status = 'neg_sea'
        elif abs(E_pin) > 500:
            status = 'diverged'

        dE = abs(E_pin - TARGET_E_RAYLEIGH)

        rms_g = float('nan')
        if shooting_wf is not None:
            r_wav = shooting_wf['r']
            G_wav = shooting_wf['G']
            G_pin_np = G_pin.squeeze().cpu().numpy()
            r_pin = np.arange(len(G_pin_np)) * DR
            G_wav_interp = np.interp(r_pin, r_wav, G_wav)
            overlap = np.trapezoid(G_wav_interp * G_pin_np, r_pin)
            if overlap < 0:
                G_pin_np = -G_pin_np
            rms_g = float(np.sqrt(np.mean((G_pin_np - G_wav_interp)**2)))

        return {
            'w_pde': w_pde, 'w_bc': w_bc, 'w_ortho': w_ortho,
            'lr': lr, 'E_init': E_init,
            'dE': dE, 'RMS_G': rms_g, 'E_pin': E_pin,
            'status': status,
        }
    except Exception as e:
        return {
            'w_pde': cfg.get('w_pde', 0), 'w_bc': cfg.get('w_bc', 0),
            'w_ortho': cfg.get('w_ortho', 0), 'lr': cfg.get('lr', 0),
            'E_init': cfg.get('E_init', 0),
            'dE': 9999, 'RMS_G': 9999, 'E_pin': 0,
            'status': f'error: {str(e)[:80]}',
        }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--workers', type=int, default=15)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    epochs = args.epochs

    if args.quick:
        w_pde_list = [5.0, 10.0, 20.0]
        w_bc_list = [3.0, 5.0, 10.0]
        w_ortho_list = [10.0, 50.0, 100.0]
        lr_list = [5e-4, 1e-3]
        E_init_list = [-10.0, -20.0, -30.0]
    else:
        w_pde_list = [1.0, 5.0, 10.0, 20.0, 50.0]
        w_bc_list = [1.0, 3.0, 5.0, 10.0, 20.0]
        w_ortho_list = [3.0, 10.0, 30.0, 50.0, 100.0, 200.0]
        lr_list = [2e-4, 5e-4, 1e-3, 2e-3]
        E_init_list = [-5.0, -10.0, -15.0, -20.0, -30.0]

    configs = [
        {'w_pde': wp, 'w_bc': wb, 'w_ortho': wo, 'lr': lr, 'E_init': ei, 'epochs': epochs}
        for wp, wb, wo, lr, ei in itertools.product(
            w_pde_list, w_bc_list, w_ortho_list, lr_list, E_init_list
        )
    ]
    total = len(configs)
    n_workers = min(args.workers, total)

    print(f"\n{'='*85}")
    print(f"  并行超参数搜索: {STATE}  目标E={TARGET_E_RAYLEIGH} MeV")
    print(f"  搜索空间: {total} 组 × {epochs}轮  |  workers={n_workers}")
    print(f"{'='*85}\n")

    t_start = time.time()
    results = []
    best_score = float('inf')
    best_cfg = None
    completed = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_single_config, c): c for c in configs}

        for future in as_completed(futures):
            completed += 1
            r = future.result()
            results.append(r)

            score = r['dE'] + (r.get('RMS_G', 0) if isinstance(r.get('RMS_G'), float) and r['RMS_G'] == r['RMS_G'] else 10)
            r['score'] = score

            marker = " ★★ BEST" if score < best_score else ""
            if score < best_score:
                best_score = score
                best_cfg = r.copy()

            elapsed = time.time() - t_start
            eta = (elapsed / completed) * (total - completed) / 60 if completed > 0 else 0

            print(f"  [{completed:>3d}/{total}] "
                  f"w_p={r['w_pde']:5.1f} w_bc={r['w_bc']:4.1f} w_or={r['w_ortho']:6.1f} "
                  f"lr={r['lr']:.0e} Ei={r['E_init']:+6.1f} → "
                  f"E={r['E_pin']:+8.2f}  dE={r['dE']:6.2f}  "
                  f"RMS={r['RMS_G']:.4f}  [{r['status']:>12s}] "
                  f" ETA={eta:.1f}min{marker}")

    total_time = time.time() - t_start

    # ── 汇总 ──
    results_sorted = sorted(results, key=lambda x: x['score'])
    top10 = results_sorted[:10]

    print(f"\n{'='*85}")
    print(f"  搜索完成! 总耗时: {total_time/60:.1f}分钟")
    print(f"{'='*85}")
    
    print(f"\n  ★ 最佳参数:")
    for k in ['w_pde','w_bc','w_ortho','lr','E_init','E_pin','dE','RMS_G']:
        print(f"    {k:<10s}= {best_cfg[k]}")

    save_path = os.path.join(OUTPUT_DIR, 'search_results.json')
    with open(save_path, 'w') as f:
        json.dump({
            'target_state': STATE, 'target_E': TARGET_E_RAYLEIGH,
            'total_configs': total, 'workers': n_workers,
            'total_time_min': round(total_time/60, 1),
            'best': best_cfg, 'top10': top10,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  完整结果: {save_path}")

    best_path = os.path.join(OUTPUT_DIR, 'best_params.json')
    with open(best_path, 'w') as f:
        json.dump({k: best_cfg[k] for k in ['w_pde','w_bc','w_ortho','lr','E_init','E_result','dE']
                   if k in best_cfg}, f, indent=2, default=float)
    print(f"  最佳参数: {best_path}")


if __name__ == '__main__':
    main()
