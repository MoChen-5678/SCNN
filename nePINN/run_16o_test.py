"""
论文 PINN 测试: 求解 ¹⁶O 中子 1s₁/₂ + 2s₁/₂ 态
=============================================

使用 Fortran shooting 方法提供的势场(POT), 
按论文方法训练: 自适应配点 + Rayleigh商 + L_conz/L_orth

用法:
    cd /home/ubuntu/rhf/plusPINN && conda run -n torch_env python run_16o_test.py
"""

import sys
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/ubuntu/rhf/PINN')
from dirac_matrix_vs_pinn import DiracPINNSolver, load_shooting_potentials, R_GRID


# ─── 配置 ──────────────────────────────────────────────
A, Z = 16, 8
tau = 'n'          # 中子
BASE_DIR = '/home/ubuntu/rhf/results/16O'
POT_DIR = os.path.join(BASE_DIR, 'POT')
WAV_DIR = os.path.join(BASE_DIR, 'WAV')
OUT_DIR = '/home/ubuntu/rhf/plusPINN/test_outputs'

os.makedirs(OUT_DIR, exist_ok=True)

# 论文超参数
MAX_EPOCHS = 80000     # 论文用80000
LR = 1e-3
PRINT_EVERY = 2000

# 要解的态 (同 κ block: κ=-1 for s_{1/2})
STATES = [
    {
        'label': '1s1/2',
        'kappa': -1,
        'nodes': 0,
        'pot_file': 'O16_state001_POT.it001.final000',
        'E_ref': -36.674504,   # Fortran 参考能量 (MeV)
    },
    {
        'label': '2s1/2',
        'kappa': -1,
        'nodes': 1,
        'pot_file': 'O16_state002_POT.it001.final000',
        'E_ref': -3.297946,    # Fortran 参考能量 (MeV)
    },
]


def main():
    print("=" * 60)
    print(f"  论文 PINN 测试: {A}{Z} {tau}  s₁/₂ block")
    print(f"  网络架构: 80神经元 × 3层 × sigmoid")
    print(f"  训练轮数: {MAX_EPOCHS}")
    print(f"  自适应网格: 200→400 点 (每2000ep加10点)")
    print("=" * 60)

    results = []
    g_gs, f_gs = None, None   # 基态波函数 (用于激发态正交约束)

    for i, state in enumerate(STATES):
        label = state['label']
        kappa = state['kappa']
        pot_path = os.path.join(POT_DIR, state['pot_file'])
        E_ref = state['E_ref']

        print(f'\n{"="*60}')
        print(f'  [{i+1}/{len(STATES)}] 求解 {tau}.{label}  κ={kappa}  E_ref={E_ref:+.4f} MeV')
        print(f'  POT文件: {state["pot_file"]}')
        if g_gs is not None:
            print(f'  正交约束: ON (基态{STATES[0]["label"]}波函数)')
        else:
            print(f'  正交约束: OFF (基态)')
        print(f'{"="*60}')

        # 加载势场
        t0 = time.time()
        potentials = load_shooting_potentials(pot_path, R_GRID)

        # 创建求解器并训练
        solver = DiracPINNSolver(A=A, Z=Z, tau=tau, kappa=kappa,
                                 potentials=potentials)

        E_pinn, history, (g_pinn, f_pinn) = solver.train(
            target_nodes=state['nodes'],
            max_epochs=MAX_EPOCHS,
            lr=LR,
            print_every=PRINT_EVERY,
            g_ref=g_gs, f_ref=f_gs,   # 激发态传入基态正交参考
            E_target=E_ref,           # ★ Fortran参考能量 → 收敛判据 |E_ray-E_ref|
        )

        elapsed = time.time() - t0

        # 结果记录
        result = {
            'label': label,
            'kappa': kappa,
            'E_pinn': E_pinn,
            'E_ref': E_ref,
            'dE': abs(E_pinn - E_ref),
            'time_s': elapsed,
            'r': solver.r_np.copy(),
            'G': g_pinn,
            'F': f_pinn,
            'history': history,
        }
        results.append(result)

        # 打印摘要
        print(f'\n  ═══════ 结果摘要 ═══════')
        print(f'  E_PINN  = {E_pinn:+.6f} MeV')
        print(f'  E_Fort  = {E_ref:+.6f} MeV')
        print(f'  |dE|    = {abs(E_pinn-E_ref):.6f} MeV ({100*abs(E_pinn-E_ref)/abs(E_ref):.3f}%)')
        print(f'  时间    = {elapsed:.1f}s')

        # 归一化检查
        dr = R_GRID[1] - R_GRID[0]
        norm = np.trapz(g_pinn**2 + f_pinn**2, dx=dr)
        print(f'  归一化  = {norm:.6f} (目标=1.000000)')

        # 保存基态供下一步正交使用
        if g_gs is None:
            g_gs, f_gs = g_pinn.copy(), f_pinn.copy()
            print(f'  → 基态波函数已保存 (用于后续正交约束)')

    # ── 绘制对比图 ──
    plot_results(results)


def plot_results(results):
    """绘制 G/F 波函数对比图"""
    n_states = len(results)
    fig, axes = plt.subplots(n_states, 2, figsize=(14, 5*n_states))

    if n_states == 1:
        axes = axes[np.newaxis, :]

    for i, res in enumerate(results):
        r = res['r']
        G = res['G']
        F = res['F']

        # G 分量
        ax_g = axes[i, 0]
        ax_g.plot(r, G, 'b-', linewidth=2, label=f'PINN')
        ax_g.set_ylabel('$G(r)$')
        ax_g.set_title(f'{res["label"]}  $\\kappa$={res["kappa"]}  '
                       f'$E_{{PINN}}$={res["E_pinn"]:+.4f} MeV\n'
                       f'$E_{{ref}}$={res["E_ref"]:+.4f}  $\\Delta E$={res["dE"]:.4f} MeV',
                       fontsize=11)
        ax_g.axhline(0, color='gray', lw=0.5)
        ax_g.grid(True, alpha=0.3)
        ax_g.set_xlim([0, min(20, r[-1])])
        if i == n_states - 1:
            ax_g.set_xlabel('r (fm)')

        # F 分量
        ax_f = axes[i, 1]
        ax_f.plot(r, F, 'r-', linewidth=2, label='PINN')
        ax_f.set_ylabel('$F(r)$')
        ax_f.set_title(f'{res["label"]} $F(r)$ component')
        ax_f.axhline(0, color='gray', lw=0.5)
        ax_f.grid(True, alpha=0.3)
        ax_f.set_xlim([0, min(20, r[-1])])
        if i == n_states - 1:
            ax_f.set_xlabel('r (fm)')

    plt.suptitle(
        f'{A}{Z} {tau.upper()} — 论文 PINN (80×3, sigm, adapt 200→400pt)\n'
        f'L_DE(ω=1) + L_conz + L_orth',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()

    outpath = os.path.join(OUT_DIR, f'O16_{tau}_s12_paper_method.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f'\n  对比图已保存: {outpath}')

    # 绘制损失曲线
    fig2, ax_loss = plt.subplots(figsize=(10, 4))
    colors = ['b', 'r', 'g', 'orange']
    for i, res in enumerate(results):
        h = res['history']
        epochs = [x['epoch'] for x in h]
        losses = [x['loss'] for x in h]
        energies = [x['E_MeV'] for x in h]
        ax_loss.semilogy(epochs, losses, '-', color=colors[i], alpha=0.7,
                         label=f'{res["label"]} Loss')
        ax_loss2 = ax_loss.twinx()
        ax_loss2.plot(epochs, energies, '--', color=colors[i], alpha=0.5,
                      label=f'{res["label"]} Energy')

    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss (log)', color='black')
    ax_loss2.set_ylabel('Energy (MeV)', color='gray')
    ax_loss.legend(loc='upper left', fontsize=9)
    ax_loss.set_title(f'Training History ({MAX_EPOCHS} epochs)')
    ax_loss.grid(True, alpha=0.3)

    outpath2 = os.path.join(OUT_DIR, f'O16_{tau}_training_history.png')
    plt.savefig(outpath2, dpi=150, bbox_inches='tight')
    print(f'  训练历史已保存: {outpath2}')

    # 最终汇总表
    print(f'\n{"═"*65}')
    print(f'  {"态":<10} {"κ":>4} {"E_PINN(MeV)":>14} {"E_Fortran(MeV)":>15} {"|ΔE|(MeV)":>12} {"误差%":>8}')
    print(f'{"-"*65}')
    for res in results:
        err_pct = 100*abs(res['dE']) / abs(res['E_ref'])
        print(f'  {res["label"]:<10} {res["kappa"]:>+4d} '
              f'{res["E_pinn"]:>+14.6f} {res["E_ref"]:>+15.6f} '
              f'{res["dE"]:>+12.6f} {err_pct:>7.3f}%')
    print(f'{"═"*65}')


if __name__ == '__main__':
    main()
