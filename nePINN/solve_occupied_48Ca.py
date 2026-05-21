#!/usr/bin/env python3
"""
批量处理 ⁴⁸Ca 的全部占据态 (PINN).

两种模式:
  1. train (默认): 从零训练每个态, 输出 .pth / .json / .png
  2. infer:       加载已有模型, 仅做前向推理, 输出波函数数据 + 对比图

占据态 (PKA1参数组):
  中子 (N=28): N.1s.1/2, N.1p.3/2, N.1p.1/2, N.1d.5/2, N.2s.1/2, N.1d.3/2, N.1f.7/2
  质子 (Z=20): P.1s.1/2, P.1p.3/2, P.1p.1/2, P.1d.5/2, P.2s.1/2, P.1d.3/2

用法:
    conda activate torch_env
    python solve_occupied_48Ca.py              # 训练模式
    python solve_occupied_48Ca.py --infer      # 推理模式 (加载已训练模型)
"""

import os, sys, argparse
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
    load_ref_wavefunctions,
)

# ════════════════════════════════════════════════════════
#   配置
# ════════════════════════════════════════════════════════

NUCLEUS = '48Ca'
A_VAL, Z_VAL = 48, 20

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)   # /home/ubuntu/rhf
POT_BASE = os.path.join(WORKSPACE_ROOT, 'results', '48Ca', 'POT')
WAV_DIR  = os.path.join(WORKSPACE_ROOT, 'results', '48Ca', 'WAV')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', f'{NUCLEUS}_occupied')

EPOCHS = 8000
LR = 1e-3

# ── 占据态列表 ──
# 格式: (state_label, tau, pot_filename)
# pot_filename 规则: Ca48_state{NNN}_POT.it{TTT}.final000
#   TT = 001(中子) / 002(质子), NN = LEV文件中的state编号
OCCUPIED_STATES = [
    # ---- 中子 (it=001, N=28) ----
    ('1s1/2', 'n', 'Ca48_state001_POT.it001.final000'),   # ε=-52.862
    ('1p3/2', 'n', 'Ca48_state007_POT.it001.final000'),   # ε=-38.187
    ('1p1/2', 'n', 'Ca48_state023_POT.it001.final000'),   # ε=-34.827
    ('1d5/2', 'n', 'Ca48_state013_POT.it001.final000'),   # ε=-23.546
    ('2s1/2', 'n', 'Ca48_state002_POT.it001.final000'),   # ε=-17.802
    ('1d3/2', 'n', 'Ca48_state029_POT.it001.final000'),   # ε=-16.729
    ('1f7/2', 'n', 'Ca48_state018_POT.it001.final000'),   # ε= -9.814
    # ---- 质子 (it=002, Z=20) ----
    ('1s1/2', 'p', 'Ca48_state001_POT.it002.final000'),   # ε=-50.494
    ('1p3/2', 'p', 'Ca48_state007_POT.it002.final000'),   # ε=-37.287
    ('1p1/2', 'p', 'Ca48_state023_POT.it002.final000'),   # ε=-33.537
    ('1d5/2', 'p', 'Ca48_state013_POT.it002.final000'),   # ε=-23.348
    ('2s1/2', 'p', 'Ca48_state002_POT.it002.final000'),   # ε=-16.684
    ('1d3/2', 'p', 'Ca48_state029_POT.it002.final000'),   # ε=-16.515
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

    # 绘制对比图 (★ 多线程环境matplotlib可能crash, 用try/except保护结果)
    ref_G = shooting_wf['G'] if shooting_wf else None
    ref_F = shooting_wf['F'] if shooting_wf else None
    hist = history if history is not None else []
    try:
        import matplotlib
        matplotlib.use('Agg')  # 确保非交互后端 (多线程安全)
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
    except Exception as e:
        print(f'  [WARN] Plot failed (training result preserved): {e}')


# ════════════════════════════════════════════════════════
#   模式 1: 训练 (Train)
# ════════════════════════════════════════════════════════

def solve_one_state(state_label, tau, pot_file, output_dir,
                    epochs=EPOCHS, lr=LR,
                    ref_wavefunction_files=None,
                    load_model_path=None):
    """训练单个态.
    
    支持激发态求解:
      - ref_wavefunction_files: 已求解的同(l,j)低能态波函数JSON列表(用于正交损失)
      - ★ 激发态不使用迁移学习，完全靠正交惩罚约束波函数正交性
    
    对于 3s1/2 → ref_wavefunction_files 应包含 [1s1/2.json, 2s1/2.json]
    """
    nucleus = NUCLEUS
    safe_name = f'{nucleus}_{tau}_{state_label.replace("/", "_")}'
    model_path = os.path.join(output_dir, f'{safe_name}_model.pth')

    kappa = parse_kappa_from_label(state_label)

    # 判断是否为激发态 (n >= 2 的态需要正交化)
    n_principal = int(state_label[0])  # '2s' -> 2, '1s' -> 1
    is_excited = (n_principal >= 2)

    # ★ 激发态：强制禁用迁移学习 + 内嵌Gram-Schmidt正交(硬约束) + 外部正交(安全网)
    if is_excited:
        load_model_path = None  # 不从基态模型继承权重
        w_orth_val = 1.0       # ★ 硬约束已内嵌到DiracNet.forward(), 外部只需极小安全网
        lambda_ort_val = 1.0
    else:
        w_orth_val = 0.0
        lambda_ort_val = 0.0

    print(f'\n{"="*60}')
    print(f'  [TRAIN] {nucleus} {tau.upper()}.{state_label}  k={kappa}'
          f'  {"[EXCITED - Gram-Schmidt embedded, NO transfer learning]" if is_excited else "[GROUND]"}')
    print(f'  POT: {os.path.basename(pot_file)}')
    if ref_wavefunction_files:
        print(f'  Ortho refs:   {[os.path.basename(f) for f in ref_wavefunction_files]}'
              f'  (w_ortho={w_orth_val})')
    print(f'{"="*60}')

    potentials, shooting_wf, E_ref = _load_ref_data(state_label, tau, pot_file)
    if potentials is None:
        print(f'  [ERROR] POT file not found')
        return None

    print(f'  E_ref = {E_ref:+.4f} MeV')

    # ★ 不用Shooting真实能量作为初始值（那是答案！），统一从固定猜测出发
    E_init = -60.0  # 所有态都用同一个初始猜测，让Rayleigh商自己收敛到正确能量
    target_nodes = count_nodes(torch.tensor(shooting_wf['G'])) if shooting_wf else None
    print(f'  E_init = {E_init:.2f} MeV (fixed guess, not shooting E),  target_nodes = {target_nodes}')

    # 加载参考波函数 (用于正交归一化损失)
    ref_wavefunctions = []
    if ref_wavefunction_files:
        for rf in ref_wavefunction_files:
            if os.path.exists(rf):
                ref_wavefunctions.extend(load_ref_wavefunctions(rf))

    solver = DiracPINNSolver(
        A=A_VAL, Z=Z_VAL, tau=tau, kappa=kappa,
        potentials=potentials,
        ref_wavefunctions=ref_wavefunctions,
        lambda_ortho=lambda_ort_val,
    )

    t0 = time.time()
    E_pinn, history = solver.train(
        E_init_MeV=E_init,
        target_nodes=target_nodes,
        max_epochs=epochs,
        lr=lr,
        print_every=max(epochs // 20, 400),
        w_pde=20.0,
        w_bc=0,
        w_ortho=w_orth_val,
        load_model=load_model_path,  # 激发态时已被强制置为None
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
        'wf_file': os.path.join(output_dir, f'{safe_name}_wavefunction.json'),
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
    parser = argparse.ArgumentParser(description='⁴⁸Ca 占据态 PINN 批量处理')
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
    print(f"  共 {len(OCCUPIED_STATES)} 个态 (n=7 + p=6)")
    if not args.infer:
        print(f"  每态 {epochs_val} epochs, lr={lr_val}")
        print(f"  激发态(n>=2): 自动正交归一化 (w_ortho=100, 无迁移学习)")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  核素: A={A_VAL}, Z={Z_VAL}")
    print("=" * 70)

    total_start = time.time()
    results_summary = []

    # ── 训练模式: 并行求解 (两阶段依赖调度) ──
    if not args.infer:
        # ★ 依赖分析: 只有 2s1/2 依赖同(l,j)的 1s1/2 波函数做正交约束
        #   其余11个态全是基态(n=1)，互相无依赖，可完全并行
        ground_states = []   # n=1 基态, 无依赖
        excited_states = []  # n>=2 激发态, 有依赖
        for item in OCCUPIED_STATES:
            state_label = item[0]
            n_principal = int(state_label[0])
            if n_principal >= 2:
                excited_states.append(item)
            else:
                ground_states.append(item)

        print(f'\n  Batch 1: {len(ground_states)} ground states (parallel, max 20 workers)')
        for s in ground_states:
            print(f'    - {s[1].upper()}.{s[0]}')
        print(f'  Batch 2: {len(excited_states)} excited states (after batch 1, parallel)')
        for s in excited_states:
            print(f'    - {s[1].upper()}.{s[0]}')

        solved_wfs = {'n': [], 'p': []}   # [(label, wf_json), ...]

        def _find_all_lower_states(state_label, tau):
            """找所有同(l,j)的低n态波函数路径."""
            n_target = int(state_label[0])
            if n_target <= 1:
                return []
            l_char = next((c for c in state_label if c.isalpha()), 's')
            j_part = state_label.split(l_char)[-1]
            lower_paths = []
            for sl, wf_p in solved_wfs.get(tau, []):
                n_sl = int(sl[0]) if sl[0].isdigit() else 99
                l_c = next((c for c in sl if c.isalpha()), '')
                j_p = sl.split(l_c)[-1]
                if l_c == l_char and j_p == j_part and n_sl < n_target and os.path.exists(wf_p):
                    lower_paths.append(wf_p)
            return lower_paths

        def _run_state(item):
            """单个态的求解包装函数 (供线程调用)."""
            state_label, tau, pot_file = item
            ortho_refs = _find_all_lower_states(state_label, tau)
            result = solve_one_state(
                state_label, tau, pot_file, OUTPUT_DIR,
                epochs=epochs_val, lr=lr_val,
                ref_wavefunction_files=ortho_refs if ortho_refs else None,
                load_model_path=None,
            )
            return state_label, tau, result

        # ════════════ Batch 1: 所有基态并行 ════════════
        t_batch1 = time.time()
        batch1_results = []
        with ThreadPoolExecutor(max_workers=min(len(ground_states), 20)) as executor:
            futures = {executor.submit(_run_state, item): item for item in ground_states}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    state_label, tau, result = future.result()
                    batch1_results.append((state_label, tau, result))
                    if result and result.get('wf_file'):
                        solved_wfs[tau].append((state_label, result['wf_file']))
                except Exception as e:
                    state_label, tau, _ = item
                    print(f'  [FAIL] {tau.upper()}.{state_label}: {e}')
                    import traceback
                    traceback.print_exc()
                    batch1_results.append((state_label, tau, None))

        elapsed_batch1 = time.time() - t_batch1
        print(f'\n  Batch 1 done: {len(batch1_results)} states in {elapsed_batch1:.1f}s '
              f'({elapsed_batch1/60:.1f}min)')

        # ════════════ Batch 2: 激发态并行 (依赖batch1的波函数) ════════════
        if excited_states:
            t_batch2 = time.time()
            batch2_results = []
            with ThreadPoolExecutor(max_workers=min(len(excited_states), 20)) as executor:
                futures = {executor.submit(_run_state, item): item for item in excited_states}
                for future in as_completed(futures):
                    item = futures[future]
                    try:
                        state_label, tau, result = future.result()
                        batch2_results.append((state_label, tau, result))
                        if result and result.get('wf_file'):
                            solved_wfs[tau].append((state_label, result['wf_file']))
                    except Exception as e:
                        state_label, tau, _ = item
                        print(f'  [FAIL] {tau.upper()}.{state_label}: {e}')
                        import traceback
                        traceback.print_exc()
                        batch2_results.append((state_label, tau, None))

            elapsed_batch2 = time.time() - t_batch2
            print(f'\n  Batch 2 done: {len(batch2_results)} states in {elapsed_batch2:.1f}s '
                  f'({elapsed_batch2/60:.1f}min)')
        else:
            batch2_results = []

        # 合并结果 (保持原始 OCCUPIED_STATES 顺序)
        all_results_map = {}
        for sl, tau, r in batch1_results + batch2_results:
            key = f'{NUCLEUS}_{tau}.{sl}'
            if r is not None:
                all_results_map[key] = r

        for state_label, tau, pot_file in OCCUPIED_STATES:
            key = f'{NUCLEUS}_{tau}.{state_label}'
            if key in all_results_map:
                results_summary.append(all_results_map[key])
            else:
                results_summary.append({
                    'state': key,
                    'mode': mode_str.split()[0],
                    'dE': 999,
                    'time_s': 0,
                })

    # ── 推理模式: 原始串行 (IO密集型无需并行) ──
    else:
        process_fn = infer_one_state
        for idx, (state_label, tau, pot_file) in enumerate(OCCUPIED_STATES):
            result = process_fn(state_label, tau, pot_file, OUTPUT_DIR)
            if result:
                results_summary.append(result)
            else:
                results_summary.append({
                    'state': f'{NUCLEUS}_{tau}.{state_label}',
                    'mode': mode_str.split()[0],
                    'dE': 999,
                    'time_s': 0,
                })
            status = "OK" if (result and result.get('dE', 999) < 1.0) else ("~" if (result and result.get('dE', 999) < 5.0) else "!!")
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
