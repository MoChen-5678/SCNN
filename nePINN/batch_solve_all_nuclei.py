#!/usr/bin/env python3
"""
批量求解 4参数集 × 15核素 的全部占据态 (PINN).

功能:
  1. 从 LEV 文件自动读取占据态 (vv != 0)
  2. 用 PINN 求解每个态 (基态8000epochs, 激发态根据主量子数n增加)
  3. 计算核宏观量: E/A, Ef(N), Ef(P), R, Rc
  4. 与 Fortran PKA1 最后一轮收敛值对比
  5. 绘制能谱对比图 (PINN vs Shooting)

并行策略:
  - 按 主量子数 n 分组: n=1 全部并行 → n=2 全部并行 → n=3 ...
  - 同一 n 组内的态互不依赖, 可真正并行
  - 不同 n 组串行执行 (高n可能需要低n波函数做正交化)

用法:
    conda activate torch_env
    python batch_solve_all_nuclei.py                    # 全部核素
    python batch_solve_all_nuclei.py --pset PKA1        # 仅 PKA1
    python batch_solve_all_nuclei.py --nucleus 16O 48Ca # 指定核素
    python batch_solve_all_nuclei.py --infer             # 推理模式
    python batch_solve_all_nuclei.py --workers 4         # 并行度(默认自动)

输出:
    outputs/batch_{pset}/{nucleus}/
      ├── {nucleus}_{tau}_{label}_model.pth        (模型权重)
      ├── {nucleus}_{tau}_{label}_wavefunction.json  (波函数JSON)
      ├── {nucleus}_{tau}_{label}_wavefunction.csv   (波函数CSV, 论文用)
      ├── {nucleus}_{tau}_{label}_comparison.png     (单态 G/F 对比图)
      ├── {nucleus}_spectrum.png          (能谱对比图 PINN vs Shooting)
      ├── {nucleus}_energy.csv            (单粒子能量对比表)
      ├── {nucleus}_bulk.csv              (宏观量 E/A Ef R Rc 对比表)
      ├── {nucleus}_wavefunctions_all.csv (全部态波函数汇总, 宽格式)
      └── {nucleus}_bulk.json             (完整结果 JSON)
"""

import os, sys, argparse, re, json, time, glob
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import HBAR_C, DR, R_GRID
from dirac_matrix_vs_pinn import (
    DiracPINNSolver, DiracNet,
    load_shooting_potentials, load_wav_wavefunction,
    parse_kappa_from_label, compute_wav_rayleigh_energy,
    plot_comparison, count_nodes, load_ref_wavefunctions,
)

# ════════════════════════════════════════════════════════
#   配置
# ════════════════════════════════════════════════════════

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)       # /home/ubuntu/rhf
RESULTS_BASE = os.path.join(PROJECT_ROOT, 'results')  # legacy: plusPINN/results/

PARAM_SETS = ['PKA1', 'PKO1', 'PKO2', 'PKO3']

# 15个核素: (dirname, A, Z, symbol) — dirname与目录名一致
NUCLEI_INFO = {
    '16O':  (16, 8,  'O'),
    '22O':  (22, 8,  'O'),
    '24O':  (24, 8,  'O'),
    '40Ca': (40, 20, 'Ca'),
    '48Ca': (48, 20, 'Ca'),
    '52Ca': (52, 20, 'Ca'),
    '60Ca': (60, 20, 'Ca'),
    '56Ni': (56, 28, 'Ni'),
    '68Ni': (68, 28, 'Ni'),
    '78Ni': (78, 28, 'Ni'),
    '124Sn':(124,50,'Sn'),
    '132Sn':(132,50,'Sn'),
    '208Pb':(208,82,'Pb'),
    '210Pb':(210,82,'Pb'),
}

BASE_EPOCHS = 8000    # n=1:8k, n=2:16k, n=3:24k, n=4:32k, n=5:40k
LR = 1e-3

# 并行配置: 每个 worker 是独立进程, 避免GIL限制
# GPU模式下建议 <= 4 (显存竞争), CPU模式可设更大
DEFAULT_MAX_WORKERS = None  # None = 自动 (CPU核数或4)

# ════════════════════════════════════════════════════════
#   占据态解析 (从 WAV 文件第一行读取 vv 占据数)
# ════════════════════════════════════════════════════════

def parse_occupied_from_wav(wav_dir, nucleus_name, pset, tau):
    """从 WAV 文件解析占据态列表.
    
    WAV 文件格式 (如 O16.G-N.PKA1):
      第1行: vv 占据数 (每列对应一个态, 0=空 1=占据 2=双占据等)
      第2行: 态名 header (r  N.1s.1/2  N.2s.1/2 ...)
      第3行+: r网格 + 波函数值
    
    返回: list of dict {
        'state': str ('N.1s.1/2'),
        'tau': str ('n'/'p'), 'kappa': int,
        'occupation': float (vv值), 'eig': float,
        'label': str ('1s1/2'),
    }
    """
    states = []
    tau_tag = 'N' if tau == 'n' else 'P'
    wav_file = os.path.join(wav_dir, f'{nucleus_name}.G-{tau_tag}.{pset}')
    
    if not os.path.exists(wav_file):
        print(f'    WARN: WAV not found: {wav_file}')
        return states
    
    with open(wav_file) as f:
        lines = f.readlines()
    
    if len(lines) < 2:
        return states
    
    # 第1行: vv 值
    vv_vals = [float(x) for x in lines[0].split()]
    
    # 第2行: 态名
    state_names = lines[1].split()
    
    # 确保对齐 (第1个列是 'r', 跳过)
    # vv 行没有 r 列, 但 header 有 r 列
    # 检查: 如果 len(vv_vals) == len(state_names)-1, 则 state_names[0]='r' 需跳过
    if state_names and state_names[0] == 'r':
        state_names = state_names[1:]
    
    assert len(vv_vals) == len(state_names), \
        f'vv({len(vv_vals)}) != names({len(state_names)}) in {wav_file}'
    
    # 尝试从 LEV 读能量 (如果存在的话)
    lev_path = os.path.join(os.path.dirname(wav_dir), 'LEV',
                            f'{nucleus_name}.psp-{tau_tag}.{pset}')
    lev_energies = {}
    if os.path.exists(lev_path):
        try:
            with open(lev_path) as lf:
                for ll in lf.readlines()[1:]:
                    pp = ll.split()
                    if len(pp) >= 5:
                        lev_energies[pp[1]] = float(pp[4])
        except Exception:
            pass
    
    for i, (vv, sname) in enumerate(zip(vv_vals, state_names)):
        if vv == 0:
            continue
        
        # 解析态名: 'N.1s.1/2' -> label='1s1/2', tau='n'
        parts = sname.split('.')
        if len(parts) < 3:
            continue
        
        raw_label = '.'.join(parts[1:])   # '1s.1/2'
        label = raw_label.replace('.', '', 1)  # '1s1/2'
        
        kappa = parse_kappa_from_label(label)
        eig = lev_energies.get(sname, 0.0)
        
        states.append({
            'idx': i,
            'state': sname,
            'tau': tau,
            'kappa': kappa,
            'occupation': vv,  # 直接使用 vv 值 (可能>1表示多粒子占据)
            'eig': eig,
            'label': label,
        })
    
    return states


def parse_states_from_pot(pot_dir, tau=None):
    """从新 POT 文件头解析态列表。

    用于没有 WAV/LEV 的目录，如 plusPINN/PKA1/208Pb/POT。这里扫描全部
    POT 态，occupation 直接读取 header；空态 occupation=0 也保留，以便
    生成全局波函数汇总。
    """
    states = []
    if not os.path.isdir(pot_dir):
        return states

    for fname in sorted(os.listdir(pot_dir)):
        if 'final000' not in fname:
            continue
        fpath = os.path.join(pot_dir, fname)
        state_name = None
        energy = None
        occ = 0.0
        with open(fpath) as f:
            for line in f:
                if not line.startswith('#'):
                    break
                m = re.search(r'State:\s+(\S+),\s+Energy=\s+([-\d.]+)', line)
                if m:
                    state_name = m.group(1)
                    energy = float(m.group(2))
                m = re.search(r'Occupation probability:\s+([-\d.]+)', line)
                if m:
                    occ = float(m.group(1))
                if state_name is not None and energy is not None and 'Columns:' in line:
                    break

        if state_name is None or energy is None:
            continue
        tau_from_state = 'p' if state_name.startswith('P.') else 'n'
        if tau is not None and tau != tau_from_state:
            continue

        raw_label = state_name[2:]       # e.g. 1s.1/2
        label = raw_label.replace('.', '', 1)
        degeneracy = abs(parse_kappa_from_label(label)) * 2
        states.append({
            'idx': len(states),
            'state': state_name,
            'tau': tau_from_state,
            'kappa': parse_kappa_from_label(label),
            'occupation': degeneracy * occ,
            'occupation_probability': occ,
            'degeneracy': degeneracy,
            'eig': energy,
            'label': label,
        })

    states.sort(key=lambda s: (s['tau'], s['eig'], s['label']))
    return states


def get_pot_filename(nucleus_dir, state_info, pset):
    """根据态信息构建 POT 文件名."""
    # POT 文件命名规则: {Nucleus}_{N|P}.{state_label_no_dash}_POT.it{001|002}.final000.{PSET}
    # 例如: O16_N.1s.1-2_POT.it001.final000.PKA1
    # 核素名格式: POT文件用 O16 (符号+质量), 不是 16O
    raw_name = os.path.basename(nucleus_dir)  # e.g. '16O'
    if raw_name in NUCLEI_INFO:
        A_val, Z_val, sym = NUCLEI_INFO[raw_name]
        nucleus_name = f'{sym}{A_val}'        # 16O -> O16
    else:
        nucleus_name = raw_name
    tau_tag = 'N' if state_info['tau'] == 'n' else 'P'
    it_num = '001' if state_info['tau'] == 'n' else '002'
    
    # state label 格式转换: '1s1/2' -> '1s.1-2' (POT文件中的格式)
    lbl = state_info['label']
    m = re.match(r'(\d)([a-z])(\d+)/(\d+)', lbl)
    if m:
        n, l, j1, j2 = m.groups()
        pot_state = f'{n}{l}.{j1}-{j2}'
    else:
        pot_state = lbl.replace('/', '-')
    
    fname = f'{nucleus_name}_{tau_tag}.{pot_state}_POT.it{it_num}.final000.{pset}'
    return fname


# ════════════════════════════════════════════════════════
#   Fortran 汇总数据解析
# ════════════════════════════════════════════════════════

def parse_fortran_summary(pka1_path):
    """从 .PKA1/.PKOx 文件最后一轮提取收敛的宏观量.
    
    返回 dict: {'E_A': float, 'Ef_n': float, 'Ef_p': float,
                 'R': float, 'Rc': float, 'iteration': int}
    或 None
    """
    if not os.path.exists(pka1_path):
        return None
    
    last_result = None
    with open(pka1_path) as f:
        for line in f:
            # 匹配格式: "  NN si = value E/A = value Ef = value value R = value Rc = value"
            m = re.search(
                r'si\s*=\s*([\d.E+-]+)\s+E/A\s*=\s*([\d.E+-]+)'
                r'\s+Ef\s*=\s*([\d.E+-]+)\s+([\d.E+-]+)'
                r'\s+R\s*=\s*([\d.E+-]+)\s+Rc\s*=\s*([\d.E+-]+)',
                line
            )
            if m:
                last_result = {
                    'E_A': float(m.group(2)),
                    'Ef_n': float(m.group(3)),
                    'Ef_p': float(m.group(4)),
                    'R': float(m.group(5)),
                    'Rc': float(m.group(6)),
                    'iteration': int(line.split()[0]),
                }
    return last_result


# ════════════════════════════════════════════════════════
#   单态求解
# ════════════════════════════════════════════════════════

def solve_one_state(state_info, pot_file_full, output_dir, wav_dir,
                   epochs=BASE_EPOCHS, lr=LR, infer_mode=False,
                   ref_wavefunction_files=None):
    """求解单个态 (训练或推理)."""
    nucleus = state_info.get('_nucleus', '?')
    tau = state_info['tau']
    label = state_info['label']
    kappa = state_info['kappa']
    safe_name = f'{nucleus}_{tau}_{label.replace("/", "_")}'
    model_path = os.path.join(output_dir, f'{safe_name}_model.pth')
    
    # 判断激发态
    n_principal = int(label[0])
    is_excited = (n_principal >= 2)
    if is_excited:
        effective_epochs = n_principal * BASE_EPOCHS
    else:
        effective_epochs = BASE_EPOCHS
    
    # ── 推理模式 ──
    if infer_mode:
        if not os.path.exists(model_path):
            return None
        return _infer_state(state_info, model_path, output_dir, wav_dir)
    
    # ── 训练模式 ──
    print(f'\n  [TRAIN] {nucleus} {tau.upper()}.{label} k={kappa}'
          f' {"[EXCITED n=%d -> %dep]" % (n_principal, effective_epochs) if is_excited else "[GROUND %dep]" % effective_epochs}')
    
    potentials = load_shooting_potentials(pot_file_full, R_GRID)
    if potentials is None:
        print(f'    ERROR: POT not found: {pot_file_full}')
        return None
    
    # 参考能量
    E_ref = state_info['eig']  # 来自 LEV 文件的 Shooting 能量
    
    # 加载 WAV 波函数作为参考
    shooting_wf = None
    if os.path.isdir(wav_dir):
        try:
            wf = load_wav_wavefunction(wav_dir, tau, state_info['state'])
            if wf is not None:
                shooting_wf = wf
                E_ray = compute_wav_rayleigh_energy(wf, potentials, kappa)
                if E_ray is not None:
                    wf['E'] = E_ray
                    E_ref = E_ray
        except Exception as e:
            print(f'    WARN: WAV load failed: {e}')
    
    E_init = -60.0
    target_nodes = None
    if shooting_wf is not None:
        target_nodes = count_nodes(torch.tensor(shooting_wf['G']))
    
    # 正交化参考波函数
    ref_wfs = []
    if ref_wavefunction_files:
        for rf in ref_wavefunction_files:
            if os.path.exists(rf):
                ref_wfs.extend(load_ref_wavefunctions(rf))
    
    w_orth = 1.0 if is_excited else 0.0
    lambda_ort = 1.0 if is_excited else 0.0
    
    solver = DiracPINNSolver(
        A=state_info['_A'], Z=state_info['_Z'], tau=tau, kappa=kappa,
        potentials=potentials, ref_wavefunctions=ref_wfs,
        lambda_ortho=lambda_ort,
    )
    
    t0 = time.time()
    # 激发态不使用迁移学习
    E_pinn, history = solver.train(
        E_init_MeV=E_init, target_nodes=target_nodes,
        max_epochs=effective_epochs, lr=lr,
        print_every=max(effective_epochs // 20, 400),
        w_pde=20.0, w_bc=0, w_ortho=w_orth,
        load_model=None if is_excited else None,
        live_plot=False,
    )
    elapsed = time.time() - t0
    G_pinn, F_pinn = solver.get_wavefunction()
    
    dE = abs(E_pinn - E_ref)
    print(f'    E_Ref={E_ref:+.3f} E_PINN={E_pinn:+.3f} dE={dE:.4f} MeV ({elapsed:.1f}s)')
    
    # 保存模型和波函数
    solver.save_model(model_path, E_MeV=E_pinn)
    _save_wf_output(safe_name, output_dir, G_pinn, F_pinn, solver.r_np,
                    E_pinn, E_ref, label, tau, shooting_wf, history)
    
    return {
        'state': f'{nucleus}_{tau}.{label}',
        'tau': tau, 'label': label, 'kappa': kappa,
        'E_ref': E_ref, 'E_pinn': E_pinn, 'dE': dE,
        'time_s': elapsed, 'epochs': effective_epochs,
        'model': model_path,
        'wf_file': os.path.join(output_dir, f'{safe_name}_wavefunction.json'),
        'occupation': state_info['occupation'],
    }


@torch.no_grad()
def _infer_state(state_info, model_path, output_dir, wav_dir):
    """推理模式: 加载模型 -> 前向传播 -> 保存."""
    nucleus = state_info.get('_nucleus', '?')
    tau = state_info['tau']
    label = state_info['label']
    kappa = state_info['kappa']
    safe_name = f'{nucleus}_{tau}_{label.replace("/", "_")}'
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    net = DiracNet(n_hidden=128, n_layers=6, activation='swish',
                   hard_normalize=True, init_energy=state_info.get('eig', -40.0))
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        net.load_state_dict(checkpoint['state_dict'], strict=False)
        loaded_E = checkpoint.get('E')
    else:
        net.load_state_dict(checkpoint, strict=False)
        loaded_E = None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device).eval()
    
    r_tensor = torch.tensor(R_GRID, dtype=torch.float32, device=device).unsqueeze(0)
    g, f = net(r_tensor, kappa=kappa, dr=DR)
    G_pinn = g.squeeze(0).cpu().numpy()
    F_pinn = f.squeeze(0).cpu().numpy()
    
    E_pinn = float(loaded_E) if loaded_E is not None else state_info['eig']
    E_ref = state_info['eig']
    
    # 尝试加载 WAV 用于绘图
    shooting_wf = None
    if os.path.isdir(wav_dir):
        try:
            shooting_wf = load_wav_wavefunction(wav_dir, tau, state_info['state'])
        except Exception:
            pass
    
    _save_wf_output(safe_name, output_dir, G_pinn, F_pinn, np.array(R_GRID),
                    E_pinn, E_ref, label, tau, shooting_wf, None)
    
    return {
        'state': f'{nucleus}_{tau}.{label}',
        'tau': tau, 'label': label,
        'E_ref': E_ref, 'E_pinn': E_pinn,
        'dE': abs(E_pinn - E_ref), 'time_s': 0.0,
        'occupation': state_info['occupation'],
    }


def _save_wf_output(safe_name, output_dir, G, F, r, E_pinn, E_ref,
                    label, tau, shooting_wf, history):
    """保存波函数 JSON + 对比图."""
    wf_path = os.path.join(output_dir, f'{safe_name}_wavefunction.json')
    with open(wf_path, 'w') as f:
        json.dump({
            'state_name': safe_name, 'r': r.tolist(),
            'G': G.tolist(), 'F': F.tolist(),
            'E_PINN': float(E_pinn), 'E_Ref': float(E_ref),
            'tau': tau, 'label': label,
        }, f, indent=2)
    
    # 绘图
    ref_G = shooting_wf['G'] if shooting_wf else None
    ref_F = shooting_wf['F'] if shooting_wf else None
    hist = history if history is not None else []
    if shooting_wf is not None:
        try:
            import matplotlib; matplotlib.use('Agg')
            plot_comparison(
                r, {'label': label, 'E': E_ref}, ref_G, ref_F,
                r, G, F, E_ref, E_pinn, hist,
                output_dir=output_dir, shooting_data=shooting_wf, tau=tau,
            )
            auto_plot = os.path.join(output_dir, f'matrix_vs_pinn_{safe_name}.png')
            plot_path = os.path.join(output_dir, f'{safe_name}_comparison.png')
            if os.path.exists(auto_plot):
                os.rename(auto_plot, plot_path)
        except Exception as e:
            print(f'    WARN: Plot failed: {e}')

    # ── 同时保存波函数CSV (方便论文使用) ──
    _save_wf_csv(safe_name, output_dir, G, F, r, E_pinn, E_ref, label, tau)


def _save_wf_csv(safe_name, output_dir, G, F, r, E_pinn, E_ref, label, tau):
    """保存单态波函数数据为 CSV."""
    csv_path = os.path.join(output_dir, f'{safe_name}_wavefunction.csv')
    import csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'r(fm)', 'G_PINN', 'F_PINN',
            f'E_PINN(MeV)', f'E_Ref(MeV)',
            'tau', 'label'
        ])
        for i in range(len(r)):
            writer.writerow([f'{r[i]:.6f}', f'{G[i]:.10e}', f'{F[i]:.10e}',
                             E_pinn if i == 0 else '', E_ref if i == 0 else '',
                             tau if i == 0 else '', label if i == 0 else ''])


# ════════════════════════════════════════════════════════
#   宏观量计算 & 能谱图绘制
# ════════════════════════════════════════════════════════

def compute_bulk_quantities(results_list, A, Z):
    """从求解结果计算核宏观量.
    
    核心公式 (用户要求重点处理):
      E_total = Σ_i (occupation_i × ε_i)   ← 单粒子能级求和 (PINN求解的ε≈Shooting)
      E/A = E_total / A
      Ef_n = 最后一个中子占据态的能量
      Ef_p = 最后一个质子占据态的能量
      R   = 均方根半径 (从波函数积分, 或近似用Shooting值)
      Rc  = 电荷均方根半径
    """
    if not results_list:
        return {}
    
    # 按 tau 分组
    n_states = [r for r in results_list if r['tau'] == 'n']
    p_states = [r for r in results_list if r['tau'] == 'p']
    
    # E_total = Σ occ_i × E_pinn_i  (PINN能量几乎等于Shooting, 所以E/A准确)
    E_n = sum(r['occupation'] * r['E_pinn'] for r in n_states) if n_states else 0
    E_p = sum(r['occupation'] * r['E_pinn'] for r in p_states) if p_states else 0
    E_total = E_n + E_p
    E_per_A = E_total / A if A > 0 else 0
    
    # Fermi energy = 最高占据态能量 (最接近0的负能量, 或最小正值)
    Ef_n = max([r['E_pinn'] for r in n_states]) if n_states else 0
    Ef_p = max([r['E_pinn'] for r in p_states]) if p_states else 0
    
    # 半径: 从波函数密度积分计算 (如果有的话)
    # 这里先返回基本信息, R/Rc 后续可以从DEN文件或波函数积分得到
    R_calc = None
    Rc_calc = None
    
    return {
        'E_total': E_total,
        'E_per_A': E_per_A,
        'Ef_n': Ef_n,
        'Ef_p': Ef_p,
        'R': R_calc,
        'Rc': Rc_calc,
        'N_n': len(n_states),
        'N_p': len(p_states),
    }


def plot_spectrum(nucleus, pset, results_list, fortran_data, output_dir):
    """绘制能谱对比图: PINN vs Shooting."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(6, len(results_list)*0.35 + 2)))
    
    # 分离中子/质子
    n_res = sorted([r for r in results_list if r['tau']=='n'], key=lambda x: x['E_pinn'])
    p_res = sorted([r for r in results_list if r['tau']=='p'], key=lambda x: x['E_pinn'])
    
    def draw_axis(ax, states, tau_label, color):
        if not states:
            ax.set_title(f'{tau_label} (no states)')
            return
        y_pos = np.arange(len(states))
        
        E_shoot = [r['E_ref'] for r in states]
        E_pinn  = [r['E_pinn'] for r in states]
        labels  = [r['label'] for r in states]
        occ    = [int(r['occupation']) for r in states]
        
        # Shooting 能级 (左)
        ax.barh(y_pos - 0.18, E_shoot, height=0.32,
               color=color, alpha=0.7, label=f'Shooting ({tau_label})', align='center')
        # PINN 能级 (右, 微偏移显示差异)
        ax.barh(y_pos + 0.18, E_pinn, height=0.32,
               color=color, edgecolor='black', linewidth=0.8, alpha=0.4,
               label=f'PINN ({tau_label})', align='center')
        
        # 标注差值
        for i, (es, ep, lab, o) in enumerate(zip(E_shoot, E_pinn, labels, occ)):
            dE = ep - es
            ax.text(max(es, ep) + 1, i, f'{dE:+.3f}', va='center', fontsize=7,
                   color='red' if abs(dE) > 0.01 else 'green')
            ax.text(min(es, ep) - 3, i, f'{lab}({o})', va='center', ha='right', fontsize=8)
        
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_ylabel('State')
        ax.set_xlabel('Energy (MeV)')
        ax.set_title(f'{nucleus} [{pset}] {tau_label} Spectrum')
        ax.legend(loc='lower right', fontsize=8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.grid(axis='x', alpha=0.3)
    
    draw_axis(ax1, n_res, 'Neutron', '#2166ac')
    draw_axis(ax2, p_res, 'Proton', '#b2182b')
    
    # 总标题含宏观量对比
    bulk = fortran_data or {}
    title = f'{nucleus} [{pset}] Energy Spectrum: PINN vs Shooting'
    if bulk:
        title += f'\n(Fortran: E/A={bulk.get("E_A","?"):.3f} R={bulk.get("R","?"):.4f} Rc={bulk.get("Rc","?"):.4f})'
    fig.suptitle(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, f'{nucleus}_spectrum.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Spectrum saved: {out_path}')


# ════════════════════════════════════════════════════════
#   CSV 输出 (论文友好格式)
# ════════════════════════════════════════════════════════

import csv as _csv_module

def save_energy_csv(nucleus, pset, results_list, fortran_data, output_dir):
    """保存单粒子能量对比表 CSV."""
    csv_path = os.path.join(output_dir, f'{nucleus}_energy.csv')
    with open(csv_path, 'w', newline='') as f:
        w = _csv_module.writer(f)
        # header
        w.writerow([
            'Nucleus', 'PSet', 'tau', 'label', 'n', 'l', 'j',
            'E_Shoot(MeV)', 'E_PINN(MeV)', 'dE(keV)',
            'occupation', 'time(s)', 'epochs',
        ])
        for r in sorted(results_list, key=lambda x: (x['tau'], x['label'])):
            lbl = r['label']
            n_pr = int(lbl[0]) if lbl[0].isdigit() else 0
            l_ch = next((c for c in lbl if c.isalpha()), '?')
            j_p = lbl.split(l_ch)[-1] if l_ch in lbl else '?'
            dE_keV = abs(r.get('E_pinn', 0) - r.get('E_ref', 0)) * 1000
            w.writerow([
                nucleus, pset, r['tau'], lbl,
                n_pr, l_ch, j_p,
                f'{r["E_ref"]:.6f}', f'{r["E_pinn"]:.6f}', f'{dE_keV:.3f}',
                r.get('occupation', ''),
                f'{r.get("time_s", 0):.1f}',
                r.get('epochs', ''),
            ])
    print(f'  Energy CSV saved: {csv_path}')
    return csv_path


def save_bulk_csv(nucleus, pset, A, Z, fortran_data, pinn_bulk, comparison,
                  output_dir):
    """保存宏观量对比表 CSV."""
    csv_path = os.path.join(output_dir, f'{nucleus}_bulk.csv')
    with open(csv_path, 'w', newline='') as f:
        w = _csv_module.writer(f)
        w.writerow(['Nucleus', 'PSet', 'A', 'Z', 'Quantity', 'Fortran', 'PINN', 'Diff'])
        fb = fortran_data or {}
        pb = pinn_bulk or {}
        cmp = comparison or {}
        
        rows = [
            ('E/A (MeV)', fb.get('E_A'), pb.get('E_per_A'), cmp.get('E_A_diff')),
            ('Ef_n (MeV)', fb.get('Ef_n'), pb.get('Ef_n'), cmp.get('Ef_n_diff')),
            ('Ef_p (MeV)', fb.get('Ef_p'), pb.get('Ef_p'), cmp.get('Ef_p_diff')),
            ('R (fm)',     fb.get('R'),     pb.get('R'),     None),
            ('Rc (fm)',    fb.get('Rc'),    pb.get('Rc'),    None),
            ('N_states',   '-',             pb.get('N_n',0)+pb.get('N_p',0), None),
        ]
        for qname, fv, pv, dv in rows:
            fv_str = f'{fv:.6f}' if isinstance(fv, (int,float)) else str(fv if fv else '-')
            pv_str = f'{pv:.6f}' if isinstance(pv, (int,float)) else str(pv if pv else '-')
            dv_str = f'{dv:.6f}' if isinstance(dv, (int,float)) and dv is not None else '-'
            w.writerow([nucleus, pset, A, Z, qname, fv_str, pv_str, dv_str])
    print(f'  Bulk CSV saved: {csv_path}')
    return csv_path


def save_wavefunctions_csv(nucleus, pset, output_dir):
    """汇总所有态的波函数到一个 CSV 文件 (长格式)."""
    json_files = sorted(glob.glob(os.path.join(
        output_dir, f'{nucleus}_*_wavefunction.json')))
    if not json_files:
        return None
    
    csv_path = os.path.join(output_dir, f'{nucleus}_wavefunctions_all.csv')
    
    # 先收集所有数据
    all_rows = []
    headers_set = False
    max_len = 0
    
    for jf in json_files:
        with open(jf) as fh:
            d = json.load(fh)
        rr = np.array(d['r'])
        GG = np.array(d['G'])
        FF = np.array(d['F'])
        max_len = max(max_len, len(rr))
        all_rows.append({
            'state': d.get('state_name', os.path.basename(jf)),
            'tau': d.get('tau', ''), 'label': d.get('label', ''),
            'E_PINN': d.get('E_PINN', 0), 'E_Ref': d.get('E_Ref', 0),
            'r': rr, 'G': GG, 'F': FF,
            'len': len(rr),
        })
    
    # 宽格式: 每行一个 r 点, 列为各态的 G/F
    with open(csv_path, 'w', newline='') as f:
        w = _csv_module.writer(f)
        # header row
        hdr = ['r(fm)']
        for row_info in all_rows:
            prefix = f"{row_info['state']}"
            hdr.extend([f'{prefix}_G', f'{prefix}_F'])
        w.writerow(hdr)
        
        # data rows
        for i in range(max_len):
            line = [f'{all_rows[0]["r"][i]:.6f}' if i < all_rows[0]['len'] else '']
            for row_info in all_rows:
                if i < row_info['len']:
                    line.append(f'{row_info["G"][i]:.10e}')
                    line.append(f'{row_info["F"][i]:.10e}')
                else:
                    line.extend(['', ''])
            w.writerow(line)
    
    print(f'  All-wavefunction CSV saved: {csv_path} ({max_len} grid points × {len(all_rows)} states)')
    return csv_path


# ════════════════════════════════════════════════════════
#   单核素处理流程
# ════════════════════════════════════════════════════════

def _find_lower(tau, label, solved_wfs):
    """找同(l,j)的低n态波函数JSON路径."""
    n_target = int(label[0])
    if n_target <= 1:
        return []
    l_char = next((c for c in label if c.isalpha()), 's')
    j_part = label.split(l_char)[-1]
    paths = []
    for sl, wp in solved_wfs.get(tau, []):
        n_sl = int(sl[0]) if sl[0].isdigit() else 99
        l_c = next((c for c in sl if c.isalpha()), '')
        j_p = sl.split(l_c)[-1] if l_c in sl else ''
        if l_c == l_char and j_p == j_part and n_sl < n_target and os.path.exists(wp):
            paths.append(wp)
    return paths


def resolve_nucleus_dir(pset, nucleus):
    """优先使用 plusPINN/{pset}/{nucleus}，不存在时回退到 plusPINN/results/{pset}/{nucleus}."""
    direct_dir = os.path.join(PROJECT_ROOT, pset, nucleus)
    if os.path.isdir(direct_dir):
        return direct_dir, 'direct'
    legacy_dir = os.path.join(RESULTS_BASE, pset, nucleus)
    if os.path.isdir(legacy_dir):
        return legacy_dir, 'results'
    return direct_dir, 'missing'


def _run_one(s, ctx):
    """包装单态求解, 自动查找正交化参考."""
    pot_fname = get_pot_filename(ctx['nucleus_dir'], s, ctx['pset'])
    pot_full = os.path.join(ctx['pot_base'], pot_fname)
    ortho_refs = _find_lower(s['tau'], s['label'], ctx['solved_wfs'])

    n_principal = int(s['label'][0])
    effective_epochs = max(BASE_EPOCHS * n_principal, BASE_EPOCHS)

    return solve_one_state(
        s, pot_full, ctx['output_dir'], ctx['wav_dir'],
        epochs=effective_epochs, lr=ctx['lr'], infer_mode=ctx['infer_mode'],
        ref_wavefunction_files=ortho_refs if ortho_refs else None,
    )


def process_nucleus(nucleus, pset, infer_mode=False, lr=LR, max_workers=None,
                    output_dir_name=None):
    """处理一个核素的全部占据态."""
    nucleus_dir, data_source = resolve_nucleus_dir(pset, nucleus)
    if not os.path.isdir(nucleus_dir):
        print(f'  SKIP {nucleus}: directory not found ({nucleus_dir})')
        return None
    
    A, Z, symbol = NUCLEI_INFO[nucleus]
    pot_base = os.path.join(nucleus_dir, 'POT')
    wav_dir = os.path.join(nucleus_dir, 'WAV')
    lev_dir = os.path.join(nucleus_dir, 'LEV')
    output_leaf = output_dir_name or nucleus
    output_dir = os.path.join(PROJECT_ROOT, 'outputs', f'batch_{pset}', output_leaf)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f'\n{"="*70}')
    print(f'  NUCLEUS: {nucleus} (A={A}, Z={Z}) | PARAM: {pset} | MODE: {"INFER" if infer_mode else "TRAIN"}')
    print(f'  DATA SOURCE: {data_source} ({nucleus_dir})')
    print(f'  POT: {pot_base}')
    print(f'  WAV: {wav_dir}')
    print(f'  OUT: {output_dir}')
    print(f'{"="*70}')
    
    # 1. 优先从 WAV 读取占据态；没有 WAV 时从 POT 读取全部态
    lev_states = []
    _sym = NUCLEI_INFO[nucleus][2]
    _A_val = NUCLEI_INFO[nucleus][0]
    nucleus_name_for_file = f'{_sym}{_A_val}'  # 16O -> O16, 208Pb -> Pb208
    for tau in ['n', 'p']:
        states = parse_occupied_from_wav(wav_dir, nucleus_name_for_file, pset, tau)
        for s in states:
            s['_A'] = A; s['_Z'] = Z; s['_nucleus'] = nucleus
        lev_states.extend(states)
        print(f"  WAV {tau}: {len(states)} occupied states (vv!=0) from G-{'N' if tau=='n' else 'P'}")

    if not lev_states:
        lev_states = parse_states_from_pot(pot_base)
        for s in lev_states:
            s['_A'] = A; s['_Z'] = Z; s['_nucleus'] = nucleus
        print(f'  POT fallback: {len(lev_states)} total states from POT headers')
    
    if not lev_states:
        print(f'  SKIP: no occupied states found')
        return None
    
    total_occ_n = sum(s['occupation'] for s in lev_states if s['tau']=='n')
    total_occ_p = sum(s['occupation'] for s in lev_states if s['tau']=='p')
    print(f'  Total: N_n={total_occ_n:.0f} N_p={total_occ_p:.0f} A={total_occ_n+total_occ_p:.0f}')
    
    # 2. 解析 Fortran 收敛结果 (文件名: O16.PKA1, 不是 16O.PKA1)
    _sym_A = f'{NUCLEI_INFO[nucleus][2]}{NUCLEI_INFO[nucleus][0]}'  # 16O -> O16
    pka1_file = os.path.join(nucleus_dir, f'{_sym_A}.{pset}')
    fort_data = parse_fortran_summary(pka1_file)
    
    # 自动确定并行度
    if max_workers is None:
        cpu_count = os.cpu_count() or 4
        # GPU模式下限制并发避免显存爆掉
        if torch.cuda.is_available():
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            # 每个PINN约需 ~500MB-1GB 显存 (128隐藏层6层网络)
            auto_workers = max(1, min(int(gpu_mem_gb / 1.5), cpu_count))
        else:
            auto_workers = min(cpu_count, 8)  # CPU模式可多开
        max_workers = auto_workers
    
    if fort_data:
        print(f'  Fortran converged (iter {fort_data["iteration"]}): '
              f'E/A={fort_data["E_A"]:.4f} Ef_n={fort_data["Ef_n"]:.3f} '
              f'Ef_p={fort_data["Ef_p"]:.3f} R={fort_data["R"]:.4f} Rc={fort_data["Rc"]:.4f}')
    
    # 3. 按 主量子数n 分组 → 组内并行, 组间串行
    from collections import OrderedDict
    n_groups = OrderedDict()
    for s in lev_states:
        n_key = int(s['label'][0])
        n_groups.setdefault(n_key, []).append(s)
    
    print(f'\n  并行策略: 按 n 分组串行, 组内并行求解')
    for n_key, group in n_groups.items():
        epochs_for_group = max(BASE_EPOCHS * n_key, BASE_EPOCHS)
        print(f'    n={n_key}: {len(group)} 态 × {epochs_for_group} epochs (并行)')
    
    # 存储已解出的波函数供正交化用
    solved_wfs = {'n': [], 'p': []}
    all_results = []
    
    # ── Worker: 单态求解 (独立进程调用) ──
    # 构建传递给 _run_one 的上下文 (spawn 需要 pickle)
    run_ctx = {
        'nucleus_dir': nucleus_dir, 'pset': pset,
        'pot_base': pot_base, 'output_dir': output_dir,
        'wav_dir': wav_dir, 'lr': lr, 'infer_mode': infer_mode,
        'solved_wfs': solved_wfs,
    }
    
    t_start = time.time()
    
    if not infer_mode:
        # ═══════════ 训练模式: 按n分组并行 ═══════════
        for n_key, group in sorted(n_groups.items()):
            effective_epochs = max(BASE_EPOCHS * n_key, BASE_EPOCHS)
            n_workers = min(len(group), max_workers) if max_workers else len(group)
            
            print(f'\n  ▶ n={n_key} batch: {len(group)} states || {n_workers} workers × {effective_epochs} epochs')
            t_batch = time.time()
            
            # 关键: 使用 spawn 上下文避免 fork 下 CUDA 问题
            ctx = mp.get_context('spawn')
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
                futures = {executor.submit(_run_one, s, run_ctx): s for s in group}
                for fut in as_completed(futures):
                    s = futures[fut]
                    try:
                        res = fut.result(timeout=3600)  # 1h timeout per state
                        if res and res.get('wf_file'):
                            solved_wfs[s['tau']].append((s['label'], res['wf_file']))
                        if res:
                            all_results.append(res)
                            dE_str = f'dE={res["dE"]:.4f}' if res.get('dE') is not None else '?'
                            print(f'    ✓ {s["state"]:20s} E_PINN={res.get("E_pinn",0):>+8.3f} {dE_str} '
                                  f'({res.get("time_s",0):.1f}s)')
                    except Exception as e:
                        print(f'    ✗ FAIL {s["state"]}: {e}')
            
            print(f'    ◆ n={n_key} done in {time.time()-t_batch:.1f}s '
                  f'({len([r for r in all_results if int(r["label"][0])==n_key])}/{len(group)} OK)')
    
    else:
        # ═══════════ 推理模式: 全部串行 ═══════════
        for n_key, group in sorted(n_groups.items()):
            for s in group:
                res = _run_one(s, run_ctx)
                if res:
                    all_results.append(res)
                    print(f'    ✓ {s["state"]:20s} dE={res["dE"]:.4f}')
    
    elapsed = time.time() - t_start
    print(f'\n  Done: {len(all_results)}/{len(lev_states)} states in {elapsed:.1f}s')
    
    if not all_results:
        return None
    
    # 4. 计算宏观量
    bulk_pinn = compute_bulk_quantities(all_results, A, Z)
    
    # 5. 绘制能谱图
    plot_spectrum(nucleus, pset, all_results, fort_data, output_dir)
    
    # 6. 汇总保存
    summary = {
        'nucleus': nucleus, 'pset': pset, 'A': A, 'Z': Z,
        'mode': 'infer' if infer_mode else 'train',
        'total_time_s': elapsed,
        'states_solved': len(all_results),
        'states_total': len(lev_states),
        'fortran': fort_data,
        'pinn_bulk': bulk_pinn,
        'comparison': {} if fort_data else None,
        'results': all_results,
    }
    
    if fort_data and bulk_pinn:
        summary['comparison'] = {
            'E_A_diff': bulk_pinn['E_per_A'] - fort_data['E_A'],
            'Ef_n_diff': bulk_pinn['Ef_n'] - fort_data['Ef_n'],
            'Ef_p_diff': bulk_pinn['Ef_p'] - fort_data['Ef_p'],
        }
        print(f'\n  ┌─────────────────────────────────────────┐')
        print(f'  │  BULK COMPARISON: {nucleus:6s} [{pset}]│')
        print(f'  ├──────────┬──────────┬──────────┬────────┤')
        print(f'  │ Quantity │  Fortran │    PINN  │  Diff  │')
        print(f'  ├──────────┼──────────┼──────────┼────────┤')
        ea_d = summary['comparison']['E_A_diff']
        efn_d = summary['comparison']['Ef_n_diff']
        efp_d = summary['comparison']['Ef_p_diff']
        print(f'  │ E/A      │ {fort_data["E_A"]:>+8.4f} │ {bulk_pinn["E_per_A"]:>+8.4f} │ {ea_d:>+6.4f}│')
        print(f'  │ Ef_n     │ {fort_data["Ef_n"]:>+8.3f} │ {bulk_pinn["Ef_n"]:>+8.3f} │ {efn_d:>+6.3f}│')
        print(f'  │ Ef_p     │ {fort_data["Ef_p"]:>+8.3f} │ {bulk_pinn["Ef_p"]:>+8.3f} │ {efp_d:>+6.3f}│')
        print(f'  │ R        │ {fort_data["R"]:>8.4f} │ {"N/A":>8s} │        │')
        print(f'  │ Rc       │ {fort_data["Rc"]:>8.4f} │ {"N/A":>8s} │        │')
        print(f'  └──────────┴──────────┴──────────┴────────┘')
    
    # 保存 JSON
    summary_path = os.path.join(output_dir, f'{nucleus}_bulk.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'  Summary: {summary_path}')
    
    # ── 7. 保存 CSV (论文友好) ──
    cmp_data = summary.get('comparison')
    save_energy_csv(nucleus, pset, all_results, fort_data, output_dir)
    save_bulk_csv(nucleus, pset, A, Z, fort_data, bulk_pinn, cmp_data, output_dir)
    save_wavefunctions_csv(nucleus, pset, output_dir)
    
    return summary


# ════════════════════════════════════════════════════════
#   主入口
# ════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='批量求解全部核素占据态 (PINN vs Shooting)')
    parser.add_argument('--pset', nargs='+', default=PARAM_SETS,
                       help=f'参数集 (默认全部: {PARAM_SETS})')
    parser.add_argument('--nucleus', nargs='+', default=list(NUCLEI_INFO.keys()),
                       help=f'核素 (默认全部: {list(NUCLEI_INFO.keys())})')
    parser.add_argument('--infer', action='store_true',
                       help='推理模式 (加载已有模型, 不训练)')
    parser.add_argument('--lr', type=float, default=LR, help=f'学习率 (默认 {LR})')
    parser.add_argument('--workers', type=int, default=None,
                       help=f'并行进程数 (默认=自动: CPU核数或4)')
    parser.add_argument('--output-dir-name', type=str, default=None,
                       help='输出子目录名 (默认核素名, 如 208Pb；可设为 Pb208-New 做对比)')
    args = parser.parse_args()
    
    mode_str = 'INFER' if args.infer else 'TRAIN'
    print('=' * 80)
    print(f'  BATCH SOLVER: {mode_str} mode')
    print(f'  Param sets: {args.pset}')
    print(f'  Nuclei:     {args.nucleus}')
    print(f'  Base epochs per ground state: {BASE_EPOCHS}')
    print(f'  Excited state scaling: n * {BASE_EPOCHS}')
    w_str = str(args.workers) if args.workers else 'AUTO (GPU-mem / CPU-count)'
    print(f'  Parallel:   max_workers={w_str}')
    print(f'  Strategy:   group-by-n (n=1‖n=2‖n=3... each in parallel)')
    print(f'  Output:     {PROJECT_ROOT}/outputs/batch_{{pset}}/{{nucleus}}/')
    print('=' * 80)
    
    grand_start = time.time()
    all_summaries = []
    
    for pset in args.pset:
        for nucleus in args.nucleus:
            if nucleus not in NUCLEI_INFO:
                print(f'  WARNING: Unknown nucleus "{nucleus}", skipping.')
                continue
            
            summary = process_nucleus(
                nucleus, pset, infer_mode=args.infer, lr=args.lr,
                max_workers=args.workers, output_dir_name=args.output_dir_name,
            )
            if summary:
                all_summaries.append(summary)
    
    total_time = time.time() - grand_start
    
    # 最终汇总表
    print(f'\n{"="*90}')
    print(f'  GRAND TOTAL: {len(all_summaries)} nuclei processed in {total_time:.1f}s ({total_time/60:.1f}min)')
    print(f'{"="*90}')
    print(f'  {"Nucleus":8s} {"PSet":6s} {"N_st":>5s} {"E/A_Fort":>9s} {"E/A_PINN":>9s} {"dE/A":>8s} '
          f'{"Ef_nF":>8s} {"Ef_nP":>8s} {"dEf_n":>7s} {"Ef_pF":>8s} {"Ef_pP":>8s} {"dEf_p":>7s}')
    print(f'  {"-"*88}')
    for s in all_summaries:
        comp = s.get('comparison')
        fb = s.get('fortran', {})
        pb = s.get('pinn_bulk', {})
        nuc = s['nucleus']
        ps = s['pset']
        ns = s['states_solved']
        if comp:
            print(f'  {nuc:8s} {ps:6s} {ns:5d} '
                  f'{fb.get("E_A",0):>+9.4f} {pb.get("E_per_A",0):>+9.4f} {comp.get("E_A_diff",0):>+8.4f} '
                  f'{fb.get("Ef_n",0):>+8.3f} {pb.get("Ef_n",0):>+8.3f} {comp.get("Ef_n_diff",0):>+7.3f} '
                  f'{fb.get("Ef_p",0):>+8.3f} {pb.get("Ef_p",0):>+8.3f} {comp.get("Ef_p_diff",0):>+7.3f}')
        else:
            print(f'  {nuc:8s} {ps:6s} {ns:5d}  (no Fortran comparison data)')
    
    # 保存全局汇总
    grand_dir = os.path.join(PROJECT_ROOT, 'outputs')
    os.makedirs(grand_dir, exist_ok=True)
    grand_path = os.path.join(grand_dir, 'batch_grand_summary.json')
    with open(grand_path, 'w') as f:
        json.dump({
            'mode': mode_str,
            'total_time_s': total_time,
            'summaries': all_summaries,
        }, f, indent=2, default=str)
    print(f'\n  Grand summary saved: {grand_path}')


if __name__ == '__main__':
    main()
