#!/usr/bin/env python3
"""
重训练 large_error_states.csv 中的大误差态
==============================================
参考 batch_solve_all_nuclei.py 的实现方式:
  1. 直接调用 DiracPINNSolver Python API (非子进程CLI)
  2. 正交参考只选同(l,j)的低n态 (如 4d5/2 → [2d5/2, 1d5/2])
  3. 逐层迁移学习: 在同lj的 n-1 模型基础上继续训
  4. 自适应早停: 80000轮上限, dE<0.1 MeV

用法:
    python retrain_large_errors.py --dry-run     # 预览
    python retrain_large_errors.py               # 执行
    python retrain_large_errors.py --nucleus 208Pb
"""

import os, sys, csv, glob, argparse, shutil, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import HBAR_C, DR, R_GRID
from dirac_matrix_vs_pinn import (
    DiracPINNSolver,
    load_shooting_potentials, parse_kappa_from_label,
    load_ref_wavefunctions, count_nodes,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
LARGE_ERROR_CSV = os.path.join(OUTPUTS_DIR, 'large_error_states.csv')

# ── 训练参数 ──
LR = 5e-4

# ── 核素符号映射 ──
NUC_SYM_MAP = {
    '16O': ('O', 16), '22O': ('O', 22), '24O': ('O', 24),
    '40Ca': ('Ca', 40), '48Ca': ('Ca', 48), '52Ca': ('Ca', 52),
    '60Ca': ('Ca', 60),
    '56Ni': ('Ni', 56), '68Ni': ('Ni', 68), '78Ni': ('Ni', 78),
    '124Sn': ('Sn', 124), '132Sn': ('Sn', 132),
    '208Pb': ('Pb', 208), '210Pb': ('Pb', 210),
}

def get_nuc_sym_A(nucleus):
    if nucleus in NUC_SYM_MAP:
        return NUC_SYM_MAP[nucleus]
    import re
    m = re.match(r'(\D+)(\d+)', nucleus)
    return (m.group(1), int(m.group(2))) if m else (nucleus, 0)


def extract_lj(label):
    """从 '3f5/2' 提取 lj='f5/2'"""
    import re
    m = re.match(r'\d+([a-z]\d+/\d+)', label)
    return m.group(1) if m else label


# ════════════════════════════════════════════════════════
#   加载 & 筛选
# ════════════════════════════════════════════════════════

def load_large_error_states(filter_nucleus=None):
    if not os.path.exists(LARGE_ERROR_CSV):
        print(f'[ERROR] 找不到 {LARGE_ERROR_CSV}')
        return []
    rows = []
    with open(LARGE_ERROR_CSV) as f:
        for row in csv.DictReader(f):
            if filter_nucleus and row['nucleus'] != filter_nucleus:
                continue
            rows.append(row)
    rows.sort(key=lambda x: (int(x['n']), -abs(float(x['dE_keV']))))
    return rows


def _label_to_filename_pattern(label):
    """'4d5/2' → '4d.5-2'"""
    import re
    m = re.match(r'(\d+)([a-z])(\d+)/(\d+)', label)
    if not m:
        return label.replace('/', '-')
    return f'{m.group(1)}{m.group(2)}.{m.group(3)}-{m.group(4)}'


def find_pot_file(nucleus, pset, tau, label):
    sym, A = get_nuc_sym_A(nucleus)
    pot_dir = os.path.join(RESULTS_DIR, pset, nucleus, 'POT')
    if not os.path.isdir(pot_dir):
        return None
    it_tag = 'it001' if tau == 'n' else 'it002'
    tau_prefix = 'N' if tau == 'n' else 'P'
    fn_pattern = _label_to_filename_pattern(label)
    target_fname = f'{sym}{A}_{tau_prefix}.{fn_pattern}_POT.{it_tag}.final000.{pset}'
    exact_path = os.path.join(pot_dir, target_fname)
    if os.path.exists(exact_path):
        return exact_path
    target_state_name = f'{tau_prefix}.{label}'
    for fname in sorted(os.listdir(pot_dir)):
        if not fname.endswith(f'.final000.{pset}'):
            continue
        fpath = os.path.join(pot_dir, fname)
        if fn_pattern in fname:
            return fpath
        try:
            with open(fpath) as f:
                for line in f:
                    if ('State:' in line and target_state_name in line) or \
                       line.strip().startswith(f'State: {target_state_name}'):
                        return fpath
                    if not line.startswith('#'):
                        break
        except:
            pass
    return None


# ════════════════════════════════════════════════════════
#   正交参考: 只选同(l,j)的低n态 (与 batch_solve_all_nuclei.py 一致)
# ════════════════════════════════════════════════════════

def find_same_lj_lower_wfs(model_dir, nucleus, tau, label, target_n):
    """
    找同核素同tau同(l,j)但n更小的已解波函数.
    
    例如 4d5/2 → [2d5/2, 1d5/2] (如果存在)
    与 batch_solve_all_nuclei.py 中 _find_lower() 逻辑一致.
    
    返回 list of dict (g, f, name) — 直接加载好的 tensor 格式.
    """
    lj = extract_lj(label)
    l_char = next((c for c in label if c.isalpha()), 's')
    j_part = label.split(l_char)[-1] if l_char in label else ''
    
    prefix = f'{nucleus}_{tau}_'
    if not os.path.isdir(model_dir):
        return []
    
    ref_paths = []
    for fname in sorted(os.listdir(model_dir)):
        if not fname.startswith(prefix) or not fname.endswith('_wavefunction.json'):
            continue
        label_part = fname[len(prefix):-len('_wavefunction.json')]
        ref_label = label_part.replace('_', '/')
        
        # 检查同(l,j)且n < target_n
        ref_lj = extract_lj(ref_label)
        ref_n_str = ref_label[0] if ref_label and ref_label[0].isdigit() else '9'
        try:
            ref_n = int(ref_n_str)
        except:
            continue
        
        if ref_lj == lj and ref_n < target_n:
            ref_paths.append(os.path.join(model_dir, fname))
    
    # 加载为 tensor 格式
    ref_wfs = []
    for rp in ref_paths:
        loaded = load_ref_wavefunctions(rp)
        ref_wfs.extend(loaded)
    
    return ref_wfs


def find_lower_n_model(model_dir, nucleus, tau, label, target_n):
    """找同(l,j)的 n-1 模型文件用于迁移学习."""
    if target_n <= 1:
        return None
    lj = extract_lj(label)
    lower_label = f'{target_n-1}{lj}'
    safe_lower = lower_label.replace('/', '_')
    candidate = os.path.join(model_dir, f'{nucleus}_{tau}_{safe_lower}_model.pth')
    return candidate if os.path.exists(candidate) else None


# ════════════════════════════════════════════════════════
#   单态训练 (直接调用 Python API, 与 batch_solve_all_nuclei.py 一致)
# ════════════════════════════════════════════════════════

def solve_one_state(item, dry_run=False):
    """
    用 DiracPINNSolver Python API 求解单个大误差态.
    
    参考 batch_solve_all_nuclei.py 的 solve_one_state():
      - 同(l,j)正交化 (lambda_ortho=1.0, w_ortho=1.0)
      - 逐层迁移学习 (load_model = n-1 同lj模型)
      - E_init = Shooting 能量
    
    返回 dict 或 None.
    """
    nucleus = item['nucleus']
    pset = item['pset']
    tau = item['tau']
    label = item['label']
    n = int(item['n'])
    E_shoot = float(item['E_Shoot_MeV'])
    
    pot_file = find_pot_file(nucleus, pset, tau, label)
    if not pot_file:
        print(f'    [SKIP] POT 文件未找到')
        return None
    
    model_dir = os.path.join(OUTPUTS_DIR, f'batch_{pset}', nucleus)
    safe_label = label.replace('/', '_')
    model_file = os.path.join(model_dir, f'{nucleus}_{tau}_{safe_label}_model.pth')
    wf_json = os.path.join(model_dir, f'{nucleus}_{tau}_{safe_label}_wavefunction.json')
    
    # ★ 正交参考: 只选同(l,j)的低n态
    ref_wfs = find_same_lj_lower_wfs(model_dir, nucleus, tau, label, n)
    
    # ★ 迁移学习: 加载同lj的n-1模型
    load_model_path = find_lower_n_model(model_dir, nucleus, tau, label, n)
    
    is_excited = n >= 2
    
    kappa = parse_kappa_from_label(label)
    A_val = get_nuc_sym_A(nucleus)[0]
    Z_val = get_nuc_sym_A(nucleus)[1]
    
    print(f'  态={tau}{label}  n={n}  κ={kappa}')
    print(f'  目标: |dE|<0.1 MeV  max=200000轮')
    print(f'  POT={os.path.basename(pot_file)}')
    print(f'  同(l,j)正交参考: {[r["name"] for r in ref_wfs]} ({len(ref_wfs)}个)')
    print(f'  迁移模型: {os.path.basename(load_model_path) if load_model_path else "无 (全新)" }')
    
    if dry_run:
        return {'status': 'DRY_RUN'}
    
    # ── 加载势场 ──
    potentials = load_shooting_potentials(pot_file, R_GRID)
    if potentials is None:
        print(f'    [ERROR] 势场加载失败: {pot_file}')
        return None
    
    # ── 创建求解器 (与 batch_solve 一致) ──
    lambda_ort = 1.0 if is_excited else 0.0
    solver = DiracPINNSolver(
        A=A_val, Z=Z_val, tau=tau, kappa=kappa,
        potentials=potentials,
        ref_wavefunctions=ref_wfs,
        lambda_ortho=lambda_ort,
    )
    
    t0 = time.time()
    w_orth = 1.0 if is_excited else 0.0
    E_pinn, history = solver.train(
        E_init_MeV=E_shoot,
        max_epochs=80000,
        lr=LR,
        print_every=1000,
        w_pde=20.0,
        w_bc=0,
        w_ortho=w_orth,
        load_model=load_model_path,
        live_plot=False,
    )
    elapsed = time.time() - t0
    
    G_pinn, F_pinn = solver.get_wavefunction()
    dE = abs(E_pinn - E_shoot)
    
    print(f'    E_Shoot={E_shoot:+.4f}  E_PINN={E_pinn:+.4f}  dE={dE:.6f} MeV  ({elapsed:.1f}s)')
    
    # ── 保存 ──
    solver.save_model(model_file, E_MeV=E_pinn)
    with open(wf_json, 'w') as f:
        json.dump({
            'state_name': f'{nucleus}_{tau}_{label.replace("/", "_")}',
            'r': R_GRID.tolist(),
            'G': G_pinn.tolist(), 'F': F_pinn.tolist(),
            'E_PINN': float(E_pinn), 'E_Ref': float(E_shoot),
            'tau': tau, 'label': label,
        }, f, indent=2)
    print(f'    已保存: {os.path.basename(model_file)}, {os.path.basename(wf_json)}')
    
    return {
        'item': item,
        'status': 'OK',
        'E_pinn': E_pinn,
        'dE_MeV': dE,
        'elapsed_s': elapsed,
        'model_file': model_file,
        'wf_json': wf_json,
        'ref_count': len(ref_wfs),
        'load_model': load_model_path,
    }


# ════════════════════════════════════════════════════════
#   CSV 更新
# ════════════════════════════════════════════════════════

def update_energy_csv(nucleus, pset, tau, label, new_E_pinn, new_epochs):
    pattern = os.path.join(OUTPUTS_DIR, f'batch_{pset}', nucleus, f'{nucleus}_energy.csv')
    matches = glob.glob(pattern)
    if not matches:
        matches = glob.glob(os.path.join(OUTPUTS_DIR, f'batch_{pset}', nucleus, '*_energy.csv'))
    if not matches:
        print(f'    [WARN] 未找到 energy.csv: {pattern}')
        return
    ecsv_path = matches[0]
    
    with open(ecsv_path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    
    updated = False
    for row in rows:
        if row.get('tau') == tau and row.get('label') == label and not updated:
            E_shoot = float(row['E_Shoot(MeV)'])
            old_de = float(row['dE(keV)'])
            new_de = (new_E_pinn - E_shoot) * 1000
            row['E_PINN(MeV)'] = f'{new_E_pinn:.6f}'
            row['dE(keV)'] = f'{new_de:.3f}'
            row['epochs'] = str(new_epochs)
            print(f'    energy.csv更新: dE {old_de:+.0f} -> {new_de:+.0f} keV')
            updated = True
            break
    
    if updated:
        with open(ecsv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def update_large_error_csv(nucleus, pset, tau, label, new_E_pinn, new_epochs):
    if not os.path.exists(LARGE_ERROR_CSV):
        return
    with open(LARGE_ERROR_CSV) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    updated = False
    for row in rows:
        if (row.get('nucleus') == nucleus and row.get('pset') == pset
                and row.get('tau') == tau and row.get('label') == label
                and not updated):
            E_shoot = float(row['E_Shoot_MeV'])
            old_de = float(row['dE_keV'])
            new_de = (new_E_pinn - E_shoot) * 1000
            row['E_PINN_MeV'] = f'{new_E_pinn:.6f}'
            row['dE_keV'] = f'{new_de:.3f}'
            row['prev_epochs'] = str(new_epochs)
            print(f'    large_error.csv更新: dE {old_de:+.0f} -> {new_de:+.0f} keV')
            updated = True
            break
    if updated:
        with open(LARGE_ERROR_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


# ════════════════════════════════════════════════════════
#   主流程
# ════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='重训练大误差态 (Python API + 同lj正交)')
    parser.add_argument('--lr', type=float, default=LR, help=f'学习率 (默认{LR})')
    parser.add_argument('--nucleus', type=str, default=None, help='限定核素')
    parser.add_argument('--dry-run', action='store_true', help='预览模式')
    args = parser.parse_args()

    print('=' * 72)
    print('  PINN 重训练脚本 (Python API, 同(l,j)正交化)')
    print(f'  数据源: large_error_states.csv')
    print(f'  自适应训练: 80000轮上限, dE<0.1MeV')
    print('=' * 72)

    states = load_large_error_states(filter_nucleus=args.nucleus)
    if not states:
        print('\n[OK] 无待重训态.')
        return

    print(f'\n共 {len(states)} 个待重训态:\n')
    print(f'{"核素":6s} {"参数集":5s} {"态":10s} {"n":>2s} '
          f'{"E_shoot":>10s} {"旧dE(keV)":>11s} {"收敛条件":>12s}')
    print('-' * 58)

    results_summary = []

    for idx, item in enumerate(states):
        nucleus = item['nucleus']
        pset = item['pset']
        tau = item['tau']
        label = item['label']
        n = int(item['n'])
        de_old = float(item['dE_keV'])
        E_shoot = float(item['E_Shoot_MeV'])

        print(f'\n{"="*60}')
        print(f'[{idx+1}/{len(states)}] {nucleus} {pset} {tau}{label}  '
              f'(旧 dE={de_old:+.0f} keV)')
        print(f'{"="*60}')

        result = solve_one_state(item, dry_run=args.dry_run)
        if result is None:
            results_summary.append({'_item': item, 'status': 'NO_POT'})
            print(f'    ✗ 跳过')
            continue
        
        if result['status'] == 'DRY_RUN':
            results_summary.append({'_item': item, 'status': 'DRY'})
            continue

        # 更新 CSV
        actual_epochs = 80000
        update_energy_csv(nucleus, pset, tau, label, result['E_pinn'], actual_epochs)
        update_large_error_csv(nucleus, pset, tau, label, result['E_pinn'], actual_epochs)

        result['old_de_keV'] = de_old
        results_summary.append(result)

    # ── 汇总 ──
    print(f'\n\n{"═" * 72}')
    print('  重训练汇总')
    print(f'{"═" * 72}')

    ok_cnt = sum(1 for r in results_summary if r.get('status') == 'OK')
    fail_cnt = len(results_summary) - ok_cnt
    print(f'  总计: {len(results_summary)}  成功: {ok_cnt}  失败: {fail_cnt}\n')

    # 输出汇总表格
    summary_csv = os.path.join(OUTPUTS_DIR, 'retrain_results.csv')
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'nucleus', 'pset', 'tau', 'label', 'n',
            'E_Shoot_MeV', 'E_PINN_new_MeV',
            'dE_old_keV', 'dE_new_keV',
            'improvement_keV', 'time_s',
            'n_ortho_refs', 'transfer_from',
        ])
        for r in results_summary:
            mark = '[OK]' if r.get('status') == 'OK' else '[FAIL]'
            item = r.get('_item', r.get('item', {}))
            
            if r.get('status') == 'OK':
                old_d = r.get('old_de_keV', 0)
                new_d = (r['E_pinn'] - float(r['item']['E_Shoot_MeV'])) * 1000
                imp = old_d - new_d
                lm = os.path.basename(r['load_model']) if r.get('load_model') else 'fresh'
                
                print(f'  {mark} {r["item"]["nucleus"]:6s} {r["item"]["pset"]:5s} '
                      f'{r["item"]["tau"]}{r["item"]["label"]:9s}  '
                      f'dE {old_d:+8.1f} -> {new_d:+8.1f} keV  '
                      f'(改善 {imp:+.1f} keV, {r["elapsed_s"]:.0f}s)')
                
                writer.writerow([
                    r['item']['nucleus'], r['item']['pset'],
                    r['item']['tau'], r['item']['label'],
                    r['item']['n'],
                    f"{r['item']['E_Shoot_MeV']}", f"{r['E_pinn']:.6f}",
                    f"{old_d:.3f}", f"{new_d:.3f}",
                    f"{imp:.3f}", f"{r['elapsed_s']:.1f}",
                    r['ref_count'], lm,
                ])
            else:
                st = r.get('status', '?')
                print(f'  {mark} {item.get("nucleus","?"):6s} {item.get("pset",""):5s} '
                      f'{item.get("tau","")}{item.get("label",""):9s}  status={st}')
                writer.writerow([
                    item.get('nucleus',''), item.get('pset',''),
                    item.get('tau',''), item.get('label',''),
                    item.get('n',''), '', '', '', '', '', '', '',
                ])

    print(f'\n  结果已保存到: {summary_csv}')


if __name__ == '__main__':
    main()
