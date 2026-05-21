#!/usr/bin/env python3
"""
后处理脚本: 汇总全部 4参数集×14核素 的宏观量对比 (PINN vs Fortran).

功能:
  1. 从 .PKA1/.PKOx 文件读取 Fortran 收敛值 (E/A, Ef_n, Ef_p, R, Rc)
  2. 从 wavefunction.json 读取 PINN 单粒子能量
  3. 计算: E/A = Σ(occ_i × ε_i) / A,  Ef_n/p = 最后占据态能量
  4. 输出: 宏观量汇总 CSV + 各核素 bulk.csv 更新

用法:
    cd /home/ubuntu/rhf/plusPINN
    source activate torch_env
    python postprocess_bulk.py
"""

import os, sys, re, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_solve_all_nuclei import (
    NUCLEI_INFO, PARAM_SETS,
    parse_fortran_summary, RESULTS_BASE, PROJECT_ROOT,
)

# ════════════════════════════════════════════════════════
#   Fortran 数据读取
# ════════════════════════════════════════════════════════

def read_fortran_all():
    """读取所有核素×参数集的 Fortran 宏观量."""
    fort_data = {}
    for nucleus in NUCLEI_INFO:
        A, Z, sym = NUCLEI_INFO[nucleus]
        sym_A = f'{sym}{A}'
        fort_data[nucleus] = {}
        for pset in PARAM_SETS:
            pset_dir = os.path.join(RESULTS_BASE, pset, nucleus)
            pka1_file = os.path.join(pset_dir, f'{sym_A}.{pset}')
            result = parse_fortran_summary(pka1_file)
            if result:
                fort_data[nucleus][pset] = result
                # 也检查 FINAL 目录中的 ALL 文件 (可能有更精确的收敛值)
                final_dir = os.path.join(pset_dir, 'FINAL')
                final_file = os.path.join(final_dir, f'{sym_A}_ALL.{result["iteration"]:03d}.final000')
                if os.path.exists(final_file):
                    pass  # 已有数据即可
            else:
                print(f'  WARN: No Fortran data for {nucleus}/{pset}')
    return fort_data


# ════════════════════════════════════════════════════════
#   PINN 数据读取 (从 wavefunction.json)
# ════════════════════════════════════════════════════════

def read_pinn_results(nucleus, pset):
    """从 outputs 目录读取该核素所有已求解态的能量."""
    output_dir = os.path.join(PROJECT_ROOT, 'outputs', f'batch_{pset}', nucleus)
    if not os.path.isdir(output_dir):
        return []

    states = []
    for fname in os.listdir(output_dir):
        if not fname.endswith('_wavefunction.json'):
            continue
        fpath = os.path.join(output_dir, fname)
        try:
            with open(fpath) as f:
                d = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        # 从文件名解析 tau 和 label: {nuc}_{tau}_{label}_wavefunction.json
        parts = fname.replace('_wavefunction.json', '').split('_')
        if len(parts) >= 3:
            tau = parts[1]  # 'n' or 'p'
            label = '_'.join(parts[2:])  # e.g., '1s1_2'
            label = label.replace('_', '/', 1)  # '1s1/2'
        else:
            continue

        states.append({
            'tau': tau,
            'label': label,
            'E_pinn': float(d.get('E_PINN', 0)),
            'E_ref': float(d.get('E_Ref', 0)),
        })
    return states


def compute_pinn_bulk(states_list, A, Z):
    """从单粒子能级计算 PINN 宏观量.

    核心公式:
      E/A = Σ(2j+1)_i × ε_i / A      (每个态的简并度×单粒子能量)
      Ef_n = 最后一个中子占据态能量
      Ef_p = 最后一个质子占据态能量

    其中 2j+1 从 label 解析: kappa → j=|κ|-1/2 → degeneracy=2j+1=2|κ|
    """
    n_states = [s for s in states_list if s['tau'] == 'n']
    p_states = [s for s in states_list if s['tau'] == 'p']

    # 总结合能: 每个占据轨道贡献 (2j+1) × ε_i (简并度从kappa计算)
    from batch_solve_all_nuclei import parse_kappa_from_label as _pk
    E_total = 0.0
    for s in states_list:
        try:
            kappa = _pk(s['label'])
            degeneracy = 2 * abs(kappa)  # 2j+1 = 2|kappa|
        except Exception:
            degeneracy = 2
        E_total += degeneracy * s['E_pinn']
    E_per_A = E_total / A if A > 0 else 0

    # 费米能 = 最后一个占据态能量
    n_sorted = sorted(n_states, key=lambda x: x['E_pinn'])
    p_sorted = sorted(p_states, key=lambda x: x['E_pinn'])
    Ef_n = n_sorted[-1]['E_pinn'] if n_sorted else 0
    Ef_p = p_sorted[-1]['E_pinn'] if p_sorted else 0

    return {
        'E_per_A': E_per_A,
        'Ef_n': Ef_n,
        'Ef_p': Ef_p,
        'N_n': len(n_states),
        'N_p': len(p_states),
        'N_total': len(states_list),
    }


# ════════════════════════════════════════════════════════
#   主流程
# ════════════════════════════════════════════════════════

def main():
    print('=' * 100)
    print('  POST-PROCESS: Bulk Properties Comparison (PINN vs Fortran)')
    print(f'  Parameter sets: {PARAM_SETS}')
    print(f'  Nuclei ({len(NUCLEI_INFO)}): {list(NUCLEI_INFO.keys())}')
    print('=' * 100)

    # 1. 读取所有 Fortran 数据
    print('\n[1/3] Reading Fortran data...')
    fort_all = read_fortran_all()
    n_fort = sum(len(v) for v in fort_all.values())
    print(f'  Loaded: {n_fort} Fortran entries')

    # 2. 遍历所有 核素×参数集
    rows = []  # 汇总表行
    grand_csv_path = os.path.join(PROJECT_ROOT, 'outputs', 'bulk_comparison_all.csv')

    print('\n[2/3] Computing PINN bulk properties...')

    header = [
        'Nucleus', 'A', 'Z', 'PSet',
        'E/A_Fort', 'E/A_PINN', 'dE/A(MeV)',
        'Ef_n_Fort', 'Ef_n_PINN', 'dEf_n(MeV)',
        'Ef_p_Fort', 'Ef_p_PINN', 'dEf_p(MeV)',
        'R_Fort(fm)', 'Rc_Fort(fm)',
        'N_n', 'N_p', 'N_states'
    ]

    for nucleus in sorted(NUCLEI_INFO.keys()):
        A, Z, sym = NUCLEI_INFO[nucleus]
        for pset in PARAM_SETS:
            fort = fort_all.get(nucleus, {}).get(pset)

            # 读取 PINN 结果
            pinn_states = read_pinn_results(nucleus, pset)

            if not pinn_states and not fort:
                continue  # 无数据跳过

            if pinn_states:
                pinn_bulk = compute_pinn_bulk(pinn_states, A, Z)
            else:
                pinn_bulk = {'E_per_A': None, 'Ef_n': None, 'Ef_p': None,
                             'N_n': 0, 'N_p': 0, 'N_total': 0}

            ea_f  = fort['E_A'] if fort else None
            ea_p  = pinn_bulk['E_per_A']
            efn_f = fort.get('Ef_n') if fort else None
            efn_p = pinn_bulk['Ef_n']
            efp_f = fort.get('Ef_p') if fort else None
            efp_p = pinn_bulk['Ef_p']
            r_f   = fort.get('R') if fort else None
            rc_f  = fort.get('Rc') if fort else None

            row = [
                nucleus, A, Z, pset,
                f'{ea_f:.4f}' if ea_f is not None else '-',
                f'{ea_p:.4f}' if ea_p is not None else '-',
                f'{ea_p - ea_f:.4f}' if (ea_p is not None and ea_f is not None) else '-',
                f'{efn_f:.3f}' if efn_f is not None else '-',
                f'{efn_p:.3f}' if efn_p is not None else '-',
                f'{efn_p - efn_f:.3f}' if (efn_p is not None and efn_f is not None) else '-',
                f'{efp_f:.3f}' if efp_f is not None else '-',
                f'{efp_p:.3f}' if efp_p is not None else '-',
                f'{efp_p - efp_f:.3f}' if (efp_p is not None and efp_f is not None) else '-',
                f'{r_f:.4f}' if r_f is not None else '-',
                f'{rc_f:.4f}' if rc_f is not None else '-',
                pinn_bulk['N_n'], pinn_bulk['N_p'], pinn_bulk['N_total'],
            ]
            rows.append(row)

            # 同时更新单个核素的 bulk.csv
            _update_single_bulk(nucleus, pset, A, Z, fort, pinn_bulk)

    # 3. 写总表
    print(f'\n[3/3] Writing {grand_csv_path} ...')

    with open(grand_csv_path, 'w') as f:
        f.write(','.join(header) + '\n')
        for row in rows:
            f.write(','.join(str(x) for x in row) + '\n')

    print(f'\n{"="*100}')
    print(f'  TOTAL: {len(rows)} entries written to {grand_csv_path}')

    # 打印汇总摘要
    _print_summary_table(rows)

    return grand_csv_path


def _update_single_bulk(nucleus, pset, A, Z, fort, pinn_bulk):
    """更新单个核素目录下的 bulk.csv."""
    output_dir = os.path.join(PROJECT_ROOT, 'outputs', f'batch_{pset}', nucleus)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f'{nucleus}_bulk.csv')

    fb = fort or {}
    pb = pinn_bulk

    with open(csv_path, 'w') as f:
        f.write('Nucleus,PSet,A,Z,Quantity,Fortran,PINN,Diff\n')
        quantities = [
            ('E/A (MeV)', fb.get('E_A'), pb.get('E_per_A')),
            ('Ef_n (MeV)', fb.get('Ef_n'), pb.get('Ef_n')),
            ('Ef_p (MeV)', fb.get('Ef_p'), pb.get('Ef_p')),
            ('R (fm)',     fb.get('R'),     None),
            ('Rc (fm)',    fb.get('Rc'),    None),
            ('N_states',  None,             pb.get('N_total')),
        ]
        for qname, fv, pv in quantities:
            diff = pv - fv if (fv is not None and pv is not None) else None
            fs = f'{fv:.6f}' if fv is not None else '-'
            ps = f'{pv:.6f}' if pv is not None else '-'
            ds = f'{diff:.6f}' if diff is not None else '-'
            f.write(f'{nucleus},{pset},{A},{Z},{qname},{fs},{ps},{ds}\n')


def _print_summary_table(rows):
    """打印终端汇总表."""
    print(f'\n{"="*110}')
    print(f'  {"Nucleus":8s} {"PSet":6s} {"A":>3s} {"Z":>3s} '
          f'| {"E/A_Fort":>9s} {"E/A_PINN":>9s} {"dE/A":>8s} '
          f'| {"Ef_nF":>8s} {"Ef_nP":>8s} {"dEf_n":>7s} '
          f'| {"Ef_pF":>8s} {"Ef_pP":>8s} {"dEf_p":>7s}'
          f'| {"R_F":>7s} {"Rc_F":>7s} | {"Ns":>4s}')
    print(f'  {"-"*108}')

    for r in rows:
        print(f'  {r[0]:8s} {r[3]:6s} {r[1]:3d} {r[2]:3d} '
              f'| {r[4]:>9s} {r[5]:>9s} {r[6]:>8s} '
              f'| {r[7]:>8s} {r[8]:>8s} {r[9]:>7s} '
              f'| {r[10]:>8s} {r[11]:>8s} {r[12]:>7s}'
              f'| {r[13]:>7s} {r[14]:>7s} | {r[17]:>4d}')

    print(f'  {"="*110}')


if __name__ == '__main__':
    main()
