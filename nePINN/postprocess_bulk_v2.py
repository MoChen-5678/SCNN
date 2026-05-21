#!/usr/bin/env python3
"""
后处理脚本 v2: 按照 Fortran Expect.f90 逻辑精确求解 E/A (结合能每核子).

核心改进:
  1. 从 .PKA1/.PKOx 文件的最后迭代 Expect 块解析完整能量分解
     (epart, ekin, dsig, dome, ..., ecou, ertn, ervt, epio, era, ecm, epai)
  2. 按 Expect.f90 Line 262-271 公式复现 E/A:
       ekin  = epart - 2*(Edirect + Eexchange + Erearr)
       etot  = ekin + Edirect + Eexchange + Erearr + Ecoul + Epair + Ecm
       E/A   = etot / A
  3. 同时输出 PINN 的 epart (= Σ occ×degeneracy×ε_i) 作为对比
  4. 明确区分 "Particle Energy/A" 与 "Binding Energy/A"

用法:
    cd /home/ubuntu/rhf/plusPINN
    conda activate torch_env
    python postprocess_bulk_v2.py
"""

import os, sys, re, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_solve_all_nuclei import (
    NUCLEI_INFO, PARAM_SETS,
    parse_fortran_summary, RESULTS_BASE, PROJECT_ROOT,
)


# ════════════════════════════════════════════════════════
#   Fortran Expect 完整能量分解解析
# ════════════════════════════════════════════════════════

def parse_fortran_expect(pka1_path):
    """从 .PKA1/.PKOx 文件的最后一个 Expect 块解析全部能量分量.

    返回 dict 或 None.
    结构:
        {
            'iteration': int,
            'particle_number': {'n': float, 'p': float, 'tot': float},
            'rms': {'n': float, 'p': float, 'tot': float},      # w/ CoM
            'rms_nocom': {'n': float, 'p': float, 'tot': float},
            'rch': float,      # charge radius w/ CoM
            'rcc': float,      # charge radius w/o CoM
            'fermi': {'n': float, 'p': float},
            # --- 能量分量 (neutron, proton, total) ---
            'epart':  {},      # Particle Energy = Σ vv·ee·mu
            'ekin':   {},      # Kinetic Energy (Dirac-derived)
            'ekin2':  {},      # Kinetic Energy (wavefunction derivative, 用于检验收敛)
            # Direct 项
            'dsig': {}, 'dome': {}, 'drho': {}, 'dcou': {}, 'drtn': {}, 'drvt': {},
            # Exchange 项
            'esig': {}, 'eome': {}, 'erho': {}, 'ecou': {},
            'ertn': {}, 'ervt': {}, 'epio': {},
            # 其他
            'era':  {},        # Rearrangement
            'epai': {},        # Pairing
            'ecom': {},        # CoM correction
            # 合计量
            'etot': {},        # Total Energy
            'ea': float,       # E/A = etot(0)/A
        }
    """
    if not os.path.exists(pka1_path):
        return None

    with open(pka1_path) as f:
        content = f.read()

    # 找所有 Expect 块，取最后一个
    expect_blocks = re.findall(
        r'\*+\s*Begin Expect\s*\*+(.*?)\*+\s*End Expect\s*\*+',
        content, re.DOTALL
    )
    if not expect_blocks:
        return None
    last_block = expect_blocks[-1]

    d = {}
    lines = last_block.strip().split('\n')

    def _parse_3col(line_pattern, key, conv=float):
        """匹配 '  label    val_n    val_p    val_tot' 格式."""
        for line in lines:
            m = re.match(line_pattern, line.strip())
            if m:
                d[key] = {'n': conv(m.group(1)), 'p': conv(m.group(2)), 'tot': conv(m.group(3))}
                return True
        return False

    def _parse_1col(line_pattern, key, conv=float):
        for line in lines:
            m = re.match(line_pattern, line.strip())
            if m:
                d[key] = conv(m.group(1))
                return True
        return False

    # 粒子数
    _parse_3col(r'particle number\s+(\S+)\s+(\S+)\s+(\S+)', 'particle_number')

    # RMS半径
    _parse_3col(r'rms-Radius w/o CoM\s+(\S+)\s+(\S+)\s+(\S+)', 'rms_nocom')
    _parse_3col(r'rms-Radius w/ CoM\s+(\S+)\s+(\S+)\s+(\S+)', 'rms')

    # 电荷半径
    _parse_1col(r'charge-Radius w/ CoM\s+(\S+)', 'rch')
    _parse_1col(r'charge-Radius w/o CoM\s+(\S+)', 'rcc')

    # 费米能 (只有 n, p 两列)
    for line in lines:
        m = re.match(r'Fermi Energy\s+(\S+)\s+(\S+)', line.strip())
        if m:
            d['fermi'] = {'n': float(m.group(1)), 'p': float(m.group(2))}
            break

    # --- 能量 ---
    _parse_3col(r'Particle Energy\s+(\S+)\s+(\S+)\s+(\S+)', 'epart')
    _parse_3col(r'Kinetic Energy\s+(\S+)\s+(\S+)\s+(\S+)', 'ekin')

    # 第二个 Kinetic Energy 行 → ekin2
    found_kin1 = False
    for line in lines:
        m = re.match(r'Kinetic Energy\s+(\S+)\s+(\S+)\s+(\S+)', line.strip())
        if m:
            if not found_kin1:
                found_kin1 = True
            else:
                d['ekin2'] = {'n': float(m.group(1)), 'p': float(m.group(2)), 'tot': float(m.group(3))}
                break

    # Direct 项
    _parse_3col(r'Direct E-sigma\s+(\S+)\s+(\S+)\s+(\S+)', 'dsig')
    _parse_3col(r'Direct E-omega\s+(\S+)\s+(\S+)\s+(\S+)', 'dome')
    _parse_3col(r'Direct E-rho\(V\)\s+(\S+)\s+(\S+)\s+(\S+)', 'drho')
    _parse_3col(r'Direct E-rho\(T\)\s+(\S+)\s+(\S+)\s+(\S+)', 'drtn')
    _parse_3col(r'Direct E-rho\(VT\)\s+(\S+)\s+(\S+)\s+(\S+)', 'drvt')
    _parse_3col(r'Direct Coulomb\s+(\S+)\s+(\S+)\s+(\S+)', 'dcou')

    # Exchange 项
    _parse_3col(r'Exchange E-sigma\s+(\S+)\s+(\S+)\s+(\S+)', 'esig')
    _parse_3col(r'Exchange E-omega\s+(\S+)\s+(\S+)\s+(\S+)', 'eome')
    _parse_3col(r'Exchange E-rho\(V\)\s+(\S+)\s+(\S+)\s+(\S+)', 'erho')
    _parse_3col(r'Exchange E-rho\(T\)\s+(\S+)\s+(\S+)\s+(\S+)', 'ertn')
    _parse_3col(r'Exchange E-rho\(VT\)\s+(\S+)\s+(\S+)\s+(\S+)', 'ervt')
    _parse_3col(r'Exchange Coulomb\s+(\S+)\s+(\S+)\s+(\S+)', 'ecou')
    _parse_3col(r'Exchange E-pion\s+(\S+)\s+(\S+)\s+(\S+)', 'epio')

    # Rearrangement
    _parse_3col(r'Rearrangement\s+(\S+)\s+(\S+)\s+(\S+)', 'era')

    # Pairing
    _parse_3col(r'Pairing Energy\s+(\S+)\s*\s*(\S+)\s+(\S+)', 'epai')

    # E-cm
    _parse_3col(r'E-cm\s+(\S+)\s+(\S+)\s+(\S+)', 'ecom')

    # Total Energy (两行: 第一行有n/p/tot, 第二行只有total=enl)
    for line in lines:
        m = re.match(r'Total Energy\s+(\S+)\s+(\S+)\s+(\S+)', line.strip())
        if m:
            d['etot'] = {'n': float(m.group(1)), 'p': float(m.group(2)), 'tot': float(m.group(3))}
            break
    # 第二个 Total Energy (enl, 用于检验)
    for line in lines:
        m = re.match(r'Total Energy\s+(\S+)', line.strip())
        if m and 'etot' in d:
            d['enl'] = float(m.group(1))
            break

    # Energy per Particle
    _parse_1col(r'Energy per Particle\s+(\S+)', 'ea')

    # iteration number: 取该 Expect 块之前的 si 行
    # 从整个文件找最后一个 si 行
    si_match = None
    for line in content.split('\n'):
        m = re.search(r'^\s*(\d+)\s+si\s*=', line)
        if m:
            si_match = m
    if si_match:
        d['iteration'] = int(si_match.group(1))

    # 验证: 用 Fortran 公式重新计算 etot 并与解析值比对
    if all(k in d for k in ['epart', 'dsig', 'dome', 'drho', 'dcou', 'drtn', 'drvt',
                              'esig', 'eome', 'erho', 'ecou', 'ertn', 'ervt', 'epio',
                              'era', 'epai', 'ecom']):
        try:
            edirect_tot = sum(d[k]['tot'] for k in ['dsig','dome','drho','dcou','drtn','drvt'])
            exch_tot   = sum(d[k]['tot'] for k in ['esig','eome','erho','ecou','ertn','ervt','epio'])
            era_tot    = d['era']['tot']
            epai_tot   = d['epai']['tot']
            ecom_tot   = d['ecom']['tot']
            coul_tot   = d.get('dcou', {}).get('tot', 0)

            # Expect.f90 Line 263: ekin = epart - 2*(direct + exchange + rearrangement)
            # 注意: era 已在此公式内, 后续 etot 求和时不要再加 era!
            # 注意: dcou/ecou 已在 direct/exchange 列表中, 不要重复加!
            ekin_calc = d['epart']['tot'] - 2*(edirect_tot + exch_tot + era_tot)

            # Expect.f90 Line 267-268: etot = ekin + dir + exch + coul + pair + ecom
            # 其中 dir 包含 dcou, exch 包含 ecou, 所以不需要再加 coul_tot!
            # 重要: era 不在这里出现(已通过ekin公式隐含)!
            etot_calc = (ekin_calc + edirect_tot + exch_tot +
                        epai_tot + ecom_tot)

            d['_verify'] = {
                'edirect_tot': edirect_tot,
                'exch_tot': exch_tot,
                'era_tot': era_tot,
                'coul_tot': coul_tot,
                'pair_tot': epai_tot,
                'ecom_tot': ecom_tot,
                'ekin_calc': ekin_calc,
                'etot_calc': etot_calc,
                'etot_parsed': d['etot']['tot'],
                'diff_etot': etot_calc - d['etot']['tot'],
                'ea_calc': etot_calc / d['particle_number']['tot'] if d['particle_number']['tot'] > 0 else 0,
                'ea_parsed': d.get('ea', 0),
            }
        except Exception as e:
            d['_verify_error'] = str(e)

    return d if len(d) > 3 else None


# ════════════════════════════════════════════════════════
#   PINN 数据读取
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

        parts = fname.replace('_wavefunction.json', '').split('_')
        if len(parts) >= 3:
            tau = parts[1]
            label = '_'.join(parts[2:])
            label = label.replace('_', '/', 1)
        else:
            continue

        states.append({
            'tau': tau,
            'label': label,
            'E_pinn': float(d.get('E_PINN', 0)),
            'E_ref': float(d.get('E_Ref', 0)),
        })
    return states


def compute_pinn_epart(states_list, A, Z):
    """
    计算 PINN 的 Particle Energy (epart) 和相关量.

    注意: 这里的 epart_PINN = Σ (2j+1)_i × ε_i_PINN
          对应 Fortran 的 epart = Σ vv(i) × ee(i) × mu(i)

    返回 dict 包含:
      - epart_tot: 总 particle energy
      - epart_per_A: 平均单粒子能量 (≠ 结合能!)
      - Ef_n, Ef_p: 费米能 (最后占据态)
      - N_n, N_p, N_total: 态数目
      - details: 每态详情列表
    """
    from dirac_matrix_vs_pinn import parse_kappa_from_label

    n_states = [s for s in states_list if s['tau'] == 'n']
    p_states = [s for s in states_list if s['tau'] == 'p']

    epart_n = 0.0
    epart_p = 0.0
    details = []

    for s in states_list:
        try:
            kappa = parse_kappa_from_label(s['label'])
            degeneracy = 2 * abs(kappa)  # 2j+1 = 2|kappa|
        except Exception:
            degeneracy = 2

        contrib = degeneracy * s['E_pinn']
        if s['tau'] == 'n':
            epart_n += contrib
        else:
            epart_p += contrib

        details.append({
            'tau': s['tau'], 'label': s['label'],
            'kappa': kappa, 'deg': degeneracy,
            'eps': s['E_pinn'], 'contrib': contrib,
        })

    epart_tot = epart_n + epart_p
    epart_per_A = epart_tot / A if A > 0 else 0

    n_sorted = sorted(n_states, key=lambda x: x['E_pinn'])
    p_sorted = sorted(p_states, key=lambda x: x['E_pinn'])
    Ef_n = n_sorted[-1]['E_pinn'] if n_sorted else 0
    Ef_p = p_sorted[-1]['E_pinn'] if p_sorted else 0

    return {
        'epart_tot': epart_tot,
        'epart_n': epart_n,
        'epart_p': epart_p,
        'epart_per_A': epart_per_A,
        'Ef_n': Ef_n,
        'Ef_p': Ef_p,
        'N_n': len(n_states),
        'N_p': len(p_states),
        'N_total': len(states_list),
        'details': details,
    }


# ════════════════════════════════════════════════════════
#   主流程
# ════════════════════════════════════════════════════════

def main():
    print('=' * 120)
    print('  POST-PROCESS v2: E/A per Fortran Expect.f90 Logic')
    print(f'  Parameter sets: {PARAM_SETS}')
    print(f'  Nuclei ({len(NUCLEI_INFO)}): {list(NUCLEI_INFO.keys())}')
    print('=' * 120)

    # 1. 解析所有 Fortran Expect 数据
    print('\n[1/4] Parsing Fortran Expect data from PKx files...')
    fort_all = {}
    for nucleus in NUCLEI_INFO:
        A, Z, sym = NUCLEI_INFO[nucleus]
        sym_A = f'{sym}{A}'
        fort_all[nucleus] = {}
        for pset in PARAM_SETS:
            pset_dir = os.path.join(RESULTS_BASE, pset, nucleus)
            pk_file = os.path.join(pset_dir, f'{sym_A}.{pset}')
            exp = parse_fortran_expect(pk_file)
            if exp:
                fort_all[nucleus][pset] = exp
                v = exp.get('_verify', {})
                ea = exp.get('ea', 0)
                print(f'  OK {nucleus:>6s}/{pset}: iter={exp.get("iteration","?"):>3d}  '
                      f'E/A={ea:>9.4f}  epart={exp.get("epart",{}).get("tot",0):>12.3f}  '
                      f'etot={exp.get("etot",{}).get("tot",0):>12.3f}'
                      f'  verify_diff={v.get("diff_etot", "?"):.2e}' if isinstance(v.get("diff_etot"), float) else '')
            else:
                print(f'  -- {nucleus:>6s}/{pset}: no data')

    # 2. 遍历 核素×参数集
    rows = []
    grand_csv = os.path.join(PROJECT_ROOT, 'outputs', 'bulk_comparison_v2.csv')

    print('\n[2/4] Computing PINN vs Fortran comparison...')
    header = [
        'Nucleus', 'A', 'Z', 'PSet',
        # Fortran 值 (来自 Expect 块)
        'F_EA(MeV)',           # E/A = etot/A  (真正的结合能)
        'F_epart/A',           # Particle Energy/A (单粒子能级加权和/A)
        'F_Ef_n', 'F_Ef_p',
        'F_R(fm)', 'F_Rc(fm)',
        'F_etot',              # 总能量
        'F_epart',             # 粒子能量总和
        # PINN 值
        'P_epart/A',           # PINN 单粒子能级加权和/A
        'P_epart_tot',         # PINN epart 总和
        'P_Ef_n', 'P_Ef_p',
        # 差异
        'd_epart/A(MeV)',     # P_epart/A - F_epart/A
        'dEf_n(MeV)', 'dEf_p(MeV)',
        'N_n', 'N_p', 'N_st',
    ]

    for nucleus in sorted(NUCLEI_INFO.keys()):
        A, Z, sym = NUCLEI_INFO[nucleus]
        for pset in PARAM_SETS:
            fort_exp = fort_all.get(nucleus, {}).get(pset)
            pinn_states = read_pinn_results(nucleus, pset)

            if not fort_exp and not pinn_states:
                continue

            # Fortran 数据
            fe = fort_exp or {}
            F_ea     = fe.get('ea')
            F_epart  = fe.get('epart', {}).get('tot')
            F_epartA = F_epart / A if (F_epart and A > 0) else None
            F_efn    = fe.get('fermi', {}).get('n')
            F_efp    = fe.get('fermi', {}).get('p')
            F_r      = fe.get('rms', {}).get('tot')
            F_rc     = fe.get('rch')
            F_etot   = fe.get('etot', {}).get('tot')

            # PINN 数据
            if pinn_states:
                pb = compute_pinn_epart(pinn_states, A, Z)
                P_epartA = pb['epart_per_A']
                P_epart  = pb['epart_tot']
                P_efn    = pb['Ef_n']
                P_efp    = pb['Ef_p']
                N_n      = pb['N_n']
                N_p      = pb['N_p']
                N_st     = pb['N_total']
            else:
                P_epartA = P_epart = P_efn = P_efp = None
                N_n = N_p = N_st = 0

            row = [
                nucleus, A, Z, pset,
                f'{F_ea:.4f}' if F_ea is not None else '-',
                f'{F_epartA:.4f}' if F_epartA is not None else '-',
                f'{F_efn:.3f}' if F_efn is not None else '-',
                f'{F_efp:.3f}' if F_efp is not None else '-',
                f'{F_r:.4f}' if F_r is not None else '-',
                f'{F_rc:.4f}' if F_rc is not None else '-',
                f'{F_etot:.3f}' if F_etot is not None else '-',
                f'{F_epart:.3f}' if F_epart is not None else '-',
                f'{P_epartA:.4f}' if P_epartA is not None else '-',
                f'{P_epart:.3f}' if P_epart is not None else '-',
                f'{P_efn:.3f}' if P_efn is not None else '-',
                f'{P_efp:.3f}' if P_efp is not None else '-',
                f'{P_epartA - F_epartA:.4f}' if (P_epartA is not None and F_epartA is not None) else '-',
                f'{P_efn - F_efn:.3f}' if (P_efn is not None and F_efn is not None) else '-',
                f'{P_efp - F_efp:.3f}' if (P_efp is not None and F_efp is not None) else '-',
                N_n, N_p, N_st,
            ]
            rows.append(row)

            # 写入单个核素详情
            _write_detail_csv(nucleus, pset, A, Z, fe, pb if pinn_states else None)

    # 3. 写总表 CSV
    print(f'\n[3/4] Writing {grand_csv} ...')
    with open(grand_csv, 'w') as f:
        f.write(','.join(header) + '\n')
        for row in rows:
            f.write(','.join(str(x) for x in row) + '\n')

    # 4. 打印汇总表
    print(f'\n[4/4] Summary Table:')
    _print_summary(rows)

    print(f'\n{"="*120}')
    print(f'  TOTAL: {len(rows)} entries -> {grand_csv}')

    # 打印物理意义说明
    _print_physics_note()

    return grand_csv


def _write_detail_csv(nucleus, pset, A, Z, fort_exp, pinn_bulk):
    """写入每个核素的详细能量分解CSV."""
    output_dir = os.path.join(PROJECT_ROOT, 'outputs', f'batch_{pset}', nucleus)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f'{nucleus}_bulk_detail.csv')

    with open(csv_path, 'w') as f:
        f.write(f'# {nucleus} {pset} Energy Decomposition (Fortran Expect.f90)\n')
        f.write('#\n')

        if fort_exp:
            v = fort_exp.get('_verify', {})
            f.write('# === Fortran Verification ===\n')
            f.write(f'# etot(parsed)  : {fort_exp.get("etot",{}).get("tot","?")}\n')
            f.write(f'# etot(calc)     : {v.get("etot_calc","?")}\n')
            f.write(f"# diff           : {v.get('diff_etot','?')}\n")
            f.write(f'# E/A(parsed)    : {fort_exp.get("ea","?")}\n')
            f.write(f'# E/A(calc)      : {v.get("ea_calc","?")}\n')
            f.write('#\n')

            f.write('Quantity,F_N(MeV),F_P(MeV),F_TOT(MeV),P_PINN(MeV/TOT),Diff(MeV)\n')

            items = [
                ('Particle Energy (epart)', 'epart'),
                ('Kinetic Energy (ekin)',    'ekin'),
                ('Direct Sigma',             'dsig'),
                ('Direct Omega',             'dome'),
                ('Direct rho(V)',            'drho'),
                ('Direct rho(T)',            'drtn'),
                ('Direct rho(VT)',           'drvt'),
                ('Direct Coulomb',           'dcou'),
                ('Exchange Sigma',           'esig'),
                ('Exchange Omega',           'eome'),
                ('Exchange rho(V)',          'erho'),
                ('Exchange rho(T)',          'ertn'),
                ('Exchange rho(VT)',         'ervt'),
                ('Exchange Coulomb',         'ecou'),
                ('Exchange Pion',            'epio'),
                ('Rearrangement',            'era'),
                ('Pairing',                  'epai'),
                ('E_cm',                     'ecom'),
                ('Total Energy (etot)',      'etot'),
            ]

            for label, key in items:
                val = fort_exp.get(key, {})
                fn = val.get('n', '') if isinstance(val, dict) else ''
                fp = val.get('p', '') if isinstance(val, dict) else ''
                ft = val.get('tot', '') if isinstance(val, dict) else ''

                # PINN 只有 epart 可比
                if pinn_bulk and key == 'epart':
                    pp = f"{pinn_bulk['epart_tot']:.3f}"
                    dd = f"{pinn_bulk['epart_tot'] - ft:.3f}" if isinstance(ft, (int,float)) else ''
                elif pinn_bulk and key == 'etot':
                    pp = '(needs potential)'
                    dd = ''
                else:
                    pp = ''
                    dd = ''

                fs_n = f'{fn:.4f}' if isinstance(fn, (int,float)) else str(fn)
                fs_p = f'{fp:.4f}' if isinstance(fp, (int,float)) else str(fp)
                fs_t = f'{ft:.4f}' if isinstance(ft, (int,float)) else str(ft)
                f.write(f'{label},{fs_n},{fs_p},{fs_t},{pp},{dd}\n')

            # E/A 行
            ea = fort_exp.get('ea')
            f.write(f'E/A (binding),,,,,{ea:.4f}' if isinstance(ea,(int,float)) else 'E/A (binding),,,,-\n')
            if pinn_bulk:
                f.write(f'epart/A (mean SP),,,,{pinn_bulk["epart_per_A"]:.4f},\n')


def _print_summary(rows):
    """打印终端汇总表."""
    sep = '-' * 120
    # row index: 0:Nuc,1:A,2:Z,3:PSet,4:F_EA,5:F_ep/A,6:F_Efn,7:F_Efp,8:R,9:Rc,
    #            10:F_etot,11:F_epart,12:P_ep/A,13:P_ep_tot,14:P_Efn,15:P_Efp,
    #            16:d_ep/A,17:d_Efn,18:d_Efp,19:Nn,20:Np,21:Ns
    print(f'  {"Nucleus":8s} {"PSet":6s} {"A":>3s} {"Z":>3s} |'
          f' {"F_E/A":>8s} {"F_ep/A":>8s} | {"P_ep/A":>8s} {"d_ep/A":>8s} |'
          f' {"F_Efn":>7s} {"P_Efn":>7s} {"d_Efn":>7s} |'
          f' {"F_Efp":>7s} {"P_Efp":>7s} {"d_Efp":>7s} |'
          f' {"R":>6s} {"Rc":>6s} | {"Ns":>3s}')
    print(f'  {sep}')

    for r in rows:
        print(f'  {str(r[0]):8s} {str(r[3]):6s} {str(r[1]):>3s} {str(r[2]):>3s} |'
              f' {str(r[4]):>8s} {str(r[5]):>8s} | {str(r[12]):>8s} {str(r[16]):>8s} |'
              f' {str(r[6]):>7s} {str(r[14]):>7s} {str(r[17]):>7s} |'
              f' {str(r[7]):>7s} {str(r[15]):>7s} {str(r[18]):>7s} |'
              f' {str(r[8]):>6s} {str(r[9]):>6s} | {str(r[21]):>3s}')

    print(f'  {sep}')


def _print_physics_note():
    """打印物理意义说明."""
    note = """
  ┌─────────────────────────────────────────────────────────────────────┐
  │  PHYSICS NOTE: Fortran E/A vs PINN epart/A                          │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │  Fortran E/A (Binding Energy per nucleon):                          │
  │    E/A = etot / A                                                    │
  │    where etot = ekin + Edir + Eexch + Erearr + Ecoul + Epair + Ecm  │
  │          ekin = epart - 2*(Edir+Eexch+Erearr)   [from Dirac eq]     │
  │                                                                     │
  │  ⇒ E/A = (epart - Edir - Eexch - Erearr + Ecoul + Epair + Ecm) / A  │
  │                                                                     │
  │  The difference between epart/A and E/A:                            │
  │    epart/A - E/A ≈ (Edir + Eexch + Erearr - Ecoul - Epair - Ecm)/A  │
  │              ≈ 12~15 MeV for medium/heavy nuclei                   │
  │                                                                     │
  │  PINN provides single-particle eigenvalues ε_i only.               │
  │  To get true binding energy E/A, one needs the full self-consistent │
  │  potential (direct+exchange terms) from RHF iteration.              │
  │                                                                     │
  │  Columns explained:                                                 │
  │    F_EA     : Fortran binding energy/nucleon (etot/A)  ← TRUE E/A   │
  │    F_epart/A: Fortran particle energy/nucleon (Σvv·ee·mu / A)       │
  │    P_epart/A: PINN mean single-particle energy (Σdeg·ε_i / A)      │
  │    d_epart/A: PINN-Fortran difference in particle energy            │
  └─────────────────────────────────────────────────────────────────────┘
"""
    print(note)


if __name__ == '__main__':
    main()
