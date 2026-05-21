#!/usr/bin/env python3
"""
通过 Fortran Expect.f90 接口计算 PINN 波函数的 E/A.

流程:
  1. ddrhf_init()          → 初始化核素 (生成 Woods-Saxon 基底)
  2. ddrhf_set_wf(PINN)    → 用 PINN 神经网络波函数覆盖 Fortran 内部波函数
  3. fortran_occup()       → 从新波函数计算占据数
  4. fortran_densit()      → 计算密度 (标量/矢量/张量)
  5. fortran_potel()       → 计算 Hartree 平均场 (直接 + Fock 交换, 完整16项泛函)
  6. fortran_expect()      → 计算总能量 (Expect.f90 Line 262-271)
  7. ddrhf_get_energy()    → 提取 E/A 和完整能量分解

关键: 步骤 3~6 等价于 Fortran 自洽循环中的一步迭代,
      但跳过 Detgff/Dirac (不更新波函数), 只算能量.
      这样得到的就是 PINN 波函数在完整 DDRHF 泛函下的真实结合能.

用法:
    conda activate torch_env
    python fortran_ea_bridge.py                    # 全部核素 × PKA1
    python fortran_ea_bridge.py --pset PKO3        # 指定参数集
    python fortran_ea_bridge.py --nucleus 16O      # 单个核素
"""

import os, sys, json, glob, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'Core-1204'))

from rhf_ctypes import (
    ddrhf_init, ddrhf_set_wf, ddrhf_get_energy, ddrhf_set_energies,
    compute_ea_from_wf, ddrhf_extract_grid,
)

# ════════════════════════════════════════════════════════
#   配置
# ════════════════════════════════════════════════════════

PSET_MAP = {
    'PKA1': 0, 'PKO1': 1, 'PKO2': 2, 'PKO3': 3,
    'DDME1': 4, 'DDME2': 5, 'PKDD': 6, 'TW99': 7, 'DDLZ1': 8,
}

NUCLEI = {
    '16O':  (16, 8),   '22O':  (22, 8),  '24O':  (24, 8),
    '40Ca': (40, 20),  '48Ca': (48, 20), '52Ca': (52, 20),
    '60Ca': (60, 20),  '56Ni': (56, 28), '68Ni': (68, 28),
    '78Ni': (78, 28),  '124Sn':(124,50), '132Sn':(132,50),
    '208Pb':(208,82),  '210Pb':(210,82),
}

MSD = 2000  # 最大网格点数 (与Fortran一致)


# ════════════════════════════════════════════════════════
#   数据加载
# ════════════════════════════════════════════════════════

def load_pinn_wavefunctions(bulk_json_path):
    """
    从 bulk.json + wavefunction.json 文件加载PINN波函数数据.
    
    返回:
        info: dict {A, Z, nucleus, pset}
        states: list[dict] 每个 dict 含:
            tau ('n'/'p'), kappa, occupation, label,
            G (np.ndarray MSD), F (np.ndarray MSD), E_pinn, it (1=n,2=p)
    """
    with open(bulk_json_path) as f:
        bulk = json.load(f)
    
    info = {
        'nucleus': bulk['nucleus'],
        'A': bulk['A'],
        'Z': bulk['Z'],
        'pset': bulk['pset'],
    }
    
    states = []
    base_dir = os.path.dirname(bulk_json_path)
    
    for res in bulk['results']:
        wf_file = res.get('wf_file', '')
        if not os.path.exists(wf_file):
            # 尝试相对路径
            wf_file = os.path.join(base_dir,
                os.path.basename(wf_file))
        
        if not os.path.exists(wf_file):
            print(f"  WARN: missing wf file for {res['state']}, skipping")
            continue
        
        with open(wf_file) as f:
            wf = json.load(f)
        
        G = np.array(wf['G'], dtype=np.float64)
        F = np.array(wf['F'], dtype=np.float64)
        
        # 补齐到 MSD 长度
        G_pad = np.zeros(MSD, dtype=np.float64)
        F_pad = np.zeros(MSD, dtype=np.float64)
        npt = min(len(G), MSD)
        G_pad[:npt] = G[:npt]
        F_pad[:npt] = F[:npt]
        
        # r网格对齐检查
        r_w = np.array(wf['r'])
        
        itype = 1 if res['tau'] == 'n' else 2
        
        states.append({
            'tau': res['tau'],
            'kappa': res['kappa'],
            'occupation': res.get('occupation', 1.0),
            'label': res['label'],
            'G': G_pad,
            'F': F_pad,
            'E_pinn': res.get('E_pinn', res.get('E_ref', 0.0)),
            'E_ref': res.get('E_ref', 0.0),
            'it': itype,
            'r_grid': r_w,
            'wf_len': len(G),
            'name_safe': os.path.basename(wf_file).replace('_wavefunction.json',''),
        })
    
    return info, states


# ════════════════════════════════════════════════════════
#   核心计算
# ════════════════════════════════════════════════════════

def compute_fortran_ea(nucleus, pset='PKA1', verbose=True):
    """
    用Fortran接口计算单个核素的PINN E/A.
    
    参数:
        nucleus: str, 核素名 (如 '16O')
        pset: str, 参数集 (如 'PKA1')
        verbose: bool, 是否打印详情
    
    返回:
        result: dict 包含所有能量分解和对比信息
    """
    if nucleus not in NUCLEI:
        return None
    
    A, Z = NUCLEI[nucleus]
    N = A - Z
    ipa = PSET_MAP.get(pset, 0)
    
    # ---- 加载PINN数据 ----
    bulk_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'outputs', f'batch_{pset}', nucleus, f'{nucleus}_bulk.json'
    )
    
    if not os.path.exists(bulk_path):
        if verbose:
            print(f"  SKIP: {bulk_path} 不存在")
        return None
    
    info, states = load_pinn_wavefunctions(bulk_path)
    n_states = len(states)
    
    if n_states == 0:
        if verbose:
            print(f"  SKIP: 无有效波函数")
        return None
    
    t0 = time.time()
    
    # ---- Step 1: 初始化Fortran ----
    ddrhf_init(ipa, Z, N, A)
    
    xr, npt_fort, dr_fort = ddrhf_extract_grid()
    
    # ---- Step 2: 准备并设置PINN波函数 ----
    kappas   = [s['kappa'] for s in states]
    occ_arr  = [s['occupation'] for s in states]
    itypes   = [s['it'] for s in states]
    
    # 构建 MSD x n_states 数组 (Fortran列主序)
    G_all = np.column_stack([s['G'] for s in states])  # (MSD, n_wf)
    F_all = np.column_stack([s['F'] for s in states])
    
    ddrhf_set_wf(n_states, kappas, occ_arr, itypes, G_all, F_all)
    
    # 设置PINN本征值 (关键: epart=Σvv·ε·μ 需要用正确的ε值)
    pinn_energies = [s['E_pinn'] for s in states]
    ddrhf_set_energies(n_states, kappas, itypes, pinn_energies)
    
    # ---- Step 3: 用Fortran计算完整能量 ----
    energy = compute_ea_from_wf()
    
    elapsed = time.time() - t0
    
    # ---- 计算epart/PINN用于对比 ----
    epart_pinn = sum(
        s['occupation'] * s['E_pinn'] * (abs(s['kappa'])*2 + 1) / max(abs(s['kappa'])*2+1, 1)
        for s in states
    )
    # 更正: degeneracy mu = 2j+1 = |2*|kappa|| (注意 kappa<0 => j=l+1/2, kappa>0 => j=l-1/2)
    epart_pinn_corrected = 0.0
    for s in states:
        ka = abs(s['kappa'])
        if s['kappa'] < 0:
            j = ka - 0.5  # kappa = -(j+1/2) => j = |kappa| - 1/2 ... 不对
        else:
            j = ka - 0.5
        # 正确公式: kappa = ±(j+1/2), degeneracy = 2j+1
        # |kappa| = j + 1/2 => j = |kappa| - 1/2, mu = 2j+1 = 2|kappa|-1
        mu = 2 * abs(s['kappa']) - 1
        epart_pinn_corrected += s['occupation'] * s['E_pinn'] * mu
    
    # ---- 构建返回结果 ----
    result = {
        'nucleus': nucleus,
        'pset': pset,
        'A': A, 'Z': Z, 'N': N,
        'n_states': n_states,
        # Fortran Expect 结果
        'F_E_total': energy['e_total'],
        'F_E_per_A': energy['e_per_A'],     # ← 这才是真正的E/A!
        'F_E_kin':   energy['e_kin'],
        'F_E_dir':   energy['e_dir'],
        'F_E_exc':   energy['e_exc'],
        'F_E_rearr': energy['e_rearr'],
        # PINN epart (平均单粒子能加权和)
        'P_epart':   epart_pinn_corrected,
        'P_epart_A': epart_pinn_corrected / A if A > 0 else 0,
        # 差值
        'd_EA_vs_epart': energy['e_per_A'] - epart_pinn_corrected/A,
        'time_s': elapsed,
    }
    
    if verbose:
        _print_result(result)
    
    return result


def _print_result(r):
    """打印单个核素的结果."""
    sep = '-' * 72
    print(f"  {r['nucleus']:6s} ({r['pset']})  A={r['A']:3d} Z={r['Z']:2d} N={r['N']:2d}  [{r['n_states']} orbits]")
    print(f"  {'─'*72}")
    print(f"  Fortran E/A (Expect.f90):     {r['F_E_per_A']:>10.4f} MeV")
    print(f"  Fortran E_total:             {r['F_E_total']:>10.4f} MeV")
    print(f"  ┌─ Energy breakdown ─────────────────────────────┐")
    print(f"  │  Kinetic:    {r['F_E_kin']:>12.4f} MeV                 │")
    print(f"  │  Direct:     {r['F_E_dir']:>12.4f} MeV                 │")
    print(f"  │  Exchange:   {r['F_E_exc']:>12.4f} MeV                 │")
    print(f"  │  Rearrange:  {r['F_E_rearr']:>12.4f} MeV                 │")
    print(f"  └────────────────────────────────────────────────┘")
    print(f"  PINN epart/A (Σdeg·ε_i)/A:   {r['P_epart_A']:>10.4f} MeV")
    print(f"  Δ(E/A − epart/A):            {r['d_EA_vs_epart']:>+10.4f} MeV")
    print(f"  [computed in {r['time_s']:.2f}s]")
    print()


# ════════════════════════════════════════════════════════
#   批量处理 & 主入口
# ════════════════════════════════════════════════════════

def run_batch(nuclei_list=None, pset='PKA1'):
    """批量计算多个核素."""
    results = []
    
    targets = nuclei_list if nuclei_list else sorted(NUCLEI.keys())
    
    print("=" * 76)
    print(f"  Fortran EA Bridge — 通过 Expect.f90 计算 PINN E/A")
    print(f"  PSet={pset}, 核素数={len(targets)}")
    print("=" * 76)
    print()
    
    for nuc in targets:
        try:
            r = compute_fortran_ea(nuc, pset=pset, verbose=True)
            if r:
                results.append(r)
        except Exception as e:
            print(f"  ERROR processing {nuc}: {e}")
            import traceback; traceback.print_exc()
            print()
    
    # 汇总表
    _print_summary_table(results)
    return results


def _print_summary_table(results):
    """打印汇总表和CSV."""
    if not results:
        print("  无结果.")
        return
    
    print()
    print("=" * 115)
    print(f"  {'Nucleus':8s} {'A':>3s} {'Z':>2s} |"
          f" {'F_E/A':>9s} {'F_Etot':>11s} |"
          f" {'P_ep/A':>9s} {'Δ':>8s} |"
          f" {'Ekin':>10s} {'Edir':>10s} {'Eex':>10s} |"
          f" {'t(s)':>5s}")
    print(f"  {'-'*8} {'-'*3} {'-'*2} |"
          f" {'-'*9} {'-'*11} |"
          f" {'-'*9} {'-'*8} |"
          f" {'-'*10} {'-'*10} {'-'*10} |"
          f" {'-'*5}")
    
    for r in results:
        print(f"  {r['nucleus']:8s} {r['A']:3d} {r['Z']:2d} |"
              f" {r['F_E_per_A']:>+9.4f} {r['F_E_total']:>+11.4f} |"
              f" {r['P_epart_A']:>+9.4f} {r['d_EA_vs_epart']:>+8.4f} |"
              f" {r['F_E_kin']:>+10.4f} {r['F_E_dir']:>+10.4f} {r['F_E_exc']:>+10.4f} |"
              f" {r['time_s']:5.2f}")
    
    print("=" * 115)
    
    # 保存CSV
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'fortran_ea_bridge_results.csv')
    
    with open(csv_path, 'w') as f:
        f.write(','.join([
            'Nucleus','A','Z','N','PSet','n_states',
            'F_E_total','F_E_per_A','F_E_kin','F_E_dir','F_E_exc','F_E_rearr',
            'P_epart','P_epart_A','d_EA_vs_epart','time_s'
        ]) + '\n')
        for r in results:
            f.write(','.join(str(r[k]) for k in [
                'nucleus','A','Z','N','pset','n_states',
                'F_E_total','F_E_per_A','F_E_kin','F_E_dir','F_E_exc','F_E_rearr',
                'P_epart','P_epart_A','d_EA_vs_epart','time_s'
            ]) + '\n')
    
    print(f"\n  CSV saved to: {csv_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='用Fortran接口计算PINN的E/A')
    parser.add_argument('--pset', default='PKA1', help='参数集 (PKA1/PKO1/PKO2/PKO3)')
    parser.add_argument('--nucleus', nargs='+', default=None, help='指定核素 (默认全部)')
    args = parser.parse_args()
    
    run_batch(nuclei_list=args.nucleus, pset=args.pset)
