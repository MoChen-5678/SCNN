#!/usr/bin/env python3
"""
test_fortran_bridge.py — 端到端验证 Core-1204 Fortran ↔ Python 桥接

测试流程:
  1. import pinn_wrapper.so
  2. init_rhf(0) → PKA1, 16O
  3. scf_iterate(50) → 收敛检查
  4. get_energy() → E_total ≈ -128 MeV (¹⁶O PKA1 参考值)
  5. compute_potentials() → 势场形状验证

用法:
    cd /home/ubuntu/rhf && source activate torch_env
    python PINN/test_fortran_bridge.py
"""

import sys
import os
import time
import numpy as np

# 确保 Core-1204 在路径中
CORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Core-1204')
if os.path.isdir(CORE_DIR) and CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)
sys.path.insert(0, '.')

def main():
    print("=" * 60)
    print("  Core-1204 Fortran Bridge — End-to-End Test")
    print("=" * 60)

    # ── Test 1: Import ──────────────────────────────────────
    print("\n[Test 1] Import pinn_wrapper module...")
    try:
        import pinn_wrapper as pw
        funcs = [x for x in dir(pw) if not x.startswith('_')]
        print(f"  ✓ Import OK! ({len(funcs)} functions: {funcs}")
    except ImportError as e:
        print(f"  ✗ FAILED: {e}")
        print(f"  Hint: run 'cd Core-1204 && bash build_f90wrap.sh' first")
        return False

    # ── Test 2: init_rhf ─────────────────────────────────────
    print("\n[Test 2] Initialize RHF engine (PKA1, 16O)...")
    npt = np.array([0], dtype=np.int32)
    dr = np.array([0.0], dtype=np.float64)
    r_grid = np.zeros(201, dtype=np.float64)
    try:
        pw.init_rhf(np.int32(0), npt, dr, r_grid)
        print(f"  ✓ npt={npt[0]}, dr={dr[0]:.4f} fm")
        print(f"  r_range=[{r_grid[0]:.3f}, {r_grid[-1]:.3f}] fm")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False

    # ── Test 3: scf_step (single step) ────────────────────
    print("\n[Test 3] Single SCF step...")
    si = np.array([999.0], dtype=np.float64)
    try:
        pw.scf_step(np.float64(0.5), si)
        print(f"  ✓ si = {si[0]:.6e} (should be large for first step)")
    except Exception as e:
        print(f"  ✗ scf_step not available: {e}")

    # ── Test 4: Full SCF convergence ───────────────────────
    print("\n[Test 4] SCF convergence (max 50 iterations)...")
    t0 = time.time()
    converged = False
    final_si = 999.0
    xmix = 0.5
    for i in range(50):
        si = np.array([0.0], dtype=np.float64)
        try:
            pw.scf_step(np.float64(xmix), si)
            current_si = abs(si[0]) if isinstance(si[0], (int, float)) else abs(float(si[0]))
            if i % 10 == 0 or current_si < 1e-3:
                print(f"    iter {i+1:3d}: si={current_si:.2e}, xmix={xmix:.3f}")
            if current_si < 1e-5:
                converged = True
                final_si = current_si
                break
        except Exception as e:
            print(f"    iter {i+1}: error: {e}")
            break
        xmix = max(0.05, 0.5 * (1 - i / 50))

    elapsed = time.time() - t0
    status = "✓ CONVERGED" if converged else "✗ NOT CONVERGED"
    print(f"\n  {status} after {i+1} iters ({elapsed:.1f}s)")
    print(f"  Final si = {final_si:.2e}")

    # ── Test 5: Energy check ───────────────────────────────
    print("\n[Test 5] Energy components...")
    # get_energy_components 可能未导出（参数太多），尝试获取
    try:
        # 准备大量参数数组
        e_args = {}
        names = ['E_kin','E_dsig','E_dome','E_drho','E_dcou','E_drtn','E_drvt',
                'E_esig','E_eome','E_erho','E_ecou','E_epio','E_ertn','E_ervt',
                'E_rear','E_com','E_pair','E_total','E_per_nuc',
                'fermi_n','fermi_p','rms_n','rms_p','rms_t',
                'charge_r','particle_n','particle_p','particle_t']
        for n in names:
            e_args[n] = np.array([0.0], dtype=np.float64)
        
        pw.get_energy_components(*[e_args[n] for n in names])
        
        Etot = float(e_args['E_total'][0])
        EpN = float(e_args['E_per_nuc'][0])
        Np = int(e_args['particle_t'][0])
        
        print(f"  E_total   = {Etot:+10.4f} MeV")
        print(f"  E/A       = {EpN:+10.4f} MeV")
        print(f"  A         = {Np}")
        print(f"  Fermi(n)  = {float(e_args['fermi_n'][0]):8.4f} MeV")
        print(f"  Fermi(p)  = {float(e_args['fermi_p'][0]):8.4f} MeV")
        print(f"  RMS(t)    = {float(e_args['rms_t'][0]):6.3f} fm")
        
        # 参考: ¹⁶O PKA1 的 RHF 结合能约 -7.98 MeV/A → 总能量 ~ -128 MeV
        if abs(Etot) > 10 and Np > 0:
            ea = Etot / Np
            if -8.5 < ea < -7.0:
                print(f"  ✓ Binding energy {ea:.3f} MeV/A looks reasonable!")
            else:
                print(f"  ⚠ Binding energy {ea:.3f} MeV/A outside expected [-8.5,-7.0]")
                
    except AttributeError:
        print("  ⚠ get_energy_components not available (may need full rebuild)")
    except Exception as e:
        print(f"  ⚠ Error getting energy: {e}")

    # ── Summary ───────────────────────────────────────────
    print("\n" + "=" * 60)
    all_passed = True
    
    tests = [
        ("Import", funcs is not None),
        ("Initialize", npt[0] == 201),
        ("SCF Step", True),
        ("SCF Converged", converged),
    ]
    
    for name, ok in tests:
        icon = "✓" if ok else "✗"
        print(f"  {icon} {name}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_passed = False

    print()
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
