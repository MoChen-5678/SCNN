#!/usr/bin/env python3
"""
Run the non-boundary-state ablation plan through dirac_matrix_vs_pinn.py.

The script keeps the same solve path as the batch workflow: every state uses
its own POT file, lower same-(l,j) states are solved first, and the adjacent
lower model is used as the transfer-learning checkpoint for the target state.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from dirac_matrix_vs_pinn import scan_pot_files


ABLATIONS = [
    {
        "name": "baseline",
        "description": "learn c, no phi, hard G endpoints, soft BC off",
        "args": [],
    },
    {
        "name": "no_hard_endpoint_no_soft_bc",
        "description": "plan item 1: hard G endpoint off, soft BC off",
        "args": ["--no-hard-g-endpoint", "--w-bc", "0"],
    },
    {
        "name": "phase_hard_endpoint",
        "description": "plan items 2-3: phi on, hard G endpoint on",
        "args": ["--boundary-phase"],
    },
    {
        "name": "phase_no_hard_endpoint",
        "description": "plan items 2-3: phi on, hard G endpoint off",
        "args": ["--boundary-phase", "--no-hard-g-endpoint"],
    },
    {
        "name": "phase_constrained",
        "description": "plan item 4: phi on with cos(k*c*R+phi)=0 constraint",
        "args": ["--boundary-phase", "--phase-constraint-weight", "1"],
    },
    {
        "name": "fixed_c_phase_constrained",
        "description": "plan item 5: c fixed to 1 with constrained phi",
        "args": ["--fixed-c", "--boundary-phase", "--phase-constraint-weight", "1"],
    },
]


def state_label_from_lev_name(name):
    label = name[2:]
    return re.sub(r"^(\d+)([a-z])\.(\d+/\d+)$", r"\1\2\3", label)


def load_lev_states(lev_dir):
    states = []
    if not lev_dir or not os.path.isdir(lev_dir):
        return states
    for fname in sorted(os.listdir(lev_dir)):
        if ".psp-" not in fname:
            continue
        path = os.path.join(lev_dir, fname)
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 6 or not parts[0].isdigit():
                    continue
                name = parts[1]
                tau = "p" if name.startswith("P.") else "n"
                states.append({
                    "lev_file": path,
                    "index": int(parts[0]),
                    "state_name": name,
                    "label": state_label_from_lev_name(name),
                    "tau": tau,
                    "mu": int(float(parts[2])),
                    "vv": float(parts[3]),
                    "E": float(parts[4]),
                })
    return states


def infer_lev_dir(pot_dir):
    parent = os.path.dirname(os.path.abspath(pot_dir))
    candidate = os.path.join(parent, "LEV")
    return candidate if os.path.isdir(candidate) else None


def merge_pot_and_lev(pot_dir, lev_dir):
    pot_states = scan_pot_files(pot_dir)
    lev_states = load_lev_states(lev_dir)
    by_key = {(s["tau"], s["label"]): s for s in pot_states}
    merged = []
    for lev in lev_states:
        pot = by_key.get((lev["tau"], lev["label"]))
        if not pot:
            continue
        item = {**pot, **lev, "pot_file": pot["pot_file"], "pot_header_E": pot["E"]}
        merged.append(item)
    if merged:
        return merged
    return pot_states


def choose_target(pot_dir, lev_dir=None, state=None, tau=None):
    states = merge_pot_and_lev(pot_dir, lev_dir)
    if state:
        matches = [s for s in states if s["label"] == state and (tau is None or s["tau"] == tau)]
    else:
        # plan.md asks for vv != 1 non-bound states; prefer partially occupied
        # states over completely empty continuum states.
        matches = [
            s for s in states
            if s["E"] is not None and s["E"] > 0 and abs(s.get("vv", s.get("occ", 1.0)) - 1.0) > 1e-8
        ]
        if tau:
            matches = [s for s in matches if s["tau"] == tau]
        partial = [s for s in matches if s.get("vv", 0.0) > 0]
        matches = partial or matches
        matches = sorted(matches, key=lambda s: (s["E"], s["tau"], s["label"]))
    if not matches:
        raise SystemExit(f"No target state found in {pot_dir}: state={state!r} tau={tau!r}")
    return matches[0]


def parse_result(output):
    matches = re.findall(r"Result:.*?E_PINN=([+-]?\d+(?:\.\d+)?)\s+dE=([+-]?\d+(?:\.\d+)?)", output)
    if not matches:
        return {"E_pinn": None, "dE": None}
    match = matches[-1]
    return {"E_pinn": float(match[0]), "dE": float(match[1])}


def run_ablation(ablation, idx, total, target, args, output_root):
    run_dir = os.path.join(output_root, ablation["name"])
    os.makedirs(run_dir, exist_ok=True)
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "dirac_matrix_vs_pinn.py"),
        "--state", target["label"],
        "--tau", target["tau"],
        "--A", str(args.A),
        "--Z", str(args.Z),
        "--pot-file", target["pot_file"],
        "--pot-dir", args.pot_dir,
        "--E-init", str(target["E"]),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--output-dir", run_dir,
        "--no-live-plot",
        *ablation["args"],
    ]
    if args.force_lower:
        cmd.append("--force-lower")
    print(f"\n[{idx}/{total}] {ablation['name']}: {ablation['description']}", flush=True)
    started = time.time()
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    elapsed = time.time() - started
    log_path = os.path.join(run_dir, "run.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(proc.stdout)

    parsed = parse_result(proc.stdout)
    row = {
        "name": ablation["name"],
        "description": ablation["description"],
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "target": target,
        "output_dir": run_dir,
        "log": log_path,
        **parsed,
    }
    return idx, row


def main():
    parser = argparse.ArgumentParser(description="Run plan.md ablations through dirac_matrix_vs_pinn.py")
    parser.add_argument("--pot-dir", default="PKA1/208Pb/POT")
    parser.add_argument("--lev-dir", default=None)
    parser.add_argument("--state", default=None, help="Target state label, e.g. 3f5/2. Default: first E>0 state")
    parser.add_argument("--tau", choices=["n", "p"], default=None)
    parser.add_argument("--A", type=int, default=208)
    parser.add_argument("--Z", type=int, default=82)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--only", nargs="*", default=None, help="Run only selected ablation names")
    parser.add_argument("--workers", type=int, default=1, help="Number of ablation subprocesses to run in parallel")
    parser.add_argument("--force-lower", action="store_true", help="Recompute automatic lower states instead of reusing existing outputs")
    args = parser.parse_args()

    lev_dir = args.lev_dir or infer_lev_dir(args.pot_dir)
    target = choose_target(args.pot_dir, lev_dir, args.state, args.tau)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_dir or os.path.join("outputs", f"plan_ablation_{target['tau']}_{target['label'].replace('/', '_')}_{timestamp}")
    os.makedirs(output_root, exist_ok=True)

    selected = [a for a in ABLATIONS if args.only is None or a["name"] in args.only]
    if not selected:
        raise SystemExit(f"No ablations selected: {args.only}")

    vv_text = f" vv={target['vv']:.6f}" if "vv" in target else ""
    print(f"Target: {target['state_name']} tau={target['tau']}{vv_text} E_ref={target['E']:+.6f} MeV")
    print(f"POT: {target['pot_file']}")
    if lev_dir:
        print(f"LEV: {lev_dir}")
    print(f"Output: {output_root}")
    n_workers = max(1, min(args.workers, len(selected)))
    print(f"Workers: requested={args.workers} active={n_workers}")

    results_by_idx = {}
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(run_ablation, ablation, idx, len(selected), target, args, output_root)
            for idx, ablation in enumerate(selected, 1)
        ]
        for future in as_completed(futures):
            idx, row = future.result()
            results_by_idx[idx] = row
            print(
                f"  done [{idx}/{len(selected)}] {row['name']}: "
                f"rc={row['returncode']} E_PINN={row['E_pinn']} dE={row['dE']} "
                f"elapsed={row['elapsed_s']:.1f}s",
                flush=True,
            )
    results = [results_by_idx[i] for i in sorted(results_by_idx)]

    ok = [r for r in results if r["returncode"] == 0 and r["dE"] is not None]
    best = min(ok, key=lambda r: r["dE"]) if ok else None
    summary = {
        "created_at": timestamp,
        "target": target,
        "epochs": args.epochs,
        "lr": args.lr,
        "workers_requested": args.workers,
        "workers_active": n_workers,
        "best": best,
        "results": results,
    }
    summary_path = os.path.join(output_root, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if best:
        print(f"\nBest: {best['name']} dE={best['dE']:.6f} MeV E_PINN={best['E_pinn']:+.6f}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
