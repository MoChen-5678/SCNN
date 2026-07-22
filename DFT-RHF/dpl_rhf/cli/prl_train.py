from __future__ import annotations

import argparse

from dpl_rhf.training.prl_hamiltonian_trainer import train_prl_hamiltonian


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PRL/AI2DFT-style local-Hamiltonian RMF trainer")
    parser.add_argument("--model", default="PKDD")
    parser.add_argument("--functional", choices=["rmf-pkdd"], default="rmf-pkdd")
    parser.add_argument("--mode", choices=["direct", "network"], default="direct")
    parser.add_argument("--direct-order", type=int, default=64)
    parser.add_argument("--direct-gate", default=None)
    parser.add_argument("--allow-unvalidated-network", action="store_true")
    parser.add_argument("--gate-residual", type=float, default=1.0e-3)
    parser.add_argument("--gate-energy-mev", type=float, default=0.5)
    parser.add_argument("--gate-radius-fm", type=float, default=0.1)
    parser.add_argument("--gate-profile-relative", type=float, default=0.1)
    parser.add_argument("--gate-level-mev", type=float, default=1.0)
    parser.add_argument("--gate-adf-mev", type=float, default=0.5)
    parser.add_argument("--checkpoint-policy", choices=["physics", "last"], default="physics")
    parser.add_argument("--z", type=int, default=8)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--a", type=int, default=None)
    parser.add_argument("--backend", choices=["torch-rmf", "fortran-fixed"], default="torch-rmf")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--activation", choices=["silu"], default="silu")
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--backtrack-threshold", type=float, default=2.0)
    parser.add_argument("--lr-decay", type=float, default=0.5)
    parser.add_argument("--max-backtracks", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--lambda-reconstruct", type=float, default=1.0e-3,
        help="PRL Hamiltonian reconstruction coefficient in MeV^-1",
    )
    parser.add_argument("--energy-gradient-weight", type=float, default=1.0)
    parser.add_argument("--derivative-order", type=int, choices=[1, 4, 5, 6, 7], default=7)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=20240623)
    parser.add_argument("--compare-scf", action="store_true")
    parser.add_argument("--out", default="outputs/prl_hamiltonian_rmf")
    return parser


def main() -> None:
    train_prl_hamiltonian(build_parser().parse_args())


if __name__ == "__main__":
    main()
