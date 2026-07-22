from __future__ import annotations

import math

import torch
from torch import nn

from dpl_rhf.functionals.pkdd_action import PKDDParameters


PHYSICAL_COMPONENT_NAMES = ("S", "V0", "V3", "VC")


def compose_hamiltonian(components: torch.Tensor, params: PKDDParameters | None = None) -> torch.Tensor:
    """Map independent RMF self-energies to shifted neutron/proton Dirac channels."""
    p = params or PKDDParameters()
    scalar, vector, isovector, coulomb = components
    return torch.stack([
        vector + scalar + isovector,
        vector - scalar + isovector - 2.0 * p.mass_n / p.hbar_c,
        vector + scalar - isovector + coulomb,
        vector - scalar - isovector + coulomb - 2.0 * p.mass_p / p.hbar_c,
    ])


def decompose_hamiltonian(
    stack: torch.Tensor, params: PKDDParameters | None = None, coulomb: torch.Tensor | None = None
) -> torch.Tensor:
    """Project shifted Dirac channels onto the four independent RMF coordinates."""
    p = params or PKDDParameters()
    scalar_n = 0.5 * (stack[0] - stack[1] - 2.0 * p.mass_n / p.hbar_c)
    scalar_p = 0.5 * (stack[2] - stack[3] - 2.0 * p.mass_p / p.hbar_c)
    vector_n = 0.5 * (stack[0] + stack[1] + 2.0 * p.mass_n / p.hbar_c)
    vector_p = 0.5 * (stack[2] + stack[3] + 2.0 * p.mass_p / p.hbar_c)
    scalar = 0.5 * (scalar_n + scalar_p)
    # vector_n=V0+V3 and vector_p=V0-V3+VC. VC is fixed by the
    # physical representation and is supplied separately when exact inversion
    # is required. For an arbitrary reconstructed H, use the proton-only
    # difference after taking the isoscalar average; this minimum-norm inverse
    # is used only for PRL residual coordinates.
    vc = torch.zeros_like(vector_n) if coulomb is None else coulomb
    vector = 0.5 * (vector_n + vector_p - vc)
    isovector = 0.5 * (vector_n - vector_p + vc)
    return torch.stack([scalar, vector, isovector, vc])


def physical_component_gradient(grad_h: torch.Tensor) -> torch.Tensor:
    """Pull dE/dH back through the exact RMF channel assembly."""
    return torch.stack([
        grad_h[0] - grad_h[1] + grad_h[2] - grad_h[3],
        grad_h.sum(dim=0),
        grad_h[0] + grad_h[1] - grad_h[2] - grad_h[3],
        grad_h[2] + grad_h[3],
    ])


class PhysicalHamiltonianRepresentation(nn.Module):
    """Hard spherical RMF map shared by direct and neural variation."""

    def __init__(self, r_grid, z: int, n: int, initial_hamiltonian=None):
        super().__init__()
        r = torch.as_tensor(r_grid, dtype=torch.float64)
        a = float(z + n)
        t = (r / r[-1]).square()
        self.register_buffer("r", r)
        self.register_buffer("radial_feature", 2.0 * t - 1.0)
        # Massive fields obey a Dirichlet condition at the finite box edge.
        # A linear zero imposes only the field value; a squared zero also
        # forces an unphysical Neumann condition and removes valid variations.
        self.register_buffer("massive_envelope", 1.0 - t)
        self.register_buffer("coulomb_envelope", (1.0 - t).square())
        self.z = int(z)
        self.a = int(a)

        p = PKDDParameters()
        charge_radius = 1.2496 * a ** (1.0 / 3.0)
        alpha_z = z / p.alpha_inv
        coulomb_base = torch.where(
            r < charge_radius,
            alpha_z * (3.0 / (2.0 * charge_radius) - r.square() / (2.0 * charge_radius**3)),
            alpha_z / r,
        )
        self.register_buffer("coulomb_base", coulomb_base)
        if initial_hamiltonian is None:
            from dpl_rhf.functionals.pkdd_rmf import PKDDRMFFunctionalSpec

            initial_hamiltonian = PKDDRMFFunctionalSpec(z, n, r.cpu().numpy()).initial_hamiltonian()
        initial_stack = torch.as_tensor(initial_hamiltonian, dtype=torch.float64)
        initial_components = decompose_hamiltonian(initial_stack, coulomb=coulomb_base)
        # The initial Woods-Saxon profile is a starting point only.  Project
        # its massive channels onto the exact finite-box boundary, while the
        # trainable correction below remains unrestricted in the interior.
        initial_components = initial_components.clone()
        initial_components[:3] *= self.massive_envelope
        initial_components[3] = coulomb_base
        self.register_buffer("initial_components", initial_components)

    def constrain(self, raw: torch.Tensor) -> torch.Tensor:
        if raw.shape != (4, self.r.numel()):
            raise ValueError(f"expected raw physical profiles (4,{self.r.numel()}), got {tuple(raw.shape)}")
        # These envelopes impose only exact spherical boundary conditions.
        # A Woods-Saxon profile and the global (N-Z)/A factor are initial
        # guesses, not consequences of the RMF Euler-Lagrange equations.
        massive_shape = self.massive_envelope
        scalar = massive_shape * raw[0]
        vector = massive_shape * raw[1]
        isovector = massive_shape * raw[2]
        coulomb = self.coulomb_base + self.coulomb_envelope * raw[3]
        correction = torch.stack([scalar, vector, isovector, coulomb - self.coulomb_base])
        return self.initial_components + correction

    def constraint_diagnostics(self, components: torch.Tensor) -> dict[str, float]:
        p = PKDDParameters()
        h = float(self.r[2] - self.r[1])
        first_interval_slopes = (components[:, 2] - components[:, 1]) / h
        massive_boundary = components[:3, -1]
        # The correction and its derivative vanish at R, so the analytic
        # Coulomb base obeys the exterior Robin condition exactly.
        coul_derivative = (components[3, -1] - components[3, -2]) / h
        discrete_robin = coul_derivative + components[3, -1] / self.r[-1]
        stack = compose_hamiltonian(components, p)
        scalar_n = 0.5 * (stack[0] - stack[1] - 2.0 * p.mass_n / p.hbar_c)
        scalar_p = 0.5 * (stack[2] - stack[3] - 2.0 * p.mass_p / p.hbar_c)
        return {
            "analytic_center_derivative_fm_minus_2": 0.0,
            "max_first_interval_slope_fm_minus_2": float(first_interval_slopes.abs().max().detach().cpu()),
            "max_massive_boundary_fm_minus_1": float(massive_boundary.abs().max().detach().cpu()),
            "analytic_coulomb_robin_fm_minus_2": 0.0,
            "discrete_coulomb_robin_fm_minus_2": float(discrete_robin.detach().cpu()),
            "max_scalar_identity_mev": float(((scalar_n - scalar_p).abs().max() * p.hbar_c).detach().cpu()),
        }


class LocalHamiltonianNet(PhysicalHamiltonianRepresentation):
    """PRL Hamiltonian network restricted to the spherical RMF coordinate space."""

    def __init__(
        self,
        r_grid,
        z: int,
        n: int,
        initial_hamiltonian,
        hidden: int = 96,
    ):
        super().__init__(r_grid, z, n, initial_hamiltonian)
        a = float(z + n)
        features = torch.stack([
            self.radial_feature,
            torch.full_like(self.r, z / a),
            torch.full_like(self.r, n / a),
            torch.full_like(self.r, a / 208.0),
        ], dim=-1)
        self.register_buffer("features", features)
        self.activation_name = "silu"
        self.networks = nn.ModuleList([self._mlp(hidden) for _ in range(4)])
        theta = torch.acos(self.radial_feature.clamp(-1.0, 1.0))
        self.register_buffer(
            "variational_basis",
            torch.stack([torch.cos(k * theta) for k in range(64)], dim=0),
        )
        self.variational_coefficients = nn.Parameter(torch.zeros((4, 64), dtype=torch.float64))
        self.reset_parameters()
        with torch.no_grad():
            initial_raw_output = torch.stack([
                network(self.features).squeeze(-1) for network in self.networks
            ])
        self.register_buffer("initial_raw_output", initial_raw_output)

    @staticmethod
    def _mlp(hidden: int) -> nn.Sequential:
        layers = [
            nn.Linear(4, hidden, dtype=torch.float64), nn.SiLU(),
            nn.Linear(hidden, hidden, dtype=torch.float64), nn.SiLU(),
            nn.Linear(hidden, hidden, dtype=torch.float64), nn.SiLU(),
        ]
        return nn.Sequential(*layers, nn.Linear(hidden, 1, dtype=torch.float64))

    def reset_parameters(self) -> None:
        for network in self.networks:
            for module in network.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        # Zero correction starts exactly from the analytic Woods-Saxon seed.

    def components(self) -> torch.Tensor:
        raw = torch.stack([network(self.features).squeeze(-1) for network in self.networks])
        raw = raw - self.initial_raw_output + self.variational_coefficients @ self.variational_basis
        return self.constrain(raw)

    def forward(self) -> torch.Tensor:
        return compose_hamiltonian(self.components())


class DirectHamiltonianParameterization(PhysicalHamiltonianRepresentation):
    """Network-free Chebyshev coordinates for validating variational RMF."""

    def __init__(self, r_grid, z: int, n: int, order: int = 32):
        super().__init__(r_grid, z, n)
        if order < 4:
            raise ValueError("direct physical basis needs order >= 4")
        theta = torch.acos(self.radial_feature.clamp(-1.0, 1.0))
        basis = torch.stack([torch.cos(k * theta) for k in range(order)], dim=0)
        self.register_buffer("basis", basis)
        coefficients = torch.zeros((4, order), dtype=torch.float64)
        self.coefficients = nn.Parameter(coefficients)

    def components(self) -> torch.Tensor:
        return self.constrain(self.coefficients @ self.basis)

    def forward(self) -> torch.Tensor:
        return compose_hamiltonian(self.components())
