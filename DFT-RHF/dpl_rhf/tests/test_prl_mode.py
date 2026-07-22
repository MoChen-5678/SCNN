from __future__ import annotations

import numpy as np
import torch
from torch import nn

from dpl_rhf.backends.base import NucleusCase
from dpl_rhf.backends.finite_difference import zhang_7point_adf_coefficients
from dpl_rhf.backends.fortran_rhf_stub import FortranRHFBackend
from dpl_rhf.backends.torch_rmf import TorchRMFBackend, noncentral_derivative_matrix
from dpl_rhf.cli.prl_train import build_parser
from dpl_rhf.functionals.pkdd_rmf import PKDDRMFFunctionalSpec
from dpl_rhf.functionals.pkdd_action import PKDDParameters, PKDDRMFFunctional
from dpl_rhf.models.hamiltonian_net import (
    DirectHamiltonianParameterization,
    LocalHamiltonianNet,
    compose_hamiltonian,
    physical_component_gradient,
)
from dpl_rhf.training.surrogate_gradient import (
    generalized_prl_gradient,
    surrogate_loss,
)


def test_surrogate_gradient_matches_backend_gradient():
    torch.set_default_dtype(torch.float64)
    hamiltonian = torch.randn(4, 16, requires_grad=True)
    reconstructed = torch.zeros_like(hamiltonian)
    grad_energy = torch.randn_like(hamiltonian)
    grad_h = generalized_prl_gradient(hamiltonian, reconstructed, grad_energy, 0.7)
    loss = surrogate_loss(hamiltonian, grad_h)
    (actual,) = torch.autograd.grad(loss, hamiltonian)
    expected = grad_h.detach() / hamiltonian.numel()
    assert torch.max(torch.abs(actual - expected)).item() < 1.0e-14


def test_prl_reconstruction_gradient_is_exact_hamiltonian_difference():
    hamiltonian = torch.arange(12, dtype=torch.float64).reshape(4, 3)
    reconstructed = torch.flip(hamiltonian, dims=(1,))
    gradient = generalized_prl_gradient(
        hamiltonian, reconstructed, torch.zeros_like(hamiltonian), 0.7
    )
    expected = 0.7 * (hamiltonian - reconstructed)
    assert torch.max(torch.abs(gradient - expected)).item() < 1.0e-14


def test_physical_component_gradient_matches_channel_assembly():
    components = torch.randn((4, 11), dtype=torch.float64, requires_grad=True)
    channel_gradient = torch.randn((4, 11), dtype=torch.float64)
    value = torch.sum(compose_hamiltonian(components) * channel_gradient)
    (actual,) = torch.autograd.grad(value, components)
    assert torch.max(torch.abs(actual - physical_component_gradient(channel_gradient))).item() < 1.0e-14


def test_7point_adf_moment_conditions():
    coeff = zhang_7point_adf_coefficients()
    offsets = np.arange(7, dtype=np.float64)
    for power in range(7):
        moment = float(np.sum(coeff * offsets**power))
        expected = 1.0 if power == 1 else 0.0
        assert abs(moment - expected) < 1.0e-10


def test_adf_uses_zero_exterior_forward_operator_and_adjoint_pair():
    r = torch.linspace(0.0, 2.0, 21, dtype=torch.float64)
    forward = noncentral_derivative_matrix(r, order=7)
    backend = TorchRMFBackend(NucleusCase("PKDD", 8, 8, 16), device=torch.device("cpu"), derivative_order=7)
    assert torch.count_nonzero(forward[-1, :-1]).item() == 0
    solver = backend.matrix_solver
    adjoint = solver.derivative_minus + solver.derivative_plus.T
    assert torch.max(torch.abs(adjoint)).item() < 1.0e-14


def test_dirac_adf_constructs_b1_b2_explicitly_with_parity_exchange():
    backend = TorchRMFBackend(NucleusCase("PKDD", 8, 8, 16), device=torch.device("cpu"), derivative_order=7)
    solver = backend.matrix_solver
    stack = torch.as_tensor(backend.initial_hamiltonian_np, dtype=torch.float64)
    size = solver.x.numel()
    # G behaves as r^(l+1): kappa=-1 (s1/2) has odd radial G, while
    # kappa=+1 (p1/2) has even radial G.
    for kappa, radial_g_is_odd in ((-1, True), (1, False)):
        matrix = solver.matrix(stack, 1.0, kappa)
        angular = torch.diag(torch.full_like(solver.x, float(kappa)) / solver.x)
        forward = solver.derivative_plus if radial_g_is_odd else solver.derivative_minus
        backward = solver.derivative_minus if radial_g_is_odd else solver.derivative_plus
        physical_b1 = -backward + angular
        physical_b2 = forward + angular
        assert torch.max(torch.abs(matrix[:size, size:] - physical_b1)).item() < 1.0e-14
        assert torch.max(torch.abs(matrix[size:, :size] - physical_b2)).item() < 1.0e-14
        assert torch.max(torch.abs(physical_b1 - physical_b2.T)).item() < 1.0e-14


def test_rhf_backend_is_reserved():
    try:
        FortranRHFBackend()
    except NotImplementedError as exc:
        assert "reserved" in str(exc)
    else:
        raise AssertionError("RHF backend should not be implemented in RMF PRL v1")


def test_prl_default_backend_is_differentiable_torch_rmf():
    args = build_parser().parse_args([])
    assert args.backend == "torch-rmf"
    assert args.energy_gradient_weight == 1.0
    assert args.lambda_reconstruct == 1.0e-3
    assert args.direct_order == 64
    assert args.derivative_order == 7
    assert args.mode == "direct"
    activation_action = next(action for action in build_parser()._actions if action.dest == "activation")
    assert activation_action.choices == ["silu"]


def test_torch_rmf_backend_gradient_is_autodiff_consistent():
    torch.set_default_dtype(torch.float64)
    backend = TorchRMFBackend(NucleusCase("PKDD", 8, 8, 16), device=torch.device("cpu"), derivative_order=7)
    hamiltonian = torch.as_tensor(backend.initial_hamiltonian_np, dtype=torch.float64).requires_grad_(True)
    result = backend.evaluate_tensor(hamiltonian)
    assert result.diagnostics["hermiticity_error"] < 2.0e-14
    assert result.diagnostics["gradient_norm"] > 0.0
    check = backend.gradient_check(hamiltonian.detach(), epsilon=1.0e-4)
    assert check["rel_error"] < 2.0e-7


def test_torch_rmf_orbitals_obey_regular_origin_boundary():
    torch.set_default_dtype(torch.float64)
    backend = TorchRMFBackend(NucleusCase("PKDD", 8, 8, 16), device=torch.device("cpu"), derivative_order=7)
    hamiltonian = torch.as_tensor(backend.initial_hamiltonian_np, dtype=torch.float64).requires_grad_(True)
    orbitals = backend.evaluate_tensor(hamiltonian).orbitals
    r = torch.as_tensor(backend.r, dtype=torch.float64)
    for i, kappa in enumerate(orbitals["kappa"]):
        l_upper, l_lower = backend.matrix_solver._component_angular_momenta(int(kappa))
        for values, angular in ((orbitals["G"][i], l_upper), (orbitals["F"][i], l_lower)):
            scaled = values[:3] / r[:3].pow(angular + 1)
            assert torch.isfinite(scaled).all()
            assert abs(float(values[-1])) < 1.0e-14


def test_kinetic_energy_uses_physically_normalized_orbitals():
    backend = TorchRMFBackend(NucleusCase("PKDD", 8, 8, 16), device=torch.device("cpu"), derivative_order=7)
    hamiltonian = torch.as_tensor(backend.initial_hamiltonian_np, dtype=torch.float64).requires_grad_(True)
    orbitals = backend.evaluate_tensor(hamiltonian).orbitals
    p = backend.functional.params
    for index in range(orbitals["G"].shape[0]):
        species = float(orbitals["species_sign"][index])
        kappa = int(orbitals["kappa"][index])
        zero_stack = torch.zeros_like(hamiltonian)
        zero_stack[1] = -2.0 * p.mass_n / p.hbar_c
        zero_stack[3] = -2.0 * p.mass_p / p.hbar_c
        free = backend.matrix_solver.matrix(zero_stack, species, kappa)
        interacting = backend.matrix_solver.matrix(hamiltonian, species, kappa)
        _, eigenvectors = torch.linalg.eigh(interacting)
        vector = eigenvectors[:, orbitals["eigen_index"][index]]
        expected = p.hbar_c * (vector @ free @ vector)
        assert abs(float(expected - orbitals["kinetic_expectation_mev"][index])) < 1.0e-10


def test_uniform_adf_metric_makes_matrix_and_density_bilinears_identical():
    backend = TorchRMFBackend(NucleusCase("PKDD", 8, 8, 16), device=torch.device("cpu"), derivative_order=7)
    hamiltonian = torch.as_tensor(backend.initial_hamiltonian_np, dtype=torch.float64).requires_grad_(True)
    result = backend.evaluate_tensor(hamiltonian)
    h = backend.functional.h
    assert torch.max(torch.abs(backend.functional.orbital_w[1:] - h)).item() < 1.0e-14
    for index in range(result.orbitals["G"].shape[0]):
        species = float(result.orbitals["species_sign"][index])
        kappa = int(result.orbitals["kappa"][index])
        offset = 0 if species > 0 else 2
        g = result.orbitals["G"][index, 1:-1]
        f = result.orbitals["F"][index, 1:-1]
        coefficients = torch.sqrt(torch.as_tensor(h)) * torch.cat([g, f])
        matrix = backend.matrix_solver.matrix(hamiltonian, species, kappa)
        matrix_expectation = coefficients @ matrix @ coefficients
        assert abs(float(torch.linalg.norm(coefficients) - 1.0)) < 1.0e-12
        assert abs(float(matrix_expectation * backend.functional.params.hbar_c - result.orbitals["epsilon"][index])) < 1.0e-9
        local_integral = h * torch.sum(
            g.square() * hamiltonian[offset, 1:-1]
            + f.square() * hamiltonian[offset + 1, 1:-1]
        )
        block_size = g.numel()
        local_matrix = (
            coefficients[:block_size].square() @ hamiltonian[offset, 1:-1]
            + coefficients[block_size:].square() @ hamiltonian[offset + 1, 1:-1]
        )
        assert abs(float(local_integral - local_matrix)) < 1.0e-12


def test_global_occupation_preserves_neutron_and_proton_numbers():
    backend = TorchRMFBackend(NucleusCase("PKDD", 8, 8, 16), device=torch.device("cpu"), derivative_order=7)
    hamiltonian = torch.as_tensor(backend.initial_hamiltonian_np, dtype=torch.float64).requires_grad_(True)
    orbitals = backend.evaluate_tensor(hamiltonian).orbitals
    particles = orbitals["occupancy"] * orbitals["degeneracy"]
    assert abs(float(particles[orbitals["species_sign"] > 0].sum()) - 8.0) < 1.0e-12
    assert abs(float(particles[orbitals["species_sign"] < 0].sum()) - 8.0) < 1.0e-12


def test_hamiltonian_network_hard_physical_boundaries():
    r = np.linspace(0.001, 20.0, 201)
    initial = np.zeros((4, r.size))
    network = LocalHamiltonianNet(r, 8, 8, initial)
    components = network.components()
    diagnostics = network.constraint_diagnostics(components)
    assert torch.max(torch.abs(components[:3, -1])).item() < 1.0e-14
    assert diagnostics["analytic_center_derivative_fm_minus_2"] == 0.0
    assert diagnostics["analytic_coulomb_robin_fm_minus_2"] == 0.0
    assert diagnostics["max_scalar_identity_mev"] < 1.0e-10
    direct = DirectHamiltonianParameterization(r, 8, 8, order=8)
    with torch.no_grad():
        direct.coefficients[0, 0] = 1.0
    scalar = direct.components()[0]
    assert abs(float(scalar[-1])) < 1.0e-14
    assert abs(float(scalar[-2] - scalar[-1])) > 1.0e-6


def test_n_equal_z_does_not_remove_isovector_variational_channel():
    r = np.linspace(0.001, 20.0, 201)
    direct = DirectHamiltonianParameterization(r, 8, 8, order=12)
    with torch.no_grad():
        direct.coefficients[2, 0] = 1.0
    isovector = direct.components()[2]
    assert torch.max(torch.abs(isovector)).item() > 0.01
    (gradient,) = torch.autograd.grad(isovector.square().sum(), direct.coefficients)
    assert torch.linalg.norm(gradient[2]).item() > 0.0


def test_representation_has_no_woods_saxon_hard_multiplier():
    r = np.linspace(0.001, 20.0, 201)
    direct = DirectHamiltonianParameterization(r, 8, 8, order=12)
    assert not hasattr(direct, "nuclear_shape")


def test_direct_parameterization_uses_same_physical_map():
    r = np.linspace(0.001, 20.0, 201)
    direct = DirectHamiltonianParameterization(r, 8, 8, order=12)
    assert direct().shape == (4, r.size)
    assert direct.components().requires_grad
    assert torch.max(torch.abs(direct.components()[:3, -1])).item() < 1.0e-14


def test_all_hamiltonian_hidden_activations_are_silu():
    r = np.linspace(0.001, 20.0, 201)
    network = LocalHamiltonianNet(r, 8, 8, np.zeros((4, r.size)))
    activations = [module for channel in network.networks for module in channel if not isinstance(module, nn.Linear)]
    assert activations
    assert all(isinstance(module, nn.SiLU) for module in activations)


def test_network_has_full_rank_variational_hamiltonian_path():
    r = np.linspace(0.001, 20.0, 201)
    network = LocalHamiltonianNet(r, 8, 8, np.zeros((4, r.size)))
    assert network.variational_coefficients.shape == (4, 64)
    with torch.no_grad():
        network.variational_coefficients[2, 0] = 1.0
    loss = network.components()[2].square().sum()
    (gradient,) = torch.autograd.grad(loss, network.variational_coefficients)
    assert torch.linalg.norm(gradient[2]).item() > 0.0


def test_complete_action_reduces_to_on_shell_energy():
    backend = TorchRMFBackend(NucleusCase("PKDD", 8, 8, 16), device=torch.device("cpu"), derivative_order=7)
    hamiltonian = torch.as_tensor(backend.initial_hamiltonian_np, dtype=torch.float64).requires_grad_(True)
    result = backend.evaluate_tensor(hamiltonian)
    assert abs(result.diagnostics["action_reduction_error_mev"]) < 1.0e-8
    assert result.diagnostics["max_orbital_norm_residual"] < 1.0e-12
    assert result.diagnostics["max_spectral_residual"] < 1.0e-10
    assert max(result.diagnostics[f"weak_el_{name}_rms"] for name in ("sigma", "omega", "rho", "coul")) < 1.0e-9
    assert abs(float(result.action_terms["action_orbital_constraints"])) < 1.0e-10


def test_pkdd_uses_distinct_neutron_and_proton_masses():
    backend = TorchRMFBackend(NucleusCase("PKDD", 8, 8, 16), device=torch.device("cpu"), derivative_order=7)
    p = backend.functional.params
    assert p.mass_n == 939.5731
    assert p.mass_p == 938.2796
    zeros = torch.zeros_like(torch.as_tensor(backend.r, dtype=torch.float64))
    potentials = backend.functional.potentials(
        rho_s_n=zeros, rho_s_p=zeros, rho_v_n=zeros, rho_v_p=zeros,
        sigma=zeros, omega=zeros, rho=zeros, coul=zeros,
    )
    assert torch.max(torch.abs(potentials["vms_n"] + 2.0 * p.mass_n / p.hbar_c)).item() < 1.0e-14
    assert torch.max(torch.abs(potentials["vms_p"] + 2.0 * p.mass_p / p.hbar_c)).item() < 1.0e-14
    expected_split = -2.0 * (p.mass_n - p.mass_p) / p.hbar_c
    actual_split = float((potentials["vms_n"] - potentials["vms_p"])[0])
    assert abs(actual_split - expected_split) < 1.0e-14


def test_pkdd_coupling_derivatives_generate_exact_rearrangement_inputs():
    r = torch.as_tensor(np.arange(201) * 0.1, dtype=torch.float64)
    r[0] = 0.001
    functional = PKDDRMFFunctional(r, 8, 8)
    rho_b = torch.linspace(1.0e-6, 0.16, r.numel(), dtype=torch.float64, requires_grad=True)
    couplings = functional.couplings(rho_b)
    for coupling, derivative in (("gsig", "dgsig"), ("gome", "dgome"), ("grho", "dgrho")):
        (actual,) = torch.autograd.grad(couplings[coupling].sum(), rho_b, retain_graph=True)
        expected = couplings[derivative] / functional.params.rho_sat
        assert torch.max(torch.abs(actual - expected)).item() < 1.0e-11


def test_local_isovector_density_sources_rho_field_even_for_n_equal_z():
    r = torch.as_tensor(np.arange(201) * 0.1, dtype=torch.float64)
    r[0] = 0.001
    functional = PKDDRMFFunctional(r, 8, 8)
    profile = 0.08 * torch.exp(-(r / 2.5).square())
    state = {
        "rho_s_n": profile,
        "rho_s_p": 0.9 * profile,
        "rho_v_n": profile,
        "rho_v_p": 0.9 * profile,
    }
    couplings = functional.couplings(state["rho_v_n"] + state["rho_v_p"])
    fields = functional.reconstruct_fields(state, couplings)
    assert torch.max(torch.abs(fields["rho"])).item() > 1.0e-6


def test_on_shell_field_action_derivative_reproduces_rearranged_potentials():
    r = torch.as_tensor(np.arange(201) * 0.1, dtype=torch.float64)
    r[0] = 0.001
    functional = PKDDRMFFunctional(r, 8, 8)
    base = 0.08 * torch.exp(-(r / 2.6).square())
    rho_s_n = (0.92 * base).clone().requires_grad_(True)
    rho_s_p = (0.86 * base).clone().requires_grad_(True)
    rho_v_n = base.clone().requires_grad_(True)
    rho_v_p = (0.9 * base).clone().requires_grad_(True)
    density = {
        "rho_s_n": rho_s_n, "rho_s_p": rho_s_p,
        "rho_v_n": rho_v_n, "rho_v_p": rho_v_p,
    }
    couplings = functional.couplings(rho_v_n + rho_v_p)
    fields = functional.reconstruct_fields(density, couplings)
    state = {**density, **fields}
    p = functional.params
    rho_s = rho_s_n + rho_s_p
    rho_b = rho_v_n + rho_v_p
    rho_3 = rho_v_n - rho_v_p
    energy = p.hbar_c * (
        0.5 * fields["sigma"] @ functional.field_full_kernels["sigma"] @ fields["sigma"]
        + torch.sum(functional.measure * couplings["gsig"] * rho_s * fields["sigma"])
        - 0.5 * fields["omega"] @ functional.field_full_kernels["omega"] @ fields["omega"]
        + torch.sum(functional.measure * couplings["gome"] * rho_b * fields["omega"])
        - 0.5 * fields["rho"] @ functional.field_full_kernels["rho"] @ fields["rho"]
        + torch.sum(functional.measure * couplings["grho"] * rho_3 * fields["rho"])
        - 0.5 * fields["coul"] @ functional.field_full_kernels["coul"] @ fields["coul"]
        + torch.sum(functional.measure * rho_v_p * fields["coul"])
    )
    gradients = torch.autograd.grad(energy, (rho_s_n, rho_s_p, rho_v_n, rho_v_p))
    potentials = functional.potentials(**state)
    interior = slice(2, -2)
    scale = p.hbar_c * functional.measure[interior]
    expected = (
        potentials["scalar_self_energy"],
        potentials["scalar_self_energy"],
        potentials["vector_isoscalar"] + potentials["vector_isovector"],
        potentials["vector_isoscalar"] - potentials["vector_isovector"] + potentials["vector_coulomb"],
    )
    for actual_gradient, expected_potential in zip(gradients, expected):
        actual = actual_gradient[interior] / scale
        assert torch.max(torch.abs(actual - expected_potential[interior])).item() < 2.0e-10


def test_torch_backend_constructs_core_mesh_without_fortran_runtime():
    backend = TorchRMFBackend(NucleusCase("PKDD", 8, 8, 16), device=torch.device("cpu"))
    assert backend.r.shape == (201,)
    assert backend.r[0] == 0.001
    assert abs(backend.r[-1] - 20.0) < 1.0e-14


def test_network_hamiltonian_obeys_book_channel_decomposition():
    p = PKDDParameters()
    r = np.linspace(0.001, 20.0, 201)
    initial = PKDDRMFFunctionalSpec(8, 8, r).initial_hamiltonian()
    network = LocalHamiltonianNet(r, 8, 8, initial)
    stack = network()
    scalar_n = 0.5 * (stack[0] - stack[1] - 2.0 * p.mass_n / p.hbar_c)
    scalar_p = 0.5 * (stack[2] - stack[3] - 2.0 * p.mass_p / p.hbar_c)
    assert torch.max(torch.abs(scalar_n - scalar_p)).item() < 1.0e-14
    vector_n = 0.5 * (stack[0] + stack[1] + 2.0 * p.mass_n / p.hbar_c)
    vector_p = 0.5 * (stack[2] + stack[3] + 2.0 * p.mass_p / p.hbar_c)
    assert torch.isfinite(vector_n).all() and torch.isfinite(vector_p).all()


if __name__ == "__main__":
    test_surrogate_gradient_matches_backend_gradient()
    print("PASS PRL surrogate gradient")
    test_7point_adf_moment_conditions()
    print("PASS 7-point ADF moment conditions")
    test_rhf_backend_is_reserved()
    print("PASS RHF reserved interface")
    test_prl_default_backend_is_differentiable_torch_rmf()
    print("PASS PRL trainer defaults to differentiable torch RMF")
    test_torch_rmf_backend_gradient_is_autodiff_consistent()
    print("PASS torch RMF autograd gradient check")
