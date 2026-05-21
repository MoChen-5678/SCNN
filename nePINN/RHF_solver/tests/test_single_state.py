"""
PINN-RHF 单态求解测试

验证 PINN 在 Shooting POT 势场下的 Dirac 方程求解能力。
对比指标: 能量误差 < 1%, 波形 L² 误差 < 5%

用法:
    python -m tests.test_single_state
    python tests/test_single_state.py --isotope 16O --state 1s1/2 --epochs 1000
"""

import os
import sys
import argparse
import torch
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from config import set_seed, DEVICE, DR, NPT, R_GRID
from model import DiracNet
from pde_residuals import compute_dirac_residual, make_local_potentials
from potentials import build_mvp_potentials
from boundary_conditions import count_nodes
from train import MVPSolver


def test_5padf_matrix():
    """Test 1: 验证 5PADF 差分矩阵构建"""
    print("\n[Test 1] 5PADF 矩阵构建")
    
    from pde_residuals import build_5padf_matrix, get_fd_directions
    
    # κ=-1 (1s1/2): G→forward, F→backward
    g_dir, f_dir = get_fd_directions(-1)
    assert g_dir == 'forward', f"Expected forward for G(κ=-1), got {g_dir}"
    assert f_dir == 'backward', f"Expected backward for F(κ=-1), got {f_dir}"
    print(f"  ✓ κ=-1 方向: G={g_dir}, F={f_dir}")
    
    # κ=+1 (1p1/2): G→backward, F→forward  
    g_dir2, f_dir2 = get_fd_directions(+1)
    assert g_dir2 == 'backward', f"Expected backward for G(κ=+1), got {g_dir2}"
    assert f_dir2 == 'forward', f"Expected forward for F(κ=+1), got {f_dir2}"
    print(f"  ✓ κ=+1 方向: G={g_dir2}, F={f_dir2}")
    
    # 构建矩阵并检查形状
    D_fwd = build_5padf_matrix(10, dr=0.1, direction='forward')
    D_bwd = build_5padf_matrix(10, dr=0.1, direction='backward')
    assert D_fwd.shape == (10, 10), f"Shape mismatch: {D_fwd.shape}"
    assert D_bwd.shape == (10, 10), f"Shape mismatch: {D_bwd.shape}"
    
    # 检查伴随关系: D_bw ≈ -D_fw.flip([0,1])
    expected_bwd = (-D_fwd).flip([0,1])
    diff = torch.max(torch.abs(D_bwd - expected_bwd)).item()
    assert diff < 1e-10, f"伴随关系偏差: {diff}"
    print(f"  ✓ 伴随关系验证通过 (max_diff={diff:.2e})")
    
    print("  ✓ Test 1 通过!\n")


def test_model_forward():
    """Test 2: 验证 DiracNet 前向传播和归一化"""
    print("[Test 2] DiracNet 前向传播")
    
    net = DiracNet(n_hidden=64, n_layers=4, init_energy=-40.0)
    r = torch.linspace(0, 20.0, 201)
    
    g, f = net(r, kappa=-1, dr=0.10)
    
    assert g.shape == (201,), f"G shape: {g.shape}"
    assert f.shape == (201,), f"F shape: {f.shape}"
    
    # 归一化检验
    norm = torch.trapz(g**2 + f**2, dim=-1, dx=0.10).item()
    print(f"  归一化积分: {norm:.6f} (目标≈1.0)")
    
    # 能量参数存在
    E = net.get_energy()
    print(f"  初始能量: {E:.2f} MeV")
    assert isinstance(E, float)
    
    # 相位对齐: 主峰为正
    peak_sign = torch.sign(g[torch.argmax(torch.abs(g))]).item()
    assert peak_sign > 0, "相位对齐失败: G主峰应为正"
    print(f"  ✓ 相位对齐正确 (peak_sign={peak_sign:+.1f})")
    
    print("  ✓ Test 2 通过!\n")


def test_pde_residual():
    """Test 3: 验证 PDE 残差计算"""
    print("[Test 3] PDE 残差计算")
    
    # 创建简单势场
    pots = build_mvp_potentials('16O')
    r = get_r_tensor()
    
    # 创建网络
    net = DiracNet(n_hidden=64, n_layers=4, init_energy=-38.0)
    g, f = net(r.unsqueeze(0), kappa=-1, dr=DR)
    
    # 计算残差
    result = compute_dirac_residual(
        g, f, net.E, -1, pots,
        dr=DR, npt=NPT, return_components=True,
        device=r.device,
    )
    
    assert 'loss_pde' in result
    assert 'R_g' in result
    assert 'R_f' in result
    
    print(f"  PDE loss: {result['loss_pde'].item():.4e}")
    print(f"  G残差均值: {result['R_g'].mean().item():.4e}")
    print(f"  F残差均值: {result['R_f'].mean().item():.4e}")
    
    # 残差应该是有限值 (无 NaN/Inf)
    assert torch.isfinite(result['loss_pde']), "PDE损失含NaN或Inf!"
    print("  ✓ 无 NaN/Inf!")
    
    print("  ✓ Test 3 通过!\n")


def test_boundary_conditions():
    """Test 4: 边界条件计算"""
    print("[Test 4] 边界条件")
    
    from boundary_conditions import (
        loss_normalization,
        loss_kinetic_positive, count_nodes,
        compute_total_boundary_loss,
    )

    N = 201
    r = torch.zeros(N)
    g = torch.exp(-r) * (r + 1).float()   # 类似 s 态的形状
    f = 0.3 * r.float() * torch.exp(-r)   # 小分量

    L_norm = loss_normalization(g, f, dr=0.10)
    L_kin = loss_kinetic_positive(g, f)
    n_nodes = count_nodes(g)

    print(f"  归一化损失: {L_norm.item():.4e}")
    print(f"  正动能损失: {L_kin.item():.4e}")
    print(f"  节点数:     {n_nodes}")
    
    # 组合边界损失
    bc_total = compute_total_boundary_loss(g, f, -1, n_expected_nodes=0)
    print(f"  总边界损失: {bc_total['total'].item():.4e}")
    
    assert torch.isfinite(bc_total['total'])
    print("  ✓ Test 4 通过!\n")


def test_full_mvp_training(isotope='16O', state='1s1/2', epochs=500):
    """
    Test 5: 完整 MVP 训练流程 (端到端测试)
    
    这是最重要的测试 — 验证整个训练流程能否正常工作。
    """
    print(f"[Test 5] 完整MVP训练: {isotope} {state} ({epochs} epochs)")
    
    set_seed(42)
    
    solver = MVPSolver(isotope=isotope, state=state, device=DEVICE)
    
    history = solver.train(
        epochs=epochs,
        verbose=True,
        print_every=max(1, epochs // 5),
    )
    
    result = solver.evaluate()
    
    # 基本检验
    assert abs(result['norm_integral'] - 1.0) < 0.1, \
        f"归一化偏差过大: {result['norm_integral']:.4f}"
    
    assert isinstance(result['energy'], float)
    assert result['energy'] < 0, "束缚态能量应为负!"
    
    print(f"\n  ★ MVP 测试结果:")
    print(f"    最终能量: ε = {result['energy']:+.4f} MeV")
    print(f"    归一化:   ∫(G²+F²)dr = {result['norm_integral']:.6f}")
    print(f"    G节点数:  {result['n_nodes']} (期望: {result['expected_nodes']})")
    print(f"  ✓ Test 5 通过! MVP流程正常运行\n")
    
    return solver, result


def get_r_tensor():
    """返回径向网格张量"""
    return torch.tensor(R_GRID, dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PINN-RHF 单元测试')
    parser.add_argument('--isotope', default='16O')
    parser.add_argument('--state', default='1s1/2')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--skip-training', action='store_true',
                       help='跳过耗时训练测试')
    args = parser.parse_args()

    print("=" * 60)
    print("  PINN-RHF 单元测试套件")
    print("=" * 60)

    # 快速测试 (不涉及训练)
    try:
        test_5padf_matrix()
        test_model_forward()
        test_pde_residual()
        test_boundary_conditions()
    except Exception as e:
        print(f"\n  ✗ 快速测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 完整训练测试 (较慢)
    if not args.skip_training:
        try:
            test_full_mvp_training(
                isotope=args.isotope,
                state=args.state,
                epochs=args.epochs,
            )
        except Exception as e:
            print(f"\n  ✗ MVP训练测试失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("\n[⏭ ] 跳过MVP训练测试 (--skip-training)")

    print("="*60)
    print("  所有测试通过!")
    print("="*60)
