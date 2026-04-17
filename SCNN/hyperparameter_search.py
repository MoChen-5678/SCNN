#!/usr/bin/env python3
"""
SCNN 控制变量法超参数搜索脚本
================================
策略: 逐个参数贪心搜索，找到最优即锁定，不回溯。
     每步只搜一个变量，其他保持最优不变。

用法:
    torchrun --nproc_per_node=2 hyperparameter_search.py
"""

import os, sys, json, time, argparse, gc
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from Model_Architecture import RHF_FNO_GRU
from Physics_Informed_Loss import calc_physics_residual
from Data_Loader import RHF_Dataset


# ============================================================
#   搜索空间 (按顺序逐个搜索，找到最优就锁定)
# ============================================================
SEARCH_ORDER = [
    ("learning_rate",    "学习率",       [1e-5, 3e-5, 5e-5, 1e-4, 2e-4]),
    ("batch_size",       "批大小",        [64, 128, 256, 512]),
    ("hidden_dim",       "FNO隐藏维",      [32, 64, 128]),
    ("gru_hidden",       "GRU隐藏维",    [512, 1024, 2048]),
    ("modes",            "傅里叶模式数",   [16, 24, 32, 48]),
    ("weight_decay",     "权重衰减",    [0.0, 1e-5, 1e-4, 1e-3]),
    ("lambda_phy_max",   "λ物理上限", [0.1, 0.2, 0.5]),
    ("phy_warmup",       "预热轮数",      [30, 50, 80]),
    ("seq_len",          "序列长度",        [5, 8, 10]),
    ("clip_grad",        "梯度裁剪",    [0.5, 1.0, 2.0]),
]

# 起始默认配置 (第一个参数从这里开始搜)
DEFAULT_CONFIG = {
    "learning_rate":    5e-5,
    "batch_size":       256,
    "hidden_dim":       64,
    "gru_hidden":       1024,
    "modes":            32,
    "weight_decay":     1e-4,
    "lambda_phy_max":   0.1,
    "phy_warmup":       50,
    "seq_len":          10,
    "clip_grad":        1.0,
}

# 固定参数
FIXED = {
    "num_epochs": 300,
    "data_dir": "/home/ubuntu/rhf/results",
    "dr": 0.10,
    "isotopes": ['16O', '40Ca', '86Kr', '210Pb'],
    "eta_min_lr": 1e-6,
    "lambda_norm": 1.0,
}

LAM_SAT_RATIO = 0.90


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def calc_stats(dataloader):
    sum_x = torch.zeros(11)
    sum_sq_x = torch.zeros(11)
    n = 0
    for x_seq, _, _ in dataloader:
        bs, sl, ch, npt = x_seq.shape
        xf = x_seq.view(-1, ch, npt).permute(1, 0, 2).reshape(ch, -1)
        sum_x += xf.sum(dim=1)
        sum_sq_x += (xf ** 2).sum(dim=1)
        n += bs * sl * npt
    mean = sum_x / n
    std = torch.sqrt(sum_sq_x / n - mean ** 2)
    std = torch.clamp(std, min=1e-8)
    return mean, std


def normalize(tensor, mean, std):
    mv = mean.view(1, 11, 1).to(tensor.device)
    sv = std.view(1, 11, 1).to(tensor.device)
    if len(tensor.shape) == 4:
        mv, sv = mv.unsqueeze(1), sv.unsqueeze(1)
    return (tensor - mv) / sv


def build_loader(config):
    datasets = []
    for iso in FIXED["isotopes"]:
        try:
            ds = RHF_Dataset(data_dir=FIXED["data_dir"], isotope=iso,
                            seq_len=config["seq_len"])
            datasets.append(ds)
        except Exception:
            pass
    combined = ConcatDataset(datasets)
    sampler = DistributedSampler(combined, shuffle=True)
    use_drop = len(combined) >= config["batch_size"] * 2
    loader = DataLoader(combined, batch_size=config["batch_size"],
                        sampler=sampler, drop_last=use_drop)
    if len(loader) == 0:
        raise RuntimeError(f"空数据! bs={config['batch_size']}")
    mean, std = calc_stats(loader)
    return loader, mean, std


def run_trial(label, config, local_rank):
    """跑一次训练，返回 λ 饱和后的评估指标"""
    device = torch.device(f"cuda:{local_rank}")
    is_main = (local_rank == 0)

    try:
        loader, mean, std = build_loader(config)
    except RuntimeError as e:
        if is_main: print(f"    {label}: 数据失败 -> {e}")
        return None

    model = RHF_FNO_GRU(in_channels=11, hidden_dim=config["hidden_dim"],
                        npt=201, gru_hidden=config["gru_hidden"],
                        modes=config["modes"]).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = AdamW(model.parameters(), lr=config["learning_rate"],
                      weight_decay=config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=FIXED["num_epochs"],
                             eta_min=FIXED["eta_min_lr"])

    r_grid = torch.arange(0, 201, device=device, dtype=torch.float32) * FIXED["dr"]
    r_grid[0] = 0.0010

    warmup = config["phy_warmup"]
    sat_thresh = LAM_SAT_RATIO * config["lambda_phy_max"]

    history_sat = []
    best_sat = float("inf")
    patience = 0
    sat_ep = 0

    t0 = time.time()
    for ep in range(1, FIXED["num_epochs"] + 1):
        loader.sampler.set_epoch(ep)
        model.train()
        running_total, nb = 0.0, 0

        if ep <= warmup:
            lam, saturated = 0.0, False
        else:
            step = ep - warmup
            lam = config["lambda_phy_max"] / (1.0 + np.exp(-(step - 25) / 5.0))
            saturated = lam >= sat_thresh

        for x_seq, y_true, kappa in loader:
            x_seq, y_true, kappa = x_seq.to(device), y_true.to(device), kappa.to(device)
            br = r_grid.unsqueeze(0).expand(x_seq.size(0), -1)
            xn, yn = normalize(x_seq, mean, std), normalize(y_true, mean, std)
            optimizer.zero_grad()
            pred = model(xn, kappa, br)
            ld = nn.MSELoss()(pred, yn)
            lp = calc_physics_residual(
                pred_tensor_norm=pred, kappa=kappa,
                stats_mean=mean, stats_std=std,
                dr=FIXED["dr"], lambda_norm=FIXED["lambda_norm"]) if lam > 0 else \
                torch.tensor(0.0, device=device)
            loss = ld + lam * lp
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["clip_grad"])
            optimizer.step()
            running_total += loss.item()
            nb += 1

        scheduler.step()
        lt = torch.tensor(running_total / max(nb, 1)).to(device)
        dist.all_reduce(lt, op=dist.ReduceOp.AVG)
        avg_loss = lt.item()

        if saturated:
            if sat_ep == 0: sat_ep = ep
            history_sat.append(avg_loss)
            if avg_loss < best_sat * 0.999:
                best_sat = avg_loss; patience = 0
            else:
                patience += 1

        if is_main and (ep % 20 == 0 or ep == 1):
            tag = "[SAT]" if saturated else ("[RAMP]" if ep > warmup else "[WARM]")
            extra = f" BestSAT={best_sat:.6f}" if best_sat < float("inf") else ""
            print(f"    {label} Ep[{ep:3d}] {tag} Loss={avg_loss:.6f} λ={lam:.4f}{extra}")

        # 同步 early stop 信号，避免 DDP 死锁
        stop_flag = torch.tensor(
            1.0 if (saturated and patience >= 20 and (ep - sat_ep) >= 30) else 0.0,
            device=device
        )
        dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
        if stop_flag.item() > 0.5:
            if is_main: print(f"    {label} Early stop @ {ep}")
            break

    elapsed = time.time() - t0
    del model, optimizer, scheduler, loader
    torch.cuda.empty_cache(); gc.collect()

    sb = best_sat if best_sat < float("inf") else float("inf")
    sf = history_sat[-1] if history_sat else float("inf")
    s10 = np.mean(history_sat[-10:]) if len(history_sat) >= 10 else \
         (np.mean(history_sat) if history_sat else float("inf"))

    return {
        "sat_best": round(sb, 8),
        "sat_final": round(sf, 8),
        "sat_avg10": round(s10, 8),
        "sat_n": len(history_sat),
        "sat_epoch": sat_ep,
        "total_ep": ep,
        "time": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="SCNN 控制变量法调参")
    parser.add_argument("--output_dir", type=str, default="/home/ubuntu/rhf/results/tune_results")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    local_rank = setup_ddp()
    is_main = (local_rank == 0)

    if is_main:
        print("=" * 65)
        print("  SCNN 控制变量法超参数搜索 (贪心/逐参锁定)")
        print(f"  Epochs/Trial: {FIXED['num_epochs']} | λ饱和阈值: {LAM_SAT_RATIO*100:.0f}%")
        print(f"  参数数量: {len(SEARCH_ORDER)} 个")
        print("=" * 65)

    # ---- 当前最优配置 ----
    current_best = dict(DEFAULT_CONFIG)
    decision_log = []  # 记录每步决策
    t_total_start = time.time()

    for step_i, (param_key, param_label, candidates) in enumerate(SEARCH_ORDER, 1):
        baseline_val = current_best[param_key]

        if is_main:
            print(f"\n{'━' * 58}")
            print(f"  [{step_i}/{len(SEARCH_ORDER)}] 搜索: {param_label} ({param_key})")
            print(f"  当前基线值 = {baseline_val}  |  候选: {candidates}")

        # 测试每个候选值
        cand_results = []
        for ci, cand_val in enumerate(candidates):
            label = f"  {cand_val}"
            trial_cfg = dict(current_best)
            trial_cfg[param_key] = cand_val

            try:
                res = run_trial(label, trial_cfg, local_rank)
                if res and dist.is_initialized():
                    dist.barrier()
            except Exception as e:
                if is_main: print(f"    {label}: 崩溃 {type(e).__name__}: {e}")
                torch.cuda.empty_cache(); gc.collect()
                if dist.is_initialized(): dist.barrier()
                continue

            if res is None:
                continue

            entry = {"value": cand_val, **res}
            cand_results.append(entry)

            if is_main:
                marker = " ← 基线" if cand_val == baseline_val else ""
                print(f"    ✓ {str(cand_val):>12s}  SAT_Best={res['sat_best']:.6f}  "
                      f"SAT_Final={res['sat_final']:.6f}  "
                      f"SatEp={res['sat_n']}  Time={res['time']:.0f}s{marker}")

        # ---- 选出本参数的最优值并锁定 ----
        if not cand_results:
            if is_main: print(f"  ⚠ 所有候选均失败，保留基线值 {baseline_val}")
            decision_log.append({
                "step": step_i, "param": param_key, "chosen": str(baseline_val),
                "reason": "全部失败_保留基线"
            })
            continue

        cand_results.sort(key=lambda x: x["sat_best"])
        winner = cand_results[0]
        chosen_val = winner["value"]

        # 锁定！更新当前最优配置
        current_best[param_key] = chosen_val
        improved = chosen_val != baseline_val

        decision_log.append({
            "step": step_i, "param": param_key, "param_zh": param_label,
            "baseline": str(baseline_val), "chosen": str(chosen_val),
            "changed": improved,
            "best_sat": winner["sat_best"],
            "all_candidates": [
                {"v": c["value"], "sat_best": c["sat_best"]} for c in cand_results
            ]
        })

        if is_main:
            arrow = " ✅ 已切换" if improved else " ➡ 保持不变"
            print(f"\n  ★ 决策: {param_label} = {chosen_val}{arrow}")
            print(f"    SAT_Best_Loss = {winner['sat_best']:.6f}")
            # 显示本轮排名
            print(f"    本轮排名:")
            for rank_idx, cr in enumerate(cand_results[:min(5, len(cand_results))], 1):
                m = " ← 最优" if rank_idx == 1 else ""
                print(f"      #{rank_idx}: v={str(cr['value']):>12s}  sat_best={cr['sat_best']:.6f}{m}")

    # ---- 最终汇总 ----
    total_elapsed = time.time() - t_total_start

    if not is_main:
        cleanup_ddp()
        return

    # 保存结果
    out_path = os.path.join(args.output_dir, "greedy_search_result.json")
    with open(out_path, "w") as f:
        json.dump({
            "final_config": current_best,
            "decision_log": decision_log,
            "total_time_sec": round(total_elapsed, 1),
            "search_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "fixed_params": dict(FIXED),
        }, f, indent=2, ensure_ascii=False, default=str)

    # 打印最终配置
    print("\n" + "=" * 65)
    print("  🏆 控制变量法搜索完成 — 最终最优配置")
    print("=" * 65)
    print(f"\n  {'参数':18s} {'最优值':>14s} {'说明'}")
    print(f"  {'─' * 50}")
    for param_key, param_label, _ in SEARCH_ORDER:
        val = current_best[param_key]
        dec = next((d for d in decision_log if d["param"] == param_key), None)
        note = "已优化" if dec and dec.get("changed") else "保留默认"
        print(f"  {param_label:18s} {str(val):>14s}  ({note})")

    print(f"\n  总耗时: {total_elapsed / 60:.1f} 分钟")
    print(f"  结果保存至: {out_path}")

    # 输出 Train.py 配置代码块
    cfg = current_best
    code = f"""
# ===== 复制到 Train.py 第78-99行超参数面板 ======
learning_rate = {cfg['learning_rate']}       # 学习率
batch_size = {cfg['batch_size']}             # 批大小
seq_len = {cfg['seq_len']}                   # 序列长度
hidden_dim = {cfg['hidden_dim']}              # FNO隐藏维度
gru_hidden = {cfg['gru_hidden']}              # GRU隐藏维度
modes = {cfg['modes']}                       # 傅里叶模式数
weight_decay = {cfg['weight_decay']}          # 权重衰减
phy_warmup = {cfg['phy_warmup']}              # 预热轮数
lambda_phy_max = {cfg['lambda_phy_max']}       # λ物理上限
clip_grad_norm = {cfg['clip_grad']}            # 梯度裁剪
"""
    print(code)

    cleanup_ddp()


if __name__ == "__main__":
    main()
