import os
import re
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


# 全局随机种子：确保多核素间划分一致
_SPLIT_SEED = 42

# ═══════════════════════════════════════════════════════════════
#   核素 → (Z, N) 映射表
#   用于宏观条件编码器 (FiLM) 的条件输入
# ═══════════════════════════════════════════════════════════════
ISOTOPE_ZN = {
    # 氧同位素链
    '16O': (8, 8), '18O': (8, 10), '20O': (8, 12), '22O': (8, 14),
    # 钙同位素链
    '36Ca': (20, 16), '38Ca': (20, 18), '40Ca': (20, 20),
    '42Ca': (20, 22), '44Ca': (20, 24), '46Ca': (20, 26), '48Ca': (20, 28),
    # 镍同位素链
    '56Ni': (28, 28), '58Ni': (28, 30), '60Ni': (28, 32), '62Ni': (28, 34),
    '64Ni': (28, 36), '68Ni': (28, 40), '72Ni': (28, 44), '78Ni': (28, 50),
    # 锡同位素链
    '100Sn': (50, 50), '112Sn': (50, 62), '116Sn': (50, 66),
    '120Sn': (50, 70), '124Sn': (50, 74), '132Sn': (50, 82),
    # 铅同位素链
    '206Pb': (82, 124), '208Pb': (82, 126), '210Pb': (82, 128),
    # 其他重要核素
    '86Kr': (36, 50), '88Sr': (38, 50), '90Zr': (40, 50), '92Mo': (42, 50),
}


def get_zn(isotope):
    """从核素名获取 (Z, N)，不在表中的尝试从名称解析"""
    if isotope in ISOTOPE_ZN:
        return ISOTOPE_ZN[isotope]
    # 尝试从名称解析: '208Pb' → A=208, 需要元素符号查Z
    try:
        m = re.match(r'(\d+)([A-Z][a-z]?)', isotope)
        if m:
            A = int(m.group(1))
            symbol = m.group(2)
            # 常见元素原子序数
            element_z = {'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,
                         'F':9,'Ne':10,'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,
                         'S':16,'Cl':17,'Ar':18,'K':19,'Ca':20,'Sc':21,'Ti':22,
                         'V':23,'Cr':24,'Mn':25,'Fe':26,'Co':27,'Ni':28,'Cu':29,
                         'Zn':30,'Ga':31,'Ge':32,'As':33,'Se':34,'Br':35,'Kr':36,
                         'Rb':37,'Sr':38,'Y':39,'Zr':40,'Nb':41,'Mo':42,'Tc':43,
                         'Ru':44,'Rh':45,'Pd':46,'Ag':47,'Cd':48,'In':49,'Sn':50,
                         'Sb':51,'Te':52,'I':53,'Xe':54,'Cs':55,'Ba':56,
                         'Pt':78,'Au':79,'Hg':80,'Tl':81,'Pb':82,'Bi':83}
            Z = element_z.get(symbol, 0)
            if Z > 0:
                return (Z, A - Z)
    except Exception:
        pass
    # 兜底：返回 (0, 0)，训练时会用默认值
    print(f"  ⚠️ 核素 {isotope} 不在 ISOTOPE_ZN 映射表中，使用默认 (Z=0, N=0)")
    return (0, 0)


def build_datasets(data_dir, isotopes, max_seq_len=10, min_seq_len=3,
                   traj_usage_ratio=0.8, mode='train',
                   target_states=None, val_ratio=0.15, test_ratio=0.15):
    """
    按轨迹（state级别）构建 train/val/test 数据集。

    核心改进：
    ──────────────────────────────
    1. 自适应序列长度
    2. Y_target 为该 state 最终 loop 的收敛态波函数（而非下一步）
    3. 每个样本附加 (Z, N) 宏观量子数标量

    划分策略：每个核素内部按轨迹随机分层抽样，保证：
      1. 同一轨迹的所有样本不会跨集泄露
      2. 各集中 kappa 分布比例一致（分层采样）
      3. 多进程(DDP)间划分结果一致（固定种子）

    返回：
    ------
    ConcatDataset 或 None（如果该模式无样本）
    每个样本为七元组: (X, Y_11ch, kappa, is_proton, actual_len, z_num, n_num)
    """
    rng = np.random.default_rng(_SPLIT_SEED)

    # 第一遍：预扫描所有核素，收集轨迹元信息
    all_trajs = []

    for iso in isotopes:
        meta = _RHF_MetaDataset(
            data_dir=data_dir, isotope=iso,
            max_seq_len=max_seq_len, min_seq_len=min_seq_len,
            traj_usage_ratio=traj_usage_ratio, target_states=target_states
        )
        if len(meta.traj_meta) > 0:
            all_trajs.extend(meta.traj_meta)

    if not all_trajs:
        print("⚠️ 未找到任何有效轨迹！")
        return None

    total_n_samples = sum(t[2] for t in all_trajs)
    print(f"📦 数据集划分总览: {len(all_trajs)} 条轨迹, {total_n_samples} 个样本 "
          f"[max_seq={max_seq_len}, min={min_seq_len}, usage={traj_usage_ratio:.0%}]")

    # 按 kappa 分组后在各组内分层采样，保证分布一致
    kappa_groups = {}
    for t in all_trajs:
        k = t[3]
        if k not in kappa_groups:
            kappa_groups[k] = []
        kappa_groups[k].append(t)

    selected_trajs = {'train': [], 'val': [], 'test': []}
    for k, trajs in kappa_groups.items():
        rng.shuffle(trajs)
        n = len(trajs)
        n_val = max(1, round(n * val_ratio))
        n_test = max(1, round(n * test_ratio))
        n_train = n - n_val - n_test
        if n_train < 1:
            n_test = max(0, n_test - 1)
            n_val = max(0, n_val - 1)
            n_train = n - n_val - n_test

        selected_trajs['train'].extend(trajs[:n_train])
        selected_trajs['val'].extend(trajs[n_train:n_train+n_val])
        selected_trajs['test'].extend(trajs[n_train+n_val:])

    # 构建目标模式的Dataset
    target_indices = selected_trajs.get(mode, [])
    datasets = []

    for iso in isotopes:
        iso_state_names = set()
        for t in target_indices:
            if t[0] == iso:
                iso_state_names.add(t[1])

        if not iso_state_names:
            continue

        ds = _RHF_Dataset(
            data_dir=data_dir, isotope=iso,
            max_seq_len=max_seq_len, min_seq_len=min_seq_len,
            traj_usage_ratio=traj_usage_ratio,
            target_states=target_states,
            allowed_states=iso_state_names
        )
        if len(ds) > 0:
            datasets.append(ds)

    if not datasets:
        return None

    from torch.utils.data import ConcatDataset
    combined = ConcatDataset(datasets)

    print(f"   📊 Train={len(selected_trajs['train'])}轨迹 | Val={len(selected_trajs['val'])}轨迹 | Test={len(selected_trajs['test'])}轨迹")
    print(f"   ✅ [{mode}] 模式: {len(combined)} 个样本")

    return combined


class _RHF_MetaDataset:
    """预扫描阶段：仅收集轨迹元信息，不加载实际数据。用于 build_datasets 的分层划分"""

    def __init__(self, data_dir, isotope, max_seq_len=10, min_seq_len=3,
                 traj_usage_ratio=0.8, target_states=None):
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.traj_usage_ratio = traj_usage_ratio
        self.target_states = target_states
        self.traj_meta = []  # [(isotope, state_name, n_samples, kappa), ...]

        wav_dir = os.path.join(data_dir, isotope, 'WAV')
        pot_dir = os.path.join(data_dir, isotope, 'POT')

        if not os.path.exists(wav_dir):
            return

        all_wav_files = sorted([f for f in os.listdir(wav_dir) if '.loop' in f])
        all_states = sorted(list(set(
            [re.match(r'(.*_state\d+)', f).group(1) for f in all_wav_files if re.match(r'(.*_state\d+)', f)]
        )))

        filtered_states = self._filter_states(all_states, all_wav_files, wav_dir)
        if not filtered_states:
            return

        # 收集每个态的元信息（含自适应seq_len和采样数估算）
        for state in filtered_states:
            state_files = sorted([f for f in all_wav_files if f.startswith(state)],
                                 key=lambda x: _extract_it_loop(x))

            # 计算有效点数
            valid_steps = []
            kappa = -1.0
            for wav_file in state_files:
                pot_file = wav_file.replace('.it', '_POT.it')
                wav_path = os.path.join(wav_dir, wav_file)
                pot_path = os.path.join(pot_dir, pot_file)
                if not os.path.exists(pot_path):
                    continue
                res = _parse_single_step(wav_path, pot_path)
                if res is not None:
                    valid_steps.append(res[0])
                    kappa = res[1]

            L_total = len(valid_steps)
            if L_total < self.min_seq_len + 1:
                continue

            # 自适应序列长度
            eff_len = max(self.min_seq_len, min(self.max_seq_len, int(L_total * self.traj_usage_ratio)))

            # 样本数 = 滑动窗口数量
            n_windows = L_total - eff_len + 1
            stride = max(1, n_windows // max(n_windows, 1)) if n_windows > 0 else 1
            num_samples = (n_windows + stride - 1) // stride
            num_samples = max(1, num_samples)

            if num_samples > 0:
                self.traj_meta.append((isotope, state, num_samples, kappa))

    def _filter_states(self, all_states, all_wav_files, wav_dir):
        """目标态筛选（简化版：final即最终数据，不再做复杂收敛性判断）"""
        if self.target_states is None:
            return all_states

        def _parse_target_lj(s):
            m = re.match(r'(\d*)([a-z])(\d+)/2', s.lower())
            return (m.group(2), int(m.group(3)) / 2.0) if m else None

        target_ljs = [(ts, _parse_target_lj(ts)) for ts in self.target_states]
        filtered = []

        for state in all_states:
            state_files = [f for f in all_wav_files if f.startswith(state)]
            if not state_files:
                continue
            first_file = os.path.join(wav_dir, state_files[0])
            try:
                with open(first_file, 'r') as f:
                    content = f.read()
                match = re.search(r'State:\s*([NP])\.(\d+)([a-z])\.(\d+)/2', content)
                if match:
                    particle, n_val, l_char, j_half = match.group(1), int(match.group(2)), match.group(3).lower(), float(match.group(4))/2.0
                    label = f"{particle}.{n_val}{l_char}.{int(j_half*2)}/2"

                    matched = False
                    for ts, lj in target_ljs:
                        normed = label.replace('.', '').lower()
                        ts_lower = ts.lower()
                        # ★ 关键修复：精确匹配主量子数 n
                        # 例: '1s1/2' 应只匹配 N.1s.1/2，不匹配 N.2s.1/2
                        if ts_lower == normed or ts_lower == f"{n_val}{l_char}{int(j_half*2)}/2":
                            matched = True; break
                        # 兼容旧逻辑：纯 lj 匹配（无前缀数字）
                        if ts_lower.startswith(f"{l_char}") and lj is not None:
                            if lj[0] == l_char and abs(lj[1] - j_half) < 0.01:
                                matched = True; break

                    if matched:
                        # 简化：不再做复杂的收敛性检查，final即最终数据
                        # 仅做极端值过滤：最终loop的 g,f 不超过 1000
                        last_file = os.path.join(wav_dir, sorted(state_files, key=lambda x: _extract_it_loop(x))[-1])
                        if _check_extreme(last_file):
                            filtered.append(state)
                        else:
                            print(f"    ⚠️ 态 {state} ({label}) 最终loop极端值过大，已跳过")
            except Exception:
                continue

        print(f"🎯 目标态筛选: {self.target_states} → 匹配到 {len(filtered)} 个态")
        return filtered


class _RHF_Dataset(Dataset):
    """
    实际数据集：支持自适应序列长度 + 演化进度通道 + (Z,N) 宏观条件。

    输出格式（七元组）：
      X: (max_seq_len, 12, npt)  — 前11维物理场 + 第12维演化进度 progress∈[0,1]
      Y: (11, npt)               — 最终收敛态波函数（仅11物理通道，不含progress）
      kappa: (,)                  — κ量子数
      is_proton: (,)              — 粒子类型
      actual_seq_len: (,)         — 该样本实际有效序列长度
      z_num: (,)                  — 原子序数 Z
      n_num: (,)                  — 中子数 N
    """

    def __init__(self, data_dir, isotope, max_seq_len=10, min_seq_len=3,
                 traj_usage_ratio=0.8, target_states=None, allowed_states=None):
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.traj_usage_ratio = traj_usage_ratio
        self.target_states = target_states
        self.allowed_states = allowed_states or set()

        # 获取该核素的 (Z, N)
        self.z_num, self.n_num = get_zn(isotope)

        wav_dir = os.path.join(data_dir, isotope, 'WAV')
        pot_dir = os.path.join(data_dir, isotope, 'POT')

        if not os.path.exists(wav_dir):
            return

        all_wav_files = sorted([f for f in os.listdir(wav_dir) if '.loop' in f])

        if not self.allowed_states:
            meta = _RHF_MetaDataset(data_dir, isotope, max_seq_len, min_seq_len,
                                     traj_usage_ratio, target_states)
            self.allowed_states = set(t[1] for t in meta.traj_meta)

        self.samples = []
        prefix_map = {'16O': 'O16_', '40Ca': 'Ca40_', '72Ni': 'Ni72_',
                       '86Kr': 'Kr86_', '210Pb': 'Pb210_'}
        iso_prefix = prefix_map.get(isotope, isotope.split('/')[-1])
        matched_states = [s for s in self.allowed_states if s.startswith(iso_prefix)]

        for state in matched_states:
            state_files = sorted([f for f in all_wav_files if f.startswith(state)],
                                 key=lambda x: _extract_it_loop(x))

            # ★ 从 state 文件名或内容中提取主量子数 n
            n_principal = 1  # 默认值
            first_wav = os.path.join(wav_dir, state_files[0]) if state_files else None
            if first_wav and os.path.exists(first_wav):
                try:
                    with open(first_wav, 'r') as f:
                        content = f.read()
                    m = re.search(r'State:\s*[NP]\.(\d+)([a-z])\.(\d+)/2', content)
                    if m:
                        n_principal = int(m.group(1))
                except Exception:
                    pass

            # 加载完整轨迹
            trajectory_data = []
            kappa = -1.0
            is_proton = 0.0

            for wav_file in state_files:
                pot_file = wav_file.replace('.it', '_POT.it')
                res = _parse_single_step(os.path.join(wav_dir, wav_file),
                                         os.path.join(pot_dir, pot_file))
                if res is not None:
                    step_tensor, parsed_kappa, parsed_proton = res
                    trajectory_data.append(step_tensor)
                    kappa = parsed_kappa
                    is_proton = parsed_proton

            L_total = len(trajectory_data)
            if L_total < self.min_seq_len + 1:
                continue

            # 堆叠成张量 (L_total, 11, npt)
            traj_tensor = torch.stack(trajectory_data)
            npt = traj_tensor.shape[-1]

            # ════════════════════════════════════════════════
            # 关键改动：Y_target 为最终收敛态波函数
            # 取该 state 排序后最后一个 loop（用户确认: final就是最终数据）
            # ════════════════════════════════════════════════
            Y_final = traj_tensor[-1]  # (11, npt) — 最终收敛态

            # 自适应序列长度
            eff_len = max(self.min_seq_len,
                          min(self.max_seq_len,
                              int(L_total * self.traj_usage_ratio)))

            # 全轨迹跨度采样
            n_windows = L_total - eff_len + 1
            if n_windows <= 0:
                starts = [0]
            elif n_windows <= eff_len:
                starts = list(range(n_windows))
            else:
                stride = max(1, n_windows // eff_len)
                starts = list(range(0, n_windows, stride))
                if starts[-1] != n_windows - 1:
                    starts.append(n_windows - 1)

            for start in starts:
                end = min(start + eff_len, L_total)
                actual_len = end - start

                # 提取子序列 (actual_len, 11, npt)
                X_11ch = traj_tensor[start:end]

                # Y目标：最终收敛态（所有窗口共享同一个 target）
                Y_11ch = Y_final

                # 第12通道：演化进度 progress ∈ [0, 1]
                indices = torch.arange(actual_len, dtype=torch.float32) + start
                progress_values = indices / max(L_total - 1, 1)
                progress_channel = progress_values.view(-1, 1, 1).expand(-1, 1, npt)

                # 拼接12维：(actual_len, 12, npt)
                X_12ch = torch.cat([X_11ch, progress_channel], dim=1)

                # Padding 到固定的 max_seq_len
                if actual_len < self.max_seq_len:
                    pad_len = self.max_seq_len - actual_len
                    last_frame = X_12ch[-1:].clone()
                    last_frame[:, 11, :] = 1.5  # progress>1.0 标记无效padding
                    pad_frames = last_frame.expand(pad_len, -1, -1).clone()

                    for p in range(pad_len):
                        pad_frames[p, 11, :] = 1.0 + (p + 1) * 0.1

                    X_padded = torch.cat([X_12ch, pad_frames], dim=0)
                else:
                    X_padded = X_12ch

                # 存储八元组（★ 新增 n_principal 主量子数）
                self.samples.append((
                    X_padded,           # (max_seq_len, 12, npt)
                    Y_11ch,             # (11, npt) — 最终收敛态（不含progress）
                    kappa,
                    is_proton,
                    actual_len,
                    self.z_num,         # 原子序数 Z
                    self.n_num,         # 中子数 N
                    float(n_principal),  # ★ 主量子数 n（区分 1s vs 2s vs 3s ...）
                ))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ==================== 向后兼容：保留原始 RHF_Dataset 接口 ====================
class RHF_Dataset(Dataset):
    """
    原始接口的向后兼容封装。
    """

    def __init__(self, data_dir, isotope, seq_len=5, mode='train', min_traj_len=None,
                 target_states=None):
        self._inner = _RHF_Dataset(
            data_dir=data_dir, isotope=isotope,
            max_seq_len=seq_len,
            min_seq_len=max(3, seq_len // 2),
            traj_usage_ratio=0.85,
            target_states=target_states
        )
        self.samples = self._inner.samples

    def __len__(self): return len(self._inner)
    def __getitem__(self, idx): return self._inner[idx]


# ==================== 工具函数 ====================

def _extract_it_loop(filename):
    match = re.search(r'\.it(\d+)\.loop(\d+)', filename)
    return int(match.group(1)) * 1000 + int(match.group(2)) if match else 0


def _fix_fortran_float(s):
    """修复 Fortran 科学计数法: '0.4215+213' → '0.4215E+213'"""
    if isinstance(s, bytes):
        s = s.decode('ascii', errors='ignore')
    s = s.strip()
    m = re.match(r'^([+-]?\d*\.?\d*)\+?(-?\d+)$', s)
    if m:
        base, exp = m.group(1), m.group(2)
        return float(f"{base}E{exp}")
    return float(s)


def _check_extreme(filepath):
    """
    极端值检查：对波函数先归一化再判断，而非直接检查原始值。

    ★ 关键改进：RHF原始数据量级差异大（不同核素/轨道可差10³倍），
              直接用绝对阈值会误杀有效数据。归一化后统一到 O(1) 量级再检查。
    """
    try:
        d = np.loadtxt(filepath, comments='#', converters={1: _fix_fortran_float, 2: _fix_fortran_float})
        r = d[:, 0]
        g, f = d[:, 1], d[:, 2]

        # ★ 先归一化（与 _parse_single_step 中完全一致的逻辑）
        norm_integral = np.trapz(g**2 + f**2, x=r)
        if norm_integral > 1e-12:
            nf = 1.0 / np.sqrt(norm_integral)
            g_norm = g * nf
            f_norm = f * nf
        else:
            return False  # 归一化积分为零 → 无效数据

        # 检查归一化后的值是否合理（O(1)量级）
        max_g = np.max(np.abs(g_norm))
        max_f = np.max(np.abs(f_norm))

        # 合理的束缚态：归一化后 |g| < 50, |f| < 50
        # （正常范围是 |g|~1~30, |f|~0.01~5；留余量防数值噪声）
        return max_g < 50.0 and max_f < 50.0
    except Exception:
        return False


def _check_converged(filepath):
    """
    保留旧接口兼容性，内部调用 _check_extreme。
    """
    return _check_extreme(filepath)


def _parse_single_step(wav_path, pot_path):
    """
    核心解析函数：提取 11 个物理通道和 kappa 量子数。

    返回: (tensor_11x201, kappa_float, is_proton_float) 或 None

    11个通道: [g, f, vps, vms, vtt, XG, XF, YG, YF, E_array, vv_array]
    """
    with open(wav_path, 'r') as f:
        lines = f.readlines()

    energy_line = [l for l in lines if 'Energy=' in l][0]
    energy = float(re.search(r'Energy=\s*([-\d\.E]+)', energy_line).group(1))

    occ_line = [l for l in lines if 'Occupation probability:' in l][0]
    vv = float(re.search(r'Occupation probability:\s+([-\d\.E]+)', occ_line).group(1))

    state_line = [l for l in lines if 'State:' in l][0]
    match = re.search(r'State:\s*([NP])\.\d+([a-z])\.(\d+)/2', state_line)
    kappa = -1.0
    is_proton = 0.0
    if match:
        particle = match.group(1)
        l_char = match.group(2)
        j_num = float(match.group(3))
        is_proton = 1.0 if particle == 'P' else 0.0
        l_map = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5, 'i': 6}
        l_val = l_map.get(l_char, 0)
        j_val = j_num / 2.0
        kappa = -(l_val + 1) if j_val > l_val else l_val

    wav_data = np.loadtxt(wav_path, comments='#', converters={1: _fix_fortran_float, 2: _fix_fortran_float})
    g, f = wav_data[:, 1], wav_data[:, 2]

    if np.max(np.abs(g)) > 100.0 or np.max(np.abs(f)) > 100.0:
        return None

    npt = len(g)
    r_grid = np.arange(npt) * 0.10
    r_grid[0] = 0.0010
    norm_integral = np.trapz(g**2 + f**2, x=r_grid)

    if norm_integral > 1e-12:
        nf = 1.0 / np.sqrt(norm_integral)
        g, f = g * nf, f * nf
    else:
        return None

    # ★ 统一相位约定：第一个峰（|g|的局部极大值）为正
    # 不同RHF计算的波函数可能有任意整体符号(ψ与-ψ满足同一方程),
    # 相位混乱会导致网络无法学习一致规律。
    # 策略：在全局范围内搜索|g|的第一个局部极大值，确保其为正。
    if len(g) > 6:
        order = 5
        abs_g = np.abs(g)
        ref_idx = None
        for i in range(order, len(g) - order):
            is_max = True
            for j in range(1, order + 1):
                if abs_g[i] <= abs_g[i - j] or abs_g[i] <= abs_g[i + j]:
                    is_max = False
                    break
            if is_max:
                ref_idx = i
                break
        if ref_idx is None:
            ref_idx = int(np.argmax(abs_g))
        if g[ref_idx] < 0:
            g, f = -g, -f

    pot_data = np.loadtxt(pot_path, comments='#')
    vps, vms, vtt = pot_data[:, 1], pot_data[:, 2], pot_data[:, 3]
    XG, XF, YG, YF = pot_data[:, 4], pot_data[:, 5], pot_data[:, 6], pot_data[:, 7]
    E_array, vv_array = np.full_like(g, energy), np.full_like(g, vv)

    tensor_np = np.stack([g, f, vps, vms, vtt, XG, XF, YG, YF, E_array, vv_array], axis=0)
    return torch.tensor(tensor_np, dtype=torch.float32), kappa, is_proton


if __name__ == "__main__":
    print("=== 数据管道测试（Y=最终收敛态 + ZN元数据 版）===")

    test_dir = '/home/ubuntu/rhf/results'
    targets = ['1s1/2', '1p1/2']
    isos = ['16O', '40Ca', '86Kr', '210Pb']

    # 测试1：新接口 build_datasets
    print("\n--- 测试 build_datasets (train/val/test 划分) ---")
    for m in ['train', 'val', 'test']:
        ds = build_datasets(test_dir, isos, max_seq_len=10, min_seq_len=3,
                           traj_usage_ratio=0.8, mode=m, target_states=targets)
        if ds is not None:
            print(f"  [{m}]: {len(ds)} samples")
            # 检查一个样本的shape
            sample = ds[0]
            print(f"       X shape: {sample[0].shape}, Y shape: {sample[1].shape}")
            print(f"       kappa={sample[2]}, is_proton={sample[3]}, actual_len={sample[4]}")
            print(f"       Z={sample[5]}, N={sample[6]}")
        else:
            print(f"  [{m}]: None")
