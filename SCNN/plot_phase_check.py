import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import glob

def fix_fortran_float(s):
    if isinstance(s, bytes):
        s = s.decode('ascii', errors='ignore')
    s = s.strip()
    m = re.match(r'^([+-]?\d*\.?\d*)\+?(-?\d+)$', s)
    return float(f'{m.group(1)}E{m.group(2)}') if m else float(s)

def find_first_peak(g, order=5):
    """找|g|的第一个显著峰位置（跳过原点平坦区域）"""
    abs_g = np.abs(g)
    significant = abs_g > 1e-4 * np.max(abs_g)
    if np.any(significant):
        return int(np.argmax(np.where(significant, abs_g, 0)))
    return int(np.argmax(abs_g))

# 找真正的 1s1/2 文件
wav_dir = '/home/ubuntu/rhf/results/16O/WAV/'
all_files = sorted(glob.glob(wav_dir + '*.loop*'))
s1_files = []
for f in all_files:
    with open(f) as fh:
        head = fh.read(500)
    if 'State:' in head:
        m = re.search(r'State:\s*[NP]\.(\d+[a-z]\.\d+/2)', head)
        if m and '1s' in m.group(1).lower():
            s1_files.append((f, m.group(1)))

print('=== 1s files ===')
for fn, st in s1_files[-5:]:
    print(f'  {st}: {fn}')

target = s1_files[-1][0] if s1_files else None
print(f'\nUsing: {target}')
with open(target) as fh:
    content = fh.read()
state_lines = [l for l in content.split('\n') if 'State:' in l]
print(f'State: {state_lines[0]}')

d = np.loadtxt(target, comments='#', converters={1: fix_fortran_float, 2: fix_fortran_float})
r, g_raw, f_raw = d[:,0], d[:,1], d[:,2]

# 归一化
norm_int = np.trapz(g_raw**2 + f_raw**2, x=r)
nf = 1.0 / np.sqrt(norm_int) if norm_int > 1e-12 else 1.0
g, f = g_raw * nf, f_raw * nf

# 找第一个真正的局部极大值（跳过原点附近）
ref_idx = find_first_peak(g, order=5)

print(f'\nFirst real peak idx={ref_idx} (r={r[ref_idx]:.2f}fm), g={g[ref_idx]:.4f}')

flipped = False
if g[ref_idx] < 0:
    g, f = -g, -f
    flipped = True

label_flip = " (flipped)" if flipped else ""
print(f'Flipped: {flipped}, now g[first_peak]={g[ref_idx]:.4f}')

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].plot(r, g, 'b-', lw=2)
axes[0].axhline(0, color='k', lw=0.5)
axes[0].scatter([r[ref_idx]], [g[ref_idx]], c='red', s=80, zorder=5,
                label=f'1st peak @ r={r[ref_idx]:.1f}fm')
axes[0].legend(fontsize=8)
axes[0].set_title(f'g(r) — First Peak > 0{label_flip}')
axes[0].set_xlabel('r (fm)')
axes[0].grid(alpha=0.3)

axes[1].plot(r, f, 'r-', lw=2)
axes[1].axhline(0, color='k', lw=0.5)
axes[1].set_title('f(r) — Small Component')
axes[1].set_xlabel('r (fm)')
axes[1].grid(alpha=0.3)

prob = g**2 + f**2
axes[2].fill_between(r, prob, alpha=0.3, color='purple')
axes[2].plot(r, prob, 'purple', lw=2)
total_norm = np.trapz(prob, x=r)
axes[2].set_title(f'|psi|^2  integral={total_norm:.4f}')
axes[2].set_xlabel('r (fm)')
axes[2].grid(alpha=0.3)

fig.suptitle('O-16 | 1s1/2 — Phase Convention: First Peak > 0',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/ubuntu/rhf/SCNN/plots/O16_1s12_phase_check.png',
            dpi=150, bbox_inches='tight')
print('\nSaved to plots/O16_1s12_phase_check.png')
