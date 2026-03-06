import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from itertools import combinations
import vitaldb
import os

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

filepath = os.path.join(os.path.dirname(__file__), 'case84_240524_03110530.vital')
outdir = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(outdir, exist_ok=True)

# ============================================================
# 데이터 로드
# ============================================================
print("Loading vital file...")
vf = vitaldb.VitalFile(filepath)

eeg_labels = ['EEG_L1', 'EEG_L2', 'EEG_R1', 'EEG_R2', 'EEG_L', 'EEG_R']
eeg_track_names = [f'X002/{l}' for l in eeg_labels]
fs = 128

eeg_data = {}
for trk, label in zip(eeg_track_names, eeg_labels):
    vals = vf.to_numpy(trk, 1/fs)
    if vals is not None:
        eeg_data[label] = vals.flatten()
        print(f"  {label}: {len(vals)} samples")

# 공통 길이로 맞추기
min_len = min(len(v) for v in eeg_data.values())
for k in eeg_data:
    eeg_data[k] = eeg_data[k][:min_len]

# NaN -> 0
for k in eeg_data:
    eeg_data[k] = np.nan_to_num(eeg_data[k], nan=0.0)

print(f"  Duration: {min_len/fs:.1f} sec ({min_len/fs/60:.1f} min)")

bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta':  (13, 30),
    'Gamma': (30, 50),
}
band_colors = {'Delta': '#FF6B6B', 'Theta': '#FFA07A', 'Alpha': '#98FB98',
               'Beta': '#87CEEB', 'Gamma': '#DDA0DD'}

channel_pairs = list(combinations(eeg_labels, 2))
pair_labels = [f"{a}-{b}" for a, b in channel_pairs]

# ============================================================
# Helper: cross-spectral density (epoch-based)
# ============================================================
def compute_cross_spectra(x, y, fs, nperseg=256, noverlap=192):
    """Epoch 기반 cross-spectrum 계산. 각 epoch의 cross-spectrum 반환."""
    step = nperseg - noverlap
    n_epochs = (len(x) - nperseg) // step + 1
    window = signal.windows.hann(nperseg)
    freqs = np.fft.rfftfreq(nperseg, 1/fs)

    Sxy_epochs = []
    Sxx_epochs = []
    Syy_epochs = []

    for i in range(n_epochs):
        start = i * step
        seg_x = x[start:start+nperseg] * window
        seg_y = y[start:start+nperseg] * window
        X = np.fft.rfft(seg_x)
        Y = np.fft.rfft(seg_y)
        Sxy_epochs.append(X * np.conj(Y))
        Sxx_epochs.append(np.abs(X)**2)
        Syy_epochs.append(np.abs(Y)**2)

    return freqs, np.array(Sxy_epochs), np.array(Sxx_epochs), np.array(Syy_epochs)


# ============================================================
# 연결성 지표 계산 함수들
# ============================================================
def calc_coherence(Sxy, Sxx, Syy):
    """Magnitude Squared Coherence"""
    num = np.abs(np.mean(Sxy, axis=0))**2
    den = np.mean(Sxx, axis=0) * np.mean(Syy, axis=0)
    return np.where(den > 0, num / den, 0)


def calc_pli(Sxy):
    """Phase Lag Index"""
    phase_diff = np.imag(Sxy)
    return np.abs(np.mean(np.sign(phase_diff), axis=0))


def calc_dpli(Sxy):
    """Directed Phase Lag Index"""
    phase_diff = np.imag(Sxy)
    return np.mean((phase_diff > 0).astype(float), axis=0)


def calc_wpli(Sxy):
    """Weighted Phase Lag Index"""
    imag_Sxy = np.imag(Sxy)
    num = np.abs(np.mean(imag_Sxy, axis=0))
    den = np.mean(np.abs(imag_Sxy), axis=0)
    return np.where(den > 0, num / den, 0)


def calc_dwpli(Sxy):
    """Debiased Weighted Phase Lag Index"""
    n = Sxy.shape[0]
    imag_Sxy = np.imag(Sxy)
    sum_imag = np.sum(imag_Sxy, axis=0)
    sum_abs_imag = np.sum(np.abs(imag_Sxy), axis=0)
    sum_sq_imag = np.sum(imag_Sxy**2, axis=0)

    num = sum_imag**2 - sum_sq_imag
    den = sum_abs_imag**2 - sum_sq_imag
    return np.where(den > 0, num / den, 0)


def calc_ple(Sxy, n_bins=12):
    """Phase Lag Entropy"""
    phase_diff = np.angle(Sxy)  # (n_epochs, n_freqs)
    n_freqs = phase_diff.shape[1]
    ple = np.zeros(n_freqs)
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    max_entropy = np.log(n_bins)

    for f_idx in range(n_freqs):
        hist, _ = np.histogram(phase_diff[:, f_idx], bins=bin_edges)
        prob = hist / hist.sum() if hist.sum() > 0 else hist
        prob = prob[prob > 0]
        entropy = -np.sum(prob * np.log(prob)) if len(prob) > 0 else 0
        ple[f_idx] = entropy / max_entropy if max_entropy > 0 else 0

    return ple


# ============================================================
# 모든 채널 쌍에 대해 지표 계산
# ============================================================
metrics = {
    'Coherence': calc_coherence,
    'PLI': calc_pli,
    'dPLI': calc_dpli,
    'wPLI': calc_wpli,
    'dwPLI': calc_dwpli,
    'PLE': calc_ple,
}

print("\nComputing connectivity metrics...")

# 결과 저장: {metric_name: {pair_label: {band_name: value}}}
band_results = {m: {} for m in metrics}
# 주파수별 결과: {metric_name: {pair_label: (freqs, values)}}
freq_results = {m: {} for m in metrics}

for (ch_a, ch_b), plabel in zip(channel_pairs, pair_labels):
    print(f"  {plabel}...")
    x = eeg_data[ch_a]
    y = eeg_data[ch_b]

    freqs, Sxy, Sxx, Syy = compute_cross_spectra(x, y, fs, nperseg=256, noverlap=192)

    for mname, mfunc in metrics.items():
        if mname == 'Coherence':
            vals = mfunc(Sxy, Sxx, Syy)
        elif mname in ('PLI', 'wPLI', 'dwPLI'):
            vals = mfunc(Sxy)
        elif mname == 'dPLI':
            vals = mfunc(Sxy)
        elif mname == 'PLE':
            vals = mfunc(Sxy)
        else:
            vals = mfunc(Sxy)

        freq_results[mname][plabel] = (freqs, vals)

        # 대역별 평균
        band_results[mname][plabel] = {}
        for bname, (f_lo, f_hi) in bands.items():
            bmask = (freqs >= f_lo) & (freqs <= f_hi)
            band_results[mname][plabel][bname] = np.mean(vals[bmask])

print("Done computing.")

# ============================================================
# 6. 주파수별 연결성 플롯 (각 지표별)
# ============================================================
print("\nPlotting frequency-resolved connectivity...")
fig, axes = plt.subplots(3, 2, figsize=(18, 15))
fig.suptitle('Case84 - Frequency-Resolved Functional Connectivity', fontsize=16, fontweight='bold')

metric_list = ['Coherence', 'PLI', 'dPLI', 'wPLI', 'dwPLI', 'PLE']
# 주요 채널쌍만 표시 (가독성)
key_pairs = ['EEG_L1-EEG_R1', 'EEG_L2-EEG_R2', 'EEG_L-EEG_R',
             'EEG_L1-EEG_L2', 'EEG_R1-EEG_R2']

colors_line = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

for idx, mname in enumerate(metric_list):
    ax = axes[idx // 2, idx % 2]
    for pidx, plabel in enumerate(key_pairs):
        if plabel in freq_results[mname]:
            f, v = freq_results[mname][plabel]
            fmask = (f >= 0.5) & (f <= 50)
            ax.plot(f[fmask], v[fmask], linewidth=1.2, label=plabel,
                    color=colors_line[pidx], alpha=0.85)

    # 대역 배경색
    for bname, (f_lo, f_hi) in bands.items():
        ax.axvspan(f_lo, f_hi, alpha=0.08, color=band_colors[bname])

    ax.set_title(mname, fontsize=13, fontweight='bold')
    ax.set_xlabel('Frequency (Hz)', fontsize=10)
    ax.set_ylabel(mname, fontsize=10)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 50)

plt.tight_layout()
plt.savefig(os.path.join(outdir, '06_connectivity_frequency.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> 06_connectivity_frequency.png saved")

# ============================================================
# 7. 대역별 연결성 히트맵 (6x6 매트릭스)
# ============================================================
print("Plotting connectivity matrices...")

for mname in metric_list:
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    fig.suptitle(f'Case84 - {mname} Connectivity Matrix by Band', fontsize=14, fontweight='bold')

    for bidx, bname in enumerate(bands.keys()):
        n_ch = len(eeg_labels)
        mat = np.zeros((n_ch, n_ch))

        for i in range(n_ch):
            for j in range(n_ch):
                if i == j:
                    mat[i, j] = 1.0 if mname != 'dPLI' else 0.5
                else:
                    a, b = eeg_labels[i], eeg_labels[j]
                    plabel = f"{a}-{b}" if f"{a}-{b}" in band_results[mname] else f"{b}-{a}"
                    if plabel in band_results[mname]:
                        val = band_results[mname][plabel][bname]
                        if mname == 'dPLI' and plabel == f"{b}-{a}":
                            val = 1 - val  # dPLI 방향성 반영
                        mat[i, j] = val

        if mname == 'dPLI':
            vmin, vmax, cmap = 0, 1, 'RdBu_r'
        else:
            vmin, vmax, cmap = 0, np.nanmax(mat[~np.eye(n_ch, dtype=bool)]) * 1.1 if np.nanmax(mat[~np.eye(n_ch, dtype=bool)]) > 0 else 1, 'YlOrRd'

        im = axes[bidx].imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap, aspect='equal')
        axes[bidx].set_xticks(range(n_ch))
        axes[bidx].set_yticks(range(n_ch))
        short_labels = [l.replace('EEG_', '') for l in eeg_labels]
        axes[bidx].set_xticklabels(short_labels, fontsize=8, rotation=45)
        axes[bidx].set_yticklabels(short_labels, fontsize=8)
        axes[bidx].set_title(bname, fontsize=11, fontweight='bold')

        # 값 표시
        for ii in range(n_ch):
            for jj in range(n_ch):
                if ii != jj:
                    axes[bidx].text(jj, ii, f'{mat[ii,jj]:.2f}', ha='center', va='center',
                                    fontsize=6, color='black' if mat[ii,jj] < vmax*0.7 else 'white')

        plt.colorbar(im, ax=axes[bidx], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fname = f'07_{mname.lower()}_matrix.png'
    plt.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {fname} saved")

# ============================================================
# 8. 대역별 종합 비교 (주요 쌍, 모든 지표)
# ============================================================
print("Plotting band-level summary...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Case84 - Connectivity Summary by Frequency Band', fontsize=16, fontweight='bold')

for idx, mname in enumerate(metric_list):
    ax = axes[idx // 3, idx % 3]
    n_bands = len(bands)
    n_pairs = len(key_pairs)
    bar_width = 0.15
    x = np.arange(n_bands)

    for pidx, plabel in enumerate(key_pairs):
        if plabel in band_results[mname]:
            vals = [band_results[mname][plabel][b] for b in bands.keys()]
            ax.bar(x + pidx * bar_width, vals, bar_width, label=plabel,
                   color=colors_line[pidx], alpha=0.8, edgecolor='black', linewidth=0.3)

    ax.set_title(mname, fontsize=13, fontweight='bold')
    ax.set_xticks(x + bar_width * (n_pairs - 1) / 2)
    ax.set_xticklabels(list(bands.keys()), fontsize=9)
    ax.set_ylabel(mname, fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')
    if idx == 0:
        ax.legend(fontsize=6, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(outdir, '08_connectivity_summary.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> 08_connectivity_summary.png saved")

# ============================================================
# 9. 시간에 따른 연결성 변화 (EEG_L - EEG_R)
# ============================================================
print("Computing time-varying connectivity (EEG_L vs EEG_R)...")
x_full = eeg_data['EEG_L']
y_full = eeg_data['EEG_R']

win_sec = 30  # 30초 윈도우
step_sec = 10  # 10초 스텝
win_samples = int(win_sec * fs)
step_samples = int(step_sec * fs)
n_windows = (len(x_full) - win_samples) // step_samples + 1

time_points = []
time_conn = {m: {b: [] for b in bands} for m in metric_list}

for w in range(n_windows):
    start = w * step_samples
    end = start + win_samples
    x_seg = x_full[start:end]
    y_seg = y_full[start:end]
    t_mid = (start + end) / 2 / fs / 60  # minutes
    time_points.append(t_mid)

    freqs_w, Sxy_w, Sxx_w, Syy_w = compute_cross_spectra(x_seg, y_seg, fs, nperseg=256, noverlap=192)

    for mname in metric_list:
        if mname == 'Coherence':
            vals = calc_coherence(Sxy_w, Sxx_w, Syy_w)
        elif mname == 'PLI':
            vals = calc_pli(Sxy_w)
        elif mname == 'dPLI':
            vals = calc_dpli(Sxy_w)
        elif mname == 'wPLI':
            vals = calc_wpli(Sxy_w)
        elif mname == 'dwPLI':
            vals = calc_dwpli(Sxy_w)
        elif mname == 'PLE':
            vals = calc_ple(Sxy_w)

        for bname, (f_lo, f_hi) in bands.items():
            bmask = (freqs_w >= f_lo) & (freqs_w <= f_hi)
            time_conn[mname][bname].append(np.mean(vals[bmask]))

time_points = np.array(time_points)

fig, axes = plt.subplots(6, 1, figsize=(16, 20), sharex=True)
fig.suptitle('Case84 - Time-Varying Connectivity (EEG_L vs EEG_R, 30s window)',
             fontsize=16, fontweight='bold')

for idx, mname in enumerate(metric_list):
    ax = axes[idx]
    for bname in bands:
        vals_t = np.array(time_conn[mname][bname])
        ax.plot(time_points, vals_t, linewidth=1.2, label=bname, color=band_colors[bname])

    ax.set_ylabel(mname, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right', ncol=5)

axes[-1].set_xlabel('Time (min)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(outdir, '09_time_varying_connectivity.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> 09_time_varying_connectivity.png saved")

# ============================================================
# 요약 테이블 출력
# ============================================================
print("\n" + "="*80)
print("FUNCTIONAL CONNECTIVITY SUMMARY (EEG_L vs EEG_R)")
print("="*80)
print(f"{'Metric':<12}", end="")
for bname in bands:
    print(f"{bname:>10}", end="")
print()
print("-"*62)

plabel = 'EEG_L-EEG_R'
for mname in metric_list:
    print(f"{mname:<12}", end="")
    for bname in bands:
        val = band_results[mname][plabel][bname]
        print(f"{val:>10.4f}", end="")
    print()

print(f"\nAll outputs saved to: {outdir}")
print("Done!")
