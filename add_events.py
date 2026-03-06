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

# ============================================================
# 이벤트 시점 (분 단위)
# ============================================================
INDUCTION_MIN = 5.13
GASOFF_MIN = 38.31

def add_event_arrows(ax, y_pos_ratio=0.92, x_axis='min'):
    """축에 induction/gas off 화살표 추가"""
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    y_arrow = ylim[0] + y_range * y_pos_ratio
    y_text = ylim[0] + y_range * 0.98

    if x_axis == 'min':
        ind_x, gas_x = INDUCTION_MIN, GASOFF_MIN
    else:  # seconds
        ind_x, gas_x = INDUCTION_MIN * 60, GASOFF_MIN * 60

    xlim = ax.get_xlim()
    # induction 표시 (xlim 범위 안일 때만)
    if xlim[0] <= ind_x <= xlim[1]:
        ax.axvline(x=ind_x, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=5)
        ax.annotate('Induction', xy=(ind_x, y_arrow),
                    xytext=(ind_x + (xlim[1]-xlim[0])*0.03, y_text),
                    fontsize=14, fontweight='bold', color='red',
                    arrowprops=dict(arrowstyle='->', color='red', lw=2.5),
                    ha='left', va='top', zorder=6)

    # gas off 표시
    if xlim[0] <= gas_x <= xlim[1]:
        ax.axvline(x=gas_x, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, zorder=5)
        ax.annotate('Gas Off', xy=(gas_x, y_arrow),
                    xytext=(gas_x - (xlim[1]-xlim[0])*0.03, y_text),
                    fontsize=14, fontweight='bold', color='blue',
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2.5),
                    ha='right', va='top', zorder=6)


# ============================================================
# 데이터 로드
# ============================================================
print("Loading vital file...")
vf = vitaldb.VitalFile(filepath)
fs = 128

numeric_tracks = ['X002/PSI', 'X002/SR', 'X002/EMG', 'X002/SEFL', 'X002/SEFR', 'X002/ARTF']
eeg_labels = ['EEG_L1', 'EEG_L2', 'EEG_R1', 'EEG_R2', 'EEG_L', 'EEG_R']
eeg_track_names = [f'X002/{l}' for l in eeg_labels]

bands = {
    'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 13),
    'Beta': (13, 30), 'Gamma': (30, 50),
}
band_colors = {'Delta': '#FF6B6B', 'Theta': '#FFA07A', 'Alpha': '#98FB98',
               'Beta': '#87CEEB', 'Gamma': '#DDA0DD'}

# EEG 데이터 로드
eeg_data = {}
for trk, label in zip(eeg_track_names, eeg_labels):
    vals = vf.to_numpy(trk, 1/fs)
    if vals is not None:
        eeg_data[label] = np.nan_to_num(vals.flatten(), nan=0.0)

min_len = min(len(v) for v in eeg_data.values())
for k in eeg_data:
    eeg_data[k] = eeg_data[k][:min_len]

# ============================================================
# 01. Vital Signs 시계열 + 이벤트
# ============================================================
print("01. Numeric timeseries with events...")
fig, axes = plt.subplots(len(numeric_tracks), 1, figsize=(16, 14), sharex=True)
fig.suptitle('Case84 - Vital Signs Time Series', fontsize=16, fontweight='bold')

for i, trk_name in enumerate(numeric_tracks):
    try:
        vals = vf.to_numpy(trk_name, 1)
        if vals is not None and len(vals) > 0:
            vals = vals.flatten()
            time_min = np.arange(len(vals)) / 60.0
            axes[i].plot(time_min, vals, linewidth=0.8)
            axes[i].set_ylabel(trk_name.split('/')[-1], fontsize=12, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
            valid = vals[~np.isnan(vals)]
            if len(valid) > 0:
                axes[i].set_title(f'{trk_name} (min={valid.min():.1f}, max={valid.max():.1f}, mean={valid.mean():.1f})',
                                  fontsize=10, loc='right')
            add_event_arrows(axes[i], y_pos_ratio=0.85)
    except Exception as e:
        axes[i].text(0.5, 0.5, f'{trk_name}: {e}', transform=axes[i].transAxes, ha='center')

axes[-1].set_xlabel('Time (min)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(outdir, '01_numeric_timeseries.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> saved")

# ============================================================
# 02. EEG 파형 + 이벤트
# ============================================================
print("02. EEG waveforms with events...")
fig, axes = plt.subplots(len(eeg_labels), 1, figsize=(16, 14), sharex=True)
fig.suptitle('Case84 - EEG Waveforms (First 30 sec)', fontsize=16, fontweight='bold')

for i, (trk_name, label) in enumerate(zip(eeg_track_names, eeg_labels)):
    if label in eeg_data:
        vals = eeg_data[label]
        time_sec = np.arange(len(vals)) / fs
        show = min(fs * 30, len(vals))
        axes[i].plot(time_sec[:show], vals[:show], linewidth=0.5, color='darkblue')
        axes[i].set_ylabel(label, fontsize=11, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        # 30초 내이므로 induction(5.13분=307.8초)은 범위 밖 -> 표시 안됨
        add_event_arrows(axes[i], x_axis='sec')

axes[-1].set_xlabel('Time (sec) - First 30 seconds', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(outdir, '02_eeg_waveforms.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> saved")

# ============================================================
# 03. PSD (주파수 도메인이므로 이벤트 화살표 불필요 - 그대로 유지)
# ============================================================
print("03. PSD - no time axis, skipping event arrows (kept as-is)")

# ============================================================
# 04. 스펙트로그램 + 이벤트
# ============================================================
print("04. Spectrogram with events...")
spec_tracks = ['X002/EEG_L', 'X002/EEG_R']
fig, axes = plt.subplots(len(spec_tracks), 1, figsize=(16, 8))
fig.suptitle('Case84 - EEG Spectrogram (Time-Frequency Analysis)', fontsize=16, fontweight='bold')

if not isinstance(axes, np.ndarray):
    axes = [axes]

for i, trk_name in enumerate(spec_tracks):
    label = trk_name.split('/')[-1]
    if label in eeg_data:
        vals = eeg_data[label]
        f, t, Sxx = signal.spectrogram(vals, fs=fs, nperseg=256, noverlap=192, window='hann')
        fmask = (f >= 0.5) & (f <= 50)
        Sxx_db = 10 * np.log10(Sxx[fmask, :] + 1e-12)
        im = axes[i].pcolormesh(t / 60.0, f[fmask], Sxx_db, shading='gouraud', cmap='jet')
        axes[i].set_ylabel('Frequency (Hz)', fontsize=11)
        axes[i].set_title(label, fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=axes[i], label='Power (dB)')

        # 이벤트 화살표 (스펙트로그램용)
        ylim = axes[i].get_ylim()
        axes[i].axvline(x=INDUCTION_MIN, color='white', linestyle='--', linewidth=2, alpha=0.9)
        axes[i].annotate('Induction', xy=(INDUCTION_MIN, ylim[1]*0.9),
                         xytext=(INDUCTION_MIN + 1.5, ylim[1]*0.95),
                         fontsize=15, fontweight='bold', color='white',
                         arrowprops=dict(arrowstyle='->', color='white', lw=3),
                         ha='left', va='top')
        axes[i].axvline(x=GASOFF_MIN, color='white', linestyle='--', linewidth=2, alpha=0.9)
        axes[i].annotate('Gas Off', xy=(GASOFF_MIN, ylim[1]*0.9),
                         xytext=(GASOFF_MIN - 1.5, ylim[1]*0.95),
                         fontsize=15, fontweight='bold', color='white',
                         arrowprops=dict(arrowstyle='->', color='white', lw=3),
                         ha='right', va='top')

axes[-1].set_xlabel('Time (min)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(outdir, '04_spectrogram.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> saved")

# ============================================================
# 05. Band power (주파수 도메인 - 이벤트 불필요, 그대로 유지)
# ============================================================
print("05. Band power - no time axis, skipping event arrows (kept as-is)")

# ============================================================
# 06. 주파수별 연결성 (주파수 도메인 - 이벤트 불필요, 그대로 유지)
# ============================================================
print("06. Freq connectivity - no time axis, skipping (kept as-is)")

# ============================================================
# 07. 연결성 매트릭스 (주파수 도메인 - 이벤트 불필요, 그대로 유지)
# ============================================================
print("07. Connectivity matrices - no time axis, skipping (kept as-is)")

# ============================================================
# 08. 대역별 종합 비교 (bar chart - 이벤트 불필요, 그대로 유지)
# ============================================================
print("08. Band summary - no time axis, skipping (kept as-is)")

# ============================================================
# 09. 시간에 따른 연결성 변화 + 이벤트
# ============================================================
print("09. Time-varying connectivity with events...")

x_full = eeg_data['EEG_L']
y_full = eeg_data['EEG_R']

def compute_cross_spectra(x, y, fs, nperseg=256, noverlap=192):
    step = nperseg - noverlap
    n_epochs = (len(x) - nperseg) // step + 1
    window = signal.windows.hann(nperseg)
    freqs = np.fft.rfftfreq(nperseg, 1/fs)
    Sxy_epochs, Sxx_epochs, Syy_epochs = [], [], []
    for i in range(n_epochs):
        s = i * step
        sx = x[s:s+nperseg] * window
        sy = y[s:s+nperseg] * window
        X, Y = np.fft.rfft(sx), np.fft.rfft(sy)
        Sxy_epochs.append(X * np.conj(Y))
        Sxx_epochs.append(np.abs(X)**2)
        Syy_epochs.append(np.abs(Y)**2)
    return freqs, np.array(Sxy_epochs), np.array(Sxx_epochs), np.array(Syy_epochs)

def calc_coherence(Sxy, Sxx, Syy):
    num = np.abs(np.mean(Sxy, axis=0))**2
    den = np.mean(Sxx, axis=0) * np.mean(Syy, axis=0)
    return np.where(den > 0, num / den, 0)

def calc_pli(Sxy):
    return np.abs(np.mean(np.sign(np.imag(Sxy)), axis=0))

def calc_dpli(Sxy):
    return np.mean((np.imag(Sxy) > 0).astype(float), axis=0)

def calc_wpli(Sxy):
    im = np.imag(Sxy)
    num = np.abs(np.mean(im, axis=0))
    den = np.mean(np.abs(im), axis=0)
    return np.where(den > 0, num / den, 0)

def calc_dwpli(Sxy):
    im = np.imag(Sxy)
    s1 = np.sum(im, axis=0)
    s2 = np.sum(np.abs(im), axis=0)
    s3 = np.sum(im**2, axis=0)
    num = s1**2 - s3
    den = s2**2 - s3
    return np.where(den > 0, num / den, 0)

def calc_ple(Sxy, n_bins=12):
    phase = np.angle(Sxy)
    n_f = phase.shape[1]
    ple = np.zeros(n_f)
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    max_e = np.log(n_bins)
    for fi in range(n_f):
        h, _ = np.histogram(phase[:, fi], bins=edges)
        p = h / h.sum() if h.sum() > 0 else h
        p = p[p > 0]
        ple[fi] = (-np.sum(p * np.log(p)) / max_e) if len(p) > 0 else 0
    return ple

metric_funcs = {
    'Coherence': lambda Sxy, Sxx, Syy: calc_coherence(Sxy, Sxx, Syy),
    'PLI': lambda Sxy, Sxx, Syy: calc_pli(Sxy),
    'dPLI': lambda Sxy, Sxx, Syy: calc_dpli(Sxy),
    'wPLI': lambda Sxy, Sxx, Syy: calc_wpli(Sxy),
    'dwPLI': lambda Sxy, Sxx, Syy: calc_dwpli(Sxy),
    'PLE': lambda Sxy, Sxx, Syy: calc_ple(Sxy),
}
metric_list = list(metric_funcs.keys())

win_sec, step_sec = 30, 10
win_samples = int(win_sec * fs)
step_samples = int(step_sec * fs)
n_windows = (len(x_full) - win_samples) // step_samples + 1

time_points = []
time_conn = {m: {b: [] for b in bands} for m in metric_list}

for w in range(n_windows):
    s = w * step_samples
    e = s + win_samples
    t_mid = (s + e) / 2 / fs / 60
    time_points.append(t_mid)
    freqs_w, Sxy_w, Sxx_w, Syy_w = compute_cross_spectra(x_full[s:e], y_full[s:e], fs)
    for mname, mfunc in metric_funcs.items():
        vals = mfunc(Sxy_w, Sxx_w, Syy_w)
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
        ax.plot(time_points, time_conn[mname][bname], linewidth=1.2,
                label=bname, color=band_colors[bname])
    ax.set_ylabel(mname, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right', ncol=5)
    add_event_arrows(ax, y_pos_ratio=0.85)

axes[-1].set_xlabel('Time (min)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(outdir, '09_time_varying_connectivity.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> saved")

print("\nAll done! Updated images saved to:", outdir)
