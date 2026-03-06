import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import vitaldb
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 파일 경로
filepath = os.path.join(os.path.dirname(__file__), 'case84_240524_03110530.vital')
outdir = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(outdir, exist_ok=True)

# vital 파일 로드
print("Loading vital file...")
vf = vitaldb.VitalFile(filepath)

# 트랙 목록 확인
track_names = vf.get_track_names()
print(f"Tracks: {track_names}")

# 관심 트랙 정의
numeric_tracks = ['X002/PSI', 'X002/SR', 'X002/EMG', 'X002/SEFL', 'X002/SEFR', 'X002/ARTF']
eeg_tracks = ['X002/EEG_L1', 'X002/EEG_L2', 'X002/EEG_R1', 'X002/EEG_R2', 'X002/EEG_L', 'X002/EEG_R']

# ============================================================
# 1. 수치 트랙 시계열 플롯 (PSI, SR, EMG, SEF, ARTF)
# ============================================================
print("Plotting numeric tracks...")
fig, axes = plt.subplots(len(numeric_tracks), 1, figsize=(16, 14), sharex=True)
fig.suptitle('Case84 - Vital Signs Time Series', fontsize=16, fontweight='bold')

for i, trk_name in enumerate(numeric_tracks):
    try:
        vals = vf.to_numpy(trk_name, 1)  # 1초 간격
        if vals is not None and len(vals) > 0:
            vals = vals.flatten()
            time_sec = np.arange(len(vals))
            time_min = time_sec / 60.0
            axes[i].plot(time_min, vals, linewidth=0.8)
            axes[i].set_ylabel(trk_name.split('/')[-1], fontsize=12, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
            # NaN이 아닌 값의 범위 표시
            valid = vals[~np.isnan(vals)]
            if len(valid) > 0:
                axes[i].set_title(f'{trk_name} (min={valid.min():.1f}, max={valid.max():.1f}, mean={valid.mean():.1f})',
                                  fontsize=10, loc='right')
        else:
            axes[i].text(0.5, 0.5, f'{trk_name}: No data', transform=axes[i].transAxes, ha='center')
    except Exception as e:
        axes[i].text(0.5, 0.5, f'{trk_name}: {e}', transform=axes[i].transAxes, ha='center')

axes[-1].set_xlabel('Time (min)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(outdir, '01_numeric_timeseries.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> 01_numeric_timeseries.png saved")

# ============================================================
# 2. EEG 파형 시계열 플롯
# ============================================================
print("Plotting EEG waveforms...")
fig, axes = plt.subplots(len(eeg_tracks), 1, figsize=(16, 14), sharex=True)
fig.suptitle('Case84 - EEG Waveforms', fontsize=16, fontweight='bold')

eeg_srate = None
eeg_data = {}

for i, trk_name in enumerate(eeg_tracks):
    try:
        # EEG 웨이브는 보통 고해상도 → 작은 간격 사용
        vals = vf.to_numpy(trk_name, 1/128)  # 128Hz
        if vals is not None and len(vals) > 0:
            vals = vals.flatten()
            eeg_data[trk_name] = vals
            eeg_srate = 128
            time_sec = np.arange(len(vals)) / 128.0
            # 처음 30초만 표시 (너무 밀집되므로)
            show_samples = min(128 * 30, len(vals))
            axes[i].plot(time_sec[:show_samples], vals[:show_samples], linewidth=0.5, color='darkblue')
            axes[i].set_ylabel(trk_name.split('/')[-1], fontsize=11, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
        else:
            axes[i].text(0.5, 0.5, f'{trk_name}: No data', transform=axes[i].transAxes, ha='center')
    except Exception as e:
        axes[i].text(0.5, 0.5, f'{trk_name}: {e}', transform=axes[i].transAxes, ha='center')

axes[-1].set_xlabel('Time (sec) - First 30 seconds', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(outdir, '02_eeg_waveforms.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> 02_eeg_waveforms.png saved")

# ============================================================
# 3. EEG 파워 스펙트럼 밀도 (PSD) 분석
# ============================================================
print("Computing Power Spectral Density...")
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle('Case84 - EEG Power Spectral Density (Welch Method)', fontsize=16, fontweight='bold')

# 주파수 대역 정의
bands = {
    'Delta (0.5-4 Hz)': (0.5, 4),
    'Theta (4-8 Hz)': (4, 8),
    'Alpha (8-13 Hz)': (8, 13),
    'Beta (13-30 Hz)': (13, 30),
    'Gamma (30-50 Hz)': (30, 50),
}
band_colors = ['#FF6B6B', '#FFA07A', '#98FB98', '#87CEEB', '#DDA0DD']

for idx, trk_name in enumerate(eeg_tracks):
    ax = axes[idx // 2, idx % 2]
    if trk_name in eeg_data:
        vals = eeg_data[trk_name]
        # NaN 제거 (연속 구간 사용)
        valid_mask = ~np.isnan(vals)
        if valid_mask.sum() > 256:
            clean = vals.copy()
            clean[np.isnan(clean)] = 0  # NaN을 0으로
            # Welch PSD
            freqs, psd = signal.welch(clean, fs=eeg_srate, nperseg=min(256, len(clean)),
                                      noverlap=128, window='hann')
            # 0.5~50Hz 범위만 표시
            mask = (freqs >= 0.5) & (freqs <= 50)
            ax.semilogy(freqs[mask], psd[mask], 'k-', linewidth=1)

            # 주파수 대역 색칠
            for (bname, (f_low, f_high)), bcolor in zip(bands.items(), band_colors):
                bmask = (freqs >= f_low) & (freqs <= f_high)
                ax.fill_between(freqs[bmask], psd[bmask], alpha=0.4, color=bcolor, label=bname)

            ax.set_title(trk_name.split('/')[-1], fontsize=12, fontweight='bold')
            ax.set_xlabel('Frequency (Hz)', fontsize=10)
            ax.set_ylabel('PSD (V²/Hz)', fontsize=10)
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend(fontsize=7, loc='upper right')
        else:
            ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center')
    else:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')

plt.tight_layout()
plt.savefig(os.path.join(outdir, '03_psd_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> 03_psd_analysis.png saved")

# ============================================================
# 4. 스펙트로그램 (시간-주파수 분석)
# ============================================================
print("Computing Spectrograms...")
# EEG_L과 EEG_R에 대해 스펙트로그램
spec_tracks = ['X002/EEG_L', 'X002/EEG_R']
fig, axes = plt.subplots(len(spec_tracks), 1, figsize=(16, 8))
fig.suptitle('Case84 - EEG Spectrogram (Time-Frequency Analysis)', fontsize=16, fontweight='bold')

if not isinstance(axes, np.ndarray):
    axes = [axes]

for i, trk_name in enumerate(spec_tracks):
    if trk_name in eeg_data:
        vals = eeg_data[trk_name]
        clean = vals.copy()
        clean[np.isnan(clean)] = 0

        f, t, Sxx = signal.spectrogram(clean, fs=eeg_srate, nperseg=256,
                                        noverlap=192, window='hann')
        # 0.5~50Hz만
        fmask = (f >= 0.5) & (f <= 50)
        Sxx_db = 10 * np.log10(Sxx[fmask, :] + 1e-12)

        im = axes[i].pcolormesh(t / 60.0, f[fmask], Sxx_db, shading='gouraud', cmap='jet')
        axes[i].set_ylabel('Frequency (Hz)', fontsize=11)
        axes[i].set_title(trk_name.split('/')[-1], fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=axes[i], label='Power (dB)')
    else:
        axes[i].text(0.5, 0.5, f'{trk_name}: No data', transform=axes[i].transAxes, ha='center')

axes[-1].set_xlabel('Time (min)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(outdir, '04_spectrogram.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> 04_spectrogram.png saved")

# ============================================================
# 5. 주파수 대역별 파워 비교 (Bar chart)
# ============================================================
print("Computing band power comparison...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Case84 - EEG Band Power Comparison (L vs R)', fontsize=16, fontweight='bold')

for idx, trk_name in enumerate(['X002/EEG_L', 'X002/EEG_R']):
    if trk_name in eeg_data:
        vals = eeg_data[trk_name]
        clean = vals.copy()
        clean[np.isnan(clean)] = 0
        freqs, psd = signal.welch(clean, fs=eeg_srate, nperseg=min(256, len(clean)),
                                  noverlap=128, window='hann')

        band_powers = []
        band_names = []
        for bname, (f_low, f_high) in bands.items():
            bmask = (freqs >= f_low) & (freqs <= f_high)
            power = np.trapz(psd[bmask], freqs[bmask])
            band_powers.append(power)
            band_names.append(bname.split(' ')[0])

        bars = axes[idx].bar(band_names, band_powers, color=band_colors, edgecolor='black', linewidth=0.5)
        axes[idx].set_title(trk_name.split('/')[-1], fontsize=13, fontweight='bold')
        axes[idx].set_ylabel('Absolute Power (V²)', fontsize=11)
        axes[idx].set_xlabel('Frequency Band', fontsize=11)

        # 상대 파워 표시
        total_power = sum(band_powers)
        if total_power > 0:
            for bar, power in zip(bars, band_powers):
                pct = power / total_power * 100
                axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                             f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    else:
        axes[idx].text(0.5, 0.5, 'No data', transform=axes[idx].transAxes, ha='center')

plt.tight_layout()
plt.savefig(os.path.join(outdir, '05_band_power.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> 05_band_power.png saved")

# ============================================================
# 요약 출력
# ============================================================
print("\n" + "="*60)
print("ANALYSIS SUMMARY")
print("="*60)
for trk_name in ['X002/EEG_L', 'X002/EEG_R']:
    if trk_name in eeg_data:
        vals = eeg_data[trk_name]
        clean = vals[~np.isnan(vals)]
        print(f"\n{trk_name}:")
        print(f"  Total samples: {len(vals)}, Valid: {len(clean)}")
        if len(clean) > 0:
            print(f"  Duration: {len(vals)/eeg_srate:.1f} sec ({len(vals)/eeg_srate/60:.1f} min)")
            print(f"  Amplitude: min={clean.min():.4f}, max={clean.max():.4f}, std={clean.std():.4f}")

            freqs, psd = signal.welch(clean, fs=eeg_srate, nperseg=min(256, len(clean)),
                                      noverlap=128, window='hann')
            for bname, (f_low, f_high) in bands.items():
                bmask = (freqs >= f_low) & (freqs <= f_high)
                power = np.trapz(psd[bmask], freqs[bmask])
                print(f"  {bname}: {power:.6f} V²")

print(f"\nOutput saved to: {outdir}")
print("Done!")
