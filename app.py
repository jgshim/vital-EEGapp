import logging
import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from itertools import combinations
import vitaldb
import tempfile
import os
import io
import pandas as pd
from collections import Counter
from math import factorial

# ============================================================
# Page config & font
# ============================================================
st.set_page_config(page_title="Vital EEG 분석", layout="wide")
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# numpy compat
_trapz = getattr(np, 'trapezoid', None) or np.trapz

# ============================================================
# Constants
# ============================================================
BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 50),
}
BAND_COLORS = {'Delta': '#FF6B6B', 'Theta': '#FFA07A', 'Alpha': '#98FB98',
               'Beta': '#87CEEB', 'Gamma': '#DDA0DD'}
BAND_COLORS_LIST = list(BAND_COLORS.values())
CONN_LINE_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
                    '#a65628', '#f781bf', '#999999']
EEG_FS = 128
WIN_SEC = 30   # sliding window length (seconds)
STEP_SEC = 10  # sliding window step (seconds)

# ============================================================
# Utility functions
# ============================================================

def compute_welch_psd(x, fs=EEG_FS, max_nperseg=256):
    """Compute Welch PSD with safe nperseg/noverlap handling.
    Returns (freqs, psd) arrays.
    """
    _nperseg = min(max_nperseg, len(x))
    _noverlap = min(128, _nperseg // 2)
    return signal.welch(x, fs=fs, nperseg=_nperseg,
                        noverlap=_noverlap, window='hann')


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()


def show_fig(fig, filename, key):
    """Display figure + PNG download button, then close.
    Renders PNG once and reuses the bytes for both display and download.
    """
    img_bytes = fig_to_bytes(fig)
    st.image(img_bytes)
    st.download_button(f"PNG 다운로드: {filename}", img_bytes,
                       filename, "image/png", key=key)
    plt.close(fig)


def show_table(df, filename, key):
    """Display dataframe + CSV download button."""
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(f"CSV 다운로드: {filename}", csv, filename,
                       "text/csv", key=key)


def add_event_lines(ax, events, x_axis='min', style='normal'):
    """Add event vertical lines + arrows."""
    if not events:
        return
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    y_range = ylim[1] - ylim[0]
    for ev in events:
        t = ev['time_min'] if x_axis == 'min' else ev['time_min'] * 60
        if not (xlim[0] <= t <= xlim[1]):
            continue
        name_lower = ev['name'].lower()
        if 'induction' in name_lower:
            base_color = 'red'
        elif 'gas' in name_lower:
            base_color = 'blue'
        else:
            base_color = 'green'
        line_c = 'white' if style == 'spectrogram' else base_color
        text_c = line_c
        ax.axvline(x=t, color=line_c, linestyle='--', linewidth=1.5,
                   alpha=0.8, zorder=5)
        ax.annotate(ev['name'], xy=(t, ylim[0] + y_range * 0.90),
                    xytext=(t + (xlim[1] - xlim[0]) * 0.02,
                            ylim[0] + y_range * 0.97),
                    fontsize=11, fontweight='bold', color=text_c,
                    arrowprops=dict(arrowstyle='->', color=text_c, lw=2),
                    ha='left', va='top', zorder=6)

# ============================================================
# Analysis functions (ported from existing scripts)
# ============================================================

def compute_cross_spectra(x, y, fs, nperseg=256, noverlap=192):
    step = nperseg - noverlap
    n_epochs = (len(x) - nperseg) // step + 1
    if n_epochs <= 0:
        freqs = np.fft.rfftfreq(nperseg, 1 / fs)
        empty = np.zeros((1, len(freqs)))
        return freqs, empty, empty, empty
    window = signal.windows.hann(nperseg)
    freqs = np.fft.rfftfreq(nperseg, 1 / fs)
    # Vectorized epoch extraction using sliding_window_view
    x_wins = np.lib.stride_tricks.sliding_window_view(x, nperseg)[::step][:n_epochs] * window
    y_wins = np.lib.stride_tricks.sliding_window_view(y, nperseg)[::step][:n_epochs] * window
    X = np.fft.rfft(x_wins, axis=1)
    Y = np.fft.rfft(y_wins, axis=1)
    Sxy = X * np.conj(Y)
    Sxx = np.abs(X) ** 2
    Syy = np.abs(Y) ** 2
    return freqs, Sxy, Sxx, Syy


def calc_coherence(Sxy, Sxx, Syy):
    num = np.abs(np.mean(Sxy, axis=0)) ** 2
    den = np.mean(Sxx, axis=0) * np.mean(Syy, axis=0)
    return np.where(den > 0, num / den, 0)


def calc_pli(Sxy, Sxx=None, Syy=None):
    return np.abs(np.mean(np.sign(np.imag(Sxy)), axis=0))


def calc_dpli(Sxy, Sxx=None, Syy=None):
    return np.mean((np.imag(Sxy) > 0).astype(float), axis=0)


def calc_wpli(Sxy, Sxx=None, Syy=None):
    im = np.imag(Sxy)
    num = np.abs(np.mean(im, axis=0))
    den = np.mean(np.abs(im), axis=0)
    return np.where(den > 0, num / den, 0)


def calc_dwpli(Sxy, Sxx=None, Syy=None):
    im = np.imag(Sxy)
    s1 = np.sum(im, axis=0)
    s2 = np.sum(np.abs(im), axis=0)
    s3 = np.sum(im ** 2, axis=0)
    num = s1 ** 2 - s3
    den = s2 ** 2 - s3
    return np.where(den > 0, num / den, 0)


def calc_ple(Sxy, Sxx=None, Syy=None, n_bins=12):
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


METRICS = {
    'Coherence': calc_coherence,
    'PLI': calc_pli,
    'dPLI': calc_dpli,
    'wPLI': calc_wpli,
    'dwPLI': calc_dwpli,
    'PLE': calc_ple,
}
METRIC_LIST = list(METRICS.keys())


def build_connectivity_matrix(eeg_labels, band_results, metric, band_name,
                              diagonal_val=0, dpli_flip=False):
    """Build n_ch x n_ch connectivity matrix from band_results dict."""
    n_ch = len(eeg_labels)
    mat = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(n_ch):
            if i == j:
                mat[i, j] = diagonal_val
            else:
                a, b = eeg_labels[i], eeg_labels[j]
                pl = f"{a}-{b}"
                pl_rev = f"{b}-{a}"
                if pl in band_results[metric]:
                    val = band_results[metric][pl].get(band_name, 0)
                elif pl_rev in band_results[metric]:
                    val = band_results[metric][pl_rev].get(band_name, 0)
                    if dpli_flip and metric == 'dPLI':
                        val = 1 - val
                else:
                    val = np.nan
                mat[i, j] = val
    return mat

# ============================================================
# Advanced analysis functions
# ============================================================

def calc_spectral_entropy(x, fs):
    """Spectral entropy via Welch PSD, normalized to [0,1]."""
    freqs, psd = compute_welch_psd(x, fs=fs)
    psd_norm = psd / psd.sum() if psd.sum() > 0 else psd
    psd_norm = psd_norm[psd_norm > 0]
    N = len(psd_norm)
    if N <= 1:
        return 0.0
    return float(-np.sum(psd_norm * np.log2(psd_norm)) / np.log2(N))


def calc_permutation_entropy(x, m=3, tau=1):
    """Permutation entropy, normalized to [0,1]."""
    n = len(x)
    if n < m * tau:
        return 0.0
    patterns = []
    for i in range(n - (m - 1) * tau):
        window = x[i:i + m * tau:tau]
        patterns.append(tuple(np.argsort(window)))
    counts = Counter(patterns)
    total = len(patterns)
    probs = np.array([c / total for c in counts.values()])
    max_ent = np.log2(factorial(m))
    if max_ent == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)) / max_ent)


def _butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = fs / 2.0
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def calc_pac_mi(x, fs, f_phase=(4, 8), f_amp=(30, 50), n_bins=18):
    """Phase-Amplitude Coupling via Tort Modulation Index (KL divergence)."""
    b_p, a_p = _butter_bandpass(f_phase[0], f_phase[1], fs)
    b_a, a_a = _butter_bandpass(f_amp[0], f_amp[1], fs)
    x_phase = signal.filtfilt(b_p, a_p, x)
    x_amp = signal.filtfilt(b_a, a_a, x)
    phase = np.angle(signal.hilbert(x_phase))
    amp = np.abs(signal.hilbert(x_amp))
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    mean_amp = np.zeros(n_bins)
    for bi in range(n_bins):
        idx = (phase >= bin_edges[bi]) & (phase < bin_edges[bi + 1])
        mean_amp[bi] = amp[idx].mean() if idx.sum() > 0 else 0
    total = mean_amp.sum()
    if total == 0:
        return 0.0, mean_amp
    p = mean_amp / total
    uniform = np.ones(n_bins) / n_bins
    kl = np.sum(p[p > 0] * np.log(p[p > 0] / uniform[p > 0]))
    mi = kl / np.log(n_bins)
    return float(mi), mean_amp


def calc_graph_metrics(matrix):
    """Compute graph theory metrics from a connectivity matrix.
    Returns: clustering_coeff, char_path_length, global_efficiency, local_efficiency
    """
    from scipy.sparse.csgraph import shortest_path
    n = matrix.shape[0]
    if n < 2:
        return 0, 0, 0, 0

    W = matrix.copy()
    np.fill_diagonal(W, 0)
    W = np.nan_to_num(W, nan=0.0)
    W = np.maximum(W, 0)

    # Clustering coefficient (weighted)
    A = (W > 0).astype(float)
    deg = A.sum(axis=1)
    W_third = np.cbrt(W)
    numerator = np.diag(W_third @ W_third @ W_third)
    denom = deg * (deg - 1)
    cc = np.where(denom > 0, numerator / denom, 0)
    clustering = float(np.mean(cc))

    # Distance matrix (inverse weights)
    with np.errstate(divide='ignore'):
        dist = np.where(W > 0, 1.0 / W, 0)
    dist[dist == 0] = np.inf
    np.fill_diagonal(dist, 0)
    sp = shortest_path(dist, directed=False)

    # Characteristic path length
    mask = ~np.eye(n, dtype=bool)
    finite_paths = sp[mask]
    finite_paths = finite_paths[np.isfinite(finite_paths)]
    cpl = float(np.mean(finite_paths)) if len(finite_paths) > 0 else float('inf')

    # Global efficiency
    with np.errstate(divide='ignore'):
        inv_sp = np.where((sp > 0) & np.isfinite(sp), 1.0 / sp, 0)
    np.fill_diagonal(inv_sp, 0)
    ge = float(inv_sp.sum() / (n * (n - 1))) if n > 1 else 0

    # Local efficiency
    le_arr = np.zeros(n)
    for i in range(n):
        neighbors = np.where(A[i] > 0)[0]
        k = len(neighbors)
        if k < 2:
            continue
        sub = dist[np.ix_(neighbors, neighbors)]
        sp_sub = shortest_path(sub, directed=False)
        with np.errstate(divide='ignore'):
            inv_sub = np.where((sp_sub > 0) & np.isfinite(sp_sub),
                               1.0 / sp_sub, 0)
        np.fill_diagonal(inv_sub, 0)
        le_arr[i] = inv_sub.sum() / (k * (k - 1))
    local_eff = float(np.mean(le_arr))

    return clustering, cpl, ge, local_eff

# ============================================================
# Cached data loading
# ============================================================

@st.cache_data(show_spinner="Vital 파일 로딩 중...")
def load_vital(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.vital') as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        vf = vitaldb.VitalFile(tmp_path)
        track_names = vf.get_track_names()

        # dtstart
        dtstart = getattr(vf, 'dtstart', None) or 0

        # Events from type==5 tracks & identify event track names
        events = []
        event_track_names = set()
        try:
            trk_items = vf.trks.items() if isinstance(vf.trks, dict) else []
            for tname, trk in trk_items:
                if getattr(trk, 'type', None) == 5:
                    event_track_names.add(tname)
                    if hasattr(trk, 'recs'):
                        for rec in trk.recs:
                            # recs are dicts: {'dt': float, 'val': str}
                            if isinstance(rec, dict):
                                ev_time = rec.get('dt')
                                ev_name = rec.get('val') or rec.get('sval')
                            else:
                                ev_time = getattr(rec, 'dt', None)
                                ev_name = getattr(rec, 'val', None) or getattr(rec, 'sval', None)
                            if ev_time is not None and ev_name is not None and dtstart:
                                rel_sec = ev_time - dtstart
                                if rel_sec >= 0:
                                    events.append({
                                        'name': str(ev_name).strip(),
                                        'time_sec': float(rel_sec),
                                        'time_min': float(rel_sec / 60),
                                    })
        except Exception as e:
            logging.warning(f"이벤트 파싱 중 오류 발생: {e}")

        # Classify tracks (exclude event tracks)
        eeg_track_names = []
        numeric_track_names = []
        for name in track_names:
            if name in event_track_names:
                continue
            short = name.split('/')[-1] if '/' in name else name
            if 'EEG' in short.upper():
                eeg_track_names.append(name)
            else:
                numeric_track_names.append(name)

        # Numeric data (1 Hz)
        numeric_data = {}
        for trk_name in numeric_track_names:
            try:
                vals = vf.to_numpy(trk_name, 1)
                if vals is not None and len(vals) > 0:
                    numeric_data[trk_name] = vals.flatten()
            except Exception as e:
                logging.warning("수치 트랙 '%s' 로딩 실패: %s", trk_name, e)

        # EEG data (128 Hz)
        eeg_data = {}
        eeg_labels = []
        for trk_name in eeg_track_names:
            try:
                vals = vf.to_numpy(trk_name, 1 / EEG_FS)
                if vals is not None and len(vals) > 0:
                    label = trk_name.split('/')[-1] if '/' in trk_name else trk_name
                    eeg_data[label] = vals.flatten()
                    eeg_labels.append(label)
            except Exception as e:
                logging.warning("EEG 트랙 '%s' 로딩 실패: %s", trk_name, e)

        # Trim & clean
        duration_sec = 0.0
        if eeg_data:
            min_len = min(len(v) for v in eeg_data.values())
            for k in eeg_data:
                eeg_data[k] = np.nan_to_num(eeg_data[k][:min_len], nan=0.0)
            duration_sec = min_len / EEG_FS

        return {
            'track_names': track_names,
            'eeg_track_names': eeg_track_names,
            'numeric_track_names': numeric_track_names,
            'eeg_labels': eeg_labels,
            'eeg_data': eeg_data,
            'numeric_data': numeric_data,
            'events': events,
            'dtstart': dtstart,
            'duration_sec': duration_sec,
        }
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@st.cache_data(show_spinner="연결성 분석 중...")
def compute_connectivity_all(file_bytes, pairs_tuple):
    data = load_vital(file_bytes)
    eeg = data['eeg_data']
    freq_results = {m: {} for m in METRIC_LIST}
    band_results = {m: {} for m in METRIC_LIST}

    for ch_a, ch_b in pairs_tuple:
        if ch_a not in eeg or ch_b not in eeg:
            continue
        plabel = f"{ch_a}-{ch_b}"
        freqs, Sxy, Sxx, Syy = compute_cross_spectra(eeg[ch_a], eeg[ch_b], EEG_FS)
        for mname, mfunc in METRICS.items():
            vals = mfunc(Sxy, Sxx, Syy)
            freq_results[mname][plabel] = (freqs, vals)
            band_results[mname][plabel] = {}
            for bname, (f_lo, f_hi) in BANDS.items():
                bmask = (freqs >= f_lo) & (freqs <= f_hi)
                band_results[mname][plabel][bname] = float(np.mean(vals[bmask]))

    return freq_results, band_results


@st.cache_data(show_spinner="시간별 연결성 계산 중...")
def compute_time_connectivity(file_bytes, ch_a, ch_b):
    data = load_vital(file_bytes)
    eeg = data['eeg_data']
    if ch_a not in eeg or ch_b not in eeg:
        return None, None
    x_full = eeg[ch_a]
    y_full = eeg[ch_b]

    win_sec, step_sec = WIN_SEC, STEP_SEC
    win_samp = int(win_sec * EEG_FS)
    step_samp = int(step_sec * EEG_FS)
    n_win = (len(x_full) - win_samp) // step_samp + 1
    if n_win <= 0:
        return None, None

    time_pts = []
    time_conn = {m: {b: [] for b in BANDS} for m in METRIC_LIST}
    for w in range(n_win):
        s = w * step_samp
        e = s + win_samp
        time_pts.append((s + e) / 2 / EEG_FS / 60)
        fw, Sxy, Sxx, Syy = compute_cross_spectra(x_full[s:e], y_full[s:e], EEG_FS)
        for mname, mfunc in METRICS.items():
            vals = mfunc(Sxy, Sxx, Syy)
            for bname, (f_lo, f_hi) in BANDS.items():
                bmask = (fw >= f_lo) & (fw <= f_hi)
                time_conn[mname][bname].append(float(np.mean(vals[bmask])))

    return np.array(time_pts), time_conn


@st.cache_data(show_spinner="고급 분석 계산 중...")
def compute_advanced_analysis(file_bytes):
    """Compute entropy time series, band power ratios, and PAC matrix."""
    data = load_vital(file_bytes)
    eeg = data['eeg_data']
    eeg_labels = data['eeg_labels']

    win_sec, step_sec = WIN_SEC, STEP_SEC
    win_samp = int(win_sec * EEG_FS)
    step_samp = int(step_sec * EEG_FS)

    # Determine number of windows from shortest channel
    if not eeg:
        return {}
    min_len = min(len(v) for v in eeg.values())
    n_win = (min_len - win_samp) // step_samp + 1
    if n_win <= 0:
        return {}

    time_pts = np.array([(w * step_samp + win_samp / 2) / EEG_FS / 60
                         for w in range(n_win)])

    spectral_entropy = {ch: np.zeros(n_win) for ch in eeg_labels}
    perm_entropy = {ch: np.zeros(n_win) for ch in eeg_labels}
    band_power_ratios = {ch: {'alpha_delta': np.zeros(n_win),
                              'theta_beta': np.zeros(n_win),
                              'dt_ab': np.zeros(n_win)} for ch in eeg_labels}

    for ch in eeg_labels:
        x = eeg[ch]
        for w in range(n_win):
            s = w * step_samp
            e = s + win_samp
            seg = x[s:e]

            spectral_entropy[ch][w] = calc_spectral_entropy(seg, EEG_FS)
            perm_entropy[ch][w] = calc_permutation_entropy(seg, m=3, tau=1)

            # Band power ratios
            freqs, psd = compute_welch_psd(seg, fs=EEG_FS)
            bp = {}
            for bname, (fl, fh) in BANDS.items():
                bm = (freqs >= fl) & (freqs <= fh)
                bp[bname] = _trapz(psd[bm], freqs[bm])

            d, t, a, b = bp['Delta'], bp['Theta'], bp['Alpha'], bp['Beta']
            band_power_ratios[ch]['alpha_delta'][w] = a / d if d > 0 else 0
            band_power_ratios[ch]['theta_beta'][w] = t / b if b > 0 else 0
            ab = a + b
            band_power_ratios[ch]['dt_ab'][w] = (d + t) / ab if ab > 0 else 0

    # PAC: comodulogram for each channel (full data)
    phase_freqs = np.arange(2, 21, 2)   # 2-20 Hz
    amp_freqs = np.arange(20, 51, 5)     # 20-50 Hz
    pac_results = {}
    pac_mean_amp = {}  # cache mean_amp per channel for Theta->Gamma PAC
    for ch in eeg_labels:
        x = eeg[ch]
        pac_mat = np.zeros((len(phase_freqs), len(amp_freqs)))
        for pi, pf in enumerate(phase_freqs):
            for ai, af in enumerate(amp_freqs):
                try:
                    mi, _ = calc_pac_mi(x, EEG_FS,
                                        f_phase=(max(pf - 1, 0.5), pf + 1),
                                        f_amp=(max(af - 2, 1), af + 2))
                    pac_mat[pi, ai] = mi
                except Exception as e:
                    logging.debug("PAC 계산 실패 (ch=%s, pf=%s, af=%s): %s", ch, pf, af, e)
                    pac_mat[pi, ai] = 0
        pac_results[ch] = pac_mat
        # Pre-compute Theta->Gamma PAC mean_amp (used for bar chart in Tab3)
        try:
            _, mean_amp_ch = calc_pac_mi(x, EEG_FS, f_phase=(4, 8), f_amp=(30, 50))
            pac_mean_amp[ch] = mean_amp_ch
        except Exception as e:
            logging.warning("Theta->Gamma PAC 계산 실패 (ch=%s): %s", ch, e)
            pac_mean_amp[ch] = np.zeros(18)

    return {
        'time_pts': time_pts,
        'spectral_entropy': spectral_entropy,
        'perm_entropy': perm_entropy,
        'band_power_ratios': band_power_ratios,
        'pac_results': pac_results,
        'pac_mean_amp': pac_mean_amp,
        'phase_freqs': phase_freqs,
        'amp_freqs': amp_freqs,
    }


# ============================================================
# MAIN APP
# ============================================================
st.title("Vital EEG 분석")

# --- Sidebar ---
with st.sidebar:
    st.header("파일 업로드")
    uploaded = st.file_uploader("`.vital` 파일 선택", type=['vital'])

    if uploaded is None:
        st.info("`.vital` 파일을 업로드하세요.")
        st.stop()

    file_bytes = uploaded.getvalue()
    data = load_vital(file_bytes)

    # File info
    st.subheader("파일 정보")
    st.write(f"**파일명:** {uploaded.name}")
    st.write(f"**전체 트랙:** {len(data['track_names'])}개")
    st.write(f"**EEG 트랙:** {len(data['eeg_labels'])}개")
    st.write(f"**수치 트랙:** {len(data['numeric_track_names'])}개")
    if data['duration_sec'] > 0:
        st.write(f"**기록 시간:** {data['duration_sec']:.0f}초 "
                 f"({data['duration_sec']/60:.1f}분)")

    # Events
    if data['events']:
        st.subheader("이벤트")
        for ev in data['events']:
            st.write(f"- **{ev['name']}** @ {ev['time_min']:.2f}분")
    else:
        st.caption("이벤트 없음")

    # Channel pair selection
    eeg_labels = data['eeg_labels']
    all_pairs = list(combinations(eeg_labels, 2))
    all_pair_labels = [f"{a}-{b}" for a, b in all_pairs]

    # Default: key pairs
    key_set = {'EEG_L1-EEG_R1', 'EEG_L2-EEG_R2', 'EEG_L-EEG_R',
               'EEG_L1-EEG_L2', 'EEG_R1-EEG_R2'}
    default_sel = [p for p in all_pair_labels if p in key_set] or all_pair_labels[:5]

    st.subheader("분석 채널 쌍")
    selected_pair_labels = st.multiselect(
        "채널 쌍 선택", all_pair_labels, default=default_sel)

    if not selected_pair_labels:
        st.warning("채널 쌍을 1개 이상 선택하세요.")
        st.stop()

    selected_pairs = tuple(
        tuple(p.split('-', 1)) for p in selected_pair_labels)

# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "파워 스펙트럼 분석", "기능적 연결성 분석", "고급 분석", "데이터 요약"])

events = data['events']
eeg_data = data['eeg_data']
numeric_data = data['numeric_data']

# ============================================================
# TAB 1: Power Spectrum Analysis
# ============================================================
with tab1:
    if not eeg_data:
        st.info("EEG 데이터가 없습니다. Vital 파일에 EEG 트랙이 포함되어 있는지 확인해 주세요.")
    if eeg_data:
        # --- 1a. Spectrogram (맨 위) ---
        st.subheader("스펙트로그램")
        n_eeg = len(eeg_labels)
        spec_labels = [l for l in eeg_labels if l in eeg_data]
        n_spec = len(spec_labels)
        fig, axes = plt.subplots(n_spec, 1, figsize=(16, 4 * n_spec),
                                 squeeze=False)
        axes = axes.flatten()
        fig.suptitle('EEG Spectrogram', fontsize=14, fontweight='bold')
        for i, label in enumerate(spec_labels):
            vals = eeg_data[label]
            f, t, Sxx = signal.spectrogram(vals, fs=EEG_FS, nperseg=256,
                                           noverlap=192, window='hann')
            fmask = (f >= 0.5) & (f <= 50)
            Sxx_db = 10 * np.log10(Sxx[fmask, :] + 1e-12)
            im = axes[i].pcolormesh(t / 60.0, f[fmask], Sxx_db,
                                    shading='gouraud', cmap='jet')
            axes[i].set_ylabel('Frequency (Hz)')
            axes[i].set_title(label, fontsize=11, fontweight='bold')
            plt.colorbar(im, ax=axes[i], label='Power (dB)')
            add_event_lines(axes[i], events, style='spectrogram')
        axes[-1].set_xlabel('Time (min)')
        plt.tight_layout()
        show_fig(fig, "spectrogram.png", "dl_spec")

    # --- 1b. Numeric time series ---
    if numeric_data:
        st.subheader("수치 트랙 시계열")
        n_num = len(numeric_data)
        fig, axes = plt.subplots(n_num, 1, figsize=(16, max(3 * n_num, 6)),
                                 sharex=True, squeeze=False)
        axes = axes.flatten()
        fig.suptitle('Vital Signs Time Series', fontsize=14, fontweight='bold')
        for i, (trk_name, vals) in enumerate(numeric_data.items()):
            time_min = np.arange(len(vals)) / 60.0
            axes[i].plot(time_min, vals, linewidth=0.8)
            short = trk_name.split('/')[-1] if '/' in trk_name else trk_name
            axes[i].set_ylabel(short, fontsize=10, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
            valid = vals[~np.isnan(vals)]
            if len(valid) > 0:
                axes[i].set_title(
                    f'{short} (min={valid.min():.1f}, max={valid.max():.1f}, '
                    f'mean={valid.mean():.1f})', fontsize=9, loc='right')
            add_event_lines(axes[i], events)
        axes[-1].set_xlabel('Time (min)')
        plt.tight_layout()
        show_fig(fig, "numeric_timeseries.png", "dl_numeric")

    # --- 1c. EEG waveforms ---
    if eeg_data:
        st.subheader("EEG 파형 (처음 30초)")
        fig, axes = plt.subplots(n_eeg, 1, figsize=(16, max(2.5 * n_eeg, 6)),
                                 sharex=True, squeeze=False)
        axes = axes.flatten()
        fig.suptitle('EEG Waveforms (First 30 sec)', fontsize=14, fontweight='bold')
        for i, label in enumerate(eeg_labels):
            vals = eeg_data[label]
            t_sec = np.arange(len(vals)) / EEG_FS
            show_n = min(EEG_FS * 30, len(vals))
            axes[i].plot(t_sec[:show_n], vals[:show_n], linewidth=0.5,
                         color='darkblue')
            axes[i].set_ylabel(label, fontsize=10, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
            add_event_lines(axes[i], events, x_axis='sec')
        axes[-1].set_xlabel('Time (sec)')
        plt.tight_layout()
        show_fig(fig, "eeg_waveforms.png", "dl_eeg_wave")

        # --- 1d. PSD ---
        st.subheader("파워 스펙트럼 밀도 (PSD)")
        n_cols = 2
        n_rows = (n_eeg + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows),
                                 squeeze=False)
        fig.suptitle('EEG Power Spectral Density (Welch)', fontsize=14,
                     fontweight='bold')
        for idx, label in enumerate(eeg_labels):
            ax = axes[idx // n_cols, idx % n_cols]
            vals = eeg_data[label]
            if len(vals) > 256:
                freqs, psd = compute_welch_psd(vals, fs=EEG_FS)
                mask = (freqs >= 0.5) & (freqs <= 50)
                ax.semilogy(freqs[mask], psd[mask], 'k-', linewidth=1)
                for (bname, (fl, fh)), bc in zip(BANDS.items(), BAND_COLORS_LIST):
                    bm = (freqs >= fl) & (freqs <= fh)
                    ax.fill_between(freqs[bm], psd[bm], alpha=0.4, color=bc,
                                    label=bname)
                if idx == 0:
                    ax.legend(fontsize=7, loc='upper right')
            ax.set_title(label, fontsize=11, fontweight='bold')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('PSD (V\u00b2/Hz)')
            ax.grid(True, alpha=0.3)
        # Hide unused axes
        for idx in range(n_eeg, n_rows * n_cols):
            axes[idx // n_cols, idx % n_cols].set_visible(False)
        plt.tight_layout()
        show_fig(fig, "psd_analysis.png", "dl_psd")

        # --- 1e. Band power bar chart ---
        st.subheader("주파수 대역별 파워")
        # Pick up to 2 representative channels (L, R or first two)
        bp_labels = []
        for candidate in ['EEG_L', 'EEG_R']:
            if candidate in eeg_data:
                bp_labels.append(candidate)
        if len(bp_labels) < 2:
            for lbl in eeg_labels:
                if lbl not in bp_labels:
                    bp_labels.append(lbl)
                if len(bp_labels) >= 2:
                    break
        n_bp = len(bp_labels)
        fig, axes = plt.subplots(1, n_bp, figsize=(7 * n_bp, 6), squeeze=False)
        axes = axes.flatten()
        fig.suptitle('EEG Band Power Comparison', fontsize=14, fontweight='bold')
        for idx, label in enumerate(bp_labels):
            vals = eeg_data[label]
            freqs, psd = compute_welch_psd(vals, fs=EEG_FS)
            bpowers, bnames = [], []
            for bname, (fl, fh) in BANDS.items():
                bm = (freqs >= fl) & (freqs <= fh)
                bpowers.append(_trapz(psd[bm], freqs[bm]))
                bnames.append(bname)
            bars = axes[idx].bar(bnames, bpowers, color=BAND_COLORS_LIST,
                                 edgecolor='black', linewidth=0.5)
            axes[idx].set_title(label, fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Absolute Power (V\u00b2)')
            total = sum(bpowers)
            if total > 0:
                for bar, pw in zip(bars, bpowers):
                    pct = pw / total * 100
                    axes[idx].text(bar.get_x() + bar.get_width() / 2,
                                   bar.get_height(), f'{pct:.1f}%',
                                   ha='center', va='bottom', fontsize=9,
                                   fontweight='bold')
        plt.tight_layout()
        show_fig(fig, "band_power.png", "dl_bandpow")

# ============================================================
# TAB 2: Functional Connectivity
# ============================================================
with tab2:
    if not eeg_data:
        st.info("EEG 데이터가 없습니다. Vital 파일에 EEG 트랙이 포함되어 있는지 확인해 주세요.")
    elif len(eeg_labels) < 2:
        st.warning("기능적 연결성 분석에는 EEG 채널이 2개 이상 필요합니다. "
                    f"현재 {len(eeg_labels)}개 채널만 감지되었습니다.")
    else:
        # Compute connectivity
        freq_results, band_results = compute_connectivity_all(
            file_bytes, selected_pairs)

        # --- 2a. Frequency-resolved connectivity ---
        st.subheader("주파수별 연결성")
        fig, axes = plt.subplots(3, 2, figsize=(18, 15), squeeze=False)
        fig.suptitle('Frequency-Resolved Functional Connectivity',
                     fontsize=14, fontweight='bold')
        for idx, mname in enumerate(METRIC_LIST):
            ax = axes[idx // 2, idx % 2]
            for pidx, plabel in enumerate(selected_pair_labels):
                if plabel in freq_results[mname]:
                    f, v = freq_results[mname][plabel]
                    fmask = (f >= 0.5) & (f <= 50)
                    c = CONN_LINE_COLORS[pidx % len(CONN_LINE_COLORS)]
                    ax.plot(f[fmask], v[fmask], linewidth=1.2, label=plabel,
                            color=c, alpha=0.85)
            for bname, (fl, fh) in BANDS.items():
                ax.axvspan(fl, fh, alpha=0.08, color=BAND_COLORS[bname])
            ax.set_title(mname, fontsize=12, fontweight='bold')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel(mname)
            ax.legend(fontsize=7, loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0.5, 50)
        plt.tight_layout()
        show_fig(fig, "connectivity_frequency.png", "dl_conn_freq")

        # --- 2b. Connectivity matrix heatmaps ---
        st.subheader("연결성 매트릭스")
        sel_metric = st.selectbox("지표 선택", METRIC_LIST, key="mat_metric")

        fig, axes = plt.subplots(1, 5, figsize=(22, 4), squeeze=False)
        axes = axes.flatten()
        fig.suptitle(f'{sel_metric} Connectivity Matrix by Band',
                     fontsize=13, fontweight='bold')
        n_ch = len(eeg_labels)
        for bidx, bname in enumerate(BANDS):
            diag = 1.0 if sel_metric != 'dPLI' else 0.5
            mat = build_connectivity_matrix(eeg_labels, band_results,
                                            sel_metric, bname,
                                            diagonal_val=diag, dpli_flip=True)

            off_diag = mat[~np.eye(n_ch, dtype=bool)]
            off_valid = off_diag[~np.isnan(off_diag)]
            if sel_metric == 'dPLI':
                vmin, vmax, cmap = 0, 1, 'RdBu_r'
            else:
                vmin = 0
                vmax = np.nanmax(off_valid) * 1.1 if len(off_valid) > 0 and np.nanmax(off_valid) > 0 else 1
                cmap = 'YlOrRd'

            im = axes[bidx].imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap,
                                   aspect='equal')
            short_labels = [l.replace('EEG_', '') for l in eeg_labels]
            axes[bidx].set_xticks(range(n_ch))
            axes[bidx].set_yticks(range(n_ch))
            axes[bidx].set_xticklabels(short_labels, fontsize=8, rotation=45)
            axes[bidx].set_yticklabels(short_labels, fontsize=8)
            axes[bidx].set_title(bname, fontsize=11, fontweight='bold')
            for ii in range(n_ch):
                for jj in range(n_ch):
                    if ii != jj and not np.isnan(mat[ii, jj]):
                        axes[bidx].text(
                            jj, ii, f'{mat[ii, jj]:.2f}', ha='center',
                            va='center', fontsize=6,
                            color='black' if mat[ii, jj] < vmax * 0.7 else 'white')
            plt.colorbar(im, ax=axes[bidx], fraction=0.046, pad=0.04)
        plt.tight_layout()
        show_fig(fig, f"matrix_{sel_metric}.png", "dl_matrix")

        # --- 2c. Band-level summary bar chart ---
        st.subheader("종합 비교 (대역별)")
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)
        fig.suptitle('Connectivity Summary by Frequency Band',
                     fontsize=14, fontweight='bold')
        n_bands = len(BANDS)
        bar_width = 0.8 / max(len(selected_pair_labels), 1)
        x = np.arange(n_bands)
        for idx, mname in enumerate(METRIC_LIST):
            ax = axes[idx // 3, idx % 3]
            for pidx, plabel in enumerate(selected_pair_labels):
                if plabel in band_results[mname]:
                    vals = [band_results[mname][plabel][b] for b in BANDS]
                    c = CONN_LINE_COLORS[pidx % len(CONN_LINE_COLORS)]
                    ax.bar(x + pidx * bar_width, vals, bar_width,
                           label=plabel, color=c, alpha=0.8,
                           edgecolor='black', linewidth=0.3)
            ax.set_title(mname, fontsize=12, fontweight='bold')
            ax.set_xticks(x + bar_width * (len(selected_pair_labels) - 1) / 2)
            ax.set_xticklabels(list(BANDS.keys()), fontsize=9)
            ax.set_ylabel(mname)
            ax.grid(True, alpha=0.2, axis='y')
            if idx == 0:
                ax.legend(fontsize=6, loc='upper right')
        plt.tight_layout()
        show_fig(fig, "connectivity_summary.png", "dl_conn_summary")

        # --- 2d. Time-varying connectivity ---
        st.subheader("시간별 연결성 변화")
        # Select pair for time-varying
        default_tv = 'EEG_L-EEG_R' if 'EEG_L-EEG_R' in selected_pair_labels else selected_pair_labels[0]
        tv_pair = st.selectbox("채널 쌍", selected_pair_labels,
                               index=selected_pair_labels.index(default_tv),
                               key="tv_pair")
        tv_a, tv_b = tv_pair.split('-', 1)
        time_pts, time_conn = compute_time_connectivity(file_bytes, tv_a, tv_b)

        if time_pts is not None and len(time_pts) > 0:
            fig, axes = plt.subplots(6, 1, figsize=(16, 20), sharex=True,
                                     squeeze=False)
            axes = axes.flatten()
            fig.suptitle(f'Time-Varying Connectivity ({tv_pair}, 30s window)',
                         fontsize=14, fontweight='bold')
            for idx, mname in enumerate(METRIC_LIST):
                ax = axes[idx]
                for bname in BANDS:
                    ax.plot(time_pts, time_conn[mname][bname], linewidth=1.2,
                            label=bname, color=BAND_COLORS[bname])
                ax.set_ylabel(mname, fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8, loc='upper right', ncol=5)
                add_event_lines(ax, events)
            axes[-1].set_xlabel('Time (min)')
            plt.tight_layout()
            show_fig(fig, f"time_connectivity_{tv_pair}.png", "dl_time_conn")
        else:
            st.warning("데이터가 충분하지 않습니다 (최소 30초 필요).")

# ============================================================
# TAB 3: Advanced Analysis
# ============================================================
with tab3:
    if not eeg_data:
        st.info("EEG 데이터가 없습니다. Vital 파일에 EEG 트랙이 포함되어 있는지 확인해 주세요.")
    else:
        st.caption("엔트로피, PAC, 그래프 이론 등 고급 분석을 수행합니다. 데이터 길이에 따라 수십 초 이상 소요될 수 있습니다.")
        adv = compute_advanced_analysis(file_bytes)
        if not adv:
            st.warning("데이터가 충분하지 않습니다 (최소 30초 필요).")
        else:
            adv_time = adv['time_pts']
            adv_se = adv['spectral_entropy']
            adv_pe = adv['perm_entropy']
            adv_bpr = adv['band_power_ratios']
            adv_pac = adv['pac_results']
            adv_pf = adv['phase_freqs']
            adv_af = adv['amp_freqs']

            # ---- A. Entropy Time Series ----
            st.subheader("A. 엔트로피 시계열")
            col_se, col_pe = st.columns(2)
            with col_se:
                st.markdown("**Spectral Entropy**")
                fig, ax = plt.subplots(figsize=(8, 4))
                for ch in eeg_labels:
                    ax.plot(adv_time, adv_se[ch], linewidth=1, label=ch)
                ax.set_xlabel('Time (min)')
                ax.set_ylabel('Spectral Entropy')
                ax.set_ylim(0, 1)
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
                ax.set_title('Spectral Entropy', fontweight='bold')
                add_event_lines(ax, events)
                plt.tight_layout()
                show_fig(fig, "spectral_entropy.png", "dl_se")

            with col_pe:
                st.markdown("**Permutation Entropy**")
                fig, ax = plt.subplots(figsize=(8, 4))
                for ch in eeg_labels:
                    ax.plot(adv_time, adv_pe[ch], linewidth=1, label=ch)
                ax.set_xlabel('Time (min)')
                ax.set_ylabel('Permutation Entropy')
                ax.set_ylim(0, 1)
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
                ax.set_title('Permutation Entropy', fontweight='bold')
                add_event_lines(ax, events)
                plt.tight_layout()
                show_fig(fig, "permutation_entropy.png", "dl_pe")

            # ---- B. Band Power Ratios ----
            st.subheader("B. 상대 대역 파워 비율")
            ratio_names = [
                ('alpha_delta', 'Alpha/Delta Ratio'),
                ('theta_beta', 'Theta/Beta Ratio'),
                ('dt_ab', '(Delta+Theta)/(Alpha+Beta)')
            ]
            # Identify EEG_L / EEG_R channels
            lr_channels = [ch for ch in eeg_labels if 'EEG_L' in ch or 'EEG_R' in ch]
            if len(lr_channels) < 2:
                lr_channels = eeg_labels[:2]

            fig, axes = plt.subplots(1, 3, figsize=(18, 5), squeeze=False)
            axes = axes.flatten()
            fig.suptitle('Band Power Ratios (30s window)', fontsize=14,
                         fontweight='bold')
            for ri, (rkey, rname) in enumerate(ratio_names):
                ax = axes[ri]
                for ch in lr_channels:
                    ax.plot(adv_time, adv_bpr[ch][rkey], linewidth=1, label=ch)
                ax.set_xlabel('Time (min)')
                ax.set_ylabel(rname)
                ax.set_title(rname, fontweight='bold')
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
                add_event_lines(ax, events)
            plt.tight_layout()
            show_fig(fig, "band_power_ratios.png", "dl_bpr")

            # ---- C. Phase-Amplitude Coupling ----
            st.subheader("C. Phase-Amplitude Coupling (PAC)")
            # Comodulogram heatmaps
            n_pac_ch = len(eeg_labels)
            n_cols_pac = min(n_pac_ch, 4)
            n_rows_pac = (n_pac_ch + n_cols_pac - 1) // n_cols_pac
            fig, axes = plt.subplots(n_rows_pac, n_cols_pac,
                                     figsize=(5 * n_cols_pac, 4 * n_rows_pac),
                                     squeeze=False)
            fig.suptitle('PAC Comodulogram (Modulation Index)',
                         fontsize=14, fontweight='bold')
            for ci, ch in enumerate(eeg_labels):
                ax = axes[ci // n_cols_pac, ci % n_cols_pac]
                im = ax.imshow(adv_pac[ch], aspect='auto', origin='lower',
                               cmap='hot', interpolation='nearest')
                ax.set_xticks(range(len(adv_af)))
                ax.set_xticklabels(adv_af, fontsize=8)
                ax.set_yticks(range(len(adv_pf)))
                ax.set_yticklabels(adv_pf, fontsize=8)
                ax.set_xlabel('Amplitude Freq (Hz)')
                ax.set_ylabel('Phase Freq (Hz)')
                ax.set_title(ch, fontweight='bold')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            # Hide unused
            for ci in range(n_pac_ch, n_rows_pac * n_cols_pac):
                axes[ci // n_cols_pac, ci % n_cols_pac].set_visible(False)
            plt.tight_layout()
            show_fig(fig, "pac_comodulogram.png", "dl_pac_como")

            # Representative channel PAC bar chart (uses cached mean_amp)
            rep_ch = eeg_labels[0]
            st.markdown(f"**대표 채널 위상-진폭 분포 ({rep_ch}, Theta→Gamma)**")
            mean_amp = adv.get('pac_mean_amp', {}).get(rep_ch)
            if mean_amp is None:
                _, mean_amp = calc_pac_mi(eeg_data[rep_ch], EEG_FS,
                                          f_phase=(4, 8), f_amp=(30, 50))
            n_bins_pac = len(mean_amp)
            bin_centers = np.linspace(-180, 180, n_bins_pac, endpoint=False)
            bin_centers += 180 / n_bins_pac
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(bin_centers, mean_amp, width=360 / n_bins_pac * 0.8,
                   color='steelblue', edgecolor='black', linewidth=0.5)
            ax.set_xlabel('Phase (degrees)')
            ax.set_ylabel('Mean Amplitude')
            ax.set_title(f'Phase-Amplitude Distribution ({rep_ch})',
                         fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            show_fig(fig, "pac_phase_amp_dist.png", "dl_pac_dist")

            # ---- D. Graph Theory Metrics ----
            st.subheader("D. 그래프 이론 지표")
            if len(eeg_labels) >= 2:
                freq_results_g, band_results_g = compute_connectivity_all(
                    file_bytes, selected_pairs)
                graph_metric_names = ['Clustering Coeff', 'Char Path Length',
                                      'Global Efficiency', 'Local Efficiency']
                sel_conn_metric = st.selectbox(
                    "연결성 지표 선택 (그래프 계산용)", METRIC_LIST,
                    key="graph_conn_metric")

                graph_rows = []
                for bname in BANDS:
                    mat = build_connectivity_matrix(eeg_labels, band_results_g,
                                                    sel_conn_metric, bname)
                    cc, cpl, ge, le = calc_graph_metrics(mat)
                    graph_rows.append({
                        'Band': bname,
                        'Clustering Coeff': f"{cc:.4f}",
                        'Char Path Length': f"{cpl:.4f}",
                        'Global Efficiency': f"{ge:.4f}",
                        'Local Efficiency': f"{le:.4f}",
                    })

                df_graph = pd.DataFrame(graph_rows)
                show_table(df_graph, "graph_metrics.csv", "dl_graph_table")

                # Bar chart
                fig, axes = plt.subplots(1, 4, figsize=(20, 5), squeeze=False)
                axes = axes.flatten()
                fig.suptitle(f'Graph Metrics ({sel_conn_metric})',
                             fontsize=14, fontweight='bold')
                band_names = list(BANDS.keys())
                for gi, gm_name in enumerate(graph_metric_names):
                    vals = [float(df_graph.iloc[bi][gm_name])
                            for bi in range(len(band_names))]
                    axes[gi].bar(band_names, vals, color=BAND_COLORS_LIST,
                                 edgecolor='black', linewidth=0.5)
                    axes[gi].set_title(gm_name, fontweight='bold')
                    axes[gi].set_ylabel(gm_name)
                    axes[gi].grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                show_fig(fig, "graph_metrics_bar.png", "dl_graph_bar")
            else:
                st.warning("그래프 이론 분석에는 EEG 채널 2개 이상 필요합니다.")

# ============================================================
# TAB 4: Data Summary
# ============================================================
with tab4:
    if not eeg_data:
        st.info("EEG 데이터가 없습니다. Vital 파일에 EEG 트랙이 포함되어 있는지 확인해 주세요.")
    # --- 3a. EEG summary ---
    st.subheader("EEG 채널 요약")
    if eeg_data:
        rows = []
        for label in eeg_labels:
            vals = eeg_data[label]
            valid = vals[vals != 0]  # non-zero (nan was replaced with 0)
            rows.append({
                'Channel': label,
                'Total Samples': len(vals),
                'Duration (sec)': f"{len(vals)/EEG_FS:.1f}",
                'Duration (min)': f"{len(vals)/EEG_FS/60:.1f}",
                'Min': f"{valid.min():.6f}" if len(valid) > 0 else 'N/A',
                'Max': f"{valid.max():.6f}" if len(valid) > 0 else 'N/A',
                'Std': f"{valid.std():.6f}" if len(valid) > 0 else 'N/A',
            })
        df_eeg = pd.DataFrame(rows)
        show_table(df_eeg, "eeg_summary.csv", "dl_eeg_sum")

    # --- 3b. Band power table ---
    st.subheader("대역별 파워")
    if eeg_data:
        bp_rows = []
        for label in eeg_labels:
            vals = eeg_data[label]
            if len(vals) <= 256:
                continue
            freqs, psd = compute_welch_psd(vals, fs=EEG_FS)
            row = {'Channel': label}
            total = 0
            band_powers = {}
            for bname, (fl, fh) in BANDS.items():
                bm = (freqs >= fl) & (freqs <= fh)
                pw = _trapz(psd[bm], freqs[bm])
                band_powers[bname] = pw
                row[f'{bname} (V\u00b2)'] = f"{pw:.6f}"
                total += pw
            for bname in BANDS:
                row[f'{bname} (%)'] = f"{band_powers[bname]/total*100:.1f}" if total > 0 else "0.0"
            bp_rows.append(row)
        if bp_rows:
            df_bp = pd.DataFrame(bp_rows)
            show_table(df_bp, "band_power.csv", "dl_bp_table")

    # --- 3c. Connectivity summary ---
    st.subheader("연결성 요약")
    if eeg_data and len(eeg_labels) < 2:
        st.caption("연결성 분석에는 EEG 채널이 2개 이상 필요합니다.")
    if eeg_data and len(eeg_labels) >= 2:
        freq_results, band_results = compute_connectivity_all(
            file_bytes, selected_pairs)
        conn_rows = []
        for plabel in selected_pair_labels:
            for mname in METRIC_LIST:
                if plabel in band_results[mname]:
                    row = {'Pair': plabel, 'Metric': mname}
                    for bname in BANDS:
                        row[bname] = f"{band_results[mname][plabel][bname]:.4f}"
                    conn_rows.append(row)
        if conn_rows:
            df_conn = pd.DataFrame(conn_rows)
            show_table(df_conn, "connectivity_summary.csv", "dl_conn_table")

    # --- 4d. Entropy summary ---
    st.subheader("엔트로피 요약")
    adv_sum = None
    if eeg_data:
        adv_sum = compute_advanced_analysis(file_bytes)
        if adv_sum:
            ent_rows = []
            for ch in eeg_labels:
                ent_rows.append({
                    'Channel': ch,
                    'Spectral Entropy (mean)': f"{np.mean(adv_sum['spectral_entropy'][ch]):.4f}",
                    'Spectral Entropy (std)': f"{np.std(adv_sum['spectral_entropy'][ch]):.4f}",
                    'Permutation Entropy (mean)': f"{np.mean(adv_sum['perm_entropy'][ch]):.4f}",
                    'Permutation Entropy (std)': f"{np.std(adv_sum['perm_entropy'][ch]):.4f}",
                })
            df_ent = pd.DataFrame(ent_rows)
            show_table(df_ent, "entropy_summary.csv", "dl_ent_sum")

    # --- 4e. Band power ratio summary ---
    st.subheader("대역 파워 비율 요약")
    if eeg_data and adv_sum:
        bpr_rows = []
        for ch in eeg_labels:
            bpr = adv_sum['band_power_ratios'][ch]
            bpr_rows.append({
                'Channel': ch,
                'Alpha/Delta (mean)': f"{np.mean(bpr['alpha_delta']):.4f}",
                'Alpha/Delta (std)': f"{np.std(bpr['alpha_delta']):.4f}",
                'Theta/Beta (mean)': f"{np.mean(bpr['theta_beta']):.4f}",
                'Theta/Beta (std)': f"{np.std(bpr['theta_beta']):.4f}",
                '(D+T)/(A+B) (mean)': f"{np.mean(bpr['dt_ab']):.4f}",
                '(D+T)/(A+B) (std)': f"{np.std(bpr['dt_ab']):.4f}",
            })
        df_bpr = pd.DataFrame(bpr_rows)
        show_table(df_bpr, "band_power_ratio_summary.csv", "dl_bpr_sum")

    # --- 4f. Graph metrics summary ---
    st.subheader("그래프 이론 지표 요약")
    if eeg_data and len(eeg_labels) < 2:
        st.caption("그래프 이론 분석에는 EEG 채널이 2개 이상 필요합니다.")
    if eeg_data and len(eeg_labels) >= 2:
        freq_results_s, band_results_s = compute_connectivity_all(
            file_bytes, selected_pairs)
        gm_rows = []
        for conn_m in METRIC_LIST:
            for bname in BANDS:
                mat = build_connectivity_matrix(eeg_labels, band_results_s,
                                                conn_m, bname)
                cc, cpl, ge, le = calc_graph_metrics(mat)
                gm_rows.append({
                    'Connectivity': conn_m,
                    'Band': bname,
                    'Clustering Coeff': f"{cc:.4f}",
                    'Char Path Length': f"{cpl:.4f}",
                    'Global Efficiency': f"{ge:.4f}",
                    'Local Efficiency': f"{le:.4f}",
                })
        df_gm = pd.DataFrame(gm_rows)
        show_table(df_gm, "graph_metrics_summary.csv", "dl_gm_sum")
