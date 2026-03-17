"""Vercel Serverless Function for Vital EEG analysis API."""
import json
import tempfile
import os
import base64
import urllib.request
import numpy as np
from scipy import signal
from scipy.sparse.csgraph import shortest_path
from scipy.ndimage import uniform_filter1d
from itertools import combinations
from collections import Counter
from math import factorial
from http.server import BaseHTTPRequestHandler
from supabase import create_client

# ── Supabase ─────────────────────────────────────────────────
SUPABASE_URL = "https://tcyapfwczeuhcqecxpdi.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRjeWFwZndjemV1aGNxZWN4cGRpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzI3NjczMDUsImV4cCI6MjA4ODM0MzMwNX0.b5oXpe23asLqGLCnHkXLf9LB2W6OoH7nBwkBN0n6dZA"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── constants ────────────────────────────────────────────────
EEG_FS = 128
WIN_SEC, STEP_SEC = 30, 10
BANDS = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 13),
         "Beta": (13, 30), "Gamma": (30, 50)}
BAND_NAMES = list(BANDS.keys())
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _to_list(obj):
    if isinstance(obj, np.ndarray):  return obj.tolist()
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, dict):        return {k: _to_list(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_list(v) for v in obj]
    return obj


# ── signal processing ───────────────────────────────────────

def compute_welch_psd(x, fs=EEG_FS):
    n = min(256, len(x))
    return signal.welch(x, fs=fs, nperseg=n, noverlap=min(128, n // 2), window="hann")


def compute_cross_spectra(x, y, fs, nperseg=256, noverlap=192):
    step = nperseg - noverlap
    n_ep = (len(x) - nperseg) // step + 1
    if n_ep <= 0:
        f = np.fft.rfftfreq(nperseg, 1 / fs)
        z = np.zeros((1, len(f)))
        return f, z, z, z
    win = signal.windows.hann(nperseg)
    f = np.fft.rfftfreq(nperseg, 1 / fs)
    xw = np.lib.stride_tricks.sliding_window_view(x, nperseg)[::step][:n_ep] * win
    yw = np.lib.stride_tricks.sliding_window_view(y, nperseg)[::step][:n_ep] * win
    X, Y = np.fft.rfft(xw, axis=1), np.fft.rfft(yw, axis=1)
    return f, X * np.conj(Y), np.abs(X)**2, np.abs(Y)**2


def calc_coherence(Sxy, Sxx, Syy):
    n = np.abs(np.mean(Sxy, 0))**2
    d = np.mean(Sxx, 0) * np.mean(Syy, 0)
    return np.where(d > 0, n / d, 0)

def calc_pli(Sxy, *_):  return np.abs(np.mean(np.sign(np.imag(Sxy)), 0))
def calc_dpli(Sxy, *_): return np.mean((np.imag(Sxy) > 0).astype(float), 0)

def calc_wpli(Sxy, *_):
    im = np.imag(Sxy)
    return np.where((d := np.mean(np.abs(im), 0)) > 0, np.abs(np.mean(im, 0)) / d, 0)

def calc_dwpli(Sxy, *_):
    im = np.imag(Sxy)
    s1, s2, s3 = np.sum(im, 0), np.sum(np.abs(im), 0), np.sum(im**2, 0)
    return np.where((d := s2**2 - s3) > 0, (s1**2 - s3) / d, 0)

def calc_ple(Sxy, *_, n_bins=12):
    phase = np.angle(Sxy)
    ple = np.zeros(phase.shape[1])
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    mx = np.log(n_bins)
    for fi in range(phase.shape[1]):
        h, _ = np.histogram(phase[:, fi], bins=edges)
        p = h / h.sum() if h.sum() > 0 else h
        p = p[p > 0]
        ple[fi] = (-np.sum(p * np.log(p)) / mx) if len(p) > 0 else 0
    return ple

METRICS = {"Coherence": calc_coherence, "PLI": calc_pli, "dPLI": calc_dpli,
           "wPLI": calc_wpli, "dwPLI": calc_dwpli, "PLE": calc_ple}
METRIC_LIST = list(METRICS.keys())


# ── advanced ────────────────────────────────────────────────

def calc_spectral_entropy(x, fs):
    _, psd = compute_welch_psd(x, fs=fs)
    pn = psd / psd.sum() if psd.sum() > 0 else psd
    pn = pn[pn > 0]
    return float(-np.sum(pn * np.log2(pn)) / np.log2(len(pn))) if len(pn) > 1 else 0.0

def calc_perm_entropy(x, m=3, tau=1):
    n = len(x)
    if n < m * tau: return 0.0
    pats = [tuple(np.argsort(x[i:i + m * tau:tau])) for i in range(n - (m - 1) * tau)]
    counts = Counter(pats)
    probs = np.array([c / len(pats) for c in counts.values()])
    mx = np.log2(factorial(m))
    return float(-np.sum(probs * np.log2(probs)) / mx) if mx > 0 else 0.0

def _butter_bp(lo, hi, fs, order=3):
    nyq = fs / 2.0
    return signal.butter(order, [max(lo / nyq, 0.001), min(hi / nyq, 0.999)], btype="band")

def calc_pac_mi(x, fs, f_phase=(4, 8), f_amp=(30, 50), n_bins=18):
    bp, ap = _butter_bp(f_phase[0], f_phase[1], fs)
    ba, aa = _butter_bp(f_amp[0], f_amp[1], fs)
    phase = np.angle(signal.hilbert(signal.filtfilt(bp, ap, x)))
    amp = np.abs(signal.hilbert(signal.filtfilt(ba, aa, x)))
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    ma = np.array([amp[(phase >= edges[i]) & (phase < edges[i+1])].mean()
                   if ((phase >= edges[i]) & (phase < edges[i+1])).sum() > 0 else 0
                   for i in range(n_bins)])
    tot = ma.sum()
    if tot == 0: return 0.0, ma
    p = ma / tot
    u = np.ones(n_bins) / n_bins
    kl = np.sum(p[p > 0] * np.log(p[p > 0] / u[p > 0]))
    return float(kl / np.log(n_bins)), ma

def calc_graph_metrics(mat):
    n = mat.shape[0]
    if n < 2: return 0, 0, 0, 0
    W = np.nan_to_num(np.maximum(mat.copy(), 0))
    np.fill_diagonal(W, 0)
    A = (W > 0).astype(float)
    deg = A.sum(1)
    W3 = np.cbrt(W)
    cc = np.where((d := deg * (deg - 1)) > 0, np.diag(W3 @ W3 @ W3) / d, 0)
    with np.errstate(divide="ignore"):
        dist = np.where(W > 0, 1.0 / W, 0)
    dist[dist == 0] = np.inf
    np.fill_diagonal(dist, 0)
    sp = shortest_path(dist, directed=False)
    mask = ~np.eye(n, dtype=bool)
    fp = sp[mask]; fp = fp[np.isfinite(fp)]
    cpl = float(np.mean(fp)) if len(fp) > 0 else float("inf")
    with np.errstate(divide="ignore"):
        inv_sp = np.where((sp > 0) & np.isfinite(sp), 1.0 / sp, 0)
    np.fill_diagonal(inv_sp, 0)
    ge = float(inv_sp.sum() / (n * (n - 1))) if n > 1 else 0
    le_arr = np.zeros(n)
    for i in range(n):
        nb = np.where(A[i] > 0)[0]
        k = len(nb)
        if k < 2: continue
        sub_sp = shortest_path(dist[np.ix_(nb, nb)], directed=False)
        with np.errstate(divide="ignore"):
            inv_sub = np.where((sub_sp > 0) & np.isfinite(sub_sp), 1.0 / sub_sp, 0)
        np.fill_diagonal(inv_sub, 0)
        le_arr[i] = inv_sub.sum() / (k * (k - 1))
    return float(np.mean(cc)), cpl, ge, float(np.mean(le_arr))

def build_conn_matrix(labels, br, metric, band, diag=0, dpli_flip=False):
    n = len(labels)
    mat = np.full((n, n), float(diag))
    for i in range(n):
        for j in range(n):
            if i == j: continue
            pl, pr = f"{labels[i]}-{labels[j]}", f"{labels[j]}-{labels[i]}"
            if pl in br[metric]:
                mat[i, j] = br[metric][pl].get(band, 0)
            elif pr in br[metric]:
                v = br[metric][pr].get(band, 0)
                mat[i, j] = (1 - v) if dpli_flip and metric == "dPLI" else v
            else:
                mat[i, j] = 0
    return mat


# ── file bytes helper ───────────────────────────────────────

def get_file_bytes(body):
    """Get vital file bytes from either storagePath or fileBase64."""
    if "storagePath" in body and body["storagePath"]:
        url = f"{SUPABASE_URL}/storage/v1/object/public/vital-files/{body['storagePath']}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            return resp.read()
    elif "fileBase64" in body and body["fileBase64"]:
        return base64.b64decode(body["fileBase64"])
    else:
        raise ValueError("No file provided (storagePath or fileBase64 required)")


# ── vital file loading ──────────────────────────────────────

def load_vital(file_bytes):
    import vitaldb
    with tempfile.NamedTemporaryFile(delete=False, suffix=".vital") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        vf = vitaldb.VitalFile(tmp_path)
        track_names = vf.get_track_names()
        dtstart = getattr(vf, "dtstart", None) or 0
        events, ev_tracks = [], set()
        try:
            for tname, trk in (vf.trks.items() if isinstance(vf.trks, dict) else []):
                if getattr(trk, "type", None) == 5:
                    ev_tracks.add(tname)
                    for rec in getattr(trk, "recs", []):
                        et = rec.get("dt") if isinstance(rec, dict) else getattr(rec, "dt", None)
                        en = (rec.get("val") or rec.get("sval")) if isinstance(rec, dict) else (getattr(rec, "val", None) or getattr(rec, "sval", None))
                        if et is not None and en is not None and dtstart:
                            rs = et - dtstart
                            if rs >= 0:
                                events.append({"name": str(en).strip(), "time_sec": float(rs), "time_min": float(rs / 60)})
        except Exception: pass

        eeg_t, num_t = [], []
        for nm in track_names:
            if nm in ev_tracks: continue
            short = nm.split("/")[-1] if "/" in nm else nm
            (eeg_t if "EEG" in short.upper() else num_t).append(nm)

        numeric_data = {}
        for tn in num_t:
            try:
                v = vf.to_numpy(tn, 1)
                if v is not None and len(v) > 0: numeric_data[tn] = v.flatten()
            except Exception: pass

        eeg_data, eeg_labels = {}, []
        for tn in eeg_t:
            try:
                v = vf.to_numpy(tn, 1 / EEG_FS)
                if v is not None and len(v) > 0:
                    label = tn.split("/")[-1] if "/" in tn else tn
                    eeg_data[label] = v.flatten()
                    eeg_labels.append(label)
            except Exception: pass

        if eeg_data:
            ml = min(len(v) for v in eeg_data.values())
            for k in eeg_data:
                eeg_data[k] = np.nan_to_num(eeg_data[k][:ml], nan=0.0)
            duration_sec = ml / EEG_FS
        else:
            duration_sec = 0.0

        return {"track_names": track_names, "eeg_labels": eeg_labels,
                "eeg_data": eeg_data, "numeric_data": numeric_data,
                "events": events, "duration_sec": duration_sec}
    finally:
        try: os.unlink(tmp_path)
        except Exception: pass


# ── action handlers ─────────────────────────────────────────

def action_load(body):
    fb = get_file_bytes(body)
    d = load_vital(fb)
    eeg_prev = {l: arr[::8].tolist() for l, arr in d["eeg_data"].items()}
    num_prev = {}
    for tn, arr in d["numeric_data"].items():
        short = tn.split("/")[-1] if "/" in tn else tn
        num_prev[short] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).tolist()
    return {"eegLabels": d["eeg_labels"], "trackNames": d["track_names"],
            "events": d["events"], "durationSec": d["duration_sec"],
            "numericTracks": list(d["numeric_data"].keys()),
            "eegPreview": eeg_prev, "numericPreview": num_prev,
            "nSamples": {l: len(a) for l, a in d["eeg_data"].items()}}


def action_spectrum(body):
    fb = get_file_bytes(body)
    d = load_vital(fb)
    result = {"psd": {}, "bandPower": {}, "spectrogram": {}}
    for label in d["eeg_labels"]:
        vals = d["eeg_data"][label]
        if len(vals) <= 256: continue
        freqs, psd = compute_welch_psd(vals)
        mask = (freqs >= 0.5) & (freqs <= 50)
        result["psd"][label] = {"freqs": freqs[mask].tolist(), "psd": psd[mask].tolist()}
        bp, total = {}, 0
        for bn, (fl, fh) in BANDS.items():
            bm = (freqs >= fl) & (freqs <= fh)
            pw = float(_trapz(psd[bm], freqs[bm]))
            bp[bn] = pw; total += pw
        result["bandPower"][label] = {
            "absolute": bp,
            "percent": {b: round(v / total * 100, 1) if total > 0 else 0 for b, v in bp.items()}}
        f, t, Sxx = signal.spectrogram(vals, fs=EEG_FS, nperseg=256, noverlap=192, window="hann")
        fm = (f >= 0.5) & (f <= 50)
        Sxx_db = 10 * np.log10(Sxx[fm, :] + 1e-12)
        step = max(1, Sxx_db.shape[1] // 500)
        result["spectrogram"][label] = {
            "freqs": f[fm].tolist(), "times": (t[::step] / 60).tolist(),
            "power": Sxx_db[:, ::step].tolist()}
    return result


def action_connectivity(body):
    fb = get_file_bytes(body)
    d = load_vital(fb)
    eeg, labels = d["eeg_data"], d["eeg_labels"]
    pairs = body.get("pairs") or [list(p) for p in combinations(labels, 2)]
    freq_r = {m: {} for m in METRIC_LIST}
    band_r = {m: {} for m in METRIC_LIST}
    for ca, cb in pairs:
        if ca not in eeg or cb not in eeg: continue
        pl = f"{ca}-{cb}"
        freqs, Sxy, Sxx, Syy = compute_cross_spectra(eeg[ca], eeg[cb], EEG_FS)
        fmask = (freqs >= 0.5) & (freqs <= 50)
        for mn, mf in METRICS.items():
            vals = mf(Sxy, Sxx, Syy)
            freq_r[mn][pl] = {"freqs": freqs[fmask].tolist(), "values": vals[fmask].tolist()}
            band_r[mn][pl] = {}
            for bn, (fl, fh) in BANDS.items():
                bm = (freqs >= fl) & (freqs <= fh)
                band_r[mn][pl][bn] = float(np.mean(vals[bm]))
    matrices = {}
    if len(labels) >= 2:
        for mn in METRIC_LIST:
            matrices[mn] = {}
            for bn in BAND_NAMES:
                diag = 0.5 if mn == "dPLI" else 1.0
                mat = build_conn_matrix(labels, band_r, mn, bn, diag=diag, dpli_flip=(mn == "dPLI"))
                matrices[mn][bn] = mat.tolist()
    return {"freqResults": _to_list(freq_r), "bandResults": _to_list(band_r),
            "matrices": matrices, "labels": labels}


def action_time_conn(body):
    fb = get_file_bytes(body)
    d = load_vital(fb)
    eeg = d["eeg_data"]
    ca, cb = body["chA"], body["chB"]
    if ca not in eeg or cb not in eeg: return {"error": "channel not found"}
    x, y = eeg[ca], eeg[cb]
    ws, ss = int(WIN_SEC * EEG_FS), int(STEP_SEC * EEG_FS)
    nw = (len(x) - ws) // ss + 1
    if nw <= 0: return {"error": "data too short"}
    tp = []
    tc = {m: {b: [] for b in BAND_NAMES} for m in METRIC_LIST}
    for w in range(nw):
        s = w * ss; e = s + ws
        tp.append((s + e) / 2 / EEG_FS / 60)
        fw, Sxy, Sxx, Syy = compute_cross_spectra(x[s:e], y[s:e], EEG_FS)
        for mn, mf in METRICS.items():
            vals = mf(Sxy, Sxx, Syy)
            for bn, (fl, fh) in BANDS.items():
                bm = (fw >= fl) & (fw <= fh)
                tc[mn][bn].append(float(np.mean(vals[bm])))
    return {"timePts": tp, "timeConn": tc}


def action_advanced(body):
    fb = get_file_bytes(body)
    d = load_vital(fb)
    eeg, labels = d["eeg_data"], d["eeg_labels"]
    if not eeg: return {"error": "no EEG data"}
    ws, ss = int(WIN_SEC * EEG_FS), int(STEP_SEC * EEG_FS)
    ml = min(len(v) for v in eeg.values())
    nw = (ml - ws) // ss + 1
    if nw <= 0: return {"error": "data too short"}
    tp = [(w * ss + ws / 2) / EEG_FS / 60 for w in range(nw)]

    se, pe, bpr = {}, {}, {}
    for ch in labels:
        x = eeg[ch]
        se[ch], pe[ch] = [], []
        bpr[ch] = {"alpha_delta": [], "theta_beta": [], "dt_ab": []}
        for w in range(nw):
            seg = x[w * ss: w * ss + ws]
            se[ch].append(calc_spectral_entropy(seg, EEG_FS))
            pe[ch].append(calc_perm_entropy(seg))
            freqs, psd = compute_welch_psd(seg)
            bp = {}
            for bn, (fl, fh) in BANDS.items():
                bm = (freqs >= fl) & (freqs <= fh)
                bp[bn] = float(_trapz(psd[bm], freqs[bm]))
            dd, tt, aa, bb = bp["Delta"], bp["Theta"], bp["Alpha"], bp["Beta"]
            bpr[ch]["alpha_delta"].append(aa / dd if dd > 0 else 0)
            bpr[ch]["theta_beta"].append(tt / bb if bb > 0 else 0)
            ab = aa + bb
            bpr[ch]["dt_ab"].append((dd + tt) / ab if ab > 0 else 0)

    pac, pac_bar = {}, {}
    pf_list, af_list = list(range(2, 21, 2)), list(range(20, 51, 5))
    for ch in labels:
        x = eeg[ch]
        mat = []
        for pf in pf_list:
            row = []
            for af in af_list:
                try:
                    mi, _ = calc_pac_mi(x, EEG_FS, f_phase=(max(pf-1, 0.5), pf+1), f_amp=(max(af-2, 1), af+2))
                    row.append(mi)
                except Exception: row.append(0)
            mat.append(row)
        pac[ch] = mat
        try:
            _, ma = calc_pac_mi(x, EEG_FS, f_phase=(4, 8), f_amp=(30, 50))
            pac_bar[ch] = ma.tolist()
        except Exception: pac_bar[ch] = [0] * 18

    pairs = body.get("pairs") or [list(p) for p in combinations(labels, 2)]
    br = {m: {} for m in METRIC_LIST}
    for ca, cb in pairs:
        if ca not in eeg or cb not in eeg: continue
        pl = f"{ca}-{cb}"
        freqs, Sxy, Sxx, Syy = compute_cross_spectra(eeg[ca], eeg[cb], EEG_FS)
        for mn, mf in METRICS.items():
            vals = mf(Sxy, Sxx, Syy)
            br[mn][pl] = {}
            for bn, (fl, fh) in BANDS.items():
                bm = (freqs >= fl) & (freqs <= fh)
                br[mn][pl][bn] = float(np.mean(vals[bm]))
    graph = {}
    if len(labels) >= 2:
        for mn in METRIC_LIST:
            graph[mn] = {}
            for bn in BAND_NAMES:
                mat = build_conn_matrix(labels, br, mn, bn)
                cc, cpl, ge, le = calc_graph_metrics(mat)
                graph[mn][bn] = {"cc": cc, "cpl": cpl, "ge": ge, "le": le}

    return {"timePts": tp, "spectralEntropy": se, "permEntropy": pe,
            "bandPowerRatios": bpr, "pac": pac, "pacBar": pac_bar,
            "phaseFreqs": pf_list, "ampFreqs": af_list,
            "graphMetrics": graph, "labels": labels}


# ── Supabase CRUD actions ────────────────────────────────────

def action_save(body):
    row = {
        "file_name": body.get("fileName", "unknown"),
        "eeg_labels": body.get("eegLabels", []),
        "track_names": body.get("trackNames", []),
        "events": body.get("events", []),
        "duration_sec": body.get("durationSec", 0),
        "n_samples": body.get("nSamples", {}),
        "numeric_preview": body.get("numericPreview", {}),
        "spectrum_result": body.get("spectrumResult"),
        "connectivity_result": body.get("connectivityResult"),
        "advanced_result": body.get("advancedResult"),
        "memo": body.get("memo", ""),
    }
    result = supabase.table("vital_items").insert(row).execute()
    return {"success": True, "id": result.data[0]["id"] if result.data else None}


def action_list_items(body):
    result = (supabase.table("vital_items")
              .select("id,file_name,eeg_labels,duration_sec,memo,created_at")
              .order("created_at", desc=True)
              .execute())
    return {"items": result.data}


def action_get_item(body):
    item_id = body.get("id")
    if not item_id:
        return {"error": "id required"}
    result = supabase.table("vital_items").select("*").eq("id", item_id).execute()
    if not result.data:
        return {"error": "not found"}
    return {"item": result.data[0]}


def action_delete_item(body):
    item_id = body.get("id")
    if not item_id:
        return {"error": "id required"}
    supabase.table("vital_items").delete().eq("id", item_id).execute()
    return {"success": True}


def action_update_memo(body):
    item_id = body.get("id")
    memo = body.get("memo", "")
    if not item_id:
        return {"error": "id required"}
    supabase.table("vital_items").update({"memo": memo, "updated_at": "now()"}).eq("id", item_id).execute()
    return {"success": True}


def action_ppg2abp(body):
    import vitaldb
    fb = get_file_bytes(body)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".vital") as tmp:
        tmp.write(fb)
        tmp_path = tmp.name

    try:
        vf = vitaldb.VitalFile(tmp_path)
        track_names = vf.get_track_names()

        # Find PLETH and IBP1 tracks
        pleth_track = next((t for t in track_names if 'PLETH' in t.upper()), None)
        ibp1_track = next((t for t in track_names if 'IBP1' in t.upper()), None)

        if not pleth_track or not ibp1_track:
            return {"error": "PLETH or IBP1 track not found", "tracks": track_names}

        SR = 125
        pleth = vf.to_numpy(pleth_track, 1/SR)
        ibp1 = vf.to_numpy(ibp1_track, 1/SR)

        if pleth is None or ibp1 is None:
            return {"error": "Failed to extract PLETH or IBP1 data"}

        pleth = pleth.flatten()
        ibp1 = ibp1.flatten()

        # Align lengths
        min_len = min(len(pleth), len(ibp1))
        pleth = pleth[:min_len]
        ibp1 = ibp1[:min_len]

        # Interpolate NaN gaps (same as step1)
        for sig in [pleth, ibp1]:
            nans = np.isnan(sig)
            if nans.any() and (~nans).sum() > 10:
                sig[nans] = np.interp(np.where(nans)[0], np.where(~nans)[0], sig[~nans])
            else:
                sig[nans] = 0.0

        # Save raw IBP1 stats before filtering (for display)
        raw_ibp1_mean = float(np.mean(ibp1[ibp1 != 0])) if np.any(ibp1 != 0) else 0.0

        # Bandpass filter (matching step1 preprocessing exactly)
        from scipy.signal import butter, filtfilt
        def bp_filter(sig, lo, hi, fs, order=4):
            nyq = fs / 2
            b, a = butter(order, [lo / nyq, hi / nyq], btype='band')
            return filtfilt(b, a, sig)

        # Apply same filters as training: PLETH 0.5-8Hz, IBP1 0.5-40Hz
        try:
            pleth = bp_filter(pleth, 0.5, 8.0, SR).astype(np.float32)
            ibp1 = bp_filter(ibp1, 0.5, 40.0, SR).astype(np.float32)
        except Exception:
            pass  # fallback to unfiltered if signal too short

        # Segment into 10s windows
        seg_len = SR * 10  # 1250 points
        n_segs = min_len // seg_len
        if n_segs == 0:
            return {"error": "Data too short for segmentation"}

        pleth_segs = pleth[:n_segs * seg_len].reshape(n_segs, seg_len)
        ibp1_segs = ibp1[:n_segs * seg_len].reshape(n_segs, seg_len)

        # Quality filter - remove segments with flat signals
        good_segs = []
        for i in range(n_segs):
            if np.std(pleth_segs[i]) > 1e-6 and np.std(ibp1_segs[i]) > 1e-6:
                good_segs.append(i)

        if len(good_segs) < 2:
            return {"error": "Not enough valid segments"}

        pleth_segs = pleth_segs[good_segs]
        ibp1_segs = ibp1_segs[good_segs]
        n_segs = len(good_segs)

        # Extract SBP/DBP from each segment (in filtered domain, zero-centered)
        true_sbp_filt = np.array([seg.max() for seg in ibp1_segs])
        true_dbp_filt = np.array([seg.min() for seg in ibp1_segs])
        # For display, add raw offset back to get physiological mmHg values
        true_sbp = true_sbp_filt + raw_ibp1_mean
        true_dbp = true_dbp_filt + raw_ibp1_mean
        true_mbp = np.array([seg.mean() for seg in ibp1_segs]) + raw_ibp1_mean

        # ---- Model 1: Baseline (Mean Prediction) ----
        train_n = int(n_segs * 0.8)
        test_n = n_segs - train_n

        mean_sbp = float(np.mean(true_sbp[:train_n]))
        mean_dbp = float(np.mean(true_dbp[:train_n]))
        baseline_sbp_pred = np.full(test_n, mean_sbp)
        baseline_dbp_pred = np.full(test_n, mean_dbp)

        # ---- Model 2: Linear Regression (PPG features -> BP) ----
        # Extract simple features: mean, std, max, min, peak-to-peak
        def extract_features(segs):
            feats = []
            for s in segs:
                feats.append([np.mean(s), np.std(s), np.max(s), np.min(s),
                              np.max(s) - np.min(s), np.median(s)])
            return np.array(feats)

        train_feats = extract_features(pleth_segs[:train_n])
        test_feats = extract_features(pleth_segs[train_n:])

        # Add bias
        train_X = np.column_stack([train_feats, np.ones(train_n)])
        test_X = np.column_stack([test_feats, np.ones(test_n)])

        # Least squares for SBP and DBP
        try:
            w_sbp, _, _, _ = np.linalg.lstsq(train_X, true_sbp[:train_n], rcond=None)
            w_dbp, _, _, _ = np.linalg.lstsq(train_X, true_dbp[:train_n], rcond=None)
            lr_sbp_pred = test_X @ w_sbp
            lr_dbp_pred = test_X @ w_dbp
        except:
            lr_sbp_pred = baseline_sbp_pred.copy()
            lr_dbp_pred = baseline_dbp_pred.copy()

        # ---- Model 3: Adaptive Waveform Scaling ----
        # Learn amplitude/offset mapping from PPG to ABP
        train_pleth_flat = pleth_segs[:train_n].flatten()
        train_ibp1_flat = ibp1_segs[:train_n].flatten()

        p_mu, p_sig = np.mean(train_pleth_flat), np.std(train_pleth_flat) + 1e-8
        a_mu, a_sig = np.mean(train_ibp1_flat), np.std(train_ibp1_flat) + 1e-8

        # Scale PPG to ABP range
        adaptive_recon = []
        for seg in pleth_segs[train_n:]:
            normalized = (seg - p_mu) / p_sig
            reconstructed = normalized * a_sig + a_mu
            adaptive_recon.append(reconstructed)
        adaptive_recon = np.array(adaptive_recon)

        # ---- Model 4: Frequency-domain Transfer Function (Wiener) ----
        nfft = seg_len
        Sxy = np.zeros(nfft // 2 + 1, dtype=complex)
        Sxx = np.zeros(nfft // 2 + 1)
        n_tf = min(train_n, 200)
        for i in range(n_tf):
            X = np.fft.rfft(pleth_segs[i])
            Y = np.fft.rfft(ibp1_segs[i])
            Sxy += Y * np.conj(X)
            Sxx += np.abs(X) ** 2
        # Wiener-style regularized transfer function
        reg = np.mean(Sxx) * 0.01
        H_avg = Sxy / (Sxx + reg)

        H_smooth_real = uniform_filter1d(np.real(H_avg), size=7)
        H_smooth_imag = uniform_filter1d(np.imag(H_avg), size=7)
        H_smooth = H_smooth_real + 1j * H_smooth_imag

        # Clip magnitude to prevent blow-up
        H_mag = np.abs(H_smooth)
        H_clip = np.where(H_mag > 50, H_smooth * 50 / (H_mag + 1e-10), H_smooth)

        abp_mu = np.mean(ibp1_segs[:train_n])
        abp_std = np.std(ibp1_segs[:train_n]) + 1e-8
        tf_recon = []
        for seg in pleth_segs[train_n:]:
            X = np.fft.rfft(seg)
            Y_est = X * H_clip
            recon = np.fft.irfft(Y_est, n=seg_len)
            # Normalize then rescale to ABP range
            r_mu, r_std = np.mean(recon), np.std(recon) + 1e-8
            recon = (recon - r_mu) / r_std * abp_std + abp_mu
            recon = np.clip(recon, abp_mu - 4 * abp_std, abp_mu + 4 * abp_std)
            tf_recon.append(recon)
        tf_recon = np.array(tf_recon)

        # ---- Compute Metrics for Test Set ----
        test_true_sbp = true_sbp[train_n:]
        test_true_dbp = true_dbp[train_n:]
        test_ibp1 = ibp1_segs[train_n:]
        test_pleth = pleth_segs[train_n:]

        models = {}

        # Baseline
        bl_sbp_mae = float(np.mean(np.abs(baseline_sbp_pred - test_true_sbp)))
        bl_dbp_mae = float(np.mean(np.abs(baseline_dbp_pred - test_true_dbp)))
        models['Baseline (Mean)'] = {'sbp_mae': bl_sbp_mae, 'dbp_mae': bl_dbp_mae, 'wf_corr': 0.0}

        # Linear Regression
        lr_sbp_mae = float(np.mean(np.abs(lr_sbp_pred - test_true_sbp)))
        lr_dbp_mae = float(np.mean(np.abs(lr_dbp_pred - test_true_dbp)))
        models['Linear Regression'] = {'sbp_mae': lr_sbp_mae, 'dbp_mae': lr_dbp_mae, 'wf_corr': 0.0}

        # Adaptive Scaling
        ad_sbp = np.array([s.max() for s in adaptive_recon]) + raw_ibp1_mean
        ad_dbp = np.array([s.min() for s in adaptive_recon]) + raw_ibp1_mean
        ad_corrs = [float(np.corrcoef(test_ibp1[i], adaptive_recon[i])[0,1])
                    for i in range(test_n) if not np.isnan(np.corrcoef(test_ibp1[i], adaptive_recon[i])[0,1])]
        models['Adaptive Scaling'] = {
            'sbp_mae': float(np.mean(np.abs(ad_sbp - test_true_sbp))),
            'dbp_mae': float(np.mean(np.abs(ad_dbp - test_true_dbp))),
            'wf_corr': float(np.mean(ad_corrs)) if ad_corrs else 0.0
        }

        # Transfer Function
        tf_sbp = np.array([s.max() for s in tf_recon]) + raw_ibp1_mean
        tf_dbp = np.array([s.min() for s in tf_recon]) + raw_ibp1_mean
        tf_corrs = [float(np.corrcoef(test_ibp1[i], tf_recon[i])[0,1])
                    for i in range(test_n) if not np.isnan(np.corrcoef(test_ibp1[i], tf_recon[i])[0,1])]
        models['Transfer Function'] = {
            'sbp_mae': float(np.mean(np.abs(tf_sbp - test_true_sbp))),
            'dbp_mae': float(np.mean(np.abs(tf_dbp - test_true_dbp))),
            'wf_corr': float(np.mean(tf_corrs)) if tf_corrs else 0.0
        }

        # Reference results from our deep learning experiments
        reference_models = {
            'XGBoost (LOSO)': {'sbp_mae': 14.98, 'dbp_mae': 8.66, 'sbp_r2': -0.19, 'dbp_bhs': 'D'},
            'LightGBM (LOSO)': {'sbp_mae': 16.09, 'dbp_mae': 8.42, 'sbp_r2': -0.34, 'dbp_bhs': 'D'},
            'Improved CNN (GPU)': {'sbp_mae': 12.64, 'dbp_mae': 7.20, 'sbp_r2': 0.17, 'dbp_bhs': 'C'},
            'ResNet1D (GPU)': {'sbp_mae': 12.25, 'dbp_mae': 7.83, 'sbp_r2': 0.20, 'dbp_bhs': 'D'},
            'CNN-LSTM (GPU)': {'wf_corr': 0.562, 'wf_rmse': 13.86, 'type': 'waveform'},
            'U-Net 1D (GPU)': {'wf_corr': 0.566, 'wf_rmse': 13.70, 'type': 'waveform'},
        }

        # Select example segments for visualization (5 segments from test set)
        n_show = min(5, test_n)
        show_idx = np.linspace(0, test_n - 1, n_show, dtype=int)

        # Downsample waveforms for JSON transfer (every 5th point)
        ds = 5
        _off = raw_ibp1_mean
        waveform_examples = []
        for idx in show_idx:
            t = (np.arange(seg_len) / SR)[::ds].tolist()
            waveform_examples.append({
                'time': t,
                'actual_abp': (ibp1_segs[train_n + idx] + _off)[::ds].tolist(),
                'ppg': pleth_segs[train_n + idx][::ds].tolist(),
                'adaptive': (adaptive_recon[idx] + _off)[::ds].tolist(),
                'transfer_fn': (tf_recon[idx] + _off)[::ds].tolist(),
                'seg_idx': int(good_segs[train_n + idx]),
            })

        # Time series of SBP/DBP for all test segments
        sbp_series = {
            'actual': test_true_sbp.tolist(),
            'baseline': baseline_sbp_pred.tolist(),
            'linear': lr_sbp_pred.tolist(),
            'adaptive': ad_sbp.tolist(),
            'transfer_fn': tf_sbp.tolist(),
        }
        dbp_series = {
            'actual': test_true_dbp.tolist(),
            'baseline': baseline_dbp_pred.tolist(),
            'linear': lr_dbp_pred.tolist(),
            'adaptive': ad_dbp.tolist(),
            'transfer_fn': tf_dbp.tolist(),
        }

        # Per-segment correlation for waveform models
        corr_series = {
            'adaptive': [float(np.corrcoef(test_ibp1[i], adaptive_recon[i])[0,1])
                        if np.std(adaptive_recon[i]) > 1e-8 else 0.0 for i in range(test_n)],
            'transfer_fn': [float(np.corrcoef(test_ibp1[i], tf_recon[i])[0,1])
                           if np.std(tf_recon[i]) > 1e-8 else 0.0 for i in range(test_n)],
        }
        # Clean NaN
        for k in corr_series:
            corr_series[k] = [0.0 if np.isnan(v) else v for v in corr_series[k]]

        # ---- Zoomed Cardiac Cycles (4s windows, 2x2 grid) & Best/Median/Worst ----
        tf_corr_arr = np.array(corr_series['transfer_fn'])
        ad_corr_arr = np.array(corr_series['adaptive'])
        rank_corr = tf_corr_arr.copy()
        rank_corr[np.isnan(rank_corr)] = -1
        sorted_idx = np.argsort(rank_corr)

        best_i = int(sorted_idx[-1])
        worst_i = int(sorted_idx[0])
        median_i = int(sorted_idx[len(sorted_idx) // 2])

        def seg_rmse(a, b):
            return float(np.sqrt(np.mean((a - b) ** 2)))

        # -- Zoomed 4s cardiac windows from 4 diverse good segments --
        top_quarter = sorted_idx[-(len(sorted_idx) // 4):]
        pick4 = np.linspace(0, len(top_quarter) - 1, min(4, len(top_quarter)), dtype=int)
        zoom_indices = [int(top_quarter[p]) for p in pick4]
        zoom_win = int(SR * 4)

        zoomed_panels = []
        for zi in zoom_indices:
            seg_a = test_ibp1[zi]
            seg_ad = adaptive_recon[zi]
            seg_tf = tf_recon[zi]
            seg_ppg = test_pleth[zi]
            mid = seg_len // 2
            s = max(0, mid - zoom_win // 2)
            e = min(seg_len, s + zoom_win)
            t4 = (np.arange(e - s) / SR).tolist()
            corr_ad = float(ad_corr_arr[zi]) if not np.isnan(ad_corr_arr[zi]) else 0.0
            corr_tf = float(tf_corr_arr[zi]) if not np.isnan(tf_corr_arr[zi]) else 0.0
            zoomed_panels.append({
                'seg_idx': int(good_segs[train_n + zi]),
                'time': t4,
                'actual': (seg_a[s:e] + _off).tolist(),
                'adaptive': (seg_ad[s:e] + _off).tolist(),
                'transfer_fn': (seg_tf[s:e] + _off).tolist(),
                'ppg': seg_ppg[s:e].tolist(),
                'corr_adaptive': corr_ad,
                'corr_tf': corr_tf,
                'rmse_adaptive': seg_rmse(seg_a[s:e], seg_ad[s:e]),
                'rmse_tf': seg_rmse(seg_a[s:e], seg_tf[s:e]),
            })

        # -- Best / Median / Worst full 10s segments --
        bmw_cases = []
        for label, idx in [('Best', best_i), ('Median', median_i), ('Worst', worst_i)]:
            seg_a = test_ibp1[idx]
            seg_ad = adaptive_recon[idx]
            seg_tf = tf_recon[idx]
            corr_ad = float(ad_corr_arr[idx]) if not np.isnan(ad_corr_arr[idx]) else 0.0
            corr_tf = float(tf_corr_arr[idx]) if not np.isnan(tf_corr_arr[idx]) else 0.0
            ds2 = 2
            bmw_cases.append({
                'label': label,
                'seg_idx': int(good_segs[train_n + idx]),
                'corr_adaptive': corr_ad,
                'corr_tf': corr_tf,
                'rmse_adaptive': seg_rmse(seg_a, seg_ad),
                'rmse_tf': seg_rmse(seg_a, seg_tf),
                'time': (np.arange(seg_len) / SR)[::ds2].tolist(),
                'actual': (seg_a + _off)[::ds2].tolist(),
                'adaptive': (seg_ad + _off)[::ds2].tolist(),
                'transfer_fn': (seg_tf + _off)[::ds2].tolist(),
            })

        return {
            'hasDL': False,
            'nSegments': n_segs,
            'trainN': train_n,
            'testN': test_n,
            'sr': SR,
            'duration_min': float(min_len / SR / 60),
            'sbpStats': {'mean': float(np.mean(true_sbp)), 'std': float(np.std(true_sbp))},
            'dbpStats': {'mean': float(np.mean(true_dbp)), 'std': float(np.std(true_dbp))},
            'models': _to_list(models),
            'referenceModels': reference_models,
            'zoomedPanels': _to_list(zoomed_panels),
            'bmwCases': _to_list(bmw_cases),
            'waveformExamples': _to_list(waveform_examples),
            'sbpSeries': _to_list(sbp_series),
            'dbpSeries': _to_list(dbp_series),
            'corrSeries': _to_list(corr_series),
        }
    finally:
        try: os.unlink(tmp_path)
        except: pass


ACTIONS = {"load": action_load, "spectrum": action_spectrum,
           "connectivity": action_connectivity, "time_conn": action_time_conn,
           "advanced": action_advanced, "save": action_save,
           "list_items": action_list_items, "get_item": action_get_item,
           "delete_item": action_delete_item, "update_memo": action_update_memo,
           "ppg2abp": action_ppg2abp}


# ── Vercel Serverless Handler ───────────────────────────────

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self):
        action = "load"
        if "?" in self.path:
            for part in self.path.split("?", 1)[1].split("&"):
                if part.startswith("action="):
                    action = part.split("=", 1)[1]

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length > 0 else {}

        fn = ACTIONS.get(action)
        if not fn:
            self._json_resp(400, {"error": f"unknown action: {action}"})
            return
        try:
            result = fn(body)
            self._json_resp(200, result)
        except Exception as e:
            self._json_resp(500, {"error": str(e)})

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json_resp(self, code, data):
        raw = json.dumps(data, ensure_ascii=False)
        raw = raw.replace(': NaN', ': null').replace('[NaN', '[null').replace(',NaN', ',null').replace(', NaN', ', null')
        raw = raw.replace(': Infinity', ': null').replace(': -Infinity', ': null')
        payload = raw.encode("utf-8")
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(payload)
