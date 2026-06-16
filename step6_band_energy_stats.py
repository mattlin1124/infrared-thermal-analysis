"""
Step 6: Band energy bar chart (single-subject) in "paper-like" style

Goal
----
Reproduce a figure similar to the PPT bar chart:
- Bars for MV / LCV / TWVV
- Frequency bands on x-axis
- Error bars from within-subject time-window variability
- Optional significance marks using Mann–Whitney U test across windows (within-subject)

Important note
--------------
This is NOT group-level inference across subjects.
The error bars and p-values come from within-trial window samples (single subject).

Inputs (from Step5)
-------------------
- step5_outputs/mv_highpass.npy
- step5_outputs/lcv_highpass.npy
- step5_outputs/twvv_highpass.npy

Outputs
-------
- step6_outputs/band_energy_stats.png
- step6_outputs/band_energy_windows.csv
- step6_outputs/band_energy_pvalues.csv
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy import signal
from scipy.stats import mannwhitneyu


# ----------------------------
# Config
# ----------------------------
FREQ_BANDS = {
    "0.005-0.0095Hz": (0.005, 0.0095),   # EDHF
    "0.0095-0.02Hz":  (0.0095, 0.02),    # Endo (NO)
    "0.02-0.06Hz":    (0.02, 0.06),      # Sympathetic
    "0.06-0.2Hz":     (0.06, 0.2),       # Myogenic
}

PAIRWISE = [("MV", "LCV"), ("MV", "TWVV"), ("LCV", "TWVV")]


@dataclass
class Paths:
    step5_dir: str
    out_dir: str


@dataclass
class Settings:
    fs: float = 2.0

    # CWT settings (match Step5)
    wavelet: str = "cmor1.5-1.0"
    freq_min: float = 0.005
    freq_max: float = 0.2
    num_freqs: int = 140

    # Windowing for within-subject variability
    window_sec: float = 60.0      # 60s per window (adjustable)
    step_sec: float = 60.0        # non-overlap by default; set smaller for overlap

    # Plot / stats
    alpha: float = 0.05           # significance threshold
    draw_significance: bool = True


# ----------------------------
# IO helpers
# ----------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_highpass_curves(step5_dir: str) -> Dict[str, np.ndarray]:
    files = {
        "MV": "mv_highpass.npy",
        "LCV": "lcv_highpass.npy",
        "TWVV": "twvv_highpass.npy",
    }
    out: Dict[str, np.ndarray] = {}
    for k, fn in files.items():
        path = os.path.join(step5_dir, fn)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing file: {path}")
        out[k] = np.load(path).astype(np.float32)
    return out


def align_curves_length(curves: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    lens = {k: len(v) for k, v in curves.items()}
    n = min(lens.values())
    if len(set(lens.values())) != 1:
        print("[Step6] Length mismatch:", lens, "-> truncate to", n)
    return {k: v[:n].copy() for k, v in curves.items()}


# ----------------------------
# Signal processing
# ----------------------------
def preprocess(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - np.mean(x)
    x = signal.detrend(x, type="linear").astype(np.float32)
    return x


def morlet_cwt_power(x: np.ndarray, fs: float, settings: Settings) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return frequencies (Hz) and power (F, T).
    """
    dt = 1.0 / fs
    freqs = np.logspace(
        np.log10(settings.freq_min),
        np.log10(settings.freq_max),
        settings.num_freqs,
    ).astype(np.float32)

    w = pywt.ContinuousWavelet(settings.wavelet)
    cf = pywt.central_frequency(w)
    scales = cf / (freqs * dt)

    coeffs, _ = pywt.cwt(x, scales, w, sampling_period=dt)
    power = (np.abs(coeffs) ** 2).astype(np.float32)
    return freqs, power


def band_power_timeseries(freqs: np.ndarray, power: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
    """
    Compute band power time series by integrating power over frequency within band.
    Output shape: (T,)
    """
    f1, f2 = band
    idx = (freqs >= f1) & (freqs <= f2)
    if not np.any(idx):
        return np.zeros(power.shape[1], dtype=np.float32)
    # integrate over frequency axis -> band energy per time
    # power[idx, :] shape (Fb, T)
    band_ts = np.trapz(power[idx, :], freqs[idx], axis=0).astype(np.float32)
    return band_ts


def window_samples(x: np.ndarray, fs: float, window_sec: float, step_sec: float) -> np.ndarray:
    """
    Slice a 1D time series into windows, return per-window mean.
    """
    n = len(x)
    w = int(round(window_sec * fs))
    s = int(round(step_sec * fs))
    if w <= 1 or s <= 0:
        raise ValueError("Invalid window_sec/step_sec.")
    samples: List[float] = []
    for start in range(0, n - w + 1, s):
        seg = x[start:start + w]
        samples.append(float(np.mean(seg)))
    if len(samples) < 3:
        # too few windows -> stats and error bars become meaningless
        print("[Step6] Warning: too few windows (<3). Consider smaller window_sec or longer signal.")
    return np.array(samples, dtype=np.float32)


# ----------------------------
# Stats + plotting
# ----------------------------
def significance_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def add_sig_bracket(ax, x1: float, x2: float, y: float, text: str) -> None:
    """
    Draw a simple bracket from x1 to x2 at height y with label text.
    """
    if not text:
        return
    h = 0.02 * y if y > 0 else 0.02
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], linewidth=1.2)
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom")


def save_csv_matrix(path: str, header: List[str], rows: List[List[str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(r) + "\n")


def main() -> None:
    paths = Paths(
        step5_dir=r"D:\mattsdata\生醫影像\infrared_hw\step5_outputs",
        out_dir=r"D:\mattsdata\生醫影像\infrared_hw\step6_outputs",
    )
    settings = Settings()
    ensure_dir(paths.out_dir)

    curves = align_curves_length(load_highpass_curves(paths.step5_dir))
    for k, v in curves.items():
        print(f"[Step6] {k} len={len(v)} std={np.std(v):.6f}")

    # Compute CWT power per tissue
    cwt: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for tissue, sig in curves.items():
        sig_p = preprocess(sig)
        freqs, power = morlet_cwt_power(sig_p, settings.fs, settings)
        cwt[tissue] = (freqs, power)

    # Build window samples per tissue per band
    # samples[tissue][band_name] = array of window means
    samples: Dict[str, Dict[str, np.ndarray]] = {t: {} for t in curves.keys()}

    for tissue in curves.keys():
        freqs_t, power_t = cwt[tissue]
        for band_name, band_rng in FREQ_BANDS.items():
            ts = band_power_timeseries(freqs_t, power_t, band_rng)
            smp = window_samples(ts, settings.fs, settings.window_sec, settings.step_sec)
            samples[tissue][band_name] = smp

    # Summaries for plotting: mean ± SEM
    band_names = list(FREQ_BANDS.keys())
    tissues = ["MV", "LCV", "TWVV"]

    means = np.zeros((len(tissues), len(band_names)), dtype=np.float32)
    sems = np.zeros((len(tissues), len(band_names)), dtype=np.float32)

    for i, tissue in enumerate(tissues):
        for j, bname in enumerate(band_names):
            x = samples[tissue][bname]
            means[i, j] = float(np.mean(x))
            # SEM
            sems[i, j] = float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) >= 2 else 0.0

    # Pairwise p-values per band
    pvals_rows: List[List[str]] = []
    pvals: Dict[Tuple[str, str, str], float] = {}  # (band, A, B) -> p

    for bname in band_names:
        for A, B in PAIRWISE:
            x = samples[A][bname]
            y = samples[B][bname]
            if len(x) < 2 or len(y) < 2:
                p = np.nan
            else:
                # two-sided Mann–Whitney U
                p = float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
            pvals[(bname, A, B)] = p
            pvals_rows.append([bname, A, B, "nan" if np.isnan(p) else f"{p:.6g}"])

    # Save window samples CSV (long format)
    win_rows: List[List[str]] = []
    for tissue in tissues:
        for bname in band_names:
            x = samples[tissue][bname]
            for idx, val in enumerate(x):
                win_rows.append([tissue, bname, str(idx), f"{float(val):.8g}"])

    save_csv_matrix(
        os.path.join(paths.out_dir, "band_energy_windows.csv"),
        header=["tissue", "band", "window_index", "band_energy"],
        rows=win_rows,
    )
    save_csv_matrix(
        os.path.join(paths.out_dir, "band_energy_pvalues.csv"),
        header=["band", "group_A", "group_B", "p_value"],
        rows=pvals_rows,
    )

    # Plot bar chart
    x = np.arange(len(band_names), dtype=np.float32)
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 5))

    bar_containers = []
    for i, tissue in enumerate(tissues):
        xi = x + (i - 1) * width
        bc = ax.bar(
            xi,
            means[i, :],
            width=width,
            yerr=sems[i, :],
            capsize=4,
            label=tissue,
        )
        bar_containers.append(bc)

    ax.set_title("Total energy in each frequency range (single-subject, window-based)")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Scale-averaged wavelet power (AU^2)")
    ax.set_xticks(x)
    ax.set_xticklabels(band_names, rotation=0)
    ax.legend()

    # Significance brackets (within-subject windows)
    if settings.draw_significance:
        for j, bname in enumerate(band_names):
            # determine baseline height above bars for this band
            y0 = float(np.max(means[:, j] + sems[:, j]))
            if y0 <= 0:
                y0 = 1e-6
            step = 0.10 * y0  # vertical step between brackets

            # x positions for each tissue bar in this band
            pos = {
                "MV": float(x[j] - width),
                "LCV": float(x[j]),
                "TWVV": float(x[j] + width),
            }

            # draw three pairwise brackets stacked
            k = 0
            for A, B in PAIRWISE:
                p = pvals[(bname, A, B)]
                txt = "" if np.isnan(p) else significance_stars(p)
                add_sig_bracket(ax, pos[A], pos[B], y0 + k * step, txt)
                k += 1

        # Add a small note in the corner
        ax.text(
            0.995,
            0.8,
            "Stars from Mann–Whitney U across time-windows (within-subject)\n(Not group-level across subjects)",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
        )

    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    out_png = os.path.join(paths.out_dir, "band_energy_stats.png")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print("[Step6] Saved:", out_png)
    print("[Step6] Saved:", os.path.join(paths.out_dir, "band_energy_windows.csv"))
    print("[Step6] Saved:", os.path.join(paths.out_dir, "band_energy_pvalues.csv"))
    print("Step 6 complete.")


if __name__ == "__main__":
    main()
