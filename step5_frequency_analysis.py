"""
Step 5: Frequency analysis (high-pass + Morlet wavelet + time-averaged power)

Inputs
------
- thermal_aligned.npy (rows, cols, frames) OR (frames, rows, cols)
- mv_mask.npy / lcv_mask.npy / twvv_mask.npy
- (optional) refined TSR curves saved as npy for MV/LCV/TWVV

Outputs
-------
- High-pass filtered curves
- Scalograms (Morlet wavelet, with COI)
- Time-averaged wavelet power spectrum
- (optional) band energy summary
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy import signal


# ----------------------------
# Config
# ----------------------------
@dataclass
class Paths:
    data_path: str
    mask_dir: str
    refined_dir: str
    out_dir: str


@dataclass
class Settings:
    fs: float = 2.0

    # High-pass filter settings
    highpass_cutoff: float = 0.004  # Hz
    highpass_order: int = 4
    highpass_ripple_db: float = 0.5
    highpass_stop_db: float = 40

    # CWT settings
    wavelet: str = "cmor1.5-1.0"
    freq_min: float = 0.005
    freq_max: float = 1.8
    num_freqs: int = 140

    # Plot settings
    cmap: str = "turbo"


FREQ_BANDS = {
    "heart_beat": (0.6, 1.8),
    "respiration": (0.2, 0.6),
    "myogenic": (0.06, 0.2),
    "sympathetic": (0.02, 0.06),
    "endo_no": (0.0095, 0.02),
    "edhf": (0.005, 0.0095),
}


# ----------------------------
# Utilities
# ----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_thermal(data_path: str) -> np.ndarray:
    """
    Load thermal_aligned.npy and standardize to (rows, cols, frames).

    Robust rule:
    - Treat the largest dimension as frames (typical videos have frames >> rows/cols).
    - Then transpose accordingly.
    """
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"找不到 thermal_aligned.npy: {data_path}")

    T = np.load(data_path).astype(np.float32)
    if T.ndim != 3:
        raise ValueError(f"資料維度錯誤: {T.shape}")

    # Decide which axis is frames: choose axis with maximum length
    ax_frames = int(np.argmax(T.shape))

    if ax_frames == 2:
        # already (rows, cols, frames)
        return T
    elif ax_frames == 0:
        # (frames, rows, cols) -> (rows, cols, frames)
        return np.transpose(T, (1, 2, 0))
    else:
        # (rows, frames, cols) -> (rows, cols, frames)
        return np.transpose(T, (0, 2, 1))


def load_masks(mask_dir: str) -> Dict[str, np.ndarray]:
    mv = np.load(os.path.join(mask_dir, "mv_mask.npy")).astype(bool)
    lcv = np.load(os.path.join(mask_dir, "lcv_mask.npy")).astype(bool)
    twvv = np.load(os.path.join(mask_dir, "twvv_mask.npy")).astype(bool)
    return {"MV": mv, "LCV": lcv, "TWVV": twvv}


def compute_tsr(T: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rows, cols, frames = T.shape
    if mask.shape != (rows, cols):
        raise ValueError(f"Mask shape mismatch: {mask.shape} vs {(rows, cols)}")
    if not np.any(mask):
        raise ValueError("Mask is empty (no True pixels). Check mask generation.")
    # vectorized mean over mask per frame
    tsr = np.empty(frames, dtype=np.float32)
    for i in range(frames):
        tsr[i] = float(T[:, :, i][mask].mean())
    return tsr


def baseline_normalize(tsr: np.ndarray, fs: float, bl_minutes: float = 10.0) -> np.ndarray:
    bl_frames = int(bl_minutes * 60 * fs)
    bl_frames = min(bl_frames, len(tsr))
    if bl_frames <= 0:
        raise ValueError("baseline frames computed as 0; check fs or signal length.")
    baseline = float(tsr[:bl_frames].mean())
    return (tsr - baseline).astype(np.float32)


def load_or_compute_curves(paths: Paths, settings: Settings) -> Dict[str, np.ndarray]:
    """
    Priority:
    1) Load refined TSR from step4_outputs if exists (supports two naming styles)
    2) Otherwise compute TSR from thermal + masks then baseline-normalize
    """
    curves: Dict[str, np.ndarray] = {}

    candidate_sets = [
        {"MV": "mv_tsr.npy", "LCV": "lcv_tsr.npy", "TWVV": "twvv_tsr.npy"},
        {"MV": "tsr_mv_refined.npy", "LCV": "tsr_lcv_refined.npy", "TWVV": "tsr_twvv_refined.npy"},
    ]

    for refined_files in candidate_sets:
        ok = all(os.path.isfile(os.path.join(paths.refined_dir, f)) for f in refined_files.values())
        if ok:
            for key, fname in refined_files.items():
                curves[key] = np.load(os.path.join(paths.refined_dir, fname)).astype(np.float32)
            print("[Step5] Loaded refined curves from:", paths.refined_dir)
            print("[Step5] Files:", refined_files)
            return curves

    print("[Step5] Refined curves not found. Fallback to TSR from thermal + masks (ΔT baseline).")
    T = load_thermal(paths.data_path)
    masks = load_masks(paths.mask_dir)
    for key, mask in masks.items():
        tsr = compute_tsr(T, mask)
        curves[key] = baseline_normalize(tsr, settings.fs)

    return curves


def align_curves_length(curves: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Truncate all curves to the minimum length (prevents mismatch in plots/CWT)."""
    lengths = {k: len(v) for k, v in curves.items()}
    n = min(lengths.values())
    if len(set(lengths.values())) != 1:
        print("[Step5] Curve length mismatch detected:", lengths)
        print(f"[Step5] Truncating all curves to n={n}")
    out = {k: v[:n].copy() for k, v in curves.items()}
    return out


def preprocess_for_frequency(x: np.ndarray) -> np.ndarray:
    """Remove mean + linear trend to reduce drift dominating low-frequency power."""
    x = np.asarray(x, dtype=np.float32)
    x = x - np.mean(x)
    x = signal.detrend(x, type="linear")
    return x.astype(np.float32)


def highpass_filter(
    x: np.ndarray,
    fs: float,
    cutoff: float,
    order: int,
    ripple_db: float,
    stop_db: float,
) -> np.ndarray:
    """Elliptic high-pass filter with sosfiltfilt."""
    sos = signal.ellip(
        order,
        ripple_db,
        stop_db,
        cutoff,
        btype="highpass",
        fs=fs,
        output="sos",
    )
    y = signal.sosfiltfilt(sos, x)
    return y.astype(np.float32)


def morlet_cwt(x: np.ndarray, fs: float, settings: Settings) -> Tuple[np.ndarray, np.ndarray]:
    dt = 1.0 / fs
    frequencies = np.logspace(
        np.log10(settings.freq_min),
        np.log10(settings.freq_max),
        settings.num_freqs,
    ).astype(np.float32)

    wavelet = pywt.ContinuousWavelet(settings.wavelet)
    cf = pywt.central_frequency(wavelet)  # central frequency (dimensionless)
    scales = cf / (frequencies * dt)      # scale = cf / (f * dt)

    coeffs, _ = pywt.cwt(x, scales, wavelet, sampling_period=dt)
    power = (np.abs(coeffs) ** 2).astype(np.float32)  # (num_freqs, time)
    return frequencies, power


def compute_coi(num_samples: int, fs: float, settings: Settings) -> np.ndarray:
    """
    Simple COI proxy for visualization:
    Use distance-to-edge to approximate maximum reliable scale near edges, then map to frequency.
    """
    dt = 1.0 / fs
    t = np.arange(num_samples, dtype=np.float32) * dt
    t_end = float(t[-1]) if num_samples > 0 else 0.0
    dist = np.minimum(t, t_end - t)
    dist = np.maximum(dist, 1e-6)

    wavelet = pywt.ContinuousWavelet(settings.wavelet)
    cf = pywt.central_frequency(wavelet)

    # A heuristic mapping: larger distance allows larger scale (lower freq) safely
    scale_edge = dist / (np.sqrt(2) * dt)
    scale_edge = np.maximum(scale_edge, 1e-6)
    coi_freq = cf / (scale_edge * dt)
    return coi_freq.astype(np.float32)


# ----------------------------
# Plotting
# ----------------------------
def plot_scalogram(
    time_min: np.ndarray,
    frequencies: np.ndarray,
    power: np.ndarray,
    coi_freq: np.ndarray,
    title: str,
    out_path: str,
    settings: Settings,
) -> None:
    """
    Scalogram (Hz on y-axis) with log y-scale so frequency fills the full plot.
    """
    plt.figure(figsize=(10, 5))

    power_log = np.log10(np.maximum(power, 1e-12)).astype(np.float32)
    vmin, vmax = np.nanpercentile(power_log, [5, 99.5])
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)):
        vmin, vmax = -12.0, -6.0
    if vmax - vmin < 0.5:
        mid = 0.5 * (vmin + vmax)
        vmin, vmax = mid - 1.5, mid + 1.5

    # 用 pcolormesh 直接在 Hz 座標上畫，y 軸再設 log
    im = plt.pcolormesh(
        time_min,
        frequencies,
        power_log,
        shading="auto",
        cmap=settings.cmap,
        vmin=vmin,
        vmax=vmax,
    )

    plt.yscale("log")
    plt.ylim(settings.freq_min, settings.freq_max)

    plt.xlabel("Time (min)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    plt.colorbar(im, label="log10(Power)")

    # COI：同樣用 Hz 畫上去（不再用 log10）
    coi_clip = np.clip(coi_freq, settings.freq_min, settings.freq_max)
    plt.plot(time_min, coi_clip, "w--", linewidth=1.5, label="COI")
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()



def plot_time_averaged(freqs: np.ndarray, spectra: Dict[str, np.ndarray], out_path: str) -> None:
    """Plot time-averaged spectrum on log-frequency axis."""
    plt.figure(figsize=(8, 5))
    for label, power in spectra.items():
        y = np.log10(np.maximum(power, 1e-12))
        plt.plot(freqs, y, label=label, linewidth=2)
    plt.xscale("log")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("log10(Time-averaged power)")
    plt.title("Time-averaged wavelet power spectrum (log scale)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def compute_band_energy(freqs: np.ndarray, power: np.ndarray) -> Dict[str, float]:
    energies: Dict[str, float] = {}
    for name, (f_low, f_high) in FREQ_BANDS.items():
        idx = (freqs >= f_low) & (freqs <= f_high)
        if not np.any(idx):
            energies[name] = 0.0
            continue
        energies[name] = float(np.trapz(power[idx], freqs[idx]))
    return energies


def plot_band_energy(energies: Dict[str, Dict[str, float]], out_path: str) -> None:
    labels = list(FREQ_BANDS.keys())
    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(10, 5))
    for i, (group, band_map) in enumerate(energies.items()):
        values = [band_map[lbl] for lbl in labels]
        plt.bar(x + i * width, values, width, label=group)

    plt.xticks(x + width, labels, rotation=30, ha="right")
    plt.ylabel("Total energy")
    plt.title("Total energy in each frequency band")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    paths = Paths(
        data_path=r"D:\mattsdata\生醫影像\infrared_hw\thermal_aligned.npy",
        mask_dir=r"D:\mattsdata\生醫影像\infrared_hw\step3_outputs",
        refined_dir=r"D:\mattsdata\生醫影像\infrared_hw\step4_outputs",
        out_dir=r"D:\mattsdata\生醫影像\infrared_hw\step5_outputs",
    )
    settings = Settings()
    ensure_dir(paths.out_dir)

    curves = load_or_compute_curves(paths, settings)
    curves = align_curves_length(curves)

    # sanity checks
    for key, curve in curves.items():
        if curve.size == 0 or np.any(np.isnan(curve)) or np.any(~np.isfinite(curve)):
            raise ValueError(f"{key} curve is empty or contains invalid values.")
        print(f"[Step5] {key} curve: len={len(curve)}, mean={np.mean(curve):.4f}, std={np.std(curve):.4f}")

    # time axis (aligned length)
    n = len(next(iter(curves.values())))
    time_min = np.arange(n, dtype=np.float32) / settings.fs / 60.0

    # filtering + save filtered
    filtered: Dict[str, np.ndarray] = {}
    for key, tsr in curves.items():
        x = preprocess_for_frequency(tsr)
        hp = highpass_filter(
            x,
            settings.fs,
            settings.highpass_cutoff,
            settings.highpass_order,
            settings.highpass_ripple_db,
            settings.highpass_stop_db,
        )
        filtered[key] = hp
        np.save(os.path.join(paths.out_dir, f"{key.lower()}_highpass.npy"), hp)
        print(f"[Step5] {key} highpass: std={np.std(hp):.6f}, min/max=({hp.min():.6f},{hp.max():.6f})")

    spectra: Dict[str, np.ndarray] = {}
    band_energies: Dict[str, Dict[str, float]] = {}

    for key, signal_hp in filtered.items():
        freqs, power = morlet_cwt(signal_hp, settings.fs, settings)
        coi = compute_coi(len(signal_hp), settings.fs, settings)

        # debug
        pmin, pmax = float(power.min()), float(power.max())
        print(f"[Step5] {key} CWT power: min/max=({pmin:.3e},{pmax:.3e})")
        if power.shape != (len(freqs), len(time_min)):
            raise ValueError(f"[Step5] power shape mismatch: {power.shape} vs {(len(freqs), len(time_min))}")

        plot_scalogram(
            time_min,
            freqs,
            power,
            coi,
            f"Morlet scalogram ({key})",
            os.path.join(paths.out_dir, f"scalogram_{key.lower()}.png"),
            settings,
        )

        # time-averaged spectrum (average over time axis)
        spectra[key] = power.mean(axis=1).astype(np.float32)
        band_energies[key] = compute_band_energy(freqs, spectra[key])

    plot_time_averaged(freqs, spectra, os.path.join(paths.out_dir, "time_averaged_power.png"))
    plot_band_energy(band_energies, os.path.join(paths.out_dir, "band_energy.png"))

    # save band energies as text
    txt_path = os.path.join(paths.out_dir, "band_energy.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for grp, emap in band_energies.items():
            f.write(f"{grp}\n")
            for band, val in emap.items():
                f.write(f"  {band}: {val:.6e}\n")
            f.write("\n")

    print("Step 5 complete. Outputs saved to:", paths.out_dir)
    print("Key outputs:",
          "scalogram_mv.png / scalogram_lcv.png / scalogram_twvv.png,",
          "time_averaged_power.png, band_energy.png, band_energy.txt")


if __name__ == "__main__":
    main()
