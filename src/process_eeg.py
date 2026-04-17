#!/usr/bin/env python3
"""Process OpenBCI Ganglion EEG data: parse, filter, analyze, and visualize."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
import mne

RAW_FILE = Path(__file__).parent.parent / "data" / "raw" / "ganglion_test_drive.txt"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PLOT_DIR = Path(__file__).parent.parent / "output" / "plots"

SFREQ = 250  # Hz
EXG_CHANNELS = ["EXG_CH0", "EXG_CH1", "EXG_CH2", "EXG_CH3"]
ACCEL_CHANNELS = ["Accel_X", "Accel_Y", "Accel_Z"]

EEG_BANDS = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 100),
}


def load_raw_data(filepath: Path) -> pd.DataFrame:
    header_lines = 0
    with open(filepath) as f:
        for line in f:
            if line.startswith("%"):
                header_lines += 1
            else:
                break

    col_line = None
    with open(filepath) as f:
        for i, line in enumerate(f):
            if line.startswith("Sample Index"):
                col_line = i
                break

    df = pd.read_csv(filepath, skiprows=col_line, skipinitialspace=True)
    df.columns = df.columns.str.strip()

    rename_map = {"Sample Index": "sample_index"}
    for i in range(4):
        rename_map[f"EXG Channel {i}"] = EXG_CHANNELS[i]
    for i, name in enumerate(ACCEL_CHANNELS):
        rename_map[f"Accel Channel {i}"] = name
    rename_map["Timestamp (Formatted)"] = "timestamp"
    rename_map["Marker Channel"] = "marker"

    df = df.rename(columns=rename_map)

    keep_cols = ["sample_index"] + EXG_CHANNELS + ACCEL_CHANNELS + ["timestamp", "marker"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    df["time_s"] = np.arange(len(df)) / SFREQ

    print(f"Loaded {len(df)} samples ({len(df)/SFREQ:.1f}s) at {SFREQ} Hz")
    return df


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()

    # Bandpass 0.5-100 Hz
    sos_bp = signal.butter(4, [0.5, 100], btype="bandpass", fs=SFREQ, output="sos")
    # Notch at 50 Hz and 60 Hz (cover both power line frequencies)
    b_50, a_50 = signal.iirnotch(50, 30, SFREQ)
    b_60, a_60 = signal.iirnotch(60, 30, SFREQ)

    for ch in EXG_CHANNELS:
        x = filtered[ch].values.astype(float)
        x = signal.sosfiltfilt(sos_bp, x)
        x = signal.filtfilt(b_50, a_50, x)
        x = signal.filtfilt(b_60, a_60, x)
        filtered[ch] = x

    print("Applied bandpass (0.5-100 Hz) and notch (50/60 Hz) filters")
    return filtered


def compute_psd(df: pd.DataFrame) -> dict:
    psd_results = {}
    for ch in EXG_CHANNELS:
        freqs, pxx = signal.welch(df[ch].values, fs=SFREQ, nperseg=min(256, len(df)))
        psd_results[ch] = (freqs, pxx)
    return psd_results


def compute_band_powers(psd_results: dict) -> pd.DataFrame:
    rows = []
    for ch, (freqs, pxx) in psd_results.items():
        freq_res = freqs[1] - freqs[0]
        total_power = np.trapezoid(pxx, dx=freq_res)
        row = {"channel": ch, "total_power": total_power}
        for band_name, (fmin, fmax) in EEG_BANDS.items():
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            band_power = np.trapezoid(pxx[idx], dx=freq_res)
            row[f"{band_name}_abs"] = band_power
            row[f"{band_name}_rel"] = band_power / total_power if total_power > 0 else 0
        rows.append(row)
    return pd.DataFrame(rows)


def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    stats = []
    for ch in EXG_CHANNELS:
        x = df[ch].values
        stats.append({
            "channel": ch,
            "mean": np.mean(x),
            "std": np.std(x),
            "min": np.min(x),
            "max": np.max(x),
            "peak_to_peak": np.ptp(x),
            "rms": np.sqrt(np.mean(x**2)),
            "kurtosis": float(pd.Series(x).kurtosis()),
            "skewness": float(pd.Series(x).skew()),
        })
    return pd.DataFrame(stats)


def plot_raw_signals(df: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Raw EXG Channels — OpenBCI Ganglion", fontsize=14)

    for i, ch in enumerate(EXG_CHANNELS):
        axes[i].plot(df["time_s"], df[ch], linewidth=0.5, color=f"C{i}")
        axes[i].set_ylabel(f"{ch}\n(µV)")
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(output_dir / "raw_signals.png", dpi=150)
    plt.close()


def plot_filtered_signals(df: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Filtered EXG Channels (0.5-100 Hz BP + 50/60 Hz Notch)", fontsize=14)

    for i, ch in enumerate(EXG_CHANNELS):
        axes[i].plot(df["time_s"], df[ch], linewidth=0.5, color=f"C{i}")
        axes[i].set_ylabel(f"{ch}\n(µV)")
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(output_dir / "filtered_signals.png", dpi=150)
    plt.close()


def plot_psd(psd_results: dict, output_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Power Spectral Density (Welch)", fontsize=14)

    for i, (ch, (freqs, pxx)) in enumerate(psd_results.items()):
        ax = axes[i // 2][i % 2]
        ax.semilogy(freqs, pxx, linewidth=0.8, color=f"C{i}")
        for band_name, (fmin, fmax) in EEG_BANDS.items():
            ax.axvspan(fmin, fmax, alpha=0.1, label=band_name)
        ax.set_title(ch)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (µV²/Hz)")
        ax.set_xlim(0, 60)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_dir / "psd.png", dpi=150)
    plt.close()


def plot_band_powers(band_df: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    bands = list(EEG_BANDS.keys())
    x = np.arange(len(bands))
    width = 0.2

    for i, (_, row) in enumerate(band_df.iterrows()):
        abs_vals = [row[f"{b}_abs"] for b in bands]
        rel_vals = [row[f"{b}_rel"] for b in bands]
        axes[0].bar(x + i * width, abs_vals, width, label=row["channel"])
        axes[1].bar(x + i * width, rel_vals, width, label=row["channel"])

    axes[0].set_title("Absolute Band Power")
    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(bands)
    axes[0].set_ylabel("Power (µV²)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].set_title("Relative Band Power")
    axes[1].set_xticks(x + width * 1.5)
    axes[1].set_xticklabels(bands)
    axes[1].set_ylabel("Relative Power")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "band_powers.png", dpi=150)
    plt.close()


def plot_accelerometer(df: pd.DataFrame, output_dir: Path):
    if not all(c in df.columns for c in ACCEL_CHANNELS):
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Accelerometer Data", fontsize=14)

    labels = ["X", "Y", "Z"]
    for i, ch in enumerate(ACCEL_CHANNELS):
        axes[i].plot(df["time_s"], df[ch], linewidth=0.5, color=f"C{i+4}")
        axes[i].set_ylabel(f"Accel {labels[i]}\n(g)")
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(output_dir / "accelerometer.png", dpi=150)
    plt.close()


def plot_spectrogram(df: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Spectrogram (Time-Frequency)", fontsize=14)

    for i, ch in enumerate(EXG_CHANNELS):
        f, t, Sxx = signal.spectrogram(df[ch].values, fs=SFREQ, nperseg=128, noverlap=96)
        axes[i].pcolormesh(t, f, 10 * np.log10(Sxx + 1e-20), shading="gouraud", cmap="viridis")
        axes[i].set_ylabel(f"{ch}\nFreq (Hz)")
        axes[i].set_ylim(0, 60)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(output_dir / "spectrogram.png", dpi=150)
    plt.close()


def create_mne_raw(df: pd.DataFrame) -> mne.io.RawArray:
    data = df[EXG_CHANNELS].values.T * 1e-6  # µV -> V for MNE
    info = mne.create_info(ch_names=EXG_CHANNELS, sfreq=SFREQ, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("OpenBCI Ganglion EEG Data Processing Pipeline")
    print("=" * 60)

    # 1. Load raw data
    print("\n[1/7] Loading raw data...")
    df_raw = load_raw_data(RAW_FILE)

    # 2. Compute raw statistics
    print("\n[2/7] Computing raw signal statistics...")
    raw_stats = compute_statistics(df_raw)
    print(raw_stats.to_string(index=False))

    # 3. Plot raw signals
    print("\n[3/7] Plotting raw signals...")
    plot_raw_signals(df_raw, PLOT_DIR)
    plot_accelerometer(df_raw, PLOT_DIR)

    # 4. Apply filters
    print("\n[4/7] Filtering signals...")
    df_filtered = apply_filters(df_raw)

    # 5. Plot filtered signals + spectrogram
    print("\n[5/7] Plotting filtered signals and spectrogram...")
    plot_filtered_signals(df_filtered, PLOT_DIR)
    plot_spectrogram(df_filtered, PLOT_DIR)

    # 6. Spectral analysis
    print("\n[6/7] Computing spectral analysis...")
    psd_results = compute_psd(df_filtered)
    plot_psd(psd_results, PLOT_DIR)

    band_powers = compute_band_powers(psd_results)
    plot_band_powers(band_powers, PLOT_DIR)

    print("\nBand Powers (relative):")
    rel_cols = ["channel"] + [f"{b}_rel" for b in EEG_BANDS]
    print(band_powers[rel_cols].to_string(index=False))

    # 7. Save processed data
    print("\n[7/7] Saving processed data...")
    df_filtered.to_csv(PROCESSED_DIR / "filtered_eeg.csv", index=False)
    band_powers.to_csv(PROCESSED_DIR / "band_powers.csv", index=False)
    raw_stats.to_csv(PROCESSED_DIR / "signal_statistics.csv", index=False)

    # Also save as MNE-compatible FIF
    raw_mne = create_mne_raw(df_filtered)
    raw_mne.save(PROCESSED_DIR / "ganglion_eeg_raw.fif", overwrite=True, verbose=False)

    print(f"\nProcessed data saved to: {PROCESSED_DIR}")
    print(f"Plots saved to: {PLOT_DIR}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Recording duration: {len(df_raw)/SFREQ:.1f} seconds")
    print(f"  Sample rate: {SFREQ} Hz")
    print(f"  Total samples: {len(df_raw)}")
    print(f"  Channels: {len(EXG_CHANNELS)} EXG + {len(ACCEL_CHANNELS)} Accel")
    print(f"  Time range: {df_raw['timestamp'].iloc[0]} → {df_raw['timestamp'].iloc[-1]}")
    dominant = band_powers.iloc[0]
    max_band = max(EEG_BANDS.keys(), key=lambda b: dominant[f"{b}_rel"])
    print(f"  Dominant band (CH0): {max_band} ({dominant[f'{max_band}_rel']*100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
