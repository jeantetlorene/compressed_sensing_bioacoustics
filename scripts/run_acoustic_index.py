from pathlib import Path
import numpy as np
import pandas as pd
import maad
from maad import sound, features

# =========================
# USER SETTINGS
# =========================

folder=r"c:\Users\loren\Documents\Postdoc\Compressed_sensing\Data\Thyolo"
AUDIO_FOLDER = Path(folder, "Audio")
OUTPUT_FOLDER = Path(folder, "Results_maad")

WINDOW_SECONDS = 10

# Spectrogram settings
N_PER_SEG = 1024   # nperseg = fs / 10, common in scikit-maad examples
NO_OVERLAP = 256

# Frequency limits
FMIN = 1000
FMAX = 10000

# ADI settings
ADI_BIN_STEP = 500
ADI_DB_THRESHOLD = -50

# Bioacoustic index frequency band
BI_FLIM = (1000, 10000)

# =========================
# FUNCTIONS
# =========================

def compute_indices_from_signal(s, fs):
    """
    Compute acoustic indices from one audio window.
    """

    # Amplitude spectrogram for ADI, ACI, BI, entropy
    Sxx_amp, tn, fn, ext = sound.spectrogram(
        s,
        fs,
        nperseg=N_PER_SEG,
        noverlap=NO_OVERLAP,
        mode="amplitude",
        detrend=False
    )

    # Power spectrogram for SNR
    Sxx_power, _, _, _ = sound.spectrogram(
        s,
        fs,
        nperseg=N_PER_SEG,
        noverlap=NO_OVERLAP,
        mode="psd",
        detrend=False
    )

    # Avoid empty or invalid windows
    if Sxx_amp.size == 0 or np.all(Sxx_amp == 0):
        return {
            "ADI": np.nan,
            "ACI": np.nan,
            "spectral_entropy": np.nan,
            "bioacoustic_index": np.nan,
            "SNR_dB": np.nan,
            "background_noise_dB": np.nan,
            "energy_dB": np.nan
        }

    # Acoustic Diversity Index
    ADI = features.acoustic_diversity_index(
        Sxx_amp,
        fn,
        fmin=FMIN,
        fmax=FMAX,
        bin_step=ADI_BIN_STEP,
        dB_threshold=ADI_DB_THRESHOLD,
        index="shannon"
    )

    # Acoustic Complexity Index
    ACI_xx, ACI_per_bin, ACI_sum = features.acoustic_complexity_index(Sxx_amp)

    # Spectral entropy
    # Returns several entropy descriptors depending on scikit-maad version.
    entropy_values = features.spectral_entropy(
        Sxx_amp,
        fn,
        flim=(FMIN, FMAX),
        display=False
    )

    # In most versions, the first returned value is the spectral entropy
    if isinstance(entropy_values, tuple):
        Hf = entropy_values[0]
    else:
        Hf = entropy_values

    # Bioacoustic Index
    BI = features.bioacoustics_index(
        Sxx_amp,
        fn,
        flim=BI_FLIM,
        R_compatible="soundecology"
    )

    # Spectral SNR
    ENRf, BGNf, SNRf, ENRf_per_bin, BGNf_per_bin, SNRf_per_bin = sound.spectral_snr(
        Sxx_power
    )

    return {
        "ADI": ADI,
        "ACI": ACI_sum,
        "spectral_entropy": Hf,
        "bioacoustic_index": BI,
        "SNR_dB": SNRf,
        "background_noise_dB": BGNf,
        "energy_dB": ENRf
    }


def process_file(filepath):
    """
    Split one wav file into windows and compute indices per window.
    """

    s, fs = sound.load(str(filepath), detrend=False)

    samples_per_window = int(WINDOW_SECONDS * fs)
    n_windows = len(s) // samples_per_window

    rows = []

    for i in range(n_windows):
        start = i * samples_per_window
        end = start + samples_per_window
        s_window = s[start:end]

        indices = compute_indices_from_signal(s_window, fs)

        rows.append({
            "file": filepath.name,
            "window_index": i,
            "start_sec": i * WINDOW_SECONDS,
            "end_sec": (i + 1) * WINDOW_SECONDS,
            "duration_sec": WINDOW_SECONDS,
            **indices
        })

    return rows


def main():
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    wav_files = sorted(AUDIO_FOLDER.glob("*.wav"))

    if len(wav_files) == 0:
        raise FileNotFoundError(f"No .wav files found in {AUDIO_FOLDER}")

    all_rows = []

    for wav_file in wav_files:
        print(f"Processing: {wav_file.name}")
        all_rows.extend(process_file(wav_file))

    window_df = pd.DataFrame(all_rows)

    window_df.to_csv(
        OUTPUT_FOLDER / "indices_per_window.csv",
        index=False
    )

    recording_summary = (
        window_df
        .groupby("file")[[
            "ADI",
            "ACI",
            "spectral_entropy",
            "bioacoustic_index",
            "SNR_dB",
            "background_noise_dB",
            "energy_dB"
        ]]
        .agg(["count", "mean", "std", "median", "min", "max"])
    )

    recording_summary.columns = [
        "_".join(col) for col in recording_summary.columns
    ]

    recording_summary = recording_summary.reset_index()

    recording_summary.to_csv(
        OUTPUT_FOLDER / "indices_per_recording.csv",
        index=False
    )

    overall_summary = window_df[[
        "ADI",
        "ACI",
        "spectral_entropy",
        "bioacoustic_index",
        "SNR_dB",
        "background_noise_dB",
        "energy_dB"
    ]].describe()

    overall_summary.to_csv(
        OUTPUT_FOLDER / "overall_summary.csv"
    )

    print("Done.")
    print(f"Processed {len(wav_files)} wav files.")
    print(f"Results saved in: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
