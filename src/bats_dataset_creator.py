from pathlib import Path
from collections import defaultdict
import json
import pickle
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm


class BatsDatasetCreator:
    """
    Creates training and testing datasets for the bats 4-class classification task.

    Labels are extracted from the filename format:
        000000_SPECIES__RECORDER_TIMESTAMP.wav  →  split('_')[1]  →  'SPECIES'

    Pipeline
    --------
    Training:
        1. Load & segment audio into non-overlapping 1-sec windows (sorted file order)
        2. Apply per-class subsampling (selected_indices_reduction_2000.npz) for
           over-represented classes (LAECAP=2, TADAEG=3)
        3. Apply window mask (window_decisions_training_mask.npy) to remove empty windows
        4. Convert to mel-spectrograms

    Testing:
        1. Load & segment audio (same as above, no class balancing)
        2. Apply window mask (window_decisions_testing_mask.npy)
        3. Convert to mel-spectrograms
    """

    def __init__(
        self,
        audio_path,
        train_txt,
        test_txt,
        label_map_path,
        filters_path,
        downsample_rate=128000,
        window_size_sec=1,
        n_fft=512,
        hop_length=384,
        n_mels=128,
        fmin=15000,
        fmax=64000,
    ):
        """
        Parameters
        ----------
        audio_path      : Path to the Audio/ folder containing .wav files
        train_txt       : Path to DataFiles/train.txt
        test_txt        : Path to DataFiles/test.txt
        label_map_path  : Path to labelName_to_labelInd JSON
        filters_path    : Path to the bats_filters/ directory
        downsample_rate : Target sample rate after resampling (default: 128000 Hz)
        window_size_sec : Duration of each window in seconds (default: 1)
        n_fft           : FFT window size for mel-spectrogram
        hop_length      : Hop length for mel-spectrogram
        n_mels          : Number of mel frequency bins
        fmin            : Minimum frequency for mel-spectrogram (Hz)
        fmax            : Maximum frequency for mel-spectrogram (Hz)
        """
        self.audio_path = Path(audio_path)
        self.train_txt = Path(train_txt)
        self.test_txt = Path(test_txt)
        self.filters_path = Path(filters_path)
        self.downsample_rate = downsample_rate
        self.window_size = window_size_sec * downsample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

        with open(label_map_path) as f:
            self.label_map = json.loads(json.load(f))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_label(self, filename):
        """Return species string from filename: '000000_CISTUGO__...' → 'CISTUGO'."""
        return filename.split("_")[1]

    def _load_audio(self, filename):
        """Load a .wav file at its native sample rate using soundfile (faster than librosa)."""
        path = self.audio_path / f"{filename}.wav"
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        return audio, sr

    def _downsample(self, audio, orig_sr):
        """Downsample audio to self.downsample_rate."""
        return librosa.resample(
            audio,
            orig_sr=orig_sr,
            target_sr=self.downsample_rate,
        ).astype(np.float32)

    def _trim_audio(self, audio, sr):
        """Remove the tail of a recording before segmentation (training only).

        Removes the last 3 seconds for recordings longer than 10 s, or the last
        2 seconds otherwise, to discard low-quality end samples.
        """
        if len(audio) // sr > 10:
            return audio[:-(sr * 3)]
        return audio[:-(sr * 2)]

    def _segment(self, audio):
        """Split audio into non-overlapping windows of self.window_size samples."""
        ws = self.window_size
        n = len(audio)
        num_frames = max(1, (n - ws) // ws + 1)
        return [audio[i * ws: i * ws + ws] for i in range(num_frames) if i * ws + ws <= n]

    def _process_file(self, filename, trim=False):
        """Load, downsample, and segment one file.

        Returns (label_str, windows) or None if skipped.
        trim : if True, remove the last 2-3 seconds before resampling (training only).
        """
        label_str = self._extract_label(filename)
        if label_str not in self.label_map:
            print(f"  Unknown label '{label_str}' in {filename} — skipped")
            return None
        try:
            audio, sr = self._load_audio(filename)
        except Exception as exc:
            print(f"  Could not load {filename}: {exc} — skipped")
            return None
        if trim:
            audio = self._trim_audio(audio, sr)
        audio_ds = self._downsample(audio, sr)
        return label_str, self._segment(audio_ds)

    def _load_and_segment_files(self, txt_path, trim=False):
        """
        Read file list, load audio sequentially, downsample, and segment.

        trim : remove last 2-3 s from each file before resampling (training only).

        Returns
        -------
        class_windows    : dict label_str → list of windows (np.ndarray)
        class_global_idx : dict label_str → list of global insertion indices
        """
        filenames = Path(txt_path).read_text().strip().split("\n")
        filenames = [f.strip() for f in filenames if f.strip()]

        class_windows = defaultdict(list)
        class_global_idx = defaultdict(list)
        global_counter = 0

        for filename in tqdm(filenames, desc=f"Loading {Path(txt_path).name}"):
            result = self._process_file(filename, trim=trim)
            if result is None:
                continue
            label_str, windows = result
            for w in windows:
                class_windows[label_str].append(w)
                class_global_idx[label_str].append(global_counter)
                global_counter += 1

        return class_windows, class_global_idx

    def _apply_class_balancing(self, class_windows, class_global_idx):
        """
        Subsample over-represented classes using pre-computed global indices.

        The npz keys are class integer indices (2=LAECAP, 3=TADAEG). Each array
        contains GLOBAL positions (into the full training window array sorted by
        processing order) to keep for that class. All other classes are kept in full.

        Output is class-grouped (all class-0 windows, then class-1, …) and within
        each over-represented class windows appear in the same order as the saved
        indices, reproducing the ordering of X_balanced in the reference notebook
        so that the pre-computed window mask aligns correctly.
        """
        # Build the full ordered array across all classes
        all_global, all_windows, all_labels = [], [], []
        for label_str, windows in class_windows.items():
            class_idx = self.label_map[label_str]
            for w, g in zip(windows, class_global_idx[label_str]):
                all_global.append(g)
                all_windows.append(w)
                all_labels.append(class_idx)

        order = np.argsort(all_global)
        X_all = np.array([all_windows[i] for i in order])
        Y_all = np.array([all_labels[i] for i in order])

        # Load selected global indices per over-represented class
        npz = np.load(self.filters_path / "selected_indices_reduction_2000.npz")
        selected_indices = {int(k): npz[k].astype(int) for k in npz.files}

        # Build output in class-grouped order, matching the notebook's X_balanced:
        #   - over-represented classes: windows in the saved (random.sample) index order
        #   - other classes: all windows in global processing order
        X_out, Y_out = [], []
        for class_idx_int in sorted(np.unique(Y_all)):
            if class_idx_int in selected_indices:
                keep_positions = selected_indices[class_idx_int]
                valid = keep_positions[keep_positions < len(X_all)]
                X_out.extend(X_all[valid])
                Y_out.extend(Y_all[valid])
            else:
                class_mask = Y_all == class_idx_int
                X_out.extend(X_all[class_mask])
                Y_out.extend(Y_all[class_mask])

        return np.array(X_out), np.array(Y_out)

    def _apply_mask(self, X, Y, mask_path):
        """Apply a pre-computed boolean mask to remove empty/noise windows."""
        mask = np.load(mask_path)
        if mask.shape[0] != X.shape[0]:
            raise ValueError(
                f"Mask length {mask.shape[0]} does not match X length {X.shape[0]}. "
                "Ensure files are processed in the same order as when the mask was created."
            )
        return X[mask], Y[mask]

    def _convert_single_to_image(self, audio):
        """Convert a 1-sec audio window to a normalized mel-spectrogram.

        Replicates Preprocessing.convert_single_to_image() from preprocess.py.
        """
        S = librosa.feature.melspectrogram(
            y=audio,
            sr=self.downsample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        image = librosa.power_to_db(S)
        mean, std = image.flatten().mean(), image.flatten().std()
        spec_norm = (image - mean) / (std + 1e-8)
        spec_scaled = (spec_norm - spec_norm.min()) / (spec_norm.max() - spec_norm.min())
        return spec_scaled

    def _convert_to_spectrograms(self, windows):
        """Batch-convert audio windows to mel-spectrograms."""
        return np.array(
            [self._convert_single_to_image(w) for w in tqdm(windows, desc="Spectrograms")]
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_training_dataset(self):
        """
        Build the training dataset.

        Steps: load & segment → class balancing → window mask → spectrograms.

        Returns
        -------
        X : np.ndarray, shape (N, n_mels, time_steps)
        Y : np.ndarray, shape (N,), integer class labels
        """
        class_windows, class_global_idx = self._load_and_segment_files(self.train_txt, trim=True)
        label_names = {v: k for k, v in self.label_map.items()}
        before = {label_names[self.label_map[s]]: len(w) for s, w in class_windows.items()}
        print(f"Before balancing: {sum(before.values())} windows — {before}")
        X, Y = self._apply_class_balancing(class_windows, class_global_idx)
        after_bal = dict(zip(*np.unique(Y, return_counts=True)))
        print(f"After  balancing: {X.shape[0]} windows — {after_bal}")
        X, Y = self._apply_mask(X, Y, self.filters_path / "window_decisions_training_mask.npy")
        after_mask = dict(zip(*np.unique(Y, return_counts=True)))
        print(f"After  mask:      {X.shape[0]} windows — {after_mask}")
        X = self._convert_to_spectrograms(X)
        return X, Y

    def create_testing_dataset(self):
        """
        Build the testing dataset.

        Steps: load & segment → window mask → spectrograms (no class balancing).

        Returns
        -------
        X : np.ndarray, shape (N, n_mels, time_steps)
        Y : np.ndarray, shape (N,), integer class labels
        """
        class_windows, class_global_idx = self._load_and_segment_files(self.test_txt)

        all_global, all_windows, all_labels = [], [], []
        for label_str, windows in class_windows.items():
            class_idx = self.label_map[label_str]
            for w, g in zip(windows, class_global_idx[label_str]):
                all_global.append(g)
                all_windows.append(w)
                all_labels.append(class_idx)

        order = np.argsort(all_global)
        X = np.array([all_windows[i] for i in order])
        Y = np.array([all_labels[i] for i in order])

        print(f"Before mask: {X.shape[0]} windows — {dict(zip(*np.unique(Y, return_counts=True)))}")
        X, Y = self._apply_mask(X, Y, self.filters_path / "window_decisions_testing_mask.npy")
        print(f"After mask: {X.shape[0]} windows — {dict(zip(*np.unique(Y, return_counts=True)))}")
        X = self._convert_to_spectrograms(X)
        return X, Y

    def save_dataset(self, X, Y, output_path, dataset_type):
        """Save X and Y as pickle files.

        Files are written to:
            output_path/X_bats_{dataset_type}.pkl
            output_path/Y_bats_{dataset_type}.pkl
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        for name, data in [("X", X), ("Y", Y)]:
            path = output_path / f"{name}_bats_{dataset_type}.pkl"
            with open(path, "wb") as f:
                pickle.dump(data, f, protocol=4)
            print(f"Saved → {path}")
