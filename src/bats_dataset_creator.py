from pathlib import Path
from glob import glob
import json
import pickle
import numpy as np
import signal
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
        sample_rate=256000,
        downsample_rate=192000,
        window_size_sec=1,
        spec_window_sec=None,
        n_fft=512,
        hop_length=384,
        n_mels=128,
        f_min=21000,
        f_max=68000,
        method_compression=None,
        parameter_compression=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        audio_path      : Path to the Audio/ folder containing .wav files
        train_txt       : Path to DataFiles/train.txt
        test_txt        : Path to DataFiles/test.txt
        label_map_path  : Path to labelName_to_labelInd JSON
        filters_path    : Path to the bats_filters/ directory
        sample_rate     : Original sample rate of the recordings
        downsample_rate : Target sample rate after resampling (default: 128000 Hz)
        window_size_sec : Duration of each window in seconds (default: 1). Must stay
                          at 1 s so the pre-computed balancing indices and window
                          masks stay aligned.
        spec_window_sec : Duration of the audio converted to each spectrogram
                          (default: None = window_size_sec). If smaller, each masked
                          window is split into non-overlapping sub-windows of this
                          length (e.g. 0.5 → two spectrograms per 1-s window, both
                          with the parent window's label).
        n_fft           : FFT window size for mel-spectrogram
        hop_length      : Hop length for mel-spectrogram
        n_mels          : Number of mel frequency bins
        f_min           : Minimum frequency for mel-spectrogram (Hz)
        f_max           : Maximum frequency for mel-spectrogram (Hz)
        method_compression   : None (wav), 'cs', or codec name e.g. 'flac'
        parameter_compression : Compression parameter (quality level, CS ratio, etc.)
        **kwargs        : Absorbs unused keys from config.preprocessing.dict()
        """
        self.audio_path = Path(audio_path)
        self.method_compression = method_compression
        self.parameter_compression = parameter_compression
        self.species_folder = Path(audio_path).parent
        self.train_txt = Path(train_txt)
        self.test_txt = Path(test_txt)
        self.filters_path = Path(filters_path)
        self.original_sample_rate = sample_rate
        self.downsample_rate = downsample_rate
        self.window_size = int(round(window_size_sec * downsample_rate))
        if spec_window_sec is None:
            spec_window_sec = window_size_sec
        self.spec_window_size = int(round(spec_window_sec * downsample_rate))
        if self.window_size % self.spec_window_size != 0:
            raise ValueError(
                f"spec_window_sec ({spec_window_sec}) must evenly divide "
                f"window_size_sec ({window_size_sec})."
            )
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = f_min
        self.fmax = f_max

        with open(label_map_path) as f:
            self.label_map = json.loads(json.load(f))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_label(self, filename):
        """Return species string from filename: '000000_CISTUGO__...' → 'CISTUGO'."""
        return filename.split("_")[1]

    def _load_audio(self, filename):
        """Load audio for one file, dispatching on self.method_compression.

        Returns (audio, sr). For CS-reconstructed .npy files sr is self.downsample_rate
        (assumed already at target rate); for wav/codec files sr is the native sample rate.
        """
        method = self.method_compression
        param  = self.parameter_compression

        if method is None:
            path = self.audio_path / f"{filename}.wav"
            audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
            return audio, sr

        elif method == "cs":
            cs_dir = self.species_folder / "Compressed_Audio" / f"cs_reconstructed_{param}"
            matches = glob(str(cs_dir / f"{filename}_*.npy"))
            if not matches:
                raise FileNotFoundError(
                    f"No CS-reconstructed file found for '{filename}' in {cs_dir}"
                )
            if len(matches) > 1:
                raise ValueError(
                    f"Multiple CS-reconstructed files found for '{filename}': {matches}"
                )
            audio = np.load(matches[0]).astype(np.float32)
            return audio, self.original_sample_rate

        else:
            codec_dir = self.species_folder / "Compressed_Audio" / f"{method}_{param}"
            path = codec_dir / f"{filename}_{method}_{param}.{method}"
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

    def _iter_windows(self, txt_path, trim=False):
        """
        Read file list, load audio sequentially, downsample, segment, and yield
        windows one at a time so that the full set of raw windows is never held
        in memory (it does not fit in RAM for the training set).

        trim : remove last 2-3 s from each file before resampling (training only).

        Yields
        ------
        (global_idx, class_idx, window) in processing order.
        """
        filenames = Path(txt_path).read_text().strip().split("\n")
        filenames = [f.strip() for f in filenames if f.strip()]

        global_counter = 0
        for filename in tqdm(filenames, desc=f"Loading {Path(txt_path).name}"):
            result = self._process_file(filename, trim=trim)
            if result is None:
                continue
            label_str, windows = result
            class_idx = self.label_map[label_str]
            for w in windows:
                yield global_counter, class_idx, w
                global_counter += 1

    def _window_to_images(self, window):
        """Split one 1-s window into sub-windows and return their spectrograms (k, n_mels, T)."""
        k = self.window_size // self.spec_window_size
        subs = window.reshape(k, self.spec_window_size)
        return np.stack([self._convert_single_to_image(s) for s in subs]).astype(np.float32)

    def _balanced_order(self, Y_all, selected_indices):
        """
        Global indices of the class-balanced training set, in X_balanced order.

        The npz keys are class integer indices (2=LAECAP, 3=TADAEG). Each array
        contains GLOBAL positions (into the full training window array sorted by
        processing order) to keep for that class. All other classes are kept in full.

        Output is class-grouped (all class-0 windows, then class-1, …) and within
        each over-represented class windows appear in the same order as the saved
        indices, reproducing the ordering of X_balanced in the reference notebook
        so that the pre-computed window mask aligns correctly.
        """
        order = []
        for class_idx_int in sorted(np.unique(Y_all)):
            if class_idx_int in selected_indices:
                keep_positions = selected_indices[class_idx_int]
                order.append(keep_positions[keep_positions < len(Y_all)])
            else:
                order.append(np.flatnonzero(Y_all == class_idx_int))
        return np.concatenate(order)

    def _load_mask(self, mask_path, expected_len):
        """Load a pre-computed boolean mask that removes empty/noise windows."""
        mask = np.load(mask_path)
        if mask.shape[0] != expected_len:
            raise ValueError(
                f"Mask length {mask.shape[0]} does not match number of windows {expected_len}. "
                "Ensure files are processed in the same order as when the mask was created."
            )
        return mask

    def _assemble(self, images_by_global, global_order, Y_all):
        """Stack spectrograms of the given global windows (sub-windows consecutive)."""
        k = self.window_size // self.spec_window_size
        first = next(iter(images_by_global.values()))
        X = np.empty((len(global_order) * k,) + first.shape[1:], dtype=np.float32)
        for i, g in enumerate(global_order):
            X[i * k:(i + 1) * k] = images_by_global[g]
        Y = np.repeat(Y_all[global_order], k)
        return X, Y

    def _convert_single_to_image(self, audio):
        """Convert an audio window to a normalized mel-spectrogram.

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_training_dataset(self):
        """
        Build the training dataset.

        Steps: load & segment → class balancing → window mask → spectrograms.
        Windows are streamed: only those that can survive class balancing are
        converted to spectrograms (raw audio windows are not kept in memory).

        Returns
        -------
        X : np.ndarray, shape (N, n_mels, time_steps)
        Y : np.ndarray, shape (N,), integer class labels
        """
        npz = np.load(self.filters_path / "selected_indices_reduction_2000.npz")
        selected_indices = {int(k): npz[k].astype(int) for k in npz.files}
        needed = set(np.concatenate(list(selected_indices.values())).tolist())

        images_by_global, labels = {}, []
        for g, class_idx, w in self._iter_windows(self.train_txt, trim=True):
            labels.append(class_idx)
            if class_idx not in selected_indices or g in needed:
                images_by_global[g] = self._window_to_images(w)
        Y_all = np.array(labels)

        label_names = {v: k for k, v in self.label_map.items()}
        before = {label_names[c]: int(n) for c, n in zip(*np.unique(Y_all, return_counts=True))}
        print(f"Before balancing: {len(Y_all)} windows — {before}")

        order = self._balanced_order(Y_all, selected_indices)
        print(f"After  balancing: {len(order)} windows — {dict(zip(*np.unique(Y_all[order], return_counts=True)))}")

        mask = self._load_mask(self.filters_path / "window_decisions_training_mask.npy", len(order))
        order = order[mask]
        print(f"After  mask:      {len(order)} windows — {dict(zip(*np.unique(Y_all[order], return_counts=True)))}")

        X, Y = self._assemble(images_by_global, order, Y_all)
        del images_by_global
        if self.spec_window_size != self.window_size:
            print(f"After  split:     {X.shape[0]} spectrograms of {self.spec_window_size} samples")
        return X, Y

    def create_testing_dataset(self):
        """
        Build the testing dataset.

        Steps: load & segment → window mask → spectrograms (no class balancing).
        Windows are streamed and only masked-in windows are converted.

        Returns
        -------
        X : np.ndarray, shape (N, n_mels, time_steps)
        Y : np.ndarray, shape (N,), integer class labels
        """
        mask = np.load(self.filters_path / "window_decisions_testing_mask.npy")

        images_by_global, labels = {}, []
        for g, class_idx, w in self._iter_windows(self.test_txt):
            labels.append(class_idx)
            if g < len(mask) and mask[g]:
                images_by_global[g] = self._window_to_images(w)
        Y_all = np.array(labels)

        print(f"Before mask: {len(Y_all)} windows — {dict(zip(*np.unique(Y_all, return_counts=True)))}")
        mask = self._load_mask(self.filters_path / "window_decisions_testing_mask.npy", len(Y_all))
        order = np.flatnonzero(mask)
        print(f"After mask: {len(order)} windows — {dict(zip(*np.unique(Y_all[order], return_counts=True)))}")

        X, Y = self._assemble(images_by_global, order, Y_all)
        del images_by_global
        if self.spec_window_size != self.window_size:
            print(f"After split: {X.shape[0]} spectrograms of {self.spec_window_size} samples")
        return X, Y

    def save_dataset(self, X, Y, output_path, dataset_type):
        """Save X and Y as pickle files.

        Files are written to:
            output_path/bats_{X|Y}_{dataset_type}.pkl                    (no compression)
            output_path/bats_{X|Y}_{dataset_type}_{method}_{param}.pkl   (with compression)
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        suffix = (
            f"_{self.method_compression}_{self.parameter_compression}"
            if self.method_compression is not None
            else ""
        )

        for name, data in [("X", X), ("Y", Y)]:
            path = output_path / dataset_type /f"bats_{name}_{dataset_type}{suffix}.pkl"
            with open(path, "wb") as f:
                pickle.dump(data, f, protocol=4)
            print(f"Saved → {path}")
