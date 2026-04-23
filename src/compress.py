import datetime
import os
import pickle
from pathlib import Path
from joblib import Parallel, delayed

import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from scipy.fft import dct, idct
from scipy.io.wavfile import write
from sklearn.linear_model import Lasso, OrthogonalMatchingPursuit
from tqdm import tqdm

class Compression_vs2:
    def __init__(self, folder_audio, folder_saved, method_compression, parameter_compression, converter_path):
        self.folder_audio=folder_audio
        self.files=[f for f in os.listdir(self.folder_audio) if f.endswith(".WAV") or f.endswith(".wav")]
        
        
        self.method_compression=method_compression
        self.parameter_compression=parameter_compression

        if self.method_compression in ["mp3", "aac"]:
            self.parameters = ["-b:a", parameter_compression]
        elif self.method_compression=="opus":
            self.parameters = ["-b:a", parameter_compression, "-ar", "24000"]
        elif self.method_compression=="ogg":
            self.parameters=["-qscale:a", str(parameter_compression)]
        elif self.method_compression=='flac':
            self.parameters=["-compression_level", str(parameter_compression)]

        self.compression_folder=Path(folder_saved, f"{method_compression}_{parameter_compression}") 
        #create the folder if doesn't exist 
        self.compression_folder.mkdir(parents=True, exist_ok=True)

        #list all the file in the folder 

        #AudioSegment.converter = converter_path

    def compress(self):
        timing=[]

        for file in self.files : 
            timing.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-5])
            print("compression file :", file)
            file_input=Path(self.folder_audio,file)
            file_output=Path(self.compression_folder) / f"{file[:-4]}_{self.method_compression}_{self.parameter_compression}.{self.method_compression}"
            #file_output=Path(self.compression_folder, file[:-4]+"."+ self.compression_method)
            audio_segment = AudioSegment.from_file(file_input)
            if self.method_compression=="aac":
                audio_segment.export(out_f=file_output, format="adts", parameters=self.parameters)
            else : 
                audio_segment.export(out_f=file_output, format=self.method_compression, parameters=self.parameters)
        return timing

class CS: 
    def __init__(self, folder_audio, folder_saved, sample_rate, frame_size, overlap, compression_rate=0.15, seed=42, n_jobs=-1):
        
        #folders
        self.folder_audio=folder_audio
        self.folder_compressed_saved=Path(folder_saved, "cs_"+str(compression_rate))
        # Create the folder if it doesn't exist
        os.makedirs(self.folder_compressed_saved, exist_ok=True)
        self.folder_reconstructed_saved=Path(folder_saved, "cs_reconstructed_"+str(compression_rate))
    
        # Create the folder if it doesn't exist
        os.makedirs(self.folder_reconstructed_saved, exist_ok=True)

        #parameters segmentation
        self.sample_rate=sample_rate
        self.frame_size=frame_size #in seconds
        self.overlap=overlap #percentage 

        #parameters compression/reconstruction
        self.compression_rate=compression_rate
        self.seed=seed
        self.n_jobs=n_jobs
        self.batch_size = 256
        self.analysis_window = np.sqrt(np.hanning(self.frame_size)).astype(np.float32)
        if not np.any(self.analysis_window):
            self.analysis_window = np.ones(self.frame_size, dtype=np.float32)


    def csmtx_dct(self, N, idx):
        K = len(idx)
        A = np.zeros((K, N))
        for i, j in enumerate(idx):
            A[i, :] = dct(np.eye(1, N, j).flatten(), norm='ortho')
        return A

    def calculate_frame_size(self, total_length ,fixed_overlap, max_frame=None, min_frame =1):
        if max_frame is None : 
            max_frame=total_length
        valid_frames = []
        for window_frame in range(min_frame, max_frame + 1):
            step = window_frame - fixed_overlap
            if step <= 0:
                continue
            num_segments = 1 + (total_length - window_frame) // step
            remainder = (total_length - window_frame) % step
            if remainder == 0:
                valid_frames.append((window_frame, num_segments))

        return valid_frames


    # Function to compress the 1D-signal
    def compress_matrix_1D(self):
        N=self.frame_size
        np.random.seed(self.seed)
        M = int(self.compression_rate * N)
        idx = np.random.choice(N, M, replace=False)
        return idx
    
    def compress_1D(self, X , idx):
        return np.array(X)[idx]

    def _get_hop_size(self):
        hop = int(round(self.frame_size * (1 - self.overlap)))
        if hop <= 0:
            raise ValueError("overlap is too large and produces a non-positive hop size.")
        return hop

    def _make_support_matrix(self, idx, support):
        basis = np.eye(self.frame_size, dtype=np.float32)[:, support]
        return idct(basis, norm="ortho", axis=0)[idx, :]

    def _iht_reconstruction_batch(
        self,
        y_batch,
        idx,
        sparsity=None,
        max_iter=60,
        tol=1e-4,
        debias=True,
    ):
        y_batch = np.asarray(y_batch, dtype=np.float32)
        if y_batch.ndim == 1:
            y_batch = y_batch[np.newaxis, :]

        batch_size = y_batch.shape[0]
        coeffs = np.zeros((batch_size, self.frame_size), dtype=np.float32)

        if sparsity is None:
            sparsity = max(8, min(len(idx) // 3, self.frame_size // 6))
        sparsity = max(1, min(int(sparsity), self.frame_size))

        for _ in range(max_iter):
            reconstructed = idct(coeffs, norm="ortho", axis=1)
            residual = y_batch - reconstructed[:, idx]

            padded_residual = np.zeros_like(coeffs)
            padded_residual[:, idx] = residual
            updated = coeffs + dct(padded_residual, norm="ortho", axis=1)

            keep_idx = np.argpartition(np.abs(updated), -sparsity, axis=1)[:, -sparsity:]
            next_coeffs = np.zeros_like(updated)
            row_ids = np.arange(batch_size)[:, None]
            next_coeffs[row_ids, keep_idx] = updated[row_ids, keep_idx]

            diff = np.linalg.norm(next_coeffs - coeffs)
            base = np.linalg.norm(coeffs) + 1e-8
            coeffs = next_coeffs
            if diff / base < tol:
                break

        if debias:
            for i in range(batch_size):
                support = np.flatnonzero(coeffs[i])
                if support.size == 0:
                    continue
                A_support = self._make_support_matrix(idx, support)
                support_coeffs, _, _, _ = np.linalg.lstsq(A_support, y_batch[i], rcond=None)
                coeffs[i, support] = support_coeffs.astype(np.float32, copy=False)

        return idct(coeffs, norm="ortho", axis=1).astype(np.float32, copy=False)

    def npy_to_wav(self, npy_file, wav_file=None, sample_rate=None):
        npy_file = Path(npy_file)
        if wav_file is None:
            wav_file = npy_file.with_suffix(".wav")
        else:
            wav_file = Path(wav_file)

        if sample_rate is None:
            sample_rate = self.sample_rate

        audio = np.load(npy_file).astype(np.float32, copy=False)
        audio = np.squeeze(audio)
        if audio.ndim != 1:
            raise ValueError("The .npy file must contain a 1D reconstructed audio signal.")

        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = np.int16(audio * 32767)
        write(wav_file, sample_rate, audio_int16)
        print(f"file saved as {wav_file}")
        return wav_file

    def convert_reconstructed_folder_to_wav(self, folder=None, sample_rate=None):
        if folder is None:
            folder = self.folder_reconstructed_saved
        else:
            folder = Path(folder)

        converted_files = []
        for npy_file in folder.glob("*_reconstructed.npy"):
            converted_files.append(self.npy_to_wav(npy_file, sample_rate=sample_rate))

        return converted_files
    
    # Function to reconstruct a single frame in a audio
    # Function to reconstruct a single frame of a segment in a audio
    def reconstruct_frame(self, y, solver, alpha, A):
        if solver == 'lasso':
            lasso = Lasso(alpha=alpha, max_iter=5000)
            lasso.fit(A, y)
            reconstructed_coeffs = lasso.coef_
        elif solver == 'omp':
            omp = OrthogonalMatchingPursuit()
            omp.fit(A, y)
            reconstructed_coeffs = omp.coef_
        else:
            raise ValueError("Unsupported solver. Use 'lasso' or 'omp'.")

        # Reconstruct the time domain signal from its frequency using IDCT
        # WE converted into DCT coeffs in the csmts_dct, it is necessary to convert it back
        X_reconstructed = idct(reconstructed_coeffs, norm='ortho')
        return X_reconstructed

    def segment_audio_fixed_window(self, audio):
        n_windows=len(audio)//self.frame_size
        #segment_samples = sample_rate * self.window_size
        #segments = [audio[i:i + segment_samples] for i in range(0, len(audio), segment_samples)]
        segments=np.split(audio[:n_windows * self.frame_size], n_windows)
        return segments
    
    def segment_audio_sliding_window(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        num_samples = len(audio)
        if num_samples < self.frame_size:
            padded = np.zeros(self.frame_size, dtype=np.float32)
            padded[:num_samples] = audio
            return padded[np.newaxis, :]

        hop_size = self._get_hop_size()
        num_frames = 1 + (num_samples - self.frame_size) // hop_size
        windows = np.lib.stride_tricks.sliding_window_view(audio, self.frame_size)
        segments = windows[::hop_size][:num_frames]
        print(num_frames)
        return np.ascontiguousarray(segments, dtype=np.float32)
    
    def overlap_add(self, frames, window=None):
        """
        frames: array-like shape (num_frames, N)  -- time-domain frames (reconstructed)
        N: int -- frame length
        hop: int -- hop size (samples between frame starts)
        window: array-like length N or None. If None, uses rectangular (no window).
        Returns: 1D numpy array with reconstructed signal
        """

        hop = self._get_hop_size()
        frames = np.asarray(frames, dtype=np.float32)
        num_frames = frames.shape[0]
        # output length: last start + N
        out_len = hop * (num_frames - 1) + self.frame_size
        out = np.zeros(out_len, dtype=np.float64)
        weight = np.zeros(out_len, dtype=np.float64)

        if window is None:
            w = np.ones(self.frame_size, dtype=np.float32)
        else:
            w = np.asarray(window, dtype=np.float32)
            assert w.shape[0] == self.frame_size

        for i in range(num_frames):
            start = i * hop
            out[start:start+self.frame_size] += frames[i] * w
            weight[start:start+self.frame_size] += w

        # avoid division by zero
        nonzero = weight > 1e-12
        out[nonzero] /= weight[nonzero]
        out[~nonzero] = 0.0

        return out

    def compress_one_file_legacy(self, audio, sample_rate, idx, file_name_no_extension):
        cs_compressed_data = []
        audio_windows = self.segment_audio_sliding_window(audio)

        for window in tqdm(audio_windows, desc="Compressing audio segments", unit="segment"):
            y = self.compress_1D(window, idx)
            cs_compressed_data.append(y)

        compressed_data = np.concatenate(cs_compressed_data)
        y_int16 = (compressed_data * 32767).astype(np.int16)
        file_name = f"{self.folder_compressed_saved}/{file_name_no_extension}_{len(cs_compressed_data)}_compressed.wav"
        write(file_name, sample_rate, y_int16)

        return cs_compressed_data


    def compress_one_file(self, audio, sample_rate, idx, file_name_no_extension):
        audio_windows = self.segment_audio_sliding_window(audio)
        compressed_batches = []

        for start in tqdm(range(0, audio_windows.shape[0], self.batch_size), desc="Compressing audio segments", unit="batch"):
            batch = audio_windows[start:start + self.batch_size]
            windowed_frames = batch * self.analysis_window[np.newaxis, :]
            compressed_batches.append(windowed_frames[:, idx].astype(np.float32, copy=False))

        compressed_frames = np.concatenate(compressed_batches, axis=0)

        file_name = Path(self.folder_compressed_saved, f"{file_name_no_extension}_{compressed_frames.shape[0]}_compressed.npy")
        np.save(file_name, compressed_frames.reshape(-1))

        return compressed_frames

    def compress_folder_legacy(self):
        timing=[]
        files=[f for f in os.listdir(self.folder_audio) if f.endswith(".wav") or f.endswith(".WAV")]
        
        idx=self.compress_matrix_1D()

        for file in files : 
            print("compression file :", file)
            timing.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-5])
          
            file_name_no_extension=file[:-4]
            audio, sample_rate = librosa.load(Path(self.folder_audio, file),sr=None)
            self.compress_one_file_legacy(audio, sample_rate, idx, file_name_no_extension)
        
        file_name=f"{self.folder_compressed_saved}/idx_matrix.pkl"
        with open(file_name, "wb") as f:
                pickle.dump(idx, f) 

        return timing

    def compress_folder(self):
        timing=[]
        files=[f for f in os.listdir(self.folder_audio) if f.endswith(".wav") or f.endswith(".WAV")]
        
        #generate the fixed compression matrix : 
        idx=self.compress_matrix_1D()


        for file in files : 
            print("compression file :", file)
            timing.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-5])
          
            file_name_no_extension=file[:-4]

            #open the file
            audio, sample_rate = librosa.load(Path(self.folder_audio, file),sr=None)
            
            
            self.compress_one_file(audio, sample_rate, idx, file_name_no_extension)
        
        #save the compression matrix 
        print("save the idx matrice")
        file_name=f"{self.folder_compressed_saved}/idx_matrix.pkl"
        with open(file_name, "wb") as f:
                pickle.dump(idx, f) 

        return timing

    def reconstruction_legacy(self, solver="lasso", alpha=1e-8, saved_in_wav=False):
        timing=[]
        files=[f for f in os.listdir(self.folder_compressed_saved) if f.endswith(".wav") and f != "idx_matrix.pkl"]
        
        with open(Path(self.folder_compressed_saved, "idx_matrix.pkl"), "rb") as f:
               idx = pickle.load(f)
        
        A = self.csmtx_dct(self.frame_size, idx)

        for file in files : 
            print("reconstruction file :", file)
            timing.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-5])
          
            file_name_no_extension=file[:-15]
            nb_windows=int(file.split("_")[-2])

            compressed_file_int16, _=sf.read(Path(self.folder_compressed_saved, file), dtype='int16')
            compressed_file = compressed_file_int16.astype(np.float32) / 32767.0
            del compressed_file_int16

            loaded_list=compressed_file.reshape((nb_windows, len(idx)))
            reconstructed_frames = Parallel(n_jobs=self.n_jobs)(
                        delayed(self.reconstruct_frame)(y, solver, alpha, A) for y in tqdm(loaded_list)
                    )


            window = np.hanning(self.frame_size)
            reconstructed_signal = self.overlap_add(reconstructed_frames, window=window)

            if saved_in_wav==True : 
                audio_int16 = np.int16(np.clip(reconstructed_signal, -1.0, 1.0) * 32767)
                write(Path(self.folder_reconstructed_saved, file_name_no_extension+"_reconstructed.wav"), self.sample_rate, audio_int16)
                saved_name = f"{file_name_no_extension}_reconstructed.wav"
            else : 
                np.save(Path(self.folder_reconstructed_saved, file_name_no_extension+"_reconstructed.npy"), reconstructed_signal.astype(np.float32, copy=False))
                saved_name = f"{file_name_no_extension}_reconstructed.npy"
            print(f"file saved as {saved_name}")

        return timing


    def reconstruction(self, solver="iht", alpha=1e-4, saved_in_wav=False):
        timing=[]
        files = [
            f
            for f in os.listdir(self.folder_compressed_saved)
            if (
                (f.endswith(".npy") or f.endswith(".wav"))
                and f != "idx_matrix.pkl"
                and "_compressed" in f
            )
        ]
        
        #open idx matrice
        with open(Path(self.folder_compressed_saved, "idx_matrix.pkl"), "rb") as f:
               idx = pickle.load(f)
        
        
        for file in files : 
            print("reconstruction file :", file)
            timing.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-5])
            if file.endswith("_compressed.npy"):
                file_name_no_extension = file[: -len("_compressed.npy")]
            else:
                file_name_no_extension = file[: -len("_compressed.wav")]

            nb_windows = int(file_name_no_extension.split("_")[-1])
            file_stem = file_name_no_extension.rsplit("_", 1)[0]

            file_path = Path(self.folder_compressed_saved, file)
            if file.endswith(".npy"):
                compressed_file = np.load(file_path).astype(np.float32, copy=False)
            else:
                compressed_file_int16, _ = sf.read(file_path, dtype="int16")
                compressed_file = compressed_file_int16.astype(np.float32) / 32767.0
                del compressed_file_int16

            loaded_list = compressed_file.reshape((nb_windows, len(idx)))

            if solver == "iht":
                reconstructed_batches = []
                for start in tqdm(range(0, loaded_list.shape[0], self.batch_size), desc="Reconstructing audio segments", unit="batch"):
                    batch = loaded_list[start:start + self.batch_size]
                    reconstructed_batches.append(
                        self._iht_reconstruction_batch(
                            batch,
                            idx,
                            sparsity=max(8, min(len(idx) // 3, self.frame_size // 6)),
                            max_iter=80,
                            tol=alpha if alpha is not None else 1e-4,
                            debias=True,
                        )
                    )
                reconstructed_frames = np.concatenate(reconstructed_batches, axis=0)
            else:
                A = self.csmtx_dct(self.frame_size, idx)
                reconstructed_frames = np.asarray(
                    [self.reconstruct_frame(y, solver, alpha, A) for y in tqdm(loaded_list)],
                    dtype=np.float32,
                )

            reconstructed_signal = self.overlap_add(reconstructed_frames, window=self.analysis_window)

    
                
            
            #save numpy file 
            if saved_in_wav==True : 
                audio_int16 = np.int16(np.clip(reconstructed_signal, -1.0, 1.0) * 32767)
                write(Path(self.folder_reconstructed_saved, file_stem+"_reconstructed.wav"), self.sample_rate, audio_int16)
                saved_name = f"{file_stem}_reconstructed.wav"
            else : 
                np.save(Path(self.folder_reconstructed_saved, file_stem+"_reconstructed.npy"), reconstructed_signal.astype(np.float32, copy=False))
                saved_name = f"{file_stem}_reconstructed.npy"
            print(f"file saved as {saved_name}")

        return timing
