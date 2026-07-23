import datetime
import logging
import os
import pickle
from pathlib import Path
from joblib import Parallel, delayed

import librosa
import numpy as np
import soundfile as sf
from scipy.signal.windows import tukey
from pydub import AudioSegment
from scipy.fft import dct, idct
from scipy.io.wavfile import write
from sklearn.linear_model import Lasso, OrthogonalMatchingPursuit
from tqdm import tqdm

logger = logging.getLogger(__name__)

class Compression:
    def __init__(self, folder_audio, folder_saved, method_compression, parameter_compression, converter_path):
        self.folder_audio=folder_audio
        self.files=[f for f in os.listdir(self.folder_audio) if f.endswith(".WAV") or f.endswith(".wav")]
        
        
        self.method_compression=method_compression
        self.parameter_compression=parameter_compression

        if self.method_compression in ["mp3", "aac"]:
            self.parameters = ["-b:a", parameter_compression]
        elif self.method_compression=="opus":
            self.parameters = ["-b:a", parameter_compression]
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
            logger.info("Compressing (codec): %s", file)
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
    def __init__(self, folder_audio, folder_saved, sample_rate, frame_size, overlap, compression_rate=0.15, seed=42, n_jobs=-1, window_type="hann"):
        
        #folders
        self.folder_audio=folder_audio
        self.folder_compressed_saved=Path(folder_saved, "cs_"+str(compression_rate))
        # Create the folder if it doesn't exist
        os.makedirs(self.folder_compressed_saved, exist_ok=True)
        self.folder_reconstructed_saved=Path(folder_saved, "cs_reconstructed_"+str(compression_rate))
    
        # Create the folder if it doesn't exist
        os.makedirs(self.folder_reconstructed_saved, exist_ok=True)
        print("folder where to save : ", self.folder_reconstructed_saved)

        #parameters segmentation
        self.sample_rate=sample_rate
        self.frame_size=frame_size #in seconds
        self.overlap=overlap #percentage 
        self.window_type=window_type

        #parameters compression/reconstruction
        self.compression_rate=compression_rate
        self.seed=seed
        self.n_jobs=n_jobs
        self.batch_size = 256



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

    def get_window(self):

        if self.window_type == "rect":
            return np.ones(self.frame_size)

        elif self.window_type == "hann":
            return np.hanning(self.frame_size)
        elif self.window_type == "tukey":
            return tukey(self.frame_size,
                        alpha=0.5)

        else:
            raise ValueError(
                f"Unknown window : {self.window_type}"
            )


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
        logger.debug("Sliding window produced %d frames", num_frames)
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
        frames = np.asarray(frames, dtype=np.float64)
        num_frames = frames.shape[0]
        # output length: last start + N
        out_len = hop * (num_frames - 1) + self.frame_size
        out = np.zeros(out_len, dtype=np.float64)
        weight = np.zeros(out_len, dtype=np.float64)

        if window is None:
            w = np.ones(self.frame_size, dtype=np.float64)
        else:
            w = np.asarray(window, dtype=np.float64)
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

    def compress_one_file(self, audio, sample_rate, idx, file_name_no_extension):
        cs_compressed_data = []
        audio_windows = self.segment_audio_sliding_window(audio)

        for window in audio_windows:
            y = self.compress_1D(window, idx)
            cs_compressed_data.append(y)

        compressed_data = np.concatenate(cs_compressed_data)
        y_int16 = (compressed_data * 32767).astype(np.int16)
        file_name = f"{self.folder_compressed_saved}/{file_name_no_extension}_{len(cs_compressed_data)}_compressed.wav"
        write(file_name, sample_rate, y_int16)

        return cs_compressed_data



    def compress_folder(self):
        timing=[]
        files=[f for f in os.listdir(self.folder_audio) if f.endswith(".wav") or f.endswith(".WAV")]
        
        idx=self.compress_matrix_1D()

        for file in files :
            logger.info("Compressing (legacy): %s", file)
            timing.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-5])

            file_name_no_extension=file[:-4]
            audio, sample_rate = librosa.load(Path(self.folder_audio, file),sr=None)
  
            self.compress_one_file(audio, sample_rate, idx, file_name_no_extension)
        
        file_name=f"{self.folder_compressed_saved}/idx_matrix.pkl"
        with open(file_name, "wb") as f:
                pickle.dump(idx, f) 

        return timing

 

    def reconstruction(self, solver="lasso", alpha=1e-8, saved_in_wav=False):
        timing=[]
        files=[f for f in os.listdir(self.folder_compressed_saved) if f.endswith(".wav") and f != "idx_matrix.pkl"]
        
        with open(Path(self.folder_compressed_saved, "idx_matrix.pkl"), "rb") as f:
               idx = pickle.load(f)
        
        A = self.csmtx_dct(self.frame_size, idx)

        for file in files :
            logger.info("Reconstructing (legacy): %s", file)
            timing.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-5])

            file_name_no_extension=file[:-15]
            print(file_name_no_extension)
            nb_windows=int(file.split("_")[-2])

            compressed_file_int16, _=sf.read(Path(self.folder_compressed_saved, file), dtype='int16')
            compressed_file = compressed_file_int16.astype(np.float32) / 32767.0
            del compressed_file_int16

            loaded_list=compressed_file.reshape((nb_windows, len(idx)))
            reconstructed_frames = Parallel(n_jobs=self.n_jobs)(
                        delayed(self.reconstruct_frame)(y, solver, alpha, A) for y in tqdm(loaded_list)
                    )


            #window = np.hanning(self.frame_size)
            window=self.get_window()
            reconstructed_signal = self.overlap_add(reconstructed_frames, window=window)

            if saved_in_wav==True : 
                audio_int16 = np.int16(np.clip(reconstructed_signal, -1.0, 1.0) * 32767)
                write(Path(self.folder_reconstructed_saved, file_name_no_extension+"_reconstructed.wav"), self.sample_rate, audio_int16)
                saved_name = f"{file_name_no_extension}_reconstructed.wav"
            else : 
                np.save(Path(self.folder_reconstructed_saved, file_name_no_extension+"_reconstructed.npy"), reconstructed_signal.astype(np.float32, copy=False))
                saved_name = f"{file_name_no_extension}_reconstructed.npy"
                print(Path(self.folder_reconstructed_saved, file_name_no_extension+"_reconstructed.npy"))
            logger.info("File saved: %s", saved_name)

        return timing


    