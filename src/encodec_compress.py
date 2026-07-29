import datetime
import os
import gc
from pathlib import Path

import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio

class EncodecCompression:
    def __init__(self, folder_audio, folder_saved, parameter_compression="6.0"):
        self.folder_audio = Path(folder_audio)
        self.files = [f for f in os.listdir(self.folder_audio) if f.endswith(".WAV") or f.endswith(".wav")]
        
        self.method_compression = "encodec"
        self.parameter_compression = str(parameter_compression)
        self.bandwidth = float(parameter_compression)
        
        self.compression_folder = Path(folder_saved) / f"{self.method_compression}_{self.parameter_compression}"
        self.compression_folder.mkdir(parents=True, exist_ok=True)

        self.latent_base_folder = Path(folder_saved).parent / 'Compressed_Latents'
        self.latent_folder = self.latent_base_folder / f"{self.method_compression}_{self.parameter_compression}"
        self.latent_folder.mkdir(parents=True, exist_ok=True)
        
        # Initialize model on CPU explicitly
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(self.bandwidth)
        self.model.to('cpu')
        
    def compress(self):
        timing = []
        
        for file in self.files:
            timing.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-5])
            print("compression file :", file)
            
            file_input = self.folder_audio / file
            file_output = self.compression_folder / f"{file[:-4]}_{self.method_compression}_{self.parameter_compression}.wav"
            file_latent = self.latent_folder / f"{file[:-4]}_{self.method_compression}_{self.parameter_compression}.pt"
            
            # Skip if already compressed
            if file_output.exists() and file_latent.exists():
                print(f"Skipping {file} (already compressed)")
                continue
            
            # Load audio on CPU explicitly
            wav, sr = torchaudio.load(file_input)
            wav = wav.to('cpu')
            
            wav = convert_audio(wav, sr, self.model.sample_rate, self.model.channels)
            wav = wav.unsqueeze(0) # add batch dimension -> [1, channels, time]
            
            # Chunking logic: 1 second chunks with 10 ms overlap
            chunk_length_sec = 1.0
            overlap_sec = 0.010
            
            chunk_length = int(chunk_length_sec * self.model.sample_rate)
            overlap = int(overlap_sec * self.model.sample_rate)
            step = chunk_length - overlap
            
            total_length = wav.shape[-1]
            reconstructed_wav = torch.zeros_like(wav)
            overlap_add_weights = torch.zeros_like(wav)
            
            # Hanning window for smooth overlap-add
            #window = torch.hann_window(chunk_length).to('cpu')
            
            # use triangular window as Meta to reconstruct the signal
            t = torch.linspace(0, 1, chunk_length + 2, device='cpu', dtype=wav.dtype
            )[1:-1]

            window = 0.5 - (t - 0.5).abs()

            all_latents = []

            # Process chunks
            with torch.no_grad():
                for start in range(0, total_length, step):
                    end = min(start + chunk_length, total_length)
                    current_chunk_length = end - start
                    
                    chunk = wav[:, :, start:end]
                    
                    # Pad if chunk is smaller than 1 second (last chunk)
                    pad_length = chunk_length - current_chunk_length
                    if pad_length > 0:
                        chunk = torch.nn.functional.pad(chunk, (0, pad_length))
                        
                    # Compress and decompress
                    encoded_frames = self.model.encode(chunk)
                    decoded_chunk = self.model.decode(encoded_frames)
                    
                    cpu_frames = [(t.cpu(), s.cpu() if s is not None else None) for t, s in encoded_frames]
                    all_latents.append(cpu_frames)

                    # Apply window
                    decoded_chunk = decoded_chunk * window
                    
                    # Add back to the full sequence (Overlap-Add)
                    reconstructed_wav[:, :, start:end] += decoded_chunk[:, :, :current_chunk_length]
                    overlap_add_weights[:, :, start:end] += window[:current_chunk_length]
                    
                    # Memory Management
                    del chunk
                    del encoded_frames
                    del decoded_chunk
                    
            # Normalize overlap
            nonzero_idx = overlap_add_weights > 1e-12
            reconstructed_wav[nonzero_idx] /= overlap_add_weights[nonzero_idx]
            reconstructed_wav[~nonzero_idx] = 0.0
            
            # Save the reconstructed audio
            torchaudio.save(file_output, reconstructed_wav.squeeze(0), self.model.sample_rate)
            
            # Save the latents
            torch.save(all_latents, file_latent)
            
            # Memory Management
            del wav
            del reconstructed_wav
            del overlap_add_weights
            del window
            gc.collect()
            
        return timing
