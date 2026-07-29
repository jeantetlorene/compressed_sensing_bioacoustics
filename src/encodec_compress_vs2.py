
import datetime
import os
import gc
from pathlib import Path

import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio

class EncodecCompression:

    def __init__(
        self,
        folder_audio,
        folder_saved,
        parameter_compression="6.0",
        chunk_length_sec=1.0,
        overlap_sec=0.010,
        block_duration_sec=600,
    ):

        self.folder_audio = Path(folder_audio)
        self.files = [
            f for f in os.listdir(self.folder_audio)
            if f.lower().endswith(".wav")
        ]

        self.method_compression = "encodec"
        self.parameter_compression = str(parameter_compression)
        self.bandwidth = float(parameter_compression)

        self.chunk_length_sec = chunk_length_sec
        self.overlap_sec = overlap_sec

        self.compression_folder = (
            Path(folder_saved)
            / f"{self.method_compression}_{self.parameter_compression}"
        )
        self.compression_folder.mkdir(parents=True, exist_ok=True)
        self.block_duration_sec=block_duration_sec

        self.latent_base_folder = Path(folder_saved).parent / "Compressed_Latents"
        self.latent_folder = (
            self.latent_base_folder
            / f"{self.method_compression}_{self.parameter_compression}"
        )
        self.latent_folder.mkdir(parents=True, exist_ok=True)

        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(self.bandwidth)
        self.model.to("cpu")
        self.model.eval()

    ####################################################################
    # Helper functions
    ####################################################################

    def _get_window(self, length, dtype):

        t = torch.linspace(
            0,
            1,
            length + 2,
            dtype=dtype,
            device="cpu"
        )[1:-1]

        return 0.5 - (t - 0.5).abs()

    def _overlap_add(self, decoded_chunks, total_length, chunk_length, step, dtype):

        reconstructed = torch.zeros(
            (1, self.model.channels, total_length),
            dtype=dtype
        )

        weights = torch.zeros_like(reconstructed)

        window = self._get_window(chunk_length, dtype)

        for i, chunk in enumerate(decoded_chunks):

            start = i * step
            end = min(start + chunk_length, total_length)

            current_length = end - start

            reconstructed[:, :, start:end] += (
                chunk[:, :, :current_length] * window[:current_length]
            )

            weights[:, :, start:end] += window[:current_length]

        mask = weights > 1e-12
        reconstructed[mask] /= weights[mask]
        reconstructed[~mask] = 0

        return reconstructed

    ####################################################################
    # Compression
    ####################################################################

    def compress(self):

        timing = []

        chunk_length = int(
            self.chunk_length_sec * self.model.sample_rate
        )

        overlap = int(
            self.overlap_sec * self.model.sample_rate
        )

        step = chunk_length - overlap

        for file in self.files:

            timing.append(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-5]
            )

            print("Compressing:", file)

            file_input = self.folder_audio / file

            file_latent = (
                self.latent_folder
                / f"{file[:-4]}_{self.method_compression}_{self.parameter_compression}.pt"
            )

            if file_latent.exists():
                print("Already compressed.")
                continue

            wav, sr = torchaudio.load(file_input)

            wav = convert_audio(
                wav,
                sr,
                self.model.sample_rate,
                self.model.channels,
            )

            wav = wav.unsqueeze(0)

            total_length = wav.shape[-1]

            latents = []

            with torch.inference_mode():

                for start in range(0, total_length, step):

                    end = min(start + chunk_length, total_length)

                    chunk = wav[:, :, start:end]

                    pad = chunk_length - chunk.shape[-1]

                    if pad > 0:
                        chunk = torch.nn.functional.pad(chunk, (0, pad))

                    encoded = self.model.encode(chunk)

                    # save only the encoded frame
                    codes, scale = encoded[0]

                    latents.append(
                        (
                            codes.cpu(),
                            scale.cpu() if scale is not None else None,
                        )
                    )

            torch.save(
                {
                    "latents": latents,
                    "length": total_length,
                    "sample_rate": self.model.sample_rate,   # 24 kHz after conversion
                    "original_sample_rate": sr,              # original WAV sampling rate
                    "chunk_length": chunk_length,
                    "step": step,
                },
                file_latent,
            )

            gc.collect()

        return timing

    ####################################################################
    # Reconstruction
    ####################################################################

    def reconstruct(self):

        for file in self.files:

            print("Reconstructing:", file)

            file_latent = (
                self.latent_folder
                / f"{file[:-4]}_{self.method_compression}_{self.parameter_compression}.pt"
            )

            file_output = (
                self.compression_folder
                / f"{file[:-4]}_{self.method_compression}_{self.parameter_compression}.wav"
            )

            if file_output.exists():
                print("Already reconstructed.")
                continue

            saved = torch.load(file_latent)
            chunk_length = saved["chunk_length"]
            step = saved["step"]

            latents = saved["latents"]
            total_length = saved["length"]

            decoded_chunks = []

            with torch.inference_mode():

                for latent in latents:

                    decoded = self.model.decode([latent])

                    decoded_chunks.append(decoded)

            reconstructed = self._overlap_add(
                decoded_chunks,
                total_length,
                chunk_length,
                step,
                decoded_chunks[0].dtype,
            )

            torchaudio.save(
                file_output,
                reconstructed.squeeze(0),
                self.model.sample_rate,
            )

            gc.collect()