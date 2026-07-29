
import datetime
import os
import time
import gc
from pathlib import Path
import soundfile as sf
import torch
#import torchaudio
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

        block_size = int(
            self.block_duration_sec * self.model.sample_rate
        )

        for file in self.files:

            timing.append(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-5]
            )

            print(f"\nCompressing: {file}")

            file_input = self.folder_audio / file

            recording_folder = (
                self.latent_folder
                / f"{file[:-4]}_{self.method_compression}_{self.parameter_compression}"
            )

            recording_folder.mkdir(parents=True, exist_ok=True)

            metadata_file = recording_folder / "metadata.pt"

            # Skip if metadata already exists
            if metadata_file.exists():
                print("Already compressed.")
                continue

            #wav, sr = torchaudio.load(file_input)

            t0 = time.time()
            wav, sr = sf.read(file_input, always_2d=True)
            wav = torch.from_numpy(wav.T).float()
            print("Load:", time.time() - t0)


            t0 = time.time()
            wav = convert_audio(
                wav,
                sr,
                self.model.sample_rate,
                self.model.channels,
            )
            print("Convert:", time.time() - t0)

            wav = wav.unsqueeze(0)

            total_length = wav.shape[-1]

            # Save metadata once
            torch.save(
                {
                    "length": total_length,
                    "sample_rate": self.model.sample_rate,
                    "original_sample_rate": sr,
                    "chunk_length": chunk_length,
                    "step": step,
                    "block_size": block_size,
                },
                metadata_file,
            )
            t0 = time.time()
            with torch.inference_mode():

                for block_idx, block_start in enumerate(
                    range(0, total_length, block_size)
                ):

                    block_end = min(
                        block_start + block_size,
                        total_length,
                    )

                    print(
                        f"  Block {block_idx + 1}: "
                        f"{block_start/self.model.sample_rate:.1f}s "
                        f"-> "
                        f"{block_end/self.model.sample_rate:.1f}s"
                    )

                    block = wav[:, :, block_start:block_end]

                    latents = []

                    for start in range(0, block.shape[-1], step):

                        end = min(
                            start + chunk_length,
                            block.shape[-1],
                        )

                        chunk = block[:, :, start:end]

                        pad = chunk_length - chunk.shape[-1]

                        if pad > 0:
                            chunk = torch.nn.functional.pad(
                                chunk,
                                (0, pad),
                            )

                        encoded = self.model.encode(chunk)

                        codes, scale = encoded[0]

                        latents.append(
                            (
                                codes.cpu(),
                                scale.cpu() if scale is not None else None,
                            )
                        )

                    block_file = (
                        recording_folder
                        / f"block_{block_idx:03d}.pt"
                    )

                    print("Encode:", time.time() - t0)


                    t0 = time.time()
                    torch.save(
                        latents,
                        block_file,
                    )
                    print("Save:", time.time() - t0)

                    del latents
                    gc.collect()

            del wav
            gc.collect()

        return timing

    ####################################################################
    # Reconstruction
    ####################################################################

    def reconstruct(self):

        for file in self.files:

            print(f"\nReconstructing: {file}")

            recording_folder = (
                self.latent_folder
                / f"{file[:-4]}_{self.method_compression}_{self.parameter_compression}"
            )

            metadata_file = recording_folder / "metadata.pt"

            file_output = (
                self.compression_folder
                / f"{file[:-4]}_{self.method_compression}_{self.parameter_compression}.wav"
            )

            if file_output.exists():
                print("Already reconstructed.")
                continue

            if not metadata_file.exists():
                print("Metadata not found.")
                continue

            metadata = torch.load(metadata_file)

            total_length = metadata["length"]
            chunk_length = metadata["chunk_length"]
            step = metadata["step"]
            block_size = metadata["block_size"]

            reconstructed = torch.zeros(
                (1, self.model.channels, total_length),
                dtype=torch.float32,
            )

            with torch.inference_mode():

                block_files = sorted(recording_folder.glob("block_*.pt"))

                for block_idx, block_file in enumerate(block_files):

                    print(f"  Block {block_idx+1}/{len(block_files)}")

                    latents = torch.load(block_file)

                    decoded_chunks = []

                    for latent in latents:

                        decoded = self.model.decode([latent])

                        decoded_chunks.append(decoded)

                    block_length = min(
                        block_size,
                        total_length - block_idx * block_size,
                    )

                    reconstructed_block = self._overlap_add(
                        decoded_chunks,
                        block_length,
                        chunk_length,
                        step,
                        decoded_chunks[0].dtype,
                    )

                    start = block_idx * block_size
                    end = start + block_length

                    reconstructed[:, :, start:end] = reconstructed_block[
                        :, :, :block_length
                    ]

                    del latents
                    del decoded_chunks
                    del reconstructed_block

                    gc.collect()

            sf.write(
                file_output,
                reconstructed.squeeze(0).T.cpu().numpy(),
                self.model.sample_rate,
            )

            del reconstructed
            gc.collect()