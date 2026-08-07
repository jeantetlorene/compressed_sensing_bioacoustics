
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
        model_name="24khz",
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
        self.block_duration_sec=block_duration_sec

        self.compression_folder = (
            Path(folder_saved)
            / f"{self.method_compression}_{self.parameter_compression}"
        )
        self.compression_folder.mkdir(parents=True, exist_ok=True)
        

        self.latent_base_folder = Path(folder_saved).parent / "Compressed_Latents"
        self.latent_folder = (
            self.latent_base_folder
            / f"{self.method_compression}_{self.parameter_compression}"
        )
        self.latent_folder.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name
        if model_name == "24khz":
            self.model = EncodecModel.encodec_model_24khz()
        elif model_name == "48khz":
            self.model = EncodecModel.encodec_model_48khz()
        else:
            raise ValueError("model_name must be '24khz' or '48khz'")
        
        self.model.set_target_bandwidth(self.bandwidth)
        self.model.to("cpu")
        self.model.eval()


    ####################################################################
    # Compression
    ####################################################################

    def compress(self):

        timing = []

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

       
            wav, sr = sf.read(file_input, always_2d=True)
            wav = torch.from_numpy(wav.T).float()
         


       
            wav = convert_audio(
                wav,
                sr,
                self.model.sample_rate,
                self.model.channels,
            )
           

            wav = wav.unsqueeze(0)

            total_length = wav.shape[-1]

            # Save metadata once
            torch.save(
                {
                    "length": total_length,
                    "sample_rate": self.model.sample_rate,
                    "original_sample_rate": sr,
                    "block_size": block_size,
                },
                metadata_file,
            )
            t1 = time.time()
            with torch.inference_mode():

                for block_idx, block_start in enumerate(
                    range(0, total_length, block_size)
                ):

                    block_end = min(
                        block_start + block_size,
                        total_length,
                    )

                    #print(
                     #   f"  Block {block_idx + 1}: "
                      #  f"{block_start/self.model.sample_rate:.1f}s "
                       # f"-> "
                        #f"{block_end/self.model.sample_rate:.1f}s"
                    #)

                    block = wav[:, :, block_start:block_end]

                 
                    encoded = self.model.encode(block)
            
                    encoded = [
                        (
                            codes.cpu(),
                            scale.cpu() if scale is not None else None,
                        )
                        for codes, scale in encoded
                    ]

              
                    torch.save(
                    encoded,
                    recording_folder / f"block_{block_idx:03d}.pt",
                    )
                    

                    del encoded
                    del block
                    gc.collect()

            del wav
            gc.collect()
            print("Encode full file:", time.time() - t1)

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
            block_size = metadata["block_size"]

            reconstructed = torch.zeros(
                (1, self.model.channels, total_length),
                dtype=torch.float32,
            )

            block_files = sorted(recording_folder.glob("block_*.pt"))

            with torch.inference_mode():

                for block_idx, block_file in enumerate(block_files):

                    print(f"  Block {block_idx+1}/{len(block_files)}")

                    encoded = torch.load(block_file)

                    decoded = self.model.decode([encoded])

                    start = block_idx * block_size
                    end = min(
                    start + decoded.shape[-1],
                    total_length,
                    )

                    reconstructed[:, :, start:end] = decoded[
                        :, :, :end - start
                    ]

                  
                    del encoded
                    del decoded
                 

                    gc.collect()

            sf.write(
                file_output,
                reconstructed.squeeze(0).T.cpu().numpy(),
                self.model.sample_rate,
            )

            del reconstructed
            gc.collect()