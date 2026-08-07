
import datetime
import os
import time
import gc
from pathlib import Path
import soundfile as sf
import numpy as np
import torch
#import torchaudio
import pprint
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
        window_duration_sec=None,
        batch_size=1,
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
        self.block_duration_sec = block_duration_sec
        self.window_duration_sec = window_duration_sec
        self.batch_size = batch_size

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


                    ##############################################################
                    # ORIGINAL ENCODEC
                    ##############################################################
                    if self.window_duration_sec is None:
                        codes, scale = self.model.encode(block)[0]

                        codes = codes.cpu().numpy().astype(np.uint16)

                        if scale is not None:
                            np.save(
                                recording_folder / f"block_{block_idx:03d}_scale.npy",
                                scale.cpu().numpy(),
                                allow_pickle=False,
                            )
                   
        

                        np.save(
                            recording_folder / f"block_{block_idx:03d}_codes.npy",
                            codes,
                            allow_pickle=False,
                        )

                
                        torch.save(
                            {
                                "window_size": None,
                                "num_windows": 1,
                            },
                            recording_folder / f"block_{block_idx:03d}_meta.pt",
                        )

                    
                    ##############################################################
                    # WINDOWED / BATCHED MODE
                    ##############################################################
                    else:

                        window_size = int(
                            self.window_duration_sec *
                            self.model.sample_rate
                        )

                        ##########################################################
                        # Create windows
                        ##########################################################

                        windows = []

                        for start in range(
                            0,
                            block.shape[-1],
                            window_size,
                        ):

                            end = min(
                                start + window_size,
                                block.shape[-1],
                            )

                            window = block[:, :, start:end]

                            if window.shape[-1] < window_size:

                                window = torch.nn.functional.pad(
                                    window,
                                    (
                                        0,
                                        window_size - window.shape[-1],
                                    ),
                                )

                            windows.append(window)

                        windows = torch.cat(
                            windows,
                            dim=0,
                        )

                        ##########################################################
                        # Encode batches
                        ##########################################################

                        num_batches = 0 

                        for i in range(
                            0,
                            windows.shape[0],
                            self.batch_size,
                        ):

                            batch = windows[
                                i:i+self.batch_size
                            ]

                            codes, scale = self.model.encode(batch)[0]
                            
                            
                            codes = codes.cpu().numpy().astype(np.uint16)


                           

                            ##########################################################
                            # Save
                            ##########################################################
                            # Save codes
                            np.save(
                                recording_folder / f"block_{block_idx:03d}_batch_{num_batches:04d}_codes.npy",
                                codes,
                                allow_pickle=False,
                            )

                            if scale is not None:
                                np.save(
                                    recording_folder /
                                    f"block_{block_idx:03d}_batch_{num_batches:04d}_scale.npy",
                                    scale.cpu().numpy(),
                                    allow_pickle=False,
                                )

                            num_batches += 1


                        torch.save(
                            {
                                "window_size": window_size,
                                "num_windows": windows.shape[0],
                                "valid_length": block.shape[-1],
                                "num_batches": num_batches,
                            },
                            recording_folder / f"block_{block_idx:03d}_meta.pt",
                        )

                        del windows
                 

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
            original_sr = metadata["original_sample_rate"]
            

            block_files = sorted(recording_folder.glob("block_*_meta.pt"))


            with sf.SoundFile(
                    file_output,
                    mode="w",
                    samplerate=self.model.sample_rate,
                    channels=self.model.channels,
                    subtype="PCM_16",
                ) as outfile:

                t1 = time.time()

                with torch.inference_mode():
                    for block_idx, block_file in enumerate(block_files):

                        prefix = block_file.stem.replace("_meta", "")
                        saved = torch.load(block_file)

                        window_size = saved["window_size"]

                        ###################################################
                        # ORIGINAL ENCODEC MODE
                        ###################################################
                        if window_size is None:

                            codes = torch.from_numpy(
                                np.load(
                                    recording_folder /
                                    f"{prefix}_codes.npy"
                                )
                            ).to(torch.int64)

                            scale_file = (
                                recording_folder /
                                f"{prefix}_scale.npy"
                            )

                            if scale_file.exists():
                                scale = torch.from_numpy(np.load(scale_file)).float()
                            else:
                                scale = None

                            decoded = self.model.decode(
                                [(codes, scale)]
                            )

                            audio = (
                                decoded.squeeze(0)
                                .T.cpu()
                                .numpy()
                            )

                            outfile.write(audio)

                            del decoded
                            del codes

                        ###################################################
                        # WINDOWED / BATCHED MODE
                        ###################################################
                        else:

                            decoded_windows = []

                            for batch_idx in range(saved["num_batches"]):

                                codes = torch.from_numpy(
                                    np.load(
                                        recording_folder /
                                        f"{prefix}_batch_{batch_idx:04d}_codes.npy"
                                    )
                                ).to(torch.int64)

                                scale_file = (
                                    recording_folder /
                                    f"{prefix}_batch_{batch_idx:04d}_scale.npy"
                                )

                                if scale_file.exists():
                                    scale = torch.from_numpy(np.load(scale_file)).float()
                                else:
                                    scale = None

                                decoded = self.model.decode(
                                    [(codes, scale)]
                                )

                                decoded_windows.append(decoded.cpu())

                                del decoded
                                del codes

                            decoded_windows = torch.cat(
                                decoded_windows,
                                dim=0,
                            )

                            reconstructed = decoded_windows.reshape(
                                -1,
                                decoded_windows.shape[-1],
                            )

                            reconstructed = reconstructed[
                                :saved["num_windows"]
                            ]

                            reconstructed = reconstructed.reshape(-1)

                            audio = reconstructed.numpy()[:saved["valid_length"]]

                            outfile.write(audio)

                            del decoded_windows
                            del reconstructed

                        gc.collect()
                print("Decode full file:", time.time() - t1)
                    

