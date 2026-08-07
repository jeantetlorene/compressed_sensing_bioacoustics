
import datetime
import os
import time
import gc
from pathlib import Path
import soundfile as sf
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
                        encoded = self.model.encode(block)
            
                        encoded = [
                            (
                                codes.cpu(),
                                scale.cpu() if scale is not None else None,
                            )
                            for codes, scale in encoded
                        ]

              
                        torch.save(
                            {
                                "window_size": None,
                                "num_windows": 1,
                                "latents": encoded,
                            },
                            recording_folder / f"block_{block_idx:03d}.pt",
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

                        batched_latents = []

                        for i in range(
                            0,
                            windows.shape[0],
                            self.batch_size,
                        ):

                            batch = windows[
                                i:i+self.batch_size
                            ]

                            encoded = self.model.encode(batch)

                            encoded = [
                                (
                                    codes.cpu(),
                                    scale.cpu() if scale is not None else None,
                                )
                                for codes, scale in encoded
                            ]

                            batched_latents.append(encoded)

                        ##########################################################
                        # Save
                        ##########################################################

                        torch.save(
                            {
                                "window_size": window_size,
                                "num_windows": windows.shape[0],
                                "valid_length": block.shape[-1],
                                "latents": batched_latents,
                            },
                            recording_folder / f"block_{block_idx:03d}.pt",
                        )

                        del windows
                        del batched_latents

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
            original_sr = metadata["original_sample_rate"]
            

            block_files = sorted(recording_folder.glob("block_*.pt"))


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

                        #print(f"  Block {block_idx+1}/{len(block_files)}")


                        saved = torch.load(block_file)
                        window_size = saved["window_size"]
                        num_windows = saved["num_windows"]
                        batched_latents = saved["latents"]

                        saved = torch.load(block_file)

                        codes = saved["latents"][0][0][0]

                        print(codes.dtype)
                        print(codes.shape)
                        print(codes.min(), codes.max())
                        print(codes.numel())
                        print(codes.element_size())
                        print("Tensor size:", codes.numel()*codes.element_size()/1024**2, "MB")



                        ###################################################
                        # ORIGINAL ENCODEC MODE
                        ###################################################

                        if window_size is None:

                            decoded = self.model.decode(
                                batched_latents
                            )

                            audio = (
                                decoded.squeeze(0)
                                .T.cpu()
                                .numpy()
                            )

                            outfile.write(audio)

                            del decoded

                        ###################################################
                        # WINDOWED / BATCHED MODE
                        ###################################################

                        else:

                            decoded_windows = []

                            for batch in batched_latents:

                                decoded = self.model.decode(batch)

                                decoded_windows.append(decoded.cpu())

                                del decoded

                            decoded_windows = torch.cat(
                                decoded_windows,
                                dim=0,
                            )

                            reconstructed = decoded_windows.reshape(
                                -1,
                                decoded_windows.shape[-1],
                            )

                            reconstructed = reconstructed[
                                :num_windows
                            ]

                            reconstructed = reconstructed.reshape(
                                -1
                            )

                            audio = reconstructed.numpy()[:saved["valid_length"]]

                            outfile.write(audio)

                            del decoded_windows
                            del reconstructed

                        gc.collect()
                print("Decode full file:", time.time() - t1)
                    

