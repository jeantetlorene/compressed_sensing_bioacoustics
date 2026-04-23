from AnnotationReader import AnnotationReader
from preprocess_old import Preprocessing 
from settings import Config
from compress import FlacCompression
from glob import glob
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pickle
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


class Preprocess:
    def __init__(self, files_path, config, preprocessing=True, data_augmentation=False):
        """
        Initializes the Preprocess class for dataset creation.

        Args:
            files_path (str or Path): Path to the dataset file (train/val/test).
            config (Config): Configuration object containing all settings.
            preprocessing (bool): Whether to apply preprocessing steps (filtering, downsampling).
            data_augmentation (bool): Whether to apply data augmentation.
        """
        self.files_path = Path(files_path)
        self.preprocessing = preprocessing
        self.data_augmentation = data_augmentation
        self.one_hot_encode = one_hot_encode
        self.config = config

        # Initialize preprocessing instance
        self.preprocess = Preprocessing(
            **self.config.preprocessing.dict(),
            species_folder=self.config.data.species_folder,
            positive_class=self.config.data.positive_class,
            negative_class=self.config.data.negative_class,
        )

        # Initialize storage for dataset
        self.X_calls = []
        self.Y_calls = []
        self.sample_rate = None

    @staticmethod
    def _one_hot_encode(Y, positive_class, negative_class):
        """Applies one-hot encoding to labels."""
        Y = np.array(Y).reshape(-1, 1)  # Ensure Y is a numpy array and reshape it

        encoder = OneHotEncoder(categories=[[negative_class, positive_class]], sparse=False)
        encoder.fit(Y)

        Y_encoded = encoder.transform(Y)
        return Y_encoded
    
    def load_files(self):
        """Loads the file list from the dataset file."""
        try:
            files = pd.read_csv(self.files_path, header=None)
            return files.values
        except Exception:
            raise ValueError(f"Error loading filenames from {self.files_path}. Check if the file is not empty.")

    def process_file(self, file_name_no_extension):
        """Processes an individual WAV file and extracts features and annotations."""
        print(f"Processing: {file_name_no_extension}")

        reader = AnnotationReader(
            self.preprocess.annotations_path,
            file_name_no_extension,
            self.preprocess.annotation_extension,
            self.preprocess.audio_extension,
            self.preprocess.positive_class
        )

        # Check if the .wav file exists before processing
        wav_file_path = Path(self.preprocess.audio_path, file_name_no_extension + self.preprocess.audio_extension)
        if str(wav_file_path) in glob(str(self.preprocess.audio_path / f"*{self.preprocess.audio_extension}")):
            print(f"Found file {file_name_no_extension}")

        # Read audio file
        audio_amps, original_sample_rate = self.preprocess.read_audio_file(str(wav_file_path))
        print(f"Sampling rate: {original_sample_rate}")

        ###########################################
        # compress to FLAC
        flac_compression = FlacCompression(str(Path(self.preprocess.audio_path,file_name_no_extension + self.preprocess.audio_extension)))
        wav_data, wav_sample_rate, flac_data = flac_compression.compress()
        ###################################################
        
        # Preprocessing steps
        if self.preprocessing:
            print("Filtering...")
            filtered = self.preprocess.butter_lowpass_filter(
                audio_amps, self.preprocess.lowpass_cutoff, self.preprocess.nyquist_rate
            )

            print("Downsampling...")
            amplitudes, sample_rate = self.preprocess.downsample_file(
                filtered, original_sample_rate, self.preprocess.downsample_rate
            )
            del filtered
        else:
            amplitudes = audio_amps
            sample_rate = original_sample_rate

        del audio_amps  # Free memory

        # Read annotation data
        print("Reading annotations...")
        df, audio_file_name = reader.get_annotation_information()

        for _, row in df.iterrows():
            start_seconds = int(round(row["Start"]))
            end_seconds = int(round(row["End"]))
            label = row["Label"]
            annotation_duration_seconds = end_seconds - start_seconds

            # Extract audio segments and corresponding labels
            X_data, y_data = self.preprocess.getXY(
                amplitudes,
                sample_rate,
                start_seconds,
                annotation_duration_seconds,
                label,
                file_name_no_extension,
                verbose=False
            )

            self.X_calls.extend(X_data)
            self.Y_calls.extend(y_data)

        self.sample_rate = sample_rate

    def create_dataset(self):
        """Main method to create the dataset."""
        files = self.load_files()

        for file in files:
            file_name_no_extension = file[0]
            self.process_file(file_name_no_extension)

        if self.data_augmentation:
            print("Applying Data Augmentation...")
            print(np.unique(self.Y_calls, return_counts=True))
            self.X_calls, self.Y_calls = self.preprocess.augment_dataset(self.X_calls, self.Y_calls)
            print("After Augmentation:")
            print(np.unique(self.Y_calls, return_counts=True))

        # Apply One-Hot Encoding if enabled
        if self.one_hot_encode:
            print("Applying One-Hot Encoding to Labels...")
            self.Y_calls = self._one_hot_encode(
                self.Y_calls, 
                self.preprocess.positive_class, 
                self.preprocess.negative_class
            )
            
        return self.X_calls, self.Y_calls, self.sample_rate
