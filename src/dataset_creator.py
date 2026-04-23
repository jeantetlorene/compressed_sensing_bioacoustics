import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path
import glob

# Assuming AnnotationReader, FlacCompression, and preprocess are imported
#from preprocess import preprocess  # Adjust import based on your file structure
from annotation_reader import AnnotationReader  # Adjust import based on your file structure
from flac_compression import FlacCompression  # Adjust import based on your file structure

class DatasetCreator:
    def __init__(self, audio_path, species_folder, annotation_extension=".csv", audio_extension=".wav", positive_class=None, lowpass_cutoff=1000, downsample_rate=16000):
        self.audio_path = audio_path
        self.species_folder = species_folder
        self.annotation_extension = annotation_extension
        self.audio_extension = audio_extension
        self.positive_class = positive_class
        self.lowpass_cutoff = lowpass_cutoff
        self.downsample_rate = downsample_rate
        self.nyquist_rate = None  # Set this if necessary

    def create_dataset(self, files_path, dataset_type="test", preprocessing=True, data_augmentation=False):
        # Read all names of the files
        try:
            files = pd.read_csv(files_path, header=None)
        except Exception:
            raise ValueError(f"Error loading filenames from {files_path}. Check if File is not empty.")

        # Initialise lists to store the X and Y values
        X_calls = []
        Y_calls = []

        for file in files.values:
            file_name_no_extension = file[0]

            print("Processing:", file_name_no_extension)

            reader = AnnotationReader(self.species_folder, file_name_no_extension, self.annotation_extension, self.audio_extension, self.positive_class)

            # Check if the .wav file exists before processing
            if str(Path(self.audio_path, file_name_no_extension + self.audio_extension)) in glob.glob(str(Path(self.audio_path) / f"*{self.audio_extension}")):
                print(f"Found file {file_name_no_extension}")

            # Read audio file
            audio_amps, original_sample_rate = preprocess.read_audio_file(
                str(Path(self.audio_path, file_name_no_extension + self.audio_extension)))

            print("sampling rate:", original_sample_rate)

            ###################################################
            # compress to FLAC
            flac_compression = FlacCompression(str(Path(self.audio_path, file_name_no_extension + self.audio_extension)))
            wav_data, wav_sample_rate, flac_data = flac_compression.compress()
            ###################################################
            if preprocessing:
                # preprocessing
                print("Filtering...")
                # Low pass filter
                filtered = preprocess.butter_lowpass_filter(audio_amps, self.lowpass_cutoff, self.nyquist_rate)
                # Downsample
                print("Downsampling...")
                amplitudes, sample_rate = preprocess.downsample_file(filtered, original_sample_rate, self.downsample_rate)
                del filtered
            else:
                amplitudes = audio_amps
                sample_rate = original_sample_rate

            del audio_amps

            # get annotation
            df, audio_file_name = reader.get_annotation_information()

            print("Reading annotations...")
            for index, row in df.iterrows():
                start_seconds = int(round(row["Start"]))
                end_seconds = int(round(row["End"]))
                label = row["Label"]
                annotation_duration_seconds = end_seconds - start_seconds

                # Extract augmented audio segments and corresponding binary labels
                X_data, y_data = preprocess.getXY(
                    amplitudes,
                    sample_rate,
                    start_seconds,
                    annotation_duration_seconds,
                    label,
                    file_name_no_extension,
                    verbose=False,
                )

                # Append the segments and labels
                X_calls.extend(X_data)
                Y_calls.extend(y_data)

        if data_augmentation:
            # Augment dataset to get a balanced dataset
            print("Augmentation")
            print(np.unique(Y_calls, return_counts=True))
            X_calls, Y_calls = preprocess.augment_dataset(X_calls, Y_calls)
            print("After Augmentation")
            print(np.unique(Y_calls, return_counts=True))

        return X_calls, Y_calls, sample_rate

    def _one_hot_encode(self, Y):
        """
        One-hot encode the labels.
        """
        # Check if the encoder is fitted
        # Reshape the labels
        Y = Y.reshape(-1, 1)

        encoder = OneHotEncoder(categories=[[preprocess.negative_class, preprocess.positive_class]])
        encoder.fit(Y)

        # Encode the labels
        Y = encoder.transform(Y)

        # Convert to numpy array
        Y = Y.toarray()
        return Y
