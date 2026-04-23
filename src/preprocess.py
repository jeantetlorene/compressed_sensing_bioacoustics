from glob import glob
import os
import re
import numpy as np
import random
import librosa
from scipy import signal
from random import randint
import pickle
import pandas as pd
from pathlib import Path
#import noisereduce as nr
from sklearn.preprocessing import OneHotEncoder


from AnnotationReader import *


class Preprocessing:
    def __init__(
        self,
        species_folder,
        sample_rate,
        lowpass_cutoff,
        downsample_rate,
        nyquist_rate,
        segment_duration,
        positive_class,
        negative_class,
        nb_negative_class,
        n_fft,
        hop_length,
        n_mels,
        f_min,
        f_max,
        annotation_extension,
        audio_extension,

        
    ) -> None:
        self.species_folder = species_folder
        self.original_sample_rate=sample_rate
        self.lowpass_cutoff = lowpass_cutoff
        self.downsample_rate = downsample_rate
        self.nyquist_rate = nyquist_rate
        self.segment_duration = segment_duration
        self.positive_class = positive_class
        self.negative_class = negative_class
        self.nb_negative_class = nb_negative_class
        self.audio_path = Path(self.species_folder, "Audio")
        
        self.annotations_path = Path(self.species_folder, "Annotations")
        self.saved_data_path = Path(self.species_folder, "Saved_Data")
        self.training_files = Path(self.species_folder, "DataFiles", "TrainingFiles.txt")
        
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.annotation_extension = annotation_extension
        self.audio_extension = audio_extension
        self.n_fft = n_fft 
        self.hop_length = hop_length
     
        
    def update_audio_path(self, audio_path):
        self.audio_path = Path(self.species_folder, audio_path)

    def read_audio_file(self, file_name, method_compression, parameter_compression):
        """
        file_name: string, name of file including extension, e.g. "audio1.wav"

        """
        print(file_name)
        # Get the path to the file
        if method_compression==None:
            audio_path=Path(self.audio_path, file_name+self.audio_extension)
        
        elif method_compression=="cs": 
            self.compressed_audio_path = Path(self.species_folder, "Compressed_Audio", f"cs_reconstructed_{parameter_compression}")
            print(self.compressed_audio_path)
            # Find file that matches pattern "{file_name}_*_reconstructed.pkl"
            matches =list(glob(f"{self.compressed_audio_path}/{file_name}_*.npy"))
            if not matches:
                raise FileNotFoundError(f"No reconstructed file found for {file_name}")
            if len(matches) > 1:
                raise ValueError(f"Multiple reconstructed files found for {file_name}: {matches}")

            audio_path = matches[0]
        else: 
            self.compressed_audio_path = Path(self.species_folder, "Compressed_Audio", f"{method_compression}_{parameter_compression}")
            audio_path=Path(self.compressed_audio_path)/ f"{file_name}_{method_compression}_{parameter_compression}.{method_compression}"
        
        # Read the amplitudes and sample rate     
        if method_compression=="cs":
            audio_amps=np.load(audio_path) 
            audio_sample_rate= self.original_sample_rate  
        else :   
            audio_amps, audio_sample_rate = librosa.load(audio_path, sr=None)
        

        return audio_amps, audio_sample_rate

    def butter_lowpass(self, cutoff, nyq_freq, order=4):
        normal_cutoff = float(cutoff) / nyq_freq
        b, a = signal.butter(order, normal_cutoff, btype="lowpass")
        return b, a

    def butter_lowpass_filter(self, data, cutoff_freq, nyq_freq, order=4):
        # Source: https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform
        b, a = self.butter_lowpass(cutoff_freq, nyq_freq, order=order)
        y = signal.filtfilt(b, a, data)
        return y

    def downsample_file(self, amplitudes, original_sr, new_sample_rate):
        """
        Downsample an audio file to a given new sample rate.
        amplitudes:
        original_sr:
        new_sample_rate:

        """
        """
        return librosa.resample(amplitudes, 
                                original_sr, 
                                new_sample_rate, 
                                res_type='kaiser_fast'), new_sample_rate
        """
        return (
            librosa.resample(
                amplitudes,
                orig_sr=original_sr,
                target_sr=new_sample_rate,
                res_type="kaiser_fast",
            ),
            new_sample_rate,
        )

    def convert_single_to_image(self, audio, sample_rate):
        """
        Convert amplitude values into a mel-spectrogram.
        """
        """
        S = librosa.feature.melspectrogram(audio, n_fft=self.n_fft,hop_length=self.hop_length, 
                                            n_mels=self.n_mels, fmin=self.f_min, fmax=self.f_max)
        """

        S = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
        )
        image = librosa.core.power_to_db(S)
        image_np = np.asmatrix(image)
        image_np_scaled_temp = image_np - np.min(image_np)
        image_np_scaled = image_np_scaled_temp / np.max(image_np_scaled_temp)
        mean = image.flatten().mean()
        std = image.flatten().std()
        eps = 1e-8
        spec_norm = (image - mean) / (std + eps)
        spec_min, spec_max = spec_norm.min(), spec_norm.max()
        spec_scaled = (spec_norm - spec_min) / (spec_max - spec_min)
        S1 = spec_scaled

        return S1

    def convert_all_to_image(self, segments, sample_rate):
        """
        Convert a number of segments into their corresponding spectrograms.
        """
        spectrograms = []
        for segment in segments:
            spectrograms.append(self.convert_single_to_image(segment, sample_rate))

        return np.array(spectrograms)

    def add_extra_dim(self, spectrograms):
        """
        Add an extra dimension to the data so that it matches
        the input requirement of Tensorflow.
        """
        spectrograms = np.reshape(
            spectrograms,
            (spectrograms.shape[0], spectrograms.shape[1], spectrograms.shape[2], 1),
        )
        return spectrograms

    def getXY(
        self,
        audio_amplitudes,
        sampling_rate,
        start_sec,
        annotation_duration_seconds,
        label,
        file_name_no_extension,
        verbose,
    ):
        """
        Extract a number of segments based on the user-annotations.
        If possible, a number of segments are extracted provided
        that the duration of the annotation is long enough. The segments
        are extracted by shifting by 1 second in time to the right.
        Each segment is then augmented a number of times based on a pre-defined
        user value.
        """

        if verbose == True:
            print("start_sec", start_sec)
            print("annotation_duration_seconds", annotation_duration_seconds)
            print("self.segment_duration ", self.segment_duration)

        X_segments = []
        Y_labels = []

        # Calculate how many segments can be extracted based on the duration of
        # the annotated duration. If the annotated duration is too short then
        # simply extract one segment. If the annotated duration is long enough
        # then multiple segments can be extracted.
        if int(annotation_duration_seconds) - self.segment_duration < 0:
            segments_to_extract = 1
        else:
            segments_to_extract = (
                int(annotation_duration_seconds) - self.segment_duration + 1
            )

        if verbose:
            print("segments_to_extract", segments_to_extract)

        if label in self.negative_class:
            if segments_to_extract > self.nb_negative_class:
                segments_to_extract = self.nb_negative_class

        for i in range(0, segments_to_extract):
            if verbose:
                print("Semgnet {} of {}".format(i, segments_to_extract - 1))
                print("*******************")

            # Set the correct location to start with.
            # The correct start is with respect to the location in time
            # in the audio file start+i*sample_rate
            start_data_observation = int(start_sec * sampling_rate) + i * (sampling_rate)
            # The end location is based off the start
            end_data_observation = start_data_observation + (
                sampling_rate * self.segment_duration
            )

            # This case occurs when something is annotated towards the end of a file
            # and can result in a segment which is too short.
            if end_data_observation > len(audio_amplitudes):
                continue

            # Extract the segment of audio
            X_audio = audio_amplitudes[start_data_observation:end_data_observation]


            if verbose == True:
                print("start frame", start_data_observation)
                print("end frame", end_data_observation)

            # Extend the augmented segments and labels
            X_segments.append(X_audio)
            Y_labels.append(label)

        return X_segments, Y_labels

    def save_data_to_pickle(self, X, Y):
        """
        Save all of the spectrograms to a pickle file.

        """
        outfile = open(Path(self.saved_data_path, "X.pkl"), "wb")
        pickle.dump(X, outfile, protocol=4)
        outfile.close()

        outfile = open(Path(self.saved_data_path, "Y.pkl"), "wb")
        pickle.dump(Y, outfile, protocol=4)
        outfile.close()

    def load_data_from_pickle(self):
        """
        Load all of the spectrograms from a pickle file

        """
        infile = open(Path(self.saved_data_path, "X.pkl"), "rb")
        X = pickle.load(infile)
        infile.close()

        infile = open(Path(self.saved_data_path, "Y.pkl"), "rb")
        Y = pickle.load(infile)
        infile.close()

        return X, Y

    def time_shifting(self, X, index):
        """
        Augment a segment of amplitude values by applying a time shift.

        Args:
            X (ndarray): Array of amplitude values.
            X_meta (ndarray): Array of corresponding metadata.
            index (list): List of indices of the files to choose from.

        Returns:
            tuple: Augmented segment and its metadata.
        """
        # Convert index to list
        index = list(index)
        # Randomly select an index from the given index list
        idx_pickup = random.sample(index, 1)

        # Retrieve the segment and metadata corresponding to the selected index
        segment = X[idx_pickup][0]

        # Randomly select time into the segments
        random_time_point_segment = randint(1, len(segment) - 1)

        # Apply time shift to the segment
        segment = self.time_shift(segment, random_time_point_segment)

        return segment

    def time_shift(self, audio, shift):
        """
        Shift amplitude values to the right by a random value.

        The amplitude values are wrapped back to the left side of the waveform.

        Args:
            audio (ndarray): Array of amplitude values representing the audio waveform.
            shift (int): Amount of shift to apply to the waveform.

        Returns:
            ndarray: Augmented waveform with the shifted amplitude values.
        """

        augmented = np.zeros(len(audio))
        augmented[0:shift] = audio[-shift:]
        augmented[shift:] = audio[:-shift]

        return augmented

    def combining(self, X, index, index_negative_class):
        """
        Combine segments to create an augmented segment.

        Randomly selects two segments from the given indices and blends them to create a new segment.
        The blending weights are set to 0.6 and 0.4.

        Args:
                X (ndarray): Input data containing segments.
                X_meta (ndarray): Metadata associated with the input data.
                index (list): List of indices of the files to choose from.

        Returns:
                tuple: Combined segment and its associated metadata.
        """
        # Convert index to list
        index=list(index)
        index_negative_class=list(index_negative_class)
        
        # Randomly select an index from the given index list
        idx_pickup=random.sample(index, 1)
            
        # Randomly select another file to combine with
        idx_combining=random.sample(index_negative_class, 1)
        
        # combine the two files with different weights
        segment=self.blend(X[idx_pickup][0], X[idx_combining][0], 0.8, 0.2)
             
        return segment


    def blend(self, audio_1, audio_2, w_1, w_2):
        """
        Blend two audio segments together using given weights.

        Takes two audio segments and blends them together using the provided weights.
        The blending weights determine the contribution of each segment in the resulting blended segment.

        Args:
            audio_1 (ndarray): First audio segment.
            audio_2 (ndarray): Second audio segment.
            w_1 (float): Weight for the first audio segment.
            w_2 (float): Weight for the second audio segment.

        Returns:
            ndarray: Blended audio segment.
        """
            
            
        augmented = w_1 * audio_1 + w_2 * audio_2
        return augmented
        
    def add_noise_gaussian(self, X, index):
        """
        Add Gaussian noise to an audio segment.

        Randomly selects an audio segment from the given indices and adds Gaussian noise to it.
        The noise is generated using a mean of 0 and a standard deviation of 0.009.

        Args:
                X (ndarray): Input data containing segments.
                X_meta (ndarray): Metadata associated with the input data.
                index (list): List of indices of the files to choose from.

        Returns:
                tuple: Segment with added Gaussian noise and its associated metadata.
        """
        # Convert index to list
        index=list(index)
           
        # Randomly select an index from the given index list
        idx_pickup=random.sample(index, 1)
        
        # Retrieve the segment corresponding to the selected index
        segment=X[idx_pickup][0]
       
        
        # Add Gaussian noise to the segment
        segment=segment+ 0.0009*np.random.normal(0,1,len(segment))
            

        return segment

    

    def augment_dataset(self, X, Y):
        
        
        X = np.asarray(X)
        
        label_to_augment = np.argmin(np.unique(np.asarray(Y), return_counts=True)[1])
        label_second_class=np.argmax(np.unique(np.asarray(Y), return_counts=True)[1])
        difference = np.max(np.unique(np.asarray(Y), return_counts=True)[1]) - np.min(
            np.unique(np.asarray(Y), return_counts=True)[1]
        )

        
        index_to_augment = np.where(np.asarray(Y)== np.unique(np.asarray(Y), return_counts=True)[0][label_to_augment])[0]
        index_second_class = np.where(np.asarray(Y)== np.unique(np.asarray(Y), return_counts=True)[0][label_second_class])[0]
        index_negative_class=np.where(np.asarray(Y)== self.negative_class)[0]
        
        
        X_augmented = []
        Y_augmented = []

        if difference > 2* np.min(np.unique(np.asarray(Y), return_counts=True)[1]):
            ## if difference too important : reduce the number of segments in the majority class 
            #1 - reduce number of segments second class
            number_to_select = 3* np.min(np.unique(np.asarray(Y), return_counts=True)[1])
            index_to_keep=np.array(random.sample(list(index_second_class), number_to_select))

            X_augmented.extend(X[index_to_keep])
            Y_augmented=[np.unique(np.asarray(Y), return_counts=True)[0][label_second_class]] * number_to_select

            #2 - augment number of minority class 
            nb_to_add= np.min(np.unique(np.asarray(Y), return_counts=True)[1]) 
            nb_to_augm_per_method=(nb_to_add//3)+1

            #add the existing examples
            X_augmented.extend(X[index_to_augment])
            Y_augmented.extend([np.unique(np.asarray(Y), return_counts=True)[0][label_to_augment]] * len(index_to_augment))

            for j in range(0,nb_to_augm_per_method ):
                X_augmented.append(self.time_shifting(X, index_to_augment))
                X_augmented.append(self.add_noise_gaussian(X, index_to_augment))
                X_augmented.append(self.combining(X, index_to_augment,index_negative_class))
                Y_augmented.extend([np.unique(np.asarray(Y), return_counts=True)[0][label_to_augment]] * 3)

        
        else : 
            #just augment the minority class to have same number than majority class
            X_augmented.extend(X)
            Y_augmented.extend(Y)
            nb_to_augm_per_method=(difference//3)+1
            for i in range(0,nb_to_augm_per_method):
                X_augmented.append(self.time_shifting(X, index_to_augment))
                X_augmented.append(self.add_noise_gaussian(X, index_to_augment))
                X_augmented.append(self.combining(X, index_to_augment,index_negative_class))
                Y_augmented.extend([np.unique(np.asarray(Y), return_counts=True)[0][label_to_augment]] * 3)


        return X_augmented, Y_augmented

    # lorene new create dataset function
    def create_dataset(self, dataset="train", method_compression=None, parameter_compression=None, preprocessing=True, data_augmentation=False, noise_reduction=False):
        # Read all names of the files
        files_path=Path(self.species_folder,"DataFiles", dataset+".txt")
        try:
            files = pd.read_csv(files_path, header=None)
        except Exception:
            raise ValueError(
                f"Error loading filenames from {files_path}. Check if File is not empty."
                )
        
        # Initialise lists to store the X and Y values
        X_calls = []
        Y_calls = []
    
        for file in files.values:
            file_name_no_extension = file[0]
    
            print("Processing:", file_name_no_extension) 
        
            reader = AnnotationReader(self.species_folder, file_name_no_extension, self.annotation_extension, self.audio_extension, self.positive_class
                )
        
    
            # Read audio file
            if self.audio_extension==".npy":
                original_sample_rate = self.original_sample_rate
                #Construct the folder path
                folder_path = Path(self.species_folder) / "Compressed_Audio" / f"{method_compression}_reconstructed_{parameter_compression}"

                # Search for the file using glob
                matching_files = list(folder_path.glob(f"{file_name_no_extension}_*.npy"))

                # Filter to select the one ending with the exact suffix
                pattern = re.compile(re.escape(file_name_no_extension) + r"(_\d+)?_reconstructed\.npy$")

                # Filter files with the correct ending
                filtered_files = [f for f in matching_files if pattern.fullmatch(f.name)]

                if filtered_files:
                    file_path = filtered_files[0]
                    audio_amps = np.load(file_path)
                else:
                    raise FileNotFoundError(f"No file found for pattern: {file_name_no_extension}_*.npy in {folder_path}")
            else : 
                audio_amps, original_sample_rate = self.read_audio_file(file_name_no_extension, method_compression, parameter_compression) 
                        
    
            #print ("sampling rate : ", original_sample_rate ) 
    

            ###################################################
            

            if preprocessing==True: 
                #preprocessing 
                #print("Filtering...") 
                # Low pass filter
                filtered = self.butter_lowpass_filter(
                            audio_amps, self.lowpass_cutoff, self.nyquist_rate)
                # Downsample            
                #print("Downsampling...") 
                amplitudes, sample_rate = self.downsample_file(
                            filtered, original_sample_rate, self.downsample_rate
                        )
                del filtered
            else : 
                amplitudes=audio_amps
                sample_rate= original_sample_rate
            
            del audio_amps

            if noise_reduction==True: 
                print("Noise reduction")
                amplitudes = nr.reduce_noise(y=amplitudes, sr=sample_rate, prop_decrease =0.80,time_mask_smooth_ms=100 )

            
            
            #get annotation
            df, _ = reader.get_annotation_information() 
    
            #print("Reading annotations...") 
            for _, row in df.iterrows():
                start_seconds = row["Start"]
                end_seconds = row["End"]
                label = row["Label"]
                annotation_duration_seconds = end_seconds - start_seconds
    
                # Extract augmented audio segments and corresponding binary labels
                X_data, y_data = self.getXY(
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
            
        print("Nb of labels ", np.unique(Y_calls, return_counts=True))   
        
        if data_augmentation : 
            # Augment dataset to get a balance dataset
            print("Augmentation ")
            print(np.unique(Y_calls, return_counts=True))
            X_calls, Y_calls = self.augment_dataset(X_calls, Y_calls)
            print("After Augmentation ")
            print(np.unique(Y_calls, return_counts=True))
            
        X_calls = self.convert_all_to_image(X_calls, sample_rate)

        # Convert to numpy arrays
        X_calls, Y_calls = np.asarray(X_calls), np.asarray(Y_calls)

        return X_calls, Y_calls
    

    def _shuffle_files_names(self, train_size=0.8, test_size=0.1, validation_size=0.1):
        # Get all file names in Audio folder
        path = Path(self.species_folder, "Audio", f"*{self.audio_extension}")
        files = glob(str(path))

        if len(files) == 0:
            raise Exception(
                f"No audio files found in {self.species_folder}/Audio.\
                Please check the audio_extension setting in the settings file."
            )
        # Shuffle the files
        np.random.shuffle(files)

        train_samples = int(np.floor(len(files) * train_size))
        test_samples = int(np.floor(len(files) * test_size))

        # Split the files into train, test, validation
        train_split = train_samples
        test_split = test_samples

        train_files = files[:train_split]
        test_files = files[train_split : train_split + test_split]
        # Use the rest for validation
        validation_files = files[train_split + test_split :]

        # Only get the file names
        train_files = [os.path.basename(file) for file in train_files]
        test_files = [os.path.basename(file) for file in test_files]
        validation_files = [os.path.basename(file) for file in validation_files]

        # Remove the file extension
        train_files = [os.path.splitext(file)[0] for file in train_files]
        test_files = [os.path.splitext(file)[0] for file in test_files]
        validation_files = [os.path.splitext(file)[0] for file in validation_files]

        # Create the folders
        os.makedirs(Path(self.species_folder, "DataFiles"), exist_ok=True)

        # Save the files as .txt
        with open(Path(self.species_folder, "DataFiles", "train.txt"), "w") as f:
            f.write("\n".join(train_files))
        with open(os.path.join(self.species_folder, "DataFiles", "test.txt"), "w") as f:
            f.write("\n".join(test_files))

        with open(Path(self.species_folder, "DataFiles", "validation.txt"), "w") as f:
            f.write("\n".join(validation_files))

    def check_distribution(self, Y):
        unique, counts = np.unique(Y, return_counts=True)
        original_distribution = dict(zip(unique, counts))
        return original_distribution

    def _one_hot_encode(self, Y):
        # Check if the encoder is fitted
        # Reshape the labels
        Y = Y.reshape(-1, 1)
        print(self.negative_class)
        print(self.positive_class)
        encoder = OneHotEncoder(categories=[[self.negative_class, self.positive_class]])
        encoder.fit(Y)
        # Encode the labels
        Y = encoder.transform(Y)
        # Convert to numpy array
        Y = Y.toarray()
        return Y
    
