# Compression
#######################################
class CS_1D:
  # Create a sensing matrix using DCT, the same as A=\Phi\Psi in the theory
  def csmtx_dct(self, N, idx):
      K = len(idx)
      A = np.zeros((K, N))
      for i, j in enumerate(idx):
          a = np.zeros(N)
          a[j] = 1
          A[i, :] = dct(a, norm='ortho')
      return A

  # Function to compress the 1D-signal
  def compress_1d(self, X, R, seed):
      np.random.seed(seed)
      N = len(X)
      M = int(R * N)  # Number of measurements
      idx = np.random.choice(N, M, replace=False)
      y = np.array(X)[idx]  # Compressed measurements
      return y, idx, N

  # Function to reconstruct a single frame in a audio
  def reconstruct_frame(self,y, idx, N, solver, alpha):
      A = self.csmtx_dct(N, idx)
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

  # Compress and reconstruct the signal frame by frame to speed up computation
  def compress_and_reconstruct(self, x, frame_size, overlap, compression_rate, solver, alpha=0.1, seed=42, n_jobs=-1):
      num_samples = len(x)
      hop_size = frame_size - overlap
      num_frames = (num_samples - frame_size) // hop_size + 1

      # Segment the signal with overlap
      x_frames = [x[i * hop_size:i * hop_size + frame_size] for i in range(num_frames)]

      compressed_data = []
      for frame in x_frames:
          y, idx, N = self.compress_1d(frame, compression_rate, seed)
          compressed_data.append((y, idx, N))

      # Parallel reconstruction
      reconstructed_frames = Parallel(n_jobs=n_jobs)(
          delayed(self.reconstruct_frame)(y, idx, N, solver, alpha) for y, idx, N in compressed_data
      )

      # Reconstruct full signal by overlap-add
      reconstructed_signal = np.zeros(num_samples)
      for i, frame in enumerate(reconstructed_frames):
          reconstructed_signal[i * hop_size:i * hop_size + frame_size] += frame

      return compressed_data[0], reconstructed_signal

########################################################
class Preprocess_CS_1D:

    def __init__(self, frame_size, overlap, max_threads, batch_size, compression_rate, solver, alpha):
        self.max_threads = max_threads
        self.batch_size = batch_size
        self.compression_rate =compression_rate
        self.solver = solver
        self.alpha = alpha
        self.frame_size =frame_size
        self.overlap = overlap

    def compress_and_reconstruct(self, audio, labels):

        compressed, reconstructed = cs.compress_and_reconstruct(audio, self.frame_size, self.overlap, self.compression_rate, self.solver, self.alpha, 42, -1)

        return compressed, reconstructed, labels

    def run_script(self, X_segments, Y_labels, saved_foder):

        store_compressed = []
        store_reconstructed =[]
        store_labels = []
        batch_times = []
        t_s = time.time()
        print(f'Run compression and reconstruction in parallel:\n')
        for start_index in range(0, len(X_segments), self.batch_size):

            end_index = start_index + self.batch_size
            batch_segments = X_segments[start_index:end_index]
            batch_labels = Y_labels[start_index:end_index]
            print(f"Batch index: ({start_index} -- {end_index})")

            print(f'Segments shape: {batch_segments.shape}')

            batch_start_time = time.time()
            # Create a thread pool with the desired threads.
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_threads) as executor:

                # Process in parallel the compression and reconstruction of each spectrogram in the batch.
                batch_output = list(executor.map(self.compress_and_reconstruct, batch_segments, batch_labels))

                compressed_segments = [extract[0] for extract in batch_output]
                reconstructed_segments = [extract[1] for extract in batch_output]
                labels = [extract[2] for extract in batch_output]

                batch_end_time = time.time()

                batch_time = batch_end_time - batch_start_time
                print(f"Compression and reconstruction time: {batch_time:.2f} seconds")

                # Outputs for the current batch.
                store_compressed.extend(compressed_segments)
                store_reconstructed.extend(reconstructed_segments)
                store_labels.extend(labels)

                batch_times.append(batch_time)
                print("done.\n")
                del batch_segments
                gc.collect()

        t_e = time.time()
        t_process = t_e-t_s

        # Compute the average time for batch processing.
        average_batch_time = sum(batch_times) / len(batch_times)

        # Average batch processing time.
        print("_______________________SUMARRY_________________________")
        print(f"Number of segments extracted: {len(X_segments)}")
        print(f"Average time to compress and reconstruct a batch: {average_batch_time:.2f} seconds")
        print(f'Processing time : {t_process:.2f} seconds')
        print("done.\n")


        # Create a folder to save the data
        #---------------------------------------------------------------------------------------------------------------------------
        if not os.path.exists(saved_foder):
            os.makedirs(saved_foder)



        X_comp, X_rec, Y_values = np.array(store_compressed), np.array(store_reconstructed), np.array(store_labels)
        # Ensure the arrays are compatible
        print(f"X_comp shape: {X_comp.shape}")
        print(f"X_rec shape: {X_rec.shape}")
        print(f"Y_values shape: {Y_values.shape}")

        # Store the results after verifying shapes
        pickle_file_1 = os.path.join(saved_foder, f"X_compressed.pkl")
        with open(pickle_file_1, 'wb') as file:
            pickle.dump(X_comp, file)

        pickle_file_2 = os.path.join(saved_foder, f"X_reconstructed.pkl")
        with open(pickle_file_2, 'wb') as file:
            pickle.dump(X_rec, file)

        pickle_file_3 = os.path.join(saved_foder, f"Y.pkl")
        with open(pickle_file_3, 'wb') as file:
            pickle.dump(Y_values, file)


########################################################
import sys
from pydub import AudioSegment
import os
import soundfile as sf

# gui.py
from PyQt5 import QtWidgets 
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QVBoxLayout, QPlainTextEdit, QDialog
from class_window import Ui_MainWindow

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
         

        # set the title
        #self.setWindowTitle("OK")
        self.setWindowIcon(QIcon('log_1.png')) 

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.pushButton.clicked.connect(self.load_data)
        self.ui.spinBox.setMinimum(0)
        self.ui.spinBox.setMaximum(999)
        self.ui.spinBox.setRange(0,320)
        #self.ui.comboBox.addItems(["8k", "16k", "32k", "64k", "128k", "192k", "256k", "320k"])
        self.ui.spinBox_2.setRange(0,10)
        self.ui.spinBox_3.setRange(0,10)
        self.ui.spinBox_4.setMinimum(0)
        self.ui.spinBox_4.setMaximum(999)
        self.ui.spinBox_4.setRange(0,320)
        self.ui.pushButton_3.clicked.connect(self.flac_conversion)
        self.ui.pushButton_2.clicked.connect(self.mp3_conversion)
        self.ui.pushButton_5.clicked.connect(self.aac_conversion)
        self.ui.pushButton_4.clicked.connect(self.ogg_conversion)

        # show all the widgets
        self.show()

    def load_data(self):
        self.input_folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")

        if self.input_folder:
            self.ui.plainTextEdit.clear()
            wav_files = [f for f in os.listdir(self.input_folder) if f.endswith('.wav')] or [f for f in os.listdir(self.input_folder) if f.endswith('.WAV')]
            self.ui.plainTextEdit.appendPlainText("WAV files found in the folder:")
            for wav_file in wav_files:
                self.ui.plainTextEdit.appendPlainText(wav_file)


    ##MP3
def mp3_conversion(self):
        if self.input_folder:
            #selected_bitrate = self.ui.comboBox.currentText()
            selected_bitrate = self.ui.spinBox.value()
            output_folder_name = f"output_mp3_{selected_bitrate}"
            self.output_folder = os.path.join(self.input_folder, output_folder_name)
            os.makedirs(self.output_folder, exist_ok=True)

            messages = []
            wav_files = [f for f in os.listdir(self.input_folder) if f.endswith('.wav')] or [f for f in os.listdir(self.input_folder) if f.endswith('.WAV')]

            self.wav_to_mp3(wav_files, selected_bitrate, messages)
        else:
            self.show_message_window(["Please load the data first!"])
    

    def wav_to_mp3(self, wav_files, selected_bitrate, messages):
        messages.append("Conversion started.")
        self.ui.progressBar.setValue(0)
        total_files = len(wav_files)
        for i, filename in enumerate(wav_files):
            #self.text_edit.appendPlainText(f"Conversion of {filename} started.")
            messages.append(f"Conversion of {filename} started.")
            input_wav_file = os.path.join(self.input_folder, filename)
            output_mp3_file = os.path.join(self.output_folder, os.path.splitext(filename)[0] + '.mp3')

            with sf.SoundFile(input_wav_file) as f:
                original_sample_rate = f.samplerate

            try:
                audio = AudioSegment.from_wav(input_wav_file)
                audio.export(output_mp3_file, format="mp3", bitrate=f"{selected_bitrate}k")
                #self.text_edit.appendPlainText(f"Converted {filename} to MP3 with sample rate {sample_rate_to_use}.")
                messages.append(f"{filename} converted to MP3.")
            except Exception as e:
                #self.text_edit.appendPlainText(f"Failed to convert {filename}: {e}")
                messages.append(f"Failed to convert {filename}: {e}")

            progress_percentage = int(((i + 1) / total_files) * 100)
            self.ui.progressBar.setValue(progress_percentage)

        messages.append("Conversion finished.")
        #self.text_edit.appendPlainText("Conversion finished.")
        self.show_message_window(messages)

    def flac_conversion(self):
        if self.input_folder:
            compression_level = self.ui.spinBox_2.value()
            output_folder_name = f"output_flac_{compression_level}"
            self.output_folder = os.path.join(self.input_folder, output_folder_name)
            os.makedirs(self.output_folder, exist_ok=True)

            messages = []
            wav_files = [f for f in os.listdir(self.input_folder) if f.endswith('.wav')] or [f for f in os.listdir(self.input_folder) if f.endswith('.WAV')]
                
                
            self.wav_to_flac(wav_files, compression_level, messages)
        else:
            self.show_message_window(["Please load the data first!"])

    def wav_to_flac(self, wav_files, compression_level, messages):
        messages.append("Conversion started.")
        self.ui.progressBar.setValue(0)
        total_files = len(wav_files)
        for i, filename in enumerate(wav_files):
            #self.text_edit.appendPlainText(f"Conversion of {filename} started.")
            messages.append(f"Conversion of {filename} started.")
            input_wav_file = os.path.join(self.input_folder, filename)
            flac_path = os.path.join(self.output_folder, os.path.splitext(filename)[0] + '.flac')

            with sf.SoundFile(input_wav_file) as f:
                original_sample_rate = f.samplerate

            try:

                # Load the WAV file
                audio = AudioSegment.from_wav(input_wav_file)
                audio.export(flac_path, format="flac", parameters=["-compression_level", str(compression_level)])
                messages.append(f"Converted {filename} to FLAC with sample rate { original_sample_rate}.")

            except Exception as e:
                #self.text_edit.appendPlainText(f"Failed to convert {filename}: {e}")
                messages.append(f"Failed to convert {filename}: {e}")

            progress_percentage = int(((i + 1) / total_files) * 100)
            self.ui.progressBar.setValue(progress_percentage)

        messages.append("Conversion finished.")
        #self.text_edit.appendPlainText("Conversion finished.")
        self.show_message_window(messages)


        ##OGG
    def ogg_conversion(self):
        if self.input_folder:
            quality_level = self.ui.spinBox_3.value()
            output_folder_name = f"output_ogg_{quality_level}"
            self.output_folder = os.path.join(self.input_folder, output_folder_name)
            os.makedirs(self.output_folder, exist_ok=True)

            messages = []
            wav_files = [f for f in os.listdir(self.input_folder) if f.endswith('.wav')] or [f for f in os.listdir(self.input_folder) if f.endswith('.WAV')]
                
                
            self.wav_to_ogg(wav_files, quality_level, messages)
        else:
            self.show_message_window(["Please load the data first!"])

    def wav_to_ogg(self, wav_files, quality_level, messages):
        messages.append("Conversion started.")
        self.ui.progressBar.setValue(0)
        total_files = len(wav_files)
        for i, filename in enumerate(wav_files):
            #self.text_edit.appendPlainText(f"Conversion of {filename} started.")
            messages.append(f"Conversion of {filename} started.")
            input_wav_file = os.path.join(self.input_folder, filename)
            ogg_path = os.path.join(self.output_folder, os.path.splitext(filename)[0] + '.ogg')

            with sf.SoundFile(input_wav_file) as f:
                original_sample_rate = f.samplerate

            try:

                # Load the WAV file
                audio = AudioSegment.from_wav(input_wav_file)
                # Set the level for ogg
                audio.export(ogg_path, format="ogg", codec="libvorbis", parameters=["-qscale:a", str(quality_level)])
                #audio.export(flac_path, format="flac", parameters=["-compression_level", str(compression_level)])
                messages.append(f"Converted {filename} to OGG with sample rate { original_sample_rate}.")


            except Exception as e:
                #self.text_edit.appendPlainText(f"Failed to convert {filename}: {e}")
                messages.append(f"Failed to convert {filename}: {e}")

            progress_percentage = int(((i + 1) / total_files) * 100)
            self.ui.progressBar.setValue(progress_percentage)

        messages.append("Conversion finished.")
        #self.text_edit.appendPlainText("Conversion finished.")
        self.show_message_window(messages)

    ## AAC
    def aac_conversion(self):
        if self.input_folder:
            #bitrate = self.ui.comboBox_2.currentText()
            bitrate = self.ui.spinBox_4.value()
            output_folder_name = f"output_aac_{bitrate}"
            self.output_folder = os.path.join(self.input_folder, output_folder_name)
            os.makedirs(self.output_folder, exist_ok=True)

            messages = []
            wav_files = [f for f in os.listdir(self.input_folder) if f.endswith('.wav')] or [f for f in os.listdir(self.input_folder) if f.endswith('.WAV')]
                
                
            self.wav_to_aac(wav_files, bitrate, messages)
        else:
            self.show_message_window(["Please load the data first!"])

    def wav_to_aac(self, wav_files, bitrate, messages):
        messages.append("Conversion started.")
        self.ui.progressBar.setValue(0)
        total_files = len(wav_files)
        for i, filename in enumerate(wav_files):
            #self.text_edit.appendPlainText(f"Conversion of {filename} started.")
            messages.append(f"Conversion of {filename} started.")
            input_wav_file = os.path.join(self.input_folder, filename)
            aac_path = os.path.join(self.output_folder, os.path.splitext(filename)[0] + '.aac')

            with sf.SoundFile(input_wav_file) as f:
                original_sample_rate = f.samplerate

            try:

                # Load the WAV file
                audio = AudioSegment.from_wav(input_wav_file)
                # Set the level for aac
                audio.export(aac_path, codec="aac", format ="adts", bitrate=f"{bitrate}k")
                messages.append(f"{filename} converted to AAC.")


            except Exception as e:
                #self.text_edit.appendPlainText(f"Failed to convert {filename}: {e}")
                messages.append(f"Failed to convert {filename}: {e}")

            progress_percentage = int(((i + 1) / total_files) * 100)
            self.ui.progressBar.setValue(progress_percentage)

        messages.append("Conversion finished.")
        #self.text_edit.appendPlainText("Conversion finished.")
        self.show_message_window(messages)

    def show_message_window(self, messages):
        self.msg_window = MessageWindow(messages)
        self.msg_window.show()
