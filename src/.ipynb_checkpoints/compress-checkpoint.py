import soundfile as sf
import io
from pydub import AudioSegment
from pathlib import Path
import librosa
import subprocess
import os
import time

class Compression:
    def __init__(self, input_wav):
        self.input_wav = input_wav
        self.wav_data, self.wav_sample_rate = librosa.load(input_wav)

    # Base method to be overridden by subclasses
    def compress(self, output_file=None):
        raise NotImplementedError("Subclasses must implement the 'compress' method.")



#########################################
class FlacCompression(Compression):
    def __init__(self, input_wav):
        super().__init__(input_wav)
        self.audio_segment = AudioSegment.from_file(input_wav)

    def compress(self, compressed_audio_path, file_name_no_extension, compression_level=8):
        output_file = Path(compressed_audio_path) / f"{file_name_no_extension}.flac"
        print('start file compression')
        start_time = time.time()
        self.audio_segment.export(out_f=output_file, format="flac", parameters=["-compression_level", str(compression_level)])
        print('end file compressionn')
        end_time = time.time()  # End time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"Compression time: {elapsed_time:.6f} seconds")
        compressed_audio_amps, compressed_sample_rate = librosa.load(str(output_file), dtype='float32')
        return compressed_audio_amps, compressed_sample_rate 
    
    def compress_without_saving(self, compressed_audio_path, file_name_no_extension, compression_level=8):
        buffer = io.BytesIO()
        self.audio_segment.export(out_f=buffer, format="flac", parameters=["-compression_level", str(compression_level)])
        buffer.seek(0)
        compressed_audio_amps, compressed_sample_rate = sf.read(buffer, dtype='float32')
        return compressed_audio_amps, compressed_sample_rate 
    
    def compress_without_bitrate_specification(self, compressed_audio_path, file_name_no_extension):
        """Compress the WAV file and save it as a FLAC file."""
        print(self.wav_data.shape)
        sf.write(str(Path(compressed_audio_path)) + "/" + str(file_name_no_extension) + '.flac', self.wav_data, self.wav_sample_rate, format='FLAC')
        compressed_audio_amps, compressed_sample_rate = librosa.load(str(Path(compressed_audio_path)) + "/" + str(file_name_no_extension) + '.flac', dtype='float32')
        return compressed_audio_amps, compressed_sample_rate

    def compress_load_flac_from_file(self, compressed_audio_path, file_name_no_extension):
        # Build the output file path.
        output_file = Path(compressed_audio_path) / f"{file_name_no_extension}.flac"
        compressed_audio_amps, compressed_sample_rate = librosa.load(str(output_file), dtype='float32')
        return compressed_audio_amps, compressed_sample_rate


###########################################
class MP3Compression(Compression):
    def __init__(self, input_wav):
        super().__init__(input_wav)
        # Load the WAV file as an AudioSegment.
        self.audio_segment = AudioSegment.from_file(input_wav)

    def compress(self, compressed_audio_path, file_name_no_extension, bitrate="192k"): # 320k / 192k / 32k (low) - 2 extremes
        output_file = Path(compressed_audio_path) / f"{file_name_no_extension}.mp3"
        ############
        print('Start file compression')
        start_time = time.time()
        self.audio_segment.export(out_f=output_file, format="mp3", parameters=["-b:a", bitrate])
        print('End file compression')
        end_time = time.time()  # End time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"Compression time: {elapsed_time:.6f} seconds")
        ##########
        compressed_audio_amps, compressed_sample_rate = librosa.load(str(output_file), dtype='float32')
        return compressed_audio_amps, compressed_sample_rate

    def compress_without_bitrate_specification(self, compressed_audio_path, file_name_no_extension):
        """Compress the WAV file and save it as a MP3 file."""
        sf.write(str(Path(compressed_audio_path)) + "/" + str(file_name_no_extension) + '.mp3', self.wav_data, self.wav_sample_rate, format='MP3')
        compressed_audio_amps, compressed_sample_rate = librosa.load(str(Path(compressed_audio_path)) + "/" + str(file_name_no_extension) + '.mp3', dtype='float32')  
        return compressed_audio_amps, compressed_sample_rate

    def compress_load_mp3_from_file(self, compressed_audio_path, file_name_no_extension):
        output_file = Path(compressed_audio_path) / f"{file_name_no_extension}.mp3"
        compressed_audio_amps, compressed_sample_rate = librosa.load(str(output_file), dtype='float32')
        return compressed_audio_amps, compressed_sample_rate


#####################
class OGGCompression(Compression):
    def __init__(self, input_wav):
        super().__init__(input_wav)
        self.audio_segment = AudioSegment.from_file(input_wav)
    
    def compress(self, compressed_audio_path, file_name_no_extension, quality=5):
        output_file = Path(compressed_audio_path) / f"{file_name_no_extension}.ogg"
        ###########
        print('start file compression')
        start_time = time.time()
        self.audio_segment.export(out_f=str(output_file), format="ogg", parameters=["-qscale:a", str(quality)])
        print('end file compression')
        end_time = time.time()  # End time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"Compression time: {elapsed_time:.6f} seconds")
        ############
        compressed_audio_amps, compressed_sample_rate = librosa.load(str(output_file), dtype='float32', sr=None)
        return compressed_audio_amps, compressed_sample_rate
        
    def compress_without_bitrate_specification(self, compressed_audio_path, file_name_no_extension):
        """Compress the WAV file and save it as a OGG file."""
        sf.write(str(Path(compressed_audio_path)) + "/" + str(file_name_no_extension) + '.ogg', self.wav_data, self.wav_sample_rate, format='OGG')
        compressed_audio_amps, compressed_sample_rate = librosa.load(str(Path(compressed_audio_path)) + "/" + str(file_name_no_extension) + '.ogg', dtype='float32')  
        return compressed_audio_amps, compressed_sample_rate

    def compress_load_ogg_from_file(self, compressed_audio_path, file_name_no_extension):
        # Build the output file path.
        output_file = Path(compressed_audio_path) / f"{file_name_no_extension}.ogg"
        compressed_audio_amps, compressed_sample_rate = librosa.load(str(output_file), dtype='float32')
        return compressed_audio_amps, compressed_sample_rate



#######################
class AACCompression(Compression):
    def __init__(self, input_wav):
        super().__init__(input_wav)
        self.audio_segment = AudioSegment.from_file(input_wav)

    def compress(self, compressed_audio_path, file_name_no_extension, bitrate="128k"):
        output_file = Path(compressed_audio_path) / f"{file_name_no_extension}.aac"
        ##########
        print('start file compression')
        self.audio_segment.export(out_f=str(output_file), format="adts", parameters=["-b:a", bitrate])
        print('end file compression')
        #########
        compressed_audio_amps, compressed_sample_rate = librosa.load(str(output_file), dtype='float32', sr=None)
        return compressed_audio_amps, compressed_sample_rate
    
    def compress_old(self, compressed_audio_path, file_name_no_extension):
        """Compress the WAV file and save it as a AAC file."""
        print('todo')
        wave_data  = AudioSegment.from_file(file = self.input_wav)
        print(type(wave_data))
        wave_data.export(out_f='test.aac', format='aac') # todo - needs ffmpeg
        compressed_audio_amps, compressed_sample_rate = librosa.load(str(Path(compressed_audio_path)) + "/" + str(file_name_no_extension) + '.aac', dtype='float32')  
        return compressed_audio_amps, compressed_sample_rate

    def compress_load_aac_from_file(self, compressed_audio_path, file_name_no_extension):
        output_file = Path(compressed_audio_path) / f"{file_name_no_extension}.aac"
        compressed_audio_amps, compressed_sample_rate = librosa.load(str(output_file), dtype='float32')
        return compressed_audio_amps, compressed_sample_rate



#######################
class CS_Compression(Compression):
    def __init__(self, input_wav):
        super().__init__(input_wav)
    
    def compress(self, compressed_audio_path, file_name_no_extension):
        print('todo')
