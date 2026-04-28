from pathlib import Path

species_settings = {
    "gibbon": {
        "data": {
            "species_folder": Path("E:/Bioacoustics_compressed_sensing/Gibbon"),
            "positive_class": "gibbon",
            "negative_class": "no-gibbon",
        },
        "preprocessing": {
            "sample_rate": 9600,
            "lowpass_cutoff": 2000,
            "downsample_rate": 4800,
            "nyquist_rate": 2400,
            "segment_duration": 4,
            "nb_negative_class": 10,
            "audio_extension": ".wav",
            "n_fft": 1024,
            "hop_length": 256,
            "n_mels": 128,
            "f_min": 1000,
            "f_max": 2000,
        },
        "cnn_architecture": {
            "conv_layers": 1,
            "conv_filters": 8,
            "dropout_rate": 0.5,
            "conv_kernel": 8,
            "max_pooling_size": 4,
            "fc_units": 32,
            "fc_layers": 2,
            "conv_padding": None,
        }
    },

    "thyolo": {
        "data": {
            "species_folder": Path("E:/Bioacoustics_compressed_sensing/Thyolo"),
            "positive_class": "thyolo-alethe",
            "negative_class": "noise",
        },
        "preprocessing": {
            "sample_rate": 32000,
            "lowpass_cutoff": 3100,
            "downsample_rate": 6400,
            "nyquist_rate": 3200,
            "segment_duration": 1,
            "nb_negative_class": 40,
            "audio_extension": ".WAV",
            "n_fft": 1024,
            "hop_length": 256,
            "n_mels": 128,
            "f_min": 1500,
            "f_max": 3000,
        },
        "cnn_architecture": {
            "conv_layers": 1,
            "conv_filters": 8,
            "dropout_rate": 0.5,
            "conv_kernel": 8,
            "max_pooling_size": 4,
            "fc_units": 32,
            "fc_layers": 2,
            "conv_padding": None,
        }
    },

    "ptw": {
        "data": {
            "species_folder": Path("E:/Bioacoustics_compressed_sensing/PTW"),
            "positive_class": "PTW",
            "negative_class": "NOISE",
        },
        "preprocessing": {
            "sample_rate": 48000,
            "lowpass_cutoff": 9000,
            "downsample_rate": 18400,
            "nyquist_rate": 9200,
            "segment_duration": 2,
            "nb_negative_class": 5,
            "audio_extension": ".WAV",
            "n_fft": 1024,
            "hop_length": 256,
            "n_mels": 128,
            "f_min": 4000,
            "f_max": 9000,
        },
        "cnn_architecture": {
            "conv_layers": 1,
            "conv_filters": 8,
            "dropout_rate": 0.5,
            "conv_kernel": 8,
            "max_pooling_size": 4,
            "fc_units": 32,
            "fc_layers": 2,
            "conv_padding": None,
        }
    },

    "bats": {
        "data": {
            "species_folder": Path("E:/Bioacoustics_compressed_sensing/Bats"),
            "positive_class":"" ,
            "negative_class": "",
        },
        "preprocessing": {
            "sample_rate": 256000,
            "lowpass_cutoff": 2000,
            "downsample_rate": 128000,
            "nyquist_rate": 2400,
            "segment_duration": 1,
            "nb_negative_class": 10,
            "audio_extension": ".wav",
            "n_fft": 512,
            "hop_length": 128*3,
            "n_mels": 128,
            "f_min": 15000,
            "f_max": 64000,
        },
        "cnn_architecture": {
            "conv_layers": 1,
            "conv_filters": 8,
            "dropout_rate": 0.5,
            "conv_kernel": 8,
            "max_pooling_size": 4,
            "fc_units": 32,
            "fc_layers": 2,
            "conv_padding": None,
        }
    }
}

def get_settings(species):
    if species in species_settings:
        return species_settings[species]
    else:
        raise ValueError(f"Unknown species '{species}'. Available options: {list(species_settings.keys())}")
