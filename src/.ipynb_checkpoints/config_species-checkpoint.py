from pathlib import Path

species_settings = {
    "gibbon": {
        "data": {
            #"species_folder": Path("../data/hainan_gibbon"),
            "species_folder": Path("D:/Bioacoustics_compressed_sensing/Gibbon"),
            "positive_class": "gibbon",
            "negative_class": "no-gibbon",
        },
        "preprocessing": {
            "lowpass_cutoff": 2000,
            "downsample_rate": 4800,
            "nyquist_rate": 2400,
            "segment_duration": 4,
            "nb_negative_class": 10,
            "audio_extension": ".wav",
            "n_fft": 1024,
            "hop_length": 256,
            "n_mels": 128,
            "f_min": 0,
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
            "species_folder": Path("D:/Bioacoustics_compressed_sensing/Thyolo"),
            "positive_class": "thyolo",
            "negative_class": "no-thyolo",
        },
        "preprocessing": {
            "lowpass_cutoff": 3100,
            "downsample_rate": 6400,
            "nyquist_rate": 3200,
            "segment_duration": 1,
            "nb_negative_class": 10,
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
            "species_folder": Path("D:/Bioacoustics_compressed_sensing/PTW"),
            "positive_class": "ptw",
            "negative_class": "no-ptw",
        },
        "preprocessing": {
            "lowpass_cutoff": 9000,
            "downsample_rate": 18400,
            "nyquist_rate": 9200,
            "segment_duration": 2,
            "nb_negative_class": 3,
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
    }
}

def get_settings(species):
    if species in species_settings:
        return species_settings[species]
    else:
        raise ValueError(f"Unknown species '{species}'. Available options: {list(species_settings.keys())}")
