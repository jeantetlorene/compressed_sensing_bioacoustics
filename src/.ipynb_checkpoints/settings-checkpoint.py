import json
import os
from dataclasses import dataclass, field, asdict


@dataclass
class BaseConfig:
    def dict(self):
        return asdict(self)




@dataclass
class DataConfig(BaseConfig):
    force_recreate_dataset: bool = False
    keep_in_memory: bool = False
    species_folder: str = ""
    train_size: float = 0.8
    test_size: float = 0.2
    reshuffle: bool = False
    positive_class: str = ""
    negative_class: str = ""


@dataclass
class PreprocessingConfig(BaseConfig):
    lowpass_cutoff: int = 2000
    downsample_rate: int = 4800
    nyquist_rate: int = 2400
    segment_duration: int = 4
    nb_negative_class: int = 20
    annotation_extension: str = "svl"
    audio_extension: str = ".wav"
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    f_min: int = 4000
    f_max: int = 9000



@dataclass
class ModelConfig(BaseConfig):
    optimizer_name: str = "adam"
    loss_function_name: str = "cross_entropy"
    num_epochs: int = 1
    batch_size: int = 128
    learning_rate: float = 0.001
    shuffle: bool = True
    metric: str = "f1"

@dataclass
class ArchitectureConfig(BaseConfig):
    conv_layers: int = 1
    conv_filters: int = 8
    dropout_rate: float = 0.5
    conv_kernel: int = 8
    max_pooling_size: int = 4
    fc_units: int = 32
    fc_layers: int = 2
    conv_padding: str = None
    stride_maxpool: int = None


@dataclass
class Config(BaseConfig):
    _input: str = field(default=None)
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    cnn_architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)

    def __post_init__(self):
        if self._input is None:
            # Use default settings
            return
        # Check if input is path to a json
        if isinstance(self._input, str):
            self._load_from_json_path(self._input)
        if isinstance(self._input, dict):
            self._set_settings(self._input)

    def get_params(self):
        params = {}
        for key, value in asdict(self).items():
            if key == "_input":
                # params["settings"] = value
                continue
            for sub_key, sub_value in value.items():
                params[f"{key}_{sub_key}"] = sub_value
        return params

    def _load_from_json_path(self, path):
        if path.endswith(".json"):
            json_path = path
        else:
            raise ValueError("Can only load from JSON path.")
        if not os.path.exists(json_path):
            raise ValueError(f"Settings File not Found at {json_path}.")
        # Load File
        with open(json_path, "r") as f:
            data = json.load(f)
        self._set_settings(data)

    def _set_settings(self, data: dict):
        for key, value in data.items():
            if hasattr(self, key):
                config_class = getattr(self, key)
                for sub_key, sub_value in value.items():
                    if hasattr(config_class, sub_key):
                        setattr(config_class, sub_key, sub_value)
                    else:
                        raise ValueError(
                            f"The {sub_key} setting you are trying to set in {key} is not valid."
                        )
