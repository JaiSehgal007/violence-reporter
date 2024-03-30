from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    dataset_dir: Path
    classes_list: list
    params_image_height: int
    params_image_width: int
    params_sequence_length: int

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir:Path
    base_model_path:Path
    updated_base_model_path: Path
    params_image_height: int
    params_image_width: int
    params_sequence_length: int
    params_include_top: bool
    params_weights: str
    params_istrainable: bool
    params_freeze_till: int
    params_classes: int