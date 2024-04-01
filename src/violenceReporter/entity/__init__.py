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

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    prediction_model_path: Path
    model_checkpoints_path: Path
    model_history: Path
    model_evaluation_history: Path
    params_epochs: int
    params_batch_size: int
    params_image_height: list
    params_image_width: list
    params_test_size: float

@dataclass(frozen=True)
class EvaluationConfig:
    root_dir: Path
    path_of_model: Path
    training_data: Path
    training_metrics: Path
    all_params: dict
    mlflow_uri: str