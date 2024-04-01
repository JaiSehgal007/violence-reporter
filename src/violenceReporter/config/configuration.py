import os
from violenceReporter.constants import *
from violenceReporter.utils.common import read_yaml,create_directories
from violenceReporter.entity import DataIngestionConfig,DataTransformationConfig,PrepareBaseModelConfig,TrainingConfig,EvaluationConfig,PredictionConfig
class ConfigurationManager:
    def __init__(
            self,
            config_filepath = CONFIG_FILE_PATH,
            params_filepath = PARAMS_FILE_PATH):
        
        self.config=read_yaml(config_filepath)
        self.params=read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self)-> DataIngestionConfig:
        config=self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config
    
    def get_data_transformation_config(self)-> DataTransformationConfig:
        config=self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir= config.root_dir,
            dataset_dir= config.dataset_dir,
            classes_list= config.classes_list,
            params_image_height= self.params.IMAGE_HEIGHT,
            params_image_width= self.params.IMAGE_WIDTH,
            params_sequence_length= self.params.SEQUENCE_LENGTH
        )

        return data_transformation_config
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES,
            params_image_height= self.params.IMAGE_HEIGHT,
            params_image_width= self.params.IMAGE_WIDTH,
            params_sequence_length= self.params.SEQUENCE_LENGTH,
            params_istrainable= self.params.TRAINABLE,
            params_freeze_till= self.params.FREEZE_TILL
        )

        return prepare_base_model_config
    
    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = self.config.data_transformation.root_dir
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            prediction_model_path= Path(training.prediction_model_path),
            model_checkpoints_path= Path(training.model_checkpoints_path),
            model_evaluation_history= Path(training.model_evaluation_history),
            model_history= Path(training.model_history),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_image_height=params.IMAGE_HEIGHT,
            params_image_width=params.IMAGE_WIDTH,
            params_test_size=params.TEST_SIZE
        )

        return training_config
    
    def get_evaluation_config(self) -> EvaluationConfig:
        create_directories([
            Path(self.config.evaluation.root_dir)
        ])

        eval_config = EvaluationConfig(
            root_dir=Path(self.config.evaluation.root_dir),
            path_of_model=Path(self.config.training.trained_model_path),
            training_metrics=Path(self.config.evaluation.training_metrics),
            training_data=Path(self.config.data_transformation.root_dir),
            mlflow_uri="https://dagshub.com/JaiSehgal007/violence-reporter.mlflow",
            all_params=self.params,
        )
        return eval_config
    
    def get_prediction_config(self) -> PredictionConfig:
        config=self.config.model_prediction
        create_directories([
            Path(config.root_dir)
        ])

        eval_config = PredictionConfig(
            root_dir=Path(config.root_dir),
            model_path=Path(config.prediction_model_path),
            classes_list=config.classes_list,
            params_image_height=self.params.IMAGE_HEIGHT,
            params_image_width=self.params.IMAGE_WIDTH,
            params_sequence_length=self.params.SEQUENCE_LENGTH
        )
        return eval_config