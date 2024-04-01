from violenceReporter import logger
from violenceReporter.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from violenceReporter.pipeline.stage_02_data_transformation import DataTransformationTrainingPipeline
from violenceReporter.pipeline.stage_03_prepare_base_model import PrepareBaseModelTrainingPipeline
from violenceReporter.pipeline.stage_04_model_training import ModelTrainingPipeline
from violenceReporter.pipeline.stage_05_model_evaluation import EvaluationPipeline
from violenceReporter.pipeline.stage_06_model_prediction import PredictionPipeline




STAGE_NAME = "Data Ingestion"
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nX================================X")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Transformation"
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
    obj = DataTransformationTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nX================================X")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Prepare Base Model"
try:
    logger.info(f"*******************")
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nX===================X")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Training"
try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nX================================X")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Evaluation"
try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = EvaluationPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Prediction"
try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = PredictionPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} stopped <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e