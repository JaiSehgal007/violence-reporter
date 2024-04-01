from violenceReporter.config.configuration import ConfigurationManager
from violenceReporter.components.model_prediction import Prediction
from violenceReporter import logger

STAGE_NAME = "Prediction"


class PredictionPipeline:
    def __init__(self):
        pass

    def main(self):
        config=ConfigurationManager()
        pred_config=config.get_prediction_config()
        prediction=Prediction(pred_config)
        prediction.predict()


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PredictionPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} stopped <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e