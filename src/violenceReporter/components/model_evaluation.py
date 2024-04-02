import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pathlib import Path
import os
from violenceReporter.entity import EvaluationConfig
from violenceReporter.utils.common import save_json

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def plot_metric(self,model_training_history, metric_name_1, metric_name_2, plot_name):
        metric_value_1 = model_training_history[metric_name_1]
        metric_value_2 = model_training_history[metric_name_2]
        epochs = range(len(metric_value_1))
    
        plt.plot(epochs, metric_value_1, 'blue', label=metric_name_1)
        plt.plot(epochs, metric_value_2, 'orange', label=metric_name_2)
        plt.title(str(plot_name))
        plt.legend()


    def generate_training_metrics(self):
        model_history = pd.read_csv(self.config.training_metrics)

        # Plot loss
        self.plot_metric(model_history, 'loss', 'val_loss', 'Total Loss vs Total Validation Loss')
        plt.savefig(os.path.join(self.config.root_dir,'loss_plot.png'))
        plt.close()

        # Plot accuracy
        self.plot_metric(model_history, 'accuracy', 'val_accuracy', 'Total Accuracy vs Total Validation Accuracy')
        plt.savefig(os.path.join(self.config.root_dir,'accuracy_plot.png'))
        plt.close()

        # Plot confusion matrix
        cm = confusion_matrix(self.labels_test, self.labels_predict)
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, fmt='g', ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(['True', 'False'])
        ax.yaxis.set_ticklabels(['NonViolence', 'Violence'])
        plt.savefig(os.path.join(self.config.root_dir,'confusion_matrix.png'))
        plt.close()

        # Save classification report
        classification_report_str = classification_report(self.labels_test, self.labels_predict)
        with open(os.path.join(self.config.root_dir,'classification_report.txt'), 'w') as f:
            f.write(classification_report_str)


        
    def test_set_generator(self):
        features = np.load(os.path.join(self.config.training_data,"features.npy"))
        labels= np.load(os.path.join(self.config.training_data,"labels.npy"))
        video_files_paths = np.load(os.path.join(self.config.training_data,"video_files_paths.npy"))

        one_hot_encoded_labels = to_categorical(labels)
        features_train, features_test, labels_train, labels_test = train_test_split(features, one_hot_encoded_labels, test_size=0.2,
                                                                            shuffle=True, random_state=42)

        self.features_test=features_test
        self.labels_test=labels_test

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    
    def model_predict(self):
        labels_predict = self.model.predict(self.features_test)
        self.labels_predict = np.argmax(labels_predict, axis=1)
        self.labels_test = np.argmax(self.labels_test, axis=1)
    
    def model_evaluate(self):
        self.score = self.model.evaluate(self.features_test, self.labels_test)
    
    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self.test_set_generator()
        self.model_evaluate()
        self.model_predict()
        self.save_score()
        self.generate_training_metrics()
    
    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )

            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(self.model, "model", registered_model_name="MobileNetV2")
            else:
                mlflow.keras.log_model(self.model, "model")
    