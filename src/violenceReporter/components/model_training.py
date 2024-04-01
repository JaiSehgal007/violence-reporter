import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
import numpy as np
import csv
import pandas as pd
from pathlib import Path
from violenceReporter.entity import TrainingConfig

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = load_model(self.config.updated_base_model_path)

    def train_test_generator(self):

        features = np.load(os.path.join(self.config.training_data,"features.npy"))
        labels= np.load(os.path.join(self.config.training_data,"labels.npy"))
        video_files_paths = np.load(os.path.join(self.config.training_data,"video_files_paths.npy"))

        one_hot_encoded_labels = to_categorical(labels)
        features_train, features_test, labels_train, labels_test = train_test_split(features, one_hot_encoded_labels, test_size=self.config.params_test_size,
                                                                            shuffle=True, random_state=42)

        return features_train, features_test, labels_train, labels_test

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    @staticmethod
    def save_csv(path, data):
        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data.keys())
            writer.writerow(data.values())

    
    def train(self):
        features_train, features_test, labels_train, labels_test=self.train_test_generator()

        early_stopping_callback = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=5, min_lr=0.00005, verbose=1)

        model_checkpoint = ModelCheckpoint(self.config.model_checkpoints_path, save_best_only=True)
        
        self.model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=["accuracy"])

        model_history = self.model.fit(
            x=features_train, 
            y=labels_train, 
            epochs=self.config.params_epochs, 
            batch_size=self.config.params_batch_size, 
            shuffle=True, 
            validation_split=self.config.params_test_size, 
            callbacks=[early_stopping_callback, reduce_lr, model_checkpoint])
        
        model_evaluation_history = self.model.evaluate(features_test, labels_test)

        self.save_model(self.config.trained_model_path,self.model)
        self.save_model(self.config.prediction_model_path,self.model)
        pd.DataFrame(model_history.history).to_csv(self.config.model_history, index=False)
        pd.DataFrame(model_evaluation_history).to_csv(self.config.model_evaluation_history, index=False)