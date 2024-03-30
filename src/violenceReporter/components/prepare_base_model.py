import tensorflow as tf
from tensorflow import keras
from keras.layers import *
from keras.models import Sequential
from keras.applications.mobilenet_v2 import MobileNetV2
from violenceReporter.entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self,config:PrepareBaseModelConfig):
        self.config=config
    
    def get_base_model(self):
        self.model = MobileNetV2(
            include_top=self.config.params_include_top, 
            weights=self.config.params_weights)
        
        self.model.trainable = self.config.params_istrainable

        self.save_model(self.config.base_model_path,model=self.model)
    
    @staticmethod
    def prepare_full_model(base_model,classes,freeze_till,sequence_length,image_height,image_width):
        for layer in base_model.layers[:freeze_till]:
            layer.trainable=False
        

        model = Sequential()
        model.add(Input(shape=(sequence_length,image_height,image_width, 3)))
        model.add(TimeDistributed(base_model))
        model.add(Dropout(0.25))
        model.add(TimeDistributed(Flatten()))

        lstm_fw = LSTM(units=32)
        lstm_bw = LSTM(units=32, go_backwards=True)
        model.add(Bidirectional(lstm_fw, backward_layer=lstm_bw))

        model.add(Dropout(0.25))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(classes, activation='softmax'))
        

        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=["accuracy"])

        model.summary()

        return model
    
    def update_base_model(self):
        self.full_model = self.prepare_full_model(
            base_model=self.model,
            classes=self.config.params_classes,
            freeze_till=self.config.params_freeze_till,
            sequence_length=self.config.params_sequence_length,
            image_height=self.config.params_image_height,
            image_width=self.config.params_image_width
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)