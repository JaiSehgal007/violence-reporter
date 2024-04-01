from collections import deque
import numpy as np
import cv2
import telepot
from datetime import datetime
import pytz
from PIL import Image, ImageEnhance
import os
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Circle
from mtcnn.mtcnn import MTCNN
from dotenv import load_dotenv
import tensorflow as tf
import time
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage
from pathlib import Path
from violenceReporter.entity import PredictionConfig
from violenceReporter import logger

class Prediction:
    def __init__(self, config: PredictionConfig):
        self.config = config

    def set_credentials(self):
        load_dotenv()
        self.STORAGE_BUCKET = os.getenv('STORAGE_BUCKET')
        self.BOT_TOKEN = os.getenv('BOT_TOKEN')
        self.CHAT_ID = os.getenv('CHAT_ID')


    def preprocess_frame(self,frame):
        resized_frame = cv2.resize(frame, (self.config.params_image_height, self.config.params_image_width))
        normalized_frame = resized_frame / 255
        return normalized_frame
    
    def img_enhance(self,frame):
        # Convert OpenCV frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Enhance sharpness
        curr_bri = ImageEnhance.Sharpness(frame_pil)
        new_bri = 1.3
        img_brightened = curr_bri.enhance(new_bri)

        # Enhance color
        curr_col = ImageEnhance.Color(img_brightened)
        new_col = 1.5
        img_col = curr_col.enhance(new_col)

        # Save enhanced image
        img_col.save(os.path.join(self.config.root_dir,"finalImage.jpg"))

        return img_col

    @staticmethod
    def get_time():
        IST = pytz.timezone('Asia/Kolkata')
        timeNow = datetime.now(IST)
        return timeNow
    
    def draw_faces(self,filename, result_list):
        # load the image
        data = plt.imread(filename)
        # plot each face as a subplot
        for i in range(len(result_list)):
            # get coordinates
            x1, y1, width, height = result_list[i]['box']
            x2, y2 = x1 + width, y1 + height
            # define subplot
            plt.subplot(1, len(result_list), i+1)
            plt.axis('off')
            # plot face
            plt.imshow(data[y1:y2, x1:x2])
        # show the plot
        plt.savefig(os.path.join(self.config.root_dir,"faces.png"))
    

    def initialize_database(self):
        try:
            cred = credentials.Certificate("firebaseKey.json")
            firebase_admin.initialize_app(cred, {'storageBucket': self.STORAGE_BUCKET})  # Initialize once
            self.db = firestore.client()

        except Exception as e:
            logger.info(f"error occured : {e}")

    def predict_webcam(self,confidence_threshold=0.75):
        SEQUENCE_LENGTH=self.config.params_sequence_length
        cap = cv2.VideoCapture(0)
        bot = telepot.Bot(self.BOT_TOKEN)
        frames_list = []
        location = "Sector 20, Noida"
        violence_image = os.path.join(self.config.root_dir,"finalImage.jpg")
        face_image = os.path.join(self.config.root_dir,"faces.png")
        no_of_detections, alert_sent = 0, 0
        last_alert_time = None

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Preprocess the frame
            normalized_frame = self.preprocess_frame(frame)
            frames_list.append(normalized_frame)

            # Ensure we have enough frames for the sequence
            if len(frames_list) == SEQUENCE_LENGTH:
                # Perform prediction
                predicted_labels_probabilities = self.model.predict(np.expand_dims(frames_list, axis=0))[0]
                predicted_label = np.argmax(predicted_labels_probabilities)
                predicted_class_name = self.config.classes_list[predicted_label]

                # Display the prediction
                confidence = predicted_labels_probabilities[predicted_label]
                print(f'Predicted: {predicted_class_name}\nConfidence: {confidence}')

                # Display "Violence" in red if confidence is above the threshold
                if predicted_class_name == "Violence" and confidence > confidence_threshold:
                    cv2.putText(frame, "Violence", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    no_of_detections += 1

                    if no_of_detections >= 7 and alert_sent == 0:
                        current_time = self.get_time()
                        self.img_enhance(frame)
                        pixels = plt.imread(violence_image)
                        detector = MTCNN()
                        faces = detector.detect_faces(pixels)
                        self.draw_faces(violence_image, faces)

                        try:
                            bot.sendMessage(self.CHAT_ID, f"VIOLENCE ALERT!! \n at location: {location} \n time: {current_time}")
                            bot.sendPhoto(self.CHAT_ID, photo=open(os.path.join(self.config.root_dir,"finalImage.jpg"), 'rb'))
                            bot.sendMessage(self.CHAT_ID, "Faces Obtained")
                            bot.sendPhoto(self.CHAT_ID, photo=open(os.path.join(self.config.root_dir,"faces.png"), 'rb'))

                            storage_client = storage.bucket()
                            storage_client.blob(violence_image).upload_from_filename(violence_image)
                            storage_client.blob(face_image).upload_from_filename(face_image)

                            # Get download URLs for the uploaded images
                            violence_image_url = storage_client.blob(violence_image).public_url
                            face_image_url = storage_client.blob(face_image).public_url

                            # Add data to Firestore with download URLs
                            self.db.collection(location).add({
                                'date': current_time,
                                'image': violence_image_url,
                                'faces': face_image_url
                            })
                            alert_sent = 1
                            last_alert_time = time.time()

                        except Exception as e:
                            print(f"Error sending elert: {e}")

                        finally:
                            pass

                # Check if it's been 5 minutes since the last alert
                if last_alert_time is not None and time.time() - last_alert_time >= 5 * 60:
                    alert_sent = 0

                # Clear the frames list for the next sequence
                frames_list = []

            # Display the webcam feed
            cv2.imshow('Violence Detector', frame)

            # Break the loop if 'q' is pressed        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the webcam and close the window outside the loop
        cap.release()
        cv2.destroyAllWindows()

        

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    
    
    def predict(self):
        self.model = self.load_model(self.config.model_path)
        self.set_credentials()
        self.initialize_database()
        self.predict_webcam()
    