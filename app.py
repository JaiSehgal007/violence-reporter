import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes
import numpy as np
import av
from collections import deque
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
from violenceReporter.config.configuration import ConfigurationManager
from violenceReporter.components.model_prediction import Prediction


# Function to run when app starts or restarts
@st.cache_resource
def setup():
    config=ConfigurationManager()
    pred_config=config.get_prediction_config()
    prediction=Prediction(pred_config)
    prediction.set_app_cred()
    return prediction

# Call setup function to run when app starts or restarts
# Prediction Object
prediction= setup()


st.title("Violence Reporter")


# Prediction function
def predict(frame: av.VideoFrame):
    global model_response

    img = frame.to_ndarray(format="bgr24")

    prediction_class=prediction.app_predict(img)
    if prediction_class!=None: 
        model_response=prediction_class
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")


# Display video stream and model response
webrtc_streamer(
    key="streamer",
    video_frame_callback=predict,
    sendback_audio=False
)

