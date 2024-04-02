# Public Violence Detection and Reporting System

This repository contains a machine learning project aimed at detecting violence in public places in real-time and reporting it. The project utilizes transfer learning techniques, leveraging the MobileNetV2 model for human detection, trained on a dataset sourced from Kaggle, specifically the [Real-Life Violence Situations Dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset). The model has been trained on 2000 videos extracted from this dataset.

## Features

- **Real-time Violence Detection**: The system is capable of detecting violence in real-time using the trained ResNet50 model.
- **Face Detection**: Utilizes the MTCNN neural network to detect faces of individuals involved in violence.
- **Reporting Mechanism**: Upon detecting violence, the system reports the incident. Currently, reporting is facilitated through Telegram and data storage is managed using Firebase.
- **High Accuracy**: The model achieves an accuracy of 98.5% on the test set.

## Future Enhancements

- **Location Detection and Reporting**: Integration of location detection to report incidents to the nearest police station.

## How to run?
### STEPS:

Clone the repository

```bash
    git clone https://github.com/JaiSehgal007/violence-detection.git
```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n reporter python=3.9
```

```bash
conda activate reporter
```

```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

### STEP 03 (optional)- Setting up MLflow

- mlflow ui

#### using dagshub for tracking MLflow files remotely 
Copy the credentials from the dagshub repository

Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI=<tracking_uri>

export MLFLOW_TRACKING_USERNAME=<user_name>

export MLFLOW_TRACKING_PASSWORD=<password>

```

### STEP 04 (optional)- Setting up DVC pipeline

1. dvc init
2. dvc repro
3. dvc dag


### STEP 05 - Running all pipelines
```bash
python main.py
```

### STEP 06 - Running streamlit app

```bash
streamlit run app.py
```

## Requirements for Alert System

The prediction model is ready after the above setup, which can be run by 

```bash
python src/violenceReporter/pipeline/stage_06_model_prediction.py
```
but it will not ensure any alert or data bse storage, to configure that set the following environment variables


1. ENV Variables
    - STORAGE_BUCKET = < configure firebase storage to get this >
    - BOT_TOKEN = < create your telegram bot to get this >
    - CHAT_ID = < create your telegram bot to get this >
    - MLFLOW_TRACKING_URI= < create and initalize dagshub repository for your project to get this >
    - MLFLOW_TRACKING_USERNAME= < create and initalize dagshub repository for your project to get this >
    - MLFLOW_TRACKING_PASSWORD= < create and initalize dagshub repository for your project to get this >

2. firebaseKey.json
    get this json file after setting up a firebase project, this can be downloded from
    - firebase console>settings>service accounts
    - Click on generate key
    - store this in the project folder

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with any improvements or suggestions.
