# Public Violence Detection and Reporting System

This repository contains a machine learning project aimed at detecting violence in public places in real-time and reporting it. The project utilizes transfer learning techniques, leveraging the MobileNetV2 model for human detection, trained on a dataset sourced from Kaggle, specifically the [Real-Life Violence Situations Dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset). The model has been trained on 2000 videos extracted from this dataset.

## Features

- **Real-time Violence Detection**: The system is capable of detecting violence in real-time using the trained ResNet50 model.
- **Face Detection**: Utilizes the MTCNN neural network to detect faces of individuals involved in violence.
- **Reporting Mechanism**: Upon detecting violence, the system reports the incident. Currently, reporting is facilitated through Telegram and data storage is managed using Firebase.
- **High Accuracy**: The model achieves an accuracy of 98.5% on the test set.

## Future Enhancements

- **Location Detection and Reporting**: Integration of location detection to report incidents to the nearest police station.

## Installation

1. Clone the repository:

```bash
    git clone https://github.com/your-username/violence-detection.git
```

2. Install the required dependencies:

3. Obtain API keys for Telegram and Firebase and configure them in the appropriate configuration files.

## Usage

1. Run the ViolenceAlertSystem Jupyter noebook after installing all the necessary dependencies

2. The system will start monitoring using the primary web camera of your device.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with any improvements or suggestions.
