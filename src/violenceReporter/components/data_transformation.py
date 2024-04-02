import cv2
import os
import numpy as np
from violenceReporter.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self,config: DataTransformationConfig):
        self.config=config

    def frames_extraction(self,video_path)->list:
        '''
        Extracts frames and normalize the frames from video
        '''
        frames_list = []
        video_reader = cv2.VideoCapture(video_path)
        video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_frames_window = max(int(video_frames_count/self.config.params_sequence_length), 1)

        for frame_counter in range(self.config.params_sequence_length):
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
            success, frame = video_reader.read()

            if not success:
                break

            resized_frame = cv2.resize(frame, (self.config.params_image_height, self.config.params_image_width))
            normalized_frame = resized_frame / 255
            frames_list.append(normalized_frame)

        video_reader.release()
        return frames_list
    

    def create_dataset(self):
        ''' 
        Extract video features and labels and convert in numpy array
        '''
        features = []
        labels = []
        video_files_paths = []

        for class_index, class_name in enumerate(self.config.classes_list):
            print(f'Extracting Data of Class: {class_name}')
            files_list = os.listdir(os.path.join(self.config.dataset_dir, class_name))

            for file_name in files_list:
                video_file_path = os.path.join(self.config.dataset_dir, class_name, file_name)
                frames = self.frames_extraction(video_file_path)

                if len(frames) == self.config.params_sequence_length:
                    features.append(frames)
                    labels.append(class_index)
                    video_files_paths.append(video_file_path)

        features = np.asarray(features)
        labels = np.array(labels)

        return features, labels, video_files_paths
    
    def transform_dataset(self):
        '''
        Save transformed dataset to specified location
        '''
        features, labels, video_files_paths = self.create_dataset()
        np.save(f"{self.config.root_dir}/features.npy", features)
        np.save(f"{self.config.root_dir}/labels.npy", labels)
        np.save(f"{self.config.root_dir}/video_files_paths.npy", video_files_paths)
        
    
    