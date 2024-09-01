import pickle
import os
import cv2
import numpy as np
import sys
sys.path.append('../') # Add the parent directory to the path to import utils
from utils import measure_distance, measure_xy_distance

class CameraMovementEstimator():
    def __init__(self, frame):
        self.min_camera_movement_distance = 5
        
        self.lk_params = {
            "winSize": (15, 15),
            "maxLevel": 2,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        }
        
        
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        masked_features = np.zeros_like(first_frame_grayscale)
        masked_features[:, 0:20] = 1 # Top part of the pitch
        masked_features[:, 900:1050] = 1 # Bottom part of the pitch
        
        self.features = {
            "maxCorners": 100,
            "qualityLevel": 0.3,
            "minDistance": 3,
            "blockSize": 7,
            "mask": masked_features
        }
        
        
        
    
    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        # Read the camera movement from a stub file if it exists
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)
        
        camera_movement = [[0, 0] * len(frames)]
        
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)
        
        for frame_num in range(1, len(frames)):
            new_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, old_features, None, **self.lk_params)
            
            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0
            for i, (new, old) in enumerate(new_features, old_features):
                new_features_point = new.ravel()
                old_features_point = old.ravel()
                distance = measure_distance(new_features_point, old_features_point)
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, new_features_point)
                    
            if max_distance > self.min_camera_movement_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(new_gray, **self.features)
            
            old_gray = new_gray.copy()
        
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(camera_movement,f)
        
        return camera_movement