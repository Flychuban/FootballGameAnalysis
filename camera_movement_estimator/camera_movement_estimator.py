import pickle
import os
import cv2
import numpy as np

class CameraMovementEstimator():
    def __init__(self, frames):
        first_frame_grayscale = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
        masked_features = np.zeros_like(first_frame_grayscale)
        masked_features[:, 0:20] = 1
        masked_features[:, 900:1050] = 1
        
        self.features = {
            "maxCorners": 100,
            "qualityLevel": 0.3,
            "minDistance": 3,
            "blockSize": 7,
            "mask": masked_features
        }
        
        for frame_num in range(1, len(frames)):
            new_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features = cv2.goodFeaturesToTrack(new_gray, **self.features)
            new_features = np.int0(new_features)
            
            # Calculate the optical flow
            flow = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, old_features, new_features)
            
            # Select good points
            good_new = flow[:, 0][flow[:, 1] == 1]
            good_old = old_features[flow[:, 1] == 1]
            
            # Calculate the movement
            movement = good_new - good_old
            camera_movement[frame_num] = movement
            
            # Update the old features and gray frame
            old_gray = new_gray
            old_features = new_features
    
    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        camera_movement = [[0, 0] * len(frames)]
        
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)
            
        return None