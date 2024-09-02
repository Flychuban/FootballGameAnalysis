import numpy as np
import cv2

class ViewTransformer:
    def __init__(self):
        self.pitch_width = 68
        self.pitch_length = 23.32
        
        self.pixel_verticies = np.array({
            [110, 1035],
            [265, 275],
            [910, 260],
            [1640, 915]
        })
        
        self.target_verticies = np.array({
            [0, self.pitch_width],
            [0, 0],
            [self.pitch_length, 0],
            [self.pitch_length, self.pitch_width]
        })
        
        self.pixel_verticies = self.pixel_verticies.astype(np.float32)
        self.target_verticies = self.target_verticies.astype(np.float32)
        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_verticies, self.target_verticies)