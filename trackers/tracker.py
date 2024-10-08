from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import sys
import numpy as np
import pandas as pd
sys.path.append('../') # Add the parent directory to the path to import utils
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        
    def add_position_to_track(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info.get('bbox', [])
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    
                    tracks[object][frame_num][track_id]['position'] = get_center_of_bbox(bbox)
                        
        
        
    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        
        # Interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        
        ball_positions = [{1 : {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]
        
        return ball_positions
        
    def detect_frames(self, frames):
        batch_size = 20
        
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model(frames[i:i + batch_size], conf=0.1, device='mps')
            detections += detections_batch
            
        return detections
            

    def get_objects_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)
        
        # Detect objects in frames using YOLO
        detections = self.detect_frames(frames)
        
        tracks = {
            "players": [],
            "ball": []
        }
        
        # Convert detections to the format required by the supervision tracker
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_reverse = {cls_name: i for i, cls_name in cls_names.items()}
            detection_supervision = sv.Detections.from_ultralytics(detection)
            # Update the tracker with the detections and track the objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            tracks["players"].append({})
            tracks["ball"].append({})
            
            
            # Add tracking only for players
            for obj in detection_with_tracks:
                bbox = obj[0].tolist()
                class_id = obj[3]
                track_id = obj[4]
                
                if class_id == cls_names_reverse['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                    
            # Detect ball with detection_supervision
            for obj in detection_supervision:
                bbox = obj[0].tolist()
                class_id = obj[3]
                
                if class_id == cls_names_reverse['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox} # Only one ball in the frame
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        
        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        
        cv2.ellipse(frame, (x_center, y2), (int(width), int(width*0.4)), angle=0.0, startAngle=-45, endAngle=235, color=color, thickness=2, lineType=cv2.LINE_4)
        
        # Draw rectangle for track id
        rectangle_height = 20
        rectangle_width = 40
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15
        
        if track_id is not None:
            cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv2.FILLED)
            
            x1_text = x1_rect + 20
            if track_id > 99:
                x1_text = x1_text - 10
            
            cv2.putText(frame, str(track_id), (int(x1_text), int(y2_rect)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame
    
    def draw_triangle(self, frame, bbox, color, track_id=None):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        triangle_points = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])
        
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED) # Fill the triangle
        cv2.polylines(frame, [triangle_points], 0, (0, 0, 0), thickness=2) # Draw the triangle border
        return frame
    
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw a semi-transparent rectangle to show the team ball control
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw the team ball control text
        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of times each team had the ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]
        team1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames) if team_1_num_frames + team_2_num_frames > 0 else 0
        team2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames) if team_1_num_frames + team_2_num_frames > 0 else 0
        
        cv2.putText(frame, f"Team 1 Ball Control: {team1*100:.2f}5", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team2*100:.2f}5", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        
        return frame
        
    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            
            # Draw players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (255, 0, 0))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)
                
                if player.get("has_ball", False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))
                
            # Draw ball
            for ball_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))
            
            # Draw team ball control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            
            output_video_frames.append(frame) 
        
        return output_video_frames