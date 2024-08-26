from ultralytics import YOLO
import supervision as sv

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        
    def detect_frames(self, frames):
        batch_size = 20
        
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model(frames[i:i + batch_size], conf=0.1, device='mps')
            detections += detections_batch
            
        return detections
            

    def get_objects_tracks(self, frames):
        # Detect objects in frames using YOLO
        detections = self.detect_frames(frames)
        
        tracks = {
            "players": [],
            "ball": []
        }
        
        # Convert detections to the format required by the supervision tracker
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_reverse = {cls_name: i for i, cls_name in enumerate(cls_names)}
            
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
        print(tracks)
        return tracks