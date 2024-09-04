import sys
sys.path.append('../') # Add the parent directory to the path to import utils
from utils import measure_distance

class SpeedAndDistanceEstimator:
    def __init__(self, speed_estimator, distance_estimator):
        self.frame_window = 5
        self.frame_rate = 24
        
    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance_covered = {}
        
        for object, object_track in tracks.items():
            if object == 'ball':
                continue
            elif object == 'player':
                number_of_frames = len(object_track)
                for frame_num in range(0, number_of_frames, self.frame_window):
                    last_frame = min(frame_num + self.frame_window, number_of_frames)
                    for track_id, _ in object_track[frame_num].items():
                        if track_id not in object_track[last_frame]:
                            continue
                        start_position = object_track[frame_num][track_id]['position_transformed']
                        end_position = object_track[last_frame][track_id]['position_transformed']
                        
                        if start_position is None or end_position is None: 
                            continue
                        distance_covered = measure_distance(start_position, end_position)
                        time_elapsed = (last_frame - frame_num) / self.frame_rate
                        speed_meters_per_second = distance_covered / time_elapsed
                        speed_km_per_hour = speed_meters_per_second * 3.6
                        
                        if object not in total_distance_covered[object]:
                            total_distance_covered[object] = 0
                        
                        total_distance_covered[object] += distance_covered
                        