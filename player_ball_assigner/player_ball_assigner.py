import sys
sys.path.append('../') # Add the parent directory to the path to import utils
from utils import get_center_of_bbox, measure_distance

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70
        
    def assign_ball_to_player(self, player, ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)
        
        minimum_distance = 999999999
        assigned_player = -1
        
        for player_id, player in player.items():
            player_bbox = player['bbox']
            
            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            
            player_distance = min(distance_left, distance_right)
            
            if player_distance < self.max_player_ball_distance:
                if player_distance < minimum_distance:
                    minimum_distance = player_distance
                    assigned_player = player_id
                    
            return assigned_player
            
            