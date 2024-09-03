from utils import read_video_frames, save_video_frames
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer

def main():
    video_path = 'input_videos/test_video.mp4'
    frames = read_video_frames(video_path)
     
    # Initialize the tracker
    tracker = Tracker('models/best.pt')
    
    tracks = tracker.get_objects_tracks(frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')
    
    # Get object positions
    tracker.add_position_to_track(tracks)
    
    # Initialize the camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(frames, read_from_stub=True, stub_path='stubs/camera_movement_stubs.pkl')
    camera_movement_estimator.add_adjust_camera_movement(tracks, camera_movement_per_frame)
    
    # Initialize the view transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_positions_to_tracks(tracks)
    
    # Interpolate the ball positions
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])
    
    # # save croped image for every player
    # for track_id, player in tracks["players"][0].items():
    #     bbox = player['bbox']
    #     frame = frames[0]
    #     cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        
    #     # Save the cropped image
    #     cv2.imwrite(f'output_videos/player_{track_id}_img.jpg', cropped_image)
    #     break
    
    # Assign team colors to players
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frames[0], tracks["players"][0])
    
    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(frames[frame_num], track['bbox'], player_id)

            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
            
        # Assign the ball to a player
    player_ball_assigner = PlayerBallAssigner()
    team_ball_control = []
    
    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_ball_assigner.assign_ball_to_player(player_track, ball_bbox)
        
        # Player was assigned to the ball
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if len(team_ball_control) > 0 else 0)
    team_ball_control = np.array(team_ball_control)
            
    # Draw the tracks on the frames
    output_video_frames = tracker.draw_annotations(frames, tracks, team_ball_control)
    
    # Draw the camera movement on the frames
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    
    save_video_frames(output_video_frames, 'output_videos/video.avi')
    
if __name__ == '__main__':
    main()