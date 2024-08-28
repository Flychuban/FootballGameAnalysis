from utils import read_video_frames, save_video_frames
from trackers import Tracker
import cv2

def main():
    video_path = 'input_videos/test_video.mp4'
    frames = read_video_frames(video_path)
     
    # Initialize the tracker
    tracker = Tracker('models/best.pt')
    
    tracks = tracker.get_objects_tracks(frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')
    
    # # save croped image for every player
    # for track_id, player in tracks["players"][0].items():
    #     bbox = player['bbox']
    #     frame = frames[0]
    #     cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        
    #     # Save the cropped image
    #     cv2.imwrite(f'output_videos/player_{track_id}_img.jpg', cropped_image)
    #     break
        
    # Draw the tracks on the frames
    output_video_frames = tracker.draw_annotations(frames, tracks)
    
    save_video_frames(output_video_frames, 'output_videos/video.avi')
    
if __name__ == '__main__':
    main()