from utils import read_video_frames, save_video_frames

def main():
    video_path = 'input_videos/test_video.mp4'
    frames = read_video_frames(video_path)
    save_video_frames(frames, 'output_videos/video.avi')
    
if __name__ == '__main__':
    main()