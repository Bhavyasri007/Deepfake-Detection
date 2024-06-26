import cv2
import os

def extract_frames(video_path, output_path):
    """Extract frames from a video."""
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return
    
    # Read and save frames
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if ret:
            # Save frame as image
            frame_path = os.path.join(output_path, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1
        else:
            break
    
    # Release the video capture object
    cap.release()

# Example usage:
video_path = "D:\\deepfake project\\data\\raw_data\\person 1\\id0_0000.mp4"  # Replace with the path to your video file
output_path = "D:\\deepfake project\\data\\frames\\person 1\\video 1"  # Choose where to save the extracted frames
extract_frames(video_path, output_path)

video_path = "D:\\deepfake project\\data\\raw_data\\person 1\\id0_0001.mp4"  # Replace with the path to your video file
output_path = "D:\\deepfake project\\data\\frames\\person 1\\video 2"  # Choose where to save the extracted frames
extract_frames(video_path, output_path)

video_path = "D:\\deepfake project\\data\\raw_data\\person 1\\id0_0002.mp4"  # Replace with the path to your video file
output_path = "D:\\deepfake project\\data\\frames\\person 1\\video 3"  # Choose where to save the extracted frames
extract_frames(video_path, output_path)

video_path = "D:\\deepfake project\\data\\raw_data\\person 1\\id0_0003.mp4"  # Replace with the path to your video file
output_path = "D:\\deepfake project\\data\\frames\\person 1\\video 4"  # Choose where to save the extracted frames
extract_frames(video_path, output_path)

video_path = "D:\\deepfake project\\data\\raw_data\\person 1\\id0_0005.mp4"  # Replace with the path to your video file
output_path = "D:\\deepfake project\\data\\frames\\person 1\\video 5"  # Choose where to save the extracted frames
extract_frames(video_path, output_path)

video_path = "D:\\deepfake project\\data\\raw_data\\person 1\\id0_0006.mp4"  # Replace with the path to your video file
output_path = "D:\\deepfake project\\data\\frames\\person 1\\video 6"  # Choose where to save the extracted frames
extract_frames(video_path, output_path)

video_path = "D:\\deepfake project\\data\\raw_data\\person 1\\id0_0007.mp4"  # Replace with the path to your video file
output_path = "D:\\deepfake project\\data\\frames\\person 1\\video 7"  # Choose where to save the extracted frames
extract_frames(video_path, output_path)

video_path = "D:\\deepfake project\\data\\raw_data\\person 1\\id0_0009.mp4"  # Replace with the path to your video file
output_path = "D:\\deepfake project\\data\\frames\\person 1\\video 8"  # Choose where to save the extracted frames
extract_frames(video_path, output_path)

