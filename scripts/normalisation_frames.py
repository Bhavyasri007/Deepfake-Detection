import cv2
import os

def normalize_frames_and_save(input_folder, output_folder):
    for frame_file in os.listdir(input_folder):
        frame_path = os.path.join(input_folder, frame_file)
        original_frame = cv2.imread(frame_path)
        normalized_frame = original_frame / 255.0  # Normalize pixel values
        # Scale back to original range and convert to integer
        normalized_frame = (normalized_frame * 255).astype(int)
        # Save normalized frame to output folder
        output_path = os.path.join(output_folder, frame_file)
        cv2.imwrite(output_path, normalized_frame)

# Example usage:
input_folder = "D:\\deepfake project\\scripts\\images"  # Replace with the path to your input folder containing frames
output_folder = "D:\\deepfake project\\rnn+efficientNet\\images" # Replace with the path to your output folder for normalized frames
normalize_frames_and_save(input_folder, output_folder)

