import cv2
import numpy as np
import os

def generate_motion_image(frame1, frame2):
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using Lucas-Kanade method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Calculate magnitude and angle of flow vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Scale the magnitude to 0-255
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Convert angle to hue
    hue = angle * 180 / np.pi / 2

    # Create an HSV image
    hsv = np.zeros((frame1.shape[0], frame1.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = hue
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Convert HSV to BGR
    motion_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return motion_image

# Create output folder if it doesn't exist
output_folder = 'processed_frames'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Read the video file
video_capture = cv2.VideoCapture('demo.mp4')

# Read the first frame
ret, prev_frame = video_capture.read()
frame_count = 0

# Loop through the frames
while True:
    # Read the next frame
    ret, next_frame = video_capture.read()
    if not ret:
        break

    # Generate motion information image
    motion_image = generate_motion_image(prev_frame, next_frame)

    # Save motion information image
    output_path = os.path.join(output_folder, f"panic_frame_{frame_count:04d}.jpg")
    cv2.imwrite(output_path, motion_image)

    # Update previous frame
    prev_frame = next_frame
    frame_count += 1

# Release video capture
video_capture.release()
