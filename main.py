import cv2
import numpy as np


# Function to detect panic behavior
def detect_panic(video_path):
    cap = cv2.VideoCapture(video_path)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Color for visualizing optical flow
    color = (0, 255, 0)

    # Initialize previous frame
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7,
                                       blockSize=7)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)

        # Filter valid points
        valid_pts = next_pts[status == 1]

        # Calculate displacement vectors
        displacement_vectors = valid_pts - prev_pts[status == 1]

        # Calculate average displacement magnitude
        avg_displacement = np.mean(np.linalg.norm(displacement_vectors, axis=1))

        # Visualize optical flow
        for (x, y), (dx, dy) in zip(valid_pts, displacement_vectors):
            cv2.arrowedLine(frame, (int(x), int(y)), (int(x + dx), int(y + dy)), color, 1)
            # Display frame with optical flow visualization
            cv2.imshow('Optical Flow', frame)



        # Update previous frame
        prev_gray = gray.copy()
        prev_pts = valid_pts.reshape(-1, 1, 2)

        # Check for panic behavior (threshold can be adjusted)
        if avg_displacement > 10:
            print("Panic behavior detected!")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Path to the input video
video_path = 'D:\\Development\\Python Projects\\Demo\\PanicDetection\\demo.mp4'

# Call the function to detect panic behavior
detect_panic(video_path)
