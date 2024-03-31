import cv2
import numpy as np
from keras.models import load_model
import joblib

# Load pre-trained models
svm_model = joblib.load('svm_model.joblib')
cnn_model = load_model('cnn_model.h5')

def generate_motion_image(frame1, frame2):
    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

def preprocess_motion_image(image):
    resized_image = cv2.resize(image, (640, 360))
    expanded_image = np.expand_dims(resized_image, axis=-1)
    return np.expand_dims(expanded_image, axis=0)

def predict_panic(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    predictions = []

    while True:
        ret, next_frame = cap.read()
        if not ret:
            break

        motion_image = generate_motion_image(prev_frame, next_frame)
        preprocessed_image = preprocess_motion_image(motion_image)

        # Extract features using CNN model
        cnn_features = cnn_model.predict(preprocessed_image)

        # Predict using SVM model
        svm_prediction = svm_model.predict(cnn_features)
        predictions.append(svm_prediction)

        # Display video with motion information
        cv2.imshow('Video with Motion Information', motion_image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        prev_frame = next_frame

    cap.release()
    cv2.destroyAllWindows()

    return predictions

# Example usage
video_path = 'non_panic.mp4'
predictions = predict_panic(video_path)

# Assuming 1 indicates panic and 0 indicates non-panic
panic_count = sum(predictions)
if panic_count > len(predictions) / 2:
    print("The video is predicted to be panic-inducing.")
else:
    print("The video is predicted to be not panic-inducing.")
