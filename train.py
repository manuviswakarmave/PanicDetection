import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import joblib

# Load motion information images
motion_images_folder = 'motion_images'
X_motion = []
y_labels = []

for filename in os.listdir(motion_images_folder):
    if filename.endswith('.jpg'):
        motion_image = cv2.imread(os.path.join(motion_images_folder, filename), cv2.IMREAD_GRAYSCALE)
        # Resize the images to match expected input shape
        motion_image = cv2.resize(motion_image, (640, 360))
        # Expand dimensions to add channel dimension
        motion_image = np.expand_dims(motion_image, axis=-1)
        X_motion.append(motion_image)
        if "panic" in filename:
            y_labels.append(1)
        else:
            y_labels.append(0)

X_motion = np.array(X_motion)
y_labels = np.array(y_labels)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_motion, y_labels, test_size=0.2, random_state=42)

# Feature extraction using a CNN model
input_shape = X_train[0].shape

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Extract features using the trained CNN model
cnn_features_train = model.predict(X_train)
cnn_features_test = model.predict(X_test)

# Train SVM classifier
svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(cnn_features_train, y_train)

# Evaluate SVM classifier
svm_predictions = svm_classifier.predict(cnn_features_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print("SVM Accuracy:", svm_accuracy)

# Save models
joblib.dump(svm_classifier, 'svm_model.joblib')
model.save('cnn_model.h5')
