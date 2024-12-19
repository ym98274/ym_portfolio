import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
import os

# Load the trained model and scaler
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '../models/pose_model.h5')
scaler_path = os.path.join(current_dir, '../models/pose_scaler.pkl')

model = tf.keras.models.load_model(model_path)
scaler = joblib.load(scaler_path)

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

def get_all_landmarks(landmarks, frame_shape):
    """Extract all face and iris landmarks."""
    landmark_list = []
    for landmark in landmarks:
        x = landmark.x * frame_shape[1]  # Convert x to pixel space
        y = landmark.y * frame_shape[0]  # Convert y to pixel space
        landmark_list.extend([x, y])  # Add both x and y to the list
    return landmark_list

def process_frame_for_prediction(frame):
    """
    Process a single frame to make predictions without rendering the frame.
    Returns the prediction: "Looking at screen" or "Not looking at screen".
    """
    # Convert the frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Get landmarks
            all_landmarks = get_all_landmarks(face_landmarks.landmark, frame.shape)

            # Scale landmarks for prediction
            try:
                scaled_landmarks = scaler.transform(np.array(all_landmarks).reshape(1, -1))
            except ValueError:
                return "Error: Mismatched input size"

            # Make predictions
            prediction = model.predict(scaled_landmarks)
            predicted_class = 1 if prediction[0][0] > 0.5 else 0
            return "Looking at screen" if predicted_class == 1 else "Not looking at screen"

    return "No face detected"