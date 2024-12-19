import os
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import time

# Dynamically resolve the path to the models directory
current_dir = os.path.dirname(os.path.abspath(__file__))  # Get current file's directory
models_dir = os.path.join(current_dir, "../models")  # Models directory at the same level as vehicle_assistant

# Load all trained models
primary_model = load_model(os.path.join(models_dir, "primary_classifier_01.keras"))
navigation_model = load_model(os.path.join(models_dir, "navigation_classifier_01.keras"))
volume_model = load_model(os.path.join(models_dir, "volume_classifier_01.keras"))

# Define class labels for secondary classifiers
navigation_classes = ["menu_next", "menu_previous"]
volume_classes = ["volume_increase", "volume_decrease"]

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Parameters
sequence_length = 10
frame_skip = 2
confidence_threshold = 0.80
prediction_gap = 1.5


class GestureDetector:
    def __init__(self):
        self.sequence = []
        self.last_prediction_time = 0
        self.volume = 0
        self.menu_state = 0
        self.running = False  # Tracks whether the detection loop is running
        self.cap = None  # Webcam capture object

    def detect_gestures(self):
        """
        Continuously process webcam frames and detect gestures.
        Yields the current volume and menu states dynamically.
        """
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Error: Unable to access the webcam.")

        self.running = True
        frame_count = 0

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.append([lm.x, lm.y, lm.z])
                    landmarks = np.array(landmarks).flatten()

                    # Add to sequence
                    self.sequence.append(landmarks)
                    if len(self.sequence) > sequence_length:
                        self.sequence.pop(0)

                    # Predict gestures when sequence is full
                    current_time = time.time()
                    if len(self.sequence) == sequence_length and current_time - self.last_prediction_time >= prediction_gap:
                        input_data = np.expand_dims(self.sequence, axis=0)

                        # Step 1: Primary Classification
                        primary_prediction = primary_model.predict(input_data, verbose=0)
                        primary_class = np.argmax(primary_prediction)
                        primary_confidence = np.max(primary_prediction)

                        if primary_confidence >= confidence_threshold:
                            if primary_class == 0:  # Navigation
                                nav_prediction = navigation_model.predict(input_data, verbose=0)
                                nav_class = np.argmax(nav_prediction)
                                nav_confidence = np.max(nav_prediction)

                                if nav_confidence >= confidence_threshold:
                                    predicted_label = navigation_classes[nav_class]
                                    if predicted_label == "menu_next":
                                        self.menu_state = (self.menu_state + 1) % 3
                                    elif predicted_label == "menu_previous":
                                        self.menu_state = (self.menu_state - 1) % 3

                            elif primary_class == 1:  # Volume Control
                                vol_prediction = volume_model.predict(input_data, verbose=0)
                                vol_class = np.argmax(vol_prediction)
                                vol_confidence = np.max(vol_prediction)

                                if vol_confidence >= confidence_threshold:
                                    predicted_label = volume_classes[vol_class]
                                    if predicted_label == "volume_increase":
                                        self.volume = min(self.volume + 10, 100)
                                    elif predicted_label == "volume_decrease":
                                        self.volume = max(self.volume - 10, 0)

                        self.last_prediction_time = current_time

            # Yield the current state
            yield {"Volume": self.volume, "Menu": f"menu_{self.menu_state + 1}"}

        self.release_resources()

    def stop(self):
        """
        Stop the gesture detection system.
        """
        self.running = False

    def release_resources(self):
        """
        Release webcam and other resources.
        """
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
