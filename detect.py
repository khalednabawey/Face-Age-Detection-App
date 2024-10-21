import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

print("Script started")

# Suppress TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

print("Initializing MediaPipe Face Detection")
# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

print("Loading the age detection model")
# Load the age detection model
model_path = os.path.join(os.getcwd(), "model/Age-Detector-Model-95.h5")
print(f"Model path: {model_path}")
age_model = load_model(model_path)

print("Model loaded successfully")

# Define age groups
age_group = {
    0: "YOUNG",
    1: "MIDDLE",
    2: "OLD"
}


def preprocess_image(image):
    image_resized = cv2.resize(
        image, (180, 180), interpolation=cv2.INTER_LINEAR)
    image_normalized = image_resized / 255.0
    return np.expand_dims(image_normalized, axis=0)


def predict_age(image):
    processed_img = preprocess_image(image)
    pred_probas = age_model.predict(processed_img)
    pred_class = pred_probas.argmax()
    return age_group[pred_class]


print("Initializing webcam")
# Initialize webcam
cap = cv2.VideoCapture(0)

print("Starting main loop")
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Convert the BGR image to RGB and process it with MediaPipe Face Detection
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image_rgb)

            if results.detections:
                for detection in results.detections:
                    # Get bounding box coordinates
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = image.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                        int(bboxC.width * iw), int(bboxC.height * ih)

                    # Extract face ROI
                    face_roi = image[y:y+h, x:x+w]

                    if face_roi.size != 0:  # Check if ROI is not empty
                        # Predict age group
                        age_prediction = predict_age(face_roi)

                        # Draw bounding box and age prediction
                        cv2.rectangle(image, (x, y), (x+w, y+h),
                                      (0, 255, 0), 2)
                        cv2.putText(image, age_prediction, (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Display the resulting frame
            cv2.imshow('Real-time Age Detection', image)

            # Break the loop if 'q' is pressed or the window is closed
            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Real-time Age Detection', cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        print("Releasing webcam and closing windows")
        cap.release()
        cv2.destroyAllWindows()

print("Script ended")
