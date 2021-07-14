import numpy as np
import cv2
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D

CURRENT_FOLDER = Path(__file__).resolve().parent

DEFAULT_FACE_DETECTOR_PATH = CURRENT_FOLDER / "haarcascade_frontalface_default.xml"
DEFAULT_EMOTION_CLASSIFIER_PATH = CURRENT_FOLDER / "model.h5"


class OpenSourceModel:

    # dictionary which assigns each label an emotion (alphabetical order)
    emotions = {
        0: "Angry",
        1: "Disgusted",
        2: "Fearful",
        3: "Happy",
        4: "Neutral",
        5: "Sad",
        6: "Surprised",
    }

    def __init__(
        self,
        face_detector_path=str(DEFAULT_FACE_DETECTOR_PATH),
        emotion_classifier_path=DEFAULT_EMOTION_CLASSIFIER_PATH,
    ):
        self._face_detector = cv2.CascadeClassifier(face_detector_path)
        self._emotion_classifier = self._load_emotion_classifier(
            emotion_classifier_path
        )

    def _load_emotion_classifier(self, model_filename):
        # Create the model
        model = Sequential()

        model.add(
            Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(48, 48, 1))
        )
        model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation="softmax"))

        model.load_weights(model_filename)

        return model

    def apply(self, frame):
        # Find haar cascade to draw bounding box around face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rectangles = self._face_detector.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5
        )

        faces = []
        for (x, y, w, h) in face_rectangles:
            # Resize face crop images 48x48
            cropped_face = cv2.resize(gray[y : y + h, x : x + w], (48, 48))
            cropped_face = np.expand_dims(np.expand_dims(cropped_face, -1), 0)

            # Filter out false positives (faces in bg + tie)
            if cropped_face.mean() < 40 or y + h > 200:
                continue

            prediction = self._emotion_classifier.predict(cropped_face)
            prediction = prediction.tolist()[0]
            emotions = {self.emotions[i]: pred for i, pred in enumerate(prediction)}

            faces.append((x, y, w, h, emotions))

        return faces
