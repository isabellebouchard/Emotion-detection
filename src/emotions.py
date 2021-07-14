import numpy as np
import argparse
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import os
import base64
import imageio
from pathlib import Path
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

VIDEO_FILENAME = (
    Path("videos") / "Les moments forts du dernier débat des chefs_360P.mp4"
)


class OpenSourceModel:

    MODEL_FILENAME = "model.h5"
    CASCADE_FILENAME = "haarcascade_frontalface_default.xml"

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

    def __init__(self):
        self._emotion_classifier = self._load_model(self.MODEL_FILENAME)
        self._face_detector = cv2.CascadeClassifier(self.CASCADE_FILENAME)

    def _load_model(self, model_filename):
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


def draw_faces(frame, faces):
    for (x, y, w, h, prediction) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Format text and colors
        for i, (emotion, pred) in enumerate(prediction.items()):
            pos = (x, y + h + 10 * i)
            color = (0, 0, 255) if pred > 0.2 else (255, 255, 255)
            text = "{}: {:.4f}".format(emotion, pred)
            cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.imshow("Video", cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))

    time.sleep(0.15)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        return


def get_model(model_type):
    if model_type == "face-pp":
        pass
    elif model_type == "open-source":
        model = OpenSourceModel()

    return model


def run(model_type):
    model = get_model(model_type)
    video_reader = imageio.get_reader(VIDEO_FILENAME, "ffmpeg")
    for i, frame in enumerate(video_reader):
        if i % 5 != 0:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        faces = model.apply(frame)
        draw_faces(frame, faces)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-type", type=str, default="open-source")
    args = parser.parse_args()

    run(args.model_type)
