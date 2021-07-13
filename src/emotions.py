import numpy as np
import argparse
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import os
import imageio
from pathlib import Path
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

VIDEO_FILENAME = Path("videos") / "Les moments forts du dernier débat des chefs_360P.mp4"
MODEL_FILENAME = "model.h5"
CASCADE_FILENAME = "haarcascade_frontalface_default.xml"

# dictionary which assigns each label an emotion (alphabetical order)
EMOTIONS = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
}

def load_model(model_filename):
    # Create the model
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    model.load_weights(model_filename)

    return model


def apply_model(model, facecasc, frame):
    # Find haar cascade to draw bounding box around face
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = cv2.resize(gray[y: y + h, x: x + w], (48, 48))

        # Filter out false positives (faces in bg + tie)
        if face.mean() < 40 or y + h > 200:
            continue

        # Resize face crop images 48x48
        cropped_img = np.expand_dims(np.expand_dims(face, -1), 0)

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Call the model
        prediction = model.predict(cropped_img).tolist()[0]

        # Format text and colors
        for i, pred in enumerate(prediction):
            pos = (x , y + h + 10 * i)
            color = (0, 0, 255) if pred > 0.2 else (255, 255, 255)

            text = EMOTIONS[i]
            text += ": "
            text += "{:.4f}".format(pred)

            cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))

    time.sleep(0.15)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return


def run():
    model = load_model(MODEL_FILENAME)
    facecasc = cv2.CascadeClassifier(CASCADE_FILENAME)

    video_reader = imageio.get_reader(VIDEO_FILENAME,  'ffmpeg')
    for i, frame in enumerate(video_reader):
        if i % 5 != 0:
            continue
        apply_model(model, facecasc, frame)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
