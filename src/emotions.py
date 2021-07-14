import argparse
import cv2
import os
import imageio
from pathlib import Path
import time

from models.face_plusplus import FacePlusPlusModel
from models.open_source import OpenSourceModel

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

VIDEO_FILENAME = (
    Path("videos") / "Les moments forts du dernier débat des chefs_360P.mp4"
)


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

    if cv2.waitKey(1) & 0xFF == ord("q"):
        return


def get_model(model_type):
    if model_type == "face-pp":
        model = FacePlusPlusModel()
    elif model_type == "open-source":
        model = OpenSourceModel()

    return model


def run(model_type, model_fps):
    faces = []

    model = get_model(model_type)
    video_reader = imageio.get_reader(VIDEO_FILENAME, "ffmpeg")

    video_fps = video_reader._meta["fps"]

    for i, frame in enumerate(video_reader):

        time0 = time.time()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if i % video_fps // model_fps == 0:
            faces = model.apply(frame)

        draw_faces(frame, faces)
        time1 = time.time()

        delay = max(0, 1 / video_fps - (time1 - time0))
        time.sleep(delay)

        if i > 10000:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-type", type=str, default="open-source")
    parser.add_argument("-f", "--model-fps", type=float, default=1)
    args = parser.parse_args()

    run(args.model_type, args.model_fps)
