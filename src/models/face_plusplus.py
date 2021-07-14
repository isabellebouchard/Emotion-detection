import os
import base64
import requests
from PIL import Image
from io import BytesIO


def _get_base64_encoded_image(image_data):
    image = Image.fromarray(image_data)

    # Create a buffer to save the image and read from
    buff = BytesIO()
    image.save(buff, format="JPEG")
    encoded_image = base64.b64encode(buff.getvalue()).decode("utf-8")

    return encoded_image


class FacePlusPlusModel:

    api_url = "https://api-us.faceplusplus.com/facepp/v3/detect"

    def __init__(self):
        self._api_key = os.environ["FACE_API_KEY"]
        self._api_secret = os.environ["FACE_API_SECRET"]

    def apply(self, frame):
        encoded_image = _get_base64_encoded_image(frame)

        request_data = {
            "api_key": self._api_key,
            "api_secret": self._api_secret,
            "image_base64": encoded_image,
            "return_attributes": "emotion",
        }

        response = requests.post(self.api_url, data=request_data)

        if response.status_code != 200:
            raise (f"FacePlusPlus API Error {response.status_code}: {response.reason}")

        response_data = response.json()

        faces = []
        for face in response_data["faces"]:
            x = face["face_rectangle"]["left"]
            y = face["face_rectangle"]["top"]
            w = face["face_rectangle"]["width"]
            h = face["face_rectangle"]["height"]

            emotions = face["attributes"]["emotion"]
            emotions = {e: p / 100 for e, p in emotions.items()}

            faces.append((x, y, w, h, emotions))

        return faces
