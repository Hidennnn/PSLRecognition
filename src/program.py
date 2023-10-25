import cv2
import joblib
import numpy as np
from PIL import ImageFont
from tensorflow.keras.models import load_model

from src.modules.custom_exceptions import PoseNotDetectedError, RightHandNotDetectedError, LeftHandNotDetectedError, \
    VectorIsNoneError
from src.modules.preprocessing import make_vector_of_points, make_distance
from src.modules.utils import rescale_img

# if you want to quit, you need to click "q"

# =============================================== OPTIONS =============================================================
CAMERA_INDEX = 0  # your camera index
PATH_TO_LABELS = r"neural_networks\train_test_data\labels.txt" # path to txt with labels
PATH_TO_MODELS = r"neural_networks\models"
PATH_TO_FONTS = r"fonts"
RESCALE_FACTOR = 50  # what percentage of the size of the original frame will be the frame after rescaling


# =============================================== PROGRAM =============================================================
# load things to prediction
scaler = joblib.load(f"{PATH_TO_MODELS}\\scaler.save")
model = load_model(f"{PATH_TO_MODELS}\\final_model.h5")

# based on labels proper caption will be chosen
labels = {}
with open(PATH_TO_LABELS, "r") as file_labels:
    count = 0
    for index, line in enumerate(file_labels):
        labels[count] = line[:-1]
        count += 1

# change labels to shorter captions
labels[21] = "ty"  # means "you" in Polish
labels[22] = "on/ona"  # means "he/she" in Polish

# get camera input
camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    raise "Camera is not available"

frame_width = int(camera.get(3))
frame_height = int(camera.get(4))

font = ImageFont.truetype(f"{PATH_TO_FONTS}\\arial.ttf", 80)
text = ""

while True:
    ret, frame = camera.read()

    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    frame = rescale_img(frame, RESCALE_FACTOR)
    vector = None
    try:
        vector = make_vector_of_points(frame)
    except (PoseNotDetectedError, RightHandNotDetectedError, LeftHandNotDetectedError):
        pass

    frame = rescale_img(frame, 50)
    try:
        dist = make_distance(vector)
        dist = scaler.transform([dist])
        predict = model.predict(dist)
        index = np.where(predict == np.amax(predict))[1][0]
        text = labels[index]
    except VectorIsNoneError:
        pass

    # Add text
    cv2.putText(frame, text, org=(100, 900), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=4, color=(255, 255, 255),
                thickness=6, lineType=cv2.LINE_AA)
    cv2.putText(frame, text, org=(100, 900), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=4, color=(0, 0, 0),
                thickness=4, lineType=cv2.LINE_AA)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
