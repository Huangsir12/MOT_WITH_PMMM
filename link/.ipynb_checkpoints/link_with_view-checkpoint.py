
from ultralytics import YOLO


# Load a classify model
model = YOLO("autodl-tmp/MOT_WITH_PMMM/runs/classify/train2/weights/best.pt")  # load a custom model
# Predict with the model
view_results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image