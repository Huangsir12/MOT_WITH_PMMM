# Run inference on 'bus.jpg'
# yolo predict model=yolo11n.pt source='https://ultralytics.com/images/bus.jpg'


# Build a new model from YAML and start training from scratch
# yolo detect train data=coco8.yaml model=yolo11n.yaml epochs=100 imgsz=640

# Start training from a pretrained *.pt model
# yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640

# Build a new model from YAML, transfer pretrained weights to it and start training
# yolo detect train data=coco8.yaml model=yolo11n.yaml pretrained=yolo11n.pt epochs=100 imgsz=640