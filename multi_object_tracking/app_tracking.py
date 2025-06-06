from collections import defaultdict
import cv2
import numpy as np
import os
import argparse
import yaml
from ultralytics import YOLO
from datetime import datetime
from bpbreid.torchreid.scripts.reID_app import inference_reid_init
from multi_object_tracking.scripts import *


__detector_models__ = {
    "yolov8n": "./pretrained_models/tracking_models/yolov8n.pt",
    "yolov8l": "./pretrained_models/tracking_models/yolov8l.pt",
    "yolov9c": "./pretrained_models/tracking_models/yolov9c.pt",
    "yolov9e": "./pretrained_models/tracking_models/yolov9e.pt",
    "yolov10x": "./pretrained_models/tracking_models/yolov10x.pt",
    "yolov11l": "./pretrained_models/tracking_models/yolov11l.pt",
    "yolov11x": "./pretrained_models/tracking_models/yolov11x.pt"
}
__tracking_models__ = {
    "ByteTrack": "bytetrack.yaml",
    "BotSort": "botsort.yaml",
    "StrongSort": "custom_tracker.yaml",
}

__reid_models__ = {
    "osnet": "osnet_x1_0",
    "resnet50": "resnet50",
    "resnet101": "resnet101",
    "bpbreid": "bpbreid",
}


class TrackObjects:
    def __init__(self, detect_model, tracking_model, reid_congig_file=None, output_dir="track_results"):

        avai_detect_models = list(__detector_models__.keys())
        if detect_model not in avai_detect_models:
            raise ValueError(
                'Invalid model name. Received "{}", '
                'but expected to be one of {}'.format(detect_model, avai_detect_models)
            )

        avai_tracking_models = list(__tracking_models__.keys())
        if tracking_model not in avai_tracking_models:
            raise ValueError(
                'Invalid model name. Received "{}", '
                'but expected to be one of {}'.format(tracking_model, avai_tracking_models)
            )
        
        self.model_path = __detector_models__[detect_model]
        self.tracker = __tracking_models__[tracking_model]
        self.model = YOLO(self.model_path)
        
        self.output_dir = os.path.join(output_dir, datetime.now().strftime("%m%d-%H%M"))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        if reid_congig_file:
            with open(reid_congig_file, 'r') as file:
                doc = yaml.safe_load(file)

            doc['inference']['dataset_folder'] = self.output_dir

            with open(reid_congig_file, 'w') as file:
                yaml.safe_dump(doc, file, default_flow_style=None)

            self.reid_inference = inference_reid_init(reid_congig_file)
        else:
            self.reid_inference = None

        self.clops_output_dir = os.path.join(self.output_dir, "clops")
        if not os.path.exists(self.clops_output_dir):
            os.makedirs(self.clops_output_dir, exist_ok=True)

        self.gallery_dataset_dir = os.path.join(self.output_dir, "gallery")
        if not os.path.exists(self.gallery_dataset_dir):
            os.makedirs(self.gallery_dataset_dir, exist_ok=True)

        self.query_dataset_dir = os.path.join(self.output_dir, "query")
        if not os.path.exists(self.query_dataset_dir):
            os.makedirs(self.query_dataset_dir, exist_ok=True)

        self.advented, self.disappeared = {}, {}
        self.boxes_before, self.track_ids_before = [], []
    

    def processing_to_reid(self, frame, boxes, track_ids, frame_count, frame_width, frame_height, 
                           track_ids_before_all, abnormal_removed):
        matched_id = {}
        # save the boxes to the output directory
        for box, track_id in zip(boxes, track_ids):
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box[:4])
            # print(f"track_id: {track_id}, x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
            crop = frame[y1:y2, x1:x2]  # Crop the detected object

            # Save the cropped image with the desired naming convention
            crop_name = f"frame_{frame_count}_ID_{track_id}.jpg"
            crop_path = os.path.join(self.clops_output_dir, crop_name)
            cv2.imwrite(crop_path, crop)

        if frame_count > 1:
            if frame_count % 100 == 0:
                clear_gallery_dataset_frame(self.gallery_dataset_dir, frame_count)
            # Get abnormal track ids
            frame_abnormal_added, self.advented = get_abnormal_added_track_ids(self.advented, boxes, track_ids, 
                                                                track_ids_before_all, 
                                                                frame_width, frame_height)    
            frame_abnormal_removed, self.disappeared = get_abnormal_removed_track_ids(self.disappeared, 
                                                                    track_ids, 
                                                                    self.boxes_before, self.track_ids_before, 
                                                                    frame_width, frame_height)
            
            # add the abnormal removed track ids to the gallery dataset
            if frame_abnormal_removed:
                haven_added_ids = add_to_gallery_dataset(self.clops_output_dir, self.gallery_dataset_dir, frame_abnormal_removed)
                abnormal_removed.extend(haven_added_ids)
                abnormal_removed = list(set(abnormal_removed))
                print(f"removed_track_id: {haven_added_ids}")
            if not abnormal_removed:
                frame_abnormal_added = []
            # add the abnormal added track ids to the query dataset
            if frame_abnormal_added:
                abnormal_removed, clear_ids = clear_gallery_dataset(self.gallery_dataset_dir, frame_count, abnormal_removed)
                pids_counts = add_to_query_dataset(self.clops_output_dir, self.query_dataset_dir, frame_abnormal_added)
                print(f"clear_ids: {clear_ids}")
                print(f"added_track_id: {pids_counts}")
            gallery_counts = get_pictures_counts(self.gallery_dataset_dir)
            query_counts = get_pictures_counts(self.query_dataset_dir)
            # start to reID
            if query_counts > 0:
                # reID the gallery and query dataset
                print("gallery_counts: ", gallery_counts)
                print("query_counts: ", query_counts)
                print("query_ids: ", frame_abnormal_added)
                print("gallery_ids: ", abnormal_removed)
                
                if gallery_counts > 0:
                    matched_id = self.reid_inference.run_tracking(pids_counts)
                    matched_id_values = list(matched_id.values())
                    if matched_id_values:
                        print(matched_id_values)
                        for item in matched_id_values:
                            abnormal_removed.remove(item)
                        remove_from_ReID_dataset(self.gallery_dataset_dir, matched_id_values)
                remove_from_ReID_dataset(self.query_dataset_dir, frame_abnormal_added)
            
        self.boxes_before, self.track_ids_before = boxes, track_ids

        return matched_id, abnormal_removed
    
    
    def track_objects(self, video_input, draw_tracks_line, conf=0.3, iou=0.5):
        
        video_path = video_input[0] if isinstance(video_input, (tuple, list)) else video_input
        
        cap = cv2.VideoCapture(video_path)
        # Get the width and height of the video frames
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create a VideoWriter object to save the annotated frames
        output_path = os.path.join(self.output_dir, os.path.basename(video_path))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))  # 30.0 is the frame rate

        # Store the track history
        track_history = defaultdict(lambda: [])
        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()
            if success:
                results = self.model.track(frame, persist=False, tracker=self.tracker, classes=0)
                # Get the boxes and track IDs
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                # Visualize the results on the frame
                annotated_frame = results[0].plot()
                # Plot the tracks
                if draw_tracks_line:
                    for box, track_id in zip(boxes, track_ids):
                        x1, y1, x2, y2 = box
                        track = track_history[track_id]
                        track.append((float(x1+x2), float(y2)))  # x, y center point
                        if len(track) > 30:  # retain 90 tracks for 90 frames
                            track.pop(0)

                        # Draw the tracking lines
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
                    
                out.write(annotated_frame)
                
                # # Display the annotated frame
                # cv2.imshow("YOLOv8 Tracking", annotated_frame)
                # Break the loop if 'q' is pressed
                # if cv2.waitKey(1) & 0xFF == ord("q"):
                #     break
            else:
                # Break the loop if the end of the video is reached
                break

        # Release the video capture object and close the display window
        cap.release()
        cv2.destroyAllWindows()
    
    
    def track_objects_with_reid(self, video_input, draw_tracks_line, conf, iou):
        video_path = video_input[0] if isinstance(video_input, (tuple, list)) else video_input
        cap = cv2.VideoCapture(video_path)
        # Get the width and height of the video frames
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create a VideoWriter object to save the annotated frames
        output_path = os.path.join(self.output_dir, os.path.basename(video_path))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))  # 30.0 is the frame rate

        frame_count = 0
        matched = []
        track_ids_before_all = []
        abnormal_removed = []
        frames, track_results_boxes, track_results_ids = [], [], []       
              
        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()
            if success:
                frame_count += 1      # Increment the frame counter
                frames.append(frame)
                # Run YOLO11 tracking on the frame, persisting tracks between frames
                # results = model.track(frame, tracker=tracker, persist=True, classes=0, conf=conf, iou=iou)
                results = self.model.track(frame, persist=True, tracker=self.tracker, classes=0)

                # Get the boxes and track IDs
                boxes = results[0].boxes.xyxy.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                print(f"track_ids: {track_ids}")
                track_results_boxes.append(boxes)
                track_results_ids.append(track_ids)

                frame_matched, abnormal_removed = self.processing_to_reid(frame, boxes, track_ids, frame_count,
                                                        frame_width, frame_height, track_ids_before_all, abnormal_removed)
                
                track_ids_before_all.extend(track_ids)
                if frame_matched:
                    matched.append(frame_matched)
                
            else:
                break   # Break the loop if the end of the video is reached
        # Release the video capture object and close the display window
        print(matched)
        renew_track_results_ids = renew_track_ids(track_results_ids, matched)
        # Visualize and save the results
        for frame, boxes, ids in zip(frames, track_results_boxes, renew_track_results_ids):
            for box, id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                label = f"id:{id} person"
                cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
                
            out.write(frame)    
        cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--detect_model', type=str, default="yolov10x", help='select from yolov8n, yolov8l, yolov9c, yolov9e, yolov10x,  yolov11l, yolov11x')
    parser.add_argument('--video_input', type=str, default="../video_datasets/door1.mp4", help='source')  # file/folder, 0 for webcam
    parser.add_argument('--tracking_model', type=str, default="BotSort", help='select from BotSort, StrongSort, ByteTrack')
    parser.add_argument('--output_dir', type=str, default="track_results", help='output folder')  # output folder
    parser.add_argument('--draw_tracks_line', type=bool, default=False, help='if draw tracks line')
    parser.add_argument('--conf', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5, help='iou threshold')

    opt = parser.parse_args()
    trackobjects = TrackObjects(opt.detect_model, opt.tracking_model)
    trackobjects.track_objects(opt.video_input, 
                               opt.draw_tracks_line, 
                               opt.conf, 
                               opt.iou)