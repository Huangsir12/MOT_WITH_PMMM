# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import argparse
import cv2
import numpy as np
from functools import partial
from pathlib import Path

import torch

import sys
sys.path.append("/root/autodl-tmp/boxmot")

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS
from boxmot.utils.checks import RequirementsChecker
from tracking.detectors import (get_yolo_inferer, default_imgsz,
                                is_ultralytics_model, is_yolox_model)
from tracking.pmmm_scripts.trackreid_pmmm import TrackReid_PMMM
from tracking.pmmm_scripts.scripts import renew_track_ids

checker = RequirementsChecker()
checker.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git', ))  # install

from ultralytics import YOLO
# from ultralytics.utils.plotting import Annotator, colors
# from ultralytics.data.utils import VID_FORMATS
# from ultralytics.utils.plotting import save_one_box

import cv2
import numpy as np
import os
import argparse


def on_predict_start(predictor, persist=False):
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    """

    assert predictor.custom_args.tracking_method in TRACKERS, \
        f"'{predictor.custom_args.tracking_method}' is not supported. Supported ones are {TRACKERS}"

    tracking_config = TRACKER_CONFIGS / (predictor.custom_args.tracking_method + '.yaml')
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.custom_args.tracking_method,
            tracking_config,
            predictor.custom_args.reid_model,
            predictor.device,
            predictor.custom_args.half,
            predictor.custom_args.per_class
        )
        # motion only modeles do not have
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)
        print(f"tracker method:{predictor.custom_args.tracking_method} tracker id:{i}")

    predictor.trackers = trackers


@torch.no_grad()
def run(args):

    if args.imgsz is None:
        args.imgsz = default_imgsz(args.yolo_model)

    # yolo = YOLO(
    #     args.yolo_model if is_ultralytics_model(args.yolo_model)
    #     else 'yolov8n.pt',
    # )
    yolo = YOLO(args.yolo_model)
    results = yolo.track(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        agnostic_nms=args.agnostic_nms,
        show=False,
        stream=True,
        device=args.device,
        show_conf=args.show_conf,
        save_txt=args.save_txt,
        show_labels=args.show_labels,
        save=args.save,
        verbose=args.verbose,
        exist_ok=args.exist_ok,
        project=args.project,
        name=args.name,
        classes=args.classes,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
        line_width=args.line_width
    )
    yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))

    track_reid = TrackReid_PMMM(args.project, args.reid_config_file, txt_name=None)

    output_path = os.path.join(track_reid.output_dir, os.path.basename(args.source))
    # Define the codec and create a VideoWriter object to save the annotated frames
    # frame_width = 1920
    # frame_height = 1080
    frame_width = 2560
    frame_height = 1440
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))  # 30.0 is the frame rate

    frame_count = 0
    matched = []
    track_ids_before_all = []
    abnormal_removed = []
    frames, track_results_boxes, track_results_ids = [], [], []

    # store custom args in predictor
    yolo.predictor.custom_args = args
    for r in results:
        frame_count += 1      # Increment the frame counter
        # Get the boxes and track IDs
        frame = r.orig_img
        frames.append(frame)
        boxes = r.boxes.xyxy.cpu()
        track_ids = r.boxes.id.int().cpu().tolist()
        print(f"track_ids: {track_ids}")
        track_results_boxes.append(boxes)
        track_results_ids.append(track_ids)
        frame_matched, abnormal_removed = track_reid.processing_to_reid(frame, boxes, track_ids, frame_count,
                                                    frame_width, frame_height, track_ids_before_all, abnormal_removed)
            
        track_ids_before_all.extend(track_ids)
        if frame_matched:
            matched.append(frame_matched)
    
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
    
    out.release()


def parse_opt():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / 'yolov10x.pt',
                        help='yolo model path')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x1_0_msmt17.pt',
                        help='reid model path')
    parser.add_argument('--tracking-method', type=str, default='botsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack, imprassoc, boosttrack')
    parser.add_argument('--source', type=str, default='/root/autodl-tmp/boxmot/data/video_datasets/door1.mp4',
                        help='file/dir/URL/glob, 0 for webcam')
    # parser.add_argument('--source', type=str, default='/root/autodl-tmp/boxmot/data/datasets/MOT17/train/MOT17-02/img1',
    #                     help='file/dir/URL/glob, 0 for webcam')
    
    parser.add_argument('--reid_config_file', type=str, default='bpbreid/configs/bpbreid/bpbreid_inference.yaml', 
                        help='path to config file')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=None,
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true',
                        help='display tracking video results')                 
    parser.add_argument('--save', action='store_true',
                        help='save video tracking results')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, default=[0],
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track_reid',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--show-labels', action='store_false',
                        help='either show all or only bboxes')
    parser.add_argument('--show-conf', action='store_false',
                        help='hide confidences when show')
    parser.add_argument('--show-trajectories', action='store_true',
                        help='show confidences')
    parser.add_argument('--save-txt', action='store_true',
                        help='save tracking results in a txt file')
    parser.add_argument('--save-id-crops', action='store_true',
                        help='save each crop to its respective id folder')
    parser.add_argument('--line-width', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--per-class', default=False, action='store_true',
                        help='not mix up classes when tracking')
    parser.add_argument('--verbose', default=True, action='store_true',
                        help='print results per frame')
    parser.add_argument('--agnostic-nms', default=False, action='store_true',
                        help='class-agnostic NMS')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(opt)
