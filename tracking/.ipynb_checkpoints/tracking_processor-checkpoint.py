"""
Tracking processor module
Handles single camera tracking with PMMM and result storage
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import json

import sys
sys.path.append("/root/autodl-tmp/MOT_WITH_PMMM")

from boxmot.tracker_zoo import create_tracker
from boxmot.utils import WEIGHTS, TRACKER_CONFIGS
from boxmot.utils.torch_utils import select_device
from boxmot.appearance.reid.auto_backend import ReidAutoBackend
from ultralytics import YOLO
from ultralytics.data.loaders import LoadImagesAndVideos
from ultralytics.utils import ops

from tracking.pmmm_scripts.trackreid_pmmm import TrackReid_PMMM
from tracking.pmmm_scripts.scripts import renew_track_ids
from tracking.database.db_manager import DatabaseManager
from tracking.utils import write_mot_results


class TrackingProcessor:
    """Processes single camera tracking with PMMM"""

    def __init__(self,
                 yolo_model_path: str = None,
                 reid_model_path: str = None,
                 tracking_method: str = 'botsort',
                 reid_config_file: str = 'bpbreid/configs/bpbreid/bpbreid_inference.yaml',
                 device: str = '0',
                 conf_threshold: float = 0.5,
                 iou_threshold: float = 0.7,
                 db_manager: Optional[DatabaseManager] = None):
        """
        Initialize tracking processor

        Args:
            yolo_model_path: Path to YOLO detection model
            reid_model_path: Path to ReID model
            tracking_method: Tracking algorithm (botsort, bytetrack, etc.)
            reid_config_file: Path to ReID configuration file
            device: Device for inference ('0', 'cpu', etc.)
            conf_threshold: Detection confidence threshold
            iou_threshold: IOU threshold for NMS
            db_manager: Database manager instance
        """
        self.yolo_model_path = yolo_model_path or str(WEIGHTS / 'yolov10x_trained_best1.pt')
        self.reid_model_path = reid_model_path or str(WEIGHTS / 'osnet_x1_0_msmt17.pt')
        self.tracking_method = tracking_method
        self.reid_config_file = reid_config_file
        self.device = select_device(device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.db_manager = db_manager

        # Initialize models
        self.yolo_model = YOLO(self.yolo_model_path)
        self.tracker = None
        self.track_reid = None

    def _init_tracker(self):
        """Initialize tracker"""
        self.tracker = create_tracker(
            self.tracking_method,
            TRACKER_CONFIGS / (self.tracking_method + '.yaml'),
            Path(self.reid_model_path).with_suffix('.pt'),
            self.device,
            False,
            False,
            None
        )

    def process_video(self,
                     video_path: str,
                     output_dir: str,
                     video_id: str,
                     scenario_name: str,
                     tracking_batch: int,
                     save_video: bool = True,
                     save_crops: bool = True) -> Dict:
        """
        Process a single video for tracking

        Args:
            video_path: Path to input video
            output_dir: Base output directory
            video_id: Video ID from database
            scenario_name: Scenario name
            tracking_batch: Tracking batch number
            save_video: Whether to save annotated video
            save_crops: Whether to save detection crops

        Returns:
            Dictionary containing tracking results and statistics
        """
        # Create output directories
        video_name = Path(video_path).stem
        results_dir = Path(output_dir) / scenario_name / f"batch_{tracking_batch:04d}" / video_name
        results_dir.mkdir(parents=True, exist_ok=True)

        crops_dir = results_dir / "crops"
        if save_crops:
            crops_dir.mkdir(exist_ok=True)

        # Initialize tracker and ReID
        self._init_tracker()
        self.track_reid = TrackReid_PMMM(
            str(results_dir),
            self.reid_config_file,
            txt_name=None
        )

        # Get video info
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Prepare video writer
        video_writer = None
        if save_video:
            output_video_path = results_dir / f"{video_name}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(output_video_path),
                fourcc,
                fps,
                (frame_width, frame_height)
            )

        # Run detection
        print(f"\nProcessing video: {video_path}")
        print(f"Output directory: {results_dir}")

        results = self.yolo_model(
            source=video_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            stream=True,
            device=self.device,
            verbose=False,
            classes=[0]  # Person class
        )

        # Process frames
        matched = []
        track_ids_before_all = []
        abnormal_removed = []
        frames, track_results_boxes, track_results_ids = [], [], []
        track_results_conf, track_results_cls = [], []
        all_mot_results = []

        frame_idx = 0
        for r in tqdm(results, desc="Processing frames", total=total_frames):
            frame_idx += 1
            frame = r.orig_img

            if r.boxes is None or len(r.boxes) == 0:
                continue

            # Get detections
            dets = torch.cat([
                r.boxes.xyxy,
                r.boxes.conf.unsqueeze(1),
                r.boxes.cls.unsqueeze(1)
            ], dim=1).cpu().numpy()

            # Get embeddings (placeholder - would use actual ReID model)
            embs = np.zeros((len(dets), 512))  # Placeholder embeddings

            # Update tracker
            tracks = self.tracker.update(dets, frame, embs, frame_width, frame_height)

            if tracks.size > 0:
                frames.append(frame.copy())
                boxes = tracks[:, 0:4]
                track_ids = tracks[:, 4].astype(np.int32).tolist()
                track_results_boxes.append(boxes)
                track_results_ids.append(track_ids)
                track_results_conf.append(tracks[:, 5])
                track_results_cls.append(tracks[:, 6])

                # Save crops
                if save_crops:
                    for box, track_id in zip(boxes, track_ids):
                        x1, y1, x2, y2 = map(int, box[:4])
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            crop_name = f"frame_{frame_idx}_ID_{track_id}.jpg"
                            crop_path = crops_dir / crop_name
                            cv2.imwrite(str(crop_path), crop)

                # Process with PMMM
                frame_matched, abnormal_removed = self.track_reid.processing_to_reid(
                    frame, boxes, track_ids, frame_idx,
                    frame_width, frame_height,
                    track_ids_before_all, abnormal_removed
                )

                track_ids_before_all.extend(track_ids)
                if frame_matched:
                    matched.append(frame_matched)

        # Renew track IDs based on PMMM matching
        print(f"\nMatched IDs: {matched}")
        renew_track_results_ids = renew_track_ids(track_results_ids, matched)

        # Generate MOT format results
        for frame_id, (frame, boxes, ids, conf, cls) in enumerate(
            zip(frames, track_results_boxes, renew_track_results_ids,
                track_results_conf, track_results_cls), start=1
        ):
            frame_id_column = np.full((boxes.shape[0], 1), frame_id, dtype=np.int32)
            mot_results = np.column_stack((
                frame_id_column,
                np.array(ids).astype(np.int32),
                ops.xyxy2ltwh(boxes).astype(np.int32),
                np.ones((boxes.shape[0], 1), dtype=np.int32),
                np.array(cls).astype(np.int32),
                conf,
            ))
            all_mot_results.append(mot_results)

            # Draw and save video
            if save_video:
                for box, id, score in zip(boxes, ids, conf):
                    x1, y1, x2, y2 = map(int, box[:4])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    label = f"ID:{id} {score:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
                video_writer.write(frame)

        # Save MOT results
        if all_mot_results:
            all_mot_results = np.vstack(all_mot_results)
        else:
            all_mot_results = np.empty((0, 0))

        txt_path = results_dir / f"{video_name}.txt"
        write_mot_results(txt_path, all_mot_results)

        if video_writer:
            video_writer.release()

        # Extract tracklet information
        tracklets_info = self._extract_tracklets(
            all_mot_results, video_id, scenario_name,
            tracking_batch, str(results_dir)
        )

        print(f"\nTracking completed!")
        print(f"Results saved to: {results_dir}")
        print(f"Total tracklets: {len(tracklets_info)}")

        return {
            'results_dir': str(results_dir),
            'txt_path': str(txt_path),
            'video_path': str(output_video_path) if save_video else None,
            'tracklets': tracklets_info,
            'total_frames': frame_idx,
            'matched_ids': matched
        }

    def _extract_tracklets(self, mot_results: np.ndarray,
                          video_id: str, scenario_name: str,
                          tracking_batch: int, results_path: str) -> List[Dict]:
        """
        Extract tracklet information from MOT results

        Args:
            mot_results: MOT format results array
            video_id: Video ID
            scenario_name: Scenario name
            tracking_batch: Tracking batch number
            results_path: Path to results directory

        Returns:
            List of tracklet information dictionaries
        """
        if mot_results.size == 0:
            return []

        tracklets = []
        unique_ids = np.unique(mot_results[:, 1])

        for track_id in unique_ids:
            track_data = mot_results[mot_results[:, 1] == track_id]

            # Get start and end frames
            start_frame = int(track_data[0, 0])
            end_frame = int(track_data[-1, 0])

            # Placeholder embeddings (would extract from crops)
            embeddings = [[0.0] * 512]  # Placeholder

            tracklet_info = {
                'video_id': video_id,
                'scenario_name': scenario_name,
                'tracking_batch': tracking_batch,
                'tracking_number': int(track_id),
                'embeddings': embeddings,
                'results_path': results_path,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'num_detections': len(track_data)
            }

            tracklets.append(tracklet_info)

        return tracklets

    def save_tracklets_to_db(self, tracklets_info: List[Dict],
                            video_start_time: datetime) -> List[str]:
        """
        Save tracklets to database

        Args:
            tracklets_info: List of tracklet information
            video_start_time: Video start time for calculating tracklet times

        Returns:
            List of tracklet IDs
        """
        tracklet_ids = []

        for tracklet in tracklets_info:
            # Calculate approximate times (assuming 30 fps)
            fps = 30
            start_offset = tracklet['start_frame'] / fps
            end_offset = tracklet['end_frame'] / fps

            from datetime import timedelta
            started_at = video_start_time + timedelta(seconds=start_offset)
            ended_at = video_start_time + timedelta(seconds=end_offset)

            tracklet_id = self.db_manager.add_tracklet(
                scenario_name=tracklet['scenario_name'],
                tracking_batch=tracklet['tracking_batch'],
                video_id=tracklet['video_id'],
                tracking_number=tracklet['tracking_number'],
                embeddings=tracklet['embeddings'],
                results_path=tracklet['results_path'],
                started_at=started_at,
                ended_at=ended_at
            )

            tracklet_ids.append(tracklet_id)

        return tracklet_ids
