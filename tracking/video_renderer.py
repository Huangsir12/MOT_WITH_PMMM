"""
Video renderer module
Re-renders tracking videos with linked person IDs (global track_id)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import sys
sys.path.append("/root/autodl-tmp/MOT_WITH_PMMM")

from tracking.database.db_manager import DatabaseManager
from tracking.utils import read_mot_results


class VideoRenderer:
    """Re-renders tracking videos with linked person IDs"""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize video renderer

        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager if db_manager else DatabaseManager

    def build_tracklet_to_person_mapping(self,
                                        scenario_name: str,
                                        tracking_batch: int,
                                        linking_batch: int) -> Tuple[Dict[str, str], Dict[str, int]]:
        """
        Build mapping from tracklet_id to person_id and person_id to global_track_id

        Args:
            scenario_name: Scenario name
            tracking_batch: Tracking batch number
            linking_batch: Linking batch number

        Returns:
            Tuple of (tracklet_to_person_map, person_to_track_id_map)
        """
        # Get all person trajectories
        person_trajectories = self.db_manager.get_person_trajectories_by_batch(
            scenario_name, tracking_batch, linking_batch
        )

        tracklet_to_person = {}
        person_to_track_id = {}

        # Assign global track_id starting from 1
        for idx, person in enumerate(person_trajectories, start=1):
            person_id = person['person_id']
            tracklet_ids = person['tracklets_list']

            # Map person_id to global track_id
            person_to_track_id[person_id] = idx

            # Map each tracklet_id to this person_id
            for tracklet_id in tracklet_ids:
                tracklet_to_person[tracklet_id] = person_id

        return tracklet_to_person, person_to_track_id

    def get_video_tracklet_mapping(self,
                                  video_id: str,
                                  tracklet_to_person: Dict[str, str],
                                  person_to_track_id: Dict[str, int]) -> Dict[int, int]:
        """
        Get mapping from original tracking_number to global track_id for a specific video

        Args:
            video_id: Video ID
            tracklet_to_person: Mapping from tracklet_id to person_id
            person_to_track_id: Mapping from person_id to global track_id

        Returns:
            Mapping from original tracking_number to global track_id
        """
        # Get all tracklets for this video
        tracklets = self.db_manager.get_tracklets_by_video(video_id)

        tracking_number_to_track_id = {}

        for tracklet in tracklets:
            tracklet_id = tracklet['tracklet_id']
            tracking_number = tracklet['tracking_number']

            # Check if this tracklet was linked
            if tracklet_id in tracklet_to_person:
                person_id = tracklet_to_person[tracklet_id]
                global_track_id = person_to_track_id[person_id]
                tracking_number_to_track_id[tracking_number] = global_track_id
            # else: tracklet was not linked (noise), skip it

        return tracking_number_to_track_id

    def render_video_with_linked_ids(self,
                                     video_path: str,
                                     mot_txt_path: str,
                                     output_video_path: str,
                                     tracking_number_to_track_id: Dict[int, int],
                                     fps: int = 30,
                                     show_conf: bool = False) -> None:
        """
        Re-render video with linked person IDs

        Args:
            video_path: Path to original video
            mot_txt_path: Path to MOT format txt file
            output_video_path: Path to save output video
            tracking_number_to_track_id: Mapping from original tracking_number to global track_id
            fps: Frame rate
            show_conf: Whether to show confidence scores
        """
        # Read MOT results
        mot_results = read_mot_results(mot_txt_path)

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        print(f"Re-rendering video: {video_path}")
        print(f"Output: {output_video_path}")
        print(f"Total frames: {total_frames}")

        frame_id = 1
        colors = self._generate_colors(len(tracking_number_to_track_id) + 100)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Get detections for current frame
            frame_dets = mot_results[mot_results[:, 0] == frame_id]

            for det in frame_dets:
                _, orig_tracking_num, x, y, w, h, _, _, conf = det

                orig_tracking_num = int(orig_tracking_num)

                # Check if this tracking_number was linked
                if orig_tracking_num not in tracking_number_to_track_id:
                    # Skip unlinked tracklets
                    continue

                global_track_id = tracking_number_to_track_id[orig_tracking_num]

                # Convert ltwh to xyxy
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)

                # Draw bounding box
                color = colors[global_track_id % len(colors)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw label
                if show_conf:
                    label = f"ID:{global_track_id} ({conf:.2f})"
                else:
                    label = f"ID:{global_track_id}"

                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 4), (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Write frame
            out.write(frame)
            frame_id += 1

        cap.release()
        out.release()
        print(f"Video rendering complete: {output_video_path}")

    def _generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for visualization"""
        colors = []
        for i in range(n):
            hue = int(180 * i / n)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append((int(color[0]), int(color[1]), int(color[2])))
        return colors

    def render_scenario(self,
                       scenario_name: str,
                       tracking_batch: int,
                       linking_batch: int,
                       output_dir: str,
                       camera_name: Optional[str] = None) -> Dict:
        """
        Re-render all videos for a scenario with linked person IDs

        Args:
            scenario_name: Scenario name
            tracking_batch: Tracking batch number
            linking_batch: Linking batch number
            output_dir: Output directory for rendered videos
            camera_name: Optional camera filter

        Returns:
            Dictionary with rendering results
        """
        print(f"\n{'=' * 80}")
        print(f"Rendering videos for scenario '{scenario_name}'")
        print(f"Tracking batch: {tracking_batch}, Linking batch: {linking_batch}")
        print(f"{'=' * 80}")

        # Build tracklet to person mapping
        tracklet_to_person, person_to_track_id = self.build_tracklet_to_person_mapping(
            scenario_name, tracking_batch, linking_batch
        )

        print(f"Found {len(person_to_track_id)} persons with global IDs: 1 - {len(person_to_track_id)}")

        # Get all videos for the scenario
        videos = self.db_manager.get_video_sources_by_scenario(scenario_name, camera_name)

        if not videos:
            print(f"No videos found for scenario '{scenario_name}'")
            return {'rendered_count': 0, 'total': 0}

        # Create output directory
        output_path = Path(output_dir) / scenario_name / f"batch_{tracking_batch:04d}"
        output_path.mkdir(parents=True, exist_ok=True)

        rendered_count = 0

        for video in videos:
            video_id = video['video_id']
            video_path = video['source_path']
            camera = video['camera_name']

            print(f"\n{'=' * 80}")
            print(f"Processing video: {camera} - {video_path}")
            print(f"{'=' * 80}")

            # Check if video file exists
            if not Path(video_path).exists():
                print(f"Warning: Video file not found: {video_path}")
                continue

            # Get tracklets for this video
            tracklets = self.db_manager.get_tracklets_by_video(video_id)
            if not tracklets:
                print(f"No tracklets found for video {video_id}")
                continue

            # Find MOT txt file for this video
            # Assuming results are in: results_base_path / scenario / batch_XXXX / video_name / video_name.txt
            tracklet_results_path = tracklets[0]['results_path']
            video_name = Path(video_path).stem
            mot_txt_path = Path(tracklet_results_path) / f"{video_name}.txt"

            if not mot_txt_path.exists():
                print(f"Warning: MOT txt file not found: {mot_txt_path}")
                continue

            # Build mapping for this video
            tracking_number_to_track_id = self.get_video_tracklet_mapping(
                video_id, tracklet_to_person, person_to_track_id
            )

            print(f"Linked tracklets: {len(tracking_number_to_track_id)}")

            # Render video
            output_video_path = output_path / video_name / f"linking_batch_{linking_batch:04d}_video.mp4"
            self.render_video_with_linked_ids(
                video_path=video_path,
                mot_txt_path=str(mot_txt_path),
                output_video_path=str(output_video_path),
                tracking_number_to_track_id=tracking_number_to_track_id,
                fps=30,
                show_conf=False
            )

            rendered_count += 1

        print(f"\n{'=' * 80}")
        print(f"Rendering complete!")
        print(f"Rendered videos: {rendered_count}/{len(videos)}")
        print(f"Output directory: {output_path}")
        print(f"{'=' * 80}")

        return {
            'rendered_count': rendered_count,
            'total': len(videos),
            'output_dir': str(output_path)
        }
