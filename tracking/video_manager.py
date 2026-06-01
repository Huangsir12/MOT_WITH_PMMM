"""
Video data source management module
Handles scanning, parsing, and registering video files from the file system
"""

import os
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from tracking.database.db_manager import DatabaseManager


class VideoDataManager:
    """Manages video data sources from file system"""

    def __init__(self, base_path: str = None,
                 db_manager: Optional[DatabaseManager] = None):
        """
        Initialize video data manager

        Args:
            base_path: Base directory containing video data
            db_manager: Database manager instance
        """
        self.base_path = Path(base_path)
        self.db_manager = db_manager

    @staticmethod
    def parse_video_filename(filename: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Parse video filename to extract start and end times

        Expected format: 2025-07-02-14-25-39_2025-07-02-14-40-40.mp4

        Args:
            filename: Video filename

        Returns:
            Tuple of (start_time, end_time) or (None, None) if parsing fails
        """
        # Remove file extension
        name_without_ext = Path(filename).stem

        # Pattern: YYYY-MM-DD-HH-MM-SS_YYYY-MM-DD-HH-MM-SS
        pattern = r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})'
        match = re.match(pattern, name_without_ext)

        if not match:
            return None, None

        try:
            start_str, end_str = match.groups()
            start_time = datetime.strptime(start_str, '%Y-%m-%d-%H-%M-%S')
            end_time = datetime.strptime(end_str, '%Y-%m-%d-%H-%M-%S')
            return start_time, end_time
        except ValueError:
            return None, None

    def scan_video_directory(self, scenario_name: Optional[str] = None,
                            camera_name: Optional[str] = None) -> List[Dict]:
        """
        Scan video directory and return list of video files

        Directory structure:
        video_data_source/
            scenario_name/
                camera_name/
                    video_file.mp4

        Args:
            scenario_name: Optional scenario filter
            camera_name: Optional camera filter

        Returns:
            List of video file information dictionaries
        """
        videos = []

        if not self.base_path.exists():
            print(f"Warning: Base path {self.base_path} does not exist")
            return videos

        # Determine which scenarios to scan
        if scenario_name:
            scenario_dirs = [self.base_path / scenario_name]
        else:
            scenario_dirs = [d for d in self.base_path.iterdir() if d.is_dir()]

        for scenario_dir in scenario_dirs:
            if not scenario_dir.exists():
                continue

            current_scenario = scenario_dir.name

            # Determine which cameras to scan
            if camera_name:
                camera_dirs = [scenario_dir / camera_name]
            else:
                camera_dirs = [d for d in scenario_dir.iterdir() if d.is_dir()]

            for camera_dir in camera_dirs:
                if not camera_dir.exists():
                    continue

                current_camera = camera_dir.name

                # Scan for video files
                for video_file in camera_dir.glob('*.mp4'):
                    start_time, end_time = self.parse_video_filename(video_file.name)

                    if start_time and end_time:
                        videos.append({
                            'scenario_name': current_scenario,
                            'camera_name': current_camera,
                            'source_path': str(video_file.absolute()),
                            'filename': video_file.name,
                            'start_time': start_time,
                            'end_time': end_time
                        })
                    else:
                        print(f"Warning: Could not parse filename: {video_file.name}")

        return videos

    def register_videos(self, scenario_name: Optional[str] = None,
                       camera_name: Optional[str] = None) -> List[str]:
        """
        Scan and register videos in database

        Args:
            scenario_name: Optional scenario filter
            camera_name: Optional camera filter

        Returns:
            List of registered video IDs
        """
        videos = self.scan_video_directory(scenario_name, camera_name)
        registered_ids = []

        for video in videos:
            try:
                video_id = self.db_manager.add_video_source(
                    scenario_name=video['scenario_name'],
                    camera_name=video['camera_name'],
                    source_path=video['source_path'],
                    start_time=video['start_time'],
                    end_time=video['end_time']
                )
                registered_ids.append(video_id)
                print(f"Registered: {video['scenario_name']}/{video['camera_name']}/{video['filename']} -> {video_id}")
            except Exception as e:
                print(f"Error registering {video['filename']}: {e}")

        return registered_ids

    def get_videos_for_tracking(self, scenario_name: str,
                               camera_name: Optional[str] = None) -> List[Dict]:
        """
        Get videos from database for tracking

        Args:
            scenario_name: Scenario identifier
            camera_name: Optional camera filter

        Returns:
            List of video source records
        """
        return self.db_manager.get_video_sources_by_scenario(scenario_name, camera_name)
