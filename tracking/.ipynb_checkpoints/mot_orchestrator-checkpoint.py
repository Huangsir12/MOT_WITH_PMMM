"""
Main orchestration script for MOT tracking system
Handles end-to-end workflow from video registration to tracking and storage
"""

import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import sys

sys.path.append("/root/autodl-tmp/MOT_WITH_PMMM")
from tracking.config.settings import get_settings
from tracking.video_manager import VideoDataManager
from tracking.tracking_processor import TrackingProcessor
from tracking.tracklet_linker import TrackletLinker
from tracking.database.db_manager import DatabaseManager
from tracking.feature_analyzer import FeatureAnalyzer

settings = get_settings()


class MOTOrchestrator:
    """Orchestrates the complete MOT tracking workflow"""

    def __init__(self):
        """
        Initialize MOT orchestrator

        Args:
            video_base_path: Base path for video data sources
            results_base_path: Base path for tracking results
            db_path: Path to database file
        """
        self.video_base_path = settings.VIDEO_BASE_PATH
        self.results_base_path = settings.RESULTS_BASE_PATH

        # Initialize managers
        self.db_manager = DatabaseManager
        self.video_manager = VideoDataManager(self.video_base_path, self.db_manager)
        self.tracking_processor = TrackingProcessor(db_manager=self.db_manager)
        self.tracklet_linker = TrackletLinker(db_manager=self.db_manager)
        self.feature_analyzer = FeatureAnalyzer(db_manager=self.db_manager)

    def register_videos(self, scenario_name: Optional[str] = None,
                       camera_name: Optional[str] = None) -> List[str]:
        """
        Step 1: Register videos from file system to database

        Args:
            scenario_name: Optional scenario filter
            camera_name: Optional camera filter

        Returns:
            List of registered video IDs
        """
        print("=" * 80)
        print("STEP 1: Registering videos from file system")
        print("=" * 80)

        video_ids = self.video_manager.register_videos(scenario_name, camera_name)

        print(f"\nTotal videos registered: {len(video_ids)}")
        return video_ids

    def process_scenario(self,
                        scenario_name: str,
                        camera_name: Optional[str] = None,
                        tracking_batch: Optional[int] = None,
                        save_video: bool = True,
                        save_crops: bool = True) -> dict:
        """
        Step 2: Process all videos for a scenario

        Args:
            scenario_name: Scenario to process
            camera_name: Optional camera filter
            tracking_batch: Optional batch number (auto-increments if None)
            save_video: Whether to save annotated videos
            save_crops: Whether to save detection crops

        Returns:
            Dictionary with processing results
        """
        print("=" * 80)
        print(f"STEP 2: Processing scenario '{scenario_name}'")
        print("=" * 80)

        # Get tracking batch number
        if tracking_batch is None:
            tracking_batch = self.db_manager.get_latest_tracking_batch(scenario_name) + 1

        print(f"Tracking batch: {tracking_batch}")

        # Get videos to process
        videos = self.video_manager.get_videos_for_tracking(scenario_name, camera_name)

        if not videos:
            print(f"No videos found for scenario '{scenario_name}'")
            return {'processed': 0, 'tracklets': []}

        print(f"Found {len(videos)} videos to process")

        all_tracklets = []
        processed_count = 0

        for video in videos:
            print(f"\n{'=' * 80}")
            print(f"Processing: {video['camera_name']} - {video['source_path']}")
            print(f"{'=' * 80}")

            try:
                # Process video
                result = self.tracking_processor.process_video(
                    video_path=video['source_path'],
                    output_dir=self.results_base_path,
                    video_id=video['video_id'],
                    scenario_name=scenario_name,
                    tracking_batch=tracking_batch,
                    save_video=save_video,
                    save_crops=save_crops
                )

                # Save tracklets to database
                video_start_time = datetime.fromisoformat(video['start_time'])
                tracklet_ids = self.tracking_processor.save_tracklets_to_db(
                    result['tracklets'],
                    video_start_time
                )

                print(f"Saved {len(tracklet_ids)} tracklets to database")

                all_tracklets.extend(tracklet_ids)
                processed_count += 1

            except Exception as e:
                print(f"Error processing video {video['source_path']}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n{'=' * 80}")
        print(f"Processing complete!")
        print(f"Processed videos: {processed_count}/{len(videos)}")
        print(f"Total tracklets: {len(all_tracklets)}")
        print(f"{'=' * 80}")

        return {
            'processed': processed_count,
            'total': len(videos),
            'tracklets': all_tracklets,
            'tracking_batch': tracking_batch
        }

    def query_tracklets(self, scenario_name: str, tracking_batch: int):
        """
        Step 3: Query tracklets for a scenario and batch

        Args:
            scenario_name: Scenario name
            tracking_batch: Tracking batch number
        """
        print("=" * 80)
        print(f"STEP 3: Querying tracklets for '{scenario_name}' batch {tracking_batch}")
        print("=" * 80)

        tracklets = self.db_manager.get_tracklets_by_batch(scenario_name, tracking_batch)

        print(f"\nFound {len(tracklets)} tracklets")

        for i, tracklet in enumerate(tracklets[:10], 1):  # Show first 10
            print(f"\n{i}. Tracklet ID: {tracklet['tracklet_id']}")
            print(f"   Video ID: {tracklet['video_id']}")
            print(f"   Tracking Number: {tracklet['tracking_number']}")
            print(f"   Time: {tracklet['started_at']} -> {tracklet['ended_at']}")
            print(f"   Results: {tracklet['results_path']}")

        if len(tracklets) > 10:
            print(f"\n... and {len(tracklets) - 10} more tracklets")

        return tracklets

    def link_tracklets(self, scenario_name: str, tracking_batch: int,
                      method: str = 'hierarchical',
                      distance_threshold: float = 0.5,
                      metric: str = 'cosine',
                      linking_batch: Optional[int] = None) -> dict:
        """
        Step 4: Link tracklets to form person trajectories

        Args:
            scenario_name: Scenario name
            tracking_batch: Tracking batch number
            method: Clustering method ('hierarchical' or 'dbscan')
            distance_threshold: Distance threshold for clustering
            metric: Distance metric ('cosine' or 'euclidean')
            linking_batch: Optional linking batch number

        Returns:
            Dictionary with linking results
        """
        print("=" * 80)
        print(f"STEP 4: Linking tracklets for '{scenario_name}' batch {tracking_batch}")
        print("=" * 80)

        result = self.tracklet_linker.link_tracklets(
            scenario_name=scenario_name,
            tracking_batch=tracking_batch,
            method=method,
            distance_threshold=distance_threshold,
            metric=metric,
            linking_batch=linking_batch,
            save_to_db=True
        )

        return result

    def query_person_trajectories(self, scenario_name: str, tracking_batch: int,
                                 linking_batch: Optional[int] = None):
        """
        Step 5: Query person trajectories

        Args:
            scenario_name: Scenario name
            tracking_batch: Tracking batch number
            linking_batch: Optional linking batch filter
        """
        print("=" * 80)
        print(f"STEP 5: Querying person trajectories for '{scenario_name}' batch {tracking_batch}")
        if linking_batch:
            print(f"Linking batch: {linking_batch}")
        print("=" * 80)

        trajectories = self.db_manager.get_person_trajectories_by_batch(
            scenario_name, tracking_batch, linking_batch
        )

        print(f"\nFound {len(trajectories)} person trajectories")

        for i, traj in enumerate(trajectories[:10], 1):  # Show first 10
            print(f"\n{i}. Person ID: {traj['person_id']}")
            print(f"   Tracklets: {len(traj['tracklets_list'])} tracklets")
            print(f"   Average Distance: {traj['average_distance']:.4f}")
            print(f"   Linking Batch: {traj['linking_batch']}")
            print(f"   Tracklet IDs: {traj['tracklets_list'][:3]}{'...' if len(traj['tracklets_list']) > 3 else ''}")

        if len(trajectories) > 10:
            print(f"\n... and {len(trajectories) - 10} more trajectories")

        return trajectories

    def visualize_person_trajectory(self, person_id: str):
        """
        Visualize a specific person trajectory with detailed tracklet information

        Args:
            person_id: Person ID to visualize
        """
        self.tracklet_linker.visualize_clusters(
            scenario_name=None,  # Will be retrieved from person record
            tracking_batch=None,
            linking_batch=None
        )

        # Get person trajectory
        person = self.db_manager.get_person_trajectory(person_id)
        if not person:
            print(f"Person ID {person_id} not found")
            return

        print("=" * 80)
        print(f"Person Trajectory Details")
        print("=" * 80)
        print(f"\nPerson ID: {person['person_id']}")
        print(f"Scenario: {person['scenario_name']}")
        print(f"Tracking Batch: {person['tracking_batch']}")
        print(f"Linking Batch: {person['linking_batch']}")
        print(f"Average Distance: {person['average_distance']:.4f}")
        print(f"Number of Tracklets: {len(person['tracklets_list'])}")

        # Get tracklet details
        tracklets = self.db_manager.get_tracklets_for_person(person_id)

        print(f"\nTracklet Details:")
        for i, tracklet in enumerate(tracklets, 1):
            print(f"\n  {i}. Tracklet ID: {tracklet['tracklet_id']}")
            print(f"     Video ID: {tracklet['video_id']}")
            print(f"     Tracking Number: {tracklet['tracking_number']}")
            print(f"     Time: {tracklet['started_at']} -> {tracklet['ended_at']}")
            print(f"     Results: {tracklet['results_path']}")

    def analyze_features(self,
                        scenario_name: str,
                        tracking_batch: int,
                        linking_batch: Optional[int] = None) -> dict:
        """
        Step 6: Analyze person attributes and group relationships

        Args:
            scenario_name: Scenario name
            tracking_batch: Tracking batch number
            linking_batch: Optional linking batch number

        Returns:
            Dictionary with feature analysis results
        """
        print("=" * 80)
        print(f"STEP 6: Analyzing person features for '{scenario_name}' batch {tracking_batch}")
        if linking_batch:
            print(f"Linking batch: {linking_batch}")
        print("=" * 80)

        # Get linking batch if not specified
        if linking_batch is None:
            linking_batch = self.db_manager.get_latest_linking_batch(scenario_name, tracking_batch)

        if linking_batch == 0:
            print("No linking batch found. Please run tracklet linking first.")
            return {'person_count': 0, 'features': []}

        # Analyze features
        result = self.feature_analyzer.analyze_scenario_features(
            scenario_name=scenario_name,
            tracking_batch=tracking_batch,
            linking_batch=linking_batch,
            results_base_path=self.results_base_path
        )

        return result

    def query_person_features(self,
                             scenario_name: str,
                             tracking_batch: int,
                             linking_batch: Optional[int] = None):
        """
        Step 7: Query and display person features

        Args:
            scenario_name: Scenario name
            tracking_batch: Tracking batch number
            linking_batch: Optional linking batch filter
        """
        print("=" * 80)
        print(f"STEP 7: Querying person features for '{scenario_name}' batch {tracking_batch}")
        if linking_batch:
            print(f"Linking batch: {linking_batch}")
        print("=" * 80)

        # Get person trajectories
        person_trajectories = self.db_manager.get_person_trajectories_by_batch(
            scenario_name, tracking_batch, linking_batch
        )

        if not person_trajectories:
            print("No person trajectories found")
            return []

        # Get features for each person
        features = []
        for person_traj in person_trajectories:
            person_id = person_traj['person_id']
            feature = self.db_manager.get_person_feature(person_id)
            if feature:
                features.append(feature)

        print(f"\nFound {len(features)} person features")

        # Display features
        for i, feature in enumerate(features[:10], 1):  # Show first 10
            print(f"\n{i}. Person ID: {feature['person_id']}")
            print(f"   Age: {feature['age']} (confidence: {feature['confidence_age']:.3f})")
            print(f"   Gender: {feature['gender']} (confidence: {feature['confidence_gender']:.3f})")
            print(f"   Clothing Style: {feature['cloth_style']} (confidence: {feature['confidence_cloth']:.3f})")
            print(f"   Bag Type: {feature['bag_type']} (confidence: {feature['confidence_bag']:.3f})")
            print(f"   Group ID: {feature['group_id'] or 'Solo'}")

        if len(features) > 10:
            print(f"\n... and {len(features) - 10} more features")

        return features


def main():
    parser = argparse.ArgumentParser(
        description="MOT Tracking System - End-to-end multi-object tracking with PMMM"
    )

    parser.add_argument(
        'command',
        choices=['register', 'process', 'query', 'link', 'query-persons', 'analyze-features', 'query-features', 'full'],
        help='Command to execute'
    )

    parser.add_argument(
        '--scenario',
        type=str,
        help='Scenario name (e.g., dajixiang, anchang)'
    )

    parser.add_argument(
        '--camera',
        type=str,
        help='Camera name (e.g., camera_001)'
    )

    parser.add_argument(
        '--batch',
        type=int,
        help='Tracking batch number (auto-increments if not specified)'
    )

    parser.add_argument(
        '--no-video',
        action='store_true',
        help='Do not save annotated videos'
    )

    parser.add_argument(
        '--no-crops',
        action='store_true',
        help='Do not save detection crops'
    )

    parser.add_argument(
        '--linking-batch',
        type=int,
        help='Linking batch number (auto-increments if not specified)'
    )

    parser.add_argument(
        '--method',
        type=str,
        default='hierarchical',
        choices=['hierarchical', 'dbscan'],
        help='Clustering method for tracklet linking'
    )

    parser.add_argument(
        '--distance-threshold',
        type=float,
        default=0.5,
        help='Distance threshold for clustering (default: 0.5)'
    )

    parser.add_argument(
        '--metric',
        type=str,
        default='cosine',
        choices=['cosine', 'euclidean'],
        help='Distance metric for clustering'
    )

    parser.add_argument(
        '--person-id',
        type=str,
        help='Person ID for visualization'
    )

    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = MOTOrchestrator()

    # Execute command
    if args.command == 'register':
        orchestrator.register_videos(args.scenario, args.camera)

    elif args.command == 'process':
        if not args.scenario:
            print("Error: --scenario is required for 'process' command")
            return

        orchestrator.process_scenario(
            scenario_name=args.scenario,
            camera_name=args.camera,
            tracking_batch=args.batch,
            save_video=not args.no_video,
            save_crops=not args.no_crops
        )

    elif args.command == 'query':
        if not args.scenario or args.batch is None:
            print("Error: --scenario and --batch are required for 'query' command")
            return

        orchestrator.query_tracklets(args.scenario, args.batch)

    elif args.command == 'link':
        if not args.scenario or args.batch is None:
            print("Error: --scenario and --batch are required for 'link' command")
            return

        orchestrator.link_tracklets(
            scenario_name=args.scenario,
            tracking_batch=args.batch,
            method=args.method,
            distance_threshold=args.distance_threshold,
            metric=args.metric,
            linking_batch=args.linking_batch
        )

    elif args.command == 'query-persons':
        if not args.scenario or args.batch is None:
            print("Error: --scenario and --batch are required for 'query-persons' command")
            return

        if args.person_id:
            # Visualize specific person
            orchestrator.visualize_person_trajectory(args.person_id)
        else:
            # Query all persons
            orchestrator.query_person_trajectories(
                args.scenario,
                args.batch,
                args.linking_batch
            )

    elif args.command == 'analyze-features':
        if not args.scenario or args.batch is None:
            print("Error: --scenario and --batch are required for 'analyze-features' command")
            return

        orchestrator.analyze_features(
            scenario_name=args.scenario,
            tracking_batch=args.batch,
            linking_batch=args.linking_batch
        )

    elif args.command == 'query-features':
        if not args.scenario or args.batch is None:
            print("Error: --scenario and --batch are required for 'query-features' command")
            return

        orchestrator.query_person_features(
            args.scenario,
            args.batch,
            args.linking_batch
        )

    elif args.command == 'full':
        if not args.scenario:
            print("Error: --scenario is required for 'full' command")
            return

        # Full workflow
        print("\n" + "=" * 80)
        print("FULL WORKFLOW: Register -> Process -> Query -> Link -> Query Persons -> Analyze Features -> Query Features")
        print("=" * 80 + "\n")

        # Step 1: Register
        orchestrator.register_videos(args.scenario, args.camera)

        # Step 2: Process
        result = orchestrator.process_scenario(
            scenario_name=args.scenario,
            camera_name=args.camera,
            tracking_batch=args.batch,
            save_video=not args.no_video,
            save_crops=not args.no_crops
        )

        # Step 3: Query tracklets
        if result['processed'] > 0:
            orchestrator.query_tracklets(args.scenario, result['tracking_batch'])

            # Step 4: Link tracklets
            link_result = orchestrator.link_tracklets(
                scenario_name=args.scenario,
                tracking_batch=result['tracking_batch'],
                method=args.method,
                distance_threshold=args.distance_threshold,
                metric=args.metric
            )

            # Step 5: Query person trajectories
            if link_result['person_count'] > 0:
                orchestrator.query_person_trajectories(
                    args.scenario,
                    result['tracking_batch'],
                    link_result['linking_batch']
                )

                # Step 6: Analyze features
                feature_result = orchestrator.analyze_features(
                    scenario_name=args.scenario,
                    tracking_batch=result['tracking_batch'],
                    linking_batch=link_result['linking_batch']
                )

                # Step 7: Query features
                if feature_result['person_count'] > 0:
                    orchestrator.query_person_features(
                        args.scenario,
                        result['tracking_batch'],
                        link_result['linking_batch']
                    )


if __name__ == "__main__":
    main()
