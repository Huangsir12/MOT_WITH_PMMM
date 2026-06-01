"""
Feature Analyzer for Person Attribute Analysis
Integrates attribute classification and group clustering
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

sys.path.append("/root/autodl-tmp/MOT_WITH_PMMM")

from attribution.inference.attribute_classifier import AttributeClassifier
from attribution.inference.group_clustering import GroupClustering
from tracking.database.db_manager import DatabaseManager


class FeatureAnalyzer:
    """
    Analyzes person features including attributes and group relationships
    """

    def __init__(self,
                 checkpoint_dir: str = None,
                 db_manager: Optional[DatabaseManager] = None,
                 device: str = 'cuda'):
        """
        Args:
            checkpoint_dir: Directory containing model checkpoints
            db_manager: Database manager instance
            device: Device for inference
        """
        self.checkpoint_dir = checkpoint_dir or '/root/autodl-tmp/MOT_WITH_PMMM/feature_extraction/checkpoints'
        self.db_manager = db_manager or DatabaseManager
        self.device = device

        # Initialize attribute classifier
        self.attribute_classifier = AttributeClassifier(
            checkpoint_dir=self.checkpoint_dir,
            device=self.device
        )

        # Initialize group clustering
        self.group_clustering = GroupClustering(
            temporal_threshold=30.0,
            spatial_threshold=100.0,
            min_overlap_ratio=0.5,
            min_group_size=2
        )

    def analyze_person_attributes(self,
                                  person_id: str,
                                  crop_dir: str) -> Dict:
        """
        Analyze attributes for a single person

        Args:
            person_id: Person ID
            crop_dir: Directory containing person crops

        Returns:
            attributes: Dictionary with attribute predictions
        """
        print(f"Analyzing attributes for person: {person_id}")

        # Classify attributes from crops
        attributes = self.attribute_classifier.classify_person_crops(
            crop_dir=crop_dir,
            person_id=person_id
        )

        return attributes

    def analyze_scenario_features(self,
                                  scenario_name: str,
                                  tracking_batch: int,
                                  linking_batch: int,
                                  results_base_path: str) -> Dict:
        """
        Analyze features for all persons in a scenario

        Args:
            scenario_name: Scenario name
            tracking_batch: Tracking batch number
            linking_batch: Linking batch number
            results_base_path: Base path for tracking results

        Returns:
            analysis_results: Dictionary with analysis results
        """
        print(f"\n{'='*80}")
        print(f"Analyzing Features for Scenario: {scenario_name}")
        print(f"Tracking Batch: {tracking_batch}, Linking Batch: {linking_batch}")
        print(f"{'='*80}\n")

        # Get person trajectories from database
        person_trajectories = self.db_manager.get_person_trajectories_by_batch(
            scenario_name=scenario_name,
            tracking_batch=tracking_batch,
            linking_batch=linking_batch
        )

        if not person_trajectories:
            print("No person trajectories found")
            return {'person_count': 0, 'features': []}

        print(f"Found {len(person_trajectories)} person trajectories")

        # Analyze attributes for each person
        person_features = []

        for person_traj in person_trajectories:
            person_id = person_traj['person_id']
            tracklets_list = person_traj['tracklets_list']

            print(f"\nProcessing person: {person_id}")
            print(f"  Tracklets: {len(tracklets_list)}")

            # Find crop directory for this person's tracklets
            # Assuming crops are organized by video/tracklet
            crop_dirs = self._find_crop_directories(
                results_base_path,
                scenario_name,
                tracking_batch,
                tracklets_list
            )

            if not crop_dirs:
                print(f"  Warning: No crops found for person {person_id}")
                continue

            # Analyze attributes
            attributes = self.attribute_classifier.classify_person_crops(
                crop_dir=crop_dirs[0],  # Use first crop directory
                person_id=person_id
            )

            # Prepare feature data
            feature_data = {
                'person_id': person_id,
                'tracklets_list_latest': tracklets_list,  # Latest tracklets
                'age': attributes['age']['prediction'],
                'gender': attributes['gender']['prediction'],
                'cloth_style': attributes['clothing']['prediction'],
                'bag_type': attributes['bag']['prediction'],
                'confidence_age': attributes['age']['confidence'],
                'confidence_gender': attributes['gender']['confidence'],
                'confidence_cloth': attributes['clothing']['confidence'],
                'confidence_bag': attributes['bag']['confidence'],
                'group_id': None  # Will be assigned later
            }

            person_features.append(feature_data)

            print(f"  Age: {feature_data['age']} (conf: {feature_data['confidence_age']:.3f})")
            print(f"  Gender: {feature_data['gender']} (conf: {feature_data['confidence_gender']:.3f})")
            print(f"  Clothing: {feature_data['cloth_style']} (conf: {feature_data['confidence_cloth']:.3f})")
            print(f"  Bag: {feature_data['bag_type']} (conf: {feature_data['confidence_bag']:.3f})")

        # Perform group clustering
        print(f"\n{'='*80}")
        print("Performing Group Clustering")
        print(f"{'='*80}\n")

        # Prepare trajectory data for clustering
        trajectory_data = []
        for person_traj in person_trajectories:
            # Get tracklet details
            tracklets = self.db_manager.get_tracklets_for_person(person_traj['person_id'])

            trajectory_data.append({
                'person_id': person_traj['person_id'],
                'tracklets': tracklets
            })

        # Cluster companions
        group_assignments = self.group_clustering.cluster_companions(trajectory_data)

        # Analyze groups
        group_stats = self.group_clustering.analyze_groups(group_assignments)

        print(f"Group Statistics:")
        print(f"  Number of groups: {group_stats['num_groups']}")
        print(f"  Grouped people: {group_stats['num_grouped_people']}")
        print(f"  Solo people: {group_stats['num_solo_people']}")
        print(f"  Average group size: {group_stats['avg_group_size']:.2f}")

        # Assign group IDs to person features
        for feature_data in person_features:
            person_id = feature_data['person_id']
            feature_data['group_id'] = group_assignments.get(person_id)

        # Save features to database
        print(f"\n{'='*80}")
        print("Saving Features to Database")
        print(f"{'='*80}\n")

        saved_count = 0
        for feature_data in person_features:
            try:
                self.db_manager.add_person_feature(**feature_data)
                saved_count += 1
            except Exception as e:
                print(f"Error saving features for {feature_data['person_id']}: {e}")

        print(f"Saved {saved_count}/{len(person_features)} person features to database")

        return {
            'person_count': len(person_features),
            'features': person_features,
            'group_stats': group_stats
        }

    def _find_crop_directories(self,
                               results_base_path: str,
                               scenario_name: str,
                               tracking_batch: int,
                               tracklets_list: List[str]) -> List[str]:
        """
        Find crop directories for given tracklets

        Args:
            results_base_path: Base path for results
            scenario_name: Scenario name
            tracking_batch: Tracking batch number
            tracklets_list: List of tracklet IDs

        Returns:
            crop_dirs: List of crop directory paths
        """
        crop_dirs = []

        # Construct base path
        batch_dir = Path(results_base_path) / scenario_name / f"batch_{tracking_batch:04d}"

        if not batch_dir.exists():
            return crop_dirs

        # Search for crop directories
        for video_dir in batch_dir.iterdir():
            if video_dir.is_dir():
                crops_dir = video_dir / "crops"
                if crops_dir.exists():
                    crop_dirs.append(str(crops_dir))

        return crop_dirs

    def query_person_features(self,
                             scenario_name: str,
                             tracking_batch: int,
                             linking_batch: Optional[int] = None) -> List[Dict]:
        """
        Query person features from database

        Args:
            scenario_name: Scenario name
            tracking_batch: Tracking batch number
            linking_batch: Optional linking batch filter

        Returns:
            features: List of person feature dictionaries
        """
        # Get person trajectories
        person_trajectories = self.db_manager.get_person_trajectories_by_batch(
            scenario_name=scenario_name,
            tracking_batch=tracking_batch,
            linking_batch=linking_batch
        )

        # Get features for each person
        features = []
        for person_traj in person_trajectories:
            person_id = person_traj['person_id']
            feature = self.db_manager.get_person_feature(person_id)
            if feature:
                features.append(feature)

        return features

    def print_feature_summary(self, features: List[Dict]):
        """
        Print summary of person features

        Args:
            features: List of person feature dictionaries
        """
        print(f"\n{'='*80}")
        print(f"Person Feature Summary")
        print(f"{'='*80}\n")

        print(f"Total persons: {len(features)}\n")

        for i, feature in enumerate(features, 1):
            print(f"{i}. Person ID: {feature['person_id']}")
            print(f"   Age: {feature['age']}")
            print(f"   Gender: {feature['gender']}")
            print(f"   Clothing Style: {feature['cloth_style']}")
            print(f"   Bag Type: {feature['bag_type']}")
            print(f"   Group ID: {feature['group_id'] or 'Solo'}")
            print()
