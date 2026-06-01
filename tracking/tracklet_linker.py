"""
Tracklet linking module
Performs clustering-based linking of tracklet fragments to form person trajectories
Optimized with temporal and spatial constraints
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from datetime import datetime
import sys
import logging
sys.path.append("/root/autodl-tmp/MOT_WITH_PMMM")

from tracking.database.db_manager import DatabaseManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


class TrackletLinker:
    """Links tracklet fragments using clustering on appearance features with temporal constraints"""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize tracklet linker

        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager

    def check_time_overlap(self, tracklet1: Dict, tracklet2: Dict) -> bool:
        """
        Check if two tracklets have overlapping time periods

        Args:
            tracklet1: First tracklet with 'started_at' and 'ended_at'
            tracklet2: Second tracklet with 'started_at' and 'ended_at'

        Returns:
            True if tracklets overlap in time, False otherwise
        """
        start1 = datetime.fromisoformat(tracklet1['started_at'])
        end1 = datetime.fromisoformat(tracklet1['ended_at'])
        start2 = datetime.fromisoformat(tracklet2['started_at'])
        end2 = datetime.fromisoformat(tracklet2['ended_at'])

        # Check if time periods overlap
        return not (end1 < start2 or end2 < start1)

    def compute_temporal_distance(self, tracklet1: Dict, tracklet2: Dict) -> float:
        """
        Compute temporal distance between two tracklets (in seconds)

        Args:
            tracklet1: First tracklet with 'started_at' and 'ended_at'
            tracklet2: Second tracklet with 'started_at' and 'ended_at'

        Returns:
            Minimum time gap in seconds (0 if overlapping)
        """
        start1 = datetime.fromisoformat(tracklet1['started_at'])
        end1 = datetime.fromisoformat(tracklet1['ended_at'])
        start2 = datetime.fromisoformat(tracklet2['started_at'])
        end2 = datetime.fromisoformat(tracklet2['ended_at'])

        # If overlapping, return 0
        if not (end1 < start2 or end2 < start1):
            return 0.0

        # Otherwise, return minimum gap
        if end1 < start2:
            gap = (start2 - end1).total_seconds()
        else:
            gap = (start1 - end2).total_seconds()

        return max(0.0, gap)

    def compute_embedding_distance(self, emb1: List[List[float]],
                                   emb2: List[List[float]],
                                   metric: str = 'cosine') -> float:
        """
        Compute distance between two sets of embeddings

        Args:
            emb1: First embedding set (multiple embeddings per tracklet)
            emb2: Second embedding set
            metric: Distance metric ('cosine' or 'euclidean')

        Returns:
            Average minimum distance between embedding sets
        """
        emb1_array = np.array(emb1)
        emb2_array = np.array(emb2)

        if metric == 'cosine':
            distances = cosine_distances(emb1_array, emb2_array)
        else:
            distances = euclidean_distances(emb1_array, emb2_array)

        # Use minimum distance for each embedding in emb1
        min_distances = distances.min(axis=1)
        return float(min_distances.mean())

    def compute_pairwise_distances(self, tracklets: List[Dict],
                                   metric: str = 'cosine',
                                   same_camera_only: bool = True) -> np.ndarray:
        """
        Compute pairwise distance matrix for all tracklets with temporal and spatial constraints

        Args:
            tracklets: List of tracklet dictionaries with embeddings, started_at, ended_at, camera_name
            metric: Distance metric for appearance features
            use_temporal: Whether to include temporal distance in combined distance
            temporal_weight: Weight for temporal distance (appearance_weight = 1 - temporal_weight)
            max_time_gap: Maximum allowed time gap in seconds (None = no limit)
            same_camera_only: Only link tracklets from the same camera

        Returns:
            Distance matrix (n_tracklets x n_tracklets)
        """
        n = len(tracklets)
        distance_matrix = np.full((n, n), np.inf)  # Initialize with infinity
        np.fill_diagonal(distance_matrix, 0.0)

        for i in range(n):
            for j in range(i + 1, n):
                # Check camera constraint
                if same_camera_only:
                    if tracklets[i].get('camera_name') != tracklets[j].get('camera_name'):
                        # Different cameras - set to infinity (will not be linked)
                        distance_matrix[i, j] = np.inf
                        distance_matrix[j, i] = np.inf
                        continue

                # Check time overlap constraint
                if self.check_time_overlap(tracklets[i], tracklets[j]):
                    # Overlapping tracklets cannot be the same person
                    distance_matrix[i, j] = np.inf
                    distance_matrix[j, i] = np.inf
                    continue

                # Compute appearance distance
                appearance_dist = self.compute_embedding_distance(
                    tracklets[i]['embeddings'],
                    tracklets[j]['embeddings'],
                    metric
                )

                distance_matrix[i, j] = appearance_dist
                distance_matrix[j, i] = appearance_dist

        return distance_matrix

    def fuse_embeddings(self, embeddings_list: List[List[List[float]]],
                       method: str = 'mean') -> List[float]:
        """
        Fuse multiple embedding sets into a single embedding

        Args:
            embeddings_list: List of embedding sets from multiple tracklets
            method: Fusion method ('mean', 'median', 'max')

        Returns:
            Fused embedding vector
        """
        # Flatten all embeddings
        all_embeddings = []
        for emb_set in embeddings_list:
            all_embeddings.extend(emb_set)

        all_embeddings = np.array(all_embeddings)

        if method == 'mean':
            fused = all_embeddings.mean(axis=0)
        elif method == 'median':
            fused = np.median(all_embeddings, axis=0)
        elif method == 'max':
            fused = all_embeddings.max(axis=0)
        else:
            fused = all_embeddings.mean(axis=0)

        return fused.tolist()

    def cluster_tracklets_hierarchical(self, tracklets: List[Dict],
                                      distance_threshold: float = 0.5,
                                      metric: str = 'cosine',
                                      linkage_method: str = 'average',
                                      same_camera_only: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster tracklets using hierarchical clustering with temporal constraints

        Args:
            tracklets: List of tracklet dictionaries
            distance_threshold: Distance threshold for clustering
            metric: Distance metric
            linkage_method: Linkage method ('average', 'complete', 'single')
            same_camera_only: Only link tracklets from same camera

        Returns:
            Tuple of (cluster_labels, distance_matrix)
        """
        if len(tracklets) == 0:
            return np.array([]), np.array([])

        if len(tracklets) == 1:
            return np.array([0]), np.array([[0.0]])

        # Compute distance matrix with temporal constraints
        log.info(f"-----正在计算特征距离-----")
        distance_matrix = self.compute_pairwise_distances(
            tracklets, metric, same_camera_only
        )

        # Replace infinity values with a large finite value for sklearn compatibility
        # Get the maximum finite distance in the matrix
        finite_distances = distance_matrix[np.isfinite(distance_matrix)]
        if len(finite_distances) > 0:
            max_finite_dist = np.max(finite_distances)
            # Replace inf with a value larger than any finite distance
            replacement_value = max_finite_dist * 10.0 + 1.0
        else:
            # All distances are infinite (no valid pairs), use a large constant
            replacement_value = 1e6

        distance_matrix_clean = np.copy(distance_matrix)
        distance_matrix_clean[np.isinf(distance_matrix_clean)] = replacement_value

        log.info(f"Replaced {np.sum(np.isinf(distance_matrix))} infinity values with {replacement_value:.2f}")

        # Perform hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric='precomputed',
            linkage=linkage_method
        )

        labels = clustering.fit_predict(distance_matrix_clean)

        return labels, distance_matrix_clean

    def cluster_tracklets_dbscan(self, tracklets: List[Dict],
                                 eps: float = 0.01,
                                 min_samples: int = 1,
                                 metric: str = 'cosine',
                                 same_camera_only: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster tracklets using DBSCAN with temporal constraints

        Args:
            tracklets: List of tracklet dictionaries
            eps: Maximum distance between samples
            min_samples: Minimum samples in a cluster
            metric: Distance metric
            same_camera_only: Only link tracklets from same camera

        Returns:
            Tuple of (cluster_labels, distance_matrix)
        """
        if len(tracklets) == 0:
            return np.array([]), np.array([])

        if len(tracklets) == 1:
            return np.array([0]), np.array([[0.0]])

        # Compute distance matrix with temporal constraints
        distance_matrix = self.compute_pairwise_distances(
            tracklets, metric, same_camera_only)

        # Replace infinity values with a large finite value for sklearn compatibility
        # Get the maximum finite distance in the matrix
        finite_distances = distance_matrix[np.isfinite(distance_matrix)]
        if len(finite_distances) > 0:
            max_finite_dist = np.max(finite_distances)
            # Replace inf with a value larger than any finite distance
            replacement_value = max_finite_dist * 10.0 + 1.0
        else:
            # All distances are infinite (no valid pairs), use a large constant
            replacement_value = 1e6

        distance_matrix_clean = np.copy(distance_matrix)
        distance_matrix_clean[np.isinf(distance_matrix_clean)] = replacement_value

        log.info(f"[DBSCAN] Replaced {np.sum(np.isinf(distance_matrix))} infinity values with {replacement_value:.2f}")

        # Perform DBSCAN clustering
        clustering = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='precomputed'
        )

        labels = clustering.fit_predict(distance_matrix_clean)

        return labels, distance_matrix_clean

    def link_tracklets(self, scenario_name: str, tracking_batch: int,
                      method: str = 'hierarchical',
                      distance_threshold: float = 0.5,
                      metric: str = 'cosine',
                      linking_batch: Optional[int] = None,
                      save_to_db: bool = True,
                      same_camera_only: bool = True) -> Dict:
        """
        Link tracklets for a specific tracking batch with temporal and spatial constraints

        Args:
            scenario_name: Scenario identifier
            tracking_batch: Tracking batch number
            method: Clustering method ('hierarchical' or 'dbscan')
            distance_threshold: Distance threshold for clustering
            metric: Distance metric ('cosine' or 'euclidean')
            linking_batch: Optional linking batch number (auto-increments if None)
            save_to_db: Whether to save results to database
            use_temporal: Whether to use temporal distance in combined distance
            temporal_weight: Weight for temporal distance (0.0-1.0)
            max_time_gap: Maximum time gap in seconds (None = no limit)
            same_camera_only: Only link tracklets from the same camera

        Returns:
            Dictionary with linking results
        """
        print(f"\n{'=' * 80}")
        print(f"Linking tracklets for scenario '{scenario_name}', batch {tracking_batch}")
        print(f"{'=' * 80}")

        # Get tracklets from database WITH camera information
        tracklets = self.db_manager.get_tracklets_by_batch_with_camera(scenario_name, tracking_batch)

        if not tracklets:
            print(f"No tracklets found for scenario '{scenario_name}', batch {tracking_batch}")
            return {'person_count': 0, 'tracklet_count': 0}

        print(f"Found {len(tracklets)} tracklets to link")

        # Display camera statistics
        camera_counts = {}
        for t in tracklets:
            camera = t.get('camera_name', 'unknown')
            camera_counts[camera] = camera_counts.get(camera, 0) + 1
        print(f"Tracklets by camera: {camera_counts}")

        # Perform clustering with temporal constraints
        if method == 'hierarchical':
            labels, distance_matrix = self.cluster_tracklets_hierarchical(
                tracklets, distance_threshold, metric, 'average', same_camera_only)
        elif method == 'dbscan':
            labels, distance_matrix = self.cluster_tracklets_dbscan(
                tracklets, distance_threshold, min_samples=2, metric=metric,
                same_camera_only=same_camera_only)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        # Get linking batch number
        if linking_batch is None:
            linking_batch = self.db_manager.get_latest_linking_batch(
                scenario_name, tracking_batch
            ) + 1

        # Group tracklets by cluster
        unique_labels = np.unique(labels)
        person_ids = []
        linking_results = []

        print(f"\nClustering results: {len(unique_labels)} clusters")

        for label in unique_labels:
            if label == -1:  # Noise in DBSCAN
                print(f"  Cluster -1 (noise): {np.sum(labels == label)} tracklets (skipped)")
                continue

            # Get tracklets in this cluster
            cluster_indices = np.where(labels == label)[0]
            cluster_tracklets = [tracklets[i] for i in cluster_indices]
            cluster_tracklet_ids = [t['tracklet_id'] for t in cluster_tracklets]

            print(f"  Cluster {label}: {len(cluster_tracklets)} tracklets")

            # Compute average distance within cluster
            if len(cluster_indices) > 1:
                cluster_distances = []
                for i in range(len(cluster_indices)):
                    for j in range(i + 1, len(cluster_indices)):
                        idx_i = cluster_indices[i]
                        idx_j = cluster_indices[j]
                        cluster_distances.append(distance_matrix[idx_i, idx_j])
                average_distance = float(np.mean(cluster_distances))
            else:
                average_distance = 0.0

            # Fuse embeddings
            embeddings_list = [t['embeddings'] for t in cluster_tracklets]
            fused_embedding = self.fuse_embeddings(embeddings_list, method='mean')

            # Save to database
            if save_to_db:
                person_id = self.db_manager.add_person_trajectory(
                    scenario_name=scenario_name,
                    tracklets_list=cluster_tracklet_ids,
                    tracking_batch=tracking_batch,
                    linking_batch=linking_batch,
                    average_distance=average_distance,
                    fused_embedding=fused_embedding
                )
                person_ids.append(person_id)

                print(f"    -> Person ID: {person_id}")
                print(f"    -> Average distance: {average_distance:.4f}")
                print(f"    -> Tracklets: {cluster_tracklet_ids}")

            linking_results.append({
                'cluster_label': int(label),
                'person_id': person_id if save_to_db else None,
                'tracklet_ids': cluster_tracklet_ids,
                'tracklet_count': len(cluster_tracklet_ids),
                'average_distance': average_distance,
                'fused_embedding': fused_embedding
            })

        print(f"\n{'=' * 80}")
        print(f"Linking complete!")
        print(f"Total persons identified: {len(person_ids)}")
        print(f"Total tracklets linked: {len(tracklets)}")
        print(f"Linking batch: {linking_batch}")
        print(f"{'=' * 80}")

        return {
            'scenario_name': scenario_name,
            'tracking_batch': tracking_batch,
            'linking_batch': linking_batch,
            'person_count': len(person_ids),
            'tracklet_count': len(tracklets),
            'person_ids': person_ids,
            'linking_results': linking_results,
            'method': method,
            'distance_threshold': distance_threshold,
            'metric': metric
        }

    def visualize_clusters(self, scenario_name: str, tracking_batch: int,
                          linking_batch: int):
        """
        Visualize clustering results (print summary)

        Args:
            scenario_name: Scenario identifier
            tracking_batch: Tracking batch number
            linking_batch: Linking batch number
        """
        trajectories = self.db_manager.get_person_trajectories_by_batch(
            scenario_name, tracking_batch, linking_batch
        )

        print(f"\n{'=' * 80}")
        print(f"Person Trajectories Summary")
        print(f"Scenario: {scenario_name}, Tracking Batch: {tracking_batch}, Linking Batch: {linking_batch}")
        print(f"{'=' * 80}")

        for i, traj in enumerate(trajectories, 1):
            print(f"\nPerson {i}:")
            print(f"  Person ID: {traj['person_id']}")
            print(f"  Tracklets: {len(traj['tracklets_list'])}")
            print(f"  Average Distance: {traj['average_distance']:.4f}")
            print(f"  Tracklet IDs: {traj['tracklets_list']}")

            # Get detailed tracklet info
            tracklets = self.db_manager.get_tracklets_for_person(traj['person_id'])
            print(f"  Tracklet Details:")
            for tracklet in tracklets:
                print(f"    - Video: {tracklet['video_id'][:8]}...")
                print(f"      Tracking #: {tracklet['tracking_number']}")
                print(f"      Time: {tracklet['started_at']} -> {tracklet['ended_at']}")

        print(f"\n{'=' * 80}")
        print(f"Total Persons: {len(trajectories)}")
        print(f"{'=' * 80}")
