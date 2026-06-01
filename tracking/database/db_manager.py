"""
MySQL Database manager using SQLAlchemy ORM
Handles all database operations for MOT tracking system
"""

import json
import uuid
from typing import List, Dict, Optional
from datetime import datetime
from sqlalchemy import func
from sqlalchemy.orm import Session

import sys
sys.path.append("/root/autodl-tmp/MOT_WITH_PMMM")

from tracking.database.connection import get_db, init_db
from tracking.database.models import VideoDataSource, TrackletsResult, PersonTrajectory, PersonFeature


class DatabaseManager:
    """Manages database operations using SQLAlchemy ORM"""

    def __init__(self):
        """Initialize database manager"""
        # Ensure tables are created
        init_db()

    # ==================== Video Data Source Operations ====================

    def add_video_source(self, scenario_name: str, camera_name: str,
                        source_path: str, start_time: datetime,
                        end_time: datetime) -> str:
        """
        Add a new video data source

        Args:
            scenario_name: Scenario identifier
            camera_name: Camera identifier
            source_path: Absolute path to video file
            start_time: Video start time
            end_time: Video end time

        Returns:
            video_id: UUID of the created video source
        """
        video_id = str(uuid.uuid4())

        with get_db() as db:
            try:
                video = VideoDataSource(
                    video_id=video_id,
                    scenario_name=scenario_name,
                    camera_name=camera_name,
                    source_path=source_path,
                    start_time=start_time,
                    end_time=end_time
                )
                db.add(video)
                db.commit()
                return video_id
            except Exception as e:
                # Check if video already exists
                existing = db.query(VideoDataSource).filter(
                    VideoDataSource.scenario_name == scenario_name,
                    VideoDataSource.camera_name == camera_name,
                    VideoDataSource.start_time == start_time
                ).first()
                if existing:
                    return existing.video_id
                raise e

    def get_video_source(self, video_id: str) -> Optional[Dict]:
        """Get video source by ID"""
        with get_db() as db:
            video = db.query(VideoDataSource).filter(
                VideoDataSource.video_id == video_id
            ).first()

            if video:
                return {
                    'id': video.id,
                    'video_id': video.video_id,
                    'scenario_name': video.scenario_name,
                    'camera_name': video.camera_name,
                    'source_path': video.source_path,
                    'start_time': video.start_time.isoformat(),
                    'end_time': video.end_time.isoformat(),
                    'created_at': video.created_at.isoformat()
                }
            return None

    def get_video_sources_by_scenario(self, scenario_name: str,
                                     camera_name: Optional[str] = None) -> List[Dict]:
        """Get all video sources for a scenario"""
        with get_db() as db:
            query = db.query(VideoDataSource).filter(
                VideoDataSource.scenario_name == scenario_name
            )

            if camera_name:
                query = query.filter(VideoDataSource.camera_name == camera_name)

            videos = query.order_by(
                VideoDataSource.camera_name,
                VideoDataSource.start_time
            ).all()

            return [
                {
                    'id': v.id,
                    'video_id': v.video_id,
                    'scenario_name': v.scenario_name,
                    'camera_name': v.camera_name,
                    'source_path': v.source_path,
                    'start_time': v.start_time.isoformat(),
                    'end_time': v.end_time.isoformat(),
                    'created_at': v.created_at.isoformat()
                }
                for v in videos
            ]

    # ==================== Tracklets Operations ====================

    def add_tracklet(self, scenario_name: str, tracking_batch: int,
                    video_id: str, tracking_number: int,
                    embeddings: List[List[float]], results_path: str,
                    started_at: datetime, ended_at: datetime) -> str:
        """Add a new tracklet result"""
        tracklet_id = str(uuid.uuid4())
        embeddings_json = json.dumps(embeddings)

        with get_db() as db:
            tracklet = TrackletsResult(
                tracklet_id=tracklet_id,
                scenario_name=scenario_name,
                tracking_batch=tracking_batch,
                video_id=video_id,
                tracking_number=tracking_number,
                embeddings=embeddings_json,
                results_path=results_path,
                started_at=started_at,
                ended_at=ended_at
            )
            db.add(tracklet)
            db.commit()
            return tracklet_id

    def get_tracklet(self, tracklet_id: str) -> Optional[Dict]:
        """Get tracklet by ID"""
        with get_db() as db:
            tracklet = db.query(TrackletsResult).filter(
                TrackletsResult.tracklet_id == tracklet_id
            ).first()

            if tracklet:
                return {
                    'id': tracklet.id,
                    'tracklet_id': tracklet.tracklet_id,
                    'scenario_name': tracklet.scenario_name,
                    'tracking_batch': tracklet.tracking_batch,
                    'video_id': tracklet.video_id,
                    'tracking_number': tracklet.tracking_number,
                    'embeddings': json.loads(tracklet.embeddings) if tracklet.embeddings else [],
                    'results_path': tracklet.results_path,
                    'started_at': tracklet.started_at.isoformat(),
                    'ended_at': tracklet.ended_at.isoformat(),
                    'operated_at': tracklet.operated_at.isoformat()
                }
            return None

    def get_tracklets_by_batch(self, scenario_name: str,
                               tracking_batch: int) -> List[Dict]:
        """Get all tracklets for a specific batch"""
        with get_db() as db:
            tracklets = db.query(TrackletsResult).filter(
                TrackletsResult.scenario_name == scenario_name,
                TrackletsResult.tracking_batch == tracking_batch
            ).order_by(TrackletsResult.started_at).all()

            return [
                {
                    'id': t.id,
                    'tracklet_id': t.tracklet_id,
                    'scenario_name': t.scenario_name,
                    'tracking_batch': t.tracking_batch,
                    'video_id': t.video_id,
                    'tracking_number': t.tracking_number,
                    'embeddings': json.loads(t.embeddings) if t.embeddings else [],
                    'results_path': t.results_path,
                    'started_at': t.started_at.isoformat(),
                    'ended_at': t.ended_at.isoformat(),
                    'operated_at': t.operated_at.isoformat()
                }
                for t in tracklets
            ]

    def get_tracklets_by_batch_with_camera(self, scenario_name: str,
                                           tracking_batch: int) -> List[Dict]:
        """
        Get all tracklets for a specific batch with camera information

        Args:
            scenario_name: Scenario name
            tracking_batch: Tracking batch number

        Returns:
            List of tracklets with camera_name field
        """
        with get_db() as db:
            # Join tracklets_result with video_data_source to get camera_name
            results = db.query(
                TrackletsResult, VideoDataSource.camera_name
            ).join(
                VideoDataSource,
                TrackletsResult.video_id == VideoDataSource.video_id
            ).filter(
                TrackletsResult.scenario_name == scenario_name,
                TrackletsResult.tracking_batch == tracking_batch
            ).order_by(TrackletsResult.started_at).all()

            return [
                {
                    'id': t.id,
                    'tracklet_id': t.tracklet_id,
                    'scenario_name': t.scenario_name,
                    'tracking_batch': t.tracking_batch,
                    'video_id': t.video_id,
                    'tracking_number': t.tracking_number,
                    'embeddings': json.loads(t.embeddings) if t.embeddings else [],
                    'results_path': t.results_path,
                    'started_at': t.started_at.isoformat(),
                    'ended_at': t.ended_at.isoformat(),
                    'operated_at': t.operated_at.isoformat(),
                    'camera_name': camera_name  # Add camera information
                }
                for t, camera_name in results
            ]

    def get_tracklets_by_video(self, video_id: str) -> List[Dict]:
        """Get all tracklets for a specific video"""
        with get_db() as db:
            tracklets = db.query(TrackletsResult).filter(
                TrackletsResult.video_id == video_id
            ).order_by(TrackletsResult.tracking_number).all()

            return [
                {
                    'id': t.id,
                    'tracklet_id': t.tracklet_id,
                    'scenario_name': t.scenario_name,
                    'tracking_batch': t.tracking_batch,
                    'video_id': t.video_id,
                    'tracking_number': t.tracking_number,
                    'embeddings': json.loads(t.embeddings) if t.embeddings else [],
                    'results_path': t.results_path,
                    'started_at': t.started_at.isoformat(),
                    'ended_at': t.ended_at.isoformat(),
                    'operated_at': t.operated_at.isoformat()
                }
                for t in tracklets
            ]

    def get_latest_tracking_batch(self, scenario_name: str) -> int:
        """Get the latest tracking batch number for a scenario"""
        with get_db() as db:
            result = db.query(func.max(TrackletsResult.tracking_batch)).filter(
                TrackletsResult.scenario_name == scenario_name
            ).scalar()

            return result if result is not None else 0

    def update_tracklet_embeddings(self, tracklet_id: str,
                                  embeddings: List[List[float]]):
        """Update embeddings for a tracklet"""
        embeddings_json = json.dumps(embeddings)

        with get_db() as db:
            db.query(TrackletsResult).filter(
                TrackletsResult.tracklet_id == tracklet_id
            ).update({'embeddings': embeddings_json})
            db.commit()

    # ==================== Person Trajectory Operations ====================

    def add_person_trajectory(self, scenario_name: str, tracklets_list: List[str],
                             tracking_batch: int, linking_batch: int,
                             average_distance: float,
                             fused_embedding: List[float]) -> str:
        """Add a new person trajectory (linked tracklets)"""
        person_id = str(uuid.uuid4())
        tracklets_json = json.dumps(tracklets_list)
        fused_embedding_json = json.dumps(fused_embedding)

        with get_db() as db:
            person = PersonTrajectory(
                person_id=person_id,
                scenario_name=scenario_name,
                tracklets_list=tracklets_json,
                tracking_batch=tracking_batch,
                linking_batch=linking_batch,
                average_distance=average_distance,
                fused_embedding=fused_embedding_json
            )
            db.add(person)
            db.commit()
            return person_id

    def get_person_trajectory(self, person_id: str) -> Optional[Dict]:
        """Get person trajectory by ID"""
        with get_db() as db:
            person = db.query(PersonTrajectory).filter(
                PersonTrajectory.person_id == person_id
            ).first()

            if person:
                return {
                    'id': person.id,
                    'person_id': person.person_id,
                    'scenario_name': person.scenario_name,
                    'tracklets_list': json.loads(person.tracklets_list),
                    'tracking_batch': person.tracking_batch,
                    'linking_batch': person.linking_batch,
                    'average_distance': person.average_distance,
                    'fused_embedding': json.loads(person.fused_embedding) if person.fused_embedding else [],
                    'created_at': person.created_at.isoformat()
                }
            return None

    def get_person_trajectories_by_batch(self, scenario_name: str,
                                        tracking_batch: int,
                                        linking_batch: Optional[int] = None) -> List[Dict]:
        """Get all person trajectories for a specific batch"""
        with get_db() as db:
            query = db.query(PersonTrajectory).filter(
                PersonTrajectory.scenario_name == scenario_name,
                PersonTrajectory.tracking_batch == tracking_batch
            )

            if linking_batch is not None:
                query = query.filter(PersonTrajectory.linking_batch == linking_batch)

            persons = query.order_by(
                PersonTrajectory.linking_batch,
                PersonTrajectory.person_id
            ).all()

            return [
                {
                    'id': p.id,
                    'person_id': p.person_id,
                    'scenario_name': p.scenario_name,
                    'tracklets_list': json.loads(p.tracklets_list),
                    'tracking_batch': p.tracking_batch,
                    'linking_batch': p.linking_batch,
                    'average_distance': p.average_distance,
                    'fused_embedding': json.loads(p.fused_embedding) if p.fused_embedding else [],
                    'created_at': p.created_at.isoformat()
                }
                for p in persons
            ]

    def get_latest_linking_batch(self, scenario_name: str, tracking_batch: int) -> int:
        """Get the latest linking batch number"""
        with get_db() as db:
            result = db.query(func.max(PersonTrajectory.linking_batch)).filter(
                PersonTrajectory.scenario_name == scenario_name,
                PersonTrajectory.tracking_batch == tracking_batch
            ).scalar()

            return result if result is not None else 0

    def get_tracklets_for_person(self, person_id: str) -> List[Dict]:
        """Get all tracklets that belong to a person"""
        person = self.get_person_trajectory(person_id)
        if not person:
            return []

        tracklet_ids = person['tracklets_list']
        tracklets = []

        with get_db() as db:
            for tracklet_id in tracklet_ids:
                tracklet = db.query(TrackletsResult).filter(
                    TrackletsResult.tracklet_id == tracklet_id
                ).first()

                if tracklet:
                    tracklets.append({
                        'id': tracklet.id,
                        'tracklet_id': tracklet.tracklet_id,
                        'scenario_name': tracklet.scenario_name,
                        'tracking_batch': tracklet.tracking_batch,
                        'video_id': tracklet.video_id,
                        'tracking_number': tracklet.tracking_number,
                        'embeddings': json.loads(tracklet.embeddings) if tracklet.embeddings else [],
                        'results_path': tracklet.results_path,
                        'started_at': tracklet.started_at.isoformat(),
                        'ended_at': tracklet.ended_at.isoformat(),
                        'operated_at': tracklet.operated_at.isoformat()
                    })

        return tracklets

    # ==================== Person Feature Operations ====================

    def add_person_feature(self, person_id: str,
                          tracklets_list_latest: List[str],
                          age: Optional[str] = None,
                          gender: Optional[str] = None,
                          cloth_style: Optional[str] = None,
                          bag_type: Optional[str] = None,
                          group_id: Optional[str] = None,
                          confidence_age: Optional[float] = None,
                          confidence_gender: Optional[float] = None,
                          confidence_cloth: Optional[float] = None,
                          confidence_bag: Optional[float] = None) -> int:
        """
        Add or update person feature

        Args:
            person_id: Person ID (foreign key to person_trajectory)
            tracklets_list_latest: Latest tracklet IDs
            age: Age classification result
            gender: Gender classification result
            cloth_style: Clothing style classification result
            bag_type: Bag type classification result
            group_id: Companion group identifier
            confidence_age: Confidence score for age
            confidence_gender: Confidence score for gender
            confidence_cloth: Confidence score for clothing
            confidence_bag: Confidence score for bag

        Returns:
            feature_id: ID of the created/updated feature record
        """
        tracklets_json = json.dumps(tracklets_list_latest)

        with get_db() as db:
            # Check if feature already exists
            existing = db.query(PersonFeature).filter(
                PersonFeature.person_id == person_id
            ).first()

            if existing:
                # Update existing feature
                existing.tracklets_list_latest = tracklets_json
                existing.age = age
                existing.gender = gender
                existing.cloth_style = cloth_style
                existing.bag_type = bag_type
                existing.group_id = group_id
                existing.confidence_age = confidence_age
                existing.confidence_gender = confidence_gender
                existing.confidence_cloth = confidence_cloth
                existing.confidence_bag = confidence_bag
                db.commit()
                return existing.id
            else:
                # Create new feature
                feature = PersonFeature(
                    person_id=person_id,
                    tracklets_list_latest=tracklets_json,
                    age=age,
                    gender=gender,
                    cloth_style=cloth_style,
                    bag_type=bag_type,
                    group_id=group_id,
                    confidence_age=confidence_age,
                    confidence_gender=confidence_gender,
                    confidence_cloth=confidence_cloth,
                    confidence_bag=confidence_bag
                )
                db.add(feature)
                db.commit()
                return feature.id

    def get_person_feature(self, person_id: str) -> Optional[Dict]:
        """Get person feature by person ID"""
        with get_db() as db:
            feature = db.query(PersonFeature).filter(
                PersonFeature.person_id == person_id
            ).first()

            if feature:
                return {
                    'id': feature.id,
                    'person_id': feature.person_id,
                    'tracklets_list_latest': json.loads(feature.tracklets_list_latest) if feature.tracklets_list_latest else [],
                    'age': feature.age,
                    'gender': feature.gender,
                    'cloth_style': feature.cloth_style,
                    'bag_type': feature.bag_type,
                    'group_id': feature.group_id,
                    'confidence_age': feature.confidence_age,
                    'confidence_gender': feature.confidence_gender,
                    'confidence_cloth': feature.confidence_cloth,
                    'confidence_bag': feature.confidence_bag,
                    'created_at': feature.created_at.isoformat(),
                    'updated_at': feature.updated_at.isoformat()
                }
            return None

    def get_person_features_by_group(self, group_id: str) -> List[Dict]:
        """Get all person features in a group"""
        with get_db() as db:
            features = db.query(PersonFeature).filter(
                PersonFeature.group_id == group_id
            ).all()

            return [
                {
                    'id': f.id,
                    'person_id': f.person_id,
                    'tracklets_list_latest': json.loads(f.tracklets_list_latest) if f.tracklets_list_latest else [],
                    'age': f.age,
                    'gender': f.gender,
                    'cloth_style': f.cloth_style,
                    'bag_type': f.bag_type,
                    'group_id': f.group_id,
                    'confidence_age': f.confidence_age,
                    'confidence_gender': f.confidence_gender,
                    'confidence_cloth': f.confidence_cloth,
                    'confidence_bag': f.confidence_bag,
                    'created_at': f.created_at.isoformat(),
                    'updated_at': f.updated_at.isoformat()
                }
                for f in features
            ]

    def query_person_features(self,
                             age: Optional[str] = None,
                             gender: Optional[str] = None,
                             cloth_style: Optional[str] = None,
                             bag_type: Optional[str] = None,
                             group_id: Optional[str] = None) -> List[Dict]:
        """
        Query person features with filters

        Args:
            age: Filter by age
            gender: Filter by gender
            cloth_style: Filter by clothing style
            bag_type: Filter by bag type
            group_id: Filter by group ID

        Returns:
            List of matching person features
        """
        with get_db() as db:
            query = db.query(PersonFeature)

            if age:
                query = query.filter(PersonFeature.age == age)
            if gender:
                query = query.filter(PersonFeature.gender == gender)
            if cloth_style:
                query = query.filter(PersonFeature.cloth_style == cloth_style)
            if bag_type:
                query = query.filter(PersonFeature.bag_type == bag_type)
            if group_id:
                query = query.filter(PersonFeature.group_id == group_id)

            features = query.all()

            return [
                {
                    'id': f.id,
                    'person_id': f.person_id,
                    'tracklets_list_latest': json.loads(f.tracklets_list_latest) if f.tracklets_list_latest else [],
                    'age': f.age,
                    'gender': f.gender,
                    'cloth_style': f.cloth_style,
                    'bag_type': f.bag_type,
                    'group_id': f.group_id,
                    'confidence_age': f.confidence_age,
                    'confidence_gender': f.confidence_gender,
                    'confidence_cloth': f.confidence_cloth,
                    'confidence_bag': f.confidence_bag,
                    'created_at': f.created_at.isoformat(),
                    'updated_at': f.updated_at.isoformat()
                }
                for f in features
            ]


# Alias for backward compatibility
DatabaseManager = DatabaseManager()
