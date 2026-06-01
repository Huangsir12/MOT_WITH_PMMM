"""
SQLAlchemy ORM models for MOT tracking system
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()


class VideoDataSource(Base):
    """Video data source model"""
    __tablename__ = 'video_data_source'

    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String(36), unique=True, nullable=False, index=True)
    scenario_name = Column(String(100), nullable=False, index=True)
    camera_name = Column(String(100), nullable=False, index=True)
    source_path = Column(String(500), nullable=False)
    start_time = Column(DateTime, nullable=False, index=True)
    end_time = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())

    __table_args__ = (
        Index('idx_scenario_camera_time', 'scenario_name', 'camera_name', 'start_time', unique=True),
    )

    def __repr__(self):
        return f"<VideoDataSource(video_id='{self.video_id}', scenario='{self.scenario_name}', camera='{self.camera_name}')>"


class TrackletsResult(Base):
    """Tracklets result model"""
    __tablename__ = 'tracklets_result'

    id = Column(Integer, primary_key=True, autoincrement=True)
    tracklet_id = Column(String(36), unique=True, nullable=False, index=True)
    scenario_name = Column(String(100), nullable=False, index=True)
    tracking_batch = Column(Integer, nullable=False, index=True)
    video_id = Column(String(36), ForeignKey('video_data_source.video_id'), nullable=False, index=True)
    tracking_number = Column(Integer, nullable=False)
    embeddings = Column(Text, nullable=True)  # JSON array
    results_path = Column(String(500), nullable=False)
    started_at = Column(DateTime, nullable=False)
    ended_at = Column(DateTime, nullable=False)
    operated_at = Column(DateTime, nullable=False, server_default=func.now())

    __table_args__ = (
        Index('idx_scenario_batch', 'scenario_name', 'tracking_batch'),
    )

    def __repr__(self):
        return f"<TrackletsResult(tracklet_id='{self.tracklet_id}', tracking_number={self.tracking_number})>"


class PersonTrajectory(Base):
    """Person trajectory model (linked tracklets)"""
    __tablename__ = 'person_trajectory'

    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(String(36), unique=True, nullable=False, index=True)
    scenario_name = Column(String(100), nullable=False, index=True)
    tracklets_list = Column(Text, nullable=False)  # JSON array of tracklet_ids
    tracking_batch = Column(Integer, nullable=False, index=True)
    linking_batch = Column(Integer, nullable=False, index=True)
    average_distance = Column(Float, nullable=True)
    fused_embedding = Column(Text, nullable=True)  # JSON array
    created_at = Column(DateTime, nullable=False, server_default=func.now())

    __table_args__ = (
        Index('idx_scenario_tracking_batch', 'scenario_name', 'tracking_batch'),
        Index('idx_scenario_linking_batch', 'scenario_name', 'linking_batch'),
    )

    def __repr__(self):
        return f"<PersonTrajectory(person_id='{self.person_id}', tracking_batch={self.tracking_batch}, linking_batch={self.linking_batch})>"


class PersonFeature(Base):
    """Person feature model (attribute analysis results)"""
    __tablename__ = 'person_feature'

    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(String(36), ForeignKey('person_trajectory.person_id'), nullable=False, index=True)
    tracklets_list_latest = Column(Text, nullable=True)  # JSON array of latest tracklet_ids
    age = Column(String(20), nullable=True)  # Age classification result
    gender = Column(String(10), nullable=True)  # Gender classification result
    cloth_style = Column(String(50), nullable=True)  # Clothing style classification result
    bag_type = Column(String(50), nullable=True)  # Bag type classification result
    group_id = Column(String(36), nullable=True, index=True)  # Companion group identifier
    confidence_age = Column(Float, nullable=True)  # Confidence score for age
    confidence_gender = Column(Float, nullable=True)  # Confidence score for gender
    confidence_cloth = Column(Float, nullable=True)  # Confidence score for clothing
    confidence_bag = Column(Float, nullable=True)  # Confidence score for bag
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index('idx_person_group', 'person_id', 'group_id'),
    )

    def __repr__(self):
        return f"<PersonFeature(person_id='{self.person_id}', age='{self.age}', gender='{self.gender}', group_id='{self.group_id}')>"
