"""
Configuration settings for MOT tracking system
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings"""
    # SSH 隧道配置 - 对应你的DBeaver SSH配置
    SSH_HOST: str = "connect.nmb2.seetacloud.com"
    SSH_PORT: int = 16000
    SSH_USER: str = "root"
    SSH_PASSWORD: str = "5pnWPeGUfrd4"  # 替换成你的真实SSH密码


    # Database Configuration
    DB_HOST: str = "localhost"
    DB_PORT: int = 3306
    DB_USER: str = "bao"
    DB_PASSWORD: str = "password123"
    DB_NAME: str = "tracking_reid"
    DB_CHARSET: str = "utf8mb4"

    # Paths Configuration
    VIDEO_BASE_PATH: str = "/root/autodl-fs/tracking_reid/video_data_source"
    RESULTS_BASE_PATH: str = "/root/autodl-fs/tracking_reid/mot_results"

    # Tracking Configuration
    DEFAULT_YOLO_MODEL: str = "yolov10x_trained_best1.pt"
    DEFAULT_REID_MODEL: str = "osnet_x1_0_msmt17.pt"
    DEFAULT_TRACKING_METHOD: str = "botsort"

    # Linking Configuration
    DEFAULT_DISTANCE_THRESHOLD: float = 0.5
    DEFAULT_CLUSTERING_METHOD: str = "hierarchical"
    DEFAULT_DISTANCE_METRIC: str = "cosine"

    class Config:
        env_file = ".env"
        case_sensitive = True

    @property
    def database_url(self) -> str:
        """Get database URL for SQLAlchemy"""
        return (
            f"mysql+pymysql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
            f"?charset={self.DB_CHARSET}"
        )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
