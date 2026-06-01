"""
Database connection and session management
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from typing import Generator
import sys
sys.path.append("/root/autodl-tmp/MOT_WITH_PMMM")

# from tracking.config.settings import get_settings
# from tracking.database.models import Base

# settings = get_settings()

# # Create database engine
# engine = create_engine(
#     settings.database_url,
#     poolclass=QueuePool,
#     pool_size=10,
#     max_overflow=20,
#     pool_pre_ping=True,
#     pool_recycle=3600,  # Recycle connections after 1 hour
#     echo=False,
# )

# ===================== 新增：导入ssh隧道核心包 =====================
from sshtunnel import SSHTunnelForwarder
import pymysql
# =================================================================

from tracking.config.settings import get_settings
from tracking.database.models import Base
from pydantic import BaseSettings  # 补上你代码里的BaseSettings

# ===================== 全局创建SSH隧道 (核心新增) =====================
# 全局SSH隧道对象，启动后一直复用，避免重复创建隧道
ssh_tunnel = None

# 初始化SSH隧道并启动
def init_ssh_tunnel():
    global ssh_tunnel
    settings = get_settings()
    if ssh_tunnel is None or not ssh_tunnel.is_active:
        ssh_tunnel = SSHTunnelForwarder(
            ssh_address_or_host=(settings.SSH_HOST, settings.SSH_PORT),  # SSH跳板机地址+端口
            ssh_username=settings.SSH_USER,                              # SSH用户名
            ssh_password=settings.SSH_PASSWORD,                          # SSH密码
            remote_bind_address=(settings.DB_HOST, settings.DB_PORT),    # 跳板机内的mysql地址+端口
            local_bind_address=('127.0.0.1', 0)                         # 本地随机端口映射，自动分配，无端口冲突
        )
        ssh_tunnel.start()
        print(f"✅ SSH隧道启动成功，本地映射端口: {ssh_tunnel.local_bind_port}")
# =================================================================

settings = get_settings()
# ===================== 提前启动SSH隧道 =====================
init_ssh_tunnel()

# Create database engine
# ===================== 修改：动态拼接带SSH隧道的数据库连接URL =====================
DATABASE_URL = f"mysql+pymysql://{settings.DB_USER}:{settings.DB_PASSWORD}@127.0.0.1:{ssh_tunnel.local_bind_port}/{settings.DB_NAME}?charset={settings.DB_CHARSET}"

engine = create_engine(
    DATABASE_URL,  # 使用拼接后的SSH隧道连接地址
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600,  # Recycle connections after 1 hour
    echo=False,
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    # print("Database tables created successfully!")


def drop_db():
    """Drop all database tables (use with caution!)"""
    Base.metadata.drop_all(bind=engine)
    print("Database tables dropped!")


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Get database session context manager

    Usage:
        with get_db() as db:
            # Your database operations
            pass
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db_session() -> Generator[Session, None, None]:
    """
    Get database session for FastAPI dependency injection

    Usage:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db_session)):
            return db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
