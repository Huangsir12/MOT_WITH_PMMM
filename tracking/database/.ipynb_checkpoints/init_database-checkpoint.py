#!/usr/bin/env python3
"""
Database initialization script
Creates MySQL database and tables
"""

import sys
sys.path.append("/root/autodl-tmp/MOT_WITH_PMMM")

import pymysql
from tracking.config.settings import get_settings
from tracking.database.connection import init_db, engine
from sqlalchemy import text

settings = get_settings()


def create_database():
    """Create database if it doesn't exist"""
    print(f"Creating database '{settings.DB_NAME}' if not exists...")

    # Connect to MySQL server without specifying database
    connection = pymysql.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        charset=settings.DB_CHARSET
    )

    try:
        with connection.cursor() as cursor:
            # Create database
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS {settings.DB_NAME} "
                f"CHARACTER SET {settings.DB_CHARSET} "
                f"COLLATE {settings.DB_CHARSET}_unicode_ci"
            )
            print(f"✓ Database '{settings.DB_NAME}' created/verified")

        connection.commit()
    finally:
        connection.close()


def create_tables():
    """Create all tables"""
    print("\nCreating tables...")
    init_db()
    print("✓ All tables created successfully")


def verify_tables():
    """Verify tables were created"""
    print("\nVerifying tables...")

    with engine.connect() as conn:
        result = conn.execute(text("SHOW TABLES"))
        tables = [row[0] for row in result]

        expected_tables = ['video_data_source', 'tracklets_result', 'person_trajectory']

        print(f"\nFound {len(tables)} tables:")
        for table in tables:
            status = "✓" if table in expected_tables else "?"
            print(f"  {status} {table}")

        missing = set(expected_tables) - set(tables)
        if missing:
            print(f"\n⚠ Missing tables: {missing}")
            return False
        else:
            print("\n✓ All expected tables exist")
            return True


def main():
    """Main initialization function"""
    print("=" * 80)
    print("MOT Tracking System - Database Initialization")
    print("=" * 80)

    print(f"\nDatabase Configuration:")
    print(f"  Host: {settings.DB_HOST}:{settings.DB_PORT}")
    print(f"  Database: {settings.DB_NAME}")
    print(f"  User: {settings.DB_USER}")
    print(f"  Charset: {settings.DB_CHARSET}")

    try:
        # Step 1: Create database
        create_database()

        # Step 2: Create tables
        create_tables()

        # Step 3: Verify tables
        success = verify_tables()

        if success:
            print("\n" + "=" * 80)
            print("✓ Database initialization completed successfully!")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("⚠ Database initialization completed with warnings")
            print("=" * 80)

    except Exception as e:
        print(f"\n✗ Error during initialization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
