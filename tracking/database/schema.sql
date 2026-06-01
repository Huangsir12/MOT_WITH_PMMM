-- Database schema for MOT tracking system

-- Video data source table
CREATE TABLE IF NOT EXISTS video_data_source (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT UNIQUE NOT NULL,
    scenario_name TEXT NOT NULL,
    camera_name TEXT NOT NULL,
    source_path TEXT NOT NULL,
    start_time DATETIME NOT NULL,
    end_time DATETIME NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(scenario_name, camera_name, start_time)
);

-- Tracklets result table
CREATE TABLE IF NOT EXISTS tracklets_result (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tracklet_id TEXT UNIQUE NOT NULL,
    scenario_name TEXT NOT NULL,
    tracking_batch INTEGER NOT NULL,
    video_id TEXT NOT NULL,
    tracking_number INTEGER NOT NULL,
    embeddings TEXT,  -- JSON array of embeddings
    results_path TEXT NOT NULL,
    started_at DATETIME NOT NULL,
    ended_at DATETIME NOT NULL,
    operated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES video_data_source(video_id),
    INDEX idx_scenario_batch (scenario_name, tracking_batch),
    INDEX idx_video_id (video_id)
);

-- Person trajectory table (linked tracklets)
CREATE TABLE IF NOT EXISTS person_trajectory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT UNIQUE NOT NULL,
    scenario_name TEXT NOT NULL,
    tracklets_list TEXT NOT NULL,  -- JSON array of tracklet_ids
    tracking_batch INTEGER NOT NULL,
    linking_batch INTEGER NOT NULL,
    average_distance REAL,
    fused_embedding TEXT,  -- JSON array of fused embedding
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_person_scenario (scenario_name),
    INDEX idx_person_tracking_batch (tracking_batch),
    INDEX idx_person_linking_batch (linking_batch)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_video_scenario ON video_data_source(scenario_name);
CREATE INDEX IF NOT EXISTS idx_video_camera ON video_data_source(camera_name);
CREATE INDEX IF NOT EXISTS idx_video_time ON video_data_source(start_time, end_time);
CREATE INDEX IF NOT EXISTS idx_tracklet_scenario ON tracklets_result(scenario_name);
CREATE INDEX IF NOT EXISTS idx_tracklet_batch ON tracklets_result(tracking_batch);
CREATE INDEX IF NOT EXISTS idx_trajectory_scenario ON person_trajectory(scenario_name);
CREATE INDEX IF NOT EXISTS idx_trajectory_tracking_batch ON person_trajectory(tracking_batch);
CREATE INDEX IF NOT EXISTS idx_trajectory_linking_batch ON person_trajectory(linking_batch);
