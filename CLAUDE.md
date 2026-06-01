# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MOT_WITH_PMMM is a Multi-Object Tracking (MOT) system that integrates the Position and Multi-step Memory Matching (PMMM) module for enhanced long-term association accuracy. The project combines YOLO detection, multiple tracking algorithms (BoTSORT, ByteTrack, etc.), and BPBreID for person re-identification.

## Core Architecture

### Three-Layer System Design

1. **Detection Layer**: YOLO-based object detection (YOLOv8/v10/v11)
2. **Tracking Layer**: Multiple tracking algorithms via BoxMOT (botsort, bytetrack, ocsort, strongsort, deepocsort, hybridsort, imprassoc, boosttrack)
3. **Re-identification Layer**: BPBreID for appearance feature extraction + PMMM module for long-term association

### Key Components

**tracking/** - Main tracking system
- `track.py` / `track_with_pmmm.py`: Inference scripts (with/without PMMM)
- `val.py`: Evaluation on MOT benchmarks (MOT17/MOT20/custom datasets)
- `mot_orchestrator.py`: Orchestrates full pipeline (video registration → tracking → tracklet linking)
- `tracking_processor.py`: Core tracking processor integrating YOLO + tracker + ReID
- `tracklet_linker.py`: Clusters tracklet fragments into person trajectories using appearance features
- `video_manager.py`: Manages video data sources and database registration
- `pmmm_scripts/trackreid_pmmm.py`: PMMM module implementation for long-term re-identification

**bpbreid/** - Person re-identification module
- `torchreid/scripts/main.py`: Training script for ReID models
- `torchreid/scripts/reID_app.py`: ReID inference interface
- `configs/bpbreid/`: Configuration files for different datasets (market1501, dukemtmc, custom datasets)
- `torchreid/data/datasets/`: Dataset loaders (add custom datasets here)

**boxmot/** - Multi-object tracking library (third-party, integrated)
- `trackers/`: Implementations of various tracking algorithms
- `appearance/`: ReID model backends and exporters
- `configs/`: Tracker configuration YAML files

**tracking/database/** - SQLite database for tracking results
- `video_data_source`: Video metadata (scenario, camera, timestamps)
- `tracklets_result`: Tracklet fragments from single-camera tracking
- `person_trajectory`: Linked person trajectories across tracklets

## Common Commands

### Training

**Train ReID model:**
```bash
conda activate pmmm
cd bpbreid/torchreid
python ./scripts/main.py --config-file configs/bpbreid/bpbreid_<dataset>_train.yaml
```

**Train YOLO detector:**
```bash
# From scratch
yolo detect train data=coco8.yaml model=yolo11n.yaml epochs=100 imgsz=640

# From pretrained
yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640
```

### Evaluation

**Evaluate on MOT benchmark:**
```bash
python tracking/val.py \
  --yolo-model WEIGHTS/'your_trained.pt' \
  --reid-model WEIGHTS/'your_trained.pt' \
  --tracking-method botsort \
  --source DATA/"datasets"/"Emporium"/"train" \
  --use-pmmm True
```

### Inference

**Track without PMMM:**
```bash
python tracking/track.py \
  --yolo-model WEIGHTS/'your_trained.pt' \
  --reid-model WEIGHTS/'your_trained.pt' \
  --tracking-method botsort
```

**Track with PMMM:**
```bash
python tracking/track_with_pmmm.py \
  --yolo-model WEIGHTS/'your_trained.pt' \
  --reid-model WEIGHTS/'your_trained.pt' \
  --tracking-method botsort
```

### Full Pipeline (MOT Orchestrator)

**Setup directory structure:**
```bash
python tracking/mot_orchestrator.py setup
```

**Register videos to database:**
```bash
python tracking/mot_orchestrator.py register --scenario <scenario_name>
```

**Process tracking:**
```bash
python tracking/mot_orchestrator.py process --scenario <scenario_name> --batch 1
```

**Link tracklets into person trajectories:**
```bash
python tracking/mot_orchestrator.py link \
  --scenario <scenario_name> \
  --batch 1 \
  --method hierarchical \
  --distance-threshold 0.5 \
  --metric cosine
```

**Query results:**
```bash
# Query tracklets
python tracking/mot_orchestrator.py query --scenario <scenario_name> --batch 1

# Query person trajectories
python tracking/mot_orchestrator.py query-persons --scenario <scenario_name> --batch 1
```

**Full pipeline (one command):**
```bash
python tracking/mot_orchestrator.py full --scenario <scenario_name>
```

## Configuration Files

**Tracker configs**: `boxmot/configs/<tracker_name>.yaml`
- Adjust tracking parameters (IOU threshold, max age, min hits, etc.)

**ReID configs**: `bpbreid/configs/bpbreid/bpbreid_<dataset>_<train|test|inference>.yaml`
- Model architecture, training hyperparameters, dataset paths
- For inference, set `inference.dataset_folder` to output directory

**YOLO data configs**: `data.yaml` or custom dataset YAML
- Dataset paths, class names, number of classes

## Adding Custom Datasets

### For ReID Training

1. Create dataset class in `bpbreid/torchreid/data/datasets/image/<dataset_name>.py`
2. Register in `bpbreid/torchreid/data/datasets/image/__init__.py`
3. Create config file `bpbreid/configs/bpbreid/bpbreid_<dataset_name>_train.yaml`
4. Organize data as: `bpbreid/datasets/<dataset_name>/train/` and `test/`

### For Detection Training

1. Organize data in YOLO format (images + labels)
2. Create `data.yaml` with paths and class definitions
3. Train with `yolo detect train data=<your_data.yaml> ...`

## PMMM Module Details

The PMMM (Position and Multi-step Memory Matching) module enhances long-term association by:
- Maintaining gallery/query datasets of detected persons
- Detecting abnormal track additions (new IDs appearing in unusual positions)
- Detecting abnormal track removals (IDs disappearing)
- Re-identifying persons using BPBreID when abnormal events occur
- Renewing track IDs based on re-identification results

Key files:
- `tracking/pmmm_scripts/trackreid_pmmm.py`: Main PMMM class
- `tracking/pmmm_scripts/scripts.py`: Helper functions for abnormal detection and dataset management

## Tracklet Linking System

The tracklet linking system clusters fragmented tracklets into complete person trajectories:

**Clustering methods:**
- `hierarchical`: Agglomerative clustering (recommended, stable)
- `dbscan`: Density-based clustering (good for noisy data)

**Distance metrics:**
- `cosine`: Recommended for normalized ReID features
- `euclidean`: For unnormalized features

**Tuning guidelines:**
- Lower threshold (0.2-0.3): Strict clustering, more persons identified
- Medium threshold (0.4-0.6): Balanced (recommended)
- Higher threshold (0.7-0.9): Loose clustering, fewer persons identified

## Video Naming Convention

Videos must follow strict naming format:
```
YYYY-MM-DD-HH-MM-SS_YYYY-MM-DD-HH-MM-SS.mp4
Example: 2025-07-02-14-25-39_2025-07-02-14-40-40.mp4
```

Directory structure:
```
video_data_source/
├── <scenario_name>/
│   ├── <camera_name>/
│   │   ├── <timestamp>_<timestamp>.mp4
│   │   └── ...
```

## Output Format

**MOT format (results.txt):**
```
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>
```

**Directory structure:**
```
mot_results/
├── <scenario_name>/
│   └── batch_<XXXX>/
│       └── <camera_name>/
│           └── <video_name>/
│               ├── results.txt      # MOT format tracking results
│               ├── results.mp4      # Visualization video
│               ├── crops/           # Cropped detections
│               ├── clops/           # PMMM crops
│               ├── gallery/         # PMMM gallery dataset
│               └── query/           # PMMM query dataset
```

## Important Notes

- Python 3.8+ required, PyTorch 2.0+ with CUDA support
- GPU memory: Ensure sufficient VRAM for YOLO + ReID models
- The system uses SQLite for tracking metadata - avoid concurrent writes
- Tracklet linking quality depends heavily on ReID feature quality
- Different scenarios may require different distance thresholds for linking
- When modifying tracking algorithms, edit configs in `boxmot/configs/` rather than code
- BPBreID uses part-based features - mask preprocessing affects performance
- For custom ReID datasets, ensure proper train/test split and identity labels
