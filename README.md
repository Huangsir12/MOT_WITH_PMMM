# MOT_WITH_PMMM

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Pytorch Version](https://img.shields.io/badge/pytorch-2.4%2B-orange)](https://pytorch.org/)

Position and Multi-step Memory Matching (PMMM) module for enhancing long-term association accuracy in Multi-Object Tracking (MOT).

## 📌 Overview

This repository contains the implementation of PMMM - a novel module that integrates positional cues with multi-frame historical context to improve long-term association accuracy in multi-object tracking scenarios.

Key features:
- Position-aware memory matching
- Multi-step historical context integration
- Enhanced long-term association
- [Add other key features]

## Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- [Other dependencies]

### Installation
```bash
git clone https://github.com/Huangsir12/MOT_WITH_PMMM.git
cd MOT_WITH_PMMM
conda create -n pmmm python=3.9
pip install -r requirements.txt
```
## 🛠️ Usage

###  Training
Firstly, we can train appearance feature representation model(reid_model)
```bash
conda activate pmmm
cd bpbreid/torchreid
python ./scripts/main.py --config-file configs/bpbreid/bpbreid_<target_dataset>_train.yaml
```
Then, we can train detection model.Train YOLO10x on the COCO8 dataset(or other coustom dataset with yolo format) for 100 epochs at image size 640. The training device can be specified using the device argument. If no argument is passed GPU device=0 will be used if available, otherwise device='cpu' will be used. See Arguments section below for a full list of training arguments.

```bash
# Build a new model from YAML and start training from scratch
yolo detect train data=coco8.yaml model=yolo11n.yaml epochs=100 imgsz=640

# Start training from a pretrained *.pt model
yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640

# Build a new model from YAML, transfer pretrained weights to it and start training
yolo detect train data=coco8.yaml model=yolo11n.yaml pretrained=yolo11n.pt epochs=100 imgsz=640
```

###  Evaluation
Evaluate a combination of detector, tracking method and ReID model on standard MOT dataset or you custom one. You can change detection model, reid model, track method and benchmark(MOT17, MOT20, you custom one)
```bash
python tracking/val.py --yolo-model WEIGHTS / 'your_trained.pt' --reid-model WEIGHTS / 'your_trained.pt' --tracking_method botsort --source DATA / "datasets" / "Emporium" / "train" --ues_pmmm True
```

### Inference
track without pmmm module
```bash
cd MOT_WITH_PMMM
python tracking/track.py --yolo-model WEIGHTS / 'your_trained.pt' --reid-model WEIGHTS / 'your_trained.pt' --tracking_method botsort
```
track with pmmm module
```bash
python tracking/track_with_pmmm.py --yolo-model WEIGHTS / 'your_trained.pt' --reid-model WEIGHTS / 'your_trained.pt' --tracking_method botsort
```

## 🧩 PMMM Module Architecture
PMMM Architecture

The PMMM module consists of:
-Position-aware Branch
-Multi-step Memory Bank
-Attention-based  Cross-frame Matching mechanism

## 📈 Performance
Benchmark Results
Dataset	MOTA ↑	IDF1 ↑	FP ↓	FN ↓	IDs ↓
MOT17	xx	xx	xx	xx	xx
MOT20	xx	xx	xx	xx	xx
Custom	xx	xx	xx	xx	xx

## 🔗 References

This project builds upon these excellent works:

1. **FairMOT** ([GitHub](https://github.com/VlSomers/bpbreid/))
   - Used the JDE-based framework as our baseline
   - Modified the original detection head implementation

2. **TransTrack** ([GitHub](https://github.com/mikel-brostrom/boxmot))
   - Adapted parts of the attention mechanism
   - Inspired our memory bank design

We sincerely thank the original authors for their work.

## 📜 Citation
If you use this work in your research, please cite:

BIBTEX
@article{yourcitation,
  title={MOT_WITH_PMMM: Position and Multi-step Memory Matching for Long-term Association},
  author={Your Name},
  journal={Journal or Conference Name},
  year={2023}
}

## 🤝 Contributing
We welcome contributions! Please see CONTRIBUTING.md for guidelines.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## ✉️ Contact
For questions or suggestions, please contact:
huangming  -@qq.com
Project Link: https://github.com/Huangsir12/MOT_WITH_PMMM