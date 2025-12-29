# Helmet Detection Implementation Plan

## Goal
Create a complete project structure for training a helmet detection model using pretrained YOLO (specifically YOLOv8 via `ultralytics`).

## Proposed File Structure
```
d:/CV_Model/
├── data/
│   ├── raw/                  # Original dataset images/labels
│   ├── processed/            # Processed/Splitted data
│   └── dataset.yaml          # YOLO dataset configuration
├── models/                   # Directory to save trained models
├── src/
│   ├── __init__.py
│   ├── train.py              # Script to run training
│   ├── predict.py            # Script to run inference/detection
│   └── utils.py              # Helper functions
├── notebooks/                # Jupyter notebooks for experiments
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## Dependencies
- `ultralytics` (YOLOv8)
- `opencv-python`
- `matplotlib`
- `numpy`
- `pandas`

## Component Details

### Configuration (`data/dataset.yaml`)
Standard YOLO configuration file defining paths to train/val images and class names.
Example classes: `['helmet', 'no-helmet', 'vest', 'no-vest']` or simply `['helmet', 'vest']` depending on your specific labeling strategy.

### Training (`src/train.py`)
Script to load a pretrained YOLOv8 model (e.g., `yolov8n.pt`) and finetune it on the custom dataset.

### Inference (`src/predict.py`)
Script to load a trained model and run detection on images, videos, or webcam feed.
