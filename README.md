# Helmet and Safety Vest Detection

This project uses YOLOv8 to detect helmets and safety vests.

## Project Structure
```
d:/CV_Model/
├── data/
│   ├── raw/                  # Place your raw images here
│   ├── processed/            # Processed data
│   └── dataset.yaml          # YOLO dataset config
├── models/                   # Trained models will be saved here
├── src/
│   ├── train.py              # Training script
│   ├── predict.py            # Inference script
│   └── utils.py              # Helper functions
├── notebooks/                # Jupyter notebooks
└── requirements.txt          # Python dependencies
```

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare your dataset:
   - Organize your images and labels in `data/images` and `data/labels` (train/val split).
   - Update `data/src/dataset.yaml` if your paths differ.

## Usage

### Training
To train the model (starts with pretrained `yolov8n.pt`):
```bash
python src/train.py
```
This will start training for 100 epochs (default) and save the best model to `models/helmet_vest_model/`.

### Inference
To run detection on an image or video:
```bash
python src/predict.py --source path/to/image.jpg
```
To run on webcam:
```bash
python src/predict.py --source 0
```
Use the `--model` argument to specify a custom model path if different from the default trained one.
