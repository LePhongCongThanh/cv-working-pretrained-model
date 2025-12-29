from ultralytics import YOLO
import os

def train_model(epochs=100, imgsz=640, batch=16):
    """
    Train YOLOv8 model on custom dataset.
    """
    # Load a pretrained model
    model = YOLO('yolov8n.pt') 

    # Define path to dataset.yaml
    # Assuming this script is run from src/ or project root, we need to handle paths correctly.
    # Recommended to run from project root: python src/train.py
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dataset_yaml = os.path.join(project_root, 'data', 'dataset.yaml')
    models_dir = os.path.join(project_root, 'models')

    print(f"Training with config: {dataset_yaml}")
    
    # Train the model
    # project argument specifies the directory to save results
    results = model.train(
        data=dataset_yaml, 
        epochs=epochs, 
        imgsz=imgsz, 
        batch=batch,
        project=models_dir,
        name='helmet_vest_model',
        exist_ok=True # overwrite existing experiment
    )
    
    # Validation
    metrics = model.val()
    print("Validation metrics:", metrics)
    
    # Export the model
    success = model.export(format='onnx')
    print("Model exported to ONNX:", success)

if __name__ == '__main__':
    # You can parse arguments here if needed
    train_model(epochs=50) # start with 50 epochs for example
