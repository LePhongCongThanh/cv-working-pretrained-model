from ultralytics import YOLO
import cv2
import os
import argparse

def run_inference(source, model_path=None):
    """
    Run inference on a source (image, video, directory, or 0 for webcam).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    if model_path is None:
        # Default to the trained model if exists, else yolov8n
        default_model = os.path.join(project_root, 'models', 'helmet_vest_model', 'weights', 'best.pt')
        if os.path.exists(default_model):
            model_path = default_model
            print(f"Using trained model: {model_path}")
        else:
            model_path = 'yolov8n.pt'
            print("Trained model not found, using generic yolov8n.pt")

    model = YOLO(model_path)

    # Run inference
    results = model.predict(source, save=True, conf=0.5, show=True)
    
    for r in results:
        print(r.boxes.data)  # print box coordinates and classes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run YOLOv8 Inference')
    parser.add_argument('--source', type=str, required=True, help='Path to image/video or 0 for webcam')
    parser.add_argument('--model', type=str, default=None, help='Path to .pt model file')
    
    args = parser.parse_args()
    
    run_inference(args.source, args.model)
