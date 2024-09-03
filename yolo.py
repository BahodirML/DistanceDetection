import torch
import cv2
import json
from pathlib import Path

# Known parameters
FOCAL_LENGTH = 800  # Example value, in pixels
KNOWN_HEIGHT = 1.7  # Example: height of a person in meters

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Path to your video file
video_path = 'cars.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Create a directory for JSON files
output_json_dir = Path('./output_json')
output_json_dir.mkdir(exist_ok=True)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Perform inference on the frame
    results = model(frame)
    
    # Extract data for each detected object
    for i, row in results.pandas().xyxy[0].iterrows():
        x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        
        # Height of the bounding box in pixels
        pixel_height = y_max - y_min
        
        # Estimate the distance to the object
        distance = (KNOWN_HEIGHT * FOCAL_LENGTH) / pixel_height
        
        # Prepare the object data
        object_data = {
            "frame_id": frame_count,
            "bounding_box": [x_min, y_min, x_max, y_max],
            "distance": distance,
            "label": row['name']  # Assuming 'name' column contains object class
        }
        
        # Save object data to a JSON file
        json_file_path = output_json_dir / f'frame_{frame_count:04d}_object_{i:02d}.json'
        with open(json_file_path, 'w') as json_file:
            json.dump(object_data, json_file, indent=4)
    
    print(f"Processed frame {frame_count:04d}")
    frame_count += 1

# Release the video capture object
cap.release()

print(f"Processed {frame_count} frames. JSON files saved in {output_json_dir}")
