from ultralytics import YOLO
import cv2
import os
import numpy as np

# Initialize YOLO model
model = YOLO('yolov8n-face.pt')  # Replace with your custom model if available

# Open video file
video_path = "source/video_2.mp4"
cap = cv2.VideoCapture(video_path)

# Create a directory to save cropped faces
output_dir = "detected_faces"
os.makedirs(output_dir, exist_ok=True)

frame_count = 0  # To track the number of frames processed
saved_faces = []  # List to store saved bounding boxes

# Function to calculate Intersection over Union (IoU)
def calculate_iou(box1, box2):
    # box format: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break  # Break the loop if the video ends
    
    frame_count += 1
    print(f"Processing frame {frame_count}")

    # Perform detection on the frame
    results = model.predict(source=frame, conf=0.5)  # Adjust confidence threshold as needed

    # Extract bounding boxes
    detections = results[0].boxes.xyxy.numpy()  # Get bounding boxes in [x1, y1, x2, y2] format
    confidence_scores = results[0].boxes.conf.numpy()  # Confidence scores

    # Loop through detections and crop faces
    for idx, box in enumerate(detections):
        x1, y1, x2, y2 = map(int, box)  # Convert bounding box to integer
        cropped_face = frame[y1:y2, x1:x2]  # Crop the face region

        # Check if the face is unique using IoU
        is_unique = True
        for saved_box in saved_faces:
            iou = calculate_iou(box, saved_box)
            if iou > 0.5:  # Adjust IoU threshold as needed (e.g., 0.5 for 50% overlap)
                is_unique = False
                break

        if is_unique:
            # Save the face and add its bounding box to the saved list
            output_path = os.path.join(output_dir, f"face_{frame_count}_{idx + 1}.jpg")
            cv2.imwrite(output_path, cropped_face)
            saved_faces.append(box)  # Save the bounding box

            print(f"Face {idx + 1} from frame {frame_count} saved at {output_path}")

    # Visualize detections on the original frame (optional)
    for box in detections:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box

    # Show the result (optional, comment out if not needed)
    cv2.imshow("Detected Faces", frame)

    # Break the loop when 'q' key is pressed (optional)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()

