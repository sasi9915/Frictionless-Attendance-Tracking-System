import cv2
from ultralytics import YOLO
import os
import time
import numpy as np
import pycuda.autoinit  # Ensures proper GPU context initialization

# -------------------------- SETUP -------------------------- #

# Define file paths
input_video_path = os.path.expanduser("~/cctv_footage.mp4")  # Change this to your video path
output_video_path = os.path.expanduser("~/output_video_with_boxes.mp4")
detected_faces_dir = os.path.expanduser("~/detected_faces")

# Ensure output directory exists
os.makedirs(detected_faces_dir, exist_ok=True)

# Load YOLO model (Check if TensorRT model exists, else create it)
model_path = os.path.expanduser("~/yolov8n-face.pt")  # PyTorch model
trt_model_path = os.path.expanduser("~/yolov8n-face.engine")  # TensorRT model

if not os.path.exists(trt_model_path):
    print("Exporting TensorRT model... This might take a while.")
    model = YOLO(model_path)
    model.export(format="engine", device=0, imgsz=640)  # Export to TensorRT on GPU
    print(f"TensorRT model saved at: {trt_model_path}")

# Load the optimized TensorRT model
model = YOLO(trt_model_path, task='detect')
print("TensorRT model loaded successfully on GPU!")

# Open the video file
cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# Video writer setup (H.265 for Jetson hardware encoding)
fourcc = cv2.VideoWriter_fourcc(*'H265')
out = cv2.VideoWriter(output_video_path, fourcc, int(fps), (frame_width, frame_height))

# ---------------------- PROCESSING LOOP ---------------------- #

frame_count = 0
face_counter = 0
start_time = time.time()

print("Starting face detection on recorded video...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    display_frame = frame.copy()

    # Process every frame (optimized for performance)
    if frame_count % 2 == 0:  # Skip alternate frames to improve FPS
        results = model.predict(frame, conf=0.5, device=0, verbose=False, imgsz=640)

        # Process detections
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # Convert to CPU
                confidences = result.boxes.conf.cpu().numpy()

                for (box, conf) in zip(boxes, confidences):
                    x1, y1, x2, y2 = map(int, box[:4])

                    # Draw bounding box and label
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, f'Face {face_counter}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Save detected face as an image
                    face_img = frame[y1:y2, x1:x2]
                    if face_img.size != 0:
                        face_filename = os.path.join(detected_faces_dir, f"face_{face_counter:04d}.jpg")
                        cv2.imwrite(face_filename, face_img)
                        face_counter += 1

    # Write the frame to the output video
    out.write(display_frame)

    # Show progress
    if frame_count % 10 == 0:
        elapsed = time.time() - start_time
        fps = frame_count / elapsed
        print(f"FPS: {fps:.1f} | Faces Detected: {face_counter} | Frames Processed: {frame_count}")

# ---------------------- CLEANUP ---------------------- #

cap.release()
out.release()
cv2.destroyAllWindows()

# Performance report
total_time = time.time() - start_time
print(f"\nFinal Performance Report:")
print(f"Total Frames Processed: {frame_count}")
print(f"Average FPS: {frame_count / total_time:.1f}")
print(f"Total Faces Detected: {face_counter}")
print(f"Processed video saved at: {output_video_path}")
print(f"Detected face images saved in: {detected_faces_dir}")
