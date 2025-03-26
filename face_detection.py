import cv2
from ultralytics import YOLO
import os
import time

# Initialize YOLOv8 face detector
model = YOLO('yolov8n-face.pt')

# Video input setup
video_path = 'source/Video 3.mp4'
cap = cv2.VideoCapture(video_path)

# Create output directories
output_dir = 'detected_faces'
os.makedirs(output_dir, exist_ok=True)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video writer setup
output_video_path = 'output_video_with_boxes.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Initialize counters and trackers
frame_count = 0
face_counter = 0
start_time = time.time()
last_boxes = []  # To maintain boxes between processed frames

print(f"Processing {video_path}...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    frame_count += 1
    display_frame = frame.copy()
    
    # Process 1 frame every 5 frames
    if frame_count % 5 == 0:
        # Perform face detection on current frame
        results = model(frame, verbose=False, conf=0.5, device='cpu')
        last_boxes = []
        
        # Process new detections
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            
            for idx, (box, conf) in enumerate(zip(boxes, confidences)):
                x1, y1, x2, y2 = map(int, box[:4])
                last_boxes.append((x1, y1, x2, y2))
                
                # Draw bounding box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f'Face {face_counter}', (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                
                # Crop and save face
                face_img = frame[y1:y2, x1:x2]
                if face_img.size != 0:
                    face_filename = os.path.join(output_dir, 
                        f"face_{face_counter:04d}_f{frame_count}.jpg")
                    cv2.imwrite(face_filename, face_img)
                    face_counter += 1
    
    # Draw previous boxes on non-processed frames
    else:
        for (x1, y1, x2, y2) in last_boxes:
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, f'Face', (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    
    # Write to output video
    out.write(display_frame)
    
    # Display real-time preview
    cv2.imshow('Live Face Detection', display_frame)
    
    # Print progress
    if frame_count % 10 == 0:
        elapsed = time.time() - start_time
        print(f"Processed {frame_count}/{total_frames} frames - Found {face_counter} faces")
    
    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

# Final report
total_time = time.time() - start_time
print(f"\nProcessing complete!")
print(f"Total faces detected: {face_counter}")
print(f"Faces saved to: {output_dir}")
print(f"Annotated video saved to: {output_video_path}")
print(f"Processing time: {total_time:.1f} seconds")