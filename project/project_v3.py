from ultralytics import YOLO
import cv2
import os
import numpy as np
import mediapipe as mp

# Initialize YOLO model
model = YOLO('yolov8n-face.pt')  # Replace with your custom model if available

# Initialize Mediapipe Face Mesh for facial landmarks
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Open video file
video_path = "source/video_2.mp4"
cap = cv2.VideoCapture(video_path)

# Create a directory to save aligned faces
output_dir = "aligned_faces"
os.makedirs(output_dir, exist_ok=True)

frame_count = 0  # To track the number of frames processed
saved_faces = []  # List to store saved bounding boxes

# Function to align the face
def align_face(image, box, landmarks):
    # Get bounding box coordinates
    x1, y1, x2, y2 = map(int, box)
    face_roi = image[y1:y2, x1:x2]

    # Extract key points for alignment
    left_eye = landmarks[33]  # Left eye landmark
    right_eye = landmarks[263]  # Right eye landmark
    nose_tip = landmarks[1]  # Nose tip landmark

    # Calculate the angle between eyes
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Get center of the eyes
    eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

    # Rotate the face to align
    rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    aligned_face = cv2.warpAffine(face_roi, rotation_matrix, (face_roi.shape[1], face_roi.shape[0]))
    
    return aligned_face

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

    # Loop through detections and crop faces
    for idx, box in enumerate(detections):
        x1, y1, x2, y2 = map(int, box)
        face_roi = frame[y1:y2, x1:x2]

        # Perform facial landmark detection
        rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        results_landmarks = face_mesh.process(rgb_face)

        if results_landmarks.multi_face_landmarks:
            landmarks = results_landmarks.multi_face_landmarks[0]
            # Convert landmarks to pixel coordinates
            height, width, _ = face_roi.shape
            landmark_points = [(int(l.x * width), int(l.y * height)) for l in landmarks.landmark]

            # Align the face
            aligned_face = align_face(frame, box, landmark_points)

            # Save aligned face
            output_path = os.path.join(output_dir, f"aligned_face_{frame_count}_{idx + 1}.jpg")
            cv2.imwrite(output_path, aligned_face)
            print(f"Aligned face {idx + 1} from frame {frame_count} saved at {output_path}")

    # Show the result (optional)
    cv2.imshow("Processed Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
