from ultralytics import YOLO
import cv2
import os

# Initialize YOLO model
# Use a pre-trained face detection model or fine-tuned YOLO model
model = YOLO('yolov8n.pt')  # Replace with your custom model if available

# Load the image
image_path = "source/frame_1.jpg"
image = cv2.imread(image_path)

# Perform detection
results = model.predict(source=image_path, conf=0.7)  # Adjust confidence threshold as needed

# Extract bounding boxes
detections = results[0].boxes.xyxy.numpy()  # Get bounding boxes in [x1, y1, x2, y2] format
confidence_scores = results[0].boxes.conf.numpy()  # Confidence scores

# Create a directory to save cropped faces
output_dir = "detected_faces"
os.makedirs(output_dir, exist_ok=True)

# Loop through detections and crop faces
for idx, box in enumerate(detections):
    x1, y1, x2, y2 = map(int, box)  # Convert bounding box to integer
    cropped_face = image[y1:y2, x1:x2]  # Crop the face region

    # Save the cropped face
    output_path = os.path.join(output_dir, f"face_{idx + 1}.jpg")
    cv2.imwrite(output_path, cropped_face)

    print(f"Face {idx + 1} saved at {output_path}")
    print(model)
# Visualize detections on the original image (optional)
for box in detections:
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box

# Show the result (optional)
cv2.imshow("Detected Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
