import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Centroid tracking class
class CentroidTracker:
    def __init__(self, max_disappeared=30):
        self.next_object_id = 0
        self.objects = {}  # {object_id: centroid}
        self.disappeared = {}  # {object_id: frames_disappeared}
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, boxes):
        if len(boxes) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        centroids = np.array([[box[0], box[1]] for box in boxes])  # Use box center (x, y)
        if len(self.objects) == 0:
            for centroid in centroids:
                self.register(centroid)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array(list(self.objects.values()))
            distances = np.sqrt(((object_centroids[:, np.newaxis] - centroids) ** 2).sum(axis=2))
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]
            used_rows, used_cols = set(), set()
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)
            unused_rows = set(range(distances.shape[0])).difference(used_rows)
            unused_cols = set(range(distances.shape[1])).difference(used_cols)
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            for col in unused_cols:
                self.register(centroids[col])
        return self.objects

# Helper function to crop face with bounding box expansion
def crop_face(frame, box, expand_ratio=0.15):
    """Crop face from frame given YOLOv8 xyxy box, expanding by expand_ratio."""
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - w * expand_ratio))
    y1 = max(0, int(y1 - h * expand_ratio))
    x2 = min(frame.shape[1], int(x2 + w * expand_ratio))
    y2 = min(frame.shape[0], int(y2 + h * expand_ratio))
    if x2 <= x1 or y2 <= y1:
        logger.warning(f"Invalid crop: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        return np.array([])
    return frame[y1:y2, x1:x2]

# Helper function to check if an image is sharp (not blurry)
def is_sharp(image, threshold=100):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return cv2.Laplacian(gray, cv2.CV_64F).var() > threshold

# Initialize YOLOv8 face detector with TensorRT engine
try:
    trt_model = YOLO("yolov8n-face.engine")
except Exception as e:
    logger.error(f"Failed to load TensorRT model: {e}")
    exit(1)

# RTSP input setup
rtsp_url = "rtsp://admin:FYP12345@10.40.16.236:554/Streaming/Channels/101"
gstreamer_pipeline = (
    f"rtspsrc location={rtsp_url} latency=200 ! "
    "rtph264depay ! h264parse ! nvv4l2decoder ! "
    "nvvidconv ! video/x-raw,format=BGR ! appsink"
)

cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    logger.warning("GStreamer pipeline failed. Falling back to FFmpeg backend...")
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        logger.error("Failed to open RTSP stream with both GStreamer and FFmpeg backends")
        exit(1)

# Reconnection logic
max_reconnect_attempts = 5
reconnect_delay = 5

def reconnect_stream(rtsp_url, gstreamer_pipeline, max_attempts, delay):
    for attempt in range(max_attempts):
        logger.info(f"Reconnection attempt {attempt + 1}/{max_attempts}")
        cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            logger.info("Reconnected successfully with GStreamer")
            return cap
        logger.warning("GStreamer reconnection failed. Trying FFmpeg...")
        cap = cv2.VideoCapture(rtsp_url)
        if cap.isOpened():
            logger.info("Reconnected successfully with FFmpeg")
            return cap
        logger.warning(f"Reconnection failed. Retrying in {delay} seconds...")
        time.sleep(delay)
    logger.error("Max reconnection attempts reached. Exiting...")
    return None

# Create output directories
output_dir = 'detected_faces'
os.makedirs(output_dir, exist_ok=True)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Fallback to 30 if FPS not detected
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
logger.info(f"RTSP Stream: {rtsp_url}, Resolution: {frame_width}x{frame_height}, FPS: {fps}")

# Video writer setup
output_video_path = 'output_video_with_boxes.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Initialize counters, trackers, and state
frame_count = 0
face_counter = 0
start_time = time.time()
tracker = CentroidTracker(max_disappeared=30)
saved_faces = {}  # {track_id: timestamp}
time_window = 5  # seconds to prevent duplicate saves
last_boxes = []

logger.info(f"Processing RTSP stream: {rtsp_url}...")

while True:
    success, frame = cap.read()
    if not success:
        logger.warning("Failed to read frame. Attempting to reconnect...")
        cap.release()
        cap = reconnect_stream(rtsp_url, gstreamer_pipeline, max_reconnect_attempts, reconnect_delay)
        if cap is None:
            break
        continue

    frame_count += 1
    display_frame = frame.copy()

    # Process every 5th frame to reduce load
    if frame_count % 5 == 0:
        # Perform face detection
        results = trt_model(frame, verbose=False, conf=0.5, device='0', imgsz=640)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()

        # Convert xyxy to xywh for centroid tracking
        xywh_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            x, y = x1 + w / 2, y1 + h / 2  # Center point
            xywh_boxes.append([x, y, w, h])
        xywh_boxes = np.array(xywh_boxes)

        # Update tracker
        tracked_objects = tracker.update(xywh_boxes)

        current_detections = []
        for i, (box, score) in enumerate(zip(boxes, scores)):
            if score > 0.5:
                face_crop = crop_face(frame, box, expand_ratio=0.15)
                if face_crop.size == 0:
                    logger.warning(f"Frame {frame_count}: Empty face crop for box {box}")
                    continue
                current_detections.append({"box": box, "score": score, "face_crop": face_crop})
            else:
                logger.info(f"Frame {frame_count}: Skipped face {i} with low confidence {score:.2f}")
        logger.info(f"Frame {frame_count}: Detected {len(boxes)} faces, {len(current_detections)} valid detections")

        # Assign track IDs to detections
        last_boxes = []
        for track_id, centroid in tracked_objects.items():
            min_dist = float('inf')
            best_detection = None
            for detection in current_detections:
                box = detection["box"]
                detection_centroid = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
                dist = np.sqrt(((centroid - detection_centroid) ** 2).sum())
                if dist < min_dist:
                    min_dist = dist
                    best_detection = detection
            if best_detection is None:
                logger.info(f"Frame {frame_count}: No detection matched for track {track_id}")
                continue
            face_crop = best_detection["face_crop"]
            score = best_detection["score"]
            box = best_detection["box"]
            current_time = time.time()

            # Store box for drawing on non-processed frames
            last_boxes.append(box)
            # Draw bounding box
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, f'Face {track_id} ({score:.2f})', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Check if this track_id was recently saved
            if track_id in saved_faces:
                last_saved_time = saved_faces[track_id]
                if (current_time - last_saved_time) < time_window:
                    logger.info(f"Frame {frame_count}: Skipping track {track_id} (recently saved at {last_saved_time})")
                    continue

            # Save the face if it is sharp and large enough
            min_size = 64
            if face_crop.size > 0 and face_crop.shape[0] >= min_size and face_crop.shape[1] >= min_size:
                if is_sharp(face_crop, threshold=100):
                    face_filename = os.path.join(output_dir, f"face_{track_id:04d}_f{frame_count}.jpg")
                    success = cv2.imwrite(face_filename, face_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    if success:
                        saved_faces[track_id] = current_time
                        face_counter += 1
                        logger.info(f"Frame {frame_count}: Saved sharp face for track {track_id} (score {score:.2f}) to {face_filename}")
                    else:
                        logger.error(f"Frame {frame_count}: Failed to save face for track {track_id} to {face_filename}")
                else:
                    logger.info(f"Frame {frame_count}: Skipped blurry face for track {track_id}")
            else:
                logger.info(f"Frame {frame_count}: Skipped small face for track {track_id}")

    # Draw previous boxes on non-processed frames
    else:
        for box in last_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, 'Face', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Write to output video
    out.write(display_frame)

    # Display real-time preview (disable to save resources if needed)
    cv2.imshow('Live Face Detection', display_frame)

    # Log progress
    if frame_count % 10 == 0:
        elapsed = time.time() - start_time
        logger.info(f"Processed {frame_count} frames - Found {face_counter} faces - Elapsed: {elapsed:.1f}s")

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

# Final report
total_time = time.time() - start_time
logger.info("\nProcessing complete!")
logger.info(f"Total frames processed: {frame_count}")
logger.info(f"Total faces saved: {face_counter}")
logger.info(f"Faces saved to: {output_dir}")
logger.info(f"Annotated video saved to: {output_video_path}")
logger.info(f"Processing time: {total_time:.1f} seconds")
logger.info(f"Average FPS: {frame_count / total_time:.1f}")
