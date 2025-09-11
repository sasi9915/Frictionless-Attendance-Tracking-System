import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
import logging
import threading
import queue
from counters_shared import zone_counters, counter_lock

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add after imports
detection_started = False
display_enabled = False  # set True if you still want OpenCV windows

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
        centroids = np.array([[box[0], box[1]] for box in boxes])
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

def reconnect_stream(rtsp_url, gstreamer_pipeline, max_attempts=5, delay=5):
    for attempt in range(max_attempts):
        cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            logger.info(f"Reconnected to stream using GStreamer pipeline on attempt {attempt + 1}")
            return cap
        cap.release()
        cap = cv2.VideoCapture(rtsp_url)
        if cap.isOpened():
            logger.info(f"Reconnected to stream using FFmpeg backend on attempt {attempt + 1}")
            return cap
        cap.release()
        logger.warning(f"Reconnect attempt {attempt + 1} failed. Retrying in {delay} seconds...")
        time.sleep(delay)
    logger.error("All reconnect attempts failed.")
    return None

def crop_face(frame, box, expand_ratio=0.15):
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

def process_stream(rtsp_url, gstreamer_pipeline, output_dir, video_out_path, zone_name, cam_type, display_queue):
    global zone_counters
    cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        logger.warning(f"{zone_name}-{cam_type}: GStreamer pipeline failed. Falling back to FFmpeg backend...")
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            logger.error(f"{zone_name}-{cam_type}: Failed to open RTSP stream with both GStreamer and FFmpeg backends")
            return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_out_path, fourcc, fps, (frame_width, frame_height))
    tracker = CentroidTracker(max_disappeared=30)
    saved_faces = {}
    counted_ids = set()  # Track IDs already counted for this stream
    time_window = 5
    last_boxes = []
    frame_count = 0
    face_counter = 0
    start_time = time.time()
    logger.info(f"Processing RTSP stream: {rtsp_url} ({zone_name}-{cam_type})...")

    while True:
        success, frame = cap.read()
        if not success:
            logger.warning(f"{zone_name}-{cam_type}: Failed to read frame. Attempting to reconnect...")
            cap.release()
            cap = reconnect_stream(rtsp_url, gstreamer_pipeline, 5, 5)
            if cap is None:
                break
            continue
        frame_count += 1
        display_frame = frame.copy()
        current_count = 0  # Real-time count for this frame
        new_ids_this_frame = set()

        if frame_count % 5 == 0:
            results = trt_model(frame, verbose=False, conf=0.5, device='0', imgsz=640)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            xywh_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                x, y = x1 + w / 2, y1 + h / 2
                xywh_boxes.append([x, y, w, h])
            xywh_boxes = np.array(xywh_boxes)
            tracked_objects = tracker.update(xywh_boxes)
            current_detections = []
            for i, (box, score) in enumerate(zip(boxes, scores)):
                if score > 0.5:
                    face_crop = crop_face(frame, box, expand_ratio=0.15)
                    if face_crop.size == 0:
                        continue
                    current_detections.append({"box": box, "score": score, "face_crop": face_crop})
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
                    continue
                face_crop = best_detection["face_crop"]
                score = best_detection["score"]
                box = best_detection["box"]
                current_time = time.time()
                last_boxes.append(box)
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f'{zone_name}-{cam_type} {track_id} ({score:.2f})', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                current_count += 1
                # ONLY COUNT IF TRACK_ID HAS NOT BEEN COUNTED YET
                if track_id not in counted_ids:
                    if track_id not in saved_faces or (current_time - saved_faces[track_id]) >= time_window:
                        min_size = 64
                        if face_crop.size > 0 and face_crop.shape[0] >= min_size and face_crop.shape[1] >= min_size:
                            if is_sharp(face_crop, threshold=100):
                                face_filename = os.path.join(output_dir, f"{zone_name}_{cam_type}_face_{track_id:04d}_f{frame_count}.jpg")
                                success = cv2.imwrite(face_filename, face_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                                if success:
                                    saved_faces[track_id] = current_time
                                    face_counter += 1
                                    new_ids_this_frame.add(track_id)
            # Detect sudden count jumps and ignore them
            num_new_ids = len(new_ids_this_frame)
            if num_new_ids > 10:
                logger.warning(f"Sudden count jump detected: {num_new_ids} new faces in frame {frame_count}. Ignoring increment for this frame.")
            else:
                counted_ids.update(new_ids_this_frame)
                with counter_lock:
                    for track_id in new_ids_this_frame:
                        if cam_type == 'entrance':
                            zone_counters[zone_name]['entrance'] += 1
                            zone_counters[zone_name]['zone_count'] += 1
                        elif cam_type == 'exit':
                            zone_counters[zone_name]['exit'] += 1
                            zone_counters[zone_name]['zone_count'] = max(0, zone_counters[zone_name]['zone_count'] - 1)
        else:
            for box in last_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f'{zone_name}-{cam_type}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            current_count = len(last_boxes)
        with counter_lock:
            cv2.putText(display_frame, f"{zone_name}-{cam_type} Real-time: {current_count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(display_frame, f"Entrance Total: {zone_counters[zone_name]['entrance']}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 128, 0), 2)
            cv2.putText(display_frame, f"Exit Total: {zone_counters[zone_name]['exit']}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 0, 0), 2)
            cv2.putText(display_frame, f"Zone Count: {zone_counters[zone_name]['zone_count']}", (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 128), 2)
        out.write(display_frame)
        display_queue.put((f'Live Face Detection - {zone_name}-{cam_type}', display_frame, current_count))
        if frame_count % 10 == 0:
            logger.info(f"{face_counter} faces ({zone_name}-{cam_type})")
    cap.release()
    out.release()
    total_time = time.time() - start_time
    logger.info(f"\n{zone_name}-{cam_type} complete!")
    logger.info(f"Total frames processed: {frame_count}")
    logger.info(f"Total faces saved: {face_counter}")
    logger.info(f"Faces saved to: {output_dir}")
    logger.info(f"Annotated video saved to: {video_out_path}")
    logger.info(f"Processing time: {total_time:.1f} seconds")
    logger.info(f"Average FPS: {frame_count / total_time:.1f}")

def start_detection():
    global detection_started
    if detection_started:
        return
    detection_started = True
    logger.info("Starting multi-zone detection threads...")
    zone_configs = {
        "zone1": {
            "entrance": {
                "url": "rtsp://admin:FYP12345@10.40.16.236:554/Streaming/Channels/101",
                "gstreamer": "rtspsrc location=rtsp://admin:FYP12345@10.40.16.236:554/Streaming/Channels/101 latency=200 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGR ! appsink",
                "output_dir": "detected_faces/zone1/entrance",
                "video_out": "output_zone1_entrance.mp4",
            },
            "exit": {
                "url": "rtsp://admin:@Deee123@10.40.16.217:554/Streaming/Channels/101",
                "gstreamer": "rtspsrc location=rtsp://admin:@Deee123@10.40.16.217:554/Streaming/Channels/101 latency=200 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGR ! appsink",
                "output_dir": "detected_faces/zone1/exit",
                "video_out": "output_zone1_exit.mp4",
            }
        },
        "zone2": {
            "entrance": {
                "url": "rtsp://admin:@Deee123@10.40.16.196:554/Streaming/Channels/101",
                "gstreamer": "rtspsrc location=rtsp://admin:@Deee123@10.40.16.196:554/Streaming/Channels/101 latency=200 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGR ! appsink",
                "output_dir": "detected_faces/zone2/entrance",
                "video_out": "output_zone2_entrance.mp4",
            },
            "exit": {
                "url": "rtsp://admin:DeeeEngex@10.40.16.214:554/Streaming/Channels/101",
                "gstreamer": "rtspsrc location=rtsp://admin:DeeeEngex@10.40.16.214:554/Streaming/Channels/101 latency=200 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGR ! appsink",
                "output_dir": "detected_faces/zone2/exit",
                "video_out": "output_zone2_exit.mp4",
            }
        }
    }
    for zone, cams in zone_configs.items():
        for cam_type, conf in cams.items():
            os.makedirs(conf['output_dir'], exist_ok=True)

    display_queue = queue.Queue()
    threads = []
    for zone, cams in zone_configs.items():
        for cam_type, conf in cams.items():
            t = threading.Thread(
                target=process_stream,
                args=(conf['url'], conf['gstreamer'], conf['output_dir'], conf['video_out'],
                      zone, cam_type, display_queue),
                daemon=True
            )
            t.start()
            threads.append(t)

    def consumer():
        while True:
            try:
                _, frame, _ = display_queue.get(timeout=5)
                if display_enabled:
                    cv2.imshow("Streams", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            except queue.Empty:
                if not any(t.is_alive() for t in threads):
                    break
        if display_enabled:
            cv2.destroyAllWindows()
        logger.info("Detection threads finished.")

    threading.Thread(target=consumer, daemon=True).start()

if __name__ == "__main__":
    start_detection()
    # Keep main thread alive
    try:
        while True:
            time.sleep(2)
    except KeyboardInterrupt:
        pass
