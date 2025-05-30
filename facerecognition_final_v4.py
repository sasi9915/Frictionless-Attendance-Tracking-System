import sqlite3
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import torch
from insightface.app import FaceAnalysis
import numpy as np
import cv2
import os
import time
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import io
import queue
import shutil
import threading

# Enhanced Face Analyzer Configuration
face_analyzer = FaceAnalysis(
    name="buffalo_l",
    providers=['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider'],
    allowed_modules=['detection', 'recognition']
)
face_analyzer.prepare(
    ctx_id=0 if torch.cuda.is_available() else -1,
    det_size=(640, 640),
    det_thresh=0.4  # Lower detection threshold for unclear faces
)

# Database and folder paths
DB_PATH = "images_all.db"
INPUT_FOLDER = os.path.expanduser("detected_faces")
UNRECOGNIZED_FOLDER = os.path.expanduser("unrecognized_faces")
PROCESSED_FOLDER = os.path.expanduser("processed_faces")

# Create folders if they don't exist
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(UNRECOGNIZED_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

face_queue = queue.Queue()

# Configuration parameters
DISPLAY_DURATION = 20000  # 15 seconds display time
SIMILARITY_THRESHOLD = 0.4  # Adjusted for unclear faces
PROCESSING_INTERVAL = 20  # Check for new images every 3 seconds

def enhance_image(image):
    """Apply multiple enhancement techniques to improve unclear faces"""
    # Convert to numpy array if it's a PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to grayscale for some operations
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Bilateral filtering for noise reduction
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(filtered, -1, kernel)
    
    # Convert back to color
    enhanced_color = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    
    # Gamma correction
    gamma = 1.5
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    enhanced_color = cv2.LUT(enhanced_color, table)
    
    return enhanced_color

def load_database():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("SELECT name, embedding, image FROM Images")
    records = cursor.fetchall()
    conn.close()

    database = [(name, np.frombuffer(embedding, dtype=np.float32), img_blob) 
               for name, embedding, img_blob in records if embedding]
    return database

def extract_features(image_path):
    try:
        # Load and enhance the image
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        
        # Apply multiple enhancement techniques
        enhanced_img = enhance_image(img_array)
        
        # Convert to BGR for InsightFace
        img_array = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)
        img_array = cv2.resize(img_array, (640, 640))
        
        # Detect faces with relaxed thresholds
        faces = face_analyzer.get(img_array)
        if not faces:
            return None
            
        # Get the face with highest detection score
        faces = sorted(faces, key=lambda x: x.det_score, reverse=True)
        embedding = faces[0].normed_embedding
        return embedding.astype(np.float32).tobytes()
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def compute_similarity(features1, features2):
    return cosine_similarity([features1], [features2])[0][0]

def update_attendance(name):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO Attendance (name, in_time) 
        VALUES (?, ?) 
        ON CONFLICT(name) DO UPDATE SET in_time=?
    """, (name, current_time, current_time))
    conn.commit()
    conn.close()
# Inside FaceRecognitionApp class

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Face Recognition System")
        self.recent_faces = []
        self.setup_ui()
        self.display_next_face()
        self.start_image_processing()

    def setup_ui(self):
        # ---- Add Top Bar (NEW) ----
        self.top_frame = tk.Frame(self.root, bg="#004080", height=60)  # Dark blue top bar
        self.top_frame.pack(side="top", fill="x")

        self.title_label = tk.Label(
            self.top_frame,
            text="Frictionless Attendance Marking System",
            bg="#004080",
            fg="white",
            font=("Helvetica", 24, "bold")
        )
        self.title_label.pack(pady=10)

        # ---- Main Frame below Top Bar ----
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky='n')

        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky='n')

        # Face Gallery Frame (left side)
        self.face_gallery_frame = ttk.Frame(self.left_frame)
        self.face_gallery_frame.pack()

        self.face_labels = []
        for i in range(5):  # 5 rows
            row = []
            for j in range(3):  # 3 columns
                label = ttk.Label(self.face_gallery_frame)
                label.grid(row=i, column=j, padx=5, pady=5)
                row.append(label)
            self.face_labels.append(row)

        # Attendance Table (right side top)
        self.attendance_table = ttk.Treeview(self.right_frame, columns=("Name", "Time"), show="headings", height=8)
        self.attendance_table.heading("Name", text="Name")
        self.attendance_table.heading("Time", text="Time")
        self.attendance_table.pack(pady=5)

        # Unrecognized Face Section
        self.unrecognized_frame = tk.Frame(self.right_frame, bg="#FFCCCC", bd=5, relief="ridge")
        self.unrecognized_frame.pack(pady=(20, 0))

        self.unrecognized_label = tk.Label(self.unrecognized_frame, borderwidth=0)
        self.unrecognized_label.pack()

        self.unrecognized_time = ttk.Label(self.right_frame)
        self.unrecognized_time.pack()


    def update_face_gallery(self, img_blob, name, time_str):
        image = Image.open(io.BytesIO(img_blob)).resize((100, 100))
        photo = ImageTk.PhotoImage(image)

        self.recent_faces.insert(0, (photo, name, time_str))
        self.recent_faces = self.recent_faces[:15]  # ✅ Keep only 15

        for idx, (photo, name, time_str) in enumerate(self.recent_faces):
            i, j = divmod(idx, 3)  # 3 columns
            if i < len(self.face_labels) and j < len(self.face_labels[i]):
                self.face_labels[i][j].configure(image=photo, text=f"{name}\n{time_str}", compound=tk.BOTTOM)
                self.face_labels[i][j].image = photo

    def update_attendance_table(self, name, time_str):
        self.attendance_table.insert("", tk.END, values=(name, time_str))

    def update_unrecognized_face(self, img_path, time_str):
        image = Image.open(img_path).resize((150, 150))

        # Add a glowing red border decoration
        border_size = 10
        img_with_border = Image.new("RGB", (image.width + 2*border_size, image.height + 2*border_size), (255, 0, 0))
        img_with_border.paste(image, (border_size, border_size))

        photo = ImageTk.PhotoImage(img_with_border)

        self.unrecognized_label.configure(image=photo)
        self.unrecognized_label.image = photo
        self.unrecognized_time.configure(text=f"⚠️ Unrecognized @ {time_str}")

    def display_next_face(self):
        try:
            if not face_queue.empty():
                face_data = face_queue.get()
                if face_data["recognized"]:
                    self.update_face_gallery(face_data["img_blob"], face_data["name"], face_data["time"])
                    self.update_attendance_table(face_data["name"], face_data["time"])
                else:
                    self.update_unrecognized_face(face_data["img_path"], face_data["time"])
        except Exception as e:
            print(f"Display error: {e}")
        finally:
            self.root.after(500, self.display_next_face)

    def start_image_processing(self):
        def loop():
            db = load_database()
            while True:
                try:
                    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                    for file in files:
                        path = os.path.join(INPUT_FOLDER, file)
                        embedding_bytes = extract_features(path)
                        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        if embedding_bytes:
                            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                            match_found = False
                            for name, db_embedding, img_blob in db:
                                sim = compute_similarity(embedding, db_embedding)
                                if sim > SIMILARITY_THRESHOLD:
                                    update_attendance(name)
                                    face_queue.put({
                                        "recognized": True,
                                        "name": name,
                                        "img_blob": img_blob,
                                        "time": time_str
                                    })
                                    match_found = True
                                    break
                            if not match_found:
                                new_path = os.path.join(UNRECOGNIZED_FOLDER, f"{time_str.replace(':','-')}_{file}")
                                shutil.move(path, new_path)
                                face_queue.put({
                                    "recognized": False,
                                    "img_path": new_path,
                                    "time": time_str
                                })
                        else:
                            print(f"No face found in {file}, skipping...")
                        shutil.move(path, os.path.join(PROCESSED_FOLDER, file))
                    time.sleep(PROCESSING_INTERVAL)
                except Exception as e:
                    print(f"Processing error: {e}")
        threading.Thread(target=loop, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
