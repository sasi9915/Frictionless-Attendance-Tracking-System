import sqlite3
import tkinter as tk
from tkinter import messagebox, Toplevel
from tkinter import Label
from PIL import Image, ImageTk
import torch
from insightface.app import FaceAnalysis
import numpy as np
import cv2
import os
import time
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import threading

# Initialize face analyzer
face_analyzer = FaceAnalysis(name="buffalo_l")
face_analyzer.prepare(ctx_id=0 if torch.cuda.is_available() else -1)

DB_PATH = "images_all.db"
INPUT_FOLDER = "/home/ukil_217/detected_faces"
PROCESSED_FOLDER = "processed_images"
UNRECOGNIZED_FOLDER = "unrecognized_faces"
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(UNRECOGNIZED_FOLDER, exist_ok=True)

# Initialize database
def initialize_database():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Images (
            name TEXT PRIMARY KEY,
            image BLOB,
            embedding BLOB
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Attendance (
            name TEXT PRIMARY KEY,
            in_time TEXT,
            out_time TEXT
        )
    """)
    conn.commit()
    conn.close()

initialize_database()

# Extract face embedding
def extract_features(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        img_array = cv2.resize(img_array, (640, 640))
        faces = face_analyzer.get(img_array)

        if not faces:
            print("❌ No face detected.")
            return None, img

        embedding = faces[0].normed_embedding
        if embedding is None:
            print("⚠️ Failed to extract embedding.")
            return None, img

        return embedding.astype(np.float32).tobytes(), img

    except Exception as e:
        print(f"⚠️ Error extracting features: {e}")
        return None, None

# Load database
def load_database():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("SELECT name, embedding FROM Images")
    records = cursor.fetchall()
    conn.close()

    database = []
    for name, embedding_data in records:
        if embedding_data:
            features = np.frombuffer(embedding_data, dtype=np.float32)
            database.append((name, features))
    return database

# Compute similarity
def compute_similarity(features1, features2):
    return cosine_similarity([features1], [features2])[0][0]

# Update attendance
def update_attendance(name):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("SELECT in_time, out_time FROM Attendance WHERE name=?", (name,))
    record = cursor.fetchone()

    if record is None:
        cursor.execute("INSERT INTO Attendance (name, in_time, out_time) VALUES (?, ?, ?)", (name, current_time, None))
    else:
        in_time, out_time = record
        if out_time is None:
            cursor.execute("UPDATE Attendance SET out_time=? WHERE name=?", (current_time, name))
        else:
            cursor.execute("UPDATE Attendance SET in_time=?, out_time=NULL WHERE name=?", (current_time, name))

    conn.commit()
    conn.close()

# Show unrecognized face (thread-safe)
def show_unrecognized_face(image_path):
    if os.path.exists(image_path):  # Check if file exists before opening
        unrecognized_window = Toplevel(root)
        unrecognized_window.title("Unrecognized Face")
        unrecognized_window.geometry("300x300")
        
        img = Image.open(image_path)
        img = img.resize((250, 250))
        img_tk = ImageTk.PhotoImage(img)
        
        label = Label(unrecognized_window, image=img_tk)
        label.image = img_tk
        label.pack()
        
        btn_close = tk.Button(unrecognized_window, text="Close", command=unrecognized_window.destroy)
        btn_close.pack(pady=10)

# Monitor folder for images
def process_images():
    database = load_database()
    while True:
        for filename in os.listdir(INPUT_FOLDER):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(INPUT_FOLDER, filename)
                print(f"📷 Processing: {filename}")
                
                if not os.path.exists(image_path):  # Double-check file existence
                    print(f"⚠️ File not found: {image_path}")
                    continue

                embedding, _ = extract_features(image_path)

                if embedding:
                    current_features = np.frombuffer(embedding, dtype=np.float32)
                    best_match = None
                    best_score = 0.0

                    for name, stored_features in database:
                        similarity = compute_similarity(current_features, stored_features)
                        if similarity > best_score and similarity > 0.6:
                            best_score = similarity
                            best_match = name

                    if best_match:
                        print(f"✅ Recognized as: {best_match} (Score: {best_score:.2f})")
                        update_attendance(best_match)
                    else:
                        print("❌ Unrecognized face detected!")
                        unrecognized_path = os.path.join(UNRECOGNIZED_FOLDER, filename)
                        os.rename(image_path, unrecognized_path)
                        root.after(100, lambda p=unrecognized_path: show_unrecognized_face(p))  # GUI-safe update

                processed_path = os.path.join(PROCESSED_FOLDER, filename)
                if os.path.exists(image_path):  # Ensure file still exists before renaming
                    os.rename(image_path, processed_path)

        time.sleep(5)  # Check every 5 seconds

# GUI Class
class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("800x600")
        self.root.configure(bg="#2c3e50")

        header = Label(root, text="Face Recognition System", font=("Arial", 20, "bold"), bg="#063970", fg="white", pady=10)
        header.pack(fill="x")

        self.result_label = tk.Label(root, text="Monitoring folder: input_images", font=("Arial", 16), fg="green", bg="#2c3e50")
        self.result_label.pack(pady=10)

        self.btn_attendance = tk.Button(root, text="View Attendance", command=self.view_attendance, font=("Arial", 14),
                                        bg="#e74c3c", fg="white", padx=10, pady=5, relief="raised", cursor="hand2")
        self.btn_attendance.pack(pady=5)

        self.start_monitoring()

    def start_monitoring(self):
        threading.Thread(target=process_images, daemon=True).start()

    def view_attendance(self):
        attendance_window = Toplevel(self.root)
        attendance_window.title("Attendance Records")
        attendance_window.geometry("600x400")
        attendance_window.configure(bg="white")

        Label(attendance_window, text="Attendance Records", font=("Helvetica", 16, "bold"), fg="black", bg="white").pack(pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
