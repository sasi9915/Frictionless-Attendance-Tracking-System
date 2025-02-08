import sqlite3
import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, Label, ttk
from PIL import Image, ImageTk
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
import numpy as np
from io import BytesIO
from datetime import datetime

# Initialize MTCNN for face detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=False, device=device)

# Load pre-trained Facenet model
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# SQLite database connection
DB_PATH = "images_all.db"

def initialize_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Images (
            name TEXT PRIMARY KEY,
            image BLOB
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

def extract_features(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        face = mtcnn(img)
        if face is None:
            return None, img
        face = face.unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(face).squeeze().cpu().numpy()
        return features / np.linalg.norm(features), img
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None

def compute_similarity(features1, features2):
    return np.linalg.norm(features1 - features2)

def load_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, image FROM Images")
    records = cursor.fetchall()
    conn.close()
    database = []
    for name, img_data in records:
        img = Image.open(BytesIO(img_data)).convert("RGB")
        face = mtcnn(img)
        if face is not None:
            face = face.unsqueeze(0).to(device)
            with torch.no_grad():
                features = model(face).squeeze().cpu().numpy()
            features = features / np.linalg.norm(features)
            database.append((name, features))
    return database

def update_attendance(name):
    conn = sqlite3.connect(DB_PATH)
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
            cursor.execute("UPDATE Attendance SET in_time=?, out_time=? WHERE name=?", (out_time, current_time, name))
    conn.commit()
    conn.close()

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition & Attendance System")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")

        tk.Label(root, text="Upload an image to recognize and mark attendance.", font=("Arial", 14, "bold"), bg="#f0f0f0").pack(pady=10)
        self.img_canvas = tk.Label(root, bg="#ffffff", relief="solid", bd=2, width=200, height=200)
        self.img_canvas.pack(pady=5)
        
        self.btn_upload = tk.Button(root, text="Upload Image", command=self.upload_image, font=("Arial", 12), bg="#4CAF50", fg="white")
        self.btn_upload.pack(pady=5)

        self.result_label = tk.Label(root, text="", font=("Arial", 14, "bold"), fg="blue", bg="#f0f0f0")
        self.result_label.pack()

        self.attendance_table = ttk.Treeview(root, columns=("Name", "In Time", "Out Time"), show="headings")
        self.attendance_table.heading("Name", text="Name")
        self.attendance_table.heading("In Time", text="In Time")
        self.attendance_table.heading("Out Time", text="Out Time")
        self.attendance_table.pack(pady=10)
        self.load_attendance()
        
        self.database = load_database()

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.display_image(file_path)
            self.recognize_face(file_path)

    def display_image(self, image_path):
        img = Image.open(image_path).resize((200, 200))
        img = ImageTk.PhotoImage(img)
        self.img_canvas.img = img
        self.img_canvas.config(image=img)

    def recognize_face(self, image_path):
        input_features, img = extract_features(image_path)
        if input_features is None:
            self.result_label.config(text="No face detected!", fg="red")
            self.show_unrecognized_face(img)
            return
        best_match = None
        min_distance = float("inf")
        for name, db_features in self.database:
            distance = compute_similarity(input_features, db_features)
            if distance < min_distance:
                min_distance = distance
                best_match = name
        threshold = 0.6
        if min_distance < threshold:
            self.result_label.config(text=f"Recognized: {best_match}", fg="green")
            update_attendance(best_match)
            self.load_attendance()
        else:
            self.result_label.config(text="Unrecognized Face", fg="red")
            self.show_unrecognized_face(img)

    def show_unrecognized_face(self, img):
        unrecognized_window = Toplevel()
        unrecognized_window.title("Unrecognized Face")
        unrecognized_window.geometry("400x400")
        Label(unrecognized_window, text="Unrecognized Face", font=("Helvetica", 16, "bold"), fg="red").pack(pady=10)
        img = img.resize((200, 200))
        img = ImageTk.PhotoImage(img)
        label = Label(unrecognized_window, image=img)
        label.image = img
        label.pack()

    def load_attendance(self):
        self.attendance_table.delete(*self.attendance_table.get_children())
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Attendance")
        for row in cursor.fetchall():
            self.attendance_table.insert("", tk.END, values=row)
        conn.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
