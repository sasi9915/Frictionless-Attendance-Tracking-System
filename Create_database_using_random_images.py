import os
import sqlite3
import torch
import cv2
import numpy as np
import pickle
from faker import Faker
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# === Configuration ===
IMAGE_FOLDER = "path_to_your_image_folder"  # <-- Replace with your folder path
DB_NAME = "unique_faces_with_images.db"
SIMILARITY_THRESHOLD = 0.6

# === Setup Faker and InsightFace ===
faker = Faker()
face_analyzer = FaceAnalysis(
    name="buffalo_l",
    providers=['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider'],
    allowed_modules=['detection', 'recognition']
)
face_analyzer.prepare(
    ctx_id=0 if torch.cuda.is_available() else -1,
    det_size=(640, 640),
    det_thresh=0.4
)

# === Setup SQLite ===
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        image_blob BLOB,
        embedding BLOB
    )
''')
conn.commit()

# === Helper Functions ===
def insert_face(name, image_blob, embedding):
    cursor.execute('''
        INSERT INTO faces (name, image_blob, embedding)
        VALUES (?, ?, ?)
    ''', (name, image_blob, pickle.dumps(embedding.astype(np.float32))))
    conn.commit()

def load_existing_embeddings():
    cursor.execute("SELECT embedding FROM faces")
    rows = cursor.fetchall()
    return [pickle.loads(row[0]) for row in rows]

def is_new_person(embedding, existing_embeddings):
    for existing in existing_embeddings:
        similarity = cosine_similarity(
            [embedding.astype(np.float32)],
            [existing.astype(np.float32)]
        )[0][0]
        if similarity >= SIMILARITY_THRESHOLD:
            return False
    return True

def get_largest_face(faces):
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

# === Main Processing ===
def process_images(folder):
    all_embeddings = []
    face_cache = []

    for filename in os.listdir(folder):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(folder, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue

        faces = face_analyzer.get(image)
        best_face = get_largest_face(faces)
        if best_face is None:
            print(f"No face found in: {filename}")
            continue

        embedding = best_face.embedding

        # Check if new person
        if is_new_person(embedding, all_embeddings):
            # Read image as binary
            with open(image_path, "rb") as img_file:
                image_bytes = img_file.read()

            name = faker.name()
            insert_face(name, image_bytes, embedding)
            all_embeddings.append(embedding)
            print(f"Inserted: {name} from {filename}")
        else:
            print(f"Duplicate face skipped: {filename}")

    print("✅ Unique faces saved to database.")

# === Run Script ===
process_images(IMAGE_FOLDER)
conn.close()
