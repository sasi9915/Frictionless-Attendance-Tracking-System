import sqlite3
import requests
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import io
import re
import torch
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Face Analysis Model
face_analyzer = FaceAnalysis(name="buffalo_l")
face_analyzer.prepare(ctx_id=0 if torch.cuda.is_available() else -1)  # Use GPU if available

# Google Sheets API Setup
SHEET_NAME = "Collecting data form (Responses)"
SERVICE_ACCOUNT_FILE = "/home/ukil_217/graphite-cell-449009-r5-7bdb9b4f8d78.json"

COLUMN_NAME_INDEX = 2  # Column index for Name (1-based index)
COLUMN_IMAGE_LINK_INDEX = 4  # Column index for Google Drive links (1-based index)

# SQLite Database Setup
DB_FILE = "images_all.db"
TABLE_NAME = "images"

# Similarity threshold (0.0 to 1.0)
SIMILARITY_THRESHOLD = 0.95

def create_database():
   """Create SQLite database with Name, Image, and Face Embeddings."""
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()
cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            image BLOB,
            embedding BLOB
        )
    """)
conn.commit()
conn.close()

def check_existing_name(name):
    """Check if a name already exists in the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(f"SELECT 1 FROM {TABLE_NAME} WHERE name = ?", (name,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists

def get_existing_embeddings():
    """Retrieve all existing embeddings from the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(f"SELECT embedding FROM {TABLE_NAME}")
    embeddings = [np.frombuffer(row[0], dtype=np.float32) for row in cursor.fetchall()]
    conn.close()
    return embeddings

def is_image_similar(new_embedding, existing_embeddings):
    """Check if new image is similar to existing ones using cosine similarity."""
    if not existing_embeddings:
        return False
    
    new_embedding = np.frombuffer(new_embedding, dtype=np.float32).reshape(1, -1)
    existing_matrix = np.array(existing_embeddings)
    
    similarities = cosine_similarity(new_embedding, existing_matrix)
    max_similarity = np.max(similarities)
    
    print(f"Max similarity found: {max_similarity:.4f}")
    return max_similarity > SIMILARITY_THRESHOLD

def get_google_sheet_data():
    """Fetch names and Google Drive image links from Google Sheets."""
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_FILE, scope)
    client = gspread.authorize(creds)

    sheet = client.open(SHEET_NAME).sheet1
    names = sheet.col_values(COLUMN_NAME_INDEX)[1:]  # Skip header
    drive_links = sheet.col_values(COLUMN_IMAGE_LINK_INDEX)[1:]  # Skip header

    return list(zip(names, drive_links))

def extract_file_id(drive_link):
    """Extract file ID from Google Drive link using regex."""
    match = re.search(r"(?:/d/|id=)([a-zA-Z0-9_-]+)", drive_link)
    return match.group(1) if match else None

def download_image(drive_link):
    """Download image from Google Drive and return both image data and bytes."""
    file_id = extract_file_id(drive_link)
    if not file_id:
        print(f"⚠️ Invalid Google Drive link: {drive_link}")
        return None, None

    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url, timeout=10)
    
    if response.status_code == 200:
        image_bytes = response.content
        image_data = io.BytesIO(image_bytes)
        return image_data, image_bytes
    
    print(f"⚠️ Failed to download: {drive_link} (Status: {response.status_code})")
    return None, None

def extract_face_embedding(image_data):
    """Extract face embedding using InsightFace."""
    try:
        image = Image.open(image_data).convert("RGB")
        img_array = np.array(image)

        faces = face_analyzer.get(img_array)
        if not faces:
            print("❌ No face detected in image.")
            return None
        
        embedding = faces[0].normed_embedding
        return embedding.astype(np.float32).tobytes()
    except Exception as e:
        print(f"⚠️ Error processing image: {e}")
        return None

def store_face_data(name, image_bytes, embedding):
    """Store data only if it doesn't already exist."""
    try:
        # First check by name
        if check_existing_name(name):
            print(f"⏩ Skipping duplicate name: {name}")
            return False
        
        # Then check by image similarity
        existing_embeddings = get_existing_embeddings()
        if is_image_similar(embedding, existing_embeddings):
            print(f"⏩ Skipping similar image for: {name}")
            return False
        
        # If checks pass, insert new record
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(f"""
            INSERT INTO {TABLE_NAME} (name, image, embedding) 
            VALUES (?, ?, ?)
        """, (name, image_bytes, embedding))
        conn.commit()
        conn.close()
        print(f"✅ Stored: {name}")
        return True
    except Exception as e:
        print(f"⚠️ Error storing data for {name}: {e}")
        return False

def main():
    create_database()
    records = get_google_sheet_data()

    for name, link in records:
        img_data, img_bytes = download_image(link)
        if img_data and img_bytes:
            embedding = extract_face_embedding(img_data)
            if embedding:
                store_face_data(name, img_bytes, embedding)

if __name__ == "__main__":
    main()