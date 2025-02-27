import sqlite3
import requests
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import io
import re
import torch
import pickle
from PIL import Image
import numpy as np
from insightface.app import FaceAnalysis

# Initialize Face Analysis Model
face_analyzer = FaceAnalysis(name="buffalo_l")
face_analyzer.prepare(ctx_id=0 if torch.cuda.is_available() else -1)  # Use GPU if available

# Google Sheets API Setup
SHEET_NAME = "Collecting data form (Responses)"
SERVICE_ACCOUNT_FILE = "/home/ukil_217/graphite-cell-449009-r5-7bdb9b4f8d78.json"

COLUMN_NAME_INDEX = 2  # Column index for Name (1-based index)
COLUMN_E_NUMBER_INDEX = 3  # Column index for E number (1-based index)
COLUMN_IMAGE_LINK_INDEX = 4  # Column index for Google Drive links (1-based index)

# SQLite Database Setup
DB_FILE = "example.db"
TABLE_NAME = "faces"

def create_database():
    """Create SQLite database with only Name, E Number, and Face Embeddings."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            e_number TEXT,
            embedding BLOB
        )
    """)
    conn.commit()
    conn.close()

def get_google_sheet_data():
    """Fetch names, E numbers, and Google Drive image links from Google Sheets."""
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_FILE, scope)
    client = gspread.authorize(creds)

    sheet = client.open(SHEET_NAME).sheet1
    names = sheet.col_values(COLUMN_NAME_INDEX)[1:]  # Skip header
    e_numbers = sheet.col_values(COLUMN_E_NUMBER_INDEX)[1:]  # Skip header
    drive_links = sheet.col_values(COLUMN_IMAGE_LINK_INDEX)[1:]  # Skip header

    return list(zip(names, e_numbers, drive_links))  # Combine data

def extract_file_id(drive_link):
    """Extract file ID from Google Drive link using regex."""
    match = re.search(r"(?:/d/|id=)([a-zA-Z0-9_-]+)", drive_link)
    return match.group(1) if match else None

def download_image(drive_link):
    """Download image from Google Drive."""
    file_id = extract_file_id(drive_link)
    if not file_id:
        return None

    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    
    if response.status_code == 200:
        return io.BytesIO(response.content)
    else:
        print(f"Failed to download: {drive_link} (Status: {response.status_code})")
        return None

def extract_face_embedding(image_data):
    """Extract face embedding using InsightFace."""
    try:
        image = Image.open(image_data).convert("RGB")
        img_array = np.array(image)
        faces = face_analyzer.get(img_array)

        if faces:
            embedding = faces[0].embedding  # Take the first detected face
            return pickle.dumps(embedding)  # Serialize for database storage
        else:
            print("No face detected in image.")
            return None
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def store_face_embedding(name, e_number, embedding):
    """Store name, E number, and face embedding in SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(f"""
        INSERT INTO {TABLE_NAME} (name, e_number, embedding) 
        VALUES (?, ?, ?)
    """, (name, e_number, embedding))
    conn.commit()
    conn.close()

def main():
    create_database()
    records = get_google_sheet_data()

    for name, e_number, link in records:
        img_data = download_image(link)
        if img_data:
            embedding = extract_face_embedding(img_data)
            if embedding:
                store_face_embedding(name, e_number, embedding)
                print(f"Stored: {name} ({e_number})")

if __name__ == "__main__":
    main()
