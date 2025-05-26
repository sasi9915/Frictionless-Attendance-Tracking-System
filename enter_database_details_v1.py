import sqlite3
import requests
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import io
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import requests
import imghdr

# Initialize FaceNet model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=False, device=device)
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# SQLite database setup
db_name = "elec_database.db"
connection = sqlite3.connect(db_name)
cursor = connection.cursor()

# Create table if it doesn't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS students (
    index_number TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    role TEXT NOT NULL,
    image_url TEXT,  -- Ensure this column exists
    feature_vector BLOB
)
''')
connection.commit()

# Authenticate Google Sheets API
def authenticate_google_sheet(credentials_file, sheet_name):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_file, scope)
    client = gspread.authorize(credentials)
    return client.open(sheet_name).sheet1

# Convert Google Drive link to direct download link
def get_downloadable_url(drive_url):
    print(f"Original URL: {drive_url}")  # Debugging
    if "id=" in drive_url:
        file_id = drive_url.split("id=")[-1]
    elif "/file/d/" in drive_url:
        file_id = drive_url.split("/file/d/")[-1].split("/")[0]
    else:
        print(f"Invalid Drive URL: {drive_url}")
        return None
    
    new_url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Converted URL: {new_url}")  # Debugging
    return new_url

def extract_features_from_url(image_url):
    try:
        print(f"Downloading image from: {image_url}")  # Debugging
        response = requests.get(image_url, stream=True)
        
        if response.status_code != 200:
            print(f"Failed to download image: {response.status_code}")
            return None

        img_data = response.content
        print(f"Downloaded image size: {len(img_data)} bytes")  # Debugging

        # Check if the content is actually an image
        img_type = imghdr.what(None, img_data)
        if img_type not in ['jpeg', 'png', 'jpg']:
            print(f"Invalid image format detected: {img_type}")
            return None
        
        img = Image.open(io.BytesIO(img_data))
        img = img.convert("RGB")  # Ensure compatibility
        img.show()  # Debugging

        face = mtcnn(img)
        if face is None:
            raise ValueError("No face detected in the image.")

        face = face.unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(face).squeeze().cpu().numpy()
        
        return features / np.linalg.norm(features)
    
    except Exception as e:
        print(f"Error extracting features: {e}")
    
    return None

# Insert or update data in database
def insert_into_database(index_number, name, role, image_url):
    try:
        feature_vector = extract_features_from_url(image_url)
        feature_vector_blob = None if feature_vector is None else sqlite3.Binary(feature_vector.tobytes())
        
        cursor.execute('''
        INSERT INTO students (index_number, name, role, image_url, feature_vector)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(index_number) DO UPDATE SET
            name=excluded.name,
            role=excluded.role,
            image_url=excluded.image_url,
            feature_vector=excluded.feature_vector
        ''', (index_number, name, role, image_url, feature_vector_blob))
        connection.commit()
        print(f"Inserted/Updated: {index_number} - {name}")
    except Exception as e:
        print(f"Database error: {e}")

# Process data from Google Sheet
def process_google_sheet(sheet):
    records = sheet.get_all_records()
    for record in records:
        try:
            name = record.get('Your Name :')
            index_number = record.get('E number :')
            drive_url = record.get('Upload a Clear photo of you : (rename the photo indexno_name)\nex.217_Irushi.jpg')
            role = "Student"
            
            if not all([name, index_number, drive_url]):
                print(f"Skipping incomplete record: {record}")
                continue
            
            image_url = get_downloadable_url(drive_url)
            if not image_url:
                print(f"Invalid image URL for {name}")
                continue
            
            insert_into_database(index_number, name, role, image_url)
        except Exception as e:
            print(f"Error processing record: {e}")

if __name__ == "__main__":
    credentials_file = "/home/ukil_217/graphite-cell-449009-r5-7bdb9b4f8d78.json"
    sheet_name = "Collecting data form (Responses)"
    
    try:
        sheet = authenticate_google_sheet(credentials_file, sheet_name)
        process_google_sheet(sheet)
    except Exception as e:
        print(f"Google Sheets API error: {e}")
    
    connection.close()
