import sqlite3
import requests
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import io
import re
from PIL import Image
import matplotlib.pyplot as plt

# Google Sheets API Setup
SHEET_NAME = "Collecting data form (Responses)"
COLUMN_INDEX = 4  # Column where Google Drive links are stored (1-based index)
SERVICE_ACCOUNT_FILE = "/home/ukil_217/graphite-cell-449009-r5-7bdb9b4f8d78.json"

# SQLite Database Setup
DB_FILE = "images.db"
TABLE_NAME = "images"

def create_database():
    """Create SQLite database and table."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT,
            image_data BLOB
        )
    """)
    conn.commit()
    conn.close()

def get_google_sheet_data():
    """Fetch Google Drive image links from Google Sheets."""
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_FILE, scope)
    client = gspread.authorize(creds)

    sheet = client.open(SHEET_NAME).sheet1
    data = sheet.col_values(COLUMN_INDEX)  # Get all values from the column
    return data[1:]  # Skip the header row

def extract_file_id(drive_link):
    """Extract file ID from Google Drive link using regex."""
    match = re.search(r"(?:/d/|id=)([a-zA-Z0-9_-]+)", drive_link)
    if match:
        return match.group(1)
    print(f"Invalid Google Drive link: {drive_link}")
    return None

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

def store_image_in_db(image_name, image_data):
    """Store image in SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(f"INSERT INTO {TABLE_NAME} (image_name, image_data) VALUES (?, ?)", (image_name, image_data))
    conn.commit()
    conn.close()

def retrieve_and_display_image(image_name):
    """Retrieve and display an image from the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute(f"SELECT image_data FROM {TABLE_NAME} WHERE image_name=?", (image_name,))
    row = cursor.fetchone()
    conn.close()

    if row:
        image_data = row[0]
        image = Image.open(io.BytesIO(image_data))
        
        # Display the image using matplotlib
        plt.imshow(image)
        plt.axis("off")  # Hide axes
        plt.show()
    else:
        print(f"Image '{image_name}' not found in database.")

def main():
    create_database()
    drive_links = get_google_sheet_data()

    for link in drive_links:
        img_data = download_image(link)
        if img_data:
            image_name = extract_file_id(link) + ".jpg"  # Use file ID as the image name
            store_image_in_db(image_name, img_data.getvalue())
            print(f"Stored: {image_name}")
            retrieve_and_display_image(image_name)

if __name__ == "__main__":
    main()
