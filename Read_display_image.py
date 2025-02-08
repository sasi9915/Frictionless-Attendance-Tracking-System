import sqlite3
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

# Google Sheets API setup
SERVICE_ACCOUNT_FILE = "/home/ukil_217/graphite-cell-449009-r5-7bdb9b4f8d78.json"
SCOPES = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/spreadsheets", 
          "https://www.googleapis.com/auth/drive"]

creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
client = gspread.authorize(creds)

# Open the Google Sheet
SHEET_NAME = "Collecting data form (Responses)"
sheet = client.open(SHEET_NAME).sheet1

# Get all values
data = sheet.get_all_values()
df = pd.DataFrame(data[1:], columns=data[0])  # First row as column names

# Print column names to verify
print(df.columns)

# Find the correct column names
image_column = 'Upload a Clear photo of you : (rename the photo indexno_name)\nex.217_Irushi.jpg'
name_column = 'Your Name :'  # Adjust this based on your actual sheet column name

# Extract image links and names
if image_column in df.columns and name_column in df.columns:
    image_links = df[image_column].tolist()
    names = df[name_column].tolist()
else:
    print("Error: Column(s) not found")
    exit()

# Function to convert Google Drive link to direct download link
def convert_drive_link(url):
    if "drive.google.com/open?id=" in url:
        file_id = url.split("id=")[-1]
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    elif "drive.google.com/file/d/" in url:
        file_id = url.split("/d/")[1].split("/")[0]
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url

# SQLite database setup
conn = sqlite3.connect("images_data.db")
cursor = conn.cursor()

# Create table if not exists
cursor.execute('''
    CREATE TABLE IF NOT EXISTS Images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        image BLOB
    )
''')
conn.commit()

# Function to store an image in the database (avoiding duplicates)
def store_image(name, url):
    # Check if the image is already stored
    cursor.execute("SELECT COUNT(*) FROM Images WHERE name=?", (name,))
    count = cursor.fetchone()[0]

    if count > 0:
        print(f"Skipping '{name}' - already exists in the database.")
        return  # Skip inserting duplicate images

    url = convert_drive_link(url)
    headers = {"User-Agent": "Mozilla/5.0"}  # Bypass restrictions for some images
    response = requests.get(url, headers=headers, stream=True)

    if response.status_code == 200:
        try:
            img_data = response.content  # Convert image to binary
            cursor.execute("INSERT INTO Images (name, image) VALUES (?, ?)", (name, img_data))
            conn.commit()
            print(f"Image '{name}' stored successfully.")
        except Exception as e:
            print("Error storing image:", e)
    else:
        print(f"Failed to download image for {name}: {url}")


# Store all images in the database
for name, link in zip(names, image_links):
    if link.strip():  # Ensure the link is not empty
        store_image(name, link)

# Function to retrieve and display an image from the database
def retrieve_image(image_id):
    cursor.execute("SELECT name, image FROM Images WHERE id=?", (image_id,))
    record = cursor.fetchone()
    
    if record:
        img_name, img_data = record
        img = Image.open(BytesIO(img_data))
        plt.imshow(img)
        plt.axis("off")
        plt.title(img_name)  # Display name as title
        plt.show()
    else:
        print("Image not found.")

# Retrieve and display an image (change ID to test different images)
retrieve_image(1)

# Close the database connection
conn.close()
