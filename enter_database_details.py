import sqlite3
import requests
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import io

# SQLite database setup
db_name = "elec_database.db"
connection = sqlite3.connect(db_name)
cursor = connection.cursor()

# Create the table if it doesn't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS students (
    index_number TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    role TEXT NOT NULL,
    image BLOB NOT NULL
)
''')
connection.commit()
print("Database setup complete.\n")

# Google Sheets API setup
def authenticate_google_sheet(credentials_file, sheet_name):
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_file, scope)
    client = gspread.authorize(credentials)
    sheet = client.open(sheet_name).sheet1  # Access the first sheet
    return sheet

# Get a direct-downloadable URL from a Google Drive sharing link
def get_downloadable_url(drive_url):
    if "id=" in drive_url:
        file_id = drive_url.split("id=")[-1]
        return f"https://drive.google.com/uc?id={file_id}&export=download"
    else:
        print(f"Invalid Google Drive URL: {drive_url}")
        return None

# Download the image as binary data
def download_image_as_blob(image_url):
    try:
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            return response.content  # Return the binary data directly
        else:
            print(f"Failed to download image from {image_url}. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

# Insert the binary image data into the SQLite database
def insert_into_database(index_number, name, role, image_blob):
    try:
        cursor.execute('''
        INSERT INTO students (index_number, name, role, image)
        VALUES (?, ?, ?, ?)
        ''', (index_number, name, role, sqlite3.Binary(image_blob)))
        connection.commit()
        print(f"Inserted data: Index: {index_number}, Name: {name}, Role: {role}\n")
    except sqlite3.IntegrityError:
        print(f"Error: Duplicate index number {index_number}. Skipping entry.\n")
    except Exception as e:
        print(f"Unexpected error while inserting into database: {e}\n")

# Process Google Sheet data and upload images as BLOBs
'''def process_google_sheet(sheet):
    records = sheet.get_all_records()
    for record in records:
        try:
            name = record['Your Name :']
            index_number = record['E number :']
            role = "Student"
            drive_url = record['Upload a Clear photo of you : (rename the photo indexno_name)ex.217_Irushi.jpg']

            image_url = get_downloadable_url(drive_url)
            if not image_url:
                print(f"Skipping {name} (Index: {index_number}) due to invalid image URL.\n")
                continue

            # Download image as binary data
            image_blob = download_image_as_blob(image_url)

            if image_blob:
                insert_into_database(index_number, name, role, image_blob)
            else:
                print(f"Failed to process image for {name} (Index: {index_number}).\n")

        except KeyError as e:
            print(f"Missing expected field in Google Sheet data: {e}")
        except Exception as e:
            print(f"Error processing record: {e}")''' 

def process_google_sheet(sheet):
    records = sheet.get_all_records()
    for record in records:
        try:
            name = record['Your Name :']
            index_number = record['E number :']
            role = "Student"

            # Debug: Print available columns
            print(f"Available columns: {list(record.keys())}")

            # Ensure the correct column name
            drive_url = record['Upload a Clear photo of you : (rename the photo indexno_name)\nex.217_Irushi.jpg']

            image_url = get_downloadable_url(drive_url)
            if not image_url:
                print(f"Skipping {name} (Index: {index_number}) due to invalid image URL.\n")
                continue

            # Download image as binary data
            image_blob = download_image_as_blob(image_url)

            if image_blob:
                insert_into_database(index_number, name, role, image_blob)
            else:
                print(f"Failed to process image for {name} (Index: {index_number}).\n")

        except KeyError as e:
            print(f"Missing expected field in Google Sheet data: {e}")
        except Exception as e:
            print(f"Error processing record: {e}")


# Main function
if __name__ == "__main__":
    credentials_file = "/home/ukil_217/graphite-cell-449009-r5-7bdb9b4f8d78.json"
    sheet_name = "Collecting data form (Responses)"

    try:
        sheet = authenticate_google_sheet(credentials_file, sheet_name)
        process_google_sheet(sheet)
    except Exception as e:
        print(f"Error setting up Google Sheets API: {e}")

    # Close the database connection when the program ends
    connection.close()
