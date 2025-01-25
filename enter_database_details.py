import sqlite3
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PIL import Image
from PIL.ExifTags import TAGS

# SQLite database setup
db_name = "user_database.db"
connection = sqlite3.connect(db_name)
cursor = connection.cursor()

# Create the table if it doesn't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS students (
    index_number INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    role TEXT NOT NULL,
    image TEXT NOT NULL
)
''')
connection.commit()

print("Database setup complete.\n")

# Function to extract metadata (if needed)
def extract_metadata(image_path):
    """
    Extract metadata from an image file. For simplicity, this function 
    uses placeholder data for 'index_number', 'name', and 'role'.
    Replace this with actual metadata extraction logic as required.
    """
    try:
        # Placeholder logic: Extract file name and create fake data
        file_name = os.path.basename(image_path)
        index_number = int(file_name.split('_')[0])  # Example: "123_name_role.jpg"
        name = file_name.split('_')[1].capitalize()
        role = file_name.split('_')[2].split('.')[0].capitalize()

        # Optional: Extract additional metadata using PIL (if needed)
        image = Image.open(image_path)
        metadata = {}
        for tag, value in image._getexif().items():
            tag_name = TAGS.get(tag, tag)
            metadata[tag_name] = value
        
        return index_number, name, role, image_path
    except Exception as e:
        print(f"Error extracting metadata from {image_path}: {e}")
        return None, None, None, None

# Function to insert data into the database
def insert_into_database(index_number, name, role, image):
    try:
        cursor.execute('''
        INSERT INTO students (index_number, name, role, image)
        VALUES (?, ?, ?, ?)
        ''', (index_number, name, role, image))
        connection.commit()
        print(f"Inserted data: Index: {index_number}, Name: {name}, Role: {role}, Image: {image}\n")
    except sqlite3.IntegrityError:
        print(f"Error: Duplicate index number {index_number}. Skipping entry.\n")
    except Exception as e:
        print(f"Unexpected error while inserting into database: {e}\n")

# File System Event Handler
class ImageHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return  # Skip directories
        if event.src_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            print(f"New image detected: {event.src_path}")
            # Extract metadata and insert into database
            index_number, name, role, image_path = extract_metadata(event.src_path)
            if index_number and name and role:
                insert_into_database(index_number, name, role, image_path)
            else:
                print(f"Skipping {event.src_path} due to incomplete metadata.\n")

# Monitor a folder for new images
def monitor_folder(folder_path):
    print(f"Monitoring folder: {folder_path}")
    event_handler = ImageHandler()
    observer = Observer()
    observer.schedule(event_handler, path=folder_path, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)  # Keep the script running
    except KeyboardInterrupt:
        print("\nStopping folder monitor...")
        observer.stop()
    observer.join()

# Main function
if __name__ == "__main__":
    folder_to_monitor = "image_uploads"  # Replace with your folder path
    os.makedirs(folder_to_monitor, exist_ok=True)  # Create folder if it doesn't exist
    monitor_folder(folder_to_monitor)

    # Close the database connection when the program ends
    connection.close()
