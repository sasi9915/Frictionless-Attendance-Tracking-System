import sqlite3
from datetime import datetime

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('attendance1.db')
cursor = conn.cursor()

# Create 'students' table (email column removed as per updated schema)
cursor.execute('''
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        image_path TEXT NOT NULL,
        timestamp TEXT
    )
''')

print("Database and 'students' table created successfully.")

# Sample data
students = [
    ("John Doe", "images/john_doe.jpg"),
    ("Jane Smith", "images/jane_smith.jpg")
]

# Insert sample data into the 'students' table
for student in students:
    cursor.execute('''
        INSERT INTO students (name, image_path, timestamp)
        VALUES (?, ?, ?)
    ''', (student[0], student[1], datetime.now().isoformat()))

# Commit changes and close the database connection
conn.commit()
conn.close()

print("Sample data added successfully.")
