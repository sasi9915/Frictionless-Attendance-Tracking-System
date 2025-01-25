import sqlite3
from datetime import datetime

# Database setup
def setup_database():
    connection = sqlite3.connect("attendance.db")
    cursor = connection.cursor()

    # Create table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('Paid', 'Non-Paid'))
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            time TEXT NOT NULL,
            FOREIGN KEY(name) REFERENCES users(name)
        )
    ''')
    
    connection.commit()
    connection.close()

# Add user to the database
def add_user(name, status):
    connection = sqlite3.connect("attendance.db")
    cursor = connection.cursor()

    try:
        cursor.execute("INSERT INTO users (name, status) VALUES (?, ?)", (name, status))
        connection.commit()
        print(f"User '{name}' added successfully.")
    except sqlite3.IntegrityError:
        print(f"User '{name}' already exists.")
    connection.close()

# Mark attendance
def mark_attendance(name):
    connection = sqlite3.connect("attendance.db")
    cursor = connection.cursor()

    cursor.execute("SELECT status FROM users WHERE name = ?", (name,))
    user = cursor.fetchone()

    if user:
        # Record attendance with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO attendance (name, time) VALUES (?, ?)", (name, timestamp))
        connection.commit()
        print(f"Attendance marked for '{name}' at {timestamp}. Status: {user[0]}")
    else:
        print(f"User '{name}' not found. Please add the user first.")
    connection.close()

# Main program
def main():
    setup_database()

    while True:
        print("\n1. Add User\n2. Mark Attendance\n3. Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            name = input("Enter user name: ").strip()
            status = input("Enter status (Paid/Non-Paid): ").strip().capitalize()
            if status in ["Paid", "Non-Paid"]:
                add_user(name, status)
            else:
                print("Invalid status. Please enter 'Paid' or 'Non-Paid'.")
        elif choice == "2":
            name = input("Enter user name to mark attendance: ").strip()
            mark_attendance(name)
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

