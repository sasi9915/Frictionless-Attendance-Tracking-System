import sqlite3
from tkinter import Tk, Toplevel, Label, Button, Entry, messagebox
from PIL import Image, ImageTk
import datetime

# Database connection
db_name = "elec_database.db"
connection = sqlite3.connect(db_name)
cursor = connection.cursor()

def check_name_and_status():
    """Check if the entered name is in the database and manage attendance."""
    name = name_entry.get().strip()

    if not name:
        messagebox.showerror("Error", "Please enter a name.")
        return

    # Check if the name exists in the students table
    cursor.execute("SELECT * FROM students WHERE name = ?", (name,))
    student = cursor.fetchone()

    if student:
        index_number = student[1]  # Assuming the second column is the index number
        handle_attendance(name, index_number)
    else:
        # Display unrecognized face window
        show_unrecognized_face_window()

def handle_attendance(name, index_number):
    """Handle attendance records for the given name and index number."""
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Check if the person already has an attendance record
    cursor.execute("SELECT * FROM attendance WHERE name = ? AND index_number = ?", (name, index_number))
    attendance_record = cursor.fetchone()

    if attendance_record:
        # Update the existing record with new arrival and exit times
        cursor.execute(
            "UPDATE attendance SET arrival_time = ?, exit_time = ? WHERE name = ? AND index_number = ?",
            (current_time, current_time, name, index_number)
        )
        messagebox.showinfo("Attendance Updated", f"Welcome back, {name}! Your attendance has been updated.")
    else:
        # Insert a new record for the first-time attendee
        cursor.execute(
            "INSERT INTO attendance (name, index_number, arrival_time, exit_time) VALUES (?, ?, ?, ?)",
            (name, index_number, current_time, current_time)
        )
        messagebox.showinfo("Attendance Recorded", f"Hello, {name}! Your attendance has been recorded.")

    connection.commit()

def show_unrecognized_face_window():
    """Display unrecognized face image."""
    unrecognized_window = Toplevel()
    unrecognized_window.title("Unrecognized Face")
    unrecognized_window.geometry("400x400")
    unrecognized_window.configure(bg="#f8d7da")

    Label(unrecognized_window, text="Unrecognized Face", font=("Helvetica", 16, "bold"), bg="#f8d7da", fg="#721c24").pack(pady=10)

    # Placeholder image for unrecognized face
    face_image = Image.open("unrecognized_face.png").resize((300, 300))  # Replace with actual image path
    face_photo = ImageTk.PhotoImage(face_image)
    Label(unrecognized_window, image=face_photo, bg="#f8d7da").pack(pady=10)

    unrecognized_window.mainloop()

def show_unpaid_face_window():
    """Display unpaid face image."""
    unpaid_window = Toplevel()
    unpaid_window.title("Unpaid Face")
    unpaid_window.geometry("400x400")
    unpaid_window.configure(bg="#fff3cd")

    Label(unpaid_window, text="Unpaid Face", font=("Helvetica", 16, "bold"), bg="#fff3cd", fg="#856404").pack(pady=10)

    # Placeholder image for unpaid face
    face_image = Image.open("unpaid_face.png").resize((300, 300))  # Replace with actual image path
    face_photo = ImageTk.PhotoImage(face_image)
    Label(unpaid_window, image=face_photo, bg="#fff3cd").pack(pady=10)

    unpaid_window.mainloop()

def welcome_screen():
    """Main Welcome Window."""
    global name_entry

    root = Tk()
    root.title("Attendance System")
    root.geometry("900x700")
    root.configure(bg="#eef2f5")

    # Add a header banner
    header_label = Label(root, text="Frictionless Attendance System", font=("Helvetica", 20, "bold"), bg="#007acc", fg="white", pady=10)
    header_label.pack(fill="x")

    Label(root, text="Enter Your Name:", font=("Arial", 14), bg="#eef2f5").pack(pady=10)
    name_entry = Entry(root, font=("Arial", 14))
    name_entry.pack(pady=5)

    Button(root, text="Submit", command=check_name_and_status, width=20, font=("Arial", 12)).pack(pady=10)
    Button(root, text="Close", command=root.destroy, width=20, font=("Arial", 12)).pack(pady=10)

    # Footer for branding
    footer_label = Label(root, text="Developed by Group 33 - University of Peradeniya", font=("Helvetica", 10, "italic"), bg="#eef2f5", fg="#333")
    footer_label.pack(side="bottom", pady=10)

    root.mainloop()

if __name__ == "__main__":
    welcome_screen()

# Close the database connection when the program ends
connection.close()
