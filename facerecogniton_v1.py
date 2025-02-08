import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import face_recognition
import os



class FaceMatchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Match Checker")
        self.root.geometry("600x400")
        
        # UI Elements
        self.label1 = tk.Label(root, text="Upload two images to check if they are the same person.")
        self.label1.pack(pady=10)
        
        self.img1_label = tk.Label(root, text="Image 1 Preview")
        self.img1_label.pack()
        self.img1_canvas = tk.Label(root)
        self.img1_canvas.pack()

        self.img2_label = tk.Label(root, text="Image 2 Preview")
        self.img2_label.pack()
        self.img2_canvas = tk.Label(root)
        self.img2_canvas.pack()
        
        self.btn_upload1 = tk.Button(root, text="Upload Image 1", command=self.upload_image1)
        self.btn_upload1.pack(pady=5)
        
        self.btn_upload2 = tk.Button(root, text="Upload Image 2", command=self.upload_image2)
        self.btn_upload2.pack(pady=5)
        
        self.btn_check = tk.Button(root, text="Check Match", command=self.check_match)
        self.btn_check.pack(pady=20)
        
        self.result_label = tk.Label(root, text="", font=("Arial", 14), fg="blue")
        self.result_label.pack()
        
        # Variables to hold image paths
        self.img1_path = None
        self.img2_path = None

    def upload_image1(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.img1_path = file_path
            self.display_image(file_path, self.img1_canvas)

    def upload_image2(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.img2_path = file_path
            self.display_image(file_path, self.img2_canvas)

    def display_image(self, image_path, canvas):
        img = Image.open(image_path)
        img = img.resize((200, 200))
        img = ImageTk.PhotoImage(img)
        canvas.img = img
        canvas.config(image=img)

    def check_match(self):
        if not self.img1_path or not self.img2_path:
            messagebox.showwarning("Error", "Please upload both images!")
            return
        
        # Perform face matching
        result = self.compare_faces(self.img1_path, self.img2_path)
        self.result_label.config(text=result)

    def compare_faces(self, img1_path, img2_path, threshold=0.6):
        # Load images
        img1 = face_recognition.load_image_file(img1_path)
        img2 = face_recognition.load_image_file(img2_path)
        
        # Extract encodings
        encodings1 = face_recognition.face_encodings(img1)
        encodings2 = face_recognition.face_encodings(img2)
        
        if not encodings1 or not encodings2:
            return "No face detected in one or both images."
        
        # Compare first detected faces
        face_distance = face_recognition.face_distance([encodings1[0]], encodings2[0])[0]
        print(f"Face Distance: {face_distance}")
        
        if face_distance < threshold:
            return "Result: Same Person"
        else:
            return "Result: Different People"

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceMatchApp(root)
    root.mainloop()
