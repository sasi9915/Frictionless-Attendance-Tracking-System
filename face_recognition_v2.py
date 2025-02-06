import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
import numpy as np

# Initialize the MTCNN for face detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=False, device=device)

# Load the pre-trained Facenet model
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)


# Function to extract facial features
def extract_features(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        face = mtcnn(img)  # Detect and align the face
        if face is None:
            raise ValueError("No face detected in the image.")
        face = face.unsqueeze(0).to(device)  # Add batch dimension and send to the correct device
        with torch.no_grad():
            features = model(face).squeeze().cpu().numpy()  # Extract features
        return features / np.linalg.norm(features)  # Normalize features
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


# Function to compute similarity score between two feature vectors
def compute_similarity(features1, features2):
    return np.linalg.norm(features1 - features2)


# Tkinter-based GUI application
class FaceMatchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Match Checker")
        self.root.geometry("700x500")

        # UI Elements
        self.label1 = tk.Label(root, text="Upload two images to check if they are the same person.", font=("Arial", 14))
        self.label1.pack(pady=10)

        # Image 1 preview
        self.img1_label = tk.Label(root, text="Image 1 Preview", font=("Arial", 12))
        self.img1_label.pack(pady=5)
        self.img1_canvas = tk.Label(root)
        self.img1_canvas.pack(pady=5)

        # Image 2 preview
        self.img2_label = tk.Label(root, text="Image 2 Preview", font=("Arial", 12))
        self.img2_label.pack(pady=5)
        self.img2_canvas = tk.Label(root)
        self.img2_canvas.pack(pady=5)

        # Buttons
        self.btn_upload1 = tk.Button(root, text="Upload Image 1", command=self.upload_image1, font=("Arial", 12))
        self.btn_upload1.pack(pady=5)

        self.btn_upload2 = tk.Button(root, text="Upload Image 2", command=self.upload_image2, font=("Arial", 12))
        self.btn_upload2.pack(pady=5)

        self.btn_check = tk.Button(root, text="Check Match", command=self.check_match, font=("Arial", 14), bg="green", fg="white")
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
        img = Image.open(image_path).resize((200, 200))
        img = ImageTk.PhotoImage(img)
        canvas.img = img
        canvas.config(image=img)

    def check_match(self):
        if not self.img1_path or not self.img2_path:
            messagebox.showwarning("Error", "Please upload both images!")
            return

        # Extract features and calculate similarity
        features1 = extract_features(self.img1_path)
        features2 = extract_features(self.img2_path)

        if features1 is None or features2 is None:
            self.result_label.config(text="Error: Could not detect a face in one or both images.")
            return

        similarity = compute_similarity(features1, features2)
        print(f"Similarity Score: {similarity}")

        # Threshold for face similarity
        threshold = 0.6
        if similarity < threshold:
            self.result_label.config(text=f"Result: Same Person (Score: {similarity:.4f})", fg="green")
        else:
            self.result_label.config(text=f"Result: Different People (Score: {similarity:.4f})", fg="red")


# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceMatchApp(root)
    root.mainloop()
