import os
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision import transforms
from PIL import Image
from tkinter import Tk, filedialog
import numpy as np

# Initialize the MTCNN for face detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, device=device)

# Load the pre-trained Facenet model
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Function to browse files
def browse_images():
    Tk().withdraw()  # Prevent the Tkinter GUI from appearing
    file_paths = filedialog.askopenfilenames()
    print(file_paths)
    return file_paths

# Preprocessing pipeline for the images
preprocess = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Function to extract features
def extract_features(image_path):
    try:
        # Load and align the image
        img = Image.open(image_path).convert('RGB')
        face = mtcnn(img)  # This gives a Torch Tensor
        if face is None:
            raise ValueError("No face detected in the image.")
        
        # Convert the Torch Tensor to a PIL Image
        face = transforms.ToPILImage()(face)  # Converts Tensor to PIL Image
        
        # Preprocess the face
        face = preprocess(face).unsqueeze(0).to(device)  # Add batch dimension
        
        # Extract features
        with torch.no_grad():
            features = model(face).squeeze().cpu().numpy()  # Convert to NumPy array
        
        # Normalize features
        features = features / np.linalg.norm(features)  # L2-normalization
        return features
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Function to compute Euclidean distance
def compute_distance(features1, features2):
    return np.linalg.norm(features1 - features2)

# Recognize faces by comparing feature vectors
def recognize_faces(feature_vectors, distance_threshold=0.6):
    recognized_faces = []
    img_names = list(feature_vectors.keys())
    for i in range(len(img_names)):
        for j in range(i + 1, len(img_names)):
            img1, img2 = img_names[i], img_names[j]
            features1, features2 = feature_vectors[img1], feature_vectors[img2]
            distance = compute_distance(features1, features2)
            if distance < distance_threshold:  # Lower distance implies more similarity
                recognized_faces.append((img1, img2, distance))
    return recognized_faces

if __name__ == "__main__":
    print("Select images to extract features.")
    image_paths = browse_images()

    if image_paths:
        feature_vectors = {}
        for image_path in image_paths:
            print(f"Processing: {image_path}")
            features = extract_features(image_path)
            if features is not None:
                feature_vectors[os.path.basename(image_path)] = features
                print(f"Feature vector for {os.path.basename(image_path)} generated.")
        
        # Compare feature vectors and recognize similar faces
        print("Comparing feature vectors to recognize faces...")
        distance_threshold = 0.6  # Adjust based on your dataset
        recognized_faces = recognize_faces(feature_vectors, distance_threshold)

        if recognized_faces:
            print("\nRecognized Similar Faces:")
            for img1, img2, distance in recognized_faces:
                print(f"{img1} and {img2} are similar with a distance of {distance:.4f}")
        else:
            print("\nNo similar faces recognized.")

        print("Feature extraction and face recognition complete!")
    else:
        print("No images selected.")
