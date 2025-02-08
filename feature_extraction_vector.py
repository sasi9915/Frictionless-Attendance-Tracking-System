import os
import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image
from tkinter import Tk, filedialog

# Function to browse files
def browse_images():
    file_paths = filedialog.askopenfilenames()
    print(file_paths)
    return file_paths


# Load the pre-trained Facenet model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Preprocessing pipeline for the images
preprocess = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Function to extract features
def extract_features(image_path):
    try:
        # Load the image
        img = Image.open(image_path).convert('RGB')
        
        # Preprocess the image
        img_tensor = preprocess(img).unsqueeze(0).to(device)  # Add batch dimension
        
        # Extract features
        with torch.no_grad():
            features = model(img_tensor).squeeze().cpu().numpy()  # Convert to NumPy array
        
        return features
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Main function
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
        
        # Save or display the feature vectors
        for img_name, features in feature_vectors.items():
            print(f"\nFeatures for {img_name}: {features}\n")
        print("Feature extraction complete!")
    else:
        print("No images selected.")
