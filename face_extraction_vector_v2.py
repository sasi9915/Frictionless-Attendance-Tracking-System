import cv2
import mediapipe as mp
import numpy as np
from numpy import dot
from numpy.linalg import norm
from tkinter import Tk
from tkinter.filedialog import askopenfilename


def extract_features(image_path):
    """
    Extract facial landmark features from an image using Mediapipe.
    
    Args:
        image_path (str): Path to the image.
    
    Returns:
        numpy.ndarray: Flattened feature vector of facial landmarks or None if no face is detected.
    """
    # Initialize Mediapipe Face Mesh module
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}.")
        return None

    # Convert the image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect face landmarks
    results = face_mesh.process(rgb_image)
    if not results.multi_face_landmarks:
        print(f"No faces detected in {image_path}.")
        return None

    # Extract the first face's landmarks
    face_landmarks = results.multi_face_landmarks[0]

    # Convert landmarks to a numpy array (468 landmarks, each with x, y, z)
    feature_vector = np.array(
        [[landmark.x, landmark.y, landmark.z] for landmark in face_landmarks.landmark]
    ).flatten()

    return feature_vector


def cosine_similarity(vec1, vec2):
    """
    Calculate the cosine similarity between two vectors.
    
    Args:
        vec1 (numpy.ndarray): First vector.
        vec2 (numpy.ndarray): Second vector.
    
    Returns:
        float: Cosine similarity value between -1 and 1.
    """
    if len(vec1) != len(vec2):
        raise ValueError("Both vectors must have the same length.")
    
    similarity = dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    return similarity


if __name__ == "__main__":
    # Allow the user to select two images
    print("Please select the first image file...")
    Tk().withdraw()  # Hide the main Tkinter window
    image_path1 = askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not image_path1:
        print("No file selected for the first image. Exiting.")
        exit()

    print("Please select the second image file...")
    image_path2 = askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not image_path2:
        print("No file selected for the second image. Exiting.")
        exit()

    # Extract features from both images
    feature_vector1 = extract_features(image_path1)
    feature_vector2 = extract_features(image_path2)

    # Check if feature extraction was successful for both images
    if feature_vector1 is None or feature_vector2 is None:
        print("Feature extraction failed for one or both images.")
        exit()

    # Calculate and display cosine similarity
    try:
        similarity = cosine_similarity(feature_vector1, feature_vector2)
        print(f"Cosine Similarity between the two images: {similarity:.4f}")
    except ValueError as e:
        print(e)
