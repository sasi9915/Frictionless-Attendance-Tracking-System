import cv2
import mediapipe as mp
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename


def extract_features(image_path):
    # Initialize Mediapipe Face Mesh module
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return None

    # Convert the image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect face landmarks
    results = face_mesh.process(rgb_image)
    if not results.multi_face_landmarks:
        print("No faces detected.")
        return None

    # Extract the first face's landmarks
    face_landmarks = results.multi_face_landmarks[0]

    # Convert landmarks to a numpy array (468 landmarks, each with x, y, z)
    feature_vector = np.array(
        [[landmark.x, landmark.y, landmark.z] for landmark in face_landmarks.landmark]
    ).flatten()

    return feature_vector


if __name__ == "__main__":
    # Create a file dialog to select the image
    print("Please select an image file...")
    Tk().withdraw()  # Hide the main Tkinter window
    image_path = askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

    if not image_path:
        print("No file selected. Exiting.")
    else:
        feature_vector = extract_features(image_path)

        if feature_vector is not None:
            print("Feature Vector:")
            print(feature_vector)
            print("Vector Length:", len(feature_vector))
        else:
            print("Failed to extract feature vector.")
