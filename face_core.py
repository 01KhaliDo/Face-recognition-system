import cv2
import os
import numpy as np
import json
from datetime import datetime
import time

# --- Konfiguration ---
DATASET_PATH = "dataset"
MODEL_FILE = "model.yml"
ROLES_FILE = "roles.json"
MAX_IMAGES = 30

# --- Initiera ---
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)

# --- Funktion: Ladda/Spara Roller ---
def load_roles():
    if os.path.exists(ROLES_FILE):
        try:
            with open(ROLES_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_role(name, role):
    roles = load_roles()
    roles[name] = role
    with open(ROLES_FILE, "w") as f:
        json.dump(roles, f)

# --- Funktion: Ansiktsdetektion (Haar Cascade) ---
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_faces(image):
    """
    Returnerar en lista med (x, y, w, h) för alla ansikten.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces


# --- Funktion: Träna modellen ---
def train_model():
    faces = []
    labels = []
    label_map = {}
    current_label = 0

    if not os.path.exists(DATASET_PATH):
        return {}

    print("Hämtar data för träning...")
    for person in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person)
        if not os.path.isdir(person_path):
            continue

        label_map[current_label] = person
        for img in os.listdir(person_path):
            img_path = os.path.join(person_path, img)
            face = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if face is not None:
                faces.append(face)
                labels.append(current_label)
        current_label += 1

    if len(faces) == 0:
        return {}

    print(f"Tränar på {len(faces)} bilder...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_FILE)
    print("Modell uppdaterad och sparad!")
    return label_map
