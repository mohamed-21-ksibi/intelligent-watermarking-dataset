import cv2
import numpy as np
import os
import random

# ================== CONFIG ==================
OUTPUT_FOLDER = r"C:\Users\ksibi\Desktop\intelligent-watermarking-dataset\watermarked_images"
LOGO_PATH = r"C:\Users\ksibi\Desktop\intelligent-watermarking-dataset\logo.png"
NUM_IMAGES = 300
LOGO_SIZE = 100
ALPHA = 0.15
HOST_IMAGES_FOLDER = r"C:\Users\ksibi\Desktop\intelligent-watermarking-dataset\original_images"
# ============================================

# Créer le dossier de sortie si il n’existe pas
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Charger ou créer un logo
if os.path.isfile(LOGO_PATH):
    logo = cv2.imread(LOGO_PATH, cv2.IMREAD_GRAYSCALE)
    logo = cv2.resize(logo, (LOGO_SIZE, LOGO_SIZE))
else:
    # Logo synthétique si aucun fichier
    logo = np.full((LOGO_SIZE, LOGO_SIZE), 255, np.uint8)
    cv2.putText(logo, "LOGO", (5, LOGO_SIZE-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3)

# Liste des images hôtes si existantes
host_files = [f for f in os.listdir(HOST_IMAGES_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Fonction pour créer une image hôte synthétique
def create_synthetic_host(h=256, w=256):
    img = np.random.randint(0, 255, (h, w), dtype=np.uint8)
    return img

# Génération des images
for i in range(NUM_IMAGES):
    # Choisir un hôte existant ou créer synthétique
    if host_files:
        img_name = np.random.choice(host_files)
        host_img = cv2.imread(os.path.join(HOST_IMAGES_FOLDER, img_name), cv2.IMREAD_GRAYSCALE)
        if host_img is None:
            host_img = create_synthetic_host()
    else:
        host_img = create_synthetic_host()

    h, w = host_img.shape
    if h < LOGO_SIZE or w < LOGO_SIZE:
        host_img = cv2.resize(host_img, (max(w, LOGO_SIZE), max(h, LOGO_SIZE)))
        h, w = host_img.shape

    # Position aléatoire du logo
    x = random.randint(0, w - LOGO_SIZE)
    y = random.randint(0, h - LOGO_SIZE)

    # Logo flou et bruité
    noisy_logo = logo.copy()
    noise = np.random.normal(0, 50, noisy_logo.shape).astype(np.uint8)
    noisy_logo = cv2.add(noisy_logo, noise)
    noisy_logo = cv2.GaussianBlur(noisy_logo, (51, 51), 0)

    # Insertion invisible
    roi = host_img[y:y+LOGO_SIZE, x:x+LOGO_SIZE]
    blended = cv2.addWeighted(roi, 1 - ALPHA, noisy_logo, ALPHA, 0)
    host_img[y:y+LOGO_SIZE, x:x+LOGO_SIZE] = blended

    # Sauvegarde
    out_name = f"watermarked_{i+1:03d}.png"
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, out_name), host_img)

print(f"✅ {NUM_IMAGES} images créées dans '{OUTPUT_FOLDER}'")
