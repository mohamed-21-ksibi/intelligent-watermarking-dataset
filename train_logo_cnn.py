import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# ================== CONFIG ==================
DATA_FOLDER = r"C:\Users\ksibi\Desktop\intelligent-watermarking-dataset\watermarked_images"
LOGO_PATH = r"C:\Users\ksibi\Desktop\intelligent-watermarking-dataset\logo.png"
MODEL_PATH = r"C:\Users\ksibi\Desktop\intelligent-watermarking-dataset\logo_cnn_model.h5"
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 50
# ============================================

# ================== CHARGEMENT DES DONNEES ==================
watermarked_files = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith('.png')]

X = []
for f in watermarked_files:
    img = cv2.imread(os.path.join(DATA_FOLDER, f), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    X.append(img)

X = np.array(X)[..., np.newaxis]

# Charger le logo clair comme cible
logo = cv2.imread(LOGO_PATH, cv2.IMREAD_GRAYSCALE)
logo = cv2.resize(logo, (IMG_SIZE, IMG_SIZE))
logo = logo.astype('float32') / 255.0
Y = np.array([logo for _ in range(len(X))])[..., np.newaxis]

# Split train/test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# ================== CONSTRUCTION DE L'AUTOENCODEUR ==================
def build_autoencoder(input_shape):
    input_img = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2,2), padding='same')(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2), padding='same')(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2,2), padding='same')(x)

    # Decoder
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2,2))(x)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2,2))(x)
    decoded = layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

autoencoder = build_autoencoder((IMG_SIZE, IMG_SIZE, 1))
autoencoder.summary()

# ================== ENTRAINEMENT ==================
history = autoencoder.fit(X_train, Y_train,
                          epochs=EPOCHS,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          validation_data=(X_test, Y_test))

# ================== SAUVEGARDE DU MODELE ==================
autoencoder.save(MODEL_PATH)
print(f"✅ Modèle sauvegardé dans : {MODEL_PATH}")

# ================== AFFICHAGE DE RESULTATS ==================
preds = autoencoder.predict(X_test[:5])

for i in range(5):
    plt.figure(figsize=(6,3))
    # Image watermarked
    plt.subplot(1,2,1)
    plt.imshow(X_test[i].squeeze(), cmap='gray')
    plt.title("Input Watermarked")
    plt.axis('off')
    # Logo reconstruit
    plt.subplot(1,2,2)
    plt.imshow(preds[i].squeeze(), cmap='gray')
    plt.title("Output Logo")
    plt.axis('off')
    plt.show()
