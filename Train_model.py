import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import cv2
import numpy as np

# Step 1: Load and Preprocess Dataset
def load_dataset(dataset_path):
    data = []
    labels = []

    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (80, 45))  # Adjust dimensions to your needs
            img = img / 255.0  # Normalize pixel values
            data.append(img)
            labels.append(label)

    return np.array(data), np.array(labels)

# Specify the path to your dataset folder
dataset_path = r"/home/prajwal/Desktop/Work/Datasets/1/images/"  # Replace with your dataset path

data, labels = load_dataset(dataset_path)

# Encode string labels into numerical format
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Step 2: Split the Dataset into Training and Validation Sets
train_data, val_data, train_labels, val_labels = train_test_split(data, encoded_labels, test_size=0.2, random_state=42)

# Step 3: Define the Model Architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(45, 80, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(set(encoded_labels)), activation='softmax')  # Adjust output units based on your classes
])

# Step 4: Compile the Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 5: Train the Model
model.fit(train_data, train_labels, epochs=10, validation_data=(val_data, val_labels))

# Step 6: Save the Trained Model
# Specify the path where you want to save the trained model
model_save_path = "/home/prajwal/Desktop/Work/Datasets/1/Model/My_model.h5"
model.save(model_save_path)
