import load_preprocessing
import enhancing_image
import memory_mapped
import QCNN_model
import training_evaluation
import os
import numpy as np
import sys
import nibabel as nib
import pennylane as qml
import tensorflow as tf


data_dir = r"../archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
output_dir = (
    r"./QCNN-RESULT/debugOutput/"
)
total_images_loaded = 1845
num_components = 5
batch_index = 0

# Step 1: Preprocess Data
# load_preprocessing.preprocess_data(data_dir, output_dir)

# Step 2: Create memory-mapped arrays
X_reduced_mmap, y_mmap = memory_mapped.create_memory_mapped_arrays(total_images_loaded, num_components)
memory_mapped.save_batched_data_to_mmap(output_dir, batch_index, X_reduced_mmap, y_mmap)

X_train, X_test, y_train, y_test = memory_mapped.load_bached_data_from_mmap(X_reduced_mmap, y_mmap, total_images_loaded, num_components)

# Step 3: Train the QCNN model
input_shape = num_components  # Using reduced feature size for training
model, history = training_evaluation.train_qcnn_model(X_train, y_train, X_test, y_test, input_shape)

# Plotting the number of epochs vs accuracy
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Step 4: Enhance images
# enhancing_image.save_enhanced_images()

