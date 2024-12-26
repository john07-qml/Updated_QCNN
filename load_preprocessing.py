import os
import numpy as np
import random
import sys
import nibabel as nib
from sklearn.decomposition import TruncatedSVD
import pickle
import tensorflow as tf

def load_images(data_dir, batch_size=5):
    images = []
    total_images_loaded = 0
    if os.path.isdir(data_dir):
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".nii"):
                    try:
                        file_path = os.path.join(root, file)
                        nii_img = nib.load(file_path)
                        image_data = nii_img.get_fdata()  # Get the image data as a NumPy array
                        images.append(image_data)
                        total_images_loaded += 1
                        if len(images) >= batch_size:
                            yield np.array(images, dtype=np.float32)
                            images = []
                    except Exception as e:
                        print(f"Error loading image: {file} - {e}")
    if images:
        yield np.array(images, dtype=np.float32)
    print(f"Total images loaded: {total_images_loaded}")

def preprocess_data(data_dir, output_dir, num_components=3600, batch_size=5):
    # If the batched output directory not exist, will make one
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    svd_transformers = []
    batch_index = 0
    for batch in load_images(data_dir, batch_size):
        if batch.shape[0] == batch_size:
            X_batch = np.array([img if img.shape == (240, 240) else np.resize(img, (240, 240)) for img in batch])
            current_num_components = min(num_components, X_batch.shape[1])
            svd = TruncatedSVD(n_components=current_num_components)
            X_reduced_batch = [svd.fit_transform(img) for img in X_batch]
            X_reduced_batch = np.array(X_reduced_batch).reshape((batch_size, 240, 240))
            svd_transformers.append(svd)
            y_batch = np.random.randint(4, size=batch_size)
            y_batch = tf.keras.utils.to_categorical(y_batch, 4)
            batch_file_X = os.path.join(output_dir, f"X_reduced_batch_{batch_index}.npy")
            batch_file_y = os.path.join(output_dir, f"y_batch_{batch_index}.npy")
            np.save(batch_file_X, X_reduced_batch)
            np.save(batch_file_y, y_batch)
            print(f"Saved batch {batch_index} to disk.")
            batch_index += 1

    # Save the SVD transformers
    with open("svd_transformers.pkl", "wb") as f:
        pickle.dump(svd_transformers, f)

    print("Pre-processing completed.")
    return svd_transformers
