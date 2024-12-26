import h5py

def enhance_images(images, model):
    enhanced_images = []
    for img in images:
        img_flat = img.flatten().reshape(1, -1)
        enhanced_img_flat = model.predict(img_flat)
        enhanced_img = enhanced_img_flat.reshape(img.shape)
        enhanced_images.append(enhanced_img)
    return np.array(enhanced_images)


def save_enhanced_images(X_reduced_mmap, svd_transformers, total_images_loaded, gba_model, output_dir):
    chunk_size = 16
    for i in range(len(svd_transformers)):
        for chunk_index in range(0, 64, chunk_size):
            start_index = i * 64 + chunk_index
            end_index = min(start_index + chunk_size, total_images_loaded)
            if start_index >= total_images_loaded:
                break

            X_reduced_chunk = X_reduced_mmap[start_index:end_index]
            svd = svd_transformers[i]

            try:
                X_reconstructed_chunk = svd.inverse_transform(X_reduced_chunk)
                X_reconstructed_chunk = X_reconstructed_chunk.reshape(-1, 240, 240, 4)

                enhanced_images_chunk = enhance_images(X_reconstructed_chunk, gba_model)

                with h5py.File(os.path.join(output_dir, f"X_enhanced_chunk_{i}_{chunk_index}.h5"), "w") as hf:
                    hf.create_dataset("data", data=enhanced_images_chunk)
                print(f"Saved enhanced chunk {i}_{chunk_index} to disk.")
            except ValueError as e:
                print(f"Error in chunk {i}_{chunk_index}: {e}. Skipping this chunk.")
                continue
