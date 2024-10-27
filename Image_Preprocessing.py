import os
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

def load_and_preprocess_image(img_path, cache_dir, output_size=(64, 64, 64)):
    """
    Load and preprocess a 3D NIfTI image.

    This function loads a NIfTI image, resizes it to the target dimensions specified by `output_size`,
    and saves the preprocessed image as a NumPy array in a cache directory.

    Note: In our preprocessing for ADNI data, we initially used `output_size=(91, 109, 91)`.
    
    Important: Other preprocessing steps such as spatial & count normalization, skull stripping,
    brain masking, and cropping were performed using PMOD.

    Parameters:
    - img_path: Path to the input NIfTI image file.
    - cache_dir: Directory to save the preprocessed images as .npy files.
    - output_size: Desired output size for the preprocessed image. Default is (64, 64, 64).

    Returns:
    - preprocessed_img_tensor: A preprocessed image as a NumPy array.
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    # Generate the cache file path
    base_name = os.path.basename(img_path).split('.')[0]
    cached_path = os.path.join(cache_dir, base_name + '_cached.npy')

    # If the preprocessed image is already cached, load it
    if os.path.exists(cached_path):
        return np.load(cached_path)

    try:
        # Load the NIfTI image
        img_data_np = nib.load(img_path).get_fdata()
        img_data_np = img_data_np.astype(np.float32)
        img_tensor = torch.tensor(img_data_np).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

        # Resize the image to the desired output size (default: 64x64x64, we used 91x109x91 for ADNI data)
        preprocessed_img_tensor = F.interpolate(
            img_tensor, size=output_size, mode='trilinear', align_corners=False
        ).squeeze(0).to(torch.float32).numpy()

        # Save the preprocessed image as a NumPy array in the cache directory
        np.save(cached_path, preprocessed_img_tensor)

        return preprocessed_img_tensor

    except Exception as e:
        print(f"Error while processing image {img_path}. Error: {e}")
        return None
