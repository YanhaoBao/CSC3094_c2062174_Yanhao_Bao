import numpy as np
from skimage.metrics import peak_signal_noise_ratio
import cv2
import os


def calculate_psnr(image, ref_image, data_range=None):
    """Calculate PSNR between two images (NumPy arrays)."""
    # Basic validation
    if image is None or ref_image is None:
        raise ValueError("Input images cannot be None")
    if image.shape != ref_image.shape:
        raise ValueError(f"Input images must have the same shape. "
                         f"Got {image.shape} and {ref_image.shape}")

    # Determine data_range if not provided
    if data_range is None:
        if image.dtype == np.uint8:
            data_range = 255
        elif image.dtype == np.uint16:
            data_range = 65535
        elif image.dtype in [np.float32, np.float64]:
            data_range = np.max(image) - np.min(image)
            if data_range == 0:  # Handle constant image
                data_range = 1.0
            elif data_range > 1.0 and np.issubdtype(image.dtype, np.floating):
                print(f"Warning: Input float image seems not to be in [0, 1] range (max-min={data_range}). "
                      "PSNR calculation might be inaccurate if data_range is not set correctly.")
        else:
            data_range = 255
            print(f"Warning: Unknown image data type {image.dtype}. "
                  f"Assuming data_range=255. Set data_range explicitly if needed.")

    # Calculate PSNR using scikit-image
    psnr_value = peak_signal_noise_ratio(ref_image, image, data_range=data_range)

    return psnr_value


# Example Usage (can be run directly for testing)
if __name__ == '__main__':
    # Create dummy images
    img1 = np.full((100, 100, 3), 200, dtype=np.uint8)
    img2 = img1.copy()
    img2[50:60, 50:60, :] = 210  # Introduce some difference

    # Calculate PSNR directly with arrays
    try:
        psnr_val = calculate_psnr(img2, img1)
        print(f"PSNR between img2 and img1: {psnr_val:.2f} dB")
    except ValueError as e:
        print(f"Error calculating PSNR: {e}")

    # Identical images
    try:
        psnr_identical = calculate_psnr(img1, img1)
        print(f"PSNR between identical images: {psnr_identical}") # Should be inf
    except ValueError as e:
        print(f"Error calculating PSNR for identical images: {e}")

    # Test with float images
    img_float1 = img1.astype(np.float32) / 255.0
    img_float2 = img2.astype(np.float32) / 255.0
    try:
        # Note: For float images in [0, 1], data_range is usually 1.0 (automatically determined)
        psnr_float = calculate_psnr(img_float2, img_float1)
        print(f"PSNR between float images: {psnr_float:.2f} dB")
    except ValueError as e:
        print(f"Error calculating PSNR for float images: {e}")
