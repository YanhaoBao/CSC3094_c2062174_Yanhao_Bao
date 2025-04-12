import numpy as np
from skimage.metrics import structural_similarity
import cv2
import os


def calculate_ssim(image_path, data_range=None, multichannel=None, channel_axis=-1, win_size=None):
    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Construct reference image path
    ref_path = os.path.splitext(image_path)[0] + '_ref' + os.path.splitext(image_path)[1]
    ref_image = cv2.imread(ref_path)

    if ref_image is None:
        raise ValueError(f"Could not read reference image at {ref_path}")

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
            if data_range == 0: data_range = 1.0  # Handle constant image
        else:
            data_range = 255  # Default guess

    # Handle multichannel deprecation and default channel axis
    if multichannel is not None:
        print("Warning: 'multichannel' argument is deprecated for SSIM. Use 'channel_axis' instead.")
        if multichannel and channel_axis is None:
            channel_axis = -1  # Assume HWC if multichannel=True and channel_axis is not set
        elif not multichannel:
            channel_axis = None  # Assume grayscale if multichannel=False

    # Check if image is likely color (3D) and set default channel_axis if needed
    if image.ndim == 3 and channel_axis is None:
        channel_axis = -1  # Default to HWC for 3D arrays if axis not specified

    # Call scikit-image SSIM function
    ssim_value = structural_similarity(
        ref_image,
        image,
        win_size=win_size,
        data_range=data_range,
        channel_axis=channel_axis
    )

    return ssim_value


# Example Usage (can be run directly for testing)
if __name__ == '__main__':

    # Create dummy images
    img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    img2 = img1.copy()
    img2 = cv2.GaussianBlur(img2, (7, 7), 1.5)  # Apply blur to reduce similarity

    # Calculate SSIM for color image (assuming HWC format)
    ssim_val_color = calculate_ssim(img1, img2)
    print(f"SSIM between img1 and blurred img2 (color): {ssim_val_color:.4f}")

    # Calculate SSIM for grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ssim_val_gray = calculate_ssim(img1_gray, img2_gray)  # channel_axis defaults to None for 2D
    print(f"SSIM between img1 and blurred img2 (grayscale): {ssim_val_gray:.4f}")

    # Identical images
    ssim_identical = calculate_ssim(img1, img1)
    print(f"SSIM between identical images: {ssim_identical:.4f}")  # Should be 1.0
