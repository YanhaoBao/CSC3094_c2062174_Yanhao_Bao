import pytest
import cv2
import numpy as np
from Metrics.Brightness import mean, YUV, HSV
from Metrics.Color import rgb_average
from Metrics.Sharpness import laplacian, sobel
from Metrics.Contrast import michelson, rms
from Metrics.PSNR import calculate_psnr
from Metrics.SSIM import calculate_ssim


@pytest.fixture
def sample_image(tmp_path):
    img_path = tmp_path / "test.jpg"

    height, width = 100, 100
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            img[i, j] = [
                int(255 * j / width),  # Blue gradient
                int(255 * i / height),  # Green gradient
                int(255 * (1 - i / height))  # Red gradient
            ]
    
    noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    
    cv2.imwrite(str(img_path), img)
    return str(img_path)


def test_brightness_metrics(sample_image):
    img = cv2.imread(sample_image)

    mean_value = mean(img)
    assert 0 <= mean_value <= 255

    yuv_value = YUV(img)
    assert 0 <= yuv_value <= 255

    hsv_value = HSV(img)
    assert 0 <= hsv_value <= 255


def test_color_metrics(sample_image):
    img = cv2.imread(sample_image)
    
    b, g, r = rgb_average(img)
    assert 0 <= b <= 255
    assert 0 <= g <= 255
    assert 0 <= r <= 255


def test_sharpness_metrics(sample_image):
    img = cv2.imread(sample_image)

    lap_value = laplacian(img)
    assert lap_value >= 0

    sob_value = sobel(img)
    assert sob_value >= 0


def test_contrast_metrics(sample_image):
    img = cv2.imread(sample_image)

    mich_value = michelson(img)
    assert 0 <= mich_value <= 1

    rms_value = rms(img)
    assert 0 <= rms_value <= 1


def test_comparison_metrics(sample_image):
    psnr_value = calculate_psnr(sample_image)
    assert psnr_value > 0

    ssim_value = calculate_ssim(sample_image)
    assert 0 <= ssim_value <= 1


def test_invalid_image():
    with pytest.raises(ValueError):
        mean("non_existent.jpg")
    
    with pytest.raises(ValueError):
        YUV("non_existent.jpg")
    
    with pytest.raises(ValueError):
        HSV("non_existent.jpg")
    
    with pytest.raises(ValueError):
        rgb_average("non_existent.jpg")
    
    with pytest.raises(ValueError):
        laplacian("non_existent.jpg")
    
    with pytest.raises(ValueError):
        sobel("non_existent.jpg")
    
    with pytest.raises(ValueError):
        michelson("non_existent.jpg")
    
    with pytest.raises(ValueError):
        rms("non_existent.jpg")
    
    with pytest.raises(ValueError):
        calculate_psnr("non_existent.jpg")
    
    with pytest.raises(ValueError):
        calculate_ssim("non_existent.jpg")
