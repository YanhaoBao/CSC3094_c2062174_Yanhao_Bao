import cv2
import numpy as np
import os


class ImageParameterAdjuster:
    def __init__(self, image):
        self.original_image = image.copy()
        self.current_image = image.copy()
        self.adjustment_history = []

    def reset(self):
        self.current_image = self.original_image.copy()
        self.adjustment_history = []
        return self.current_image

    def adjust_brightness(self, factor):
        if not 0.0 <= factor <= 4.0:
            raise ValueError("Brightness factor must be between 0.0 and 4.0")

        adjusted = cv2.convertScaleAbs(self.current_image, alpha=factor, beta=0)
        self.current_image = adjusted
        self.adjustment_history.append(("brightness", factor))
        return adjusted

    def adjust_contrast(self, factor):
        if not 0.5 <= factor <= 3.0:
            raise ValueError("Contrast factor must be between 0.5 and 3.0")

        mean_intensity = np.mean(self.current_image)
        adjusted = np.clip(factor * (self.current_image - mean_intensity) + mean_intensity, 0, 255).astype(np.uint8)

        self.current_image = adjusted
        self.adjustment_history.append(("contrast", factor))
        return adjusted

    def adjust_gaussian_blur(self, kernel_size):
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Kernel size must be positive and odd")

        blurred = cv2.GaussianBlur(
            self.current_image, (kernel_size, kernel_size), sigmaX=0
        )
        self.current_image = blurred
        self.adjustment_history.append(("blur", kernel_size))
        return blurred

    def adjust_motion_blur(self, kernel_size, angle=0):
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Kernel size must be positive and odd")

        kernel = np.zeros((kernel_size, kernel_size))
        center = (kernel_size - 1) // 2
        radians = np.deg2rad(angle)
        dx = np.cos(radians)
        dy = np.sin(radians)

        for i in range(kernel_size):
            x = int(center + (i - center) * dx)
            y = int(center + (i - center) * dy)
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1
        kernel /= np.sum(kernel)

        blurred = cv2.filter2D(self.current_image, -1, kernel)
        self.current_image = blurred
        self.adjustment_history.append(("motion_blur", (kernel_size, angle)))
        return blurred

    def adjust_color_balance(self, red_gain=1.0, green_gain=1.0, blue_gain=1.0):
        gains = [blue_gain, green_gain, red_gain]

        for gain in gains:
            if not 0.0 <= gain <= 4.0:
                raise ValueError("Color gain must be between 0.0 and 4.0")

        adjusted = self.current_image.copy().astype(np.float32)
        for c in range(3):
            adjusted[:, :, c] = np.clip(adjusted[:, :, c] * gains[c], 0, 255)

        self.current_image = adjusted.astype(np.uint8)
        self.adjustment_history.append(("color", (red_gain, green_gain, blue_gain)))
        return self.current_image

    def adjust_parameters(self, params_dict):
        self.reset()  

        if "brightness" in params_dict:
            self.adjust_brightness(params_dict["brightness"])

        if "contrast" in params_dict:
            self.adjust_contrast(params_dict["contrast"])

        if "gaussian_blur" in params_dict:
            self.adjust_gaussian_blur(params_dict["gaussian_blur"])

        if "motion_blur" in params_dict:
            mb_params = params_dict["motion_blur"]
            self.adjust_motion_blur(mb_params["kernel_size"], mb_params.get("angle", 0))

        if "color_balance" in params_dict:
            cb = params_dict["color_balance"]
            self.adjust_color_balance(
                cb.get("red", 1.0), cb.get("green", 1.0), cb.get("blue", 1.0)
            )

        return self.current_image

    def batch_adjust(self, adjustments):
        results = []
        for params in adjustments:
            self.reset()
            try:
                adjusted = self.adjust_parameters(params)
                results.append({"params": params, "image": adjusted})
            except Exception as e:
                results.append({"error": str(e), "params": params})
        return results

    def save_image(self, output_dir="adjusted_images"):
        """Save the current adjusted image with a meaningful name based on adjustments"""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename based on adjustment history
        filename_parts = []
        for adjustment, value in self.adjustment_history:
            if adjustment == "brightness":
                filename_parts.append(f"bright_{value:.2f}")
            elif adjustment == "contrast":
                filename_parts.append(f"contrast_{value:.2f}")
            elif adjustment == "blur":
                filename_parts.append(f"blur_{value}")
            elif adjustment == "motion_blur":
                kernel_size, angle = value
                filename_parts.append(f"motion_{kernel_size}_{angle}")
            elif adjustment == "color":
                r, g, b = value
                filename_parts.append(f"color_{r:.2f}_{g:.2f}_{b:.2f}")
        
        if not filename_parts:
            filename = "original.jpg"
        else:
            filename = "_".join(filename_parts) + ".jpg"
        
        # Save the image
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, self.current_image)
        return output_path

    @staticmethod
    def generate_preset_adjustments():
        return [{"brightness": f} for f in [0.5, 0.8, 1.2, 1.5]] + [
            {"gaussian_blur": k} for k in [5, 9, 15]
        ]
