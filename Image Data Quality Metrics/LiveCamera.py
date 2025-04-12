import cv2
import torch
from torchvision import models, transforms
from Metrics import Brightness, Color, Sharpness, Contrast
import requests
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet-50 model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load ImageNet category label
imagenet_classes_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
response = requests.get(imagenet_classes_url)
imagenet_classes = response.text.split("\n")
imagenet_classes = [line.strip() for line in imagenet_classes if line.strip()]

def calculate_metrics(frame):
    """Calculate all metrics for a frame"""
    # Brightness
    b_mean = Brightness.mean(frame)
    b_y = Brightness.YUV(frame)
    b_hsv = Brightness.HSV(frame)

    # Sharpness
    s_lap = Sharpness.laplacian(frame)
    s_sobel = Sharpness.sobel(frame)

    # Color
    b, g, r = Color.rgb_average(frame)

    # Contrast
    c_mich = Contrast.michelson(frame)
    c_rms = Contrast.rms(frame)

    return {
        "brightness": {
            "mean": b_mean,
            "yuv": b_y,
            "hsv": b_hsv
        },
        "sharpness": {
            "laplacian": s_lap,
            "sobel": s_sobel
        },
        "color": {
            "b": b,
            "g": g,
            "r": r
        },
        "contrast": {
            "michelson": c_mich,
            "rms": c_rms
        }
    }

def draw_metrics(frame, metrics):
    """Draw metrics on the frame"""
    height, width = frame.shape[:2]

    # Create a semi-transparent overlay for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Draw metrics
    y_pos = 40
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    color = (255, 255, 255)

    # Brightness
    cv2.putText(frame, f"Brightness (Mean): {metrics['brightness']['mean']:.2f}",
                (20, y_pos), font, font_scale, color, font_thickness)
    y_pos += 30

    # Sharpness
    cv2.putText(frame, f"Sharpness (Lap): {metrics['sharpness']['laplacian']:.2f}",
                (20, y_pos), font, font_scale, color, font_thickness)
    y_pos += 30

    # Color
    cv2.putText(frame,
                f"Color (B,G,R): ({metrics['color']['b']:.0f}, {metrics['color']['g']:.0f}, {metrics['color']['r']:.0f})",
                (20, y_pos), font, font_scale, color, font_thickness)
    y_pos += 30

    # Contrast
    cv2.putText(frame, f"Contrast (Mich): {metrics['contrast']['michelson']:.2f}",
                (20, y_pos), font, font_scale, color, font_thickness)

    return frame

def main():
    cap = cv2.VideoCapture(0)
    fps = 0
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't open the camera")
            break

        # Calculate FPS
        frame_count += 1
        if frame_count >= 30:
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            frame_count = 0
            start_time = time.time()

        # Calculate metrics
        metrics = calculate_metrics(frame)

        # Draw metrics on frame
        frame = draw_metrics(frame, metrics)

        # Draw FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show frame
        cv2.imshow("Real-time Analysis", frame)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()