import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
from datetime import datetime
from torchvision import models, transforms
from PIL import Image
from Metrics import Brightness, Color, Sharpness, Contrast
import requests

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

save_dir = "captured_images"
os.makedirs(save_dir, exist_ok=True)


def capture_image():
    """Open the camera, take a picture and save it"""
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't open the camera")
            break

        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, frame)
            print(f"Image saved to: {filepath}")
            cap.release()
            cv2.destroyAllWindows()
            return filepath

        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            print("exit")
            return None


def classify_image(image_path):
    """Using ResNet-50 model to classify an image"""
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted_idx = output.max(1)
        predicted_class = imagenet_classes[predicted_idx]

    return predicted_class


results = []


def process_image(image_path):
    img = cv2.imread(image_path)

    # Brightness
    b_mean = Brightness.mean(img)
    b_y = Brightness.YUV(img)
    b_hsv = Brightness.HSV(img)

    # Sharpness
    s_lap = Sharpness.laplacian(img)
    s_sobel = Sharpness.sobel(img)

    # Color
    b, g, r = Color.rgb_average(img)

    # Contrast
    c_mich = Contrast.michelson(img)
    c_rms = Contrast.rms(img)

    # Classification
    label = classify_image(image_path)

    # Append to results
    results.append({
        "Image": image_path,
        "Brightness_Mean": b_mean,
        "Brightness_Y": b_y,
        "Brightness_HSV": b_hsv,
        "Sharpness_Laplacian": s_lap,
        "Sharpness_Sobel": s_sobel,
        "Color_B": b,
        "Color_G": g,
        "Color_R": r,
        "Contrast_Michelson": c_mich,
        "Contrast_RMS": c_rms,
        "Predicted_Class": label
    })

    # Print result
    print(f"Image: {image_path}")
    print(f"Prediction: {label}")
    print(f"Brightness -> Mean: {b_mean:.2f}, Y: {b_y:.2f}, HSV: {b_hsv:.2f}")
    print(f"Sharpness -> Laplacian: {s_lap:.2f}, Sobel: {s_sobel:.2f}")
    print(f"Color -> B: {b:.0f}, G: {g:.0f}, R: {r:.0f}")
    print(f"Contrast -> Michelson: {c_mich:.2f}, RMS: {c_rms:.2f}")


def save_results():
    df = pd.DataFrame(results)
    excel_path = os.path.join("results.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"Results saved to: {excel_path}")

    # Visualizations
    sns.set(style="whitegrid")
    for column in ["Brightness_Mean", "Sharpness_Laplacian", "Contrast_Michelson"]:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[column], kde=True)
        plt.title(f"Distribution of {column}")
        plt.tight_layout()
        fig_path = os.path.join(f"{column}_hist.png")
        plt.savefig(fig_path)
        print(f"Saved: {fig_path}")


if __name__ == "__main__":
    while True:
        path = capture_image()
        if path:
            process_image(path)
            again = input("Capture another? (y/n): ")
            if again.lower() != 'y':
                save_results()
                break
        else:
            break
