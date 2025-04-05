import os
import torch
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms, models
from Metrics.Brightness import mean as brightness
from Metrics.Sharpness import laplacian as sharpness
from Metrics.Color import rgb_average as color
from Metrics.Contrast import rms as contrast
from PIL import Image

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet-50
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
model.eval()
imagenet_classes = models.ResNet50_Weights.IMAGENET1K_V1.meta["categories"]

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset path
data_dir = os.path.join("data", "ILSVRC2012_img_val")

# Collect images (if labels known)
image_files = [os.path.join(data_dir, img) for img in os.listdir(data_dir) if img.endswith(".JPEG")]
labels = ["unknown"] * len(image_files)  # 可替换成真实标签

# Evaluation
results = []

for path, label in zip(image_files[:100], labels[:100]):  # 测试前100张
    img = cv2.imread(path)

    # Calculate metrics
    b_mean = brightness(img)
    s_lap = sharpness(img)
    r, g, b = color(img)
    c_rms = contrast(img)

    # Classification
    image = Image.open(path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        _, pred_idx = torch.max(output, 1)
        pred_label = imagenet_classes[pred_idx.item()]

    correct = 0
    results.append({
        "Image": path,
        "Label": label,
        "Predicted": pred_label,
        "Correct": correct,
        "Brightness": b_mean,
        "Sharpness": s_lap,
        "Color_R": r,
        "Contrast": c_rms
    })

# DataFrame
df = pd.DataFrame(results)
print("DataFrame Columns:", df.columns)

# Plot
plt.figure(figsize=(10, 6))
window = 10
x = range(len(df) - window + 1)


def moving_avg(column):
    if column not in df.columns:
        print(f"[ERROR] Column '{column}' not found in DataFrame!")
        return [0] * len(x)
    return df[column].rolling(window=window).mean().iloc[window - 1:]


def moving_acc():
    return df["Correct"].rolling(window=window).mean().iloc[window - 1:]

plt.plot(x, moving_avg("Brightness"), label="Brightness")
plt.plot(x, moving_avg("Sharpness"), label="Sharpness")
plt.plot(x, moving_avg("Color_R"), label="Color R")
plt.plot(x, moving_avg("Contrast"), label="Contrast")
plt.ylabel("Metric Value")
plt.xlabel("Image Index (Rolling Window Avg)")
plt.legend()
plt.title("Image Quality Metrics over Test Samples")
plt.tight_layout()
plt.savefig("output/quality_vs_index.png")
plt.show()

# Accuracy summary
top1 = df["Correct"].mean()
print(f"Top-1 Accuracy: {top1 * 100:.2f}%")