import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.datasets as datasets
from main import process_image
from scipy.stats import pearsonr

# Load Food-101 Dataset
food101_classes = list(datasets.Food101(root="./data", split="test").classes)

# Save outputs
results = []
correct_predictions = 0
total_images = 0

# Food-101 Dataset Path
image_folder = os.path.join("./data/food-101/images")
image_files = []

# Traverse all categories and obtain the paths of .jpg images.
for category in os.listdir(image_folder):
    category_path = os.path.join(image_folder, category)
    if os.path.isdir(category_path):
        for img in os.listdir(category_path):
            if img.endswith(".jpg") or img.endswith(".png"):
                image_files.append((category, os.path.join(category_path, img)))

print(f"Checking Image Folder: {image_folder}")
print(f"Found {len(image_files)} images.")

# Test first 100 image
for i, (true_label, image_path) in enumerate(image_files[:100]):
    print(f"Processing image {i + 1}: {image_path}")

    predicted_class, brightness, sharpness, (color_b, color_g, color_r) = process_image(image_path)

    #
    is_correct = 1 if predicted_class.lower() == true_label.lower() else 0
    correct_predictions += is_correct
    total_images += 1

    # Save data into list
    results.append({
        "True Label": true_label,
        "Predicted Label": predicted_class,
        "Is Correct": is_correct,
        "Brightness": brightness,
        "Sharpness": sharpness,
        "Color_B": color_b,
        "Color_G": color_g,
        "Color_R": color_r
    })

# Calculate Accuracy
accuracy = (correct_predictions / total_images) * 100
print(f"Accuracy: {accuracy:.2f}%")

# Save the output into an Excel file
df = pd.DataFrame(results)
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

excel_path = os.path.join(output_dir, "results.xlsx")
df.to_excel(excel_path, index=False)

# Calculate correlation
brightness_corr, _ = pearsonr(df["Brightness"], df["Is Correct"])
sharpness_corr, _ = pearsonr(df["Sharpness"], df["Is Correct"])
color_b_corr, _ = pearsonr(df["Color_B"], df["Is Correct"])
color_g_corr, _ = pearsonr(df["Color_G"], df["Is Correct"])
color_r_corr, _ = pearsonr(df["Color_R"], df["Is Correct"])

# Print Correlation
print("\nPearson Correlation with Accuracy:")
print(f"Brightness Correlation: {brightness_corr:.3f}")
print(f"Sharpness Correlation: {sharpness_corr:.3f}")
print(f"Color_B Correlation: {color_b_corr:.3f}")
print(f"Color_G Correlation: {color_g_corr:.3f}")
print(f"Color_R Correlation: {color_r_corr:.3f}")

# Visualize the results of Pearson Correlation
sns.set_theme(style="whitegrid")

#Brightness and Accuracy
plt.figure(figsize=(6, 4))
sns.regplot(x=df["Brightness"], y=df["Is Correct"], logistic=True, scatter_kws={"alpha": 0.5})
plt.title("Brightness vs Accuracy")
plt.xlabel("Brightness")
plt.ylabel("Correct Prediction (0=Wrong, 1=Correct)")
plt.savefig(os.path.join(output_dir, "brightness_vs_accuracy.png"))
plt.show()

#Sharpness and Accuracy
plt.figure(figsize=(6, 4))
sns.regplot(x=df["Sharpness"], y=df["Is Correct"], logistic=True, scatter_kws={"alpha": 0.5})
plt.title("Sharpness vs Accuracy")
plt.xlabel("Sharpness")
plt.ylabel("Correct Prediction (0=Wrong, 1=Correct)")
plt.savefig(os.path.join(output_dir, "sharpness_vs_accuracy.png"))
plt.show()

# Color (R) and Accuracy
plt.figure(figsize=(6, 4))
sns.regplot(x=df["Color_R"], y=df["Is Correct"], logistic=True, scatter_kws={"alpha": 0.5}, color="red")
plt.title("Color (Red) vs Accuracy")
plt.xlabel("Red Color Value")
plt.ylabel("Correct Prediction (0=Wrong, 1=Correct)")
plt.savefig(os.path.join(output_dir, "color_r_vs_accuracy.png"))
plt.show()
