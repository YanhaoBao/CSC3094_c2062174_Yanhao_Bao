import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import random
import requests
from tqdm import tqdm
from torchvision import models, transforms
from Metrics import Brightness, Color, Sharpness, Contrast, SSIM, PSNR
from Adjuster import ImageParameterAdjuster
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directories for saving results
results_dir = "results"
adjusted_dir = "adjusted_images"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(adjusted_dir, exist_ok=True)

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

def classify_image(image_path):
    """Use ResNet50 model to classify an image."""
    # Convert image path to PIL Image
    try:
        pil_img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return {
            "predictions": [
                {
                    "label": "unknown",
                    "probability": 0.0
                }
            ]
        }

    image = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted_idx = output.max(1)
        predicted_class = imagenet_classes[predicted_idx]

    probs = torch.nn.functional.softmax(output[0], dim=0)
    return {
        "predictions": [
            {
                "label": predicted_class,
                "probability": probs[predicted_idx].item()
            }
        ]
    }

results = []

def process_image(image_path, is_adjusted=False, adjustment_params=None):
    """Process an image and calculate all metrics"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Calculate SSIM and PSNR
    ref_path = os.path.splitext(image_path)[0] + '_ref' + os.path.splitext(image_path)[1]
    if os.path.exists(ref_path):
        ssim_res = SSIM.calculate_ssim(image_path)
        PSNR_res = PSNR.calculate_psnr(image_path)
    else:
        ssim_res = None
        PSNR_res = None

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

    # Create result dictionary
    result = {
        "Image": image_path,
        "Is_Adjusted": is_adjusted,
        "Adjustment_Params": str(adjustment_params) if adjustment_params else None,
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
        "Predicted_Class": label["predictions"][0]["label"],
        "Classification_Confidence": label["predictions"][0]["probability"]
    }

    # Add SSIM and PSNR
    if ssim_res is not None:
        result["SSIM"] = ssim_res
    if PSNR_res is not None:
        result["PSNR"] = PSNR_res

    results.append(result)
    return result

def apply_parameter_adjustments(image_path, adjustment_params):
    """Apply parameter adjustments to an image and save the result"""
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    # Create adjuster instance
    adjuster = ImageParameterAdjuster(img)

    # Apply adjustments
    adjusted_img = adjuster.adjust_parameters(adjustment_params)

    # Generate filename based on adjustments
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    if "brightness" in adjustment_params:
        base_name += f"_bright_{adjustment_params['brightness']:.2f}"
    if "gaussian_blur" in adjustment_params:
        base_name += f"_blur_{adjustment_params['gaussian_blur']}"
    if "color_balance" in adjustment_params:
        cb = adjustment_params["color_balance"]
        base_name += f"_color_{cb.get('red', 1.0):.2f}_{cb.get('green', 1.0):.2f}_{cb.get('blue', 1.0):.2f}"

    # Save adjusted image
    adjusted_path = os.path.join(adjusted_dir, base_name + ".jpg")
    cv2.imwrite(adjusted_path, adjusted_img)

    # Save original image as reference with _ref suffix
    ref_path = os.path.join(adjusted_dir, base_name + "_ref.jpg")
    cv2.imwrite(ref_path, img)

    return adjusted_path

def save_results():
    """Save results to Excel and create visualizations, including accuracy per adjustment type."""
    if not results:
        print("No results to save")
        return

    # Save to Excel
    df = pd.DataFrame(results)
    excel_path = os.path.join(results_dir, "imagenet_metrics.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"\nResults saved to: {excel_path}")

    # Create visualizations
    print("\nCreating visualizations...")
    
    # Create directory for visualizations
    viz_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # --- Accuracy Calculation and Plotting per Parameter --- 

    # Get original image accuracy
    original_accuracy = df[~df['Is_Adjusted']]['Classification_Confidence'].mean()

    # Function to plot accuracy vs parameter value
    def plot_accuracy_vs_param(param_name, values_dict, xlabel, filename):
        plt.figure(figsize=(10, 6))
        # Add original accuracy as a reference point
        param_values = [values_dict['Original']['value']] + [v['value'] for k, v in values_dict.items() if k != 'Original']
        accuracies = [original_accuracy] + [v['accuracy'] for k, v in values_dict.items() if k != 'Original']
        
        # Sort by parameter value for plotting
        sorted_indices = np.argsort(param_values)
        param_values = [param_values[i] for i in sorted_indices]
        accuracies = [accuracies[i] for i in sorted_indices]
        labels = [str(p) for p in param_values]
        
        plt.plot(labels, accuracies, marker='o', linestyle='-')
        plt.title(f'Accuracy vs {param_name.capitalize()}')
        plt.xlabel(xlabel)
        plt.ylabel('Accuracy (Avg. Confidence)')
        plt.ylim(0, max(accuracies) * 1.1 if accuracies else 1) # Adjust ylim based on data
        plt.grid(True)
        # Add value labels
        for i, txt in enumerate(accuracies):
            plt.annotate(f'{txt:.3f}', (labels[i], accuracies[i]), textcoords="offset points", xytext=(0,5), ha='center')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, filename))
        plt.close()
        print(f"Generated plot: {filename}")

    # --- Contrast Accuracy --- 
    contrast_accuracies = {'Original': {'value': 1.0, 'accuracy': original_accuracy}}
    for params_str in df[df['Is_Adjusted']]['Adjustment_Params'].unique():
        if params_str and 'contrast' in params_str and len(eval(params_str)) == 1: # Ensure it's only contrast
            try:
                params = eval(params_str)
                value = params['contrast']
                acc = df[df['Adjustment_Params'] == params_str]['Classification_Confidence'].mean()
                contrast_accuracies[f"Contrast_{value}"] = {'value': value, 'accuracy': acc}
            except Exception as e:
                print(f"Error processing contrast params {params_str}: {e}")
    if len(contrast_accuracies) > 1:
        plot_accuracy_vs_param("contrast", contrast_accuracies, "Contrast Factor", "accuracy_vs_contrast.png")

    # --- Brightness Accuracy ---
    brightness_accuracies = {'Original': {'value': 1.0, 'accuracy': original_accuracy}}
    for params_str in df[df['Is_Adjusted']]['Adjustment_Params'].unique():
        if params_str and 'brightness' in params_str and len(eval(params_str)) == 1: # Ensure it's only brightness
            try:
                params = eval(params_str)
                value = params['brightness']
                acc = df[df['Adjustment_Params'] == params_str]['Classification_Confidence'].mean()
                brightness_accuracies[f"Brightness_{value}"] = {'value': value, 'accuracy': acc}
            except Exception as e:
                print(f"Error processing brightness params {params_str}: {e}")
    if len(brightness_accuracies) > 1:
       plot_accuracy_vs_param("brightness", brightness_accuracies, "Brightness Factor", "accuracy_vs_brightness.png")

    # --- Blur Accuracy --- 
    # Note: Blur value 0 corresponds to original
    blur_accuracies = {'Original': {'value': 0, 'accuracy': original_accuracy}}
    for params_str in df[df['Is_Adjusted']]['Adjustment_Params'].unique():
        if params_str and 'gaussian_blur' in params_str and len(eval(params_str)) == 1: # Ensure it's only blur
            try:
                params = eval(params_str)
                value = params['gaussian_blur'] # Kernel size
                acc = df[df['Adjustment_Params'] == params_str]['Classification_Confidence'].mean()
                blur_accuracies[f"Blur_{value}"] = {'value': value, 'accuracy': acc}
            except Exception as e:
                print(f"Error processing blur params {params_str}: {e}")
    if len(blur_accuracies) > 1:
        plot_accuracy_vs_param("blur", blur_accuracies, "Gaussian Blur Kernel Size", "accuracy_vs_blur.png")

    # --- Color Balance Accuracy --- 
    color_accuracies = {'Original': {'value': '1.0_1.0_1.0', 'accuracy': original_accuracy}}
    for params_str in df[df['Is_Adjusted']]['Adjustment_Params'].unique():
         if params_str and 'color_balance' in params_str and len(eval(params_str)) == 1: # Ensure it's only color
            try:
                params = eval(params_str)
                cb = params['color_balance']
                # Create a consistent string representation for the label
                value_str = f"{cb.get('red', 1.0):.1f}_{cb.get('green', 1.0):.1f}_{cb.get('blue', 1.0):.1f}"
                acc = df[df['Adjustment_Params'] == params_str]['Classification_Confidence'].mean()
                color_accuracies[f"Color_{value_str}"] = {'value': value_str, 'accuracy': acc}
            except Exception as e:
                print(f"Error processing color params {params_str}: {e}")
                
    # Plotting color needs custom x-axis labels
    if len(color_accuracies) > 1:
        plt.figure(figsize=(12, 6))
        labels = [k for k, v in color_accuracies.items()]
        accuracies = [v['accuracy'] for k, v in color_accuracies.items()]
        plt.plot(range(len(labels)), accuracies, marker='o', linestyle='-')
        plt.title('Accuracy vs Color Balance')
        plt.xlabel('Color Balance (R_G_B)')
        plt.ylabel('Accuracy (Avg. Confidence)')
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        plt.ylim(0, max(accuracies) * 1.1 if accuracies else 1)
        plt.grid(True)
        # Add value labels
        for i, txt in enumerate(accuracies):
             plt.annotate(f'{txt:.3f}', (i, accuracies[i]), textcoords="offset points", xytext=(0,5), ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "accuracy_vs_color.png"))
        plt.close()
        print(f"Generated plot: accuracy_vs_color.png")

    # --- Original Correlation Heatmap and Metric Distributions --- (Keep these as before)
    
    # Create correlation heatmap
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    # Exclude Is_Adjusted if it exists from corr calculation if desired
    if 'Is_Adjusted' in numeric_cols:
         numeric_cols = numeric_cols.drop('Is_Adjusted')
         
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation between Metrics (All Images)")
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "correlation_heatmap.png"))
    plt.close()
    print(f"Generated plot: correlation_heatmap.png")
    
    # Plot distributions for key metrics (optional, can be removed if too many plots)
    sns.set(style="whitegrid")
    metrics_to_plot = ["Brightness_Mean", "Sharpness_Laplacian", "Contrast_Michelson", "SSIM", "PSNR", "Classification_Confidence"]
    for metric in metrics_to_plot:
        if metric in df.columns: 
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x=metric, bins=30, kde=True)
            plt.title(f"Distribution of {metric} (All Images)")
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f"{metric}_distribution.png"))
            plt.close()
            print(f"Generated plot: {metric}_distribution.png")
    
    print(f"\nVisualizations saved to: {viz_dir}")
    # Removed the summary print as individual plots are now generated

def main():
    # --- Define specific adjustment parameters --- 
    contrast_values = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    brightness_values = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    # Use odd kernel sizes for blur
    blur_values = [3, 5, 7, 9, 11] # Kernel sizes must be positive odd integers
    color_values = [
        {"red": 0.8, "green": 1.0, "blue": 1.2},
        {"red": 0.9, "green": 1.0, "blue": 1.1},
        {"red": 1.0, "green": 1.0, "blue": 1.0}, # Original
        {"red": 1.1, "green": 1.0, "blue": 0.9},
        {"red": 1.2, "green": 1.0, "blue": 0.8}
    ]

    adjustment_params_list = []
    # Add contrast adjustments (excluding 1.0 as it's original)
    adjustment_params_list.extend([{"contrast": v} for v in contrast_values if v != 1.0])
    # Add brightness adjustments (excluding 1.0)
    adjustment_params_list.extend([{"brightness": v} for v in brightness_values if v != 1.0])
    # Add blur adjustments (excluding 0, which is handled by original)
    adjustment_params_list.extend([{"gaussian_blur": v} for v in blur_values])
    # Add color adjustments (excluding 1.0, 1.0, 1.0)
    adjustment_params_list.extend([{"color_balance": v} for v in color_values if v != {"red": 1.0, "green": 1.0, "blue": 1.0}])
    
    # --- Get all images from the dataset directory --- 
    data_dir = os.path.join("data", "ILSVRC2012_img_val")
    all_images = []
    print(f"Searching for images in: {data_dir}")
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith('.jpeg'):
                file_path = os.path.join(root, file)
                all_images.append(file_path)

    if not all_images:
        print(f"Error: No JPEG images found in {data_dir}. Please check the path.")
        return

    print(f"Found {len(all_images)} images in {data_dir}.")

    # --- Select a random sample of images --- 
    num_samples = 10000
    if len(all_images) >= num_samples:
        selected_images = random.sample(all_images, num_samples)
        print(f"\nRandomly selected {num_samples} images for processing.")
    else:
        selected_images = all_images # Use all if fewer than num_samples available
        print(f"\nWarning: Fewer than {num_samples} images found. Processing all {len(selected_images)} images.")

    print(f"Processing {len(selected_images)} images with {len(adjustment_params_list)} adjustments each...")
    
    # Initialize results list
    global results
    results = [] 
    
    for img_path in tqdm(selected_images, desc="Processing images"):
        # Process original image first
        process_image(img_path, is_adjusted=False, adjustment_params=None)

        # Apply each defined adjustment and process
        for params in adjustment_params_list:
            adjusted_path = apply_parameter_adjustments(img_path, params)
            if adjusted_path:
                process_image(adjusted_path, is_adjusted=True, adjustment_params=params)

    # --- Save results and generate visualizations --- 
    save_results()
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
