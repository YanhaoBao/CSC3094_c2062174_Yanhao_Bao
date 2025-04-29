import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import random

# --- Local Imports ---
from Adjuster import ImageParameterAdjuster
from Metrics import Brightness, Contrast, Sharpness, Color

DATASET_DIR = os.path.join("data", "ILSVRC2012_img_val")
NUM_SAMPLES = 1000
OUTPUT_DIR = "Metrics Validation results/results/metric_evaluations_avg"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameter Ranges to Test (Values from main.py)
BRIGHTNESS_VALUES = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6] # Added 1.0 for baseline
CONTRAST_VALUES = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
# Added 1 (no blur) to the list from main.py
BLUR_VALUES = [1, 3, 5, 7, 9, 11]
# Specific color combinations from main.py
COLOR_COMBINATIONS = [
    {"red": 1.0, "green": 1.0, "blue": 1.0}, # Original baseline
    {"red": 0.8, "green": 1.0, "blue": 1.2},
    {"red": 0.9, "green": 1.0, "blue": 1.1},
    {"red": 1.1, "green": 1.0, "blue": 0.9},
    {"red": 1.2, "green": 1.0, "blue": 0.8}
]

# --- Helper Functions ---

def load_image(path):
    """Loads an image using OpenCV."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at: {path}")
    return img

def plot_results(applied_values, calculated_values, xlabel, ylabel, title, filename):
    """Generates and saves a plot."""
    plt.figure(figsize=(8, 6))
    plt.plot(applied_values, calculated_values, marker='o', linestyle='-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved plot: {filepath}")

def get_image_samples(dataset_dir, num_samples):
    """Finds all JPEG images and returns a random sample of paths."""
    all_image_paths = []
    print(f"Searching for JPEG images in: {dataset_dir}...")
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            # Check for both .jpeg and .jpg extensions (case-insensitive)
            if file.lower().endswith(('.jpeg', '.jpg')):
                all_image_paths.append(os.path.join(root, file))

    if not all_image_paths:
        raise FileNotFoundError(f"No JPEG or JPG images found in {dataset_dir}")

    print(f"Found {len(all_image_paths)} images.")

    if len(all_image_paths) < num_samples:
        print(f"Warning: Found fewer images ({len(all_image_paths)}) than requested ({num_samples}). Using all found images.")
        return all_image_paths
    else:
        selected_paths = random.sample(all_image_paths, num_samples)
        print(f"Randomly selected {num_samples} images.")
        return selected_paths

# --- Evaluation Functions ---

def evaluate_brightness(sample_image_paths):
    """Evaluates brightness metric against applied brightness adjustment across multiple images."""
    print("\nEvaluating Brightness (Averaged)...")
    applied_factors = []
    avg_calculated_metrics = []

    for factor in tqdm(BRIGHTNESS_VALUES, desc="Brightness Values"):
        metric_values_for_factor = []
        num_processed = 0
        for img_path in sample_image_paths:
            try:
                base_image = load_image(img_path)
                adjuster = ImageParameterAdjuster(base_image.copy())
                adjusted_image = adjuster.adjust_brightness(factor)
                metric_value = Brightness.HSV(adjusted_image)
                metric_values_for_factor.append(metric_value)
                num_processed += 1
            except FileNotFoundError:
                # print(f"Warning: Skipping missing image {img_path}") # Can be noisy
                continue
            except ValueError as e:
                # print(f"Warning: Skipping image {img_path} for factor {factor}: {e}") # Can be noisy
                continue
            except Exception as e:
                print(f"Warning: Error processing image {img_path} for factor {factor}: {e}")
                continue

        if num_processed > 0:
            avg_metric = np.mean(metric_values_for_factor)
            applied_factors.append(factor)
            avg_calculated_metrics.append(avg_metric)
        else:
            print(f"Warning: No images successfully processed for brightness factor {factor}")

    plot_results(applied_factors, avg_calculated_metrics,
                 "Applied Brightness Factor",
                 f"Avg. Calculated Brightness (HSV Value over {NUM_SAMPLES} images)",
                 "Average Brightness Metric Evaluation",
                 "avg_brightness_evaluation.png")

def evaluate_contrast(sample_image_paths):
    """Evaluates contrast metric against applied contrast adjustment across multiple images."""
    print("\nEvaluating Contrast (Averaged)...")
    applied_factors = []
    avg_calculated_metrics = []

    for factor in tqdm(CONTRAST_VALUES, desc="Contrast Values"):
        metric_values_for_factor = []
        num_processed = 0
        for img_path in sample_image_paths:
            try:
                base_image = load_image(img_path)
                adjuster = ImageParameterAdjuster(base_image.copy())
                adjusted_image = adjuster.adjust_contrast(factor)
                metric_value = Contrast.rms(adjusted_image)
                metric_values_for_factor.append(metric_value)
                num_processed += 1
            except FileNotFoundError:
                continue
            except ValueError as e:
                continue
            except Exception as e:
                print(f"Warning: Error processing image {img_path} for factor {factor}: {e}")
                continue

        if num_processed > 0:
            avg_metric = np.mean(metric_values_for_factor)
            applied_factors.append(factor)
            avg_calculated_metrics.append(avg_metric)
        else:
            print(f"Warning: No images successfully processed for contrast factor {factor}")

    plot_results(applied_factors, avg_calculated_metrics,
                 "Applied Contrast Factor",
                 f"Avg. Calculated Contrast (RMS over {NUM_SAMPLES} images)",
                 "Average Contrast Metric Evaluation",
                 "avg_contrast_evaluation.png")

def evaluate_blur(sample_image_paths):
    """Evaluates sharpness metric against applied Gaussian blur across multiple images."""
    print("\nEvaluating Blur (Sharpness Metric, Averaged)...")
    applied_kernel_sizes = []
    avg_calculated_metrics = []

    for k_size in tqdm(BLUR_VALUES, desc="Blur Kernel Sizes"):
        metric_values_for_ksize = []
        num_processed = 0
        for img_path in sample_image_paths:
            try:
                base_image = load_image(img_path)
                adjuster = ImageParameterAdjuster(base_image.copy())
                if k_size == 1:
                    adjusted_image = base_image.copy()
                else:
                    adjusted_image = adjuster.adjust_gaussian_blur(k_size)
                metric_value = Sharpness.laplacian(adjusted_image)
                metric_values_for_ksize.append(metric_value)
                num_processed += 1
            except FileNotFoundError:
                continue
            except ValueError as e:
                continue
            except Exception as e:
                print(f"Warning: Error processing image {img_path} for k_size {k_size}: {e}")
                continue

        if num_processed > 0:
            avg_metric = np.mean(metric_values_for_ksize)
            applied_kernel_sizes.append(k_size)
            avg_calculated_metrics.append(avg_metric)
        else:
            print(f"Warning: No images successfully processed for blur k_size {k_size}")

    plot_results(applied_kernel_sizes, avg_calculated_metrics,
                 "Applied Gaussian Blur Kernel Size (Odd Numbers)",
                 f"Avg. Calculated Sharpness (Laplacian Variance over {NUM_SAMPLES} images)",
                 "Average Sharpness Metric Evaluation (vs. Blur)",
                 "avg_sharpness_vs_blur_evaluation.png")

def evaluate_color(sample_image_paths):
    """Evaluates color metric against applied color gain adjustments (one channel at a time) across multiple images."""
    print("\nEvaluating Color Balance (Averaged)...")

    # --- Evaluate Specific Color Combinations from main.py --- 
    applied_combination_labels = []
    avg_means_per_combination = {'b': [], 'g': [], 'r': []}

    for combo in tqdm(COLOR_COMBINATIONS, desc="Color Combinations"):
        means_for_combo = {'b': [], 'g': [], 'r': []}
        num_processed = 0
        # Create a label for the plot axis
        combo_label = f"R:{combo['red']:.1f}_G:{combo['green']:.1f}_B:{combo['blue']:.1f}"

        for img_path in sample_image_paths:
            try:
                base_image = load_image(img_path)
                adjuster = ImageParameterAdjuster(base_image.copy())
                # Apply the specific R, G, B gains from the current combination
                adjusted_image = adjuster.adjust_color_balance(red_gain=combo['red'], green_gain=combo['green'], blue_gain=combo['blue'])
                b_mean, g_mean, r_mean = Color.rgb_average(adjusted_image)
                means_for_combo['b'].append(b_mean)
                means_for_combo['g'].append(g_mean)
                means_for_combo['r'].append(r_mean)
                num_processed += 1
            except FileNotFoundError:
                continue
            except ValueError as e:
                continue # Errors might occur if gains are outside Adjuster limits
            except Exception as e:
                 print(f"Warning: Error processing image {img_path} for color combo {combo_label}: {e}")
                 continue

        if num_processed > 0:
            applied_combination_labels.append(combo_label)
            avg_means_per_combination['b'].append(np.mean(means_for_combo['b']))
            avg_means_per_combination['g'].append(np.mean(means_for_combo['g']))
            avg_means_per_combination['r'].append(np.mean(means_for_combo['r']))
        else:
             print(f"Warning: No images successfully processed for color combo {combo_label}")

    # Plot Color Combination vs Average R, G, B means
    plt.figure(figsize=(12, 7)) # Adjusted figure size for potentially long labels
    x_indices = np.arange(len(applied_combination_labels))
    plt.plot(x_indices, avg_means_per_combination['r'], marker='o', linestyle='-', color='red', label='Avg Red Channel')
    plt.plot(x_indices, avg_means_per_combination['g'], marker='s', linestyle='--', color='green', label='Avg Green Channel')
    plt.plot(x_indices, avg_means_per_combination['b'], marker='^', linestyle=':', color='blue', label='Avg Blue Channel')

    plt.xlabel("Applied Color Combination (R_G_B)")
    plt.xticks(x_indices, applied_combination_labels, rotation=45, ha='right') # Use labels for x-axis ticks
    plt.ylabel(f"Avg. Calculated Channel Value (over {NUM_SAMPLES} images)")
    plt.title("Average Color Metric Evaluation (Specific Combinations)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    filepath = os.path.join(OUTPUT_DIR, "avg_color_evaluation_combinations.png")
    plt.savefig(filepath)
    plt.close()
    print(f"Saved plot: {filepath}")

# --- Main Execution ---
if __name__ == "__main__":
    try:
        sample_image_paths = get_image_samples(DATASET_DIR, NUM_SAMPLES)

        if not sample_image_paths:
            print("No images selected for evaluation. Exiting.")
        else:
            evaluate_brightness(sample_image_paths)
            evaluate_contrast(sample_image_paths)
            evaluate_blur(sample_image_paths)
            evaluate_color(sample_image_paths)

            print(f"\nEvaluation complete. Plots saved in: {OUTPUT_DIR}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the DATASET_DIR is correct and contains JPEG images.")
    except ImportError as e:
        print(f"Error: {e}")
        print("Please ensure necessary libraries (cv2, numpy, matplotlib) and local modules (Adjuster, Metrics) are installed and accessible.")
        if "matplotlib" in str(e):
            print("Try running: pip install matplotlib")
    except Exception as e:
        print(f"An unexpected error occurred: {e}") 