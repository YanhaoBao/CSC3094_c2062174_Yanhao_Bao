# Image Data Quality Metrics

This project analyzes various image quality metrics and provides tools for evaluating their behavior under different image adjustments. It includes functionality for processing image datasets, visualizing results, and a web interface for live camera analysis and adjustment.

## Features

*   Calculates various image quality metrics: Brightness (Mean, YUV, HSV), Sharpness (Laplacian, Sobel), Color (RGB Average), Contrast (Michelson, RMS), SSIM, PSNR.
*   Applies image adjustments: Brightness, Contrast, Gaussian Blur, Color Balance.
*   Classifies images using a pre-trained ResNet-50 model.
*   Evaluates metric performance against controlled image adjustments using the ImageNet validation set (`evaluate_metrics.py`).
*   Provides a main script (`main.py`) to process images, calculate metrics, apply adjustments, and save results to Excel with visualizations.
*   Includes a Flask-based web UI (`web_ui.py`) for real-time camera feed analysis, allowing interactive adjustment of parameters and viewing classification results.
*   Offers a live camera view with overlayed metrics (`LiveCamera.py`).

## Setup

1.  **Clone the repository (if applicable)**
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download Data (if needed):** https://image-net.org/download.php Download dataset from this website. The datasets used in this project are ILSVRC2012_devkit_t12 and ILSVRC2012_img_val

## Usage

### Main Analysis (`main.py`)

*   Place images to be analyzed in a directory (e.g., create an `data/` folder).
*   Modify the `IMAGE_DIR` variable in `main.py` if necessary.
*   Run the script:
    ```bash
    python main.py
    ```
*   Results (metrics, classifications) will be saved in `results/imagenet_metrics.xlsx`.
*   Visualizations comparing metrics and classification accuracy under different adjustments will be saved in `results/visualizations/`.
*   Adjusted images will be saved in `adjusted_images/`.

### Metric Evaluation (`evaluate_metrics.py`)

*   Ensure the ImageNet validation dataset is in `data/ILSVRC2012_img_val`.
*   Run the script:
    ```bash
    python evaluate_metrics.py
    ```
*   Plots evaluating metric responses to controlled adjustments will be saved in `Metrics Validation results/results/metric_evaluations_avg/`.

### Web UI (`web_ui.py`)

*   Run the Flask application:
    ```bash
    python web_ui.py
    ```
*   Open your web browser and navigate to `http://127.0.0.1:5000`.
*   The interface will show multiple camera feeds: one original and others with different adjustable parameters (brightness, blur, color, contrast).
*   Use the sliders to adjust parameters in real-time and observe the effects on the image and the classification results.

### Live Camera Metrics (`LiveCamera.py`)

*   Run the script:
    ```bash
    python LiveCamera.py
    ```
*   A window will open showing the live camera feed with calculated metrics overlaid. Press 'q' to quit.

## Project Structure

```
    Image Data Quality Metrics/
    ├── Metrics/                  # Modules for calculating individual metrics
    │   ├── Brightness.py
    │   ├── Color.py
    │   ├── Contrast.py
    │   ├── Sharpness.py
    │   ├── PSNR.py
    │   ├── SSIM.py
    │   └── ...
    ├── data/                     # Data directory (e.g., for datasets like ImageNet)
    │   └── ILSVRC2012_img_val/
    ├── adjusted_images/          # Output directory for adjusted images from main.py
    ├── results/                  # Output directory for main.py results (Excel, visualizations)
    │   └── visualizations/
    ├── Metrics Validation results/ # Output directory for evaluate_metrics.py results
    │   └── results/
    │       └── metric_evaluations_avg/
    ├── templates/                # HTML templates for the web UI
    │   └── webCamera.html
    ├── test/                     # Potential directory for unit tests
    ├── Adjuster.py               # Class for applying image parameter adjustments
    ├── evaluate_metrics.py       # Script to evaluate metric performance
    ├── LiveCamera.py             # Script for simple live camera view with metrics
    ├── main.py                   # Main script for batch processing and analysis
    ├── web_ui.py                 # Flask application for the interactive web UI
    │ 
    README.md                 # This file
    │
    requirements.txt          # Project dependencies
``` 