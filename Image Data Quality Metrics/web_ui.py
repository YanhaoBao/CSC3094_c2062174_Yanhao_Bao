from flask import Flask, render_template, Response, request, jsonify
import cv2
import threading
import time
from Adjuster import ImageParameterAdjuster
from main import classify_image
from PIL import Image
from collections import deque
import random
from Metrics.PSNR import calculate_psnr
from Metrics.SSIM import calculate_ssim
import numpy as np
import warnings

app = Flask(__name__)
lock = threading.Lock()

IMG_SIZE = (640, 480)
REFRESH_RATE = 0.15
HISTORY_LENGTH = 10000
CLASSIFICATION_INTERVAL = 0.5

# Updated PROCESSORS dictionary
PROCESSORS = {
    "original": {"params": {}, "controls": []},
    "adjustable": {
        "params": {
            "brightness": 1.0,
            "gaussian_blur": 1, # Use internal name
            "red_gain": 1.0,
            "green_gain": 1.0,
            "blue_gain": 1.0,
            "contrast_factor": 1.0,
        },
        "controls": [
            # Control tuples: (param_name_frontend, min, max, default, step)
            ("brightness", 0.0, 4.0, 1.0, 0.1),
            ("blur_size", 1, 31, 1, 2), # Frontend uses blur_size
            ("red_gain", 0.0, 4.0, 1.0, 0.1),
            ("green_gain", 0.0, 4.0, 1.0, 0.1),
            ("blue_gain", 0.0, 4.0, 1.0, 0.1),
            ("contrast_factor", 0.5, 3.0, 1.0, 0.1),
        ],
    },
}

# Initialize current_data structure dynamically
current_data = {}
for name, config in PROCESSORS.items():
    data_entry = {
        "frame": None,
        "prediction": "N/A",
        "params": config["params"].copy(),
        "latest_metrics": { # Stores latest calculated/measured values
            "timestamp": None,
            "probability": 0.0, # Model output probability
            "ssim": None,       # Calculated SSIM (vs original for adjustable, 1.0 for original)
            "psnr": None        # Calculated PSNR (vs original for adjustable, inf for original)
        },
        "history": { # History remains for the chart
            "timestamps": deque(maxlen=HISTORY_LENGTH),
            "probability": deque(maxlen=HISTORY_LENGTH),
            "psnr": deque(maxlen=HISTORY_LENGTH),
            "ssim": deque(maxlen=HISTORY_LENGTH),
        },
    }
    # Add parameter history only for adjustable
    if name == "adjustable":
        for param_key in config["params"].keys():
             data_entry["history"][param_key] = deque(maxlen=HISTORY_LENGTH)
    # Add measured metrics slots only for original
    elif name == "original":
        data_entry["latest_metrics"]["measured_brightness"] = None
        data_entry["latest_metrics"]["measured_contrast"] = None
        data_entry["latest_metrics"]["measured_sharpness"] = None # Renamed from measured_blur (Laplacian Variance)
        data_entry["latest_metrics"]["measured_avg_red"] = None
        data_entry["latest_metrics"]["measured_avg_green"] = None
        data_entry["latest_metrics"]["measured_avg_blue"] = None

    current_data[name] = data_entry


@app.route("/")
def index():
    # Pass the updated PROCESSORS structure, specifically for controls in adjustable
    return render_template("webCamera.html", processors=PROCESSORS)


@app.route("/video/<name>")
def video_feed(name):
    if name not in current_data:
        return "Invalid processor name", 404
    def generate():
        while True:
            with lock:
                frame = current_data[name]["frame"]
            if frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(REFRESH_RATE)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/predictions")
def get_predictions():
    with lock:
        return jsonify(
            {name: data["prediction"] for name, data in current_data.items()}
        )


@app.route("/update", methods=["POST"])
def update_params():
    data = request.get_json()
    name = data["name"]
    param = data["param"] # This is the param name from the frontend slider (e.g., 'brightness', 'blur_size')
    value = float(data["value"])

    if name == "adjustable":
        with lock:
            params = current_data[name]["params"]
            if param == "brightness":
                params["brightness"] = max(0.0, min(4.0, value))
            elif param == "blur_size": # Frontend sends 'blur_size'
                value = int(value)
                value = value if value % 2 == 1 else value + 1
                params["gaussian_blur"] = max(1, min(31, value)) # Update internal 'gaussian_blur'
            elif param in ["red_gain", "green_gain", "blue_gain"]:
                params[param] = max(0.0, min(4.0, value))
            elif param == "contrast_factor":
                params["contrast_factor"] = max(0.5, min(3.0, value))
            # else: # Optional: handle unknown param?
            #     return jsonify(success=False, message="Unknown parameter"), 400
    else:
        return jsonify(success=False, message="Invalid processor name for update"), 400


    return jsonify(success=True)


# --- Image Analysis Helper Functions ---

def calculate_sharpness(image):
    """Calculates sharpness using Laplacian variance. Higher is sharper."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Variance of Laplacian
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var

def calculate_brightness_contrast(image):
    """Calculates average brightness and contrast (std dev)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()
    contrast = gray.std()
    return brightness, contrast

def calculate_color_averages(image):
    """Calculates the average value for each color channel (BGR order)."""
    # Calculate mean for each channel
    avg_b, avg_g, avg_r = cv2.mean(image)[:3] # cv2.mean returns mean for each channel + alpha if present
    return avg_r, avg_g, avg_b # Return in RGB order for consistency

# --- Main Image Processing Loop ---
def image_processor():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open the Camera")

    last_classification_time = 0

    while True:
        ret, base_image = cap.read()
        if not ret:
            print("Failed to read image")
            continue
        base_image = cv2.resize(base_image, IMG_SIZE)
        start_time = time.time()
        current_time = time.time() # Get current time early

        # --- Analyze the base image (for 'original' display and history) ---
        measured_sharpness = calculate_sharpness(base_image) # Renamed from measured_blur
        measured_brightness, measured_contrast = calculate_brightness_contrast(base_image)
        measured_avg_r, measured_avg_g, measured_avg_b = calculate_color_averages(base_image)

        # Update the 'original' stream's latest metrics AND history
        with lock:
            orig_latest = current_data["original"]["latest_metrics"]
            orig_history = current_data["original"]["history"]

            # Update latest
            orig_latest["timestamp"] = current_time
            orig_latest["measured_sharpness"] = measured_sharpness # Renamed
            orig_latest["measured_brightness"] = measured_brightness
            orig_latest["measured_contrast"] = measured_contrast
            orig_latest["measured_avg_red"] = measured_avg_r
            orig_latest["measured_avg_green"] = measured_avg_g
            orig_latest["measured_avg_blue"] = measured_avg_b
            
            # Append to history (ensure history deques exist from init)
            if "measured_sharpness" in orig_history: orig_history["measured_sharpness"].append(measured_sharpness) # Renamed
            if "measured_brightness" in orig_history: orig_history["measured_brightness"].append(measured_brightness)
            if "measured_contrast" in orig_history: orig_history["measured_contrast"].append(measured_contrast)
            if "measured_avg_red" in orig_history: orig_history["measured_avg_red"].append(measured_avg_r)
            if "measured_avg_green" in orig_history: orig_history["measured_avg_green"].append(measured_avg_g)
            if "measured_avg_blue" in orig_history: orig_history["measured_avg_blue"].append(measured_avg_b)

        # --- Process 'original' and 'adjustable' streams ---
        for name in PROCESSORS:
            img = base_image.copy()

            with lock:
                params = current_data[name]["params"].copy()

            psnr_val = None
            ssim_val = None
            pred_label = "N/A" # Initialize prediction label
            pred_prob = 0.0 # Initialize prediction probability

            if name == "adjustable":
                adjuster = ImageParameterAdjuster(img)
                adjuster.adjust_brightness(params.get("brightness", 1.0))
                adjuster.adjust_gaussian_blur(params.get("gaussian_blur", 1))
                adjuster.adjust_color_balance(
                    params.get("red_gain", 1.0),
                    params.get("green_gain", 1.0),
                    params.get("blue_gain", 1.0),
                )
                adjuster.adjust_contrast(params.get("contrast_factor", 1.0))
                img = adjuster.current_image

                # Calculate metrics vs original...
                try:
                    img_uint8 = img.astype(np.uint8)
                    base_image_uint8 = base_image.astype(np.uint8)
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero encountered in scalar divide')
                        psnr_calc = calculate_psnr(img_uint8, base_image_uint8)
                    psnr_val = float(psnr_calc) if np.isfinite(psnr_calc) else None
                    ssim_val = calculate_ssim(img_uint8, base_image_uint8, channel_axis=-1)
                except Exception as e:
                    print(f"Error calculating metrics for {name}: {e}")

            elif name == "original":
                 # For the original image, set fixed metrics
                ssim_val = 1.0
                psnr_val = None # PSNR is infinite, represented as None

            # --- Classification --- (Runs for both original and adjustable)
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            run_classification = (current_time - last_classification_time) >= CLASSIFICATION_INTERVAL

            if run_classification:
                pred = classify_image(pil_img)
                pred_label = pred['predictions'][0]['label']
                pred_prob = pred['predictions'][0]['probability']
                if name == list(PROCESSORS.keys())[-1]: # If last processor (adjustable)
                     last_classification_time = current_time
            else:
                # Reuse last prediction/probability
                with lock:
                    last_pred_text = current_data[name]["prediction"]
                    last_prob = current_data[name]["latest_metrics"]["probability"]
                    # (Parsing logic as before)
                    try:
                        if last_pred_text != "N/A" and last_pred_text != "Processing..." and last_prob is not None:
                             pred_label = last_pred_text.split('(')[0].strip()
                             pred_prob = last_prob
                        elif '(' in last_pred_text and '%)' in last_pred_text:
                            parts = last_pred_text.split('(')
                            pred_label = parts[0].strip()
                            prob_part = parts[1].split('%')[0]
                            pred_prob = float(prob_part) / 100.0
                        else:
                            pred_label = last_pred_text
                            pred_prob = 0.0
                    except Exception as e:
                        print(f"Error reusing prediction text '{last_pred_text}': {e}")
                        pred_label = last_pred_text
                        pred_prob = 0.0

            # --- Encode and Store Data --- 
            _, jpeg = cv2.imencode(".jpg", img)

            with lock:
                # Store frame and prediction text
                current_data[name]["frame"] = jpeg.tobytes()
                current_prediction_text = f"{pred_label} ({pred_prob*100:.1f}%)" if pred_label not in ["N/A", "Processing..."] else pred_label
                current_data[name]["prediction"] = current_prediction_text
                
                # Update latest metrics (calculated/fixed ones for this stream)
                latest = current_data[name]["latest_metrics"]
                # Only update timestamp if not original (original timestamp updated earlier)
                if name != "original":
                    latest["timestamp"] = current_time
                latest["probability"] = pred_prob
                latest["ssim"] = ssim_val
                latest["psnr"] = psnr_val
                # Note: Measured values for 'original' were updated outside this loop

                # Record history (for charts) - Common metrics and timestamp
                history = current_data[name]["history"]
                history["timestamps"].append(current_time)
                history["probability"].append(pred_prob)
                history["psnr"].append(psnr_val)
                history["ssim"].append(ssim_val)
                # Note: Measured value history for 'original' was updated earlier

                # Record parameter history only for adjustable
                if name == "adjustable":
                    for param_key, param_value in params.items():
                         if param_key in history:
                            history[param_key].append(param_value)

        # --- Loop Timing --- 
        elapsed = time.time() - start_time
        time.sleep(max(REFRESH_RATE - elapsed, 0))


@app.route('/latest_metrics/<name>')
def latest_metrics(name):
    if name not in current_data:
        return jsonify({"error": "Invalid processor name"}), 404

    with lock:
        metrics = current_data[name]["latest_metrics"].copy()

    # --- Formatting --- 
    # Format calculated metrics
    if metrics.get("psnr") is not None:
        metrics["psnr"] = "inf" if np.isinf(metrics["psnr"]) else f"{metrics['psnr']:.2f}"
    if metrics.get("ssim") is not None:
        metrics["ssim"] = f"{metrics['ssim']:.3f}"
    if metrics.get("probability") is not None:
        metrics["probability"] = f"{metrics['probability']*100:.1f}%"

    # Format measured metrics (if they exist for this name, i.e., 'original')
    if metrics.get("measured_brightness") is not None:
        metrics["measured_brightness"] = f"{metrics['measured_brightness']:.1f}"
    if metrics.get("measured_contrast") is not None:
        metrics["measured_contrast"] = f"{metrics['measured_contrast']:.1f}"
    if metrics.get("measured_sharpness") is not None: # Renamed from measured_blur
        metrics["measured_sharpness"] = f"{metrics['measured_sharpness']:.1f}"
    if metrics.get("measured_avg_red") is not None:
        metrics["measured_avg_red"] = f"{metrics['measured_avg_red']:.1f}"
    if metrics.get("measured_avg_green") is not None:
        metrics["measured_avg_green"] = f"{metrics['measured_avg_green']:.1f}"
    if metrics.get("measured_avg_blue") is not None:
        metrics["measured_avg_blue"] = f"{metrics['measured_avg_blue']:.1f}"

    return jsonify(metrics)


@app.route('/metrics_history/<name>')
def metrics_history(name):
    if name not in current_data:
        return jsonify({"error": "Invalid processor name"}), 404

    # Define potential keys to fetch
    metric_keys = ["probability", "psnr", "ssim"]
    parameter_keys = []
    measured_keys = []

    if name == "adjustable":
        parameter_keys = list(PROCESSORS["adjustable"]["params"].keys())
    elif name == "original":
        # Define keys for measured history to fetch for original
        measured_keys = [
            "measured_brightness", "measured_contrast", "measured_sharpness", # Renamed from measured_blur
            "measured_avg_red", "measured_avg_green", "measured_avg_blue"
        ]

    with lock:
        history = current_data[name]["history"]
        timestamps = list(history.get("timestamps", deque()))

        response_data = {"timestamps": timestamps}
        # Combine metrics + params OR metrics + measured based on name
        all_keys_to_fetch = metric_keys + (parameter_keys if name == "adjustable" else measured_keys)

        for key in all_keys_to_fetch:
            # Get data if key exists in the history for this processor
            response_data[key] = list(history.get(key, deque()))

    # Ensure all returned lists have the same length, matching timestamps
    min_len = len(response_data["timestamps"])
    final_response = {"timestamps": response_data["timestamps"][:min_len]}
    for key in all_keys_to_fetch:
        # Use .get(key, []) to handle cases where the key might not exist
        final_response[key] = response_data.get(key, [])[:min_len]

    return jsonify(final_response)


if __name__ == "__main__":
    # Add measured history deques during initialization
    for name, data_entry in current_data.items():
        if name == "original":
             data_entry["history"]["measured_brightness"] = deque(maxlen=HISTORY_LENGTH)
             data_entry["history"]["measured_contrast"] = deque(maxlen=HISTORY_LENGTH)
             data_entry["history"]["measured_sharpness"] = deque(maxlen=HISTORY_LENGTH) # Renamed from measured_blur
             data_entry["history"]["measured_avg_red"] = deque(maxlen=HISTORY_LENGTH)
             data_entry["history"]["measured_avg_green"] = deque(maxlen=HISTORY_LENGTH)
             data_entry["history"]["measured_avg_blue"] = deque(maxlen=HISTORY_LENGTH)

    threading.Thread(target=image_processor, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False)
