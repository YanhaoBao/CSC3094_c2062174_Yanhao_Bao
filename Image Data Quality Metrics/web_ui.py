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

PROCESSORS = {
    "original": {"params": {}, "controls": []},
    "low_light": {
        "params": {"brightness": 1.0},
        "controls": [("brightness", 0.0, 4.0, 1.0, 0.1)],
    },
    "blurry": {
        "params": {"gaussian_blur": 1},
        "controls": [("blur_size", 1, 31, 1, 2)],
    },
    "color_balance": {
        "params": {"red_gain": 1.0, "green_gain": 1.0, "blue_gain": 1.0},
        "controls": [
            ("red_gain", 0.0, 4.0, 1.0, 0.1),
            ("green_gain", 0.0, 4.0, 1.0, 0.1),
            ("blue_gain", 0.0, 4.0, 1.0, 0.1),
        ],
    },
    "contrast": {
        "params": {"contrast_factor": 1.0},
        "controls": [("contrast_factor", 0.5, 3.0, 1.0, 0.1)],
    },
}

# Initialize current_data structure dynamically
current_data = {}
for name, config in PROCESSORS.items():
    data_entry = {
        "frame": None,
        "prediction": "N/A",
        "params": config["params"].copy(),
        "history": { # Changed from 'metrics' to 'history'
            "timestamps": deque(maxlen=HISTORY_LENGTH),
            "probability": deque(maxlen=HISTORY_LENGTH),
            "psnr": deque(maxlen=HISTORY_LENGTH), # Keep for potential future use
            "ssim": deque(maxlen=HISTORY_LENGTH), # Keep for potential future use
        },
    }
    # Add specific parameter history deques based on the processor type
    if name == "low_light":
        data_entry["history"]["brightness"] = deque(maxlen=HISTORY_LENGTH)
    elif name == "blurry":
        # Use internal param name 'gaussian_blur' for history key
        data_entry["history"]["gaussian_blur"] = deque(maxlen=HISTORY_LENGTH)
    elif name == "color_balance":
        data_entry["history"]["red_gain"] = deque(maxlen=HISTORY_LENGTH)
        data_entry["history"]["green_gain"] = deque(maxlen=HISTORY_LENGTH)
        data_entry["history"]["blue_gain"] = deque(maxlen=HISTORY_LENGTH)
    elif name == "contrast":
        data_entry["history"]["contrast_factor"] = deque(maxlen=HISTORY_LENGTH)
    current_data[name] = data_entry


@app.route("/")
def index():
    return render_template("webCamera.html", processors=PROCESSORS)


@app.route("/video/<name>")
def video_feed(name):
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
    param = data["param"]
    value = float(data["value"])

    with lock:
        if name == "low_light" and param == "brightness":
            current_data[name]["params"]["brightness"] = max(0.0, min(4.0, value))
        elif name == "blurry" and param == "blur_size":
            value = int(value)
            value = value if value % 2 == 1 else value + 1
            current_data[name]["params"]["gaussian_blur"] = max(1, min(31, value))
        elif name == "color_balance":
            current_data[name]["params"][param] = max(0.0, min(4.0, value))
        elif name == "contrast" and param == "contrast_factor":
            current_data[name]["params"]["contrast_factor"] = max(0.5, min(3.0, value))

    return jsonify(success=True)


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
        for name in PROCESSORS:
            img = base_image.copy()
            with lock:
                params = current_data[name]["params"].copy()

            psnr_val = None
            ssim_val = None

            if name != "original":
                adjuster = ImageParameterAdjuster(img)
                if name == "low_light":
                    adjuster.adjust_brightness(params.get("brightness", 1.0))
                elif name == "blurry":
                    adjuster.adjust_gaussian_blur(params.get("gaussian_blur", 1))
                elif name == "color_balance":
                    adjuster.adjust_color_balance(
                        params.get("red_gain", 1.0),
                        params.get("green_gain", 1.0),
                        params.get("blue_gain", 1.0),
                    )
                elif name == "contrast":
                    adjuster.adjust_contrast(params.get("contrast_factor", 1.0))
                img = adjuster.current_image

                try:
                    img_uint8 = img.astype(np.uint8)
                    base_image_uint8 = base_image.astype(np.uint8)
                    
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero encountered in scalar divide')
                        psnr_calc = calculate_psnr(img_uint8, base_image_uint8)
                    
                    psnr_val = float(psnr_calc) if np.isfinite(psnr_calc) else None

                    ssim_val = calculate_ssim(img_uint8, base_image_uint8, channel_axis=-1)

                except ValueError as e:
                    print(f"Error calculating metrics for {name}: {e}")
                except Exception as e:
                    print(f"Unexpected error calculating metrics for {name}: {e}")
            
            if name == "original":
                ssim_val = 1.0
                psnr_val = None

            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            current_time = time.time()
            run_classification = (current_time - last_classification_time) >= CLASSIFICATION_INTERVAL
            pred_label = "Processing..."
            pred_prob = 0.0

            if run_classification:
                pred = classify_image(pil_img)
                pred_label = pred['predictions'][0]['label']
                pred_prob = pred['predictions'][0]['probability']
                if name == list(PROCESSORS.keys())[-1]:
                    last_classification_time = current_time
            else:
                with lock:
                    last_pred_text = current_data[name]["prediction"]
                    try:
                        parts = last_pred_text.split('(')
                        pred_label = parts[0].strip()
                        pred_prob = float(parts[1].replace('%) ', '').replace('%)\'', '')) / 100.0
                    except:
                        pred_label = last_pred_text
                        pred_prob = 0.0

            _, jpeg = cv2.imencode(".jpg", img)

            with lock:
                current_data[name]["frame"] = jpeg.tobytes()
                current_data[name]["prediction"] = (
                    f"{pred_label} ({pred_prob*100:.1f}%)"
                )
                # Record metrics and parameters in history
                history = current_data[name]["history"]
                history["timestamps"].append(current_time)
                history["probability"].append(pred_prob)
                history["psnr"].append(psnr_val)
                history["ssim"].append(ssim_val)

                # Record parameters based on processor type
                if name == "low_light":
                    history["brightness"].append(params.get("brightness", 1.0))
                elif name == "blurry":
                    history["gaussian_blur"].append(params.get("gaussian_blur", 1))
                elif name == "color_balance":
                    history["red_gain"].append(params.get("red_gain", 1.0))
                    history["green_gain"].append(params.get("green_gain", 1.0))
                    history["blue_gain"].append(params.get("blue_gain", 1.0))
                elif name == "contrast":
                    history["contrast_factor"].append(params.get("contrast_factor", 1.0))

        elapsed = time.time() - start_time
        time.sleep(max(REFRESH_RATE - elapsed, 0))


@app.route('/metrics_history/<name>')
def metrics_history(name):
    if name not in current_data:
        return jsonify({"error": "Invalid processor name"}), 404

    # Define potential keys to fetch
    metric_keys = ["probability", "psnr", "ssim"]
    parameter_keys = []
    if name == "low_light": parameter_keys = ["brightness"]
    elif name == "blurry": parameter_keys = ["gaussian_blur"]
    elif name == "color_balance": parameter_keys = ["red_gain", "green_gain", "blue_gain"]
    elif name == "contrast": parameter_keys = ["contrast_factor"]

    with lock:
        history = current_data[name]["history"]
        timestamps = list(history.get("timestamps", deque()))

        response_data = {"timestamps": timestamps}
        all_keys_to_fetch = metric_keys + parameter_keys

        for key in all_keys_to_fetch:
            # Get data if key exists for this processor, otherwise empty list
            response_data[key] = list(history.get(key, deque()))

    # Ensure all returned lists have the same length, matching timestamps
    min_len = len(response_data["timestamps"])
    final_response = {"timestamps": response_data["timestamps"][:min_len]}
    for key in all_keys_to_fetch:
        # Use .get(key, []) to handle cases where the key might not have been added (e.g., params for 'original')
        final_response[key] = response_data.get(key, [])[:min_len]

    return jsonify(final_response)


if __name__ == "__main__":
    threading.Thread(target=image_processor, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False)
