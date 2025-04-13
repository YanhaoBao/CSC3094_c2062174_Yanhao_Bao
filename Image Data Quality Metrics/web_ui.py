from flask import Flask, render_template, Response, request, jsonify
import cv2
import threading
import time
import numpy as np
from Adjuster import ImageParameterAdjuster
from main import classify_image
from PIL import Image

app = Flask(__name__)
lock = threading.Lock()

IMG_SIZE = (640, 480)
REFRESH_RATE = 0.1  # 秒

# 处理配置
PROCESSORS = {
    "original": {"params": {}, "controls": []},
    "low_light": {
        "params": {"brightness": 1.0},
        "controls": [("brightness", 0.0, 2.0, 1.0, 0.1)],
    },
    "blurry": {
        "params": {"gaussian_blur": 1},
        "controls": [("blur_size", 1, 15, 1, 2)],
    },
    "color_balance": {
        "params": {"red_gain": 1.0, "green_gain": 1.0, "blue_gain": 1.0},
        "controls": [
            ("red_gain", 0.0, 2.0, 1.0, 0.1),
            ("green_gain", 0.0, 2.0, 1.0, 0.1),
            ("blue_gain", 0.0, 2.0, 1.0, 0.1),
        ],
    },
}

# 共享状态
current_data = {
    name: {"frame": None, "prediction": "N/A", "params": config["params"].copy()}
    for name, config in PROCESSORS.items()
}


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
            current_data[name]["params"]["brightness"] = max(0.0, min(2.0, value))
        elif name == "blurry" and param == "blur_size":
            value = int(value)
            value = value if value % 2 == 1 else value + 1
            current_data[name]["params"]["gaussian_blur"] = max(1, min(15, value))
        elif name == "color_balance":
            current_data[name]["params"][param] = max(0.0, min(2.0, value))

    return jsonify(success=True)


def image_processor():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头")

    while True:
        ret, base_image = cap.read()
        if not ret:
            print("摄像头读取失败")
            continue
        base_image = cv2.resize(base_image, IMG_SIZE)
        start_time = time.time()
        for name in PROCESSORS:
            img = base_image.copy()
            with lock:
                params = current_data[name]["params"].copy()

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
                img = adjuster.current_image

            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            pred = classify_image(pil_img)
            _, jpeg = cv2.imencode(".jpg", img)

            with lock:
                current_data[name]["frame"] = jpeg.tobytes()
                current_data[name]["prediction"] = (
                    f"{pred['predictions'][0]['label']} "
                    f"({pred['predictions'][0]['probability']*100:.1f}%)"
                )

        elapsed = time.time() - start_time
        time.sleep(max(REFRESH_RATE - elapsed, 0))


if __name__ == "__main__":
    threading.Thread(target=image_processor, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False)
