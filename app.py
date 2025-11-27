from flask import Flask, render_template, request, send_file, jsonify
import os
import cv2
import numpy as np
import json

from preprocess import preprocess_pipeline
from masking import build_mask_hsv, morphology_cleanup, remove_small_components
from components_utils import find_components
from export_utils import export_transparent_png, export_components_csv, write_thresholds_json

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
PREVIEW_FOLDER = "static"  # we'll save preview.png into static/

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(PREVIEW_FOLDER, exist_ok=True)


@app.route("/preview", methods=["POST"])
def preview():
    try:
        image_file = request.files.get("image")
        if not image_file:
            return jsonify({"error": "No image provided"}), 400

        # HSV + morphology inputs
        h_min = int(request.form.get("h_min", 0))
        h_max = int(request.form.get("h_max", 179))
        s_min = int(request.form.get("s_min", 40))
        s_max = int(request.form.get("s_max", 255))
        v_min = int(request.form.get("v_min", 60))
        v_max = int(request.form.get("v_max", 255))
        open_k = int(request.form.get("open_k", 3))
        close_k = int(request.form.get("close_k", 7))
        area_min = int(request.form.get("area_min", 150))
        clahe = "clahe" in request.form

        # Decode image
        file_bytes = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Failed to read image"}), 400

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess
        rgb_proc, hsv_proc = preprocess_pipeline(
            image,
            use_grayworld=False,
            use_bilateral_filter=False,
            use_clahe_v_enhancement=clahe,
            bilateral_d=9,
            bilateral_sigma_color=75.0,
            bilateral_sigma_space=75.0,
            clahe_clip_limit=2.0,
            scale=1.0,
        )

        # Build mask from HSV thresholds
        lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
        mask = build_mask_hsv(hsv_proc, lower, upper)
        mask = morphology_cleanup(mask, 0, open_k, close_k, True, True)
        mask = remove_small_components(mask, area_min)

        # ---- NEW: grayscale background, color only inside mask ----
        gray = cv2.cvtColor(rgb_proc, cv2.COLOR_RGB2GRAY)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        mask_3 = (mask > 0)[..., None]  # shape (H, W, 1), bool
        result = np.where(mask_3, rgb_proc, gray_rgb)

        # Save preview
        preview_name = "preview.png"
        preview_path = os.path.join(PREVIEW_FOLDER, preview_name)
        cv2.imwrite(preview_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

        num = len(find_components(mask, area_min))
        return jsonify(
            {
                "preview_url": f"/static/{preview_name}",
                "num_detected": num,
            }
        )

    except Exception:
        import traceback

        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if not file:
            return "No file uploaded", 400

        # Save upload
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Load image
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Form params
        hmin = int(request.form.get("h_min", 0))
        hmax = int(request.form.get("h_max", 179))
        smin = int(request.form.get("s_min", 40))
        smax = int(request.form.get("s_max", 255))
        vmin = int(request.form.get("v_min", 60))
        vmax = int(request.form.get("v_max", 255))
        open_k = int(request.form.get("open_k", 3))
        close_k = int(request.form.get("close_k", 7))
        area_min = int(request.form.get("area_min", 150))
        use_clahe = bool(request.form.get("clahe"))

        # Preprocess
        rgb_proc, hsv_proc = preprocess_pipeline(
            image,
            use_grayworld=False,
            use_bilateral_filter=False,
            use_clahe_v_enhancement=use_clahe,
            bilateral_d=9,
            bilateral_sigma_color=75.0,
            bilateral_sigma_space=75.0,
            clahe_clip_limit=2.0,
            scale=1.0,
        )

        # Mask from HSV
        lower = np.array([hmin, smin, vmin], dtype=np.uint8)
        upper = np.array([hmax, smax, vmax], dtype=np.uint8)
        raw_mask = build_mask_hsv(hsv_proc, lower, upper)
        mask_clean = morphology_cleanup(raw_mask, 0, open_k, close_k, True, True)
        final_mask = remove_small_components(mask_clean, area_min)

                # ---- grayscale background, color only inside mask ----
        gray = cv2.cvtColor(rgb_proc, cv2.COLOR_RGB2GRAY)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        mask_3 = (final_mask > 0)[..., None]
        result = np.where(mask_3, rgb_proc, gray_rgb)

        # Save result
        output_path = os.path.join(PROCESSED_FOLDER, "result.png")
        cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

        # ---- NEW: compute hold centers for selection canvas ----
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            final_mask, connectivity=8
        )
        holds = []
        for label_id in range(1, num_labels):  # skip background 0
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            cx, cy = centroids[label_id]
            holds.append(
                {
                    "id": label_id - 1,
                    "cx": float(cx),
                    "cy": float(cy),
                    "area": area,
                }
            )

        num_detected = len(holds)
        components_json = json.dumps(holds)

        return render_template(
            "index.html",
            result_image="result.png",
            num_detected=num_detected,
            components_json=components_json,   # NEW
        )

    # return render_template("index.html", result_image=None)
    return render_template(
        "index.html",
        result_image=None,
        num_detected=0,
        components_json="[]",   # NEW
    )


@app.route("/processed/<filename>")
def processed_file(filename):
    return send_file(os.path.join(PROCESSED_FOLDER, filename), mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True)