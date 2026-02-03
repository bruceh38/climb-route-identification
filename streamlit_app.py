import streamlit as st
import cv2
import numpy as np
import os
import io
import csv
import json

from preprocess import preprocess_pipeline
from masking import build_mask_hsv, morphology_cleanup, remove_small_components
from components_utils import find_components
from color_ops import kmeans_hsv, bounds_from_center

st.set_page_config(page_title="Climbing Route Detection", layout="wide")
st.title("Climbing Route Identifier")

uploaded_file = st.file_uploader("Upload a climbing wall image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img_width = st.slider("Image display width (px)", 100, 1600, 800)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    st.image(image, caption="Original Image", width=img_width)


    st.subheader("⚙️ Preprocessing Options")
    _, _, c3 = st.columns(3)
    with c3:
        use_clahe = st.checkbox("Apply contrast enhancement (CLAHE)", value=True)

    colp1, colp2 = st.columns(2)
    with colp1:
        bilat_d = st.slider("Bilateral filter kernel diameter", 3, 15, 9, step=2)
        bilat_sc = st.slider("Color influence for bilateral filter", 20, 150, 75)
    with colp2:
        bilat_ss = st.slider("Distance influence for bilateral filter", 20, 150, 75)
        clahe_clip = st.slider("CLAHE clip limit (higher = more contrast)", 1.0, 5.0, 2.0)

    proc_rgb, proc_hsv = preprocess_pipeline(
        image,
        use_grayworld=False,
        use_bilateral_filter=False,
        use_clahe_v_enhancement=use_clahe,
        bilateral_d=bilat_d,
        bilateral_sigma_color=float(bilat_sc),
        bilateral_sigma_space=float(bilat_ss),
        clahe_clip_limit=float(clahe_clip),
        scale=1.0,
    )

    st.subheader("🎨 Color Selection (HSV)")
    method = st.radio("Color Selection Method", ["Manual", "KMeans (Suggest)"])
    if method == "Manual":
        col1, col2 = st.columns(2)
        with col1:
            h_min = st.slider("Hue minimum (e.g., red = 0)", 0, 179, 0)
            s_min = st.slider("Saturation minimum (how pure the color is)", 0, 255, 40)
            v_min = st.slider("Brightness minimum (darker colors lower)", 0, 255, 60)
        with col2:
            h_max = st.slider("Hue maximum", 0, 179, 179)
            s_max = st.slider("Saturation maximum", 0, 255, 255)
            v_max = st.slider("Brightness maximum", 0, 255, 255)
        lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
    else:
        k = st.slider("Number of dominant color suggestions (KMeans K)", 5, 8, 6)
        centers, _labels, counts = kmeans_hsv(proc_hsv, k=k)
        st.caption("Click a cluster to use its center ± margins")
        cols = st.columns(min(k, 6))
        chosen = st.session_state.get("_km_choice", 0)
        for i in range(k):
            center = centers[i]
            pct = (counts[i] / max(1, counts.sum())) * 100.0
            rgb = cv2.cvtColor(np.uint8([[center]]), cv2.COLOR_HSV2RGB)[0][0]
            hex_color = '#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2]))
            if cols[i % len(cols)].button(f"Use C{i} ({pct:.1f}%)\n{hex_color}"):
                st.session_state["_km_choice"] = i
                chosen = i
        lower, upper = bounds_from_center(centers[chosen])

    st.subheader("🧼 Mask Cleanup Options")
    _, m2, m3 = st.columns(3)
    with m2:
        do_open = st.checkbox("Apply morphological opening (remove noise)", value=True)
        open_k = st.slider("Opening kernel size", 1, 15, 3, step=2)
    with m3:
        do_close = st.checkbox("Apply morphological closing (fill gaps)", value=True)
        close_k = st.slider("Closing kernel size", 1, 21, 7, step=2)
    area_min = st.slider("Minimum pixel area to keep (removes tiny spots)", 0, 5000, 150, step=10)

    raw_mask = build_mask_hsv(proc_hsv, lower, upper)
    cleaned = morphology_cleanup(raw_mask, 0, open_k, close_k, do_open, do_close)
    final_mask = remove_small_components(cleaned, area_min)

    result = cv2.bitwise_and(proc_rgb, proc_rgb, mask=final_mask)

    st.subheader("🖼️ Final Result")
    st.image(result, caption="Masked Route (preprocessed)", width=img_width)

    components = find_components(final_mask, area_min=area_min)
    st.info(f"Detected holds: {len(components)}  |  Total area: {int(sum(c['area'] for c in components))} px²")

    st.subheader("🗕️ Export Options")
    colx, coly, colz = st.columns(3)
    export_base = os.path.splitext(uploaded_file.name)[0]

    # Build in-memory exports (Streamlit Cloud friendly)
    rgba = np.dstack([proc_rgb, (final_mask > 0).astype(np.uint8) * 255])
    _, png_buf = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    png_bytes = png_buf.tobytes()

    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)
    writer.writerow(["area", "centroid_x", "centroid_y", "x", "y", "w", "h", "circularity"])
    for c in components:
        cx, cy = c["centroid"]
        x, y, w, h = c["bbox"]
        writer.writerow([c["area"], cx, cy, x, y, w, h, c["circularity"]])
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    thresholds_json = json.dumps(
        {
            "lower_hsv": lower.tolist() if lower is not None else None,
            "upper_hsv": upper.tolist() if upper is not None else None,
            "num_holds": len(components),
        },
        indent=2,
    ).encode("utf-8")

    colx.download_button(
        "Download Transparent PNG",
        data=png_bytes,
        file_name=f"{export_base}_holds.png",
        mime="image/png",
    )
    coly.download_button(
        "Download Components CSV",
        data=csv_bytes,
        file_name=f"{export_base}_components.csv",
        mime="text/csv",
    )
    colz.download_button(
        "Download HSV Thresholds JSON",
        data=thresholds_json,
        file_name=f"{export_base}_thresholds.json",
        mime="application/json",
    )
