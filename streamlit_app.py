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

# ---- sensible defaults stored in session_state ----
defaults = {
    "h_min": 0,
    "h_max": 179,
    "s_min": 40,
    "s_max": 255,
    "v_min": 60,
    "v_max": 255,
    "area_min": 150,
    "use_clahe": True,
    "do_open": True,
    "do_close": True,
    "open_k": 3,
    "close_k": 7,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

presets = [
    {
        "name": "Bright holds",
        "hint": "High contrast colors",
        "h_min": 0,
        "h_max": 179,
        "s_min": 40,
        "v_min": 70,
        "area_min": 140,
        "use_clahe": True,
    },
    {
        "name": "Pastel holds",
        "hint": "Softer colors",
        "h_min": 0,
        "h_max": 179,
        "s_min": 20,
        "v_min": 70,
        "area_min": 160,
        "use_clahe": True,
    },
    {
        "name": "Dim lighting",
        "hint": "Low light gyms",
        "h_min": 0,
        "h_max": 179,
        "s_min": 30,
        "v_min": 40,
        "area_min": 140,
        "use_clahe": True,
    },
    {
        "name": "Reset",
        "hint": "Factory defaults",
        **defaults,
    },
]


def apply_preset(preset):
    for k in ["h_min", "h_max", "s_min", "s_max", "v_min", "v_max", "area_min", "use_clahe"]:
        if k in preset:
            st.session_state[k] = preset[k]


st.header("Step 1 · Upload")
uploaded_file = st.file_uploader("Upload a climbing wall image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img_width = st.slider("Preview size", 400, 1200, 800, help="Adjust on-screen preview only.")
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    st.image(image, caption="Original wall", width=img_width)

    st.header("Step 2 · Pick the hold color")

    preset_cols = st.columns(len(presets))
    for col, preset in zip(preset_cols, presets):
        if col.button(preset["name"]):
            apply_preset(preset)
            st.toast(f"Applied preset: {preset['name']} ({preset['hint']})")
    st.caption("Presets set color sensitivity and brightness tweaks automatically.")

    h_min, h_max = st.slider(
        "Color range (Hue)",
        0,
        179,
        (st.session_state["h_min"], st.session_state["h_max"]),
        help="Drag the handles to cover the main color of the holds.",
    )
    st.session_state["h_min"], st.session_state["h_max"] = h_min, h_max

    area_min = st.slider(
        "Ignore tiny spots (px²)",
        50,
        2000,
        st.session_state["area_min"],
        step=10,
        help="Higher = fewer tiny specks kept.",
    )
    st.session_state["area_min"] = area_min

    use_clahe = st.checkbox(
        "Brighten dim walls",
        value=st.session_state["use_clahe"],
        help="Boosts contrast in darker photos.",
    )
    st.session_state["use_clahe"] = use_clahe

    with st.expander("Advanced fine-tuning (optional)", expanded=False):
        st.markdown("**Color detail**")
        s_min = st.slider("Minimum color strength (Saturation)", 0, 255, st.session_state["s_min"])
        v_min = st.slider("Minimum brightness (Value)", 0, 255, st.session_state["v_min"])
        st.session_state["s_min"], st.session_state["v_min"] = s_min, v_min

        st.markdown("**Cleanup**")
        do_open = st.checkbox("Remove specks (opening)", value=st.session_state["do_open"])
        open_k = st.slider("Speck removal strength", 1, 15, st.session_state["open_k"], step=2)
        do_close = st.checkbox("Fill small gaps (closing)", value=st.session_state["do_close"])
        close_k = st.slider("Gap filling strength", 1, 21, st.session_state["close_k"], step=2)
        st.session_state["do_open"], st.session_state["do_close"] = do_open, do_close
        st.session_state["open_k"], st.session_state["close_k"] = open_k, close_k

        st.markdown("**Auto pick color (optional)**")
        if st.button("Suggest a color from the image"):
            centers, _labels, counts = kmeans_hsv(hsv_image, k=6)
            # choose most frequent cluster
            main_idx = int(np.argmax(counts))
            lower_suggest, upper_suggest = bounds_from_center(centers[main_idx])
            st.session_state["h_min"], st.session_state["h_max"] = int(lower_suggest[0]), int(upper_suggest[0])
            st.session_state["s_min"] = int(lower_suggest[1])
            st.session_state["v_min"] = int(lower_suggest[2])
            st.toast("Auto-selected a color range from the image.")

    # build HSV thresholds from state
    lower = np.array(
        [st.session_state["h_min"], st.session_state["s_min"], st.session_state["v_min"]], dtype=np.uint8
    )
    upper = np.array(
        [st.session_state["h_max"], st.session_state.get("s_max", 255), st.session_state.get("v_max", 255)],
        dtype=np.uint8,
    )

    # preprocessing
    proc_rgb, proc_hsv = preprocess_pipeline(
        image,
        use_grayworld=False,
        use_bilateral_filter=False,
        use_clahe_v_enhancement=st.session_state["use_clahe"],
        bilateral_d=9,
        bilateral_sigma_color=75.0,
        bilateral_sigma_space=75.0,
        clahe_clip_limit=2.0,
        scale=1.0,
    )

    raw_mask = build_mask_hsv(proc_hsv, lower, upper)
    cleaned = morphology_cleanup(
        raw_mask,
        0,
        st.session_state["open_k"],
        st.session_state["close_k"],
        st.session_state["do_open"],
        st.session_state["do_close"],
    )
    final_mask = remove_small_components(cleaned, st.session_state["area_min"])

    result = cv2.bitwise_and(proc_rgb, proc_rgb, mask=final_mask)

    st.header("Step 3 · Check result")
    st.image(result, caption="Masked route", width=img_width)

    components = find_components(final_mask, area_min=st.session_state["area_min"])
    coverage = np.count_nonzero(final_mask) / float(final_mask.size)
    if coverage < 0.002:
        st.warning(f"Detected {len(components)} holds. Mask is very small—try a wider color range or a preset.")
    elif coverage < 0.02:
        st.info(f"Detected {len(components)} holds. Looks okay; widen color range if holds are missing.")
    else:
        st.success(f"Detected {len(components)} holds. Coverage looks good.")

    st.header("Step 4 · Download")
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
        "Download photo with holds",
        data=png_bytes,
        file_name=f"{export_base}_holds.png",
        mime="image/png",
    )
    coly.download_button(
        "Download hold list (CSV)",
        data=csv_bytes,
        file_name=f"{export_base}_components.csv",
        mime="text/csv",
    )
    colz.download_button(
        "Download color settings (JSON)",
        data=thresholds_json,
        file_name=f"{export_base}_thresholds.json",
        mime="application/json",
    )
else:
    st.info("Upload a wall photo to begin.")
