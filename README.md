# Climbing Route Identifier (Streamlit)

A Streamlit app that isolates climbing holds in a wall photo using HSV masking and basic morphology.

## Quick start (local)
1. Use Python 3.10 or 3.11 (recommended for Streamlit Cloud; avoid 3.14 for now).
2. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. `pip install --upgrade pip`  # ensures build backends install cleanly
4. `streamlit run streamlit_app.py`
5. Open the URL Streamlit prints (default http://localhost:8501).

## Usage
- Upload a JPG/PNG of the wall.
- Choose preprocessing/HSV ranges; the mask and result update inline.
- Exports (PNG with transparency, components CSV, HSV JSON) download directly—no disk writes.

## Deploy to Streamlit Community Cloud
1. Push this repo to GitHub.
2. In share.streamlit.io, create an app pointing to `streamlit_app.py` on `main`.
3. (Optional) Set Python version to 3.10/3.11; leave secrets empty for now.

## Notes
- Dependencies are pinned for reproducible cloud builds.
- Uses headless OpenCV to avoid GUI issues on Streamlit Cloud.
- No persistent storage is required; everything is kept in-memory per request.
