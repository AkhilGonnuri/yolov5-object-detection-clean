# app_streamlit.py
# Streamlit UI for YOLOv5 image upload -> detection -> display
# Author: G Chaitanya Sai (gcsai)
# Place this file at the repo root (same folder as detect.py)

import subprocess
import time
from pathlib import Path

import streamlit as st

# ---------- Config ----------
AUTHOR = "G Chaitanya Sai (gcsai)"
WEIGHTS = "yolov5s.pt"  # pretrained weights (auto-download)
OUTPUT_PROJECT = "runs/streamlit"  # folder where detect.py writes results
OUTPUT_NAME = "output"
USE_CPU = True  # Set False to allow GPU (if you have CUDA configured)
# -----------------------------

st.set_page_config(page_title=f"YOLOv5 Demo — {AUTHOR}", layout="wide")
st.title("YOLOv5 — Upload Image & Run Detection")
st.markdown(
    f"**Author:** {AUTHOR}  \n\n"
    "Upload an image and YOLOv5 will run object detection. Results are saved in "
    f"`{OUTPUT_PROJECT}/{OUTPUT_NAME}/`."
)

# Device selection notice
if USE_CPU:
    st.info(
        "Running inference on **CPU**. If you have a CUDA-enabled GPU, set `USE_CPU = False` in this file for GPU inference."
    )

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    # prepare paths
    tmp_dir = Path(OUTPUT_PROJECT)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    input_path = tmp_dir / "input.jpg"
    out_dir = tmp_dir / OUTPUT_NAME
    out_img = out_dir / "input.jpg"

    # Save uploaded file
    with open(input_path, "wb") as f:
        f.write(uploaded.getbuffer())

    # Build command
    device_flag = "--device cpu" if USE_CPU else ""
    # Ensure paths quoted for Windows
    cmd = (
        f'python detect.py --weights "{WEIGHTS}" --source "{input_path}" '
        f'--project "{tmp_dir}" --name "{OUTPUT_NAME}" --exist-ok {device_flag}'
    )

    st.text("Running YOLOv5 detect (this may take a few seconds)...")
    with st.spinner("Detecting..."):
        try:
            # Run detect.py as subprocess
            completed = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            # Optionally show some output logs in the app (short)
            if completed.stdout:
                st.text("detect.py output (tail):")
                st.code("\n".join(completed.stdout.splitlines()[-10:]))
        except subprocess.CalledProcessError as e:
            st.error("Detection failed. See the terminal for full details.")
            # show limited stderr for quick hint
            if e.stderr:
                st.code("\n".join(e.stderr.splitlines()[-10:]))
        else:
            # give a moment to ensure file system updated
            time.sleep(0.25)

            if out_img.exists():
                # Display input and output side-by-side
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Input Image")
                    st.image(str(input_path), use_column_width=True)
                with col2:
                    st.subheader("YOLOv5 Prediction")
                    st.image(str(out_img), use_column_width=True)

                    # Download button for result image
                    with open(out_img, "rb") as f:
                        st.download_button(
                            label="Download prediction image",
                            data=f,
                            file_name="yolov5_prediction.jpg",
                            mime="image/jpeg",
                        )

                # Show raw labels (YOLO txt) if present
                label_txt = out_dir / "labels" / "input.txt"
                if label_txt.exists():
                    st.subheader("Raw labels (YOLO format)")
                    st.code(label_txt.read_text())
                else:
                    st.info("No YOLO label TXT found (labels may not be saved depending on detect.py options).")
            else:
                st.error("Prediction image not found. Check terminal logs and ensure detect.py ran successfully.")

st.markdown("---")
st.markdown(
    "Notes:  \n"
    "- This demo runs `detect.py` as a subprocess for simplicity.  \n"
    "- For production, consider importing the model and running inference in-memory (faster).  \n"
    "- To allow GPU inference set `USE_CPU = False` above and ensure CUDA + correct torch build are installed."
)
