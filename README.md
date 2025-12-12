<div align="center">
  <a href="https://www.ultralytics.com" target="_blank">
    <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png" alt="Ultralytics YOLO Banner">
  </a>
</div>

<p align="center">
  <a href="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml">
    <img src="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml/badge.svg" alt="YOLOv5 CI Testing">
  </a>
  <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab">
  </a>
</p>

# 🚀 YOLOv5 — Object Detection (Customized by **G Chaitanya Sai**)

<div align="center">
  <img src="main_page.jpg" alt="Project Main Page" width="100%">
</div>

---

This repository is **my customized version of YOLOv5**, forked from the official Ultralytics project.  
It is fully configured to run **object detection on images, videos, webcam input**, and more.

I also added an optional **Streamlit Web UI**, allowing users to upload images and instantly view detections.

---

# 📸 Example Detection Results

<div align="center">

### **Input Image**

<img src="data/images/zidane.jpg" alt="Input" width="45%">

### **Prediction Output**

<img src="runs/detect/exp/zidane.jpg" alt="Prediction" width="45%">

</div>

Run this to generate the above output:

```bash
python detect.py --weights yolov5s.pt --source data/images/zidane.jpg
What This Project Does

Performs real-time object detection using YOLOv5

Supports images, videos, webcam, URLs, YouTube links, folders

Automatically downloads pretrained weights

Fully ready for transfer learning

Includes an easy Streamlit web interface

Models Used
Model	Size	Speed	Accuracy
yolov5s.pt	Small	Fast	Good
yolov5m.pt	Medium	Moderate	Better
yolov5l.pt	Large	Slower	High
yolov5x.pt	XL	Slowest	Highest Accuracy

💡 Default in this repo = yolov5s.pt (best for demos)

Why This Project Is Useful

Excellent for Resumes, GitHub Portfolio, Interviews

Shows practical knowledge of:

Deep Learning

Computer Vision

PyTorch

Deployment (Streamlit Web App)

Contains:

Pretrained weights

Real dataset support

Complete training pipeline

Clean project structure

This is a full end-to-end AI project, not just a script.

⭐ Features

🚀 Real-time object detection

📸 Web UI for uploading and detecting

🎯 80-class COCO pretrained model

⚙️ Supports transfer learning on custom datasets

📂 Clear and professional project structure

💾 Auto-saving of prediction results

🔥 Completely ready for deployment

Project Structure
yolov5/
│── app_streamlit.py        # Streamlit interface
│── detect.py               # Inference script
│── train.py                # Training script
│── main_page.jpg           # Banner image
│── data/
│   ├── images/
│   │   ├── zidane.jpg
│   │   ├── bus.jpg
│   ├── coco128.yaml
│── models/                 # YOLO models
│── utils/                  # Utility functions
│── runs/                   # Outputs (ignored by Git)
│── requirements.txt
│── README.md
Setup Instructions (Windows PowerShell)
1️⃣ Create & activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

2️⃣ Upgrade pip & install dependencies
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install ultralytics streamlit opencv-python

▶️ Running Inference
Image
python detect.py --weights yolov5s.pt --source data/images/zidane.jpg


Output saved to:

runs/detect/exp/

Webcam
python detect.py --weights yolov5s.pt --source 0

Optional: Run Streamlit Web App

Upload images → Get detections instantly.

streamlit run app_streamlit.py


App opens at:

🔗 http://localhost:8501

🏋️ Fine-Tuning on a Custom Dataset (Transfer Learning)

Create a YAML file:

train: path/to/train/images
val: path/to/val/images
names: ["class1", "class2"]


Train:

python train.py --img 640 --batch 16 --epochs 50 --data custom.yaml --weights yolov5s.pt


YOLOv5 automatically handles:

Pretrained weight loading

Fine-tuning

Logging metrics

Saving best models

📁 Recommended .gitignore
.vscode/
.venv/
__pycache__/
runs/
*.pt
*.onnx
*.engine

🎉 Author

G Chaitanya Sai
Customized and enhanced YOLOv5 for real-world usage, deployment, and portfolio-ready presentation.

```
