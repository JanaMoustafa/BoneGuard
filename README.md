# BoneGuard — Bone Tumour X-ray Report Web Application

A Flask web application that accepts X-ray images, patient age, and gender to generate AI-powered research reports with tumour segmentation and multi-task classification.

## Features

- Upload bone X-ray images (PNG, JPG, JPEG)
- CLAHE preprocessing pipeline for consistent image analysis
- U-Net tumour segmentation with overlay visualization
- Multi-task ConvNeXt classification (tumour type, neoplasm, skeletal region, bone type)
- PDF report generation with clinical formatting
- TinyDB local storage for submission history
- Input validation and error handling

## Requirements

- Python 3.10+
- PyTorch
- timm
- Flask
- TinyDB
- OpenCV
- Pillow
- fpdf2
- numpy
- matplotlib

## Setup

```bash
pip install flask tinydb torch timm opencv-python Pillow fpdf2 numpy matplotlib
```

## Usage

```bash
cd /path/to/GraduationProject
python -m flask --app report_web.app run --debug --port 5000
```

Then open http://127.0.0.1:5000 in your browser.

## Project Structure

```
report_web/
├── __init__.py              # Package init
├── app.py                   # Flask routes, validation, TinyDB
├── report_inference.py      # Model architectures, preprocessing, PDF generation
├── static/
│   └── style.css            # Clinical light theme CSS
└── templates/
    ├── base.html            # Base layout
    ├── upload.html          # Upload form with drag-and-drop
    ├── report.html          # Report results page
    └── error.html           # Error page

website_data/
├── uploads/                 # Uploaded X-ray images (UUID-renamed)
├── reports/                 # Generated PDF reports
├── overlays/                # Generated segmentation overlays
└── db/
    └── reports.json         # TinyDB database
```

## Trained Model Checkpoints

The app requires pre-trained checkpoints in `outputs/report_system/`:
- `best_REPORT_TumourSeg_UNet.pt` — U-Net segmentation model
- `best_REPORT_MultiTask_ConvNeXtTiny.pt` — Multi-task ConvNeXt classifier

## Disclaimer

This is a **research prototype** developed as a graduation project. It is not validated for clinical use and must not be used for medical diagnosis or treatment decisions.
