"""
report_web.report_inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Standalone inference module extracted from notebook Section 6.
Loads pre-trained checkpoints, preprocesses uploaded X-ray images,
runs segmentation + multi-task classification, and generates PDF reports.

No notebook execution required — all model architectures and helpers
are self-contained in this file.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image as PILImage
from fpdf import FPDF
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────
#  Paths & directories
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "report_system"
WEBSITE_DATA = PROJECT_ROOT / "website_data"
UPLOAD_DIR = WEBSITE_DATA / "uploads"
REPORT_DIR = WEBSITE_DATA / "reports"
OVERLAY_DIR = WEBSITE_DATA / "overlays"

for _d in (UPLOAD_DIR, REPORT_DIR, OVERLAY_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
REPORT_CONFIDENCE_THRESHOLD = 0.55
REPORT_SEG_MASK_THRESHOLD_DEFAULT = 0.50
REPORT_SEG_MIN_MASK_RATIO = 0.001
REPORT_GENDER_TO_ID = {"f": 0.0, "female": 0.0, "m": 1.0, "male": 1.0}

# ImageNet normalization (same as notebook)
REPORT_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
REPORT_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────────────
#  Image preprocessing (exact copy from notebook cell 30)
# ──────────────────────────────────────────────────────────────────────

def _load_and_resize(img_path: str | Path, size: tuple[int, int] = IMG_SIZE):
    """Load with PIL (handles JPEG/PNG/BMP/TIFF) and resize to target size."""
    try:
        pil_img = PILImage.open(img_path).convert("RGB")
        img = np.array(pil_img, dtype=np.uint8)
    except Exception:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
    return img.astype(np.uint8)


def _apply_clahe_and_normalize(img_uint8: np.ndarray):
    """
    Steps (all in uint8 space to avoid float artefacts):
    1. Per-image min-max stretch
    2. Convert to grayscale
    3. Apply CLAHE (enhances local contrast without colour-space artefacts)
    4. Stack to 3 channels and normalise to [0, 1]
    """
    lo, hi = int(img_uint8.min()), int(img_uint8.max())
    if hi - lo < 5:
        return None
    img_stretched = np.clip(
        (img_uint8.astype(np.float32) - lo) / (hi - lo) * 255.0, 0, 255
    ).astype(np.uint8)
    gray = cv2.cvtColor(img_stretched, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    img_out = np.stack([gray_eq, gray_eq, gray_eq], axis=-1)
    return img_out.astype(np.float32) / 255.0


def full_preprocessing_pipeline(img_path: str | Path) -> np.ndarray | None:
    """
    Robust pipeline:
    PIL load -> resize -> contrast stretch -> CLAHE (grayscale) -> [0,1] float32
    Returns None for unreadable or flat (blank) images.
    """
    img = _load_and_resize(str(img_path))
    if img is None:
        return None
    return _apply_clahe_and_normalize(img)


def preprocess_external_image(image_path: str | Path) -> np.ndarray:
    """Preprocess an uploaded X-ray image. Uses the full CLAHE pipeline
    with a simple-resize fallback."""
    img = full_preprocessing_pipeline(str(image_path))
    if img is not None:
        return img.astype(np.float32)
    # Fallback: simple PIL resize + normalize
    pil_img = PILImage.open(image_path).convert("RGB").resize(IMG_SIZE)
    return (np.asarray(pil_img, dtype=np.float32) / 255.0).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
#  Model architectures (exact copies from notebook)
# ──────────────────────────────────────────────────────────────────────

class _ReportDoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ReportTinyUNet(nn.Module):
    """Lightweight U-Net for tumour segmentation."""

    def __init__(self, in_channels: int = 3, base: int = 24):
        super().__init__()
        self.enc1 = _ReportDoubleConv(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = _ReportDoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = _ReportDoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = _ReportDoubleConv(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _ReportDoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _ReportDoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _ReportDoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


class ReportMultiTaskConvNeXt(nn.Module):
    """Multi-task ConvNeXt classifier for tumour type, neoplasm,
    skeletal type, and bone type."""

    def __init__(self, model_name: str, pretrained: bool = False,
                 num_tumor: int = 10, num_neoplasm: int = 3,
                 num_skeletal: int = 3, num_bones: int = 13):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        feat_dim = int(getattr(self.backbone, "num_features"))
        self.meta_branch = nn.Sequential(
            nn.Linear(2, 32), nn.BatchNorm1d(32), nn.GELU(),
            nn.Dropout(0.20), nn.Linear(32, 16), nn.GELU(),
        )
        fused_dim = feat_dim + 16
        self.shared = nn.Sequential(
            nn.Dropout(0.35), nn.Linear(fused_dim, 384),
            nn.GELU(), nn.Dropout(0.35),
        )
        self.tumor_head = nn.Linear(384, num_tumor)
        self.neoplasm_head = nn.Linear(384, num_neoplasm)
        self.skeletal_head = nn.Linear(384, num_skeletal)
        self.bones_head = nn.Linear(384, num_bones)

    def forward(self, image, meta):
        img_feat = self.backbone(image)
        if img_feat.ndim > 2:
            img_feat = img_feat.mean(dim=tuple(range(2, img_feat.ndim)))
        meta_feat = self.meta_branch(meta)
        shared = self.shared(torch.cat([img_feat, meta_feat], dim=1))
        return {
            "tumor": self.tumor_head(shared),
            "neoplasm": self.neoplasm_head(shared),
            "skeletal": self.skeletal_head(shared),
            "bones": self.bones_head(shared),
        }


# ──────────────────────────────────────────────────────────────────────
#  Model loading (singleton — loaded once and cached)
# ──────────────────────────────────────────────────────────────────────
_seg_model: ReportTinyUNet | None = None
_cls_model: ReportMultiTaskConvNeXt | None = None
_cls_meta: dict | None = None  # checkpoint metadata (class names, age stats)


def _load_seg_model() -> ReportTinyUNet:
    global _seg_model
    if _seg_model is not None:
        return _seg_model
    ckpt_path = CHECKPOINT_DIR / "best_REPORT_TumourSeg_UNet.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing segmentation checkpoint: {ckpt_path}")
    model = ReportTinyUNet().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    _seg_model = model
    return _seg_model


def _load_cls_model() -> tuple[ReportMultiTaskConvNeXt, dict]:
    global _cls_model, _cls_meta
    if _cls_model is not None and _cls_meta is not None:
        return _cls_model, _cls_meta
    ckpt_path = CHECKPOINT_DIR / "best_REPORT_MultiTask_ConvNeXtTiny.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing classifier checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    class_names = ckpt["class_names"]
    model = ReportMultiTaskConvNeXt(
        model_name=ckpt.get("model_name", "convnext_tiny"),
        pretrained=False,
        num_tumor=len(class_names["tumor"]),
        num_neoplasm=len(class_names["neoplasm"]),
        num_skeletal=len(class_names["skeletal"]),
        num_bones=len(class_names["bones"]),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    _cls_model = model
    _cls_meta = {
        "class_names": class_names,
        "age_mean": float(ckpt.get("age_mean", 35.45)),
        "age_std": float(ckpt.get("age_std", 20.94)),
    }
    return _cls_model, _cls_meta


# ──────────────────────────────────────────────────────────────────────
#  Inference helpers
# ──────────────────────────────────────────────────────────────────────

def _encode_user_meta(age: float, gender: str, age_mean: float, age_std: float) -> np.ndarray:
    age_z = (float(age) - age_mean) / max(age_std, 1e-6)
    gender_val = REPORT_GENDER_TO_ID.get(str(gender).strip().lower(), 0.5)
    return np.asarray([[age_z, gender_val]], dtype=np.float32)


@torch.no_grad()
def _predict_classifier(image_arr: np.ndarray, meta_arr: np.ndarray) -> dict[str, np.ndarray]:
    model, meta = _load_cls_model()
    img_t = torch.from_numpy(image_arr.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    img_t = (img_t - REPORT_MEAN) / REPORT_STD
    img_t = img_t.to(DEVICE).float()
    meta_t = torch.from_numpy(meta_arr.astype(np.float32)).to(DEVICE).float()
    out = model(img_t, meta_t)
    probs = {
        task: torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        for task, logits in out.items()
    }
    return probs


@torch.no_grad()
def _predict_segmentation(image_arr: np.ndarray):
    model = _load_seg_model()
    img_t = (
        torch.from_numpy(image_arr.astype(np.float32))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(DEVICE)
        .float()
    )
    prob = torch.sigmoid(model(img_t)).detach().cpu().numpy()[0, 0]
    threshold = REPORT_SEG_MASK_THRESHOLD_DEFAULT
    pred_mask = (prob >= threshold).astype(np.uint8)
    return prob, pred_mask, threshold


# ──────────────────────────────────────────────────────────────────────
#  Label interpretation helpers (from notebook cell 243)
# ──────────────────────────────────────────────────────────────────────

def _label_from_prob(prob: np.ndarray, names: list[str]):
    idx = int(np.argmax(prob))
    confidence = float(prob[idx])
    label = names[idx]
    if confidence < REPORT_CONFIDENCE_THRESHOLD:
        return label, confidence, f"uncertain - specialist review recommended ({label})"
    return label, confidence, label


def _imaging_risk(tumor_present: bool, tumor_ratio: float,
                  neoplasm_label: str, uncertain_ratio: bool = False) -> str:
    if not tumor_present:
        return "No visible tumour pattern detected by model."
    if uncertain_ratio:
        return "Imaging-risk estimate unavailable - segmentation review recommended."
    if neoplasm_label == "malignant" or tumor_ratio >= 0.04:
        return "High imaging-risk proxy - urgent specialist review recommended."
    if tumor_ratio >= 0.015:
        return "Moderate imaging-risk proxy - orthopedic oncology/radiology review recommended."
    return "Low imaging-risk proxy - clinical correlation and follow-up recommended."


def _instructions(tumor_present: bool, neoplasm_label: str,
                  uncertain: bool = False) -> list[str]:
    if uncertain:
        return [
            "Prediction uncertainty is elevated; obtain radiologist/specialist review.",
            "Do not use this prototype as a standalone diagnosis.",
        ]
    if not tumor_present:
        return [
            "No tumour pattern was confidently detected.",
            "If pain, swelling, or symptoms persist, seek clinical evaluation.",
        ]
    if neoplasm_label == "malignant":
        return [
            "Urgent orthopedic oncology referral is recommended.",
            "Consider MRI/CT and biopsy according to clinician judgment.",
            "Review segmentation overlay and original X-ray with a radiologist.",
        ]
    if neoplasm_label == "benign":
        return [
            "Orthopedic/radiology follow-up is recommended.",
            "Monitor symptoms and consider additional imaging if clinically indicated.",
        ]
    return ["Specialist review recommended because the image appears abnormal or uncertain."]


# ──────────────────────────────────────────────────────────────────────
#  Overlay generation
# ──────────────────────────────────────────────────────────────────────

def _generate_overlay(image_arr: np.ndarray, mask: np.ndarray,
                      out_path: Path) -> Path:
    overlay = image_arr.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] + 0.45 * mask, 0, 1)
    plt.imsave(str(out_path), overlay)
    return out_path


# ──────────────────────────────────────────────────────────────────────
#  PDF report (from notebook cell 243)
# ──────────────────────────────────────────────────────────────────────

class ReportPDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 8, "Bone Tumour X-ray Research Report", ln=1, align="C")
        self.set_font("Arial", "", 9)
        self.cell(
            0, 6,
            "Research decision-support prototype - not a clinical diagnosis",
            ln=1, align="C",
        )
        self.ln(3)

    def section_title(self, title: str):
        self.ln(2)
        self.set_x(self.l_margin)
        self.set_font("Arial", "B", 12)
        self.set_fill_color(230, 235, 245)
        usable_width = self.w - self.l_margin - self.r_margin
        self.cell(usable_width, 8, title, ln=1, fill=True)
        self.set_x(self.l_margin)

    def key_value(self, key: str, value: str):
        self.set_font("Arial", "B", 10)
        key_text = str(key)[:35] + ":"
        value_text = str(value)
        left_margin = self.l_margin
        usable_width = self.w - self.l_margin - self.r_margin
        key_width = min(55, usable_width * 0.36)
        value_width = max(usable_width - key_width, 40)
        self.set_x(left_margin)
        self.cell(key_width, 7, key_text, border=0)
        self.set_font("Arial", "", 10)
        self.multi_cell(value_width, 7, value_text)
        self.set_x(self.l_margin)

    def safe_multicell(self, text: str, h: int = 7, font_size: int = 10):
        self.set_font("Arial", "", font_size)
        self.set_x(self.l_margin)
        usable_width = self.w - self.l_margin - self.r_margin
        self.multi_cell(usable_width, h, str(text))
        self.set_x(self.l_margin)


# ──────────────────────────────────────────────────────────────────────
#  Main report generation function
# ──────────────────────────────────────────────────────────────────────

def generate_web_patient_report(
    image_path: str | Path,
    age: float,
    gender: str,
    report_id: str | None = None,
) -> dict:
    """
    Generate a full patient report from an uploaded X-ray image.

    Parameters
    ----------
    image_path : path to the uploaded X-ray image
    age : patient age in years
    gender : 'M', 'F', 'male', or 'female'
    report_id : optional UUID; generated if not supplied

    Returns
    -------
    dict with pdf_path, overlay_path, and all model predictions
    """
    if report_id is None:
        report_id = uuid.uuid4().hex

    image_path = Path(image_path)

    # --- Load class names from checkpoint ---
    _, cls_meta = _load_cls_model()
    class_names = cls_meta["class_names"]
    tumor_names = class_names["tumor"]
    neoplasm_names = class_names["neoplasm"]
    skeletal_names = class_names["skeletal"]
    bones_names = class_names["bones"]
    tumor_to_id = {name: i for i, name in enumerate(tumor_names)}

    # --- Preprocess ---
    image_arr = preprocess_external_image(image_path)
    meta_arr = _encode_user_meta(age, gender, cls_meta["age_mean"], cls_meta["age_std"])

    # --- Run models ---
    cls_probs = _predict_classifier(image_arr, meta_arr)
    seg_prob, seg_mask, seg_threshold = _predict_segmentation(image_arr)

    # --- Interpret results ---
    tumor_label, tumor_conf, tumor_display = _label_from_prob(cls_probs["tumor"], tumor_names)
    neoplasm_label, neoplasm_conf, neoplasm_display = _label_from_prob(cls_probs["neoplasm"], neoplasm_names)
    skeletal_label, skeletal_conf, skeletal_display = _label_from_prob(cls_probs["skeletal"], skeletal_names)
    bones_label, bones_conf, bones_display = _label_from_prob(cls_probs["bones"], bones_names)

    no_tumor_key = "no_tumor_or_unlabeled"
    no_tumor_idx = tumor_to_id.get(no_tumor_key, 0)
    tumor_present_prob = float(1.0 - cls_probs["tumor"][no_tumor_idx])
    tumor_present = tumor_present_prob >= 0.50

    mask_ratio = float(seg_mask.mean())
    mask_pixels = int(seg_mask.sum())

    if not tumor_present and tumor_present_prob < 0.35:
        tumor_ratio_text = "0 / no visible tumour mask"
        risk_text = _imaging_risk(False, 0.0, neoplasm_label)
        ratio_uncertain = False
    elif mask_ratio < REPORT_SEG_MIN_MASK_RATIO:
        tumor_ratio_text = "uncertain - segmentation review recommended"
        ratio_uncertain = True
        risk_text = _imaging_risk(True, mask_ratio, neoplasm_label, uncertain_ratio=True)
    else:
        tumor_ratio_text = f"{mask_ratio * 100:.2f}% of resized image area ({mask_pixels} pixels)"
        ratio_uncertain = False
        risk_text = _imaging_risk(True, mask_ratio, neoplasm_label)

    # --- Generate overlay ---
    overlay_path = OVERLAY_DIR / f"{report_id}_segmentation_overlay.png"
    _generate_overlay(image_arr, seg_mask, overlay_path)

    # --- Generate PDF ---
    uncertain = (
        any(conf < REPORT_CONFIDENCE_THRESHOLD for conf in [tumor_conf, neoplasm_conf, skeletal_conf, bones_conf])
        or ratio_uncertain
    )
    instr = _instructions(tumor_present, neoplasm_label, uncertain=uncertain)

    pdf_path = REPORT_DIR / f"{report_id}_patient_report.pdf"

    pdf = ReportPDF()
    pdf.add_page()

    pdf.section_title("Patient Input")
    pdf.key_value("Image", image_path.name)
    pdf.key_value("Age", str(age))
    pdf.key_value("Gender", str(gender))
    pdf.key_value("Generated", datetime.now().strftime("%Y-%m-%d %H:%M"))

    pdf.section_title("Model Prediction Summary")
    pdf.key_value(
        "Tumour detected",
        f'{"Yes" if tumor_present else "No/low probability"} (P tumour={tumor_present_prob:.3f})',
    )
    pdf.key_value("Tumour type", f"{tumor_display} (confidence={tumor_conf:.3f})")
    pdf.key_value("Neoplasm", f"{neoplasm_display} (confidence={neoplasm_conf:.3f})")
    pdf.key_value("Skeletal type", f"{skeletal_display} (confidence={skeletal_conf:.3f})")
    pdf.key_value("Bone type", f"{bones_display} (confidence={bones_conf:.3f})")

    pdf.section_title("Segmentation-Based Tumour Ratio")
    pdf.key_value("Tumour ratio", tumor_ratio_text)
    pdf.key_value("Mask threshold", f"{seg_threshold:.3f}")
    pdf.key_value("Imaging-risk estimate", risk_text)
    if overlay_path.exists():
        pdf.set_x(pdf.l_margin)
        pdf.image(str(overlay_path), x=25, w=160)
        pdf.ln(3)
        pdf.set_x(pdf.l_margin)

    pdf.section_title("Instructions")
    pdf.set_font("Arial", "", 10)
    for item in instr:
        pdf.safe_multicell("- " + item, h=7, font_size=10)

    pdf.section_title("Disclaimer")
    pdf.set_font("Arial", "", 9)
    pdf.safe_multicell(
        "This report is generated by a research prototype. It is not a medical "
        "diagnosis and must not replace radiologist or physician assessment.",
        h=6, font_size=9,
    )
    pdf.output(str(pdf_path))

    return {
        "report_id": report_id,
        "pdf_path": str(pdf_path),
        "overlay_path": str(overlay_path),
        "tumor_present_probability": round(tumor_present_prob, 4),
        "tumor_type": tumor_label,
        "tumor_type_display": tumor_display,
        "tumor_confidence": round(tumor_conf, 4),
        "neoplasm": neoplasm_label,
        "neoplasm_display": neoplasm_display,
        "neoplasm_confidence": round(neoplasm_conf, 4),
        "skeletal_type": skeletal_label,
        "skeletal_type_display": skeletal_display,
        "skeletal_confidence": round(skeletal_conf, 4),
        "bones_type": bones_label,
        "bones_type_display": bones_display,
        "bones_confidence": round(bones_conf, 4),
        "tumour_ratio": round(mask_ratio, 6),
        "tumour_ratio_text": tumor_ratio_text,
        "risk_estimate": risk_text,
    }
