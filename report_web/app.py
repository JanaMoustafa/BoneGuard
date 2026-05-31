"""
report_web.app
~~~~~~~~~~~~~~
Flask web application for generating bone tumour patient reports.

Routes
------
GET  /                          Upload form
POST /reports                   Submit X-ray, age, gender -> generate report
GET  /reports/<report_id>       View report results
GET  /reports/<report_id>/pdf   Download generated PDF
GET  /reports/<report_id>/overlay  View/download segmentation overlay
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from tinydb import TinyDB, Query
from werkzeug.utils import secure_filename

from report_web.report_inference import (
    UPLOAD_DIR,
    WEBSITE_DATA,
    generate_web_patient_report,
)

# ──────────────────────────────────────────────────────────────────────
#  App factory
# ──────────────────────────────────────────────────────────────────────

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)
app.secret_key = "boneguard-dev-secret-key-change-in-production"

# ──────────────────────────────────────────────────────────────────────
#  TinyDB setup
# ──────────────────────────────────────────────────────────────────────
DB_DIR = WEBSITE_DATA / "db"
DB_DIR.mkdir(parents=True, exist_ok=True)
db = TinyDB(str(DB_DIR / "reports.json"), indent=2)
ReportQuery = Query()

# ──────────────────────────────────────────────────────────────────────
#  Validation helpers
# ──────────────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _validate_file(file_storage):
    """Return (is_valid, error_message)."""
    if file_storage is None or file_storage.filename == "":
        return False, "No X-ray image was uploaded."
    ext = Path(file_storage.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file type '{ext}'. Please upload PNG, JPG, or JPEG."
    return True, None


def _validate_age(age_str: str | None):
    """Return (age_float, error_message)."""
    if not age_str or age_str.strip() == "":
        return None, "Age is required."
    try:
        age = float(age_str)
    except (ValueError, TypeError):
        return None, f"Invalid age value: '{age_str}'."
    if age < 0 or age > 120:
        return None, f"Age must be between 0 and 120 (got {age})."
    return age, None


def _normalize_gender(gender_str: str | None) -> str:
    """Normalize gender to M/F or return as-is for neutral fallback."""
    if not gender_str:
        return "unknown"
    g = gender_str.strip().lower()
    if g in ("m", "male"):
        return "M"
    if g in ("f", "female"):
        return "F"
    return gender_str  # model will map to 0.5 neutral


# ──────────────────────────────────────────────────────────────────────
#  Template context
# ──────────────────────────────────────────────────────────────────────

@app.context_processor
def inject_globals():
    return {"current_year": datetime.now().year}


# ──────────────────────────────────────────────────────────────────────
#  Routes
# ──────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Show the upload form."""
    return render_template("upload.html")


@app.route("/reports", methods=["POST"])
def create_report():
    """Accept uploaded X-ray + age + gender, run inference, store results."""
    # --- Validate inputs ---
    errors = []

    file = request.files.get("xray")
    file_ok, file_err = _validate_file(file)
    if not file_ok:
        errors.append(file_err)

    age, age_err = _validate_age(request.form.get("age"))
    if age_err:
        errors.append(age_err)

    gender = _normalize_gender(request.form.get("gender"))

    if errors:
        for err in errors:
            flash(err, "error")
        return redirect(url_for("index"))

    # --- Save upload with UUID filename ---
    report_id = uuid.uuid4().hex
    original_filename = secure_filename(file.filename) or "unknown.jpg"
    ext = Path(original_filename).suffix.lower()
    upload_filename = f"{report_id}{ext}"
    upload_path = UPLOAD_DIR / upload_filename
    file.save(str(upload_path))

    # --- Create initial DB record ---
    timestamp = datetime.now().isoformat(timespec="seconds")
    doc = {
        "report_id": report_id,
        "timestamp": timestamp,
        "original_filename": original_filename,
        "upload_path": str(upload_path),
        "age": age,
        "gender": gender,
        "pdf_path": None,
        "overlay_path": None,
        "tumor_present_probability": None,
        "tumor_type": None,
        "tumor_type_display": None,
        "tumor_confidence": None,
        "neoplasm": None,
        "neoplasm_display": None,
        "neoplasm_confidence": None,
        "skeletal_type": None,
        "skeletal_type_display": None,
        "skeletal_confidence": None,
        "bones_type": None,
        "bones_type_display": None,
        "bones_confidence": None,
        "tumour_ratio": None,
        "tumour_ratio_text": None,
        "risk_estimate": None,
        "status": "processing",
        "error": None,
    }
    db.insert(doc)

    # --- Run inference ---
    try:
        result = generate_web_patient_report(
            image_path=str(upload_path),
            age=age,
            gender=gender,
            report_id=report_id,
        )
        # Update DB with results
        db.update(
            {
                "pdf_path": result["pdf_path"],
                "overlay_path": result["overlay_path"],
                "tumor_present_probability": result["tumor_present_probability"],
                "tumor_type": result["tumor_type"],
                "tumor_type_display": result["tumor_type_display"],
                "tumor_confidence": result["tumor_confidence"],
                "neoplasm": result["neoplasm"],
                "neoplasm_display": result["neoplasm_display"],
                "neoplasm_confidence": result["neoplasm_confidence"],
                "skeletal_type": result["skeletal_type"],
                "skeletal_type_display": result["skeletal_type_display"],
                "skeletal_confidence": result["skeletal_confidence"],
                "bones_type": result["bones_type"],
                "bones_type_display": result["bones_type_display"],
                "bones_confidence": result["bones_confidence"],
                "tumour_ratio": result["tumour_ratio"],
                "tumour_ratio_text": result["tumour_ratio_text"],
                "risk_estimate": result["risk_estimate"],
                "status": "completed",
                "error": None,
            },
            ReportQuery.report_id == report_id,
        )
        flash("Report generated successfully!", "success")

    except Exception as e:
        db.update(
            {"status": "failed", "error": str(e)},
            ReportQuery.report_id == report_id,
        )
        flash(f"Report generation failed: {e}", "error")

    return redirect(url_for("view_report", report_id=report_id))


@app.route("/reports/<report_id>")
def view_report(report_id: str):
    """Display the generated report results."""
    results = db.search(ReportQuery.report_id == report_id)
    if not results:
        return render_template(
            "error.html",
            title="Report Not Found",
            message=f"No report found with ID '{report_id[:12]}...'. It may have been deleted.",
        ), 404

    # TinyDB returns a list; convert first match to a simple namespace
    report_data = results[0]
    report = type("Report", (), report_data)()

    return render_template("report.html", report=report)


@app.route("/reports/<report_id>/pdf")
def get_pdf(report_id: str):
    """Download the generated PDF report."""
    results = db.search(ReportQuery.report_id == report_id)
    if not results:
        return render_template(
            "error.html", title="Report Not Found",
            message="The requested report does not exist.",
        ), 404

    pdf_path = results[0].get("pdf_path")
    if not pdf_path or not Path(pdf_path).exists():
        return render_template(
            "error.html", title="PDF Not Available",
            message="The PDF report file was not found. The report may have failed to generate.",
        ), 404

    return send_file(
        pdf_path,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"bone_report_{report_id[:8]}.pdf",
    )


@app.route("/reports/<report_id>/overlay")
def get_overlay(report_id: str):
    """Serve the segmentation overlay image."""
    results = db.search(ReportQuery.report_id == report_id)
    if not results:
        return render_template(
            "error.html", title="Report Not Found",
            message="The requested report does not exist.",
        ), 404

    overlay_path = results[0].get("overlay_path")
    if not overlay_path or not Path(overlay_path).exists():
        return render_template(
            "error.html", title="Overlay Not Available",
            message="The segmentation overlay was not found.",
        ), 404

    return send_file(overlay_path, mimetype="image/png")


# ──────────────────────────────────────────────────────────────────────
#  Error handlers
# ──────────────────────────────────────────────────────────────────────

@app.errorhandler(404)
def page_not_found(e):
    return render_template(
        "error.html",
        title="Page Not Found",
        message="The page you're looking for doesn't exist.",
    ), 404


@app.errorhandler(500)
def internal_error(e):
    return render_template(
        "error.html",
        title="Server Error",
        message="An internal error occurred. Please try again.",
    ), 500
