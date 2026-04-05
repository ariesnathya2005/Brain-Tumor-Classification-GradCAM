"""
Flask server for NeuroScan AI Frontend
Serves the professional HTML interface and proxies API calls to the FastAPI backend
"""
import os
import base64

from flask import Flask, render_template, jsonify, request, send_from_directory, send_file
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

from datetime import datetime
import requests

app = Flask(__name__, template_folder="templates")

# FastAPI backend URL
BACKEND_URL = "http://127.0.0.1:8000"

# Ensure templates directory exists
os.makedirs("templates", exist_ok=True)

@app.route("/")
def index():
    """Serve the main application page"""
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Proxy prediction request to FastAPI backend
    Expects multipart form data with 'file' field
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        # Read file bytes
        file_bytes = file.read()

        # Call FastAPI backend /predict_explain endpoint
        files = {"file": file_bytes}
        response = requests.post(f"{BACKEND_URL}/predict_explain", files=files)

        if response.status_code != 200:
            return jsonify({"error": f"Backend error: {response.text}"}), response.status_code

        data = response.json()
        return jsonify({
            "success": True,
            "prediction": data.get("top1_class"),
            "confidence": data.get("top1_conf"),
            "top2_class": data.get("top2_class"),
            "top2_conf": data.get("top2_conf"),
            "probabilities": data.get("probabilities"),
            "original_image": data.get("original_image_b64"),
            "gradcam_overlay": data.get("gradcam_overlay_b64"),
            "boxes_overlay": data.get("boxes_overlay_b64"),
            "tumor_area_percentage": data.get("tumor_area_percentage"),
            "severity": data.get("severity"),
            "warning": data.get("warning"),
            "explanation": data.get("explanation"),
        })

    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Cannot connect to backend. Is it running on port 8000?"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/download_pdf", methods=["POST"])
def download_pdf():
    data = request.json
    
    patient_id = data.get("patient_id", "PT-10234")
    name = data.get("name", "Arun Kumar")
    age_gender = data.get("age_gender", "45 / Male")
    predicted = data.get("predicted", "Glioma")
    confidence = data.get("confidence", "91.4")
    severity = data.get("severity", "High Risk")
    area = data.get("area", "18.5")
    location = data.get("location", "Right Hemisphere")
    
    # Original image and gradcam
    orig_b64 = data.get("original_image_b64", "")
    gradcam_b64 = data.get("gradcam_overlay_b64", "")
    
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    content_list = []
    
    # 🏥 Header
    content_list.append(Paragraph("<b>NeuroScan AI Diagnostic System</b>", styles['Title']))
    content_list.append(Paragraph("Brain Tumor MRI Analysis Report", styles['Heading2']))
    content_list.append(Spacer(1, 12))
    
    # Patient Info
    content_list.append(Paragraph(f"<b>Patient ID:</b> {patient_id}", styles['Normal']))
    content_list.append(Paragraph(f"<b>Name:</b> {name}", styles['Normal']))
    content_list.append(Paragraph(f"<b>Age/Gender:</b> {age_gender}", styles['Normal']))
    content_list.append(Spacer(1, 12))
    
    # 🧠 Diagnosis Table
    diagnosis_data = [
        ["Parameter", "Value"],
        ["Predicted Condition", predicted],
        ["Confidence Level", f"{confidence}%"],
        ["Severity Level", severity],
        ["Affected Area", f"{area}%"],
        ["Region Location", location]
    ]
    
    table = Table(diagnosis_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke)
    ]))
    
    content_list.append(Paragraph("<b>Diagnosis Summary</b>", styles['Heading3']))
    content_list.append(table)
    content_list.append(Spacer(1, 20))
    
    # 🖼️ Images Section
    try:
        if orig_b64 and gradcam_b64:
            img1_data = io.BytesIO(base64.b64decode(orig_b64))
            img2_data = io.BytesIO(base64.b64decode(gradcam_b64))
            img1 = RLImage(img1_data, width=2.5*inch, height=2.5*inch)
            img2 = RLImage(img2_data, width=2.5*inch, height=2.5*inch)
            
            img_table = Table([[img1, img2]])
            content_list.append(Paragraph("<b>Visual Analysis</b>", styles['Heading3']))
            content_list.append(img_table)
            content_list.append(Paragraph("Highlighted regions indicate important areas.", styles['Italic']))
    except Exception as e:
        print("Image processing error:", e)
        content_list.append(Paragraph("Images not found.", styles['Normal']))
        
    content_list.append(Spacer(1, 20))
    
    # 🧠 AI Explanation
    explanation = f"""
    The analysis of the MRI scan suggests the presence of {predicted}.
    The model demonstrates a confidence level of {confidence}%.
    The region of interest is located in the {location}, covering approximately {area}% of the scan.
    The highlighted areas contributed most to the model's prediction.
    Further clinical evaluation is recommended.
    """
    
    content_list.append(Paragraph("<b>AI Clinical Insight</b>", styles['Heading3']))
    content_list.append(Paragraph(explanation.replace('\n', ' '), styles['Normal']))
    content_list.append(Spacer(1, 20))
    
    # 📊 Confidence Chart (Text-based)
    content_list.append(Paragraph("<b>Prediction Confidence Distribution</b>", styles['Heading3']))
    
    # create text bars
    probs = data.get("probabilities", {})
    if not probs:
        probs = {predicted.lower(): float(confidence)/100}
    
    # Sorted probabilities
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    for cls_name, prob_val in sorted_probs:
        pct = int(prob_val * 100)
        blocks = "█" * int(pct / 3.3) # scale somewhat close to 20 blocks max
        line = f"{cls_name.capitalize().ljust(14)} {blocks} {pct}%"
        content_list.append(Paragraph(line, styles['Normal']))
        
    content_list.append(Spacer(1, 20))
    
    # ⚠️ Disclaimer
    disclaimer = """
    This report is generated using an AI-based system for educational and research purposes only.
    It does not replace professional medical diagnosis.
    """
    
    content_list.append(Paragraph("<b>Disclaimer</b>", styles['Heading3']))
    content_list.append(Paragraph(disclaimer.replace('\n', ' '), styles['Normal']))
    content_list.append(Spacer(1, 20))
    
    # Footer
    content_list.append(Paragraph("Generated by NeuroScan AI System", styles['Normal']))
    content_list.append(Paragraph("Developed by: Aries Nathya", styles['Normal']))
    content_list.append(Paragraph(f"Year: 2026", styles['Normal']))
    
    doc.build(content_list)
    pdf_buffer.seek(0)
    
    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name=f"Brain_Tumor_Report_{patient_id}.pdf",
        mimetype="application/pdf"
    )

@app.route("/api/health"
, methods=["GET"])
def health():
    """Check if backend is available"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        if response.status_code == 200:
            return jsonify({"status": "ok", "backend": response.json()})
        return jsonify({"status": "backend_error"}), 503
    except:
        return jsonify({"status": "backend_unavailable"}), 503

if __name__ == "__main__":
    # Check if backend is running
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=1)
        print(f"✓ Backend is running: {r.json()}")
    except:
        print("⚠ Backend is not running on port 8000. Please start it first:")
        print("  source .venv/bin/activate && uvicorn backend_api:app --host 127.0.0.1 --port 8000")

    print("Starting Flask frontend on http://127.0.0.1:5005")
    app.run(host="127.0.0.1", port=5005, debug=False)
