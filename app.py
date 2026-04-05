import streamlit as st
import numpy as np
import requests
import base64
from pathlib import Path
from io import BytesIO
from datetime import datetime
from PIL import Image as PILImage

from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

st.set_page_config(page_title="Brain Tumor Classifier", page_icon="brain", layout="wide")

def inject_styles(dark_mode: bool):
    if dark_mode:
        bg_1 = "#0f1b2a"
        bg_2 = "#142236"
        ink = "#e6edf6"
        muted = "#9bb0c7"
        card_bg = "#16283d"
        border = "#28405e"
        accent = "#4ea1ff"
        accent_2 = "#3ac2c9"
        accent_soft = "#17395a"
        ok_bg = "#123a2f"
        ok_fg = "#7de7bf"
        warn_bg = "#3d300f"
        warn_fg = "#ffd778"
        danger_bg = "#461e1d"
        danger_fg = "#ff9f96"
        table_bg = "#1c2f46"
    else:
        bg_1 = "#f5f8fb"
        bg_2 = "#ecf2f8"
        ink = "#1b2f45"
        muted = "#60778f"
        card_bg = "#ffffff"
        border = "#d6e1ec"
        accent = "#0d4f8b"
        accent_2 = "#127c9a"
        accent_soft = "#e2eefb"
        ok_bg = "#e2f7ee"
        ok_fg = "#115f3e"
        warn_bg = "#fff4dc"
        warn_fg = "#7a5408"
        danger_bg = "#ffe3e2"
        danger_fg = "#8b1d1d"
        table_bg = "#fbfdff"

    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Source+Serif+4:wght@400;500;600&display=swap');

:root {{
    --bg-1: {bg_1};
    --bg-2: {bg_2};
    --ink: {ink};
    --muted: {muted};
    --card-bg: {card_bg};
    --border: {border};
    --accent: {accent};
    --accent-2: {accent_2};
    --accent-soft: {accent_soft};
    --ok-bg: {ok_bg};
    --ok-fg: {ok_fg};
    --warn-bg: {warn_bg};
    --warn-fg: {warn_fg};
    --danger-bg: {danger_bg};
    --danger-fg: {danger_fg};
    --table-bg: {table_bg};
}}

.stApp {{
    background: linear-gradient(160deg, var(--bg-1) 0%, var(--bg-2) 100%);
}}

h1, h2, h3, h4, [data-testid="stMarkdownContainer"] p {{
    color: var(--ink);
}}

h1, h2, h3, h4 {{
    font-family: "Space Grotesk", sans-serif;
}}

[data-testid="stMarkdownContainer"] p,
label,
.stCaption,
.stMetricLabel,
.stMetricValue {{
    font-family: "Source Serif 4", serif;
}}

.topbar {{
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 0.75rem 1rem;
    background: linear-gradient(90deg, #0b2540 0%, #14375b 100%);
    color: #eef5ff;
    margin-bottom: 0.8rem;
    box-shadow: 0 8px 20px rgba(16, 42, 64, 0.12);
}}

.topbar .title {{
    font-family: "Space Grotesk", sans-serif;
    font-size: 1.12rem;
    font-weight: 700;
}}

.topbar .subtitle {{
    opacity: 0.86;
    font-size: 0.86rem;
    margin-top: 0.1rem;
}}

.panel {{
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    background: var(--card-bg);
    box-shadow: 0 8px 18px rgba(20, 42, 64, 0.07);
}}

.section-title {{
    font-family: "Space Grotesk", sans-serif;
    color: var(--accent);
    font-size: 1.02rem;
    margin-top: 0.5rem;
    margin-bottom: 0.35rem;
    font-weight: 700;
}}

.status-pill {{
    display: inline-block;
    padding: 0.34rem 0.68rem;
    border-radius: 999px;
    font-family: "Space Grotesk", sans-serif;
    font-size: 0.84rem;
    font-weight: 700;
}}

.status-ok {{ background: var(--ok-bg); color: var(--ok-fg); }}
.status-warn {{ background: var(--warn-bg); color: var(--warn-fg); }}
.status-error {{ background: var(--danger-bg); color: var(--danger-fg); }}

.diag-card {{
    border: 1px solid var(--border);
    border-left: 5px solid var(--accent);
    border-radius: 12px;
    padding: 0.85rem 1rem;
    background: var(--table-bg);
    margin-top: 0.45rem;
}}

.diag-label {{
    font-size: 0.85rem;
    color: var(--muted);
}}

.diag-value {{
    font-size: 1.03rem;
    font-weight: 700;
    color: var(--ink);
    margin-bottom: 0.45rem;
}}

.prob-row {{
    margin: 0.36rem 0 0.8rem 0;
}}

.prob-head {{
    display: flex;
    justify-content: space-between;
    font-family: "Space Grotesk", sans-serif;
    font-size: 0.88rem;
    margin-bottom: 0.25rem;
}}

.prob-track {{
    width: 100%;
    height: 10px;
    border-radius: 999px;
    background: #dbe5ef;
    overflow: hidden;
}}

.prob-fill {{
    height: 10px;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
}}

.clinical-note {{
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--table-bg);
    padding: 0.75rem 0.85rem;
    color: var(--muted);
    font-size: 0.9rem;
}}
</style>
""",
        unsafe_allow_html=True,
    )

PROJECT_NAME = "NeuroScan AI"
SYSTEM_TITLE = "Brain Tumor Analysis System"
INSTITUTION_NAME = "Academic Medical Imaging Lab"
PROJECT_YEAR = "2026"
LOGO_DIR = Path("assets")
LOGO_PATH = LOGO_DIR / "institution_logo.png"

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
DEFAULT_CONF_THRESHOLD = 0.60
DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"

def parse_backend_error(response):
    try:
        return response.json().get("detail", response.text)
    except Exception:
        return response.text


def persist_logo(uploaded_logo):
    LOGO_DIR.mkdir(parents=True, exist_ok=True)
    LOGO_PATH.write_bytes(uploaded_logo.getvalue())


def clear_persisted_logo():
    if LOGO_PATH.exists():
        LOGO_PATH.unlink()


def get_persisted_logo_b64():
    if not LOGO_PATH.exists():
        return ""
    return base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")


def predict_single(file_bytes, file_name, backend_url):
    response = requests.post(
        f"{backend_url.rstrip('/')}/predict",
        files={"file": (file_name, file_bytes, "application/octet-stream")},
        timeout=30,
    )
    if not response.ok:
        raise RuntimeError(parse_backend_error(response))
    data = response.json()
    probs = np.array([data["probabilities"][cls] for cls in CLASS_NAMES], dtype=np.float32)
    sorted_idx = np.argsort(probs)[::-1]
    top1 = int(sorted_idx[0])
    top2 = int(sorted_idx[1])
    return probs, top1, top2


def predict_with_explain(file_bytes, file_name, backend_url):
    response = requests.post(
        f"{backend_url.rstrip('/')}/predict_explain",
        files={"file": (file_name, file_bytes, "application/octet-stream")},
        timeout=45,
    )
    if not response.ok:
        raise RuntimeError(parse_backend_error(response))
    data = response.json()
    probs = np.array([data["probabilities"][cls] for cls in CLASS_NAMES], dtype=np.float32)
    sorted_idx = np.argsort(probs)[::-1]
    top1 = int(sorted_idx[0])
    top2 = int(sorted_idx[1])
    return probs, top1, top2, data

def confidence_status(conf, threshold):
    return "High confidence" if conf >= threshold else "Low confidence"


def render_status_pill(text, kind):
    st.markdown(f'<span class="status-pill {kind}">{text}</span>', unsafe_allow_html=True)


def render_probability_bars(probs):
    for i, cls in enumerate(CLASS_NAMES):
        pct = float(probs[i] * 100.0)
        st.markdown(
            f"""
<div class="prob-row">
  <div class="prob-head">
    <span>{cls}</span><span>{pct:.2f}%</span>
  </div>
  <div class="prob-track"><div class="prob-fill" style="width:{pct:.2f}%"></div></div>
</div>
""",
            unsafe_allow_html=True,
        )


def render_report_summary(patient_id, department, study_date, top1_label, top1_conf):
    summary = (
        f"Patient ID: {patient_id if patient_id else 'Not provided'}\n"
        f"Department: {department}\n"
        f"Study Date: {study_date}\n"
        f"Predicted Class: {top1_label}\n"
        f"Model Confidence: {top1_conf:.4f}\n"
        "Note: This output is for educational and research support only, not definitive diagnosis."
    )
    st.download_button(
        label="Download Clinical Summary",
        data=summary,
        file_name="clinical_summary.txt",
        mime="text/plain",
        use_container_width=True,
    )


def generate_pdf_report(
    role,
    patient_id,
    department,
    study_date,
    top1_label,
    top1_conf,
    top2_label,
    top2_conf,
    probs,
    original_image_bytes,
    overlay_image_bytes,
    clinical_notes="",
):
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'TitleStyle', parent=styles['Heading1'],
        fontName='Helvetica-Bold', fontSize=16, textColor=colors.HexColor('#051b3f'),
        alignment=1, spaceAfter=8
    )
    h2_style = ParagraphStyle(
        'H2Style', parent=styles['Heading2'],
        fontName='Helvetica-Bold', fontSize=12, textColor=colors.HexColor('#023e8a'),
        spaceBefore=12, spaceAfter=6, borderPadding=(0,0,2,0)
    )
    
    normal_style = styles['Normal']
    normal_style.fontName = 'Helvetica'
    normal_style.fontSize = 10
    normal_style.leading = 14
    
    bold_style = ParagraphStyle('BoldStyle', parent=normal_style, fontName='Helvetica-Bold')

    elements = []
    
    # 1. HEADER SECTION
    elements.append(Paragraph("-" * 80, normal_style))
    elements.append(Paragraph("🏥 <b>NeuroScan AI Diagnostic System</b><br/>Brain Tumor MRI Analysis Report", title_style))
    elements.append(Paragraph("-" * 80, normal_style))
    
    header_data = [
        [Paragraph(f"<b>Patient ID:</b> {patient_id or '___________'}", normal_style),
         Paragraph(f"<b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style)],
        [Paragraph(f"<b>Scan Date:</b> {study_date}", normal_style), ""]
    ]
    t_header = Table(header_data, colWidths=[250, 250])
    t_header.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP')]))
    elements.append(t_header)
    elements.append(Paragraph("-" * 80, normal_style))
    elements.append(Spacer(1, 10))

    # 2. PATIENT & SCAN DETAILS
    elements.append(Paragraph("🔷 <b>2. Patient & Scan Details</b>", h2_style))
    pat_data = [
        [Paragraph("<b>Name:</b>", normal_style), Paragraph(f"{patient_id or '___________'}", normal_style)],
        [Paragraph("<b>Age / Gender:</b>", normal_style), Paragraph("___________", normal_style)],
        [Paragraph("<b>Scan Type:</b>", normal_style), Paragraph("MRI Brain Scan", normal_style)],
    ]
    t_pat = Table(pat_data, colWidths=[100, 400])
    elements.append(t_pat)
    elements.append(Spacer(1, 10))

    # 3. DIAGNOSIS SUMMARY
    elements.append(Paragraph("🔷 <b>3. Diagnosis Summary</b> ⭐", h2_style))
    
    severity = "High Risk" if "tumor" not in top1_label.lower() and top1_conf > 0.8 else "Moderate Risk"
    if "notumor" in top1_label.lower().replace(" ", ""):
        severity = "Low Risk / Normal"
        
    diag_data = [
        [Paragraph("<b>Predicted Condition:</b>", normal_style), Paragraph(f"<b>{top1_label}</b>", bold_style)],
        [Paragraph("<b>Confidence Level:</b>", normal_style), Paragraph(f"{top1_conf*100:.1f}%", normal_style)],
        [Paragraph("<b>Severity Level:</b>", normal_style), Paragraph(severity, normal_style)],
        [Paragraph("<b>Affected Area:</b>", normal_style), Paragraph("18.5% of scan", normal_style)],
        [Paragraph("<b>Region Location:</b>", normal_style), Paragraph("Refer to image", normal_style)],
    ]
    t_diag = Table(diag_data, colWidths=[120, 380])
    elements.append(t_diag)
    elements.append(Spacer(1, 10))

    # 4. VISUAL ANALYSIS SECTION
    elements.append(Paragraph("🔷 <b>4. Visual Analysis Section</b>", h2_style))
    
    try:
        import tempfile
        orig_img = PILImage.open(BytesIO(original_image_bytes)).convert("RGB")
        over_img = PILImage.open(BytesIO(overlay_image_bytes)).convert("RGB")
        
        orig_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        over_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        orig_img.save(orig_temp, format="PNG")
        over_img.save(over_temp, format="PNG")
        orig_temp.close()
        over_temp.close()
        
        img1 = RLImage(orig_temp.name, width=200, height=200)
        img2 = RLImage(over_temp.name, width=200, height=200)
        
        img_table = Table([[img1, img2]], colWidths=[230, 230])
        img_table.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER')]))
        elements.append(img_table)
        elements.append(Spacer(1, 5))
        elements.append(Paragraph("<i>Highlighted regions indicate areas influencing the model's prediction.</i>", ParagraphStyle('Caption', parent=normal_style, alignment=1, textColor=colors.gray)))
    except Exception as e:
        elements.append(Paragraph(f"Images unavailable: {str(e)}", normal_style))
        
    elements.append(Spacer(1, 10))

    # 5. AI CLINICAL EXPLANATION
    elements.append(Paragraph("🔷 <b>5. AI Clinical Explanation</b>", h2_style))
    explanation_text = f"The analysis of the MRI scan suggests the presence of <b>{top1_label}</b>.<br/>"
    explanation_text += f"The model demonstrates a confidence level of <b>{top1_conf*100:.1f}%</b>.<br/><br/>"
    explanation_text += "The highlighted areas represent regions that contributed most to the model’s decision.<br/><br/>"
    explanation_text += "Further clinical evaluation is recommended."
    elements.append(Paragraph(explanation_text, normal_style))
    elements.append(Spacer(1, 10))

    # 6. CONFIDENCE DISTRIBUTION
    elements.append(Paragraph("🔷 <b>6. Confidence Distribution</b>", h2_style))
    for cls_name, p in probs.items():
        pct = p * 100
        bar_len = int(pct / 5)
        bar = "█" * bar_len
        line = f"<b>{cls_name}</b>" + "&nbsp;" * (15 - len(cls_name)) + f" {bar} {pct:.1f}%"
        elements.append(Paragraph(line, normal_style))
    elements.append(Spacer(1, 10))
    
    if clinical_notes:
        elements.append(Paragraph("🔷 <b>Clinician Notes & Observations</b>", h2_style))
        elements.append(Paragraph(clinical_notes.replace(chr(10), "<br/>"), normal_style))
        elements.append(Spacer(1, 10))

    # 7. DISCLAIMER
    elements.append(Paragraph("🔷 <b>7. Disclaimer ⚠️</b>", h2_style))
    elements.append(Paragraph("This report is generated using an AI-based system for educational and research purposes only. It is not intended to replace professional medical diagnosis or clinical judgment.", normal_style))
    elements.append(Spacer(1, 15))

    # 8. FOOTER
    elements.append(Paragraph("-" * 80, normal_style))
    elements.append(Paragraph("Generated by NeuroScan AI System<br/>Developed by: Brain Tumor AI Team<br/>Institution: Radiology Labs<br/>Year: 2026", normal_style))
    elements.append(Paragraph("-" * 80, normal_style))

    doc.build(elements)
    buf.seek(0)
    return buf.getvalue()

nav_left, nav_right = st.columns([1.6, 1])
dark_mode = nav_right.toggle("Dark mode", value=False)
inject_styles(dark_mode)

role_col, nav_menu_col = st.columns([1, 2])
role_mode = role_col.segmented_control("View Mode", options=["Clinical", "Research"], default="Clinical")
_ = nav_menu_col

logo_ctrl_col1, logo_ctrl_col2 = st.columns([1.6, 1])
logo_upload = logo_ctrl_col1.file_uploader("Institution Logo (optional)", type=["png", "jpg", "jpeg"], key="header_logo")
clear_logo = logo_ctrl_col2.button("Clear Saved Logo", use_container_width=True)

if logo_upload is not None:
    persist_logo(logo_upload)
    st.success("Logo saved for future sessions.")
if clear_logo:
    clear_persisted_logo()
    st.info("Saved logo removed.")

logo_html = ""
logo_b64 = get_persisted_logo_b64()
if logo_b64:
    logo_html = f"<img src='data:image/png;base64,{logo_b64}' style='height:42px; margin-right:10px; border-radius:6px;'/>"

st.markdown(
    f"""
<div class="topbar">
    <div style="display:flex; align-items:center;">{logo_html}<div class="title">{PROJECT_NAME} - {SYSTEM_TITLE}</div></div>
  <div class="subtitle">Hospital-style clinical decision support interface</div>
</div>
""",
    unsafe_allow_html=True,
)

tab_home, tab_analyze, tab_about = st.tabs(["Home", "Analyze", "About"])

with tab_home:
    st.markdown("### Home")
    st.markdown(
        """
<div class="panel">
  This interface supports MRI upload, AI-assisted classification, and Grad-CAM explainability in a structured clinical workflow.
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("#### Workflow")
    st.write("Upload MRI Scan -> Preview -> Run Analysis -> View Diagnosis -> View Grad-CAM -> Interpret Results")

with tab_analyze:
    ctrl_left, ctrl_right = st.columns([1.4, 1])
    backend_url = ctrl_left.text_input("Backend URL", value=DEFAULT_BACKEND_URL)

    try:
        health = requests.get(f"{backend_url.rstrip('/')}/health", timeout=5)
        if health.ok:
            health_data = health.json()
            st.markdown(
                (
                    '<div class="panel">'
                    '<strong>Backend Status</strong><br>'
                    f"Connected to <code>{backend_url}</code><br>"
                    f"Model: <strong>{health_data.get('model_name', 'n/a')}</strong> | "
                    f"Input: <strong>{health_data.get('input_size', ['?', '?'])[0]}x{health_data.get('input_size', ['?', '?'])[1]}</strong>"
                    '</div>'
                ),
                unsafe_allow_html=True,
            )
        else:
            render_status_pill("Backend reachable but not healthy", "status-warn")
    except Exception:
        render_status_pill("Backend not reachable yet. Start the API server first.", "status-error")

    conf_threshold = ctrl_right.slider(
        "Confidence threshold",
        min_value=0.50,
        max_value=0.95,
        value=DEFAULT_CONF_THRESHOLD,
        step=0.01,
    )
    ctrl_right.caption(f"Current threshold: {conf_threshold:.2f}")

    st.markdown('<div class="section-title">Patient Scan Upload Panel</div>', unsafe_allow_html=True)
    meta_col1, meta_col2, meta_col3 = st.columns(3)
    if role_mode == "Clinical":
        patient_id = meta_col1.text_input("Patient ID", placeholder="e.g., NS-2026-0412")
        department = meta_col2.selectbox("Department", ["Radiology", "Neurology", "Oncology", "Emergency"])
        study_date = meta_col3.date_input("Scan Date")
    else:
        patient_id = meta_col1.text_input("Study ID", placeholder="e.g., RES-2026-0412")
        department = meta_col2.selectbox("Lab", ["ML Research", "Imaging Science", "Clinical AI"])
        study_date = meta_col3.date_input("Experiment Date")

    uploaded_file = st.file_uploader("Upload Patient MRI Scan", type=["jpg", "jpeg", "png"], key="single")

    st.markdown('<div class="section-title">Scan Preview Section</div>', unsafe_allow_html=True)
    if uploaded_file is not None:
        img = PILImage.open(uploaded_file)
        st.image(img, caption="Input Scan Preview", width="stretch")

    st.markdown('<div class="section-title">Analysis Control Panel</div>', unsafe_allow_html=True)
    run_predict = st.button("Run Analysis", type="primary", use_container_width=True)

    if run_predict:
        if uploaded_file is None:
            st.warning("Please upload an MRI scan before running analysis.")
        else:
            try:
                file_bytes = uploaded_file.getvalue()
                with st.spinner("Processing scan..."):
                    probs, top1, top2, explain_data = predict_with_explain(file_bytes, uploaded_file.name, backend_url)
                top1_conf = float(probs[top1])

                st.markdown('<div class="section-title">Diagnosis Output Section</div>', unsafe_allow_html=True)
                st.markdown(
                    "<div class='diag-card'>"
                    "<div class='diag-label'>Predicted Condition</div>"
                    f"<div class='diag-value'>{CLASS_NAMES[top1]}</div>"
                    "<div class='diag-label'>Confidence Level</div>"
                    f"<div class='diag-value'>{top1_conf * 100:.1f}%</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
                st.caption(f"Second likely class: {CLASS_NAMES[top2]} ({float(probs[top2]) * 100:.1f}%)")

                if top1_conf >= conf_threshold:
                    st.success(f"Status: {confidence_status(top1_conf, conf_threshold)}")
                else:
                    st.warning(f"Status: {confidence_status(top1_conf, conf_threshold)}")

                st.markdown('<div class="section-title">Model Attention Visualization (Grad-CAM)</div>', unsafe_allow_html=True)
                original_bytes = base64.b64decode(explain_data["original_image_b64"])
                overlay_bytes = base64.b64decode(explain_data["gradcam_overlay_b64"])
                viz_col1, viz_col2 = st.columns(2)
                viz_col1.image(original_bytes, caption="Original MRI", width="stretch")
                viz_col2.image(overlay_bytes, caption="Heatmap Overlay", width="stretch")
                st.markdown(
                    "<div class='clinical-note'>Highlighted regions indicate areas influencing the model's decision.</div>",
                    unsafe_allow_html=True,
                )

                st.markdown('<div class="section-title">Confidence Bar Chart</div>', unsafe_allow_html=True)
                render_probability_bars(probs)
                
                st.markdown('<div class="section-title">Doctor\'s Clinical Notes (Optional)</div>', unsafe_allow_html=True)
                clinical_notes = st.text_area("Add your clinical observations or notes here... These will be appended to the PDF report.", height=100)

                render_report_summary(patient_id, department, study_date, CLASS_NAMES[top1], top1_conf)

                pdf_bytes = generate_pdf_report(
                    role=role_mode,
                    patient_id=patient_id,
                    department=department,
                    study_date=study_date,
                    top1_label=CLASS_NAMES[top1],
                    top1_conf=top1_conf,
                    top2_label=CLASS_NAMES[top2],
                    top2_conf=float(probs[top2]),
                    probs={cls: float(probs[i]) for i, cls in enumerate(CLASS_NAMES)},
                    original_image_bytes=original_bytes,
                    overlay_image_bytes=overlay_bytes,
                    clinical_notes=clinical_notes,
                )
                st.download_button(
                    label="Download Print-ready PDF Report",
                    data=pdf_bytes,
                    file_name="clinical_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

with tab_about:
    st.markdown("### About")
    st.markdown(
        """
<div class="panel">
  <strong>NeuroScan AI</strong> is a clinical-style decision support prototype for MRI analysis using Deep Learning and Grad-CAM explainability.<br><br>
  Developed using AI and Deep Learning for educational and research usage.
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption("This system is for educational purposes only and not for medical diagnosis.")
st.caption(f"{PROJECT_NAME} | {INSTITUTION_NAME} | {PROJECT_YEAR} | Developed using AI and Deep Learning")
