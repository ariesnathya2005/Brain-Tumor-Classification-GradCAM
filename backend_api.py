import base64
from io import BytesIO

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
MODEL_PATH = "brain_tumor_model.h5"

app = FastAPI(title="Brain Tumor Inference API", version="1.0.0")
model = tf.keras.models.load_model(MODEL_PATH)

def resolve_input_size(m):
    input_shape = m.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    if len(input_shape) == 4 and input_shape[1] and input_shape[2]:
        return (int(input_shape[1]), int(input_shape[2]))
    return (128, 128)

def resolve_preprocess_fn(m):
    name = (m.layers[0].name if m.layers else m.name).lower()
    if "efficientnet" in name:
        return tf.keras.applications.efficientnet.preprocess_input
    if "mobilenet" in name:
        return tf.keras.applications.mobilenet_v2.preprocess_input
    return lambda x: x / 255.0

IMG_SIZE = resolve_input_size(model)
PREPROCESS_FN = resolve_preprocess_fn(model)


def get_base_model(m):
    if hasattr(m, "layers") and m.layers:
        first_layer = m.layers[0]
        if hasattr(first_layer, "get_layer") and hasattr(first_layer, "input"):
            return first_layer
    return m


def resolve_last_conv_layer_name(m):
    base = get_base_model(m)
    for layer in reversed(base.layers):
        output_tensor = getattr(layer, "output", None)
        if output_tensor is not None:
            shape = getattr(output_tensor, "shape", None)
            if shape is not None and len(shape) == 4:
                return layer.name
    return None


LAST_CONV_LAYER_NAME = resolve_last_conv_layer_name(model)


def load_image_arrays(file_bytes: bytes):
    image = Image.open(BytesIO(file_bytes)).convert("RGB")
    # Resize specifically for the model input
    model_image = image.resize(IMG_SIZE)
    # Resize specifically for high-res output on the frontend (e.g. 512x512)
    output_image = image.resize((512, 512))
    
    original = np.array(model_image, dtype=np.uint8)
    high_res_rgb = np.array(output_image, dtype=np.uint8)
    
    model_input = np.expand_dims(original.astype(np.float32), axis=0)
    model_input = PREPROCESS_FN(model_input)
    return high_res_rgb, model_input


def run_inference(model_input):
    probs = model.predict(model_input, verbose=0)[0]
    sorted_idx = np.argsort(probs)[::-1]
    top1 = int(sorted_idx[0])
    top2 = int(sorted_idx[1])
    return probs, top1, top2


def make_gradcam_heatmap(model_input, pred_index=None):
    base = get_base_model(model)
    last_conv_name = LAST_CONV_LAYER_NAME
    if last_conv_name is None:
        return None

    _ = model(model_input, training=False)
    last_conv_layer = base.get_layer(last_conv_name)
    last_conv_model = tf.keras.models.Model(base.input, last_conv_layer.output)

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    if base is model:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    else:
        for layer in model.layers[1:]:
            x = layer(x)
    classifier_model = tf.keras.models.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        conv_outputs = last_conv_model(model_input)
        tape.watch(conv_outputs)
        predictions = classifier_model(conv_outputs)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def overlay_gradcam_on_image(original_rgb, heatmap, alpha=0.45):
    """
    Professional Hospital-Grade Grad-CAM overlay with aggressive tumor region highlighting
    Ensures regions are visible even with subtle heatmaps
    """
    if heatmap is None:
        return original_rgb
    
    # Prepare heatmap with intensity enhancement
    heatmap_uint8 = np.uint8(255 * heatmap)
    h_target, w_target = original_rgb.shape[0], original_rgb.shape[1]
    heatmap_uint8 = cv2.resize(heatmap_uint8, (w_target, h_target))
    
    # Enhance heatmap contrast to make regions more visible
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    heatmap_enhanced = clahe.apply(heatmap_uint8)
    
    # Create color-mapped heatmap (JET colormap: blue=low, red=high attention)
    heatmap_color = cv2.applyColorMap(heatmap_enhanced, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Blend with original
    overlay = cv2.addWeighted(original_rgb, 1 - alpha, heatmap_color, alpha, 0).astype(np.uint8)
    pure_heatmap_overlay = overlay.copy()
    
    # Create boxes overlay on original image without heatmap
    boxes_overlay = original_rgb.copy()
    
    # Detect high-activation regions with LOWER threshold for better visibility
    threshold = 0.25  # Much lower threshold to catch subtle regions
    binary_mask = (heatmap > threshold).astype(np.uint8)
    binary_mask_resized = cv2.resize(binary_mask, (w_target, h_target))
    
    # Aggressive morphological operations to connect and clean regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary_mask_resized = cv2.morphologyEx(binary_mask_resized, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary_mask_resized = cv2.morphologyEx(binary_mask_resized, cv2.MORPH_OPEN, kernel, iterations=1)
    # Dilate to make regions more prominent
    binary_mask_resized = cv2.dilate(binary_mask_resized, kernel, iterations=1)
    
    # Find contours of suspicious regions
    contours, _ = cv2.findContours(binary_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_regions = 0
    
    # Draw professional clinical annotations on detected regions
    if len(contours) > 0:
        # Sort contours by area (largest first = most suspicious)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]  # Top 5 regions
        
        confidence_colors = {
            "HIGH": (0, 0, 255),      # RED (BGR format: actually (255, 0, 0) in RGB)
            "MEDIUM": (0, 165, 255),  # ORANGE
            "LOW": (0, 255, 255)      # YELLOW
        }
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 50:  # Reduced threshold to catch small regions
                continue
            
            detected_regions += 1
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Ensure bounds are within image
            x = max(0, x)
            y = max(0, y)
            x_end = min(IMG_SIZE[1], x + w)
            y_end = min(IMG_SIZE[0], y + h)
            
            # Determine severity based on region heatmap intensity
            region_heatmap_x = max(0, x // 6)
            region_heatmap_y = max(0, y // 6)
            region_heatmap_x_end = min(heatmap.shape[1], (x_end) // 6)
            region_heatmap_y_end = min(heatmap.shape[0], (y_end) // 6)
            
            if region_heatmap_x_end > region_heatmap_x and region_heatmap_y_end > region_heatmap_y:
                region_heatmap = heatmap[region_heatmap_y:region_heatmap_y_end, 
                                        region_heatmap_x:region_heatmap_x_end]
                avg_intensity = np.mean(region_heatmap) if region_heatmap.size > 0 else 0
            else:
                avg_intensity = 0.3
            
            # Determine severity
            if avg_intensity > 0.5:
                severity = "HIGH"
                color = (0, 0, 255)  # RED in BGR
            elif avg_intensity > 0.35:
                severity = "MEDIUM"
                color = (0, 165, 255)  # ORANGE in BGR
            else:
                severity = "LOW"
                color = (0, 255, 255)  # YELLOW in BGR
            
            # Convert to RGB for overlay
            color_rgb = (color[2], color[1], color[0])
            
            # Draw THICK bounding box (very visible)
            cv2.rectangle(boxes_overlay, (x, y), (x_end, y_end), color_rgb, 4)
            
            # Draw corner markers (like medical annotations) - thicker
            corner_size = 20
            thickness = 3
            # Top-left
            cv2.line(boxes_overlay, (x, y), (x+corner_size, y), color_rgb, thickness)
            cv2.line(boxes_overlay, (x, y), (x, y+corner_size), color_rgb, thickness)
            # Top-right
            cv2.line(boxes_overlay, (x_end, y), (x_end-corner_size, y), color_rgb, thickness)
            cv2.line(boxes_overlay, (x_end, y), (x_end, y+corner_size), color_rgb, thickness)
            # Bottom-left
            cv2.line(boxes_overlay, (x, y_end), (x+corner_size, y_end), color_rgb, thickness)
            cv2.line(boxes_overlay, (x, y_end), (x, y_end-corner_size), color_rgb, thickness)
            # Bottom-right
            cv2.line(boxes_overlay, (x_end, y_end), (x_end-corner_size, y_end), color_rgb, thickness)
            cv2.line(boxes_overlay, (x_end, y_end), (x_end, y_end-corner_size), color_rgb, thickness)
            
            # Draw circle at centroid - LARGER and VERY visible
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(boxes_overlay, (cx, cy), 12, color_rgb, -1)  # Filled circle
                cv2.circle(boxes_overlay, (cx, cy), 12, (255, 255, 255), 3)  # White outline
            
            # Professional label with severity
            label_text = f"Region {detected_regions}: {severity}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
            
            # Position label above the box
            label_x = max(5, x)
            label_y = max(30, y - 15)
            
            # Draw thick black background for label
            cv2.rectangle(boxes_overlay, 
                         (label_x - 8, label_y - text_size[1] - 10),
                         (label_x + text_size[0] + 8, label_y + 8),
                         (0, 0, 0), -1)
            
            # Draw bright white text
            cv2.putText(boxes_overlay, label_text, (label_x, label_y), 
                       font, font_scale, (255, 255, 255), thickness)
            
            # Add confidence percentage below label
            confidence_text = f"Attention: {int(avg_intensity*100)}%"
            cv2.putText(boxes_overlay, confidence_text, (label_x, label_y + 25), 
                       font, 0.65, color_rgb, 2)
    
    # FALLBACK: If no contours detected, find intensity peaks and mark them
    if detected_regions == 0:
        # Find local maxima in heatmap
        heatmap_resized = cv2.resize(heatmap, (w_target, h_target))
        threshold_fallback = 0.4
        peaks = np.where(heatmap_resized > threshold_fallback)
        
        if len(peaks[0]) > 0:
            # Cluster nearby pixels
            points = np.column_stack((peaks[1], peaks[0]))
            
            # Find top intensity point and mark it
            intensities = heatmap_resized[peaks]
            if len(intensities) > 0:
                max_idx = np.argmax(intensities)
                peak_x = points[max_idx][0]
                peak_y = points[max_idx][1]
                
                # Draw circle at peak
                cv2.circle(boxes_overlay, (int(peak_x), int(peak_y)), 15, (0, 0, 255), -1)
                cv2.circle(boxes_overlay, (int(peak_x), int(peak_y)), 15, (255, 255, 255), 3)
                
                # Label it
                cv2.putText(boxes_overlay, "SUSPICIOUS AREA", 
                           (int(peak_x) + 20, int(peak_y)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    return pure_heatmap_overlay, boxes_overlay


def encode_rgb_image_to_base64(image_rgb):
    ok, buffer = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")

@app.get("/health")
def health() -> dict:
    model_name = model.layers[0].name if model.layers else model.name
    return {
        "status": "ok",
        "model_loaded": True,
        "model_name": model_name,
        "input_size": [IMG_SIZE[0], IMG_SIZE[1]],
        "last_conv_layer": LAST_CONV_LAYER_NAME,
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    try:
        file_bytes = await file.read()
        _, model_input = load_image_arrays(file_bytes)
        probs, top1, top2 = run_inference(model_input)

        return {
            "top1_class": CLASS_NAMES[top1],
            "top1_conf": float(probs[top1]),
            "top2_class": CLASS_NAMES[top2],
            "top2_conf": float(probs[top2]),
            "probabilities": {cls: float(probs[i]) for i, cls in enumerate(CLASS_NAMES)},
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc


@app.post("/predict_explain")
async def predict_explain(file: UploadFile = File(...)) -> dict:
    try:
        file_bytes = await file.read()
        original_rgb, model_input = load_image_arrays(file_bytes)
        probs, top1, top2 = run_inference(model_input)
        heatmap = make_gradcam_heatmap(model_input, pred_index=top1)
        pure_heatmap_overlay, boxes_overlay = overlay_gradcam_on_image(original_rgb, heatmap)
        
        tumor_area_percentage = 0.0
        location = "unspecified"
        if heatmap is not None:
            # Calculate what percentage of the image is "highly activated"
            highly_activated = np.sum(heatmap > 0.25)
            tumor_area_percentage = float(highly_activated / heatmap.size * 100)
            
            # Calculate region location based on heatmap
            h, w = heatmap.shape
            y_indices, x_indices = np.where(heatmap > 0.25)
            if len(y_indices) > 0 and len(x_indices) > 0:
                cy = np.mean(y_indices)
                cx = np.mean(x_indices)
                vertical = "upper" if cy < h / 2 else "lower"
                horizontal = "left" if cx < w / 2 else "right"
                location = f"{vertical} {horizontal}"
            else:
                location = "central"
            
        top1_class = CLASS_NAMES[top1]
        confidence = float(probs[top1])
        
        # 1. Tumor Severity & 7. Confidence Warning Logic
        if top1_class == "notumor":
            severity = "Low Risk"
        else:
            if confidence > 0.85 and tumor_area_percentage > 5.0:
                severity = "High Risk"
            elif confidence > 0.60:
                severity = "Moderate Risk"
            else:
                severity = "Low Risk"
                
        warning = None
        if confidence < 0.70:
            warning = "⚠️ Low confidence prediction — further medical review recommended"
            
        # 4. AI Explanation text
        formatted_class_name = top1_class.capitalize() if top1_class != "notumor" else "No Tumor"
        if top1_class == "notumor":
            explanation = (
                "The AI analysis of this MRI scan did not detect significant abnormal tissue patterns typically associated with tumors. "
                "The regions analyzed appear structurally consistent with healthy brain tissue. "
                "However, as AI is an assistive screening tool, a routine review by a neurologist or radiologist is always recommended for complete assurance."
            )
        else:
            explanation = (
                f"The analysis of the MRI scan suggests the presence of {formatted_class_name}.\n"
                f"The model demonstrates a confidence level of {confidence*100:.1f}%.\n\n"
                f"The region of interest is predominantly located in the {location} region of the brain, "
                f"with approximately {tumor_area_percentage:.1f}% of the scan area showing abnormal activation.\n\n"
                "The highlighted areas represent regions that contributed most to the model's decision. "
                "Further clinical evaluation is recommended."
            )

        return {
            "top1_class": top1_class,
            "top1_conf": confidence,
            "top2_class": CLASS_NAMES[top2],
            "top2_conf": float(probs[top2]),
            "probabilities": {cls: float(probs[i]) for i, cls in enumerate(CLASS_NAMES)},
            "original_image_b64": encode_rgb_image_to_base64(original_rgb),
            "gradcam_overlay_b64": encode_rgb_image_to_base64(pure_heatmap_overlay),
            "boxes_overlay_b64": encode_rgb_image_to_base64(boxes_overlay),
            "tumor_area_percentage": tumor_area_percentage,
            "severity": severity,
            "warning": warning,
            "explanation": explanation
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction explain failed: {exc}") from exc
