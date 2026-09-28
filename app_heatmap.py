import json
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "training_output" / "models" / "best_crop_model.pth"
JSON_DIR = BASE_DIR / "Crop_Papers" / "extracted_results"

DISEASE_MODEL_ROOT = BASE_DIR / "disease_models"

# Each crop has its own disease model.
DISEASE_MODEL_MAP = {
    "Arhar": "arhar_disease_model.pth",
    "Bajra": "bajra_disease_model.pth",
    "Cotton": "cotton_disease_model.pth",
    "Cucumber": "cucumber_disease_model.pth",
    "Fodder": "fodder_disease_model.pth",
    "Maize": "maize_disease_model.pth",
    "Masoor": "masoor_disease_model.pth",
    "Matar": "matar_disease_model.pth",
    "Moongfali": "moongfali_disease_model.pth",
    "Potato": "potato_disease_model.pth",
    "Rice": "rice_disease_model.pth",
    "Soybean": "soybean_disease_model.pth",
    "Urad": "urad_disease_model.pth",
    "Watermelon": "watermelon_disease_model.pth",
    "Wheat": "wheat_disease_model.pth",
}

CLASS_NAMES = [
    "Arhar", "Bajra", "Cotton", "Cucumber", "Fodder",
    "Maize", "Masoor", "Matar", "Moongfali", "Potato",
    "Rice", "Soybean", "Urad", "Watermelon", "Wheat"
]

JSON_FILE_MAP = {
    "Arhar": "arhar.json",
    "Bajra": "bajra.json",
    "Cotton": "cotton.json",
    "Cucumber": "cucumber.json",
    "Fodder": "fodder.json",
    "Maize": "maize.json",
    "Masoor": "masoor.json",
    "Matar": "matar.json",
    "Moongfali": "moongfali.json",
    "Potato": "patato.json",
    "Rice": "rice.json",
    "Soybean": "soybean.json",
    "Urad": "urad.json",
    "Watermelon": "watermelon.json",
    "Wheat": "wheat.json",
}

st.set_page_config(
    page_title="Krishi AI",
    page_icon="🌱",
    layout="wide",
)

# ============================================================
# UI - SAME DARK STYLE / CLEAN LAYOUT
# ============================================================
st.markdown("""
<style>
.block-container {
    padding-top: 3.2rem !important;
    padding-bottom: 2rem;
    max-width: 1400px;
}

header[data-testid="stHeader"] {
    height: 2.8rem;
}

div[data-testid="stAppViewContainer"] > .main {
    padding-top: 0 !important;
}

.upload-title {
    font-size: 28px;
    font-weight: 750;
    margin-bottom: 4px;
}

.upload-subtitle {
    font-size: 15px;
    color: #a1a1aa;
    margin-bottom: 12px;
}

.crop-label {
    font-size: 13px;
    letter-spacing: 2px;
    color: #a1a1aa;
    margin-bottom: 5px;
}

.crop-name {
    font-size: 36px;
    font-weight: 800;
    margin-bottom: 4px;
}

.conf-title {
    font-size: 20px;
    font-weight: 700;
    margin-top: 2px;
    margin-bottom: 5px;
}

.top-title {
    font-size: 25px;
    font-weight: 750;
    margin-top: 25px;
    margin-bottom: 14px;
}

.pred-row {
    display: flex;
    justify-content: space-between;
    padding: 7px 0;
    font-size: 16px;
}

.pred-name {
    font-weight: 650;
}

.pred-confidence {
    font-weight: 500;
}

.section-title {
    font-size: 25px;
    font-weight: 750;
    margin-top: 34px;
    margin-bottom: 12px;
}

.disease-card {
    padding: 18px 20px;
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 14px;
    background: rgba(128,128,128,.06);
    margin-bottom: 15px;
}

.disease-name {
    font-size: 27px;
    font-weight: 800;
}

.info-heading {
    font-size: 18px;
    font-weight: 750;
    margin-top: 12px;
    margin-bottom: 6px;
}

.info-text {
    font-size: 15px;
    line-height: 1.55;
}

.muted {
    color: #a1a1aa;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL
# IMPORTANT: matches trained classifier.1.1.* architecture
# ============================================================
@st.cache_resource
def load_model():
    model = models.efficientnet_v2_s(weights=None)

    # Training checkpoint expects:
    # classifier.1.1.weight
    # classifier.1.1.bias
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(
            model.classifier[1].in_features,
            len(CLASS_NAMES)
        )
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# DISEASE MODEL
# ============================================================
@st.cache_resource
def load_disease_model(crop):
    model_path = DISEASE_MODEL_ROOT / crop / "models" / DISEASE_MODEL_MAP[crop]

    if not model_path.exists():
        return None, model_path, []

    # Read the exact class mapping saved with the crop-specific model.
    metadata_candidates = [
        DISEASE_MODEL_ROOT / crop / "metadata.json",
        DISEASE_MODEL_ROOT / crop / "model_metadata.json",
        DISEASE_MODEL_ROOT / crop / "reports" / "metadata.json",
    ]

    class_names = None
    for metadata_path in metadata_candidates:
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                class_names = (
                    meta.get("class_names")
                    or meta.get("classes")
                    or meta.get("disease_classes")
                )
                if class_names:
                    break
            except Exception:
                pass

    # Fallback: the checkpoint may itself contain class names.
    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False
    )

    if class_names is None and isinstance(checkpoint, dict):
        class_names = (
            checkpoint.get("class_names")
            or checkpoint.get("classes")
            or checkpoint.get("disease_classes")
        )

    if not class_names:
        raise RuntimeError(
            f"Could not find disease class mapping for {crop}. "
            f"Expected metadata.json/model_metadata.json or class_names "
            f"inside {model_path.name}."
        )

    # IMPORTANT: disease models were trained with a plain Linear head:
    # classifier.1.weight / classifier.1.bias
    # The crop classifier is different and uses classifier.1.1.*.
    model = models.efficientnet_v2_s(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, len(class_names)
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model, model_path, class_names



def generate_gradcam(model, image, class_index):
    """
    Generate a Grad-CAM heatmap for the disease model's predicted class.
    The heatmap is based on EfficientNetV2-S final convolutional features.
    """
    target_layer = model.features[-1]

    activations = []
    gradients = []

    def forward_hook(module, inputs, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        if grad_output and grad_output[0] is not None:
            gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        tensor = transform(image).unsqueeze(0)

        model.zero_grad(set_to_none=True)

        output = model(tensor)
        score = output[0, class_index]
        score.backward()

        if not activations or not gradients:
            return None

        activation = activations[0].detach()
        gradient = gradients[0].detach()

        # Global-average-pool gradients to obtain channel importance.
        weights = gradient.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of feature maps.
        cam = (weights * activation).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        # Resize CAM to the original uploaded image size.
        cam = torch.nn.functional.interpolate(
            cam,
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0].cpu().numpy()

        # Normalize to 0..1.
        cam -= cam.min()
        max_value = cam.max()
        if max_value > 1e-8:
            cam /= max_value
        else:
            return None

        # Create a red/yellow heatmap without requiring OpenCV.
        heatmap = np.zeros((cam.shape[0], cam.shape[1], 3), dtype=np.float32)

        # Blue -> cyan -> yellow -> red style gradient.
        heatmap[..., 0] = np.clip(2.0 * cam, 0.0, 1.0)
        heatmap[..., 1] = np.clip(2.0 * (1.0 - np.abs(cam - 0.5) * 2.0), 0.0, 1.0)
        heatmap[..., 2] = np.clip(2.0 * (1.0 - cam), 0.0, 1.0)

        heatmap = (heatmap * 255).astype(np.uint8)

        # Blend the heatmap over the original image.
        original = np.array(image).astype(np.float32)
        overlay = (0.45 * original + 0.55 * heatmap).clip(0, 255).astype(np.uint8)

        return Image.fromarray(overlay)

    finally:
        forward_handle.remove()
        backward_handle.remove()



def predict_disease(image, crop):
    model, model_path, class_names = load_disease_model(crop)

    if model is None:
        return None, None, model_path

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)[0]

    top_probs, top_indices = torch.topk(
        probabilities,
        k=min(len(class_names), 5)
    )

    predictions = [
        (class_names[int(index)], float(probability))
        for probability, index in zip(top_probs, top_indices)
    ]

    return predictions[0], predictions, model_path


def normalize_name(value):
    """Normalize model/JSON disease names for matching."""
    return (
        clean_text(value)
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace("/", " ")
    ).strip()


def find_matching_record(records, disease_name):
    """Find the research JSON record matching the disease-model prediction."""
    target = normalize_name(disease_name)

    # Exact normalized match first.
    for record in records:
        name = first_value(
            record,
            ["disease_name", "Disease Name", "diseaseName"]
        )
        if normalize_name(name) == target:
            return record

    # Controlled fallback for minor naming differences.
    target_compact = target.replace(" ", "")
    for record in records:
        name = first_value(
            record,
            ["disease_name", "Disease Name", "diseaseName"]
        )
        candidate = normalize_name(name).replace(" ", "")
        if candidate == target_compact:
            return record

    return None


# ============================================================
# JSON HELPERS
# ============================================================
def clean_text(value):
    if value is None:
        return ""

    if isinstance(value, list):
        return " ".join(
            t for t in (clean_text(x) for x in value) if t
        )

    if isinstance(value, dict):
        return " ".join(
            t for t in (clean_text(v) for v in value.values()) if t
        )

    return str(value).strip()


def first_value(d, keys):
    if not isinstance(d, dict):
        return ""

    for key in keys:
        if key in d:
            text = clean_text(d[key])
            if text:
                return text
    return ""


def values_from_section(record, section_names, item_names):
    section = None

    for key in section_names:
        if isinstance(record, dict) and key in record:
            section = record[key]
            break

    if section is None:
        return []

    if isinstance(section, dict):
        for key in item_names:
            if key in section:
                value = section[key]
                if isinstance(value, list):
                    return [clean_text(x) for x in value if clean_text(x)]
                text = clean_text(value)
                return [text] if text else []

        # Fallback: flatten only useful text
        result = []
        for key, value in section.items():
            text = clean_text(value)
            if text:
                result.append(text)
        return result

    if isinstance(section, list):
        return [clean_text(x) for x in section if clean_text(x)]

    text = clean_text(section)
    return [text] if text else []


def find_disease_records(data):
    found = []

    def walk(obj):
        if isinstance(obj, dict):
            disease = first_value(
                obj,
                ["disease_name", "Disease Name", "diseaseName"]
            )

            if disease:
                found.append(obj)

            for value in obj.values():
                walk(value)

        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)

    # Preserve JSON order, remove duplicates.
    unique = []
    seen = set()

    for record in found:
        marker = json.dumps(
            record,
            sort_keys=True,
            ensure_ascii=False
        )
        if marker not in seen:
            seen.add(marker)
            unique.append(record)

    return unique


def load_json_for_crop(crop):
    filename = JSON_FILE_MAP[crop]
    path = JSON_DIR / filename

    if not path.exists():
        return None, path

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f), path


def predict_crop(image):
    model = load_model()

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)[0]

    top_probs, top_indices = torch.topk(
        probabilities,
        k=min(5, len(CLASS_NAMES))
    )

    return [
        (CLASS_NAMES[int(index)], float(probability))
        for probability, index
        in zip(top_probs, top_indices)
    ]


def show_bullets(items, limit=7):
    cleaned = []
    seen = set()

    for item in items:
        item = clean_text(item)
        if not item or item in seen:
            continue
        seen.add(item)
        cleaned.append(item)

    if not cleaned:
        st.markdown(
            '<div class="muted">Not available in the research record.</div>',
            unsafe_allow_html=True
        )
        return

    for item in cleaned[:limit]:
        st.markdown(f"- {item}")


# ============================================================
# HEADER
# ============================================================
st.markdown(
    '<div class="upload-title">📷 Upload Plant / Crop Photo</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="upload-subtitle">Upload a clear crop/leaf image</div>',
    unsafe_allow_html=True
)

uploaded = st.file_uploader(
    "Upload Plant / Crop Photo",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    label_visibility="collapsed"
)

if uploaded is None:
    st.stop()

image = Image.open(uploaded).convert("RGB")

# ============================================================
# PREDICTION AREA
# ============================================================
with st.spinner("Identifying crop..."):
    predictions = predict_crop(image)

crop, confidence = predictions[0]

left, right = st.columns([1.05, 1.25], gap="large")

with left:
    st.image(
        image,
        use_container_width=True
    )

with right:
    st.markdown(
        '<div class="crop-label">IDENTIFIED CROP</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="crop-name">🌱 {crop}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="conf-title">Confidence: {confidence * 100:.2f}%</div>',
        unsafe_allow_html=True
    )

    st.progress(confidence)

    st.markdown(
        '<div class="top-title">🔎 Top Predictions</div>',
        unsafe_allow_html=True
    )

    for i, (name, prob) in enumerate(predictions, start=1):
        st.markdown(
            f"""
            <div class="pred-row">
                <span class="pred-name">{i}. {name}</span>
                <span class="pred-confidence">{prob * 100:.2f}%</span>
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================================
# DISEASE MODEL + RESEARCH INFORMATION
# ============================================================
st.markdown(
    '<div class="section-title">🩺 Disease Detection</div>',
    unsafe_allow_html=True
)

# Load ONLY the disease model belonging to the identified crop.
with st.spinner(f"Running {crop} disease model..."):
    disease_prediction, disease_predictions, disease_model_path = predict_disease(
        image,
        crop
    )

if disease_prediction is None:
    st.error(
        f"Disease model for **{crop}** was not found.\n\n"
        f"Expected model: `{disease_model_path}`"
    )
    st.stop()

disease_name, disease_confidence = disease_prediction

# ============================================================
# VISUAL EXPLANATION / GRAD-CAM
# ============================================================
st.markdown(
    '<div class="section-title">🔥 Visual Explanation</div>',
    unsafe_allow_html=True
)

gradcam_image = None
try:
    # Find the predicted disease class index from the crop-specific model.
    disease_model, _, disease_class_names = load_disease_model(crop)
    predicted_class_index = (
        disease_class_names.index(disease_name)
        if disease_model is not None and disease_name in disease_class_names
        else None
    )

    if disease_model is not None and predicted_class_index is not None:
        gradcam_image = generate_gradcam(
            disease_model,
            image,
            predicted_class_index
        )
except Exception as e:
    gradcam_image = None

v1, v2 = st.columns(2, gap="large")

with v1:
    st.markdown("**Uploaded Image**")
    st.image(
        image,
        use_container_width=True
    )

with v2:
    st.markdown("**Grad-CAM Heatmap**")
    if gradcam_image is not None:
        st.image(
            gradcam_image,
            use_container_width=True
        )
        st.caption(
            f"Red/yellow regions show areas that contributed more strongly "
            f"to the **{disease_name}** prediction."
        )
    else:
        st.warning("Heatmap could not be generated for this image.")

d1, d2 = st.columns([1.15, 1], gap="large")

with d1:
    st.markdown(
        f"""
        <div class="disease-card">
            <div class="crop-label">IMAGE-BASED DISEASE PREDICTION</div>
            <div class="disease-name">🦠 {disease_name}</div>
            <div class="conf-title">
                Confidence: {disease_confidence * 100:.2f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with d2:
    st.markdown("### Top Disease Predictions")
    for i, (name, prob) in enumerate(disease_predictions, start=1):
        st.markdown(
            f"""
            <div class="pred-row">
                <span class="pred-name">{i}. {name}</span>
                <span class="pred-confidence">{prob * 100:.2f}%</span>
            </div>
            """,
            unsafe_allow_html=True
        )

# ------------------------------------------------------------
# Load ONLY this crop's individual research JSON.
# ------------------------------------------------------------
data, json_path = load_json_for_crop(crop)

if data is None:
    st.warning(f"Research JSON not found for {crop}: {json_path}")
    st.stop()

records = find_disease_records(data)

if not records:
    st.info("No structured disease records were found in this crop JSON.")
    st.stop()

# Match the image-predicted disease to the corresponding research record.
record = find_matching_record(records, disease_name)

if record is None:
    st.warning(
        f"Research record for **{disease_name}** was not found in "
        f"`{json_path.name}`. Showing available diseases below."
    )

    available = [
        first_value(
            r,
            ["disease_name", "Disease Name", "diseaseName"]
        )
        for r in records
    ]
    available = [x for x in available if x]

    if available:
        st.markdown("**Available research diseases:**")
        for name in available:
            st.markdown(f"- {name}")

    st.stop()

selected_name = first_value(
    record,
    ["disease_name", "Disease Name", "diseaseName"]
) or disease_name

disease_type = first_value(
    record,
    ["disease_type", "Disease Type", "type"]
)

causal_agent = first_value(
    record,
    ["causal_agent", "Causal Agent", "causalAgent"]
)

pathogen_type = first_value(
    record,
    ["pathogen_type", "Pathogen Type"]
)

affected_parts = values_from_section(
    record,
    ["affected_plant_parts", "Affected Plant Parts"],
    ["general", "General"]
)

general_symptoms = values_from_section(
    record,
    ["symptoms", "Symptoms"],
    ["general", "General"]
)

early_symptoms = values_from_section(
    record,
    ["symptoms", "Symptoms"],
    ["early_symptoms", "Early Symptoms", "early"]
)

advanced_symptoms = values_from_section(
    record,
    ["symptoms", "Symptoms"],
    ["advanced_symptoms", "Advanced Symptoms", "advanced"]
)

field_identification = values_from_section(
    record,
    ["identification", "Identification"],
    ["field_identification", "Field Identification"]
)

causes = values_from_section(
    record,
    ["causes_and_risk_factors", "Causes And Risk Factors",
     "causes", "Causes"],
    ["general", "General", "other_conditions", "Other Conditions"]
)

spread = values_from_section(
    record,
    ["disease_development_and_spread", "Disease Development And Spread",
     "disease_development", "Disease Development"],
    ["general", "General"]
)

management = values_from_section(
    record,
    ["management", "Management"],
    [
        "integrated_management", "Integrated Management",
        "chemical_management", "Chemical Management",
        "biological_management", "Biological Management",
        "cultural_management", "Cultural Management",
        "farmer_action_points", "Farmer Action Points"
    ]
)

prevention = values_from_section(
    record,
    ["prevention", "Prevention"],
    ["general", "General", "farmer_action_points", "Farmer Action Points"]
)

st.markdown(
    f"""
    <div class="disease-card">
        <div class="crop-label">RESEARCH RECORD MATCH</div>
        <div class="disease-name">📚 {selected_name}</div>
        <div class="muted">
            Source: {json_path.name}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# FIVE USEFUL THINGS
# ============================================================
c1, c2 = st.columns(2, gap="large")

with c1:
    st.markdown("### 1. How did it happen?")
    cause_output = []

    if disease_type:
        cause_output.append(f"Disease type: {disease_type}")

    if causal_agent:
        cause_output.append(f"Caused by: {causal_agent}")

    if pathogen_type:
        cause_output.append(f"Pathogen type: {pathogen_type}")

    cause_output.extend(causes)
    cause_output.extend(spread)

    show_bullets(cause_output)

with c2:
    st.markdown("### 2. Symptoms")

    symptom_output = []
    symptom_output.extend(general_symptoms)

    if affected_parts:
        symptom_output.extend(
            [f"Affected part: {x}" for x in affected_parts]
        )

    show_bullets(symptom_output)

    if early_symptoms:
        st.markdown("**Early symptoms**")
        show_bullets(early_symptoms, limit=5)

    if advanced_symptoms:
        st.markdown("**Advanced symptoms**")
        show_bullets(advanced_symptoms, limit=5)

with c1:
    st.markdown("### 3. How will it progress?")

    progress_output = []
    progress_output.extend(early_symptoms)
    progress_output.extend(advanced_symptoms)
    progress_output.extend(field_identification)

    show_bullets(progress_output)

with c2:
    st.markdown("### 4. How to treat / manage it?")
    show_bullets(management)

st.markdown("### 5. How to prevent it?")
show_bullets(prevention)

st.markdown(
    f"""
    <div class="muted">
        Pipeline: Crop classifier → {crop} disease model →
        {disease_name} → {json_path.name}
    </div>
    """,
    unsafe_allow_html=True
)
