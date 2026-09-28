# KriShield 🌾🛡️

**KriShield** is an AI-powered Crop Classification & Plant Disease Detection platform designed to help farmers and agronomists diagnose crop diseases early with deep learning and visual interpretability (Grad-CAM heatmaps).

---

## 🌟 Key Features

- **Multi-Crop Classification**: Supports 15 crop varieties (Arhar, Bajra, Cotton, Cucumber, Fodder, Maize, Masoor, Matar, Moongfali, Potato, Rice, Soybean, Urad, Watermelon, Wheat).
- **Disease Identification**: Dedicated CNN models for each crop to diagnose specific diseases accurately.
- **Grad-CAM Visual Heatmaps**: Explainable AI visualization highlighting regions of interest on infected leaves.
- **Agronomy Knowledge Base**: Comprehensive remedies, chemical treatments, and organic prevention advice extracted from agricultural research.
- **Modern Interactive Dashboard**: Clean Streamlit web interface with real-time inference and analysis.

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/gogulgupta/KriShield.git
cd KriShield
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run app_heatmap.py
```

---

## 📁 Repository Structure

```
├── app_heatmap.py              # Main Streamlit web application
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules (datasets & caches excluded)
├── disease_models/             # Pretrained crop disease models & metadata
│   ├── Arhar/
│   ├── Cotton/
│   ├── Rice/
│   ├── Wheat/
│   └── ...
├── training_output/            # Primary crop classifier model & class mappings
│   └── models/
│       └── best_crop_model.pth
└── Crop_Papers/                # Extracted agronomy disease knowledge & remedies
    └── extracted_results/
        ├── wheat.json
        ├── rice.json
        └── ...
```

---

## ☁️ Deployment Guide

### Recommended Platforms (Streamlit + PyTorch):
1. **Streamlit Community Cloud** (Recommended - Free & 1-Click):
   - Link your GitHub repo `gogulgupta/KriShield`
   - Main file path: `app_heatmap.py`
   - Deploy instantly!

2. **Hugging Face Spaces**:
   - Create a new Space with SDK: `Streamlit`
   - Connect repository for fast cloud inference.

3. **Render / Railway / Docker**:
   - Deploy as a containerized web service.

---

## 📄 License
Open source for educational and agricultural research purposes.
