# Quick Start Guide - Deepfake Detection Web App

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Start the Server
```bash
python app.py
```

### Step 3: Open Your Browser
Navigate to: **http://localhost:5000**

---

## 📋 What You'll See

1. **Upload Interface** - Click button to select an image
2. **Analysis** - Click "Analyze Image" button
3. **Results** - View prediction, confidence score, and probabilities

---

## ✅ Supported Files
- JPG/JPEG
- PNG
- BMP
- TIFF
- Max size: 10MB

---

## 🔧 Troubleshooting

### Model Not Found?
Make sure `best_hybrid_model.pth` is in the project directory.

### Port Already in Use?
Change the port:
```bash
# Windows
set PORT=8080
python app.py

# Linux/Mac
export PORT=8080
python app.py
```

### Slow Predictions?
- GPU: 0.5-2 seconds (normal)
- CPU: 2-5 seconds (normal)

---

## 📖 Full Documentation
See [README.md](README.md) for complete documentation.

---

## 🎯 Example Usage

1. Start server: `python app.py`
2. Open browser: `http://localhost:5000`
3. Upload a test image
4. View results instantly!

**That's it! You're ready to detect deepfakes! 🔍**
