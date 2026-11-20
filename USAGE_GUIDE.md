# Usage Guide - Deepfake Detection Web Interface

## 🚀 Quick Start

### Step 1: Start the Server
```bash
python app.py
```

You should see:
```
============================================================
🚀 Deepfake Detection Web Application
============================================================
Server: http://0.0.0.0:5000
Mode: Production
Device: cuda
Model: Loaded ✓
============================================================
```

### Step 2: Open Your Browser
Navigate to: **http://localhost:5000**

---

## 📸 Using the Interface

### Upload Section

You'll see:
- 🖼️ Image icon
- **"Select an Image to Analyze"** heading
- File format info: "Supported: JPG, PNG, BMP, TIFF (Max 10MB)"
- **Blue "Choose Image File" button**
- Disabled "Analyze Image" button

### Step-by-Step Process

#### 1️⃣ Select an Image
Click the **"Choose Image File"** button

Your file browser will open

#### 2️⃣ Choose Your Image
Navigate to your image file

Select a JPG, PNG, BMP, or TIFF file (max 10MB)

Click "Open"

#### 3️⃣ Confirm Selection
You'll see:
- ✓ Filename displayed in green box
- File size shown below filename
- "Analyze Image" button now enabled (blue)

#### 4️⃣ Analyze
Click the **"Analyze Image"** button

#### 5️⃣ Wait for Processing
You'll see:
- Loading spinner animation
- "Analyzing Image..." message
- "Please wait while our AI model processes your image"

**Processing time:** 0.5-5 seconds depending on your hardware

#### 6️⃣ View Results
Results display shows:

**Left Side:**
- Your uploaded image

**Right Side:**
- **Prediction Label** (large, color-coded)
  - ✅ REAL (green) or ⚠️ FAKE (red)
- **Confidence Score** (percentage)
- **Probability Bars:**
  - Green bar for REAL probability
  - Red bar for FAKE probability
  - Percentages shown
- **Metadata:**
  - Processing time
  - Timestamp

#### 7️⃣ Analyze Another Image
Click **"Analyze Another Image"** button

Interface resets to upload section

Repeat from step 1

---

## 🎯 Example Workflow

### Scenario: Testing a Suspicious Image

1. **Start:** `python app.py`
2. **Open:** Browser → http://localhost:5000
3. **Click:** "Choose Image File" button
4. **Select:** suspicious_photo.jpg
5. **See:** "✓ suspicious_photo.jpg" (2.3 MB)
6. **Click:** "Analyze Image"
7. **Wait:** ~2 seconds
8. **Result:** 
   - ⚠️ FAKE ⚠️
   - Confidence: 94.52%
   - Real: 5.48% | Fake: 94.52%
   - Processing Time: 1.87s

---

## 💡 Tips for Best Results

### Image Quality
- ✅ Use clear, well-lit images
- ✅ Higher resolution is better
- ✅ Avoid heavily compressed images
- ❌ Don't use blurry or dark images

### File Size
- ✅ Under 10MB (enforced)
- ✅ 1-5MB is optimal
- ⚠️ Very small files may have less detail

### Interpreting Results

#### High Confidence (>90%)
**Very Reliable** - Trust this prediction
- Example: 95% FAKE → Likely manipulated
- Example: 97% REAL → Likely authentic

#### Medium Confidence (70-90%)
**Fairly Reliable** - Good indication
- Example: 85% REAL → Probably authentic
- Example: 78% FAKE → Likely manipulated

#### Low Confidence (<70%)
**Less Certain** - Review carefully
- Example: 65% REAL → Uncertain
- Example: 60% FAKE → Uncertain
- Consider testing with different image

---

## ⚠️ Error Messages

### "Invalid file type"
**Problem:** Wrong file format
**Solution:** Use JPG, PNG, BMP, or TIFF only

### "File is too large"
**Problem:** File exceeds 10MB
**Solution:** Compress or resize the image

### "Model not available"
**Problem:** Model file missing
**Solution:** Ensure `best_hybrid_model.pth` exists

### "Unable to process image"
**Problem:** Corrupted or invalid image
**Solution:** Try a different image file

### "Network error"
**Problem:** Server connection issue
**Solution:** Check if server is running

---

## 🔧 Troubleshooting

### Server Won't Start

**Check:**
1. Is Python installed? `python --version`
2. Are dependencies installed? `pip install -r requirements.txt`
3. Does model file exist? Check for `best_hybrid_model.pth`
4. Is port 5000 available?

**Fix:**
```bash
# Change port if needed
set PORT=8080  # Windows
export PORT=8080  # Linux/Mac
python app.py
```

### Slow Predictions

**Normal Times:**
- GPU: 0.5-2 seconds ✅
- CPU: 2-5 seconds ✅

**If slower:**
1. Check console for "Using device: cuda" or "cpu"
2. Close other applications
3. Ensure model is loaded (check startup logs)

### Button Not Working

**Try:**
1. Refresh the page (F5)
2. Clear browser cache
3. Try a different browser
4. Check browser console for errors (F12)

---

## 📊 Understanding the Results

### Prediction Label
- **✅ REAL**: Image appears authentic
- **⚠️ FAKE**: Image appears manipulated/AI-generated

### Confidence Score
- Shows how certain the model is
- Higher = more confident
- Displayed as percentage (0-100%)

### Probability Bars
- **Green (REAL)**: Likelihood image is authentic
- **Red (FAKE)**: Likelihood image is manipulated
- Both add up to 100%

### Processing Time
- How long analysis took
- Shown in seconds (s) or milliseconds (ms)
- Varies by hardware and image size

### Timestamp
- When the analysis was performed
- Useful for record keeping

---

## 🎓 Advanced Usage

### Testing Multiple Images

**Efficient Workflow:**
1. Prepare images in a folder
2. Start server once
3. For each image:
   - Upload
   - Analyze
   - Note results
   - Click "Analyze Another"
4. Model stays loaded = fast predictions

### Batch Testing Tips
- Keep browser tab open
- Model stays in memory
- Subsequent predictions are faster
- No need to restart server

### Recording Results
- Take screenshots of results
- Note confidence scores
- Compare similar images
- Track patterns

---

## 🌐 Accessing from Other Devices

### Same Network Access

**On Server Computer:**
1. Find your IP address:
   - Windows: `ipconfig`
   - Linux/Mac: `ifconfig`
2. Note the IP (e.g., 192.168.1.100)

**On Other Device:**
1. Connect to same WiFi network
2. Open browser
3. Go to: `http://192.168.1.100:5000`
4. Use interface normally

### Mobile Access
- Works on phones and tablets
- Interface is responsive
- Touch-friendly buttons
- Same functionality

---

## 📱 Mobile Usage

### On Smartphone/Tablet

**Interface Adapts:**
- Single column layout
- Larger touch targets
- Simplified display
- Same features

**Upload on Mobile:**
1. Tap "Choose Image File"
2. Select from:
   - Camera (take photo)
   - Gallery (existing photo)
   - Files (any image)
3. Tap "Analyze Image"
4. View results

---

## ✅ Best Practices

### Do's
✅ Use high-quality images
✅ Check confidence scores
✅ Test multiple similar images
✅ Keep server running for multiple tests
✅ Note unusual results

### Don'ts
❌ Don't upload sensitive personal images
❌ Don't rely solely on low-confidence results
❌ Don't use extremely small images
❌ Don't test with corrupted files
❌ Don't share results without context

---

## 🎉 You're Ready!

The interface is simple and intuitive:
1. **Click** button
2. **Select** image
3. **Analyze** and view results
4. **Repeat** as needed

**Happy deepfake detecting!** 🔍
