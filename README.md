# Deepfake Detection Web Application

An advanced Flask-based web interface for detecting deepfake images using a hybrid CNN-Transformer model. Upload images through your browser and get real-time predictions about whether they're real or AI-generated/manipulated.

## Features

- 🎯 **Real-time Detection**: Fast inference using pre-trained hybrid CNN-Transformer model
- 🖼️ **Simple Upload Interface**: Easy-to-use image upload with file selection button
- 📊 **Visual Results**: Confidence scores and probability distributions with animated charts
- 📱 **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- 🔒 **Secure**: Input validation, file size limits, and security headers
- ⚡ **Fast**: Model kept in memory for quick predictions (0.5-5 seconds per image)

## Architecture

The application uses:
- **Backend**: Flask web framework with PyTorch for model inference
- **Model**: Hybrid CNN-Transformer (EfficientNet-B0 + Transformer Encoder)
- **Frontend**: Modern HTML5, CSS3, and vanilla JavaScript
- **Device Support**: Automatic GPU/CPU detection

## Prerequisites

- Python 3.8 or higher
- Trained model file: `best_hybrid_model.pth` (from training script)
- CUDA-capable GPU (optional, for faster inference)

## Installation

### 1. Clone or navigate to the project directory

```bash
cd /path/to/your/project
```

### 2. Create a virtual environment (recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Ensure model file exists

Make sure `best_hybrid_model.pth` is in the project root directory. If it's located elsewhere, you can set the `MODEL_PATH` environment variable:

```bash
# Windows
set MODEL_PATH=path\to\your\model.pth

# Linux/Mac
export MODEL_PATH=path/to/your/model.pth
```

## Usage

### Starting the Server

**Development Mode:**
```bash
python app.py
```

**Production Mode (Linux):**
```bash
# Using Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

**Production Mode (Windows):**
```bash
# Using Waitress
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 app:app
```

### Accessing the Application

Once the server is running, open your web browser and navigate to:

```
http://localhost:5000
```

Or if accessing from another device on the same network:

```
http://YOUR_IP_ADDRESS:5000
```

### Using the Web Interface

1. **Upload an Image**:
   - Click "Choose Image File" button to select an image from your computer

2. **Analyze**:
   - Click the "Analyze Image" button
   - Wait for the model to process (typically 0.5-5 seconds)

3. **View Results**:
   - See the prediction (REAL or FAKE)
   - Check the confidence score
   - Review probability distributions
   - View processing time and timestamp

4. **Analyze Another**:
   - Click "Analyze Another Image" to test more images

## Configuration

### Environment Variables

You can configure the application using environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `best_hybrid_model.pth` | Path to the trained model file |
| `HOST` | `0.0.0.0` | Server host address |
| `PORT` | `5000` | Server port |
| `FLASK_ENV` | `production` | Set to `development` for debug mode |

**Example:**
```bash
# Windows
set MODEL_PATH=models\best_model.pth
set PORT=8080
set FLASK_ENV=development
python app.py

# Linux/Mac
export MODEL_PATH=models/best_model.pth
export PORT=8080
export FLASK_ENV=development
python app.py
```

## File Structure

```
project/
├── app.py                      # Flask application and routes
├── model_manager.py            # Model loading and management
├── prediction_service.py       # Prediction logic and validation
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
├── best_hybrid_model.pth       # Trained model (not in repo)
├── static/
│   ├── css/
│   │   └── style.css          # Application styling
│   └── js/
│       └── app.js             # Frontend JavaScript
├── templates/
│   └── index.html             # Main HTML template
└── uploads/                    # Temporary upload folder (auto-created)
```

## API Endpoints

### GET /
Returns the main HTML page.

### POST /predict
Upload an image for prediction.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `image` file

**Response:**
```json
{
  "success": true,
  "prediction": "real",
  "confidence": 0.9542,
  "probabilities": {
    "real": 0.9542,
    "fake": 0.0458
  },
  "processing_time": 1.234,
  "timestamp": "2025-11-08T10:30:45.123456"
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)

**Maximum file size:** 10MB

## Troubleshooting

### Model Not Found Error

**Problem:** Server starts but shows "Model not available" error.

**Solution:**
1. Ensure `best_hybrid_model.pth` exists in the project directory
2. Check file permissions
3. Set `MODEL_PATH` environment variable if model is elsewhere

### CUDA Out of Memory

**Problem:** GPU runs out of memory during inference.

**Solution:**
1. The application will automatically fall back to CPU
2. Close other GPU-intensive applications
3. Use a smaller batch size (already set to 1 for single predictions)

### Slow Predictions

**Problem:** Predictions take too long (>5 seconds).

**Solution:**
1. Check if CUDA is available: Look for "Using device: cuda" in console
2. If using CPU, predictions will be slower (2-5 seconds is normal)
3. Ensure no other heavy processes are running

### Port Already in Use

**Problem:** Error: "Address already in use"

**Solution:**
1. Change the port: `python app.py` with `PORT=8080` environment variable
2. Or kill the process using port 5000:
   - Windows: `netstat -ano | findstr :5000` then `taskkill /PID <PID> /F`
   - Linux/Mac: `lsof -ti:5000 | xargs kill -9`

### File Upload Fails

**Problem:** Image upload returns an error.

**Solution:**
1. Check file size (must be < 10MB)
2. Verify file format (JPG, PNG, BMP, TIFF only)
3. Ensure file is not corrupted
4. Check browser console for JavaScript errors

## Performance

- **Model Loading**: 2-5 seconds (one-time on startup)
- **Prediction Time**: 
  - GPU: 0.5-2 seconds
  - CPU: 2-5 seconds
- **Supported Concurrent Users**: Depends on hardware (typically 5-10 on consumer hardware)

## Security Considerations

- File type validation (whitelist approach)
- File size limits (10MB max)
- Secure filename handling
- Security headers (XSS, clickjacking protection)
- Input sanitization

**For production deployment:**
- Use HTTPS
- Set `FLASK_ENV=production`
- Use a production WSGI server (Gunicorn/Waitress)
- Implement rate limiting
- Add authentication if needed

## Development

### Running in Development Mode

```bash
set FLASK_ENV=development  # Windows
export FLASK_ENV=development  # Linux/Mac
python app.py
```

Development mode enables:
- Auto-reload on code changes
- Detailed error pages
- Debug logging

### Testing

Test the application manually:
1. Start the server
2. Upload various image types
3. Test with different file sizes
4. Try invalid files
5. Test on different browsers and devices

## License

This project is part of a deepfake detection system. Please ensure compliance with your institution's policies and applicable laws when using this software.

## Credits

- **Model Architecture**: Hybrid CNN-Transformer (EfficientNet-B0 + Transformer Encoder)
- **Framework**: Flask, PyTorch
- **Frontend**: HTML5, CSS3, JavaScript

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review console logs for error messages
3. Ensure all prerequisites are met
4. Verify model file integrity

---

**Note**: This application requires a trained model file (`best_hybrid_model.pth`). Train the model using `deepfake.py` before running the web application.
