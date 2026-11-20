# Design Document - Flask Web Interface for Deepfake Detection

## Overview

The Flask web interface will provide a modern, user-friendly web application for deepfake detection. The system follows a client-server architecture where the frontend (HTML/CSS/JavaScript) communicates with a Flask backend that interfaces with the pre-trained PyTorch model. The design emphasizes real-time feedback, responsive UI, and efficient model inference.

### Key Design Principles

- **Simplicity**: Clean, intuitive interface requiring minimal user interaction
- **Performance**: Fast inference with model kept in memory, asynchronous processing
- **Responsiveness**: Mobile-first design that works across all devices
- **Reliability**: Comprehensive error handling and validation
- **Scalability**: Stateless design supporting multiple concurrent users

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Web Browser                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Frontend (HTML/CSS/JavaScript)             │  │
│  │  - Upload Interface                                  │  │
│  │  - Result Display                                    │  │
│  │  - AJAX Communication                                │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ HTTP/HTTPS
                            │ (POST /predict, GET /)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Flask Application                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                  Route Handlers                      │  │
│  │  - index() → Serve HTML                             │  │
│  │  - predict() → Handle predictions                   │  │
│  │  - health() → Health check                          │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Prediction Service                      │  │
│  │  - Image validation                                  │  │
│  │  - Image preprocessing                               │  │
│  │  - Model inference                                   │  │
│  │  - Result formatting                                 │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │               Model Manager                          │  │
│  │  - Load model on startup                            │  │
│  │  - Maintain model in memory                         │  │
│  │  - Device management (CPU/GPU)                      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              PyTorch Model (HybridModel)                    │
│  - EfficientNet-B0 CNN Backbone                            │
│  - Transformer Encoder                                      │
│  - Classification Head                                      │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

**Backend:**
- Flask 2.3+ (Web framework)
- PyTorch 2.0+ (Deep learning)
- Pillow (PIL) 10.0+ (Image processing)
- Werkzeug (File handling, security)

**Frontend:**
- HTML5 (Structure)
- CSS3 with Flexbox/Grid (Styling)
- Vanilla JavaScript (Interactivity, AJAX)
- No heavy frameworks (lightweight, fast loading)

**Deployment:**
- Python 3.8+
- Compatible with Windows/Linux
- Development server for testing
- Production-ready with Gunicorn/Waitress

## Components and Interfaces

### 1. Flask Application (app.py)

**Purpose**: Main application entry point, route definitions, and server configuration

**Key Components:**

```python
class FlaskApp:
    - app: Flask instance
    - model_manager: ModelManager instance
    - config: Application configuration
    
    Methods:
    - __init__(): Initialize Flask app and load model
    - configure_routes(): Set up URL routes
    - run(): Start the development server
```

**Routes:**

| Route | Method | Purpose | Response |
|-------|--------|---------|----------|
| `/` | GET | Serve main HTML page | HTML template |
| `/predict` | POST | Handle image upload and prediction | JSON with results |
| `/health` | GET | Health check endpoint | JSON status |
| `/static/<path>` | GET | Serve static files (CSS, JS, images) | Static file |

### 2. Model Manager (model_manager.py)

**Purpose**: Handle model loading, device management, and maintain model state

**Class Structure:**

```python
class ModelManager:
    Attributes:
    - device: torch.device (CPU or CUDA)
    - model: HybridModel instance
    - class_names: List[str] = ['fake', 'real']
    - transform: torchvision.transforms.Compose
    - model_path: str = 'best_hybrid_model.pth'
    
    Methods:
    - __init__(model_path: str)
    - load_model() -> HybridModel
    - get_device() -> torch.device
    - is_model_loaded() -> bool
```

**Model Loading Strategy:**
- Load model once during application startup
- Keep model in memory for fast inference
- Use singleton pattern to ensure single model instance
- Automatic device detection (CUDA if available, else CPU)

### 3. Prediction Service (prediction_service.py)

**Purpose**: Handle image preprocessing, model inference, and result formatting

**Class Structure:**

```python
class PredictionService:
    Attributes:
    - model_manager: ModelManager
    - max_image_size: int = 10 * 1024 * 1024  # 10MB
    - allowed_extensions: Set[str] = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    Methods:
    - validate_image(file) -> Tuple[bool, str]
    - preprocess_image(image: PIL.Image) -> torch.Tensor
    - predict(image_file) -> Dict
    - format_results(output: torch.Tensor) -> Dict
```

**Prediction Flow:**

```
1. Receive uploaded file
2. Validate file type and size
3. Load image with PIL
4. Apply transformations:
   - Resize to 224x224
   - Convert to tensor
   - Normalize with ImageNet stats
5. Add batch dimension
6. Move to device (CPU/GPU)
7. Run model inference
8. Apply softmax for probabilities
9. Extract prediction and confidence
10. Format and return results
```

### 4. Frontend Interface (templates/index.html)

**Purpose**: Provide user interface for image upload and result display

**Layout Structure:**

```html
<body>
  <header>
    <h1>Deepfake Detection System</h1>
    <p>Upload an image to check if it's real or AI-generated</p>
  </header>
  
  <main>
    <section id="upload-section">
      <div id="drop-zone">
        <!-- Drag & drop area -->
      </div>
      <input type="file" id="file-input" accept="image/*">
      <button id="upload-btn">Analyze Image</button>
    </section>
    
    <section id="loading-section" style="display:none">
      <!-- Loading spinner and message -->
    </section>
    
    <section id="results-section" style="display:none">
      <div id="image-preview">
        <!-- Display uploaded image -->
      </div>
      
      <div id="prediction-results">
        <h2 id="prediction-label"></h2>
        <p id="confidence-score"></p>
        
        <div id="probability-bars">
          <div class="prob-bar">
            <label>REAL</label>
            <div class="bar-container">
              <div class="bar real-bar"></div>
              <span class="percentage"></span>
            </div>
          </div>
          
          <div class="prob-bar">
            <label>FAKE</label>
            <div class="bar-container">
              <div class="bar fake-bar"></div>
              <span class="percentage"></span>
            </div>
          </div>
        </div>
        
        <div id="metadata">
          <p>Processing Time: <span id="proc-time"></span></p>
          <p>Timestamp: <span id="timestamp"></span></p>
        </div>
      </div>
      
      <button id="analyze-another">Analyze Another Image</button>
    </section>
    
    <section id="error-section" style="display:none">
      <div class="error-message"></div>
      <button id="try-again">Try Again</button>
    </section>
  </main>
  
  <footer>
    <p>Powered by Hybrid CNN-Transformer Model</p>
  </footer>
</body>
```

### 5. Frontend JavaScript (static/js/app.js)

**Purpose**: Handle user interactions, AJAX requests, and dynamic UI updates

**Key Functions:**

```javascript
// File handling
function handleFileSelect(event)
function handleDragOver(event)
function handleDrop(event)
function validateFile(file)

// API communication
async function uploadAndPredict(file)
function sendPredictionRequest(formData)

// UI updates
function showLoading()
function hideLoading()
function displayResults(data)
function displayError(message)
function updateProbabilityBars(realProb, fakeProb)
function resetInterface()

// Utility
function formatTimestamp(date)
function formatProcessingTime(ms)
```

**AJAX Request Format:**

```javascript
// Request
POST /predict
Content-Type: multipart/form-data
Body: {
  image: <File>
}

// Response (Success)
{
  "success": true,
  "prediction": "real" | "fake",
  "confidence": 0.95,
  "probabilities": {
    "real": 0.95,
    "fake": 0.05
  },
  "processing_time": 1.23,
  "timestamp": "2025-11-08T10:30:45"
}

// Response (Error)
{
  "success": false,
  "error": "Error message"
}
```

### 6. Styling (static/css/style.css)

**Purpose**: Provide modern, responsive styling

**Design System:**

```css
/* Color Palette */
--primary-bg: #1a1a2e
--secondary-bg: #16213e
--accent-color: #0f3460
--success-color: #27ae60
--danger-color: #e74c3c
--text-primary: #eaeaea
--text-secondary: #a0a0a0

/* Typography */
--font-primary: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif
--font-size-base: 16px
--font-size-large: 24px
--font-size-xlarge: 32px

/* Spacing */
--spacing-xs: 8px
--spacing-sm: 16px
--spacing-md: 24px
--spacing-lg: 32px
--spacing-xl: 48px

/* Breakpoints */
--mobile: 320px
--tablet: 768px
--desktop: 1024px
--wide: 1440px
```

**Responsive Design Strategy:**

```css
/* Mobile First Approach */
/* Base styles for mobile (320px+) */

@media (min-width: 768px) {
  /* Tablet adjustments */
}

@media (min-width: 1024px) {
  /* Desktop adjustments */
}

@media (min-width: 1440px) {
  /* Wide screen adjustments */
}
```

## Data Models

### Image Upload Data

```python
class ImageUpload:
    """Represents an uploaded image file"""
    filename: str
    content_type: str
    size: int  # bytes
    data: bytes
```

### Prediction Result

```python
class PredictionResult:
    """Represents a prediction result"""
    success: bool
    prediction: str  # 'real' or 'fake'
    confidence: float  # 0.0 to 1.0
    probabilities: Dict[str, float]  # {'real': 0.95, 'fake': 0.05}
    processing_time: float  # seconds
    timestamp: str  # ISO format
    error: Optional[str] = None
```

### Application Configuration

```python
class Config:
    """Application configuration"""
    # Flask settings
    SECRET_KEY: str = os.urandom(24)
    MAX_CONTENT_LENGTH: int = 10 * 1024 * 1024  # 10MB
    UPLOAD_FOLDER: str = 'uploads'
    
    # Model settings
    MODEL_PATH: str = 'best_hybrid_model.pth'
    IMG_SIZE: int = 224
    
    # Server settings
    HOST: str = '0.0.0.0'
    PORT: int = 5000
    DEBUG: bool = False
```

## Error Handling

### Error Categories and Responses

| Error Type | HTTP Status | User Message | Log Level |
|------------|-------------|--------------|-----------|
| Model not found | 500 | "Model not available. Please train the model first." | ERROR |
| Invalid file type | 400 | "Please upload a valid image file (JPG, PNG, BMP, TIFF)" | WARNING |
| File too large | 413 | "Image file is too large. Maximum size is 10MB." | WARNING |
| Corrupted image | 400 | "Unable to process image. File may be corrupted." | WARNING |
| Processing error | 500 | "An error occurred during analysis. Please try again." | ERROR |
| Out of memory | 500 | "Server is busy. Please try again in a moment." | ERROR |

### Error Handling Strategy

**Backend:**

```python
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validation
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        # Process
        result = prediction_service.predict(file)
        return jsonify(result), 200
        
    except ValueError as e:
        # Validation errors
        return jsonify({'success': False, 'error': str(e)}), 400
    except FileNotFoundError as e:
        # Model not found
        return jsonify({'success': False, 'error': 'Model not available'}), 500
    except Exception as e:
        # Unexpected errors
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'success': False, 'error': 'Processing failed'}), 500
```

**Frontend:**

```javascript
async function uploadAndPredict(file) {
    try {
        showLoading();
        const response = await sendPredictionRequest(file);
        
        if (response.success) {
            displayResults(response);
        } else {
            displayError(response.error);
        }
    } catch (error) {
        displayError('Network error. Please check your connection.');
    } finally {
        hideLoading();
    }
}
```

## Testing Strategy

### Unit Tests

**Backend Tests (test_app.py):**

```python
class TestModelManager:
    - test_model_loading()
    - test_device_detection()
    - test_model_not_found()

class TestPredictionService:
    - test_image_validation()
    - test_file_size_limit()
    - test_invalid_file_type()
    - test_image_preprocessing()
    - test_prediction_format()

class TestFlaskRoutes:
    - test_index_route()
    - test_predict_route_success()
    - test_predict_route_no_file()
    - test_predict_route_invalid_file()
    - test_health_check()
```

**Frontend Tests (test_frontend.js):**

```javascript
describe('File Validation', () => {
    test('accepts valid image files')
    test('rejects invalid file types')
    test('rejects oversized files')
})

describe('UI Updates', () => {
    test('displays loading state')
    test('displays results correctly')
    test('displays errors correctly')
    test('resets interface')
})

describe('API Communication', () => {
    test('sends correct request format')
    test('handles successful response')
    test('handles error response')
    test('handles network errors')
})
```

### Integration Tests

```python
class TestEndToEnd:
    - test_upload_real_image()
    - test_upload_fake_image()
    - test_multiple_predictions()
    - test_concurrent_requests()
```

### Manual Testing Checklist

- [ ] Upload various image formats (JPG, PNG, BMP, TIFF)
- [ ] Test with real and fake images
- [ ] Test file size limits
- [ ] Test invalid file types
- [ ] Test drag and drop functionality
- [ ] Test on different browsers (Chrome, Firefox, Safari, Edge)
- [ ] Test on different devices (Desktop, Tablet, Mobile)
- [ ] Test with slow network connection
- [ ] Test error scenarios
- [ ] Test multiple consecutive uploads

## Performance Considerations

### Optimization Strategies

1. **Model Loading**
   - Load model once at startup
   - Keep in memory for fast inference
   - Use model.eval() to disable training mode

2. **Image Processing**
   - Resize images efficiently with PIL
   - Use batch size of 1 for single predictions
   - Optimize tensor operations

3. **Frontend Performance**
   - Minimize JavaScript bundle size
   - Use CSS animations for smooth transitions
   - Lazy load images
   - Compress static assets

4. **Caching**
   - Cache static files with appropriate headers
   - Use browser caching for CSS/JS

5. **Concurrent Requests**
   - Flask handles requests sequentially by default
   - For production, use Gunicorn with multiple workers
   - Implement request queuing if needed

### Expected Performance Metrics

- **Model Loading Time**: 2-5 seconds (one-time on startup)
- **Prediction Time**: 0.5-2 seconds per image (GPU), 2-5 seconds (CPU)
- **Page Load Time**: < 1 second
- **Upload Time**: Depends on image size and network (typically < 1 second for 2MB image)
- **Total User Wait Time**: 1-6 seconds from upload to results

## Security Considerations

### Input Validation

- Validate file extensions (whitelist approach)
- Check file MIME types
- Limit file size (10MB max)
- Sanitize filenames
- Verify image integrity with PIL

### File Handling

- Store uploads in temporary directory
- Clean up files after processing
- Use secure filename generation
- Prevent directory traversal attacks

### Flask Security

```python
# Security headers
@app.after_request
def set_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

# CSRF protection (if using forms)
app.config['SECRET_KEY'] = os.urandom(24)

# Limit request size
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
```

### Production Deployment

- Use HTTPS in production
- Set DEBUG=False
- Use environment variables for secrets
- Implement rate limiting
- Add authentication if needed
- Use production WSGI server (Gunicorn/Waitress)

## Deployment Guide

### Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
python app.py

# Access at http://localhost:5000
```

### Production Deployment

**Using Gunicorn (Linux):**

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

**Using Waitress (Windows):**

```bash
waitress-serve --host=0.0.0.0 --port=5000 app:app
```

### Environment Variables

```bash
# Optional configuration
export FLASK_ENV=production
export MODEL_PATH=best_hybrid_model.pth
export PORT=5000
export HOST=0.0.0.0
```

### Docker Deployment (Optional)

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## Future Enhancements

### Potential Features

1. **Batch Processing**: Upload and analyze multiple images at once
2. **History**: Save and view previous predictions
3. **API Authentication**: Add API keys for programmatic access
4. **Advanced Visualizations**: Show attention maps, feature visualizations
5. **Video Support**: Extend to video deepfake detection
6. **Model Comparison**: Compare results from multiple models
7. **Export Results**: Download results as PDF/CSV
8. **Real-time Webcam**: Analyze images from webcam
9. **Confidence Threshold**: Allow users to set custom thresholds
10. **Explainability**: Add LIME/SHAP explanations for predictions

### Scalability Considerations

- Implement Redis for caching
- Use message queue (Celery) for async processing
- Deploy with load balancer for multiple instances
- Use cloud storage for uploaded files
- Implement database for logging predictions
- Add monitoring and analytics (Prometheus, Grafana)
