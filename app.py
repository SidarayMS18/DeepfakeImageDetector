from flask import Flask, render_template, request, jsonify
import os
import logging
from werkzeug.utils import secure_filename
from model_manager import ModelManager
from prediction_service import PredictionService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_PATH'] = os.getenv('MODEL_PATH', 'best_hybrid_model.pth')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize model and prediction service
try:
    logger.info("="*60)
    logger.info("Initializing Deepfake Detection Web Application")
    logger.info("="*60)
    
    model_manager = ModelManager(model_path=app.config['MODEL_PATH'])
    prediction_service = PredictionService(model_manager)
    
    logger.info("Application initialized successfully!")
    logger.info(f"Device: {model_manager.device}")
    logger.info(f"Model loaded from: {app.config['MODEL_PATH']}")
    logger.info("="*60)
    
except FileNotFoundError as e:
    logger.error("="*60)
    logger.error("INITIALIZATION FAILED")
    logger.error("="*60)
    logger.error(str(e))
    logger.error("Please ensure 'best_hybrid_model.pth' exists in the current directory")
    logger.error("or set the MODEL_PATH environment variable to the correct path.")
    logger.error("="*60)
    model_manager = None
    prediction_service = None
except Exception as e:
    logger.error("="*60)
    logger.error("INITIALIZATION FAILED")
    logger.error("="*60)
    logger.error(f"Unexpected error: {str(e)}")
    logger.error("="*60)
    model_manager = None
    prediction_service = None


@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle image upload and prediction
    
    Expected: multipart/form-data with 'image' file
    Returns: JSON with prediction results or error
    """
    # Check if model is loaded
    if prediction_service is None:
        return jsonify({
            'success': False,
            'error': 'Model not available. Please ensure the model file exists and restart the server.'
        }), 500
    
    # Check if image file is in request
    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No image file provided'
        }), 400
    
    file = request.files['image']
    
    # Check if file was selected
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected'
        }), 400
    
    try:
        # Run prediction
        result = prediction_service.predict(file)
        
        # Return appropriate status code
        if result['success']:
            return jsonify(result), 200
        else:
            # Determine status code based on error type
            error_msg = result.get('error', '').lower()
            if 'invalid' in error_msg or 'file type' in error_msg:
                status_code = 400
            elif 'too large' in error_msg:
                status_code = 413
            elif 'corrupted' in error_msg:
                status_code = 400
            else:
                status_code = 500
            
            return jsonify(result), status_code
            
    except Exception as e:
        logger.error(f"Error in predict route: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'An error occurred during prediction. Please try again.'
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    if model_manager and model_manager.is_model_loaded():
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'device': str(model_manager.device)
        }), 200
    else:
        return jsonify({
            'status': 'unhealthy',
            'model_loaded': False,
            'error': 'Model not loaded'
        }), 503


@app.after_request
def set_security_headers(response):
    """Add security headers to all responses"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': 'File is too large. Maximum size is 10MB.'
    }), 413


@app.errorhandler(400)
def bad_request(error):
    """Handle bad request error"""
    return jsonify({
        'success': False,
        'error': 'Bad request. Please check your input.'
    }), 400


@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server error"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'success': False,
        'error': 'An internal server error occurred. Please try again later.'
    }), 500


if __name__ == '__main__':
    # Get configuration from environment variables
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV', 'production') == 'development'
    
    # Display startup information
    print("\n" + "="*60)
    print("🚀 Deepfake Detection Web Application")
    print("="*60)
    print(f"Server: http://{host}:{port}")
    print(f"Mode: {'Development' if debug else 'Production'}")
    
    if model_manager:
        print(f"Device: {model_manager.device}")
        print(f"Model: Loaded ✓")
    else:
        print("Model: Not Loaded ✗")
        print("⚠️  Warning: Application will not function without a model")
    
    print("="*60)
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    # Run the Flask app
    app.run(host=host, port=port, debug=debug)
