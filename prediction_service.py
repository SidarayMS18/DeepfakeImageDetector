import torch
import torch.nn.functional as F
from PIL import Image
import io
import time
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional
from werkzeug.datastructures import FileStorage

logger = logging.getLogger(__name__)


class PredictionService:
    """Service for handling image predictions"""
    
    # Maximum file size: 10MB
    MAX_IMAGE_SIZE = 10 * 1024 * 1024
    
    # Allowed image extensions
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    def __init__(self, model_manager):
        """
        Initialize the PredictionService
        
        Args:
            model_manager: ModelManager instance with loaded model
        """
        self.model_manager = model_manager
        logger.info("PredictionService initialized")
    
    def validate_image(self, file: FileStorage) -> Tuple[bool, str]:
        """
        Validate the uploaded image file
        
        Args:
            file: The uploaded file from Flask request
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        # Check if file exists
        if not file or file.filename == '':
            return False, "No file provided"
        
        # Check file extension
        filename = file.filename.lower()
        file_ext = None
        for ext in self.ALLOWED_EXTENSIONS:
            if filename.endswith(ext):
                file_ext = ext
                break
        
        if not file_ext:
            allowed = ', '.join(sorted(self.ALLOWED_EXTENSIONS))
            return False, f"Invalid file type. Allowed types: {allowed}"
        
        # Check file size
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > self.MAX_IMAGE_SIZE:
            max_size_mb = self.MAX_IMAGE_SIZE / (1024 * 1024)
            return False, f"File too large. Maximum size is {max_size_mb}MB"
        
        if file_size == 0:
            return False, "File is empty"
        
        return True, ""
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess the image for model inference
        
        Args:
            image: PIL Image object
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Apply transformations (resize, normalize, etc.)
        image_tensor = self.model_manager.transform(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        # Move to device
        image_tensor = image_tensor.to(self.model_manager.device)
        
        return image_tensor
    
    def format_results(self, output: torch.Tensor, processing_time: float) -> Dict:
        """
        Format the model output into a structured result dictionary
        
        Args:
            output: Raw model output tensor
            processing_time: Time taken for processing in seconds
            
        Returns:
            Dict: Formatted prediction results
        """
        # Apply softmax to get probabilities
        probabilities = F.softmax(output, dim=1)
        
        # Get prediction and confidence
        confidence, prediction_idx = torch.max(probabilities, 1)
        
        # Extract values
        predicted_class = self.model_manager.class_names[prediction_idx.item()]
        confidence_score = confidence.item()
        
        # Get individual probabilities
        probs = probabilities[0].cpu().detach().numpy()
        fake_prob = float(probs[0])
        real_prob = float(probs[1])
        
        # Create result dictionary
        result = {
            'success': True,
            'prediction': predicted_class,
            'confidence': round(confidence_score, 4),
            'probabilities': {
                'real': round(real_prob, 4),
                'fake': round(fake_prob, 4)
            },
            'processing_time': round(processing_time, 3),
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def predict(self, file: FileStorage) -> Dict:
        """
        Main prediction method that orchestrates the entire prediction pipeline
        
        Args:
            file: The uploaded image file
            
        Returns:
            Dict: Prediction results or error information
        """
        start_time = time.time()
        
        try:
            # Step 1: Validate the file
            is_valid, error_message = self.validate_image(file)
            if not is_valid:
                logger.warning(f"Validation failed: {error_message}")
                return {
                    'success': False,
                    'error': error_message
                }
            
            # Step 2: Load and verify the image
            try:
                image_bytes = file.read()
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                logger.info(f"Image loaded: {image.size}")
            except Exception as e:
                logger.error(f"Error loading image: {str(e)}")
                return {
                    'success': False,
                    'error': 'Unable to process image. File may be corrupted.'
                }
            
            # Step 3: Preprocess the image
            try:
                image_tensor = self.preprocess_image(image)
            except Exception as e:
                logger.error(f"Error preprocessing image: {str(e)}")
                return {
                    'success': False,
                    'error': 'Error preprocessing image.'
                }
            
            # Step 4: Run model inference
            try:
                with torch.no_grad():
                    output = self.model_manager.model(image_tensor)
                logger.info("Inference completed successfully")
            except Exception as e:
                logger.error(f"Error during inference: {str(e)}")
                return {
                    'success': False,
                    'error': 'Error during model inference.'
                }
            
            # Step 5: Format and return results
            processing_time = time.time() - start_time
            result = self.format_results(output, processing_time)
            
            logger.info(
                f"Prediction: {result['prediction']} "
                f"(confidence: {result['confidence']:.2%}, "
                f"time: {result['processing_time']:.2f}s)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Unexpected error in predict(): {str(e)}")
            return {
                'success': False,
                'error': 'An unexpected error occurred during prediction.'
            }
