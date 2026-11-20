import torch
import torch.nn as nn
from torchvision import transforms, models
import os
import logging

logger = logging.getLogger(__name__)


class HybridModel(nn.Module):
    """Hybrid CNN-Transformer model for deepfake detection"""
    
    def __init__(self, num_classes=2, d_model=1280, nhead=8, num_encoder_layers=3, 
                 dim_feedforward=1024, dropout=0.1):
        """
        Args:
            num_classes (int): Number of output classes (2 for real/fake).
            d_model (int): The embedding dimension required by the Transformer.
                           Must match the CNN backbone's output channels.
                           EfficientNet-B0 outputs 1280 channels.
            nhead (int): Number of attention heads in the Transformer.
            num_encoder_layers (int): Number of Transformer encoder layers.
            dim_feedforward (int): Hidden dimension of the Transformer's feedforward network.
            dropout (float): Dropout rate.
        """
        super(HybridModel, self).__init__()
        
        # CNN Backbone - EfficientNet-B0
        self.cnn_backbone = models.efficientnet_b0(weights=None).features
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        # Classification Head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        """Forward pass through the model"""
        # 1. Pass through CNN backbone
        x = self.cnn_backbone(x)
        
        # 2. Prepare for Transformer
        b, c, h, w = x.shape
        x = x.flatten(2)  # Flatten spatial dimensions
        x = x.permute(2, 0, 1)  # (seq_len, batch, channels)
        
        # 3. Pass through Transformer Encoder
        x = self.transformer_encoder(x)
        
        # 4. Global Pooling
        x = x.permute(1, 2, 0)  # (batch, channels, seq_len)
        x = self.global_pool(x)
        x = x.squeeze(-1)  # (batch, channels)

        # 5. Pass through Classifier
        x = self.classifier(x)
        return x


class ModelManager:
    """Manages model loading, device selection, and transformations"""
    
    def __init__(self, model_path='best_hybrid_model.pth'):
        """
        Initialize the ModelManager
        
        Args:
            model_path (str): Path to the trained model weights file
        """
        self.model_path = model_path
        self.device = self.get_device()
        self.class_names = ['fake', 'real']
        
        # Define image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load the model
        self.model = self.load_model()
        
        logger.info(f"ModelManager initialized with device: {self.device}")
    
    def get_device(self):
        """
        Automatically detect and return the appropriate device (CUDA or CPU)
        
        Returns:
            torch.device: The device to use for inference
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("CUDA is available. Using GPU for inference.")
        else:
            device = torch.device("cpu")
            logger.info("CUDA not available. Using CPU for inference.")
        
        return device
    
    def load_model(self):
        """
        Load the trained model from disk
        
        Returns:
            HybridModel: The loaded model in evaluation mode
            
        Raises:
            FileNotFoundError: If the model file doesn't exist
            RuntimeError: If there's an error loading the model
        """
        if not os.path.exists(self.model_path):
            error_msg = (
                f"Model file not found at '{self.model_path}'. "
                "Please ensure the model has been trained and the file exists."
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            # Initialize model architecture
            model = HybridModel(num_classes=2)
            
            # Load weights
            logger.info(f"Loading model from {self.model_path}...")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle both state_dict and full model save formats
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            # Move to device and set to evaluation mode
            model = model.to(self.device)
            model.eval()
            
            logger.info("Model loaded successfully!")
            return model
            
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def is_model_loaded(self):
        """
        Check if the model is loaded and ready
        
        Returns:
            bool: True if model is loaded, False otherwise
        """
        return self.model is not None
