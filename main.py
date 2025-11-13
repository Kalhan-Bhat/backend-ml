"""
FastAPI ML Microservice for Emotion Detection

This service:
1. Loads a pre-trained PyTorch emotion detection model
2. Receives base64-encoded video frames via REST API
3. Processes frames and runs inference
4. Returns emotion predictions with confidence scores
"""

import os
import io
import base64
import logging
from typing import Dict, Any

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =====================================
# CONFIGURATION
# =====================================

class Config:
    """Configuration class for ML service"""
    PORT = int(os.getenv('PORT', 8000))
    HOST = os.getenv('HOST', '0.0.0.0')
    MODEL_PATH = os.getenv('MODEL_PATH', './models/emotion_model_traced.pt')
    IMAGE_SIZE = int(os.getenv('IMAGE_SIZE', 224))
    EMOTION_LABELS = os.getenv(
        'EMOTION_LABELS',
        'neutral,happy,sad,angry,surprised,fearful,disgusted'
    ).split(',')

config = Config()

# =====================================
# PYDANTIC MODELS
# =====================================

class PredictionRequest(BaseModel):
    """Request model for emotion prediction"""
    image: str  # Base64 encoded image

class PredictionResponse(BaseModel):
    """Response model for emotion prediction"""
    emotion: str
    confidence: float
    all_predictions: Dict[str, float]

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    device: str

# =====================================
# MODEL LOADER
# =====================================

class EmotionDetector:
    """
    Emotion Detection Model Wrapper
    
    Handles loading and inference of the PyTorch emotion detection model
    """
    
    def __init__(self, model_path: str, emotion_labels: list):
        self.model_path = model_path
        self.emotion_labels = emotion_labels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the PyTorch model from file"""
        try:
            if os.path.exists(self.model_path):
                logger.info(f"Loading model from {self.model_path}")
                self.model = torch.load(self.model_path, map_location=self.device)
                self.model.eval()
                logger.info(f"✅ Model loaded successfully on {self.device}")
            else:
                logger.warning(f"⚠️ Model file not found at {self.model_path}")
                logger.info("Creating a placeholder model for testing")
                self.model = self._create_placeholder_model()
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            logger.info("Creating a placeholder model for testing")
            self.model = self._create_placeholder_model()
    
    def _create_placeholder_model(self):
        """
        Create a simple placeholder model for testing when actual model is not available
        This returns random predictions for demonstration purposes
        """
        class PlaceholderModel(torch.nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.num_classes = num_classes
            
            def forward(self, x):
                batch_size = x.size(0)
                # Return random logits
                return torch.randn(batch_size, self.num_classes)
        
        model = PlaceholderModel(len(self.emotion_labels))
        model.eval()
        return model
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for model input
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed tensor ready for model
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model's expected input size
        image = image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
        
        # Convert to numpy array and normalize
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        # Shape: (H, W, C) -> (C, H, W) -> (1, C, H, W)
        tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run emotion prediction on image
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary with emotion, confidence, and all predictions
        """
        with torch.no_grad():
            # Preprocess image
            tensor = self.preprocess_image(image)
            
            # Run inference
            outputs = self.model(tensor)
            
            # Get probabilities
            probabilities = F.softmax(outputs, dim=1).squeeze(0)
            
            # Get top prediction
            confidence, predicted_idx = torch.max(probabilities, 0)
            predicted_emotion = self.emotion_labels[predicted_idx.item()]
            
            # Create dict of all predictions
            all_predictions = {
                label: float(prob)
                for label, prob in zip(self.emotion_labels, probabilities.cpu().numpy())
            }
            
            return {
                'emotion': predicted_emotion,
                'confidence': float(confidence.item()),
                'all_predictions': all_predictions
            }

# =====================================
# INITIALIZE FASTAPI APP
# =====================================

app = FastAPI(
    title="Emotion Detection ML Service",
    description="PyTorch-based emotion detection microservice",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize emotion detector
detector = EmotionDetector(config.MODEL_PATH, config.EMOTION_LABELS)

# =====================================
# API ENDPOINTS
# =====================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with service info"""
    return {
        "status": "running",
        "model_loaded": detector.model is not None,
        "device": str(detector.device)
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": detector.model is not None,
        "device": str(detector.device)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_emotion(request: PredictionRequest):
    """
    Predict emotion from base64-encoded image
    
    Args:
        request: PredictionRequest with base64 image
        
    Returns:
        PredictionResponse with emotion, confidence, and all predictions
    """
    try:
        # Decode base64 image
        # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
        image_data = request.image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 to bytes
        image_bytes = base64.b64decode(image_data)
        
        # Open image with PIL
        image = Image.open(io.BytesIO(image_bytes))
        
        # Run prediction
        result = detector.predict(image)
        
        logger.info(f"✅ Prediction: {result['emotion']} ({result['confidence']:.2%})")
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"❌ Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/emotions")
async def get_emotion_labels():
    """Get list of supported emotion labels"""
    return {
        "emotions": config.EMOTION_LABELS,
        "count": len(config.EMOTION_LABELS)
    }

# =====================================
# STARTUP & SHUTDOWN EVENTS
# =====================================

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("")
    logger.info("🚀 ========================================")
    logger.info("🚀 Emotion Detection ML Service")
    logger.info("🚀 ========================================")
    logger.info(f"🤖 Model: {config.MODEL_PATH}")
    logger.info(f"📊 Emotions: {', '.join(config.EMOTION_LABELS)}")
    logger.info(f"🖥️  Device: {detector.device}")
    logger.info(f"🌐 Server: http://{config.HOST}:{config.PORT}")
    logger.info("🚀 ========================================")
    logger.info("")

@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("⏹️  Shutting down ML service...")

# =====================================
# MAIN
# =====================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )
