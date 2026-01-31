"""
Utility functions for model loading and management.
"""
import os
import json
import pickle
import logging
from typing import Tuple, Optional, List
# Don't import tensorflow at top level - import lazily when needed
import numpy as np
from sklearn.preprocessing import LabelBinarizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_path: str = "skin_cancer_model.h5"):
    """
    Load the trained Keras model from file.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded Keras model or None if loading fails
    """
    # Lazy import tensorflow only when needed
    import tensorflow as tf
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return None
    
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


def load_label_binarizer(binarizer_path: str = "label_binarizer.pkl") -> Optional[LabelBinarizer]:
    """
    Load the label binarizer from file.
    
    Args:
        binarizer_path: Path to the label binarizer pickle file
        
    Returns:
        Loaded LabelBinarizer or None if loading fails
    """
    if not os.path.exists(binarizer_path):
        logger.warning(f"Label binarizer not found: {binarizer_path}")
        return None
    
    try:
        with open(binarizer_path, "rb") as f:
            lb = pickle.load(f)
        logger.info(f"Label binarizer loaded successfully from {binarizer_path}")
        return lb
    except Exception as e:
        logger.error(f"Error loading label binarizer: {e}")
        return None


def load_class_names(class_names_path: str = "class_names.json") -> List[str]:
    """
    Load class names from JSON file.
    
    Args:
        class_names_path: Path to the class names JSON file
        
    Returns:
        List of class names or empty list if loading fails
    """
    if not os.path.exists(class_names_path):
        logger.warning(f"Class names file not found: {class_names_path}")
        # Return default class names as fallback
        return ['benign', 'malignant']
    
    try:
        with open(class_names_path, "r") as f:
            class_names = json.load(f)
        logger.info(f"Class names loaded: {class_names}")
        return class_names
    except Exception as e:
        logger.error(f"Error loading class names: {e}")
        return ['benign', 'malignant']


def load_model_metadata(metadata_path: str = "model_metadata.json") -> Optional[dict]:
    """
    Load model metadata from JSON file.
    
    Args:
        metadata_path: Path to the metadata JSON file
        
    Returns:
        Dictionary containing model metadata or None if loading fails
    """
    if not os.path.exists(metadata_path):
        logger.warning(f"Model metadata file not found: {metadata_path}")
        return None
    
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        logger.info("Model metadata loaded successfully")
        return metadata
    except Exception as e:
        logger.error(f"Error loading model metadata: {e}")
        return None


def get_class_names_from_binarizer(lb: LabelBinarizer) -> List[str]:
    """
    Extract class names from a LabelBinarizer object.
    
    Args:
        lb: LabelBinarizer object
        
    Returns:
        List of class names
    """
    if hasattr(lb, 'classes_'):
        return lb.classes_.tolist()
    return []


def preprocess_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
    """
    Preprocess an image for model prediction.
    Uses EfficientNet's preprocess_input to match training preprocessing.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for the image (height, width)
        
    Returns:
        Preprocessed image array or None if processing fails
    """
    try:
        # Use PIL instead of deprecated tensorflow.keras.preprocessing
        from PIL import Image
        from tensorflow.keras.applications.efficientnet import preprocess_input
        
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img, dtype=np.float32)
        # Use EfficientNet's preprocess_input (normalizes to [-1, 1] range)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {e}")
        return None

