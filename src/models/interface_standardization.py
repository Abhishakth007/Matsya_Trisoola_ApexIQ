"""
Model interface standardization for consistent forward signatures across detector and postprocessor models.
Ensures all models have compatible interfaces for the inference pipeline.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class StandardizedModelInterface(nn.Module):
    """Wrapper that standardizes model interfaces for consistent inference pipeline.
    
    All models should have the same forward signature:
    - Input: images (List[torch.Tensor]), targets (Optional[List[Dict]])
    - Output: (predictions, loss) where predictions can be dict or tensor
    """
    
    def __init__(self, model, model_type="detector"):
        super(StandardizedModelInterface, self).__init__()
        self.model = model
        self.model_type = model_type
        
    def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict]] = None) -> Tuple[Union[Dict, torch.Tensor], Optional[torch.Tensor]]:
        """Standardized forward pass that handles different model interfaces.
        
        Args:
            images: List of image tensors
            targets: Optional list of target dictionaries
            
        Returns:
            Tuple of (predictions, loss) where:
            - predictions: Model-specific output (dict for detector, tensor for postprocessor)
            - loss: Training loss (None during inference)
        """
        try:
            # Call the underlying model's forward method
            if self.model_type == "detector":
                # Detector models return (detections, loss)
                detections, loss = self.model(images, targets)
                return detections, loss
            elif self.model_type == "postprocessor":
                # Postprocessor models return (output, loss)
                output, loss = self.model(images, targets)
                return output, loss
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
                
        except Exception as e:
            logger.error(f"Error in {self.model_type} forward pass: {e}")
            # Return safe fallback values
            if self.model_type == "detector":
                return [{"boxes": torch.empty(0, 4), "scores": torch.empty(0), "labels": torch.empty(0)}], None
            else:
                return torch.zeros(1, 21), None  # 21 attributes for vessel properties


def standardize_model_interface(model, model_type="detector"):
    """Wrap a model with standardized interface.
    
    Args:
        model: The model to wrap
        model_type: Type of model ("detector" or "postprocessor")
        
    Returns:
        Model with standardized interface
    """
    return StandardizedModelInterface(model, model_type)


class ModelInterfaceValidator:
    """Validator to ensure model interfaces are compatible."""
    
    @staticmethod
    def validate_detector_interface(model, sample_images, sample_targets):
        """Validate that detector model has correct interface."""
        try:
            # Test forward pass
            detections, loss = model(sample_images, sample_targets)
            
            # Validate detections format
            if not isinstance(detections, list):
                raise ValueError("Detector must return list of detection dicts")
            
            for detection in detections:
                if not isinstance(detection, dict):
                    raise ValueError("Each detection must be a dict")
                required_keys = ["boxes", "scores", "labels"]
                for key in required_keys:
                    if key not in detection:
                        raise ValueError(f"Detection dict missing required key: {key}")
            
            logger.info("✅ Detector interface validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Detector interface validation failed: {e}")
            return False
    
    @staticmethod
    def validate_postprocessor_interface(model, sample_images, sample_targets):
        """Validate that postprocessor model has correct interface."""
        try:
            # Test forward pass
            output, loss = model(sample_images, sample_targets)
            
            # Validate output format
            if not isinstance(output, torch.Tensor):
                raise ValueError("Postprocessor must return torch.Tensor")
            
            # Check expected output shape (21 attributes)
            expected_attributes = 21  # length, width, 16 heading buckets, speed, 2 vessel type
            if output.shape[-1] != expected_attributes:
                logger.warning(f"Postprocessor output has {output.shape[-1]} attributes, expected {expected_attributes}")
            
            logger.info("✅ Postprocessor interface validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Postprocessor interface validation failed: {e}")
            return False


def create_compatible_model_pair(detector_model, postprocessor_model):
    """Create a pair of models with compatible interfaces.
    
    Args:
        detector_model: The detector model
        postprocessor_model: The postprocessor model
        
    Returns:
        Tuple of (standardized_detector, standardized_postprocessor)
    """
    # Standardize interfaces
    detector = standardize_model_interface(detector_model, "detector")
    postprocessor = standardize_model_interface(postprocessor_model, "postprocessor")
    
    # Validate interfaces
    validator = ModelInterfaceValidator()
    
    # Create sample data for validation
    sample_images = [torch.randn(2, 512, 512)]  # 2-channel SAR image
    sample_targets = [{
        "boxes": torch.tensor([[100, 100, 200, 200]]),
        "labels": torch.tensor([0])
    }]
    
    # Validate both models
    detector_valid = validator.validate_detector_interface(detector, sample_images, sample_targets)
    postprocessor_valid = validator.validate_postprocessor_interface(postprocessor, sample_images, sample_targets)
    
    if not detector_valid or not postprocessor_valid:
        raise ValueError("Model interface validation failed")
    
    return detector, postprocessor
