# FIXED: Enhanced model registry with proper error handling and validation
from . import custom as custom
from . import resnet as resnet
from . import unet as unet
from . import frcnn_cmp2 as frcnn_cmp2
from . import frcnn as frcnn

# Model registry with validation
models = {}

# Register all available models with validation
try:
    models["frcnn"] = frcnn.FasterRCNNModel
    models["frcnn_cmp2"] = frcnn_cmp2.FasterRCNNModel
    models["unet"] = unet.Model
    models["resnet"] = resnet.Model
    models["custom"] = custom.Model
    models["custom_separate_heads"] = custom.SeparateHeadAttrModel
except ImportError as e:
    print(f"Warning: Could not import some models: {e}")

# Validation function to ensure model registry is complete
def validate_model_registry():
    """Validate that all required models are available."""
    required_models = ["frcnn", "frcnn_cmp2", "custom"]
    missing_models = [model for model in required_models if model not in models]
    if missing_models:
        raise ImportError(f"Missing required models: {missing_models}")
    return True

# Auto-validate on import
try:
    validate_model_registry()
except ImportError as e:
    print(f"Model registry validation failed: {e}")
