"""
Configuration Loader and Validator

This module provides centralized configuration loading, validation,
and access for the vessel detection system.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Centralized configuration loader with validation and environment variable support."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to configuration file. If None, uses default location.
        """
        if config_path is None:
            # Default config path relative to project root
            project_root = os.getenv("PROJECT_ROOT", ".")
            config_path = os.path.join(project_root, "src", "config", "config.yml")
        
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self.config = yaml.safe_load(file)
            
            if not self.config:
                raise ValueError("Configuration file is empty or invalid")
            
            logger.info(f"✅ Configuration loaded successfully from {self.config_path}")
            
            # Validate configuration structure
            self._validate_config_structure()
            
        except Exception as e:
            logger.error(f"❌ Failed to load configuration: {e}")
            # Load minimal default configuration
            self._load_default_config()
    
    def _load_default_config(self) -> None:
        """Load minimal default configuration when main config fails."""
        logger.warning("Loading minimal default configuration")
        self.config = {
            "main": {
                "sentinel1_detector": "./data/model_artifacts/sentinel-1/frcnn_cmp2/3dff445",
                "sentinel1_postprocessor": "./data/model_artifacts/sentinel-1/attr/c34aa37",
                "sentinel2_detector": "./data/model_artifacts/sentinel-2/frcnn_cmp2/default",
                "sentinel2_postprocessor": "./data/model_artifacts/sentinel-2/attr/default",
                "default_window_size": 2048,
                "default_padding": 400,
                "default_overlap": 20,
                "default_confidence_threshold": 0.9,
                "default_nms_threshold": 10.0,
                "detector_batch_size": 4,
                "postprocessor_batch_size": 32,
                "max_gpu_memory_mb": 8000,
                "chunked_processing_threshold_mb": 1000,
                "save_crops": True,
                "debug_mode": False,
                "remove_clouds": False,
                "log_level": "INFO",
                "log_file": "./vessel_detection.log"
            }
        }
    
    def _validate_config_structure(self) -> None:
        """Validate that required configuration sections exist."""
        required_sections = ["main"]
        required_main_keys = [
            "sentinel1_detector", "sentinel1_postprocessor",
            "sentinel2_detector", "sentinel2_postprocessor"
        ]
        
        # Check required sections
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Check required main keys
        main_config = self.config.get("main", {})
        for key in required_main_keys:
            if key not in main_config:
                raise ValueError(f"Missing required configuration key: main.{key}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Configuration key path (e.g., "main.sentinel1_detector")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            if default is not None:
                logger.debug(f"Configuration key '{key_path}' not found, using default: {default}")
                return default
            else:
                logger.warning(f"Configuration key '{key_path}' not found and no default provided")
                return None
    
    def get_channel_config(self, catalog: str) -> Dict[str, Any]:
        """
        Get channel configuration for a specific catalog.
        
        Args:
            catalog: Imagery catalog ("sentinel1" or "sentinel2")
            
        Returns:
            Channel configuration dictionary
        """
        channel_config = self.get(f"main.channel_management.{catalog}", {})
        
        # Provide defaults if not configured
        if catalog == "sentinel1":
            defaults = {
                "base_channels": ["vh", "vv"],
                "overlap_channels": ["vh_overlap0", "vv_overlap0", "vh_overlap1", "vv_overlap1"],
                "synthetic_overlap_strategy": "spatial_averaging",
                "fallback_channel_count": 6,
                "enable_intelligent_duplication": True
            }
        elif catalog == "sentinel2":
            defaults = {
                "base_channels": ["B02", "B03", "B04"],
                "overlap_channels": [],
                "fallback_channel_count": 3,
                "enable_intelligent_duplication": False
            }
        else:
            defaults = {}
        
        # Merge defaults with configured values
        for key, default_value in defaults.items():
            if key not in channel_config:
                channel_config[key] = default_value
        
        return channel_config
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get image processing configuration."""
        processing_config = self.get("main.image_processing", {})
        
        # Provide defaults
        defaults = {
            "max_offset": 8,
            "step_size": 2,
            "alignment_search_window": 16,
            "gaussian_filter_sigma": 1.0,
            "noise_injection_level": 0.05,
            "median_filter_size": 3
        }
        
        # Merge defaults with configured values
        for key, default_value in defaults.items():
            if key not in processing_config:
                processing_config[key] = default_value
        
        return processing_config
    
    def get_error_handling_config(self) -> Dict[str, Any]:
        """Get error handling configuration."""
        error_config = self.get("main.error_handling", {})
        
        # Provide defaults
        defaults = {
            "max_retries": 3,
            "graceful_degradation": True,
            "resource_cleanup_timeout": 30,
            "user_friendly_errors": True,
            "enable_fallback_channels": True,
            "log_full_tracebacks": False
        }
        
        # Merge defaults with configured values
        for key, default_value in defaults.items():
            if key not in error_config:
                error_config[key] = default_value
        
        return error_config
    
    def get_model_paths(self, catalog: str) -> Dict[str, str]:
        """
        Get model paths for a specific catalog.
        
        Args:
            catalog: Imagery catalog ("sentinel1" or "sentinel2")
            
        Returns:
            Dictionary with detector and postprocessor paths
        """
        return {
            "detector": self.get(f"main.{catalog}_detector"),
            "postprocessor": self.get(f"main.{catalog}_postprocessor")
        }
    
    def resolve_paths(self, base_path: str) -> str:
        """
        Resolve configuration paths relative to base path.
        
        Args:
            base_path: Base path for resolution
            
        Returns:
            Resolved path
        """
        if base_path.startswith("${PROJECT_ROOT}"):
            project_root = os.getenv("PROJECT_ROOT", ".")
            return base_path.replace("${PROJECT_ROOT}", project_root)
        elif base_path.startswith("./") or base_path.startswith("../"):
            # Relative path, resolve against current working directory
            return os.path.abspath(base_path)
        else:
            return base_path
    
    def validate_model_paths(self) -> bool:
        """Validate that all configured model paths exist."""
        catalogs = ["sentinel1", "sentinel2"]
        all_valid = True
        
        for catalog in catalogs:
            try:
                paths = self.get_model_paths(catalog)
                for model_type, path in paths.items():
                    if path:
                        resolved_path = self.resolve_paths(path)
                        if not os.path.exists(resolved_path):
                            logger.error(f"Model path does not exist: {catalog}.{model_type} = {resolved_path}")
                            all_valid = False
                        else:
                            logger.debug(f"✅ Model path valid: {catalog}.{model_type} = {resolved_path}")
                    else:
                        logger.warning(f"Model path not configured: {catalog}.{model_type}")
            except Exception as e:
                logger.error(f"Error validating model paths for {catalog}: {e}")
                all_valid = False
        
        return all_valid
    
    def reload(self) -> None:
        """Reload configuration from file."""
        logger.info("Reloading configuration...")
        self.load_config()
    
    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration dictionary."""
        return self.config.copy()


# Global configuration instance
_config_loader: Optional[ConfigLoader] = None


def get_config() -> ConfigLoader:
    """Get global configuration loader instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


def get_config_value(key_path: str, default: Any = None) -> Any:
    """Get configuration value using global loader."""
    return get_config().get(key_path, default)


def get_channel_config(catalog: str) -> Dict[str, Any]:
    """Get channel configuration using global loader."""
    return get_config().get_channel_config(catalog)


def get_processing_config() -> Dict[str, Any]:
    """Get processing configuration using global loader."""
    return get_config().get_processing_config()


def get_error_handling_config() -> Dict[str, Any]:
    """Get error handling configuration using global loader."""
    return get_config().get_error_handling_config()


def get_model_paths(catalog: str) -> Dict[str, str]:
    """Get model paths using global loader."""
    return get_config().get_model_paths(catalog)


# Export main functions
__all__ = [
    'ConfigLoader', 'get_config', 'get_config_value',
    'get_channel_config', 'get_processing_config',
    'get_error_handling_config', 'get_model_paths'
]
