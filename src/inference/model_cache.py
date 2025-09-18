"""
Model Caching System for Vessel Detection System
Phase 3: Model Loading and Inference Optimization

This module implements:
1. Model Caching System
2. Parallel Model Loading
3. Lazy Loading Strategy
4. Model optimization and warmup
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Union
import warnings
from functools import lru_cache
import threading

import torch
import torch.nn as nn

from src.models import models
from .memory_manager import MemoryManager, get_memory_manager

logger = logging.getLogger(__name__)

class ModelCache:
    """
    Advanced model caching system for vessel detection.
    
    Implements singleton pattern, parallel loading, and memory optimization
    to reduce model loading overhead.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelCache, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        """
        Initialize model cache (only once due to singleton).
        
        Args:
            memory_manager: Memory manager instance
        """
        if hasattr(self, '_initialized'):
            return
        
        self.memory_manager = memory_manager or get_memory_manager()
        self.device = self.memory_manager.device
        
        # Model cache storage
        self.cached_models: Dict[str, nn.Module] = {}
        self.model_configs: Dict[str, Dict] = {}
        self.model_metadata: Dict[str, Dict] = {}
        
        # Performance tracking
        self.load_times: Dict[str, float] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Thread safety
        self._cache_lock = threading.Lock()
        
        # Model warmup cache
        self.warmed_models: set = set()
        
        self._initialized = True
        logger.info(f"🚀 Model Cache initialized for {self.device}")
    
    def get_model(self, model_dir: str, 
                 example: Optional[List] = None,
                 force_reload: bool = False) -> nn.Module:
        """
        Get model from cache or load if not cached.
        
        Args:
            model_dir: Directory containing model files
            example: Example input for model initialization
            force_reload: Force reload even if cached
            
        Returns:
            Loaded model
        """
        try:
            cache_key = self._get_cache_key(model_dir)
            
            with self._cache_lock:
                # Check if model is already cached
                if not force_reload and cache_key in self.cached_models:
                    self.cache_hits += 1
                    logger.info(f"📋 Model cache hit: {cache_key}")
                    return self.cached_models[cache_key]
                
                self.cache_misses += 1
                logger.info(f"📥 Loading model: {model_dir}")
                
                # Load model
                start_time = time.time()
                model = self._load_model_internal(model_dir, example)
                load_time = time.time() - start_time
                
                # Cache the model
                self.cached_models[cache_key] = model
                self.load_times[cache_key] = load_time
                
                # Warm up model if not already warmed
                if cache_key not in self.warmed_models:
                    self._warmup_model(model, cache_key)
                
                logger.info(f"✅ Model loaded and cached in {load_time:.3f}s: {cache_key}")
                return model
                
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
    
    def _load_model_internal(self, model_dir: str, example: Optional[List] = None) -> nn.Module:
        """
        Internal model loading with error handling.
        
        Args:
            model_dir: Model directory
            example: Example input
            
        Returns:
            Loaded model
        """
        try:
            # Load model using existing models module
            model = models.load_model(model_dir, example=example, device=self.device)
            
            # Set model to evaluation mode
            model.eval()
            
            # Store model metadata
            cache_key = self._get_cache_key(model_dir)
            self.model_metadata[cache_key] = {
                'model_dir': model_dir,
                'device': str(self.device),
                'load_time': time.time(),
                'model_type': type(model).__name__
            }
            
            return model
            
        except Exception as e:
            logger.error(f"Internal model loading failed: {e}")
            raise
    
    def _get_cache_key(self, model_dir: str) -> str:
        """Generate cache key for model directory."""
        return os.path.abspath(model_dir)
    
    def _warmup_model(self, model: nn.Module, cache_key: str) -> None:
        """
        Warm up model with dummy input to avoid first-run overhead.
        
        Args:
            model: Model to warm up
            cache_key: Cache key for tracking
        """
        try:
            logger.info(f"🔥 Warming up model: {cache_key}")
            
            # Create dummy input based on model type
            dummy_input = self._create_dummy_input(model)
            
            if dummy_input is not None:
                with torch.no_grad():
                    # Run forward pass to warm up
                    _ = model(dummy_input)
                
                self.warmed_models.add(cache_key)
                logger.info(f"✅ Model warmed up: {cache_key}")
            
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    def _create_dummy_input(self, model: nn.Module) -> Optional[List]:
        """
        Create dummy input for model warmup.
        
        Args:
            model: Model to create input for
            
        Returns:
            Dummy input or None if not supported
        """
        try:
            # Try to infer input shape from model
            if hasattr(model, 'num_channels'):
                channels = model.num_channels
            else:
                # Default to 3 channels (RGB)
                channels = 3
            
            # Create dummy input
            dummy_tensor = torch.randn(1, channels, 224, 224, device=self.device)
            return [dummy_tensor, None]
            
        except Exception as e:
            logger.warning(f"Dummy input creation failed: {e}")
            return None
    
    def load_models_parallel(self, detector_dir: str, postprocess_dir: str,
                           example_detector: Optional[List] = None,
                           example_postprocess: Optional[List] = None) -> Tuple[nn.Module, nn.Module]:
        """
        Load detector and postprocessor models in parallel.
        
        Args:
            detector_dir: Detector model directory
            postprocess_dir: Postprocessor model directory
            example_detector: Example input for detector
            example_postprocess: Example input for postprocessor
            
        Returns:
            Tuple of (detector_model, postprocessor_model)
        """
        try:
            import concurrent.futures
            
            logger.info("🔄 Loading models in parallel...")
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both model loading tasks
                detector_future = executor.submit(
                    self.get_model, detector_dir, example_detector
                )
                postprocess_future = executor.submit(
                    self.get_model, postprocess_dir, example_postprocess
                )
                
                # Wait for both to complete
                detector_model = detector_future.result()
                postprocess_model = postprocess_future.result()
            
            total_time = time.time() - start_time
            logger.info(f"✅ Parallel model loading completed in {total_time:.3f}s")
            
            return detector_model, postprocess_model
            
        except Exception as e:
            logger.error(f"Parallel model loading failed: {e}")
            # Fallback to sequential loading
            logger.info("🔄 Falling back to sequential loading...")
            detector_model = self.get_model(detector_dir, example_detector)
            postprocess_model = self.get_model(postprocess_dir, example_postprocess)
            return detector_model, postprocess_model
    
    def preload_models(self, model_dirs: List[str]) -> None:
        """
        Preload models in background for faster access.
        
        Args:
            model_dirs: List of model directories to preload
        """
        try:
            logger.info(f"🔄 Preloading {len(model_dirs)} models...")
            
            for model_dir in model_dirs:
                try:
                    # Load model in background
                    self.get_model(model_dir)
                    logger.info(f"✅ Preloaded: {model_dir}")
                except Exception as e:
                    logger.warning(f"Preload failed for {model_dir}: {e}")
            
            logger.info("✅ Model preloading completed")
            
        except Exception as e:
            logger.error(f"Model preloading failed: {e}")
    
    def clear_cache(self, model_dir: Optional[str] = None) -> None:
        """
        Clear model cache.
        
        Args:
            model_dir: Specific model to clear (None for all)
        """
        try:
            with self._cache_lock:
                if model_dir is None:
                    # Clear all models
                    for cache_key in list(self.cached_models.keys()):
                        del self.cached_models[cache_key]
                    self.model_metadata.clear()
                    self.load_times.clear()
                    self.warmed_models.clear()
                    logger.info("🗑️ All models cleared from cache")
                else:
                    # Clear specific model
                    cache_key = self._get_cache_key(model_dir)
                    if cache_key in self.cached_models:
                        del self.cached_models[cache_key]
                        if cache_key in self.model_metadata:
                            del self.model_metadata[cache_key]
                        if cache_key in self.load_times:
                            del self.load_times[cache_key]
                        self.warmed_models.discard(cache_key)
                        logger.info(f"🗑️ Model cleared from cache: {cache_key}")
            
            # Force garbage collection
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Cache clearing failed: {e}")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        try:
            total_load_time = sum(self.load_times.values())
            avg_load_time = total_load_time / len(self.load_times) if self.load_times else 0
            
            return {
                'cached_models': len(self.cached_models),
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses),
                'total_load_time': total_load_time,
                'average_load_time': avg_load_time,
                'warmed_models': len(self.warmed_models)
            }
            
        except Exception as e:
            logger.error(f"Cache stats retrieval failed: {e}")
            return {}
    
    def optimize_model(self, model: nn.Module, model_name: str) -> nn.Module:
        """
        Optimize model for inference.
        
        Args:
            model: Model to optimize
            model_name: Name of model for logging
            
        Returns:
            Optimized model
        """
        try:
            logger.info(f"⚡ Optimizing model: {model_name}")
            
            # Set to evaluation mode
            model.eval()
            
            # Enable optimizations
            if hasattr(torch, 'jit') and hasattr(torch.jit, 'optimize_for_inference'):
                try:
                    model = torch.jit.optimize_for_inference(model)
                    logger.info(f"✅ JIT optimization applied to {model_name}")
                except Exception as e:
                    logger.warning(f"JIT optimization failed for {model_name}: {e}")
            
            # Enable CUDA optimizations
            if self.device.type == 'cuda':
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                logger.info(f"✅ CUDA optimizations enabled for {model_name}")
            
            return model
            
        except Exception as e:
            logger.error(f"Model optimization failed for {model_name}: {e}")
            return model


class LazyModelLoader:
    """
    Lazy loading implementation for models.
    """
    
    def __init__(self, model_cache: ModelCache):
        """
        Initialize lazy model loader.
        
        Args:
            model_cache: Model cache instance
        """
        self.model_cache = model_cache
        self.loaded_models: Dict[str, nn.Module] = {}
    
    def get_model_lazy(self, model_dir: str, 
                      example: Optional[List] = None) -> nn.Module:
        """
        Get model with lazy loading.
        
        Args:
            model_dir: Model directory
            example: Example input
            
        Returns:
            Loaded model
        """
        cache_key = self.model_cache._get_cache_key(model_dir)
        
        if cache_key not in self.loaded_models:
            self.loaded_models[cache_key] = self.model_cache.get_model(model_dir, example)
        
        return self.loaded_models[cache_key]


# Global model cache instance
_global_model_cache: Optional[ModelCache] = None

def get_model_cache(memory_manager: Optional[MemoryManager] = None) -> ModelCache:
    """Get global model cache instance."""
    global _global_model_cache
    
    if _global_model_cache is None:
        _global_model_cache = ModelCache(memory_manager)
    
    return _global_model_cache


# Convenience functions for easy usage
def load_models_optimized(detector_dir: str, postprocess_dir: str,
                         memory_manager: Optional[MemoryManager] = None,
                         example_detector: Optional[List] = None,
                         example_postprocess: Optional[List] = None) -> Tuple[nn.Module, nn.Module]:
    """
    Load detector and postprocessor models with optimization.
    
    Args:
        detector_dir: Detector model directory
        postprocess_dir: Postprocessor model directory
        memory_manager: Memory manager instance
        example_detector: Example input for detector
        example_postprocess: Example input for postprocessor
        
    Returns:
        Tuple of (detector_model, postprocessor_model)
    """
    model_cache = get_model_cache(memory_manager)
    return model_cache.load_models_parallel(
        detector_dir, postprocess_dir, example_detector, example_postprocess
    )


def get_model_optimized(model_dir: str, 
                       memory_manager: Optional[MemoryManager] = None,
                       example: Optional[List] = None) -> nn.Module:
    """
    Get optimized model from cache.
    
    Args:
        model_dir: Model directory
        memory_manager: Memory manager instance
        example: Example input
        
    Returns:
        Loaded and optimized model
    """
    model_cache = get_model_cache(memory_manager)
    model = model_cache.get_model(model_dir, example)
    return model_cache.optimize_model(model, os.path.basename(model_dir))
