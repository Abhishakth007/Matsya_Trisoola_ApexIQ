"""
Advanced Memory Management System for Vessel Detection

This module provides comprehensive memory monitoring, optimization,
and cleanup for the vessel detection pipeline.
"""

import os
import gc
import time
import logging
import psutil
import threading
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager
import torch

logger = logging.getLogger(__name__)


class MemoryManager:
    """Advanced memory management with monitoring and optimization."""
    
    def __init__(self, max_memory_mb: int = 8000, alert_threshold_mb: int = 6000):
        """
        Initialize memory manager.
        
        Args:
            max_memory_mb: Maximum allowed memory usage in MB
            alert_threshold_mb: Memory threshold for warnings in MB
        """
        self.max_memory_mb = max_memory_mb
        self.alert_threshold_mb = alert_threshold_mb
        self.peak_memory = 0
        self.memory_history = []
        self.monitoring_active = False
        self.monitor_thread = None
        
        # GDAL memory optimization
        self._optimize_gdal_memory()
        
        logger.info(f"🧠 Memory Manager initialized - Max: {max_memory_mb}MB, Alert: {alert_threshold_mb}MB")
    
    def _optimize_gdal_memory(self):
        """Optimize GDAL memory settings."""
        try:
            from osgeo import gdal
            # Set GDAL cache to prevent memory bloat
            gdal.SetCacheMax(512 * 1024 * 1024)  # 512MB max cache
            gdal.SetConfigOption("GDAL_NUM_THREADS", "2")  # Limit threads
            gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
            logger.info("✅ GDAL memory optimization applied")
        except ImportError:
            logger.warning("⚠️ GDAL not available for memory optimization")
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            return memory_mb
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return 0.0
    
    def get_system_memory_info(self) -> Dict[str, float]:
        """Get comprehensive system memory information."""
        try:
            vm = psutil.virtual_memory()
            return {
                "total_gb": vm.total / 1024 / 1024 / 1024,
                "available_gb": vm.available / 1024 / 1024 / 1024,
                "used_gb": vm.used / 1024 / 1024 / 1024,
                "percent_used": vm.percent
            }
        except Exception as e:
            logger.warning(f"Failed to get system memory info: {e}")
            return {}
    
    def log_memory(self, stage: str) -> float:
        """Log memory usage at different stages."""
        memory_mb = self.get_memory_usage()
        self.peak_memory = max(self.peak_memory, memory_mb)
        
        # Add to history
        self.memory_history.append({
            "timestamp": time.time(),
            "stage": stage,
            "memory_mb": memory_mb
        })
        
        # Check thresholds
        if memory_mb > self.max_memory_mb:
            logger.error(f"🚨 CRITICAL: Memory usage {memory_mb:.1f}MB exceeds limit {self.max_memory_mb}MB")
        elif memory_mb > self.alert_threshold_mb:
            logger.warning(f"⚠️ WARNING: High memory usage {memory_mb:.1f}MB at {stage}")
        else:
            logger.info(f"🧠 Memory Usage - {stage}: {memory_mb:.1f}MB")
        
        return memory_mb
    
    def start_memory_monitoring(self, interval_seconds: int = 30):
        """Start continuous memory monitoring in background thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    memory_mb = self.get_memory_usage()
                    if memory_mb > self.alert_threshold_mb:
                        logger.warning(f"🔄 Memory Monitor: {memory_mb:.1f}MB (Peak: {self.peak_memory:.1f}MB)")
                    time.sleep(interval_seconds)
                except Exception as e:
                    logger.error(f"Memory monitoring error: {e}")
                    time.sleep(interval_seconds)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"🔄 Memory monitoring started (interval: {interval_seconds}s)")
    
    def stop_memory_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("🔄 Memory monitoring stopped")
    
    def force_memory_cleanup(self):
        """Force comprehensive memory cleanup."""
        initial_memory = self.get_memory_usage()
        
        # Python garbage collection
        collected = gc.collect()
        
        # PyTorch CUDA cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection again
        gc.collect()
        
        final_memory = self.get_memory_usage()
        memory_freed = initial_memory - final_memory
        
        logger.info(f"🧹 Memory cleanup: {initial_memory:.1f}MB → {final_memory:.1f}MB (freed {memory_freed:.1f}MB)")
        
        return memory_freed
    
    def optimize_for_low_memory(self):
        """Apply low-memory optimizations."""
        logger.info("🔧 Applying low-memory optimizations...")
        
        # Reduce GDAL cache further
        try:
            from osgeo import gdal
            gdal.SetCacheMax(256 * 1024 * 1024)  # 256MB max cache
            logger.info("✅ GDAL cache reduced to 256MB")
        except ImportError:
            pass
        
        # Force cleanup
        self.force_memory_cleanup()
        
        logger.info("✅ Low-memory optimizations applied")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory usage summary."""
        current_memory = self.get_memory_usage()
        system_info = self.get_system_memory_info()
        
        return {
            "current_memory_mb": current_memory,
            "peak_memory_mb": self.peak_memory,
            "memory_history": self.memory_history,
            "system_memory": system_info,
            "memory_efficiency": (current_memory / self.peak_memory * 100) if self.peak_memory > 0 else 0
        }
    
    def check_memory_health(self) -> bool:
        """Check if memory usage is healthy."""
        current_memory = self.get_memory_usage()
        return current_memory < self.alert_threshold_mb
    
    @contextmanager
    def memory_context(self, context_name: str, cleanup_threshold_mb: Optional[int] = None):
        """Context manager for memory-intensive operations."""
        start_memory = self.get_memory_usage()
        start_time = time.time()
        
        logger.info(f"🚀 Starting memory context: {context_name}")
        self.log_memory(f"{context_name}_start")
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self.get_memory_usage()
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            logger.info(f"✅ Memory context {context_name} completed in {duration:.1f}s")
            logger.info(f"   Memory delta: {memory_delta:+.1f}MB")
            
            # Auto-cleanup if threshold exceeded
            if cleanup_threshold_mb and abs(memory_delta) > cleanup_threshold_mb:
                logger.info(f"🧹 Auto-cleanup triggered (delta: {memory_delta:.1f}MB)")
                self.force_memory_cleanup()
            
            self.log_memory(f"{context_name}_end")


# Global memory manager instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager


def log_memory(stage: str) -> float:
    """Log memory usage using global manager."""
    return get_memory_manager().log_memory(stage)


def force_memory_cleanup() -> float:
    """Force memory cleanup using global manager."""
    return get_memory_manager().force_memory_cleanup()


@contextmanager
def memory_context(context_name: str, cleanup_threshold_mb: Optional[int] = None):
    """Memory context manager using global manager."""
    with get_memory_manager().memory_context(context_name, cleanup_threshold_mb):
        yield


# Export main functions
__all__ = [
    'MemoryManager', 'get_memory_manager', 'log_memory',
    'force_memory_cleanup', 'memory_context'
]
