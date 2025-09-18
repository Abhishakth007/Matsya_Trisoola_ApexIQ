"""
Memory Management Module for Vessel Detection System
Phase 1: Memory Management Revolution

This module implements:
1. Chunked Processing Architecture
2. Tensor Memory Pool Management  
3. Data Type Optimization
4. Memory-aware processing strategies
"""

import gc
import logging
import math
import os
import time
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import psutil
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Advanced memory management for vessel detection pipeline.
    
    Implements chunked processing, tensor pooling, and memory optimization
    strategies to handle large satellite images efficiently.
    """
    
    def __init__(self, device: torch.device, target_gpu_usage: float = 0.6):
        """
        Initialize memory manager.
        
        Args:
            device: Target device (CPU/GPU)
            target_gpu_usage: Target GPU memory usage (0.0-1.0)
        """
        self.device = device
        self.target_gpu_usage = target_gpu_usage
        self.tensor_pools: Dict[Tuple[int, ...], List[torch.Tensor]] = {}
        self.allocated_tensors: List[torch.Tensor] = []
        
        # Memory monitoring
        self.memory_history: List[Dict[str, float]] = []
        self.peak_memory_usage = 0.0
        
        # Performance tracking
        self.operation_times: Dict[str, List[float]] = {}
        
        logger.info(f"🚀 Memory Manager initialized for {device}")
        self._log_memory_status("initialization")
    
    def _log_memory_status(self, stage: str) -> Dict[str, float]:
        """Log current memory usage and return status dict."""
        try:
            # CPU memory
            process = psutil.Process(os.getpid())
            cpu_memory_mb = process.memory_info().rss / 1024 / 1024
            
            # GPU memory
            gpu_memory_mb = 0.0
            gpu_memory_percent = 0.0
            if self.device.type == 'cuda' and torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.memory_allocated(self.device) / 1024 / 1024
                gpu_memory_percent = torch.cuda.memory_allocated(self.device) / torch.cuda.max_memory_allocated(self.device) * 100
            
            status = {
                'stage': stage,
                'cpu_memory_mb': cpu_memory_mb,
                'gpu_memory_mb': gpu_memory_mb,
                'gpu_memory_percent': gpu_memory_percent,
                'timestamp': time.time()
            }
            
            self.memory_history.append(status)
            self.peak_memory_usage = max(self.peak_memory_usage, cpu_memory_mb + gpu_memory_mb)
            
            logger.info(f"💾 Memory Status [{stage}]: CPU={cpu_memory_mb:.1f}MB, GPU={gpu_memory_mb:.1f}MB ({gpu_memory_percent:.1f}%)")
            
            return status
            
        except Exception as e:
            logger.warning(f"Memory monitoring failed: {e}")
            return {'stage': stage, 'error': str(e)}
    
    def get_available_memory(self) -> Dict[str, float]:
        """Get available memory information."""
        try:
            # CPU memory
            cpu_memory = psutil.virtual_memory()
            cpu_available_mb = cpu_memory.available / 1024 / 1024
            
            # GPU memory
            gpu_available_mb = 0.0
            if self.device.type == 'cuda' and torch.cuda.is_available():
                gpu_total = torch.cuda.get_device_properties(self.device).total_memory / 1024 / 1024
                gpu_allocated = torch.cuda.memory_allocated(self.device) / 1024 / 1024
                gpu_available_mb = gpu_total - gpu_allocated
            
            return {
                'cpu_available_mb': cpu_available_mb,
                'gpu_available_mb': gpu_available_mb,
                'cpu_total_mb': cpu_memory.total / 1024 / 1024,
                'gpu_total_mb': gpu_total if self.device.type == 'cuda' else 0.0
            }
        except Exception as e:
            logger.warning(f"Memory info retrieval failed: {e}")
            return {'cpu_available_mb': 0.0, 'gpu_available_mb': 0.0}
    
    def calculate_optimal_chunk_size(self, image_shape: Tuple[int, ...], 
                                   window_size: int, 
                                   batch_size: int) -> Dict[str, int]:
        """
        Calculate optimal chunk size based on available memory.
        
        Args:
            image_shape: Shape of input image (C, H, W)
            window_size: Size of sliding window
            batch_size: Desired batch size
            
        Returns:
            Dict with optimal chunk parameters
        """
        try:
            available_memory = self.get_available_memory()
            
            # Calculate memory requirements
            channels, height, width = image_shape
            window_memory_mb = (channels * window_size * window_size * 4) / (1024 * 1024)  # float32
            batch_memory_mb = window_memory_mb * batch_size
            
            # Target GPU memory usage
            target_gpu_memory = available_memory['gpu_available_mb'] * self.target_gpu_usage
            
            # Calculate optimal batch size
            optimal_batch_size = max(1, min(batch_size, int(target_gpu_memory / window_memory_mb)))
            
            # Calculate chunk dimensions
            max_chunk_height = min(height, int(np.sqrt(target_gpu_memory * 1024 * 1024 / (channels * 4))))
            max_chunk_width = min(width, max_chunk_height)
            
            # Ensure chunks are at least window_size
            chunk_height = max(window_size, max_chunk_height)
            chunk_width = max(window_size, max_chunk_width)
            
            # Round to nearest multiple of window_size for efficiency
            chunk_height = (chunk_height // window_size) * window_size
            chunk_width = (chunk_width // window_size) * window_size
            
            result = {
                'chunk_height': chunk_height,
                'chunk_width': chunk_width,
                'optimal_batch_size': optimal_batch_size,
                'estimated_memory_mb': batch_memory_mb,
                'available_memory_mb': target_gpu_memory
            }
            
            logger.info(f"📏 Optimal chunk size: {chunk_height}x{chunk_width}, batch_size={optimal_batch_size}")
            logger.info(f"💾 Estimated memory: {batch_memory_mb:.1f}MB / {target_gpu_memory:.1f}MB available")
            
            return result
            
        except Exception as e:
            logger.error(f"Chunk size calculation failed: {e}")
            # Fallback to conservative values
            return {
                'chunk_height': window_size * 2,
                'chunk_width': window_size * 2,
                'optimal_batch_size': 1,
                'estimated_memory_mb': 0.0,
                'available_memory_mb': 0.0
            }
    
    def create_tensor_pool(self, shape: Tuple[int, ...], pool_size: int = 10) -> None:
        """
        Create a pool of reusable tensors.
        
        Args:
            shape: Tensor shape
            pool_size: Number of tensors in pool
        """
        try:
            if shape not in self.tensor_pools:
                self.tensor_pools[shape] = []
                
                for _ in range(pool_size):
                    # Use torch.empty() for better performance
                    tensor = torch.empty(shape, dtype=torch.float32, device=self.device)
                    self.tensor_pools[shape].append(tensor)
                
                logger.info(f"🏊 Created tensor pool: {shape} x {pool_size}")
                
        except Exception as e:
            logger.error(f"Tensor pool creation failed: {e}")
    
    def get_tensor_from_pool(self, shape: Tuple[int, ...]) -> Optional[torch.Tensor]:
        """Get a tensor from pool or create new one if pool is empty."""
        try:
            if shape in self.tensor_pools and self.tensor_pools[shape]:
                tensor = self.tensor_pools[shape].pop()
                # Clear tensor data for reuse
                tensor.zero_()
                return tensor
            else:
                # Create new tensor if pool is empty
                tensor = torch.empty(shape, dtype=torch.float32, device=self.device)
                self.allocated_tensors.append(tensor)
                return tensor
                
        except Exception as e:
            logger.error(f"Tensor pool retrieval failed: {e}")
            return None
    
    def return_tensor_to_pool(self, tensor: torch.Tensor, shape: Tuple[int, ...]) -> None:
        """Return tensor to pool for reuse."""
        try:
            if shape not in self.tensor_pools:
                self.tensor_pools[shape] = []
            
            # Clear tensor data
            tensor.zero_()
            self.tensor_pools[shape].append(tensor)
            
        except Exception as e:
            logger.error(f"Tensor pool return failed: {e}")
    
    def chunk_image(self, image: torch.Tensor, chunk_height: int, chunk_width: int, 
                   overlap: int = 0) -> List[Tuple[torch.Tensor, Tuple[int, int]]]:
        """
        Split image into overlapping chunks for memory-efficient processing.
        
        Args:
            image: Input image tensor (C, H, W)
            chunk_height: Height of each chunk
            chunk_width: Width of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of (chunk, (row_offset, col_offset)) tuples
        """
        try:
            start_time = time.time()
            channels, height, width = image.shape
            
            chunks = []
            
            # Calculate step sizes
            step_height = chunk_height - overlap
            step_width = chunk_width - overlap
            
            # Generate chunk coordinates
            for row in range(0, height, step_height):
                for col in range(0, width, step_width):
                    # Calculate chunk boundaries
                    row_start = row
                    row_end = min(row + chunk_height, height)
                    col_start = col
                    col_end = min(col + chunk_width, width)
                    
                    # Extract chunk
                    chunk = image[:, row_start:row_end, col_start:col_end]
                    
                    # Pad chunk if necessary
                    if chunk.shape[1] < chunk_height or chunk.shape[2] < chunk_width:
                        padded_chunk = torch.zeros(channels, chunk_height, chunk_width, 
                                                 dtype=chunk.dtype, device=chunk.device)
                        padded_chunk[:, :chunk.shape[1], :chunk.shape[2]] = chunk
                        chunk = padded_chunk
                    
                    chunks.append((chunk, (row_start, col_start)))
            
            elapsed = time.time() - start_time
            logger.info(f"✂️ Image chunked into {len(chunks)} chunks in {elapsed:.3f}s")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Image chunking failed: {e}")
            return [(image, (0, 0))]
    
    def optimize_data_type(self, tensor: torch.Tensor, target_dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Optimize tensor data type for memory efficiency.
        
        Args:
            tensor: Input tensor
            target_dtype: Target data type
            
        Returns:
            Optimized tensor
        """
        try:
            if tensor.dtype != target_dtype:
                # Use in-place conversion when possible
                if tensor.is_contiguous():
                    tensor = tensor.to(target_dtype, non_blocking=True)
                else:
                    tensor = tensor.contiguous().to(target_dtype, non_blocking=True)
            
            return tensor
            
        except Exception as e:
            logger.error(f"Data type optimization failed: {e}")
            return tensor
    
    def clear_memory(self, force_gc: bool = False) -> None:
        """
        Clear memory and perform garbage collection.
        
        Args:
            force_gc: Force garbage collection
        """
        try:
            # Clear tensor pools
            for shape, pool in self.tensor_pools.items():
                pool.clear()
            
            # Clear allocated tensors
            self.allocated_tensors.clear()
            
            # Clear GPU cache
            if self.device.type == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection if requested
            if force_gc:
                gc.collect()
            
            self._log_memory_status("memory_clear")
            logger.info("🧹 Memory cleared")
            
        except Exception as e:
            logger.error(f"Memory clearing failed: {e}")
    
    def get_memory_stats(self) -> Dict[str, Union[float, int]]:
        """Get comprehensive memory statistics."""
        try:
            current_status = self._log_memory_status("stats")
            
            return {
                'peak_memory_usage_mb': self.peak_memory_usage,
                'current_cpu_memory_mb': current_status.get('cpu_memory_mb', 0.0),
                'current_gpu_memory_mb': current_status.get('gpu_memory_mb', 0.0),
                'tensor_pools_count': len(self.tensor_pools),
                'allocated_tensors_count': len(self.allocated_tensors),
                'memory_history_length': len(self.memory_history),
                'total_operations': len(self.operation_times)
            }
            
        except Exception as e:
            logger.error(f"Memory stats retrieval failed: {e}")
            return {}
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup memory."""
        self.clear_memory(force_gc=True)


class ChunkedProcessor:
    """
    Chunked processing implementation for large images.
    """
    
    def __init__(self, memory_manager: MemoryManager):
        """
        Initialize chunked processor.
        
        Args:
            memory_manager: Memory manager instance
        """
        self.memory_manager = memory_manager
        self.device = memory_manager.device
    
    def process_image_chunks(self, image: torch.Tensor, 
                           window_size: int,
                           batch_size: int,
                           processor_func,
                           **kwargs) -> List:
        """
        Process image in chunks using the provided processor function.
        
        Args:
            image: Input image tensor
            window_size: Sliding window size
            batch_size: Batch size for processing
            processor_func: Function to process each chunk
            **kwargs: Additional arguments for processor function
            
        Returns:
            List of results from all chunks
        """
        try:
            # Calculate optimal chunk size
            chunk_params = self.memory_manager.calculate_optimal_chunk_size(
                image.shape, window_size, batch_size
            )
            
            # Split image into chunks
            chunks = self.memory_manager.chunk_image(
                image, 
                chunk_params['chunk_height'], 
                chunk_params['chunk_width']
            )
            
            results = []
            
            for i, (chunk, (row_offset, col_offset)) in enumerate(chunks):
                logger.info(f"🔄 Processing chunk {i+1}/{len(chunks)} at offset ({row_offset}, {col_offset})")
                
                # Process chunk
                chunk_result = processor_func(chunk, **kwargs)
                
                # Adjust coordinates in results
                if isinstance(chunk_result, list):
                    for result in chunk_result:
                        if isinstance(result, dict) and 'row' in result and 'col' in result:
                            result['row'] += row_offset
                            result['col'] += col_offset
                
                results.extend(chunk_result if isinstance(chunk_result, list) else [chunk_result])
                
                # Clear memory after each chunk
                if i % 5 == 0:  # Clear every 5 chunks
                    self.memory_manager.clear_memory()
            
            logger.info(f"✅ Processed {len(chunks)} chunks, got {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Chunked processing failed: {e}")
            return []


# Global memory manager instance
_global_memory_manager: Optional[MemoryManager] = None

def get_memory_manager(device: torch.device = None) -> MemoryManager:
    """Get global memory manager instance."""
    global _global_memory_manager
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if _global_memory_manager is None or _global_memory_manager.device != device:
        _global_memory_manager = MemoryManager(device)
    
    return _global_memory_manager
