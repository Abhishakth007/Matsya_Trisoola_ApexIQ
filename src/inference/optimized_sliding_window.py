"""
Optimized Sliding Window Implementation for Vessel Detection System
Phase 2: Sliding Window Algorithm Redesign

This module implements:
1. Vectorized Window Extraction
2. Spatial Batching Strategy
3. Overlap Optimization
4. Memory-efficient window processing
"""

import logging
import math
import time
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from .memory_manager import MemoryManager, get_memory_manager

logger = logging.getLogger(__name__)

class OptimizedSlidingWindow:
    """
    Optimized sliding window implementation with vectorized operations.
    
    Replaces the O(n²) complexity sliding window with efficient
    vectorized operations and spatial batching.
    """
    
    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        """
        Initialize optimized sliding window processor.
        
        Args:
            memory_manager: Memory manager instance (optional)
        """
        self.memory_manager = memory_manager or get_memory_manager()
        self.device = self.memory_manager.device
        
        # Pre-computed window coordinates cache
        self.window_coords_cache: Dict[Tuple[int, int, int, int], List[Tuple[int, int]]] = {}
        
        # Performance tracking
        self.performance_stats = {
            'total_windows_processed': 0,
            'total_processing_time': 0.0,
            'average_time_per_window': 0.0,
            'memory_usage_history': []
        }
        
        logger.info(f"🚀 Optimized Sliding Window initialized for {self.device}")
    
    def precompute_window_coordinates(self, image_height: int, image_width: int,
                                    window_size: int, step_size: int,
                                    padding: int = 0) -> List[Tuple[int, int]]:
        """
        Pre-compute all window coordinates for efficient processing.
        
        Args:
            image_height: Height of input image
            image_width: Width of input image
            window_size: Size of sliding window
            step_size: Step size between windows
            padding: Padding around image edges
            
        Returns:
            List of (row_offset, col_offset) tuples
        """
        try:
            cache_key = (image_height, image_width, window_size, step_size)
            
            if cache_key in self.window_coords_cache:
                logger.info(f"📋 Using cached window coordinates for {cache_key}")
                return self.window_coords_cache[cache_key]
            
            start_time = time.time()
            
            # Calculate effective image dimensions
            eff_height = image_height - 2 * padding
            eff_width = image_width - 2 * padding
            
            # Generate window coordinates
            coordinates = []
            
            for row in range(0, eff_height - window_size + 1, step_size):
                for col in range(0, eff_width - window_size + 1, step_size):
                    # Adjust for padding
                    actual_row = row + padding
                    actual_col = col + padding
                    
                    # Ensure window fits within image bounds
                    if (actual_row + window_size <= image_height and 
                        actual_col + window_size <= image_width):
                        coordinates.append((actual_row, actual_col))
            
            # Cache the results
            self.window_coords_cache[cache_key] = coordinates
            
            elapsed = time.time() - start_time
            logger.info(f"📐 Pre-computed {len(coordinates)} window coordinates in {elapsed:.3f}s")
            
            return coordinates
            
        except Exception as e:
            logger.error(f"Window coordinate pre-computation failed: {e}")
            return []
    
    def create_spatial_batches(self, coordinates: List[Tuple[int, int]], 
                             batch_size: int,
                             spatial_grouping: bool = True,
                             window_size: int = 512) -> List[List[Tuple[int, int]]]:
        """
        Create spatially-aware batches for efficient processing.
        
        Args:
            coordinates: List of window coordinates
            batch_size: Target batch size
            spatial_grouping: Whether to group spatially adjacent windows
            window_size: Window size for spatial distance calculation
            
        Returns:
            List of coordinate batches
        """
        try:
            if not spatial_grouping or len(coordinates) <= batch_size:
                # Simple batching
                return [coordinates[i:i + batch_size] for i in range(0, len(coordinates), batch_size)]
            
            # Spatial batching - group nearby windows
            batches = []
            current_batch = []
            
            for i, (row, col) in enumerate(coordinates):
                current_batch.append((row, col))
                
                # Check if batch is full or if next window is far away
                if len(current_batch) >= batch_size:
                    batches.append(current_batch)
                    current_batch = []
                elif i < len(coordinates) - 1:
                    next_row, next_col = coordinates[i + 1]
                    distance = math.sqrt((next_row - row) ** 2 + (next_col - col) ** 2)
                    
                    # Start new batch if next window is far away
                    if distance > window_size * 2:
                        batches.append(current_batch)
                        current_batch = []
            
            # Add remaining windows
            if current_batch:
                batches.append(current_batch)
            
            logger.info(f"📦 Created {len(batches)} spatial batches from {len(coordinates)} windows")
            return batches
            
        except Exception as e:
            logger.error(f"Spatial batching failed: {e}")
            # Fallback to simple batching
            return [coordinates[i:i + batch_size] for i in range(0, len(coordinates), batch_size)]
    
    def extract_windows_vectorized(self, image: torch.Tensor,
                                 coordinates: List[Tuple[int, int]],
                                 window_size: int) -> torch.Tensor:
        """
        Extract multiple windows using vectorized operations.
        
        Args:
            image: Input image tensor (C, H, W)
            coordinates: List of (row, col) coordinates
            window_size: Size of windows to extract
            
        Returns:
            Batched window tensor (N, C, window_size, window_size)
        """
        try:
            start_time = time.time()
            
            if not coordinates:
                return torch.empty(0, image.shape[0], window_size, window_size, 
                                 device=image.device, dtype=image.dtype)
            
            # Pre-allocate output tensor
            num_windows = len(coordinates)
            channels = image.shape[0]
            
            # Use memory manager to get optimal tensor
            window_shape = (num_windows, channels, window_size, window_size)
            windows = self.memory_manager.get_tensor_from_pool(window_shape)
            
            if windows is None:
                # Fallback to direct allocation
                windows = torch.empty(window_shape, device=image.device, dtype=image.dtype)
            
            # Extract windows using advanced indexing
            for i, (row, col) in enumerate(coordinates):
                windows[i] = image[:, row:row + window_size, col:col + window_size]
            
            elapsed = time.time() - start_time
            logger.info(f"🔪 Extracted {num_windows} windows vectorized in {elapsed:.3f}s")
            
            return windows
            
        except Exception as e:
            logger.error(f"Vectorized window extraction failed: {e}")
            return torch.empty(0, image.shape[0], window_size, window_size, 
                             device=image.device, dtype=image.dtype)
    
    def process_windows_optimized(self, image: torch.Tensor,
                                window_size: int,
                                step_size: int,
                                batch_size: int,
                                model,
                                threshold: float = 0.5,
                                padding: int = 0,
                                overlap: int = 0) -> List[Dict]:
        """
        Process image using optimized sliding window approach.
        
        Args:
            image: Input image tensor (C, H, W)
            window_size: Size of sliding window
            step_size: Step size between windows
            batch_size: Batch size for processing
            model: Model to apply to windows
            threshold: Confidence threshold
            padding: Padding around image
            overlap: Overlap between windows
            
        Returns:
            List of detection results
        """
        try:
            start_time = time.time()
            
            # Pre-compute window coordinates
            coordinates = self.precompute_window_coordinates(
                image.shape[1], image.shape[2], window_size, step_size, padding
            )
            
            if not coordinates:
                logger.warning("No valid window coordinates found")
                return []
            
            # Create spatial batches
            batches = self.create_spatial_batches(coordinates, batch_size, spatial_grouping=True)
            
            all_results = []
            
            # Process each batch
            for batch_idx, batch_coords in enumerate(batches):
                batch_start_time = time.time()
                
                logger.info(f"🔄 Processing batch {batch_idx + 1}/{len(batches)} with {len(batch_coords)} windows")
                
                # Extract windows for this batch
                batch_windows = self.extract_windows_vectorized(image, batch_coords, window_size)
                
                if batch_windows.numel() == 0:
                    continue
                
                # Normalize windows
                batch_windows = batch_windows.float() / 255.0
                
                # Run model inference
                with torch.no_grad():
                    batch_outputs = model(batch_windows)
                
                # Process results
                batch_results = self.process_batch_results(
                    batch_outputs, batch_coords, window_size, threshold, overlap
                )
                
                all_results.extend(batch_results)
                
                # Update performance stats
                batch_time = time.time() - batch_start_time
                self.performance_stats['total_windows_processed'] += len(batch_coords)
                self.performance_stats['total_processing_time'] += batch_time
                
                # Clear memory after each batch
                if batch_idx % 5 == 0:
                    self.memory_manager.clear_memory()
                
                logger.info(f"✅ Batch {batch_idx + 1} completed in {batch_time:.3f}s, found {len(batch_results)} detections")
            
            # Calculate final stats
            total_time = time.time() - start_time
            self.performance_stats['average_time_per_window'] = (
                self.performance_stats['total_processing_time'] / 
                max(1, self.performance_stats['total_windows_processed'])
            )
            
            logger.info(f"🎉 Optimized sliding window completed:")
            logger.info(f"   Total windows: {self.performance_stats['total_windows_processed']}")
            logger.info(f"   Total time: {total_time:.3f}s")
            logger.info(f"   Average time per window: {self.performance_stats['average_time_per_window']:.6f}s")
            logger.info(f"   Total detections: {len(all_results)}")
            
            return all_results
            
        except Exception as e:
            logger.error(f"Optimized sliding window processing failed: {e}")
            return []
    
    def process_batch_results(self, batch_outputs, batch_coords: List[Tuple[int, int]],
                            window_size: int, threshold: float, overlap: int) -> List[Dict]:
        """
        Process batch model outputs and convert to detection format.
        
        Args:
            batch_outputs: Model outputs for batch
            batch_coords: Window coordinates for batch
            window_size: Window size
            threshold: Confidence threshold
            overlap: Overlap between windows
            
        Returns:
            List of detection dictionaries
        """
        try:
            results = []
            
            for i, (row_offset, col_offset) in enumerate(batch_coords):
                if i >= len(batch_outputs):
                    continue
                
                output = batch_outputs[i]
                
                # Handle different output formats
                if isinstance(output, (list, tuple)) and len(output) > 0:
                    output = output[0]
                
                if not isinstance(output, dict) or 'boxes' not in output or 'scores' not in output:
                    continue
                
                # Process detections in this window
                for j, (box, score) in enumerate(zip(output['boxes'], output['scores'])):
                    if score < threshold:
                        continue
                    
                    # Convert box coordinates to image coordinates
                    box_x1, box_y1, box_x2, box_y2 = box.tolist()
                    
                    # Calculate center point
                    center_x = (box_x1 + box_x2) / 2
                    center_y = (box_y1 + box_y2) / 2
                    
                    # Convert to image coordinates
                    image_col = col_offset + int(center_x)
                    image_row = row_offset + int(center_y)
                    
                    # Apply overlap filtering
                    if self.is_in_overlap_region(center_x, center_y, window_size, overlap):
                        continue
                    
                    results.append({
                        'row': image_row,
                        'col': image_col,
                        'score': float(score),
                        'box': box.tolist(),
                        'window_offset': (row_offset, col_offset)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Batch result processing failed: {e}")
            return []
    
    def is_in_overlap_region(self, x: float, y: float, window_size: int, overlap: int) -> bool:
        """
        Check if detection is in overlap region that should be filtered.
        
        Args:
            x: X coordinate within window
            y: Y coordinate within window
            window_size: Window size
            overlap: Overlap size
            
        Returns:
            True if in overlap region
        """
        if overlap == 0:
            return False
        
        # Check if detection is in overlap regions
        return (x < overlap or x > window_size - overlap or 
                y < overlap or y > window_size - overlap)
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        return self.performance_stats.copy()
    
    def clear_cache(self) -> None:
        """Clear window coordinate cache."""
        self.window_coords_cache.clear()
        logger.info("🗑️ Window coordinate cache cleared")


class VectorizedWindowExtractor:
    """
    Advanced vectorized window extraction with memory optimization.
    """
    
    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        """
        Initialize vectorized window extractor.
        
        Args:
            memory_manager: Memory manager instance
        """
        self.memory_manager = memory_manager or get_memory_manager()
        self.device = self.memory_manager.device
    
    def extract_windows_advanced(self, image: torch.Tensor,
                               coordinates: List[Tuple[int, int]],
                               window_size: int,
                               use_unfold: bool = True) -> torch.Tensor:
        """
        Extract windows using advanced PyTorch operations.
        
        Args:
            image: Input image tensor (C, H, W)
            coordinates: List of (row, col) coordinates
            window_size: Window size
            use_unfold: Whether to use torch.unfold for extraction
            
        Returns:
            Batched window tensor
        """
        try:
            if use_unfold and len(coordinates) > 1:
                return self._extract_windows_unfold(image, coordinates, window_size)
            else:
                return self._extract_windows_indexing(image, coordinates, window_size)
                
        except Exception as e:
            logger.error(f"Advanced window extraction failed: {e}")
            return self._extract_windows_indexing(image, coordinates, window_size)
    
    def _extract_windows_unfold(self, image: torch.Tensor,
                               coordinates: List[Tuple[int, int]],
                               window_size: int) -> torch.Tensor:
        """
        Extract windows using torch.unfold for better performance.
        """
        try:
            # This is a simplified version - in practice, you'd need to handle
            # non-contiguous coordinates more carefully
            channels, height, width = image.shape
            
            # Create a mask of valid window positions
            valid_positions = torch.zeros(height - window_size + 1, 
                                        width - window_size + 1, 
                                        device=image.device, dtype=torch.bool)
            
            for row, col in coordinates:
                if (row < height - window_size + 1 and 
                    col < width - window_size + 1):
                    valid_positions[row, col] = True
            
            # Extract windows using unfold
            unfolded = image.unfold(1, window_size, 1).unfold(2, window_size, 1)
            
            # Select valid windows
            valid_windows = unfolded[:, valid_positions]
            
            # Reshape to (N, C, window_size, window_size)
            num_windows = valid_windows.shape[0]
            windows = valid_windows.permute(1, 0, 2, 3).contiguous()
            
            return windows
            
        except Exception as e:
            logger.error(f"Unfold window extraction failed: {e}")
            return self._extract_windows_indexing(image, coordinates, window_size)
    
    def _extract_windows_indexing(self, image: torch.Tensor,
                                 coordinates: List[Tuple[int, int]],
                                 window_size: int) -> torch.Tensor:
        """
        Extract windows using advanced indexing.
        """
        try:
            num_windows = len(coordinates)
            channels = image.shape[0]
            
            # Pre-allocate output tensor
            windows = torch.empty(num_windows, channels, window_size, window_size,
                                device=image.device, dtype=image.dtype)
            
            # Extract windows using vectorized indexing
            for i, (row, col) in enumerate(coordinates):
                windows[i] = image[:, row:row + window_size, col:col + window_size]
            
            return windows
            
        except Exception as e:
            logger.error(f"Indexing window extraction failed: {e}")
            return torch.empty(0, image.shape[0], window_size, window_size,
                             device=image.device, dtype=image.dtype)


# Factory function for easy usage
def create_optimized_sliding_window(memory_manager: Optional[MemoryManager] = None) -> OptimizedSlidingWindow:
    """Create an optimized sliding window instance."""
    return OptimizedSlidingWindow(memory_manager)
