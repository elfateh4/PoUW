"""
GPU Acceleration Support for PoUW

This module provides GPU acceleration capabilities for:
- Neural network training and inference
- Mining operations
- Verification processes
- Data processing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Optional, Any, Tuple
import logging
import warnings
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class GPUManager:
    """
    Manages GPU resources and acceleration for PoUW operations
    """
    
    def __init__(self, device_preference: Optional[str] = None):
        self.device_preference = device_preference
        self.device = self._initialize_device()
        self.scaler = GradScaler() if self.device.type == 'cuda' else None
        self.memory_fraction = 0.8  # Reserve 80% of GPU memory
        
        self._log_gpu_info()
    
    def _initialize_device(self) -> torch.device:
        """Initialize and configure GPU device"""
        
        if self.device_preference:
            try:
                device = torch.device(self.device_preference)
                if device.type == 'cuda' and not torch.cuda.is_available():
                    logger.warning(f"CUDA requested but not available, falling back to CPU")
                    device = torch.device('cpu')
                return device
            except:
                logger.warning(f"Invalid device {self.device_preference}, using auto-detection")
        
        # Auto-detect best device
        if torch.cuda.is_available():
            # Select GPU with most memory
            best_gpu = 0
            max_memory = 0
            
            for i in range(torch.cuda.device_count()):
                memory = torch.cuda.get_device_properties(i).total_memory
                if memory > max_memory:
                    max_memory = memory
                    best_gpu = i
            
            device = torch.device(f'cuda:{best_gpu}')
            torch.cuda.set_device(device)
            
            # Set memory fraction
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(self.memory_fraction, device)
            
            return device
        else:
            return torch.device('cpu')
    
    def _log_gpu_info(self):
        """Log GPU information"""
        logger.info(f"Using device: {self.device}")
        
        if self.device.type == 'cuda':
            gpu_props = torch.cuda.get_device_properties(self.device)
            memory_gb = gpu_props.total_memory / 1024**3
            
            logger.info(f"GPU: {gpu_props.name}")
            logger.info(f"GPU Memory: {memory_gb:.1f} GB")
            try:
                cuda_version = torch.version.cuda  # type: ignore
                logger.info(f"CUDA Version: {cuda_version}")
            except AttributeError:
                logger.info("CUDA Version: Unknown")
            logger.info(f"GPU Count: {torch.cuda.device_count()}")
            
            if self.scaler:
                logger.info("Mixed precision training enabled")
        else:
            logger.info("Using CPU - consider GPU acceleration for better performance")
    
    @property
    def is_gpu_available(self) -> bool:
        """Check if GPU is available and being used"""
        return self.device.type == 'cuda'
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get detailed device information"""
        info = {
            'device': str(self.device),
            'device_type': self.device.type,
            'is_cuda': self.device.type == 'cuda',
            'mixed_precision': self.scaler is not None
        }
        
        if self.device.type == 'cuda':
            props = torch.cuda.get_device_properties(self.device)
            try:
                cuda_version = torch.version.cuda  # type: ignore
            except AttributeError:
                cuda_version = 'Unknown'
            info.update({
                'gpu_name': props.name,
                'total_memory_gb': props.total_memory / 1024**3,
                'allocated_memory_gb': torch.cuda.memory_allocated(self.device) / 1024**3,
                'reserved_memory_gb': torch.cuda.memory_reserved(self.device) / 1024**3,
                'device_count': torch.cuda.device_count(),
                'cuda_version': cuda_version,
                'compute_capability': f"{props.major}.{props.minor}"
            })
        
        return info
    
    def optimize_model_for_gpu(self, model: nn.Module) -> nn.Module:
        """Optimize model for GPU acceleration"""
        
        model = model.to(self.device)
        
        if self.device.type == 'cuda':
            # Enable optimizations
            if hasattr(torch.backends.cudnn, 'benchmark'):
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            
            # Consider model parallelism for large models
            if torch.cuda.device_count() > 1:
                logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
                model = nn.DataParallel(model)
        
        return model
    
    def optimize_optimizer(self, optimizer: optim.Optimizer) -> optim.Optimizer:
        """Optimize optimizer for GPU"""
        
        if self.device.type == 'cuda':
            # Enable specific CUDA optimizations
            for param_group in optimizer.param_groups:
                for param in param_group['params']:
                    if param.device != self.device:
                        param.data = param.data.to(self.device)
                        if param.grad is not None:
                            param.grad.data = param.grad.data.to(self.device)
        
        return optimizer
    
    @contextmanager
    def autocast_context(self):
        """Context manager for automatic mixed precision"""
        if self.device.type == 'cuda' and self.scaler:
            with autocast():
                yield
        else:
            yield
    
    def scale_loss_and_backward(self, loss: torch.Tensor, optimizer: optim.Optimizer):
        """Handle scaled backward pass for mixed precision"""
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad()
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    def clear_cache(self):
        """Clear GPU memory cache"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage"""
        if self.device.type == 'cuda':
            return {
                'allocated_gb': torch.cuda.memory_allocated(self.device) / 1024**3,
                'reserved_gb': torch.cuda.memory_reserved(self.device) / 1024**3,
                'max_allocated_gb': torch.cuda.max_memory_allocated(self.device) / 1024**3,
                'max_reserved_gb': torch.cuda.max_memory_reserved(self.device) / 1024**3
            }
        else:
            return {
                'allocated_gb': 0.0,
                'reserved_gb': 0.0,
                'max_allocated_gb': 0.0,
                'max_reserved_gb': 0.0
            }


class GPUAcceleratedTrainer:
    """
    GPU-accelerated training wrapper for PoUW ML training
    """
    
    def __init__(self, gpu_manager: GPUManager, enable_amp: bool = True):
        self.gpu_manager = gpu_manager
        self.enable_amp = enable_amp and gpu_manager.is_gpu_available
        self.device = gpu_manager.device
        
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare model for GPU training"""
        model = self.gpu_manager.optimize_model_for_gpu(model)
        
        # Additional GPU-specific optimizations
        if self.gpu_manager.is_gpu_available:
            # Compile model for newer PyTorch versions
            if hasattr(torch, 'compile') and hasattr(model, 'forward'):
                try:
                    model = torch.compile(model)  # type: ignore
                    logger.info("Model compiled for GPU optimization")
                except Exception as e:
                    logger.warning(f"Model compilation failed: {e}")
        
        return model
    
    def prepare_data(self, data: torch.Tensor) -> torch.Tensor:
        """Prepare data for GPU processing"""
        if isinstance(data, torch.Tensor):
            return data.to(self.device, non_blocking=True)
        elif isinstance(data, (list, tuple)):
            return [self.prepare_data(item) for item in data]
        elif isinstance(data, dict):
            return {key: self.prepare_data(value) for key, value in data.items()}
        else:
            return data
    
    def train_batch(self, model: nn.Module, batch_data: Tuple[torch.Tensor, torch.Tensor], 
                   optimizer: optim.Optimizer, criterion: nn.Module) -> Dict[str, Any]:
        """Train single batch with GPU acceleration"""
        
        inputs, targets = batch_data
        inputs = self.prepare_data(inputs)
        targets = self.prepare_data(targets)
        
        optimizer.zero_grad()
        
        with self.gpu_manager.autocast_context():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # Handle backward pass with mixed precision
        if self.enable_amp and self.gpu_manager.scaler:
            self.gpu_manager.scaler.scale(loss).backward()
            self.gpu_manager.scaler.step(optimizer)
            self.gpu_manager.scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            if outputs.dim() > 1 and outputs.size(1) > 1:
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == targets).float().mean().item()
            else:
                accuracy = 0.0
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'memory_usage': self.gpu_manager.get_memory_usage()
        }
    
    def validate_batch(self, model: nn.Module, batch_data: Tuple[torch.Tensor, torch.Tensor], 
                      criterion: nn.Module) -> Dict[str, float]:
        """Validate single batch with GPU acceleration"""
        
        inputs, targets = batch_data
        inputs = self.prepare_data(inputs)
        targets = self.prepare_data(targets)
        
        model.eval()
        with torch.no_grad():
            with self.gpu_manager.autocast_context():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # Calculate accuracy
            if outputs.dim() > 1 and outputs.size(1) > 1:
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == targets).float().mean().item()
            else:
                accuracy = 0.0
        
        model.train()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy
        }


class GPUAcceleratedMiner:
    """
    GPU-accelerated mining operations for PoUW
    """
    
    def __init__(self, gpu_manager: GPUManager):
        self.gpu_manager = gpu_manager
        self.device = gpu_manager.device
    
    def accelerate_gradient_computation(self, model: nn.Module, 
                                      gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """Accelerate gradient computations on GPU"""
        
        if not self.gpu_manager.is_gpu_available:
            return gradients
        
        # Move gradients to GPU for parallel processing
        gpu_gradients = []
        for grad in gradients:
            if isinstance(grad, torch.Tensor):
                gpu_gradients.append(grad.to(self.device, non_blocking=True))
            else:
                gpu_gradients.append(torch.tensor(grad, device=self.device))
        
        # Parallel gradient processing
        with torch.no_grad():
            # Example: parallel norm computation
            grad_norms = torch.stack([torch.norm(grad) for grad in gpu_gradients])
            
            # Move back to CPU if needed
            processed_gradients = [grad.cpu() for grad in gpu_gradients]
        
        return processed_gradients
    
    def accelerate_nonce_generation(self, model_weights: torch.Tensor, 
                                   gradients: torch.Tensor, 
                                   target_difficulty: int) -> Optional[int]:
        """GPU-accelerated nonce generation for mining"""
        
        if not self.gpu_manager.is_gpu_available:
            return None
        
        # Move data to GPU
        weights_gpu = model_weights.to(self.device, non_blocking=True)
        gradients_gpu = gradients.to(self.device, non_blocking=True)
        
        # Parallel nonce search
        with torch.no_grad():
            # Create batch of potential nonces
            batch_size = 1024  # Process multiple nonces in parallel
            nonce_candidates = torch.arange(batch_size, device=self.device)
            
            # Vectorized hash computation simulation
            # In practice, this would be actual hash computation
            combined_data = torch.cat([
                weights_gpu.flatten(),
                gradients_gpu.flatten(),
                nonce_candidates.float()
            ], dim=0)
            
            # Simulate finding valid nonce
            # Real implementation would use actual hashing
            hash_values = torch.sum(combined_data) % (2**target_difficulty)
            
            if hash_values.item() < target_difficulty:
                return int(nonce_candidates[0].item())
        
        return None


class GPUMemoryManager:
    """
    Manages GPU memory allocation and optimization
    """
    
    def __init__(self, gpu_manager: GPUManager):
        self.gpu_manager = gpu_manager
        self.memory_checkpoints: List[Dict[str, Any]] = []
    
    def create_checkpoint(self, name: str):
        """Create memory usage checkpoint"""
        if self.gpu_manager.is_gpu_available:
            usage = self.gpu_manager.get_memory_usage().copy()
            # Store additional metadata separately to avoid type conflicts
            checkpoint_data = {
                **usage,
                'checkpoint_name': name,
                'timestamp': torch.cuda.Event(enable_timing=True) if self.gpu_manager.is_gpu_available else None
            }
            self.memory_checkpoints.append(checkpoint_data)
            logger.debug(f"Memory checkpoint '{name}': {usage['allocated_gb']:.2f} GB allocated")
    
    def optimize_memory_usage(self, target_memory_gb: Optional[float] = None):
        """Optimize GPU memory usage"""
        if not self.gpu_manager.is_gpu_available:
            return
        
        current_usage = self.gpu_manager.get_memory_usage()
        
        if target_memory_gb and current_usage['allocated_gb'] > target_memory_gb:
            # Clear cache and force garbage collection
            self.gpu_manager.clear_cache()
            
            # Additional optimizations
            torch.cuda.synchronize()
            
            logger.info(f"Memory optimization applied. "
                       f"Usage: {current_usage['allocated_gb']:.2f} GB")
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Generate memory usage report"""
        if not self.gpu_manager.is_gpu_available:
            return {'status': 'GPU not available'}
        
        current = self.gpu_manager.get_memory_usage()
        
        report = {
            'current_usage': current,
            'checkpoints': self.memory_checkpoints[-10:],  # Last 10 checkpoints
            'recommendations': []
        }
        
        # Generate recommendations
        if current['allocated_gb'] > 0.8 * current.get('total_gb', float('inf')):
            report['recommendations'].append("High memory usage - consider reducing batch size")
        
        if len(self.memory_checkpoints) > 1:
            growth = current['allocated_gb'] - self.memory_checkpoints[-2]['allocated_gb']
            if growth > 1.0:  # >1GB growth
                report['recommendations'].append("Rapid memory growth detected - check for memory leaks")
        
        return report


# Utility functions for PoUW integration
def create_gpu_manager(device_preference: Optional[str] = None) -> GPUManager:
    """Factory function to create GPU manager"""
    return GPUManager(device_preference)

def enable_gpu_optimizations():
    """Enable global GPU optimizations"""
    if torch.cuda.is_available():
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        # Set memory management
        torch.cuda.empty_cache()
        
        logger.info("GPU optimizations enabled")
        return True
    else:
        logger.warning("GPU not available - optimizations skipped")
        return False

def benchmark_gpu_performance(gpu_manager: GPUManager, 
                            model: nn.Module, 
                            sample_input: torch.Tensor) -> Dict[str, Any]:
    """Benchmark GPU performance for a model"""
    
    model = gpu_manager.optimize_model_for_gpu(model)
    sample_input = sample_input.to(gpu_manager.device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(sample_input)
    
    torch.cuda.synchronize() if gpu_manager.is_gpu_available else None
    
    # Benchmark inference
    start_time = torch.cuda.Event(enable_timing=True) if gpu_manager.is_gpu_available else None
    end_time = torch.cuda.Event(enable_timing=True) if gpu_manager.is_gpu_available else None
    
    import time
    cpu_start = time.time()
    
    if start_time:
        start_time.record()  # type: ignore
    
    with torch.no_grad():
        for _ in range(100):
            _ = model(sample_input)
    
    if end_time and start_time:
        end_time.record()  # type: ignore
        torch.cuda.synchronize()
        gpu_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
    else:
        gpu_time = time.time() - cpu_start
    
    avg_inference_time = gpu_time / 100
    throughput = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
    
    return {
        'average_inference_time_ms': avg_inference_time * 1000,
        'throughput_samples_per_second': throughput,
        'device': str(gpu_manager.device),
        'memory_usage_gb': gpu_manager.get_memory_usage().get('allocated_gb', 0)
    }
