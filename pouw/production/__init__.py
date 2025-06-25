"""
Production-ready features for PoUW blockchain implementation.

This module provides enterprise-grade capabilities including:
- Real dataset integration with multiple formats
- GPU acceleration and performance optimization
- Large-scale model support (>14M parameters)
- Cross-validation and model comparison
- Comprehensive performance monitoring
"""

from .datasets import ProductionDatasetManager, DatasetMetadata

from .monitoring import (
    PerformanceMonitor,
    PerformanceProfiler,
    OptimizationManager,
    PerformanceMetrics,
    SystemHealth,
    monitor_mining_performance,
    monitor_training_performance,
    monitor_verification_performance,
)

from .gpu_acceleration import (
    GPUManager,
    GPUAcceleratedTrainer,
    GPUAcceleratedMiner,
    GPUMemoryManager,
)

from .large_models import LargeModelArchitectures, LargeModelManager, ModelConfig

from .cross_validation import (
    ModelArchitectures,
    CrossValidationManager,
    HyperparameterOptimizer,
    ModelArchitectureConfig,
    ValidationResults,
    HyperparameterConfig,
)

__all__ = [
    # Dataset management
    "ProductionDatasetManager",
    "DatasetMetadata",
    # Performance monitoring
    "PerformanceMonitor",
    "PerformanceProfiler",
    "OptimizationManager",
    "PerformanceMetrics",
    "SystemHealth",
    "monitor_mining_performance",
    "monitor_training_performance",
    "monitor_verification_performance",
    # GPU acceleration
    "GPUManager",
    "GPUAcceleratedTrainer",
    "GPUAcceleratedMiner",
    "GPUMemoryManager",
    # Large model support
    "LargeModelArchitectures",
    "LargeModelManager",
    "ModelConfig",
    # Cross-validation
    "ModelArchitectures",
    "CrossValidationManager",
    "HyperparameterOptimizer",
    "ModelArchitectureConfig",
    "ValidationResults",
    "HyperparameterConfig",
]

# Version info
__version__ = "1.0.0"
__author__ = "PoUW Development Team"
