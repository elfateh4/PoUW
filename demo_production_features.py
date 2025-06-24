#!/usr/bin/env python3
"""
Production Features Demo for PoUW

This script demonstrates all the production-ready capabilities of the PoUW system:
1. Real dataset integration with multiple formats
2. GPU acceleration and performance optimization
3. Large-scale model support (>14M parameters)
4. Cross-validation and model comparison
5. Comprehensive performance monitoring

Run with: python demo_production_features.py
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from pouw.production import (
    ProductionDatasetManager,
    PerformanceMonitor,
    GPUManager, 
    LargeModelArchitectures,
    CrossValidationManager,
    ModelArchitectures
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionFeaturesDemo:
    """Demo showcasing all production features"""
    
    def __init__(self):
        self.perf_monitor = PerformanceMonitor()
        self.gpu_manager = GPUManager()
        self.dataset_manager = ProductionDatasetManager()
        self.cv_manager = CrossValidationManager(self.gpu_manager)
        
        logger.info("Production Features Demo initialized")
    
    async def demo_dataset_integration(self):
        """Demonstrate real dataset integration capabilities"""
        logger.info("=== Dataset Integration Demo ===")
        
        try:
            # Show available datasets
            available = self.dataset_manager.list_available_datasets()
            logger.info(f"Available datasets: {list(available.keys())}")
            
            # Load MNIST dataset
            logger.info("Loading MNIST dataset...")
            mnist_metadata = self.dataset_manager.load_torchvision_dataset('mnist')
            logger.info(f"MNIST loaded: {mnist_metadata.size} samples, shape {mnist_metadata.input_shape}")
            
            # Split dataset
            logger.info("Splitting dataset...")
            splits = self.dataset_manager.split_dataset('mnist')
            logger.info(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
            
            # Create dataloaders
            logger.info("Creating dataloaders...")
            dataloaders = self.dataset_manager.create_dataloaders('mnist', batch_size=64)
            logger.info(f"Created dataloaders: {list(dataloaders.keys())}")
            
            # Cache dataset for future use
            self.dataset_manager.cache_dataset('mnist')
            logger.info("Dataset cached successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Dataset integration demo failed: {e}")
            return False
    
    async def demo_gpu_acceleration(self):
        """Demonstrate GPU acceleration capabilities"""
        logger.info("=== GPU Acceleration Demo ===")
        
        try:
            # Check GPU availability
            device_info = {
                'device': str(self.gpu_manager.device),
                'cuda_available': self.gpu_manager.device.type == 'cuda',
                'memory_info': self.gpu_manager.get_memory_info() if self.gpu_manager.device.type == 'cuda' else None
            }
            logger.info(f"GPU Info: {device_info}")
            
            # Create a simple model for testing
            import torch
            import torch.nn as nn
            
            model = nn.Sequential(
                nn.Linear(784, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
            )
            
            # Move to GPU if available
            model = self.gpu_manager.setup_model(model)
            logger.info(f"Model moved to device: {next(model.parameters()).device}")
            
            # Test GPU memory management
            if self.gpu_manager.device.type == 'cuda':
                memory_before = self.gpu_manager.get_memory_info()
                logger.info(f"Memory before allocation: {memory_before}")
                
                # Allocate some tensors
                large_tensor = torch.randn(1000, 1000).to(self.gpu_manager.device)
                
                memory_after = self.gpu_manager.get_memory_info()
                logger.info(f"Memory after allocation: {memory_after}")
                
                # Clean up
                del large_tensor
                torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"GPU acceleration demo failed: {e}")
            return False
    
    async def demo_large_models(self):
        """Demonstrate large-scale model support"""
        logger.info("=== Large Models Demo ===")
        
        try:
            # Create different large model architectures
            architectures = {
                'large_cnn': LargeModelArchitectures.create_large_cnn(num_classes=10, width_multiplier=2.0),
                'large_transformer': LargeModelArchitectures.create_transformer(
                    vocab_size=1000, d_model=512, nhead=8, num_layers=6
                ),
                'large_resnet': LargeModelArchitectures.create_large_resnet(num_classes=10, width_multiplier=2.0)
            }
            
            for name, model in architectures.items():
                param_count = sum(p.numel() for p in model.parameters())
                logger.info(f"{name}: {param_count:,} parameters")
                
                # Check if it's truly large-scale (>14M parameters)
                if param_count > 14_000_000:
                    logger.info(f"✓ {name} qualifies as large-scale model")
                else:
                    logger.info(f"  {name} is a medium-scale model")
            
            # Demonstrate memory estimation
            from pouw.production.large_models import LargeModelManager
            manager = LargeModelManager()
            
            sample_model = architectures['large_cnn']
            memory_req = manager.estimate_memory_requirements(sample_model, batch_size=32)
            logger.info(f"Estimated memory requirement: {memory_req:.2f} GB")
            
            return True
            
        except Exception as e:
            logger.error(f"Large models demo failed: {e}")
            return False
    
    async def demo_cross_validation(self):
        """Demonstrate cross-validation and model comparison"""
        logger.info("=== Cross-Validation Demo ===")
        
        try:
            # Get model architectures for comparison
            model_configs = ModelArchitectures.get_standard_architectures(data_type='images')
            logger.info(f"Available model architectures: {list(model_configs.keys())}")
            
            # For demo purposes, create simple synthetic data
            import torch
            from torch.utils.data import TensorDataset
            
            # Create synthetic MNIST-like data
            X = torch.randn(1000, 1, 28, 28)  # 1000 samples
            y = torch.randint(0, 10, (1000,))  # 10 classes
            dataset = TensorDataset(X, y)
            
            # Configure cross-validation
            self.cv_manager.configure_cv(
                dataset=dataset,
                model_configs=model_configs,
                n_folds=3,  # Reduced for demo
                metrics=['accuracy', 'f1_macro']
            )
            
            logger.info("Running cross-validation (this may take a while)...")
            
            # Run cross-validation on a subset of models for demo
            demo_models = ['simple_cnn', 'mlp']  # Just test these for speed
            demo_configs = {k: v for k, v in model_configs.items() if k in demo_models}
            
            # Override with simpler configs for demo
            self.cv_manager.configure_cv(
                dataset=dataset,
                model_configs=demo_configs,
                n_folds=3,
                metrics=['accuracy']
            )
            
            results = self.cv_manager.run_cross_validation(max_epochs=2)  # Quick training
            
            # Display results
            for model_name, result in results.items():
                mean_acc = result.mean_metrics['accuracy']
                std_acc = result.std_metrics['accuracy']
                logger.info(f"{model_name}: {mean_acc:.3f} ± {std_acc:.3f} accuracy")
            
            return True
            
        except Exception as e:
            logger.error(f"Cross-validation demo failed: {e}")
            return False
    
    async def demo_performance_monitoring(self):
        """Demonstrate comprehensive performance monitoring"""
        logger.info("=== Performance Monitoring Demo ===")
        
        try:
            # Start performance monitoring
            self.perf_monitor.start_monitoring()
            logger.info("Performance monitoring started")
            
            # Simulate some computational work
            logger.info("Simulating computational workload...")
            
            import torch
            import time
            
            # CPU-intensive task
            with self.perf_monitor.profile("matrix_multiplication"):
                for i in range(5):
                    a = torch.randn(500, 500)
                    b = torch.randn(500, 500)
                    c = torch.mm(a, b)
                    time.sleep(0.1)
            
            # Memory-intensive task
            with self.perf_monitor.profile("memory_allocation"):
                tensors = []
                for i in range(10):
                    tensor = torch.randn(100, 100, 100)
                    tensors.append(tensor)
                    time.sleep(0.05)
                
                # Clean up
                del tensors
            
            # Get performance metrics
            await asyncio.sleep(1)  # Let monitoring collect data
            
            # Get system health
            health = self.perf_monitor.get_system_health()
            logger.info(f"System Health - CPU: {health.cpu_usage:.1f}%, Memory: {health.memory_usage:.1f}%")
            
            # Get performance report
            report = self.perf_monitor.generate_performance_report()
            logger.info("Performance Report Generated:")
            
            if report['operations']:
                for op_name, metrics in report['operations'].items():
                    logger.info(f"  {op_name}: {metrics['mean_time']:.3f}s avg, {metrics['count']} runs")
            
            # Stop monitoring
            self.perf_monitor.stop_monitoring()
            logger.info("Performance monitoring stopped")
            
            return True
            
        except Exception as e:
            logger.error(f"Performance monitoring demo failed: {e}")
            return False
    
    async def demo_integration(self):
        """Demonstrate integration of all production features"""
        logger.info("=== Integration Demo ===")
        
        try:
            # This would be a real integration test with PoUW components
            # For now, just show that all components can work together
            
            logger.info("Testing integrated workflow...")
            
            # 1. Load dataset with monitoring
            with self.perf_monitor.profile("dataset_loading"):
                if 'mnist' not in self.dataset_manager.list_loaded_datasets():
                    self.dataset_manager.load_torchvision_dataset('mnist')
            
            # 2. Create model with GPU support
            model = LargeModelArchitectures.create_large_cnn(num_classes=10, width_multiplier=1.0)
            model = self.gpu_manager.setup_model(model)
            
            # 3. Get dataloaders
            dataloaders = self.dataset_manager.create_dataloaders('mnist', batch_size=32)
            
            # 4. Quick training simulation with monitoring
            with self.perf_monitor.profile("training_simulation"):
                # Simulate a few training steps
                import torch.nn as nn
                import torch.optim as optim
                
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                
                # Get a few batches
                train_loader = dataloaders['train']
                for i, (data, target) in enumerate(train_loader):
                    if i >= 3:  # Just a few batches for demo
                        break
                    
                    data, target = data.to(self.gpu_manager.device), target.to(self.gpu_manager.device)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    if i == 0:
                        logger.info(f"Training step {i+1}: loss = {loss.item():.4f}")
            
            logger.info("✓ All production features integrated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Integration demo failed: {e}")
            return False
    
    async def run_all_demos(self):
        """Run all production feature demonstrations"""
        logger.info("🚀 Starting Production Features Demo")
        logger.info("=" * 60)
        
        demos = [
            ("Dataset Integration", self.demo_dataset_integration),
            ("GPU Acceleration", self.demo_gpu_acceleration),
            ("Large Models", self.demo_large_models),
            ("Cross-Validation", self.demo_cross_validation),
            ("Performance Monitoring", self.demo_performance_monitoring),
            ("Integration", self.demo_integration)
        ]
        
        results = {}
        start_time = time.time()
        
        for demo_name, demo_func in demos:
            logger.info(f"\n🔄 Running {demo_name} Demo...")
            try:
                success = await demo_func()
                results[demo_name] = "✅ SUCCESS" if success else "❌ FAILED"
            except Exception as e:
                logger.error(f"Demo {demo_name} crashed: {e}")
                results[demo_name] = "💥 CRASHED"
        
        total_time = time.time() - start_time
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("📊 PRODUCTION FEATURES DEMO RESULTS")
        logger.info("=" * 60)
        
        for demo_name, result in results.items():
            logger.info(f"{demo_name:<25}: {result}")
        
        success_count = sum(1 for r in results.values() if r == "✅ SUCCESS")
        total_count = len(results)
        
        logger.info(f"\nSummary: {success_count}/{total_count} demos successful")
        logger.info(f"Total time: {total_time:.2f} seconds")
        
        if success_count == total_count:
            logger.info("🎉 All production features are working correctly!")
        else:
            logger.warning("⚠️  Some production features need attention")
        
        return success_count == total_count


async def main():
    """Main demo function"""
    try:
        demo = ProductionFeaturesDemo()
        success = await demo.run_all_demos()
        
        if success:
            logger.info("\n✅ Production Features Demo completed successfully!")
            return 0
        else:
            logger.error("\n❌ Some production features failed")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Demo interrupted by user")
        return 2
    except Exception as e:
        logger.error(f"\n💥 Demo crashed with error: {e}")
        return 3


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
