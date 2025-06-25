#!/usr/bin/env python3
"""
Production Features Showcase for PoUW

This demo showcases all the production features implemented for the PoUW system:
1. Real Dataset Integration
2. Performance Monitoring
3. GPU Acceleration
4. Large-Scale Model Support
5. Cross-Validation and Multiple Architectures

Run this demo to see the production capabilities in action.
"""

import asyncio
import logging
import time
import torch
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import PoUW production features
from pouw.production import (
    ProductionDatasetManager,
    PerformanceMonitor,
    GPUManager,
    LargeModelArchitectures,
    CrossValidationManager,
    ModelArchitectures,
)


class ProductionShowcase:
    """Comprehensive demonstration of PoUW production features"""

    def __init__(self):
        self.temp_dir = Path("/tmp/pouw_demo")
        self.temp_dir.mkdir(exist_ok=True)

        # Initialize production components
        self.dataset_manager = ProductionDatasetManager(
            data_root=str(self.temp_dir / "data"), cache_dir=str(self.temp_dir / "cache")
        )
        self.perf_monitor = PerformanceMonitor()
        self.gpu_manager = GPUManager()
        self.cv_manager = CrossValidationManager(self.gpu_manager)

    def print_section(self, title: str):
        """Print a formatted section header"""
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    def demonstrate_dataset_integration(self):
        """Demonstrate real dataset integration capabilities"""
        self.print_section("1. Real Dataset Integration")

        print("Available datasets:")
        available = self.dataset_manager.list_available_datasets()
        for name, desc in available.items():
            print(f"  - {name}: {desc}")

        print("\n🔄 Loading MNIST dataset...")
        try:
            with self.perf_monitor.profile("dataset_loading"):
                metadata = self.dataset_manager.load_torchvision_dataset("mnist", download=True)

            print(f"✅ Dataset loaded successfully!")
            print(f"   Name: {metadata.name}")
            print(f"   Size: {metadata.size:,} samples")
            print(f"   Classes: {metadata.num_classes}")
            print(f"   Input shape: {metadata.input_shape}")
            print(f"   Data format: {metadata.data_format}")

        except Exception as e:
            print(f"⚠️  MNIST download failed (no internet?): {e}")
            print("   Creating synthetic dataset instead...")

            # Create synthetic dataset for demo
            X = torch.randn(1000, 1, 28, 28)
            y = torch.randint(0, 10, (1000,))
            synthetic_dataset = torch.utils.data.TensorDataset(X, y)
            print(f"✅ Synthetic dataset created with {len(synthetic_dataset)} samples")
            return synthetic_dataset

        # Get actual dataset
        dataloaders = self.dataset_manager.create_dataloaders("mnist", batch_size=64)
        train_dataset = dataloaders["train"].dataset
        return train_dataset

    def demonstrate_performance_monitoring(self):
        """Demonstrate performance monitoring capabilities"""
        self.print_section("2. Performance Monitoring & Optimization")

        print("🔄 Starting performance monitoring...")
        self.perf_monitor.start_monitoring()

        # Simulate some operations
        operations = [
            ("cpu_intensive_task", 0.5),
            ("memory_allocation", 0.3),
            ("quick_computation", 0.1),
        ]

        for op_name, duration in operations:
            print(f"   Running {op_name}...")
            with self.perf_monitor.profile(op_name):
                # Simulate work
                start_time = time.time()
                while time.time() - start_time < duration:
                    _ = torch.randn(100, 100) @ torch.randn(100, 100)

        self.perf_monitor.stop_monitoring()

        print("✅ Performance monitoring complete!")

        # Generate performance report
        report = self.perf_monitor.generate_performance_report()
        print(f"   Operations monitored: {len(report['operation_performance'])}")

        for op_name, metrics in report["operation_performance"].items():
            exec_time = metrics["execution_time"]["mean"]
            cpu_usage = metrics["cpu_usage"]["mean"]
            print(f"   - {op_name}: {exec_time:.3f}s, CPU: {cpu_usage:.1f}%")

        # System health
        health = self.perf_monitor.collect_system_health()
        print(f"\n📊 System Health:")
        print(f"   CPU Usage: {health.cpu_usage:.1f}%")
        print(f"   Memory Usage: {health.memory_usage:.1f}%")
        print(f"   Disk Usage: {health.disk_usage:.1f}%")

        if health.gpu_info:
            print(f"   GPU Available: Yes")
            for i, gpu in enumerate(health.gpu_info.get("devices", [])):
                print(f"   GPU {i}: {gpu.get('name', 'Unknown')}")
        else:
            print(f"   GPU Available: No")

    def demonstrate_gpu_acceleration(self):
        """Demonstrate GPU acceleration capabilities"""
        self.print_section("3. GPU Acceleration")

        print(f"🔄 GPU Manager initialized")
        print(f"   Device: {self.gpu_manager.device}")
        print(f"   Device type: {self.gpu_manager.device.type}")

        if self.gpu_manager.device.type == "cuda":
            print(f"   GPU Name: {torch.cuda.get_device_name()}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Test GPU acceleration with model training
        print(f"\n🔄 Testing GPU acceleration...")

        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(784, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
        )

        # Move to GPU
        model = model.to(self.gpu_manager.device)
        print(f"   Model moved to: {next(model.parameters()).device}")

        # Test forward pass
        with self.perf_monitor.profile("gpu_forward_pass"):
            batch_size = 1000
            x = torch.randn(batch_size, 784).to(self.gpu_manager.device)
            y = model(x)

        print(f"✅ GPU forward pass completed")
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {y.shape}")
        print(f"   Input device: {x.device}")
        print(f"   Output device: {y.device}")

    def demonstrate_large_models(self):
        """Demonstrate large-scale model support"""
        self.print_section("4. Large-Scale Model Support")

        print("🔄 Creating large-scale models...")

        # Large CNN
        print("\n📊 Large CNN Model:")
        large_cnn = LargeModelArchitectures.create_large_cnn(
            num_classes=1000, input_channels=3, width_multiplier=1.5
        )
        cnn_params = sum(p.numel() for p in large_cnn.parameters())
        print(f"   Parameters: {cnn_params:,}")
        print(f"   Memory estimate: ~{cnn_params * 4 / 1e6:.1f} MB")

        # Transformer model
        print("\n📊 Large Transformer Model:")
        transformer = LargeModelArchitectures.create_transformer_model(
            vocab_size=10000, d_model=512, num_heads=8, num_layers=6
        )
        transformer_params = sum(p.numel() for p in transformer.parameters())
        print(f"   Parameters: {transformer_params:,}")
        print(f"   Memory estimate: ~{transformer_params * 4 / 1e6:.1f} MB")

        # ResNet model
        print("\n📊 Large ResNet Model:")
        resnet = LargeModelArchitectures.create_large_resnet(
            num_classes=1000, layers=[3, 8, 36, 3], width_multiplier=1.2  # ResNet-152 variant
        )
        resnet_params = sum(p.numel() for p in resnet.parameters())
        print(f"   Parameters: {resnet_params:,}")
        print(f"   Memory estimate: ~{resnet_params * 4 / 1e6:.1f} MB")

        print(f"\n✅ Large models created successfully!")
        total_params = cnn_params + transformer_params + resnet_params
        print(f"   Total parameters across all models: {total_params:,}")

        # Test gradient checkpointing
        print(f"\n🔄 Testing gradient checkpointing...")
        large_cnn.enable_gradient_checkpointing()
        transformer.enable_gradient_checkpointing()
        resnet.enable_gradient_checkpointing()
        print(f"✅ Gradient checkpointing enabled for memory efficiency")

        return large_cnn

    def demonstrate_cross_validation(self, dataset):
        """Demonstrate cross-validation and multiple architectures"""
        self.print_section("5. Cross-Validation & Multiple Architectures")

        print("🔄 Setting up cross-validation experiment...")

        # Register standard architectures for image data
        input_info = {"type": "image", "input_channels": 1, "num_classes": 10}
        self.cv_manager.register_standard_architectures(input_info)

        print(f"   Registered architectures: {len(self.cv_manager.model_configs)}")
        for name in self.cv_manager.model_configs.keys():
            print(f"   - {name}")

        # Create small dataset for quick demo
        print(f"\n🔄 Preparing dataset for cross-validation...")
        if hasattr(dataset, "__len__") and len(dataset) > 500:
            # Use subset for quick demo
            indices = torch.randperm(len(dataset))[:500].tolist()
            subset = torch.utils.data.Subset(dataset, indices)
            demo_dataset = subset
        else:
            demo_dataset = dataset

        print(f"   Demo dataset size: {len(demo_dataset)}")

        # Run quick cross-validation (2-fold, 3 epochs for speed)
        print(f"\n🔄 Running cross-validation (2-fold, 3 epochs)...")

        start_time = time.time()
        with self.perf_monitor.profile("cross_validation"):
            results = self.cv_manager.run_cross_validation(
                demo_dataset, k_folds=2, epochs=3, batch_size=64, stratified=True
            )
        cv_time = time.time() - start_time

        print(f"✅ Cross-validation completed in {cv_time:.1f}s")

        # Display results
        print(f"\n📊 Cross-Validation Results:")
        for model_name, result in results.items():
            acc = result.mean_metrics["accuracy"]
            std = result.std_metrics["accuracy"]
            params = result.total_parameters
            time_taken = result.training_time

            print(f"   {model_name}:")
            print(f"     Accuracy: {acc:.4f} ± {std:.4f}")
            print(f"     Parameters: {params:,}")
            print(f"     Training time: {time_taken:.1f}s")

        # Get best model
        best_model, best_results = self.cv_manager.get_best_model("accuracy")
        print(f"\n🏆 Best Model: {best_model}")
        print(f"   Accuracy: {best_results.mean_metrics['accuracy']:.4f}")
        print(f"   Parameters: {best_results.total_parameters:,}")

        return results

    def generate_comprehensive_report(self, cv_results):
        """Generate a comprehensive report of all production features"""
        self.print_section("📋 Comprehensive Production Features Report")

        # Performance summary
        perf_report = self.perf_monitor.generate_performance_report()

        print("🔧 Performance Summary:")
        print(f"   Total operations monitored: {len(perf_report['operation_performance'])}")

        total_time = sum(
            metrics["execution_time"]["mean"]
            for metrics in perf_report["operation_performance"].values()
        )
        print(f"   Total execution time: {total_time:.2f}s")

        # System capabilities
        print(f"\n💻 System Capabilities:")
        print(f"   Device: {self.gpu_manager.device}")
        print(f"   GPU Acceleration: {'Yes' if self.gpu_manager.device.type == 'cuda' else 'No'}")

        health = self.perf_monitor.collect_system_health()
        print(f"   Current CPU Usage: {health.cpu_usage:.1f}%")
        print(f"   Current Memory Usage: {health.memory_usage:.1f}%")

        # Dataset capabilities
        print(f"\n📊 Dataset Support:")
        available_datasets = self.dataset_manager.list_available_datasets()
        print(f"   Supported datasets: {len(available_datasets)}")
        for name in list(available_datasets.keys())[:5]:  # Show first 5
            print(f"   - {name}")
        if len(available_datasets) > 5:
            print(f"   ... and {len(available_datasets) - 5} more")

        # Model architectures
        print(f"\n🧠 Model Architecture Support:")
        print(f"   Registered architectures: {len(self.cv_manager.model_configs)}")

        if cv_results:
            best_model, best_results = self.cv_manager.get_best_model("accuracy")
            print(f"   Best performing model: {best_model}")
            print(f"   Best accuracy: {best_results.mean_metrics['accuracy']:.4f}")

        # Optimization recommendations
        if perf_report.get("optimization_recommendations"):
            print(f"\n⚡ Optimization Recommendations:")
            for rec in perf_report["optimization_recommendations"][:3]:  # Show top 3
                print(f"   - {rec}")

        print(f"\n✅ Production Features Demonstration Complete!")
        print(f"   All major production capabilities successfully tested")
        print(f"   System is ready for production deployment")


async def main():
    """Main demonstration function"""
    print("🚀 PoUW Production Features Showcase")
    print("=====================================")
    print("This demo showcases all production-ready features of the PoUW system.")

    showcase = ProductionShowcase()
    cv_results = None

    try:
        # 1. Dataset Integration
        dataset = showcase.demonstrate_dataset_integration()

        # 2. Performance Monitoring
        showcase.demonstrate_performance_monitoring()

        # 3. GPU Acceleration
        showcase.demonstrate_gpu_acceleration()

        # 4. Large Models
        large_model = showcase.demonstrate_large_models()

        # 5. Cross-Validation
        cv_results = showcase.demonstrate_cross_validation(dataset)

        # 6. Comprehensive Report
        showcase.generate_comprehensive_report(cv_results)

    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Cleanup
        if hasattr(showcase, "perf_monitor"):
            showcase.perf_monitor.stop_monitoring()
        print("\n🧹 Cleanup completed")


if __name__ == "__main__":
    asyncio.run(main())
