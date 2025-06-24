"""
Integration Tests for Production Features

Tests that production features work correctly with the existing PoUW system.
"""

import pytest
import asyncio
import tempfile
import torch
import numpy as np
from pathlib import Path

from pouw.production import (
    ProductionDatasetManager,
    PerformanceMonitor,
    GPUManager,
    LargeModelArchitectures,
    CrossValidationManager,
    ModelArchitectures
)


class TestProductionIntegration:
    """Test integration of production features with PoUW system"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def dataset_manager(self, temp_dir):
        """Create dataset manager for testing"""
        return ProductionDatasetManager(data_root=temp_dir, cache_dir=temp_dir)
    
    @pytest.fixture
    def perf_monitor(self):
        """Create performance monitor for testing"""
        return PerformanceMonitor()
    
    @pytest.fixture
    def gpu_manager(self):
        """Create GPU manager for testing"""
        return GPUManager()
    
    @pytest.fixture
    def cv_manager(self, gpu_manager):
        """Create cross-validation manager for testing"""
        return CrossValidationManager(gpu_manager)
    
    def test_dataset_manager_initialization(self, dataset_manager):
        """Test that dataset manager initializes correctly"""
        assert dataset_manager is not None
        assert len(dataset_manager.list_loaded_datasets()) == 0
        available = dataset_manager.list_available_datasets()
        assert 'mnist' in available
        assert 'cifar10' in available
    
    def test_performance_monitor_basic(self, perf_monitor):
        """Test basic performance monitoring functionality"""
        perf_monitor.start_monitoring()
        
        # Test profiling context manager
        with perf_monitor.profile("test_operation"):
            import time
            time.sleep(0.1)
        
        perf_monitor.stop_monitoring()
        
        # Check that metrics were recorded
        report = perf_monitor.generate_performance_report()
        assert 'operation_performance' in report
        assert 'test_operation' in report['operation_performance']
    
    def test_gpu_manager_device_detection(self, gpu_manager):
        """Test GPU manager device detection"""
        assert gpu_manager.device is not None
        assert hasattr(gpu_manager.device, 'type')
        
        # Test moving model to device
        model = torch.nn.Linear(10, 5)
        model_on_device = model.to(gpu_manager.device)
        assert next(model_on_device.parameters()).device == gpu_manager.device
    
    def test_large_model_creation(self):
        """Test creation of large-scale models"""
        # Test CNN
        cnn = LargeModelArchitectures.create_large_cnn(num_classes=10, width_multiplier=2.0)
        cnn_params = sum(p.numel() for p in cnn.parameters())
        assert cnn_params > 1_000_000  # Should be substantial
        
        # Test Transformer
        transformer = LargeModelArchitectures.create_transformer_model(
            vocab_size=1000, d_model=256, num_heads=8, num_layers=4
        )
        transformer_params = sum(p.numel() for p in transformer.parameters())
        assert transformer_params > 1_000_000
    
    def test_model_architectures_registry(self, cv_manager):
        """Test model architecture registry"""
        # Test registering standard architectures
        input_info = {'type': 'image', 'input_channels': 1, 'num_classes': 10}
        cv_manager.register_standard_architectures(input_info)
        
        assert len(cv_manager.model_configs) > 0
        assert 'cnn_classifier' in cv_manager.model_configs
        
        # Test tabular data architectures
        cv_manager_tabular = CrossValidationManager(cv_manager.gpu_manager)
        tabular_info = {'type': 'tabular', 'input_size': 784, 'num_classes': 10}
        cv_manager_tabular.register_standard_architectures(tabular_info)
        
        assert len(cv_manager_tabular.model_configs) > 0
        assert 'mlp_small' in cv_manager_tabular.model_configs
    
    @pytest.mark.asyncio
    async def test_dataset_loading_with_monitoring(self, dataset_manager, perf_monitor):
        """Test dataset loading with performance monitoring"""
        perf_monitor.start_monitoring()
        
        try:
            # Load dataset with monitoring
            with perf_monitor.profile("dataset_loading"):
                # Use a small dataset for testing
                try:
                    metadata = dataset_manager.load_torchvision_dataset('mnist', download=True)
                    assert metadata.name == 'mnist'
                    assert metadata.size > 0
                    assert metadata.num_classes == 10
                except Exception as e:
                    # If MNIST download fails (no internet), create synthetic data
                    pytest.skip(f"Could not download MNIST: {e}")
            
            # Check that monitoring recorded the operation
            report = perf_monitor.generate_performance_report()
            assert 'dataset_loading' in report['operation_performance']
            
        finally:
            perf_monitor.stop_monitoring()
    
    def test_cross_validation_setup(self, cv_manager):
        """Test cross-validation setup with synthetic data"""
        # Create simple synthetic dataset
        X = torch.randn(100, 1, 28, 28)
        y = torch.randint(0, 10, (100,))
        dataset = torch.utils.data.TensorDataset(X, y)
        
        # Register standard architectures first
        input_info = {'type': 'image', 'input_channels': 1, 'num_classes': 10}
        cv_manager.register_standard_architectures(input_info)
        
        # Verify configuration
        assert len(cv_manager.model_configs) > 0
        assert 'cnn_classifier' in cv_manager.model_configs
    
    @pytest.mark.asyncio
    async def test_integrated_workflow(self, dataset_manager, perf_monitor, gpu_manager):
        """Test integrated workflow with all production features"""
        perf_monitor.start_monitoring()
        
        try:
            # 1. Create synthetic dataset for testing
            with perf_monitor.profile("data_preparation"):
                X = torch.randn(50, 1, 28, 28)  # Small dataset for testing
                y = torch.randint(0, 10, (50,))
                synthetic_dataset = torch.utils.data.TensorDataset(X, y)
            
            # 2. Create model with GPU support
            with perf_monitor.profile("model_creation"):
                model = LargeModelArchitectures.create_large_cnn(
                    num_classes=10, 
                    input_channels=1,  # Match our synthetic data (1 channel)
                    width_multiplier=0.5  # Smaller for testing
                )
                model = model.to(gpu_manager.device)
            
            # 3. Simulate training with monitoring
            with perf_monitor.profile("training_simulation"):
                dataloader = torch.utils.data.DataLoader(synthetic_dataset, batch_size=16)
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                
                # One training step
                for data, target in dataloader:
                    data, target = data.to(gpu_manager.device), target.to(gpu_manager.device)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    break  # Just one batch for testing
            
            # 4. Verify monitoring recorded everything
            report = perf_monitor.generate_performance_report()
            expected_operations = ['data_preparation', 'model_creation', 'training_simulation']
            
            for op in expected_operations:
                assert op in report['operation_performance'], f"Operation {op} not recorded"
            
            # 5. Check system health
            health = perf_monitor.collect_system_health()
            assert health.cpu_usage >= 0
            assert health.memory_usage >= 0
            
        finally:
            perf_monitor.stop_monitoring()
    
    def test_production_features_imports(self):
        """Test that all production features can be imported correctly"""
        # This test verifies the __init__.py exports work correctly
        from pouw.production import (
            ProductionDatasetManager,
            DatasetMetadata,
            PerformanceMonitor,
            PerformanceProfiler,
            OptimizationManager,
            PerformanceMetrics,
            SystemHealth,
            GPUManager,
            GPUAcceleratedTrainer,
            GPUAcceleratedMiner,
            GPUMemoryManager,
            LargeModelArchitectures,
            LargeModelManager,
            ModelConfig,
            ModelArchitectures,
            CrossValidationManager,
            HyperparameterOptimizer,
            ModelArchitectureConfig,
            ValidationResults,
            HyperparameterConfig
        )
        
        # Basic instantiation test
        dataset_mgr = ProductionDatasetManager()
        perf_mon = PerformanceMonitor()
        gpu_mgr = GPUManager()
        cv_mgr = CrossValidationManager(gpu_mgr)
        
        assert dataset_mgr is not None
        assert perf_mon is not None
        assert gpu_mgr is not None
        assert cv_mgr is not None


class TestProductionFeatureConfiguration:
    """Test configuration and customization of production features"""
    
    def test_dataset_manager_configuration(self):
        """Test dataset manager configuration options"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ProductionDatasetManager(
                data_root=tmpdir,
                cache_dir=tmpdir
            )
            
            assert str(manager.data_root) == tmpdir
            assert str(manager.cache_dir) == tmpdir
    
    def test_gpu_manager_configuration(self):
        """Test GPU manager configuration options"""
        # Test with explicit device preference
        gpu_manager = GPUManager(device_preference='cpu')
        assert gpu_manager.device.type == 'cpu'
        
        # Test auto-detection
        auto_manager = GPUManager()
        assert auto_manager.device is not None
    
    def test_performance_monitor_configuration(self):
        """Test performance monitor configuration"""
        monitor = PerformanceMonitor(
            history_size=100
        )
        
        assert monitor.history_size == 100


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
