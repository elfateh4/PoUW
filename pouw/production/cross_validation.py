"""
Cross-Validation and Multiple Model Architectures for PoUW

This module provides comprehensive support for:
- K-fold cross-validation
- Multiple model architecture evaluation
- Model selection and comparison
- Ensemble methods
- Hyperparameter optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import time
import itertools
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ModelArchitectureConfig:
    """Configuration for a model architecture"""
    name: str
    model_class: type
    model_kwargs: Dict[str, Any]
    optimizer_class: type = optim.Adam
    optimizer_kwargs: Dict[str, Any] = field(default_factory=lambda: {'lr': 0.001})
    scheduler_class: Optional[type] = None
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    criterion_class: type = nn.CrossEntropyLoss
    criterion_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResults:
    """Results from cross-validation or model evaluation"""
    model_name: str
    fold_results: List[Dict[str, float]]
    mean_metrics: Dict[str, float]
    std_metrics: Dict[str, float]
    best_fold: int
    worst_fold: int
    training_time: float
    total_parameters: int


@dataclass
class HyperparameterConfig:
    """Configuration for hyperparameter optimization"""
    param_name: str
    param_type: str  # 'choice', 'range', 'log_range'
    values: Union[List[Any], Tuple[float, float]]


class ModelArchitectures:
    """
    Collection of various model architectures for comparison
    """
    
    @staticmethod
    def simple_mlp(input_size: int, hidden_sizes: List[int], num_classes: int, 
                   dropout: float = 0.2) -> nn.Module:
        """Simple Multi-Layer Perceptron"""
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        
        return nn.Sequential(*layers)
    
    @staticmethod
    def cnn_classifier(input_channels: int = 3, num_classes: int = 10,
                      base_channels: int = 32) -> nn.Module:
        """Convolutional Neural Network classifier"""
        
        class CNNClassifier(nn.Module):
            def __init__(self, input_channels, num_classes, base_channels):
                super().__init__()
                
                self.features = nn.Sequential(
                    nn.Conv2d(input_channels, base_channels, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    
                    nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    
                    nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    
                    nn.AdaptiveAvgPool2d((4, 4))
                )
                
                self.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(base_channels * 4 * 16, base_channels * 4),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(base_channels * 4, num_classes)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                return x
        
        return CNNClassifier(input_channels, num_classes, base_channels)
    
    @staticmethod
    def resnet_block_model(input_channels: int = 3, num_classes: int = 10,
                          num_blocks: int = 3) -> nn.Module:
        """ResNet-style model with residual blocks"""
        
        class ResidualBlock(nn.Module):
            def __init__(self, in_channels, out_channels, stride=1):
                super().__init__()
                
                self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_channels)
                
                self.shortcut = nn.Sequential()
                if stride != 1 or in_channels != out_channels:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                        nn.BatchNorm2d(out_channels)
                    )
            
            def forward(self, x):
                out = torch.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += self.shortcut(x)
                out = torch.relu(out)
                return out
        
        class ResNetModel(nn.Module):
            def __init__(self, input_channels, num_classes, num_blocks):
                super().__init__()
                
                self.conv1 = nn.Conv2d(input_channels, 64, 7, stride=2, padding=3, bias=False)
                self.bn1 = nn.BatchNorm2d(64)
                self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
                
                self.layer1 = self._make_layer(64, 64, num_blocks, stride=1)
                self.layer2 = self._make_layer(64, 128, num_blocks, stride=2)
                self.layer3 = self._make_layer(128, 256, num_blocks, stride=2)
                
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(256, num_classes)
            
            def _make_layer(self, in_channels, out_channels, num_blocks, stride):
                layers = [ResidualBlock(in_channels, out_channels, stride)]
                for _ in range(1, num_blocks):
                    layers.append(ResidualBlock(out_channels, out_channels, 1))
                return nn.Sequential(*layers)
            
            def forward(self, x):
                x = torch.relu(self.bn1(self.conv1(x)))
                x = self.maxpool(x)
                
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)
                return x
        
        return ResNetModel(input_channels, num_classes, num_blocks)
    
    @staticmethod
    def attention_model(input_size: int, num_classes: int, 
                       num_heads: int = 8, embed_dim: int = 256) -> nn.Module:
        """Attention-based model for tabular data"""
        
        class AttentionModel(nn.Module):
            def __init__(self, input_size, num_classes, num_heads, embed_dim):
                super().__init__()
                
                self.embedding = nn.Linear(input_size, embed_dim)
                self.pos_encoding = nn.Parameter(torch.randn(1, 1, embed_dim))
                
                self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
                self.norm1 = nn.LayerNorm(embed_dim)
                
                self.ffn = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.ReLU(),
                    nn.Linear(embed_dim * 4, embed_dim)
                )
                self.norm2 = nn.LayerNorm(embed_dim)
                
                self.classifier = nn.Linear(embed_dim, num_classes)
            
            def forward(self, x):
                # Assume x is (batch_size, input_size)
                x = self.embedding(x).unsqueeze(1)  # (batch_size, 1, embed_dim)
                x = x + self.pos_encoding
                
                # Self-attention
                attn_out, _ = self.attention(x, x, x)
                x = self.norm1(x + attn_out)
                
                # Feed-forward
                ffn_out = self.ffn(x)
                x = self.norm2(x + ffn_out)
                
                # Classification
                x = x.squeeze(1)  # (batch_size, embed_dim)
                x = self.classifier(x)
                return x
        
        return AttentionModel(input_size, num_classes, num_heads, embed_dim)


class CrossValidationManager:
    """
    Manages cross-validation experiments for multiple model architectures
    """
    
    def __init__(self, gpu_manager, performance_monitor=None):
        self.gpu_manager = gpu_manager
        self.performance_monitor = performance_monitor
        self.device = gpu_manager.device
        
        self.model_configs: Dict[str, ModelArchitectureConfig] = {}
        self.results: Dict[str, ValidationResults] = {}
        
    def register_model_architecture(self, config: ModelArchitectureConfig):
        """Register a model architecture for evaluation"""
        self.model_configs[config.name] = config
        logger.info(f"Registered model architecture: {config.name}")
    
    def register_standard_architectures(self, input_info: Dict[str, Any]):
        """Register standard model architectures based on input information"""
        
        input_type = input_info.get('type', 'tabular')  # 'tabular', 'image', 'sequence'
        num_classes = input_info.get('num_classes', 10)
        
        if input_type == 'tabular':
            input_size = input_info.get('input_size', 784)
            
            # Simple MLP
            mlp_config = ModelArchitectureConfig(
                name='mlp_small',
                model_class=lambda: ModelArchitectures.simple_mlp(
                    input_size, [128, 64], num_classes, dropout=0.2
                ),
                model_kwargs={}
            )
            self.register_model_architecture(mlp_config)
            
            # Larger MLP
            mlp_large_config = ModelArchitectureConfig(
                name='mlp_large',
                model_class=lambda: ModelArchitectures.simple_mlp(
                    input_size, [512, 256, 128], num_classes, dropout=0.3
                ),
                model_kwargs={}
            )
            self.register_model_architecture(mlp_large_config)
            
            # Attention model
            attention_config = ModelArchitectureConfig(
                name='attention',
                model_class=lambda: ModelArchitectures.attention_model(
                    input_size, num_classes, num_heads=8, embed_dim=256
                ),
                model_kwargs={}
            )
            self.register_model_architecture(attention_config)
            
        elif input_type == 'image':
            input_channels = input_info.get('input_channels', 3)
            
            # CNN classifier
            cnn_config = ModelArchitectureConfig(
                name='cnn_classifier',
                model_class=lambda: ModelArchitectures.cnn_classifier(
                    input_channels, num_classes, base_channels=32
                ),
                model_kwargs={}
            )
            self.register_model_architecture(cnn_config)
            
            # Larger CNN
            cnn_large_config = ModelArchitectureConfig(
                name='cnn_large',
                model_class=lambda: ModelArchitectures.cnn_classifier(
                    input_channels, num_classes, base_channels=64
                ),
                model_kwargs={}
            )
            self.register_model_architecture(cnn_large_config)
            
            # ResNet-style model
            resnet_config = ModelArchitectureConfig(
                name='resnet_small',
                model_class=lambda: ModelArchitectures.resnet_block_model(
                    input_channels, num_classes, num_blocks=2
                ),
                model_kwargs={}
            )
            self.register_model_architecture(resnet_config)
        
        logger.info(f"Registered {len(self.model_configs)} standard architectures for {input_type} data")
    
    def run_cross_validation(self, dataset, k_folds: int = 5, 
                           epochs: int = 10, batch_size: int = 32,
                           stratified: bool = True) -> Dict[str, ValidationResults]:
        """Run k-fold cross-validation for all registered models"""
        
        logger.info(f"Starting {k_folds}-fold cross-validation for {len(self.model_configs)} models")
        
        # Prepare dataset indices
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        
        # Create labels for stratification
        if stratified:
            labels = []
            for i in range(dataset_size):
                _, label = dataset[i]
                labels.append(label)
            
            kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
            fold_splits = list(kfold.split(indices, labels))
        else:
            kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            fold_splits = list(kfold.split(indices))
        
        # Run cross-validation for each model
        all_results = {}
        
        for model_name, config in self.model_configs.items():
            logger.info(f"Running cross-validation for {model_name}")
            
            fold_results = []
            start_time = time.time()
            
            for fold_idx, (train_indices, val_indices) in enumerate(fold_splits):
                logger.info(f"  Fold {fold_idx + 1}/{k_folds}")
                
                # Create data loaders
                train_sampler = SubsetRandomSampler(train_indices)
                val_sampler = SubsetRandomSampler(val_indices)
                
                train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
                val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
                
                # Create model
                model = config.model_class()
                model = model.to(self.device)
                
                # Create optimizer and criterion
                optimizer = config.optimizer_class(model.parameters(), **config.optimizer_kwargs)
                criterion = config.criterion_class(**config.criterion_kwargs)
                
                # Create scheduler if specified
                scheduler = None
                if config.scheduler_class:
                    scheduler = config.scheduler_class(optimizer, **config.scheduler_kwargs)
                
                # Train model
                fold_result = self._train_and_evaluate(
                    model, train_loader, val_loader, optimizer, criterion, 
                    epochs, scheduler, fold_idx
                )
                
                fold_results.append(fold_result)
            
            training_time = time.time() - start_time
            
            # Calculate statistics across folds
            mean_metrics = {}
            std_metrics = {}
            metric_names = fold_results[0].keys()
            
            for metric in metric_names:
                values = [result[metric] for result in fold_results]
                mean_metrics[metric] = np.mean(values)
                std_metrics[metric] = np.std(values)
            
            # Find best and worst folds
            accuracies = [result['accuracy'] for result in fold_results]
            best_fold = np.argmax(accuracies)
            worst_fold = np.argmin(accuracies)
            
            # Get model parameter count
            model = config.model_class()
            total_params = sum(p.numel() for p in model.parameters())
            
            results = ValidationResults(
                model_name=model_name,
                fold_results=fold_results,
                mean_metrics=mean_metrics,
                std_metrics=std_metrics,
                best_fold=best_fold,
                worst_fold=worst_fold,
                training_time=training_time,
                total_parameters=total_params
            )
            
            all_results[model_name] = results
            self.results[model_name] = results
            
            logger.info(f"  {model_name} - Accuracy: {mean_metrics['accuracy']:.4f} ± {std_metrics['accuracy']:.4f}")
        
        return all_results
    
    def _train_and_evaluate(self, model, train_loader, val_loader, optimizer, criterion,
                          epochs, scheduler, fold_idx) -> Dict[str, float]:
        """Train and evaluate model for one fold"""
        
        model.train()
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_data in train_loader:
                if len(batch_data) == 2:
                    inputs, targets = batch_data
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
            
            if scheduler:
                scheduler.step()
        
        # Evaluation
        model.eval()
        all_predictions = []
        all_targets = []
        total_val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                if len(batch_data) == 2:
                    inputs, targets = batch_data
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    _, predicted = torch.max(outputs.data, 1)
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    
                    total_val_loss += loss.item()
                    num_val_batches += 1
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        
        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'val_loss': avg_val_loss
        }
    
    def get_best_model(self, metric: str = 'accuracy') -> Tuple[str, ValidationResults]:
        """Get the best performing model based on specified metric"""
        
        if not self.results:
            raise ValueError("No validation results available")
        
        best_score = -float('inf')
        best_model = None
        best_results = None
        
        for model_name, results in self.results.items():
            score = results.mean_metrics.get(metric, -float('inf'))
            if score > best_score:
                best_score = score
                best_model = model_name
                best_results = results
        
        return best_model, best_results
    
    def generate_comparison_report(self) -> Dict[str, Any]:
        """Generate comprehensive comparison report"""
        
        if not self.results:
            return {'error': 'No validation results available'}
        
        # Model rankings by different metrics
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        rankings = {}
        
        for metric in metrics:
            sorted_models = sorted(
                self.results.items(),
                key=lambda x: x[1].mean_metrics.get(metric, 0),
                reverse=True
            )
            rankings[metric] = [(model, results.mean_metrics.get(metric, 0)) for model, results in sorted_models]
        
        # Best model overall
        best_model, best_results = self.get_best_model('accuracy')
        
        # Performance summary
        summary = {}
        for model_name, results in self.results.items():
            summary[model_name] = {
                'mean_accuracy': results.mean_metrics.get('accuracy', 0),
                'std_accuracy': results.std_metrics.get('accuracy', 0),
                'parameters': results.total_parameters,
                'training_time': results.training_time,
                'efficiency_score': results.mean_metrics.get('accuracy', 0) / (results.total_parameters / 1e6)  # Accuracy per million parameters
            }
        
        report = {
            'timestamp': time.time(),
            'num_models_evaluated': len(self.results),
            'best_model': {
                'name': best_model,
                'accuracy': best_results.mean_metrics.get('accuracy', 0),
                'parameters': best_results.total_parameters
            },
            'rankings': rankings,
            'performance_summary': summary,
            'detailed_results': {
                name: {
                    'mean_metrics': results.mean_metrics,
                    'std_metrics': results.std_metrics,
                    'parameters': results.total_parameters,
                    'training_time': results.training_time
                }
                for name, results in self.results.items()
            }
        }
        
        return report
    
    def export_results(self, filepath: str):
        """Export validation results to JSON file"""
        
        report = self.generate_comparison_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Exported validation results to {filepath}")


class HyperparameterOptimizer:
    """
    Hyperparameter optimization for PoUW models
    """
    
    def __init__(self, cv_manager: CrossValidationManager):
        self.cv_manager = cv_manager
        self.hyperparameter_configs: List[HyperparameterConfig] = []
        self.optimization_results: List[Dict[str, Any]] = []
    
    def add_hyperparameter(self, config: HyperparameterConfig):
        """Add hyperparameter to optimize"""
        self.hyperparameter_configs.append(config)
    
    def grid_search(self, base_config: ModelArchitectureConfig, dataset,
                   k_folds: int = 3, max_combinations: int = 50) -> Dict[str, Any]:
        """Perform grid search optimization"""
        
        if not self.hyperparameter_configs:
            raise ValueError("No hyperparameters configured for optimization")
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(max_combinations)
        
        logger.info(f"Starting grid search with {len(param_combinations)} combinations")
        
        best_score = -float('inf')
        best_params = None
        best_results = None
        
        for i, params in enumerate(param_combinations):
            logger.info(f"Testing combination {i + 1}/{len(param_combinations)}: {params}")
            
            # Create modified config
            modified_config = self._apply_hyperparameters(base_config, params)
            
            # Register and test this configuration
            test_name = f"grid_search_{i}"
            modified_config.name = test_name
            
            self.cv_manager.register_model_architecture(modified_config)
            
            # Run cross-validation
            cv_results = self.cv_manager.run_cross_validation(
                dataset, k_folds=k_folds, epochs=5  # Reduced epochs for speed
            )
            
            results = cv_results[test_name]
            score = results.mean_metrics.get('accuracy', 0)
            
            if score > best_score:
                best_score = score
                best_params = params
                best_results = results
            
            # Store results
            self.optimization_results.append({
                'parameters': params,
                'score': score,
                'std': results.std_metrics.get('accuracy', 0),
                'results': results
            })
        
        optimization_summary = {
            'best_score': best_score,
            'best_parameters': best_params,
            'best_results': best_results,
            'all_results': self.optimization_results,
            'total_combinations_tested': len(param_combinations)
        }
        
        logger.info(f"Grid search completed. Best accuracy: {best_score:.4f} with params: {best_params}")
        
        return optimization_summary
    
    def _generate_param_combinations(self, max_combinations: int) -> List[Dict[str, Any]]:
        """Generate parameter combinations for grid search"""
        
        param_values = {}
        for config in self.hyperparameter_configs:
            if config.param_type == 'choice':
                param_values[config.param_name] = config.values
            elif config.param_type == 'range':
                start, end = config.values
                param_values[config.param_name] = np.linspace(start, end, 5).tolist()
            elif config.param_type == 'log_range':
                start, end = config.values
                param_values[config.param_name] = np.logspace(np.log10(start), np.log10(end), 5).tolist()
        
        # Generate all combinations
        param_names = list(param_values.keys())
        param_value_lists = [param_values[name] for name in param_names]
        
        all_combinations = list(itertools.product(*param_value_lists))
        
        # Limit combinations if too many
        if len(all_combinations) > max_combinations:
            # Random sampling
            import random
            all_combinations = random.sample(all_combinations, max_combinations)
        
        # Convert to list of dictionaries
        combinations = []
        for combo in all_combinations:
            param_dict = {name: value for name, value in zip(param_names, combo)}
            combinations.append(param_dict)
        
        return combinations
    
    def _apply_hyperparameters(self, base_config: ModelArchitectureConfig, 
                             params: Dict[str, Any]) -> ModelArchitectureConfig:
        """Apply hyperparameters to create new configuration"""
        
        # Deep copy the base configuration
        import copy
        new_config = copy.deepcopy(base_config)
        
        # Apply hyperparameters
        for param_name, param_value in params.items():
            if param_name.startswith('optimizer_'):
                # Optimizer hyperparameters
                opt_param = param_name.replace('optimizer_', '')
                new_config.optimizer_kwargs[opt_param] = param_value
            elif param_name.startswith('model_'):
                # Model hyperparameters
                model_param = param_name.replace('model_', '')
                new_config.model_kwargs[model_param] = param_value
            else:
                # General hyperparameters
                if hasattr(new_config, param_name):
                    setattr(new_config, param_name, param_value)
        
        return new_config


# Utility functions
def create_standard_cv_experiment(gpu_manager, dataset, input_info: Dict[str, Any],
                                k_folds: int = 5, epochs: int = 10) -> Dict[str, Any]:
    """Create and run a standard cross-validation experiment"""
    
    cv_manager = CrossValidationManager(gpu_manager)
    cv_manager.register_standard_architectures(input_info)
    
    results = cv_manager.run_cross_validation(dataset, k_folds, epochs)
    report = cv_manager.generate_comparison_report()
    
    return {
        'cv_manager': cv_manager,
        'results': results,
        'report': report
    }

def quick_model_comparison(gpu_manager, dataset, input_info: Dict[str, Any]) -> str:
    """Quick comparison of model architectures"""
    
    experiment = create_standard_cv_experiment(gpu_manager, dataset, input_info, k_folds=3, epochs=5)
    
    best_model, best_results = experiment['cv_manager'].get_best_model()
    
    summary = f"Best Model: {best_model}\n"
    summary += f"Accuracy: {best_results.mean_metrics['accuracy']:.4f} ± {best_results.std_metrics['accuracy']:.4f}\n"
    summary += f"Parameters: {best_results.total_parameters:,}\n"
    summary += f"Training Time: {best_results.training_time:.2f}s\n"
    
    return summary
