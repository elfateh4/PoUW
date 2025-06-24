"""
Production Dataset Management for PoUW

This module provides real dataset integration with support for:
- Common ML datasets (MNIST, CIFAR-10, etc.)
- Multiple data formats (CSV, HDF5, images)
- Data preprocessing and normalization
- Integration with PoUW data management pipeline
"""

import os
import json
import hashlib
import pickle
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import logging

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import h5py

logger = logging.getLogger(__name__)


@dataclass
class DatasetMetadata:
    """Metadata for production datasets"""
    name: str
    size: int
    num_classes: int
    input_shape: Tuple[int, ...]
    data_format: str  # 'images', 'tabular', 'text', 'audio'
    hash_value: str
    preprocessing: Dict[str, Any]
    splits: Dict[str, float]  # train/val/test ratios


class ProductionDatasetManager:
    """
    Manages real datasets for production PoUW deployment.
    Supports multiple data formats and integrates with PoUW data pipeline.
    """
    
    SUPPORTED_DATASETS = {
        'mnist': 'MNIST handwritten digits',
        'cifar10': 'CIFAR-10 image classification',
        'cifar100': 'CIFAR-100 image classification',
        'fashionmnist': 'Fashion-MNIST clothing classification',
        'custom_csv': 'Custom CSV tabular data',
        'custom_hdf5': 'Custom HDF5 data',
        'custom_images': 'Custom image dataset'
    }
    
    def __init__(self, data_root: str = "./data", cache_dir: str = "./cache"):
        self.data_root = Path(data_root)
        self.cache_dir = Path(cache_dir)
        self.data_root.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Dataset registry
        self.datasets: Dict[str, DatasetMetadata] = {}
        self.loaded_datasets: Dict[str, Any] = {}
        
        logger.info(f"Production dataset manager initialized")
        logger.info(f"Data root: {self.data_root}")
        logger.info(f"Cache directory: {self.cache_dir}")
    
    def load_torchvision_dataset(self, name: str, download: bool = True) -> DatasetMetadata:
        """Load standard torchvision datasets"""
        
        # Define transforms for different datasets
        transforms_dict = {
            'mnist': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]),
            'cifar10': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            'cifar100': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            'fashionmnist': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        }
        
        transform = transforms_dict.get(name.lower())
        if not transform:
            raise ValueError(f"Unsupported torchvision dataset: {name}")
        
        # Load dataset
        dataset_class = {
            'mnist': torchvision.datasets.MNIST,
            'cifar10': torchvision.datasets.CIFAR10,
            'cifar100': torchvision.datasets.CIFAR100,
            'fashionmnist': torchvision.datasets.FashionMNIST
        }[name.lower()]
        
        # Load train and test sets
        train_dataset = dataset_class(
            root=str(self.data_root), 
            train=True, 
            download=download, 
            transform=transform
        )
        test_dataset = dataset_class(
            root=str(self.data_root), 
            train=False, 
            download=download, 
            transform=transform
        )
        
        # Calculate dataset hash
        sample_data = str(train_dataset[0]).encode()
        hash_value = hashlib.sha256(sample_data).hexdigest()[:16]
        
        # Determine properties
        if name.lower() in ['mnist', 'fashionmnist']:
            input_shape = (1, 28, 28)
            num_classes = 10
        elif name.lower() == 'cifar10':
            input_shape = (3, 32, 32)
            num_classes = 10
        elif name.lower() == 'cifar100':
            input_shape = (3, 32, 32)
            num_classes = 100
        
        metadata = DatasetMetadata(
            name=name.lower(),
            size=len(train_dataset) + len(test_dataset),
            num_classes=num_classes,
            input_shape=input_shape,
            data_format='images',
            hash_value=hash_value,
            preprocessing={'normalize': True, 'transform': str(transform)},
            splits={'train': 0.8, 'val': 0.1, 'test': 0.1}
        )
        
        # Store datasets
        self.loaded_datasets[name.lower()] = {
            'train': train_dataset,
            'test': test_dataset,
            'metadata': metadata
        }
        self.datasets[name.lower()] = metadata
        
        logger.info(f"Loaded {name} dataset: {metadata.size} samples, {metadata.num_classes} classes")
        return metadata
    
    def load_csv_dataset(self, filepath: str, target_column: str, 
                        dataset_name: str = "custom_csv") -> DatasetMetadata:
        """Load tabular data from CSV file"""
        
        df = pd.read_csv(filepath)
        
        # Separate features and targets
        X = df.drop(columns=[target_column]).values.astype(np.float32)
        y = df[target_column].values
        
        # Handle categorical targets
        if y.dtype == object:
            unique_labels = sorted(list(set(y)))
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            y = np.array([label_to_idx[label] for label in y])
            num_classes = len(unique_labels)
        else:
            num_classes = len(np.unique(np.array(y)))
        
        # Calculate hash
        data_hash = hashlib.sha256(str(df.head()).encode()).hexdigest()[:16]
        
        # Safely get input shape
        shape_list = list(X.shape)
        if len(shape_list) >= 2:
            input_shape = tuple(int(shape_list[i]) for i in range(1, len(shape_list)))
        else:
            input_shape = (int(shape_list[0]),)
        
        metadata = DatasetMetadata(
            name=dataset_name,
            size=len(df),
            num_classes=num_classes,
            input_shape=input_shape,
            data_format='tabular',
            hash_value=data_hash,
            preprocessing={'standardized': False},
            splits={'train': 0.7, 'val': 0.15, 'test': 0.15}
        )
        
        # Create custom dataset class
        class TabularDataset(Dataset):
            def __init__(self, X, y):
                self.X = torch.FloatTensor(X)
                self.y = torch.LongTensor(y)
            
            def __len__(self):
                return len(self.X)
            
            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]
        
        dataset = TabularDataset(X, y)
        
        self.loaded_datasets[dataset_name] = {
            'full': dataset,
            'X': X,
            'y': y,
            'metadata': metadata
        }
        self.datasets[dataset_name] = metadata
        
        logger.info(f"Loaded CSV dataset {dataset_name}: {metadata.size} samples, {metadata.num_classes} classes")
        return metadata
    
    def load_hdf5_dataset(self, filepath: str, data_key: str = 'data', 
                         labels_key: str = 'labels', dataset_name: str = "custom_hdf5") -> DatasetMetadata:
        """Load data from HDF5 file"""
        
        with h5py.File(filepath, 'r') as f:
            # Get datasets with proper type checking
            if data_key not in f or labels_key not in f:
                raise ValueError(f"Keys '{data_key}' or '{labels_key}' not found in HDF5 file")
            
            data_dataset = f[data_key]
            labels_dataset = f[labels_key]
            
            # Verify these are actually datasets
            if not isinstance(data_dataset, h5py.Dataset) or not isinstance(labels_dataset, h5py.Dataset):
                raise ValueError("HDF5 objects are not datasets")
            
            X = np.array(data_dataset[:])
            y = np.array(labels_dataset[:])
        
        # Convert to appropriate types
        X = X.astype(np.float32)
        if y.dtype == object:
            unique_labels = sorted(list(set(y)))
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            y = np.array([label_to_idx[label] for label in y])
        
        num_classes = len(np.unique(y))
        
        # Calculate hash
        data_hash = hashlib.sha256(str(X[:100]).encode()).hexdigest()[:16]
        
        # Determine input shape safely
        shape_list = list(X.shape)
        if len(shape_list) > 2:
            input_shape = tuple(int(dim) for dim in shape_list[1:])
            data_format = 'images' if len(shape_list) == 4 else 'multidimensional'
        elif len(shape_list) == 2:
            input_shape = (int(shape_list[1]),)
            data_format = 'tabular'
        else:
            input_shape = (int(shape_list[0]),)
            data_format = 'vector'
        
        metadata = DatasetMetadata(
            name=dataset_name,
            size=len(X),
            num_classes=num_classes,
            input_shape=input_shape,
            data_format=data_format,
            hash_value=data_hash,
            preprocessing={'normalized': False},
            splits={'train': 0.7, 'val': 0.15, 'test': 0.15}
        )
        
        # Create dataset
        class HDF5Dataset(Dataset):
            def __init__(self, X, y):
                self.X = torch.FloatTensor(X)
                self.y = torch.LongTensor(y)
            
            def __len__(self):
                return len(self.X)
            
            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]
        
        dataset = HDF5Dataset(X, y)
        
        self.loaded_datasets[dataset_name] = {
            'full': dataset,
            'X': X,
            'y': y,
            'metadata': metadata
        }
        self.datasets[dataset_name] = metadata
        
        logger.info(f"Loaded HDF5 dataset {dataset_name}: {metadata.size} samples, {metadata.num_classes} classes")
        return metadata
    
    def split_dataset(self, dataset_name: str, train_ratio: float = 0.7, 
                     val_ratio: float = 0.15, test_ratio: float = 0.15) -> Dict[str, Any]:
        """Split dataset into train/validation/test sets"""
        
        if dataset_name not in self.loaded_datasets:
            raise ValueError(f"Dataset {dataset_name} not loaded")
        
        dataset_info = self.loaded_datasets[dataset_name]
        
        if 'full' in dataset_info:
            # Handle custom datasets
            dataset = dataset_info['full']
            total_size = len(dataset)
            
            # Calculate split sizes
            train_size = int(train_ratio * total_size)
            val_size = int(val_ratio * total_size)
            test_size = total_size - train_size - val_size
            
            # Random split
            train_set, val_set, test_set = torch.utils.data.random_split(
                dataset, [train_size, val_size, test_size]
            )
            
            splits = {
                'train': train_set,
                'val': val_set,
                'test': test_set
            }
            
        else:
            # Handle torchvision datasets that come pre-split
            train_dataset = dataset_info['train']
            test_dataset = dataset_info['test']
            
            # Further split training into train/val
            train_size = int(train_ratio * len(train_dataset))
            val_size = len(train_dataset) - train_size
            
            train_set, val_set = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
            
            splits = {
                'train': train_set,
                'val': val_set,
                'test': test_dataset
            }
        
        # Update metadata
        self.datasets[dataset_name].splits = {
            'train': train_ratio, 'val': val_ratio, 'test': test_ratio
        }
        
        # Store splits
        self.loaded_datasets[dataset_name]['splits'] = splits
        
        logger.info(f"Split {dataset_name}: train={len(splits['train'])}, "
                   f"val={len(splits['val'])}, test={len(splits['test'])}")
        
        return splits
    
    def create_dataloaders(self, dataset_name: str, batch_size: int = 32, 
                          num_workers: int = 0) -> Dict[str, DataLoader]:
        """Create PyTorch DataLoaders for training"""
        
        if dataset_name not in self.loaded_datasets:
            raise ValueError(f"Dataset {dataset_name} not loaded")
        
        dataset_info = self.loaded_datasets[dataset_name]
        
        if 'splits' not in dataset_info:
            # Auto-split if not already done
            self.split_dataset(dataset_name)
            dataset_info = self.loaded_datasets[dataset_name]
        
        splits = dataset_info['splits']
        
        dataloaders = {}
        for split_name, dataset in splits.items():
            shuffle = (split_name == 'train')  # Only shuffle training data
            
            dataloaders[split_name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
        
        logger.info(f"Created DataLoaders for {dataset_name} with batch_size={batch_size}")
        return dataloaders
    
    def get_dataset_info(self, dataset_name: str) -> Optional[DatasetMetadata]:
        """Get metadata for a loaded dataset"""
        return self.datasets.get(dataset_name)
    
    def list_available_datasets(self) -> Dict[str, str]:
        """List all supported datasets"""
        return self.SUPPORTED_DATASETS.copy()
    
    def list_loaded_datasets(self) -> List[str]:
        """List currently loaded datasets"""
        return list(self.datasets.keys())
    
    def export_dataset_metadata(self, filepath: str):
        """Export dataset metadata to JSON file"""
        metadata_dict = {}
        for name, metadata in self.datasets.items():
            metadata_dict[name] = {
                'name': metadata.name,
                'size': metadata.size,
                'num_classes': metadata.num_classes,
                'input_shape': metadata.input_shape,
                'data_format': metadata.data_format,
                'hash_value': metadata.hash_value,
                'preprocessing': metadata.preprocessing,
                'splits': metadata.splits
            }
        
        with open(filepath, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        logger.info(f"Exported metadata for {len(metadata_dict)} datasets to {filepath}")
    
    def cache_dataset(self, dataset_name: str):
        """Cache processed dataset to disk"""
        if dataset_name not in self.loaded_datasets:
            raise ValueError(f"Dataset {dataset_name} not loaded")
        
        cache_path = self.cache_dir / f"{dataset_name}.pkl"
        
        with open(cache_path, 'wb') as f:
            pickle.dump(self.loaded_datasets[dataset_name], f)
        
        logger.info(f"Cached dataset {dataset_name} to {cache_path}")
    
    def load_cached_dataset(self, dataset_name: str) -> bool:
        """Load dataset from cache"""
        cache_path = self.cache_dir / f"{dataset_name}.pkl"
        
        if not cache_path.exists():
            return False
        
        try:
            with open(cache_path, 'rb') as f:
                dataset_info = pickle.load(f)
            
            self.loaded_datasets[dataset_name] = dataset_info
            self.datasets[dataset_name] = dataset_info['metadata']
            
            logger.info(f"Loaded cached dataset {dataset_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load cached dataset {dataset_name}: {e}")
            return False


# Convenience functions for common datasets
def load_mnist(manager: ProductionDatasetManager) -> DatasetMetadata:
    """Load MNIST dataset"""
    return manager.load_torchvision_dataset('mnist')

def load_cifar10(manager: ProductionDatasetManager) -> DatasetMetadata:
    """Load CIFAR-10 dataset"""
    return manager.load_torchvision_dataset('cifar10')

def load_cifar100(manager: ProductionDatasetManager) -> DatasetMetadata:
    """Load CIFAR-100 dataset"""
    return manager.load_torchvision_dataset('cifar100')

def load_fashion_mnist(manager: ProductionDatasetManager) -> DatasetMetadata:
    """Load Fashion-MNIST dataset"""
    return manager.load_torchvision_dataset('fashionmnist')


# Integration with PoUW data management
def integrate_with_pouw_data_manager(dataset_manager: ProductionDatasetManager, 
                                   pouw_data_manager, dataset_name: str):
    """
    Integrate production dataset with PoUW data management system
    (Reed-Solomon encoding, consistent hashing, etc.)
    """
    if dataset_name not in dataset_manager.loaded_datasets:
        raise ValueError(f"Dataset {dataset_name} not loaded")
    
    metadata = dataset_manager.get_dataset_info(dataset_name)
    dataset_info = dataset_manager.loaded_datasets[dataset_name]
    
    # Serialize dataset for PoUW storage
    serialized_data = pickle.dumps(dataset_info)
    
    # Store with Reed-Solomon encoding
    success = pouw_data_manager.store_dataset(
        dataset_id=dataset_name,
        data=serialized_data
    )
    
    if success:
        logger.info(f"Integrated {dataset_name} with PoUW data management")
        return True
    else:
        logger.error(f"Failed to integrate {dataset_name} with PoUW data management")
        return False
