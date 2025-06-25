"""
Large-Scale Model Support for PoUW

This module provides support for training and deploying large neural networks
(>14M parameters) in the PoUW system with optimizations for memory efficiency,
model parallelism, and distributed computing.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.checkpoint import checkpoint
from typing import Dict, List, Optional, Any, Tuple, Union
import math
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for large-scale models"""

    model_type: str  # 'cnn', 'transformer', 'resnet', 'custom'
    num_parameters: int
    memory_requirement_gb: float
    recommended_batch_size: int
    supports_gradient_checkpointing: bool = True
    supports_model_parallelism: bool = True


class LargeModelArchitectures:
    """
    Collection of large-scale model architectures optimized for PoUW
    """

    @staticmethod
    def create_large_cnn(
        num_classes: int = 1000, input_channels: int = 3, width_multiplier: float = 2.0
    ) -> nn.Module:
        """Create large CNN architecture"""

        class LargeCNN(nn.Module):
            def __init__(self, num_classes, input_channels, width_mult):
                super().__init__()

                base_channels = int(64 * width_mult)

                self.features = nn.Sequential(
                    # Initial layers
                    nn.Conv2d(input_channels, base_channels, 7, stride=2, padding=3),
                    nn.BatchNorm2d(base_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(3, stride=2, padding=1),
                    # Block 1
                    *self._make_layer(base_channels, base_channels * 2, 3),
                    *self._make_layer(base_channels * 2, base_channels * 4, 4),
                    *self._make_layer(base_channels * 4, base_channels * 8, 6),
                    *self._make_layer(base_channels * 8, base_channels * 16, 3),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )

                self.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(base_channels * 16, base_channels * 8),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(base_channels * 8, base_channels * 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(base_channels * 4, num_classes),
                )

                self.gradient_checkpointing = False

            def _make_layer(self, in_channels, out_channels, num_blocks):
                layers = []
                for i in range(num_blocks):
                    layers.extend(
                        [
                            nn.Conv2d(
                                in_channels if i == 0 else out_channels, out_channels, 3, padding=1
                            ),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=True),
                        ]
                    )
                return layers

            def forward(self, x):
                if self.gradient_checkpointing and self.training:
                    x = checkpoint(self.features, x)
                else:
                    x = self.features(x)

                # Ensure x is a tensor before flattening
                assert isinstance(x, torch.Tensor), f"Expected tensor, got {type(x)}"
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                return x

            def enable_gradient_checkpointing(self):
                self.gradient_checkpointing = True

            def disable_gradient_checkpointing(self):
                self.gradient_checkpointing = False

        return LargeCNN(num_classes, input_channels, width_multiplier)

    @staticmethod
    def create_transformer_model(
        vocab_size: int = 50000,
        d_model: int = 1024,
        num_heads: int = 16,
        num_layers: int = 24,
        seq_length: int = 512,
    ) -> nn.Module:
        """Create large transformer model"""

        class LargeTransformer(nn.Module):
            def __init__(self, vocab_size, d_model, num_heads, num_layers, seq_length):
                super().__init__()

                self.d_model = d_model
                self.seq_length = seq_length

                self.embedding = nn.Embedding(vocab_size, d_model)
                self.positional_encoding = nn.Parameter(torch.randn(seq_length, d_model))

                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=d_model * 4,
                    dropout=0.1,
                    batch_first=True,
                )

                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                self.output_projection = nn.Linear(d_model, vocab_size)

                self.gradient_checkpointing = False

                # Initialize weights
                self._init_weights()

            def _init_weights(self):
                for module in self.modules():
                    if isinstance(module, (nn.Linear, nn.Embedding)):
                        module.weight.data.normal_(mean=0.0, std=0.02)
                        if isinstance(module, nn.Linear) and module.bias is not None:
                            module.bias.data.zero_()

            def forward(self, input_ids, attention_mask=None):
                seq_len = input_ids.size(1)

                # Embeddings
                x = self.embedding(input_ids)
                x = x + self.positional_encoding[:seq_len]

                # Transformer layers
                if self.gradient_checkpointing and self.training:
                    x = checkpoint(self.transformer, x, attention_mask)
                else:
                    x = self.transformer(x, src_key_padding_mask=attention_mask)

                # Output projection
                x = self.output_projection(x)
                return x

            def enable_gradient_checkpointing(self):
                self.gradient_checkpointing = True

            def disable_gradient_checkpointing(self):
                self.gradient_checkpointing = False

        return LargeTransformer(vocab_size, d_model, num_heads, num_layers, seq_length)

    @staticmethod
    def create_large_resnet(
        num_classes: int = 1000,
        layers: List[int] = [3, 8, 36, 3],  # ResNet-152 variant
        width_multiplier: float = 2.0,
    ) -> nn.Module:
        """Create large ResNet architecture"""

        class BasicBlock(nn.Module):
            expansion = 1

            def __init__(self, inplanes, planes, stride=1, downsample=None):
                super().__init__()
                self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(planes)
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(planes)
                self.downsample = downsample
                self.stride = stride

            def forward(self, x):
                identity = x

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                out = self.bn2(out)

                if self.downsample is not None:
                    identity = self.downsample(x)

                out += identity
                out = self.relu(out)

                return out

        class LargeResNet(nn.Module):
            def __init__(self, block, layers, num_classes, width_mult):
                super().__init__()

                self.inplanes = int(64 * width_mult)

                self.conv1 = nn.Conv2d(3, self.inplanes, 7, stride=2, padding=3, bias=False)
                self.bn1 = nn.BatchNorm2d(self.inplanes)
                self.relu = nn.ReLU(inplace=True)
                self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

                self.layer1 = self._make_layer(block, int(64 * width_mult), layers[0])
                self.layer2 = self._make_layer(block, int(128 * width_mult), layers[1], stride=2)
                self.layer3 = self._make_layer(block, int(256 * width_mult), layers[2], stride=2)
                self.layer4 = self._make_layer(block, int(512 * width_mult), layers[3], stride=2)

                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(int(512 * width_mult) * block.expansion, num_classes)

                self.gradient_checkpointing = False

                # Initialize weights
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)

            def _make_layer(self, block, planes, blocks, stride=1):
                downsample = None
                if stride != 1 or self.inplanes != planes * block.expansion:
                    downsample = nn.Sequential(
                        nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                        nn.BatchNorm2d(planes * block.expansion),
                    )

                layers = []
                layers.append(block(self.inplanes, planes, stride, downsample))
                self.inplanes = planes * block.expansion
                for _ in range(1, blocks):
                    layers.append(block(self.inplanes, planes))

                return nn.Sequential(*layers)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)

                if self.gradient_checkpointing and self.training:
                    x = checkpoint(self.layer1, x)
                    x = checkpoint(self.layer2, x)
                    x = checkpoint(self.layer3, x)
                    x = checkpoint(self.layer4, x)
                else:
                    x = self.layer1(x)
                    x = self.layer2(x)
                    x = self.layer3(x)
                    x = self.layer4(x)

                x = self.avgpool(x)
                # Ensure x is a tensor before flattening
                assert isinstance(x, torch.Tensor), f"Expected tensor, got {type(x)}"
                x = torch.flatten(x, 1)
                x = self.fc(x)

                return x

            def enable_gradient_checkpointing(self):
                self.gradient_checkpointing = True

            def disable_gradient_checkpointing(self):
                self.gradient_checkpointing = False

        return LargeResNet(BasicBlock, layers, num_classes, width_multiplier)


class LargeModelManager:
    """
    Manages large-scale models for PoUW with memory and performance optimizations
    """

    def __init__(self, gpu_manager, enable_distributed: bool = False):
        self.gpu_manager = gpu_manager
        self.enable_distributed = enable_distributed
        self.model_configs: Dict[str, ModelConfig] = {}

        if enable_distributed:
            self._init_distributed()

    def _init_distributed(self):
        """Initialize distributed training"""
        try:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            logger.info(
                f"Distributed training initialized: rank {dist.get_rank()}/{dist.get_world_size()}"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize distributed training: {e}")
            self.enable_distributed = False

    def register_model_config(self, model_name: str, config: ModelConfig):
        """Register a model configuration"""
        self.model_configs[model_name] = config
        logger.info(f"Registered model config: {model_name} ({config.num_parameters:,} parameters)")

    def create_large_model(self, model_type: str, **kwargs) -> nn.Module:
        """Create a large-scale model"""

        if model_type == "large_cnn":
            model = LargeModelArchitectures.create_large_cnn(**kwargs)
            num_params = sum(p.numel() for p in model.parameters())
            memory_gb = self._estimate_memory_requirement(model)

            config = ModelConfig(
                model_type="cnn",
                num_parameters=num_params,
                memory_requirement_gb=memory_gb,
                recommended_batch_size=self._recommend_batch_size(memory_gb),
                supports_gradient_checkpointing=True,
                supports_model_parallelism=True,
            )

        elif model_type == "large_transformer":
            model = LargeModelArchitectures.create_transformer_model(**kwargs)
            num_params = sum(p.numel() for p in model.parameters())
            memory_gb = self._estimate_memory_requirement(model)

            config = ModelConfig(
                model_type="transformer",
                num_parameters=num_params,
                memory_requirement_gb=memory_gb,
                recommended_batch_size=self._recommend_batch_size(memory_gb),
                supports_gradient_checkpointing=True,
                supports_model_parallelism=True,
            )

        elif model_type == "large_resnet":
            model = LargeModelArchitectures.create_large_resnet(**kwargs)
            num_params = sum(p.numel() for p in model.parameters())
            memory_gb = self._estimate_memory_requirement(model)

            config = ModelConfig(
                model_type="resnet",
                num_parameters=num_params,
                memory_requirement_gb=memory_gb,
                recommended_batch_size=self._recommend_batch_size(memory_gb),
                supports_gradient_checkpointing=True,
                supports_model_parallelism=True,
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.register_model_config(model_type, config)

        logger.info(f"Created {model_type}: {num_params:,} parameters, {memory_gb:.2f} GB memory")
        return model

    def _estimate_memory_requirement(self, model: nn.Module) -> float:
        """Estimate memory requirement for a model"""
        num_params = sum(p.numel() for p in model.parameters())

        # Rough estimation: 4 bytes per parameter (float32) * 3 (weights + gradients + optimizer state)
        memory_bytes = num_params * 4 * 3
        memory_gb = memory_bytes / (1024**3)

        return memory_gb

    def _recommend_batch_size(self, memory_gb: float) -> int:
        """Recommend batch size based on memory requirement"""
        if not self.gpu_manager.is_gpu_available:
            return 8  # Conservative for CPU

        # Get available GPU memory
        gpu_info = self.gpu_manager.get_device_info()
        available_memory = gpu_info.get("total_memory_gb", 8) * 0.8  # Use 80% of available memory

        # Estimate batch size (rough approximation)
        if memory_gb < available_memory / 4:
            return 32
        elif memory_gb < available_memory / 2:
            return 16
        elif memory_gb < available_memory:
            return 8
        else:
            return 4

    def optimize_large_model(self, model: nn.Module, model_name: Optional[str] = None) -> nn.Module:
        """Apply optimizations for large models"""

        # Get model config if available
        config = self.model_configs.get(model_name) if model_name else None

        # Move to GPU
        model = self.gpu_manager.optimize_model_for_gpu(model)

        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, "enable_gradient_checkpointing") and callable(
            getattr(model, "enable_gradient_checkpointing")
        ):
            model.enable_gradient_checkpointing()  # type: ignore
            logger.info("Gradient checkpointing enabled")

        # Distributed training setup
        if self.enable_distributed and dist.is_initialized():
            model = DDP(model, device_ids=[torch.cuda.current_device()])
            logger.info("Model wrapped with DistributedDataParallel")

        # Model compilation for optimization (PyTorch 2.0+)
        if hasattr(torch, "compile"):
            try:
                compiled_model = torch.compile(model, mode="max-autotune")  # type: ignore
                model = compiled_model  # type: ignore
                logger.info("Model compiled for optimization")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")

        return model

    def create_efficient_dataloader(
        self, dataset, batch_size: Optional[int] = None, num_workers: Optional[int] = None
    ) -> DataLoader:
        """Create memory-efficient dataloader for large models"""

        if batch_size is None:
            # Use recommended batch size based on available memory
            batch_size = 16  # Default conservative value

        if num_workers is None:
            num_workers = min(8, torch.get_num_threads())

        # Distributed sampler if using distributed training
        sampler = None
        shuffle = True
        if self.enable_distributed and dist.is_initialized():
            sampler = DistributedSampler(dataset)
            shuffle = False

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=self.gpu_manager.is_gpu_available,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else 2,
        )

        logger.info(f"Created dataloader: batch_size={batch_size}, num_workers={num_workers}")
        return dataloader

    def get_model_info(self, model: nn.Module) -> Dict[str, Any]:
        """Get comprehensive information about a model"""

        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Calculate model size in MB
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        model_size_mb = (param_size + buffer_size) / 1024 / 1024

        info = {
            "total_parameters": num_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": model_size_mb,
            "estimated_memory_gb": self._estimate_memory_requirement(model),
            "supports_gradient_checkpointing": hasattr(model, "enable_gradient_checkpointing"),
            "device": str(next(model.parameters()).device),
            "is_distributed": isinstance(model, DDP),
        }

        return info

    def save_large_model(
        self, model: nn.Module, filepath: str, save_optimizer: bool = False, optimizer=None
    ):
        """Save large model with optimizations"""

        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Prepare state dict
        if isinstance(model, DDP):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        save_dict = {"model_state_dict": state_dict, "model_info": self.get_model_info(model)}

        if save_optimizer and optimizer:
            save_dict["optimizer_state_dict"] = optimizer.state_dict()

        # Save with compression for large models
        torch.save(save_dict, filepath, _use_new_zipfile_serialization=True)

        file_size_mb = Path(filepath).stat().st_size / 1024 / 1024
        logger.info(f"Saved large model to {filepath} ({file_size_mb:.2f} MB)")

    def load_large_model(
        self, model: nn.Module, filepath: str, load_optimizer: bool = False, optimizer=None
    ) -> nn.Module:
        """Load large model with optimizations"""

        checkpoint = torch.load(filepath, map_location=self.gpu_manager.device)

        # Load model state
        if isinstance(model, DDP):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state if requested
        if load_optimizer and optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Log model info
        if "model_info" in checkpoint:
            info = checkpoint["model_info"]
            logger.info(
                f"Loaded model: {info['total_parameters']:,} parameters, "
                f"{info['model_size_mb']:.2f} MB"
            )

        return model


# Utility functions
def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def enable_gradient_checkpointing(model: nn.Module):
    """Enable gradient checkpointing for memory efficiency"""
    if hasattr(model, "enable_gradient_checkpointing") and callable(
        getattr(model, "enable_gradient_checkpointing")
    ):
        model.enable_gradient_checkpointing()  # type: ignore
        logger.info("Gradient checkpointing enabled")
    else:
        logger.warning("Model does not support gradient checkpointing")


def estimate_training_memory(
    model: nn.Module, batch_size: int, sequence_length: Optional[int] = None
) -> float:
    """Estimate memory requirement for training"""

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())

    # Memory for parameters (weights)
    param_memory = num_params * 4  # 4 bytes for float32

    # Memory for gradients
    grad_memory = param_memory

    # Memory for optimizer states (assume Adam: 2x parameters)
    optimizer_memory = param_memory * 2

    # Memory for activations (rough estimate)
    if sequence_length:
        # For transformer models
        activation_memory = batch_size * sequence_length * 1024 * 4  # Rough estimate
    else:
        # For CNN models
        activation_memory = batch_size * 224 * 224 * 3 * 4  # Rough estimate for 224x224 images

    total_memory_bytes = param_memory + grad_memory + optimizer_memory + activation_memory
    total_memory_gb = total_memory_bytes / (1024**3)

    return total_memory_gb
