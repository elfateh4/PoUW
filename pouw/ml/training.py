"""
Machine Learning training coordination for PoUW.
"""

import hashlib
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod


@dataclass
class MiniBatch:
    """Mini-batch data structure"""

    batch_id: str
    data: np.ndarray
    labels: np.ndarray
    epoch: int

    def get_hash(self) -> str:
        """Calculate hash of mini-batch"""
        data_bytes = self.data.tobytes() + self.labels.tobytes()
        return hashlib.sha256(data_bytes).hexdigest()

    def size(self) -> int:
        """Get size of mini-batch in bytes"""
        return self.data.nbytes + self.labels.nbytes


@dataclass
class GradientUpdate:
    """Gradient update message"""

    miner_id: str
    task_id: str
    iteration: int
    epoch: int
    indices: List[int]  # Indices of weights that changed
    values: List[float]  # New values for those weights
    timestamp: int = field(default_factory=lambda: int(time.time()))

    def to_message_map(self, tau: float) -> bytes:
        """Convert to compressed message map format"""
        message_map = []
        for i, (idx, val) in enumerate(zip(self.indices, self.values)):
            # Pack index and sign bit into 4 bytes
            sign_bit = 1 if val > tau else 0
            packed = (sign_bit << 31) | (idx & 0x7FFFFFFF)
            message_map.extend(packed.to_bytes(4, "big"))
        return bytes(message_map)

    def get_hash(self) -> str:
        """Calculate hash of gradient update"""
        data = json.dumps(
            {
                "miner_id": self.miner_id,
                "task_id": self.task_id,
                "iteration": self.iteration,
                "indices": self.indices,
                "values": self.values,
            },
            sort_keys=True,
        )
        return hashlib.sha256(data.encode()).hexdigest()


@dataclass
class IterationMessage:
    """IT_RES message sent by miners"""

    version: int
    task_id: str
    msg_type: str = "IT_RES"
    epoch: int = 0
    iteration: int = 0
    gradient_updates: Optional[GradientUpdate] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    start_time: int = 0
    finish_time: int = 0
    batch_hash: str = ""
    peer_updates_hash: str = ""
    model_state_hash: str = ""
    gradient_residual_hash: str = ""
    zero_nonce_block_hash: str = ""
    peer_messages_hash: str = ""
    signature: Optional[bytes] = None

    def serialize(self) -> bytes:
        """Serialize message for network transmission"""
        data = {
            "version": self.version,
            "task_id": self.task_id,
            "msg_type": self.msg_type,
            "epoch": self.epoch,
            "iteration": self.iteration,
            "gradient_updates": self.gradient_updates.__dict__ if self.gradient_updates else None,
            "metrics": self.metrics,
            "start_time": self.start_time,
            "finish_time": self.finish_time,
            "batch_hash": self.batch_hash,
            "peer_updates_hash": self.peer_updates_hash,
            "model_state_hash": self.model_state_hash,
            "gradient_residual_hash": self.gradient_residual_hash,
            "zero_nonce_block_hash": self.zero_nonce_block_hash,
            "peer_messages_hash": self.peer_messages_hash,
        }
        return json.dumps(data).encode()

    def get_hash(self) -> str:
        """Calculate message hash"""
        data = self.serialize()
        return hashlib.sha256(data).hexdigest()


class MLModel(ABC):
    """Abstract base class for ML models"""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Make the model callable"""
        return self.forward(x)

    @abstractmethod
    def get_weights(self) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def set_weights(self, weights: Dict[str, torch.Tensor]):
        pass

    @abstractmethod
    def get_gradients(self) -> Dict[str, torch.Tensor]:
        pass


class SimpleMLP(nn.Module, MLModel):
    """Simple Multi-Layer Perceptron"""

    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Get current model weights"""
        return {name: param.clone() for name, param in self.named_parameters()}

    def set_weights(self, weights: Dict[str, torch.Tensor]):
        """Set model weights"""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in weights:
                    param.copy_(weights[name])

    def get_gradients(self) -> Dict[str, torch.Tensor]:
        """Get current gradients"""
        return {
            name: param.grad.clone() if param.grad is not None else torch.zeros_like(param)
            for name, param in self.named_parameters()
        }


class DistributedTrainer:
    """Coordinates distributed ML training for PoUW"""

    def __init__(self, model: MLModel, task_id: str, miner_id: str, tau: float = 1e-4):
        self.model = model
        self.task_id = task_id
        self.miner_id = miner_id
        self.tau = tau  # Threshold for gradient updates

        self.current_epoch = 0
        self.current_iteration = 0
        self.gradient_residual = {}  # Accumulated gradients
        self.peer_updates = []  # Updates from other miners
        self.message_history = []  # History of all messages

        # Initialize gradient residual
        self._initialize_gradient_residual()

    def _initialize_gradient_residual(self):
        """Initialize gradient residual to zeros"""
        # Use getattr to safely access named_parameters
        named_params_func = getattr(self.model, "named_parameters", None)
        if named_params_func is not None:
            for name, param in named_params_func():
                self.gradient_residual[name] = torch.zeros_like(param)
        else:
            # Fallback for models without named_parameters
            weights = self.model.get_weights()
            for name, param in weights.items():
                self.gradient_residual[name] = torch.zeros_like(param)

    def process_iteration(
        self, batch: MiniBatch, optimizer: optim.Optimizer, criterion: nn.Module
    ) -> Tuple[IterationMessage, Dict[str, float]]:
        """Process one training iteration following Algorithm 1 from the paper"""

        start_time = int(time.time() * 1000)  # milliseconds

        # Step 1: Load mini-batch
        x_batch = torch.tensor(batch.data, dtype=torch.float32)
        y_batch = torch.tensor(batch.labels, dtype=torch.long)

        # Step 2: Apply peer updates to local model
        self._apply_peer_updates()

        # Step 3: Forward pass and compute gradients
        optimizer.zero_grad()
        outputs = self.model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()

        # Step 4: Get local gradients
        local_gradients = self.model.get_gradients()

        # Step 5: Update gradient residual
        for name in local_gradients:
            self.gradient_residual[name] += local_gradients[name]

        # Step 6: Extract significant updates (threshold τ)
        gradient_update = self._extract_gradient_updates()

        # Step 7: Apply local updates to model
        self._apply_local_updates(gradient_update)

        # Step 8: Calculate metrics
        with torch.no_grad():
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == y_batch).float().mean().item()
            metrics = {"loss": loss.item(), "accuracy": accuracy}

        finish_time = int(time.time() * 1000)

        # Step 9: Create iteration message
        message = IterationMessage(
            version=1,
            task_id=self.task_id,
            epoch=self.current_epoch,
            iteration=self.current_iteration,
            gradient_updates=gradient_update,
            metrics=metrics,
            start_time=start_time,
            finish_time=finish_time,
            batch_hash=batch.get_hash(),
            peer_updates_hash=self._calculate_peer_updates_hash(),
            model_state_hash=self._calculate_model_state_hash(),
            gradient_residual_hash=self._calculate_gradient_residual_hash(),
            zero_nonce_block_hash="",  # Set during mining
            peer_messages_hash="",  # Set with peer messages
        )

        self.message_history.append(message)
        self.current_iteration += 1

        return message, metrics

    def _apply_peer_updates(self):
        """Apply updates received from other miners"""
        with torch.no_grad():
            for update in self.peer_updates:
                weights = self.model.get_weights()
                for idx, value in zip(update.indices, update.values):
                    # Find the parameter and index within it
                    param_name, param_idx = self._linear_index_to_param(idx)
                    if param_name in weights:
                        flat_param = weights[param_name].flatten()
                        if param_idx < len(flat_param):
                            flat_param[param_idx] -= value  # SGD update
                        weights[param_name] = flat_param.reshape(weights[param_name].shape)

                self.model.set_weights(weights)

        self.peer_updates.clear()

    def _extract_gradient_updates(self) -> GradientUpdate:
        """Extract gradient updates that exceed threshold τ"""
        indices = []
        values = []

        linear_idx = 0
        for name, grad_residual in self.gradient_residual.items():
            flat_grad = grad_residual.flatten()

            for i, val in enumerate(flat_grad):
                if abs(val.item()) > self.tau:
                    indices.append(linear_idx + i)
                    values.append(val.item())

                    # Reset residual for this index
                    flat_grad[i] = 0.0

            # Update the residual
            self.gradient_residual[name] = flat_grad.reshape(grad_residual.shape)
            linear_idx += flat_grad.numel()

        return GradientUpdate(
            miner_id=self.miner_id,
            task_id=self.task_id,
            iteration=self.current_iteration,
            epoch=self.current_epoch,
            indices=indices,
            values=values,
        )

    def _apply_local_updates(self, gradient_update: GradientUpdate):
        """Apply local gradient updates to the model"""
        with torch.no_grad():
            weights = self.model.get_weights()

            for idx, value in zip(gradient_update.indices, gradient_update.values):
                param_name, param_idx = self._linear_index_to_param(idx)
                if param_name in weights:
                    flat_param = weights[param_name].flatten()
                    if param_idx < len(flat_param):
                        flat_param[param_idx] -= value  # SGD update
                    weights[param_name] = flat_param.reshape(weights[param_name].shape)

            self.model.set_weights(weights)

    def _linear_index_to_param(self, linear_idx: int) -> Tuple[str, int]:
        """Convert linear index to parameter name and index within parameter"""
        current_idx = 0

        # Use getattr to safely access named_parameters
        named_params_func = getattr(self.model, "named_parameters", None)
        if named_params_func is not None:
            for name, param in named_params_func():
                param_size = param.numel()
                if linear_idx < current_idx + param_size:
                    return name, linear_idx - current_idx
                current_idx += param_size
        else:
            # Fallback using get_weights
            weights = self.model.get_weights()
            for name, param in weights.items():
                param_size = param.numel()
                if linear_idx < current_idx + param_size:
                    return name, linear_idx - current_idx
                current_idx += param_size

        raise IndexError(f"Linear index {linear_idx} out of bounds")

    def _calculate_peer_updates_hash(self) -> str:
        """Calculate hash of applied peer updates"""
        if not self.peer_updates:
            return "0" * 64

        data = json.dumps([update.__dict__ for update in self.peer_updates], sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

    def _calculate_model_state_hash(self) -> str:
        """Calculate hash of current model state"""
        weights = self.model.get_weights()
        weights_bytes = b""
        for name in sorted(weights.keys()):
            weights_bytes += weights[name].detach().numpy().tobytes()

        return hashlib.sha256(weights_bytes).hexdigest()

    def _calculate_gradient_residual_hash(self) -> str:
        """Calculate hash of gradient residual"""
        residual_bytes = b""
        for name in sorted(self.gradient_residual.keys()):
            residual_bytes += self.gradient_residual[name].detach().numpy().tobytes()

        return hashlib.sha256(residual_bytes).hexdigest()

    def add_peer_update(self, update: GradientUpdate):
        """Add update received from peer miner"""
        self.peer_updates.append(update)

    def get_model_weights_for_nonce(self) -> bytes:
        """Get model weights for nonce generation"""
        weights = self.model.get_weights()
        weights_bytes = b""
        for name in sorted(weights.keys()):
            weights_bytes += weights[name].detach().numpy().tobytes()
        return weights_bytes

    def get_local_gradients_for_nonce(self) -> bytes:
        """Get local gradients for nonce generation"""
        gradients_bytes = b""
        for name in sorted(self.gradient_residual.keys()):
            gradients_bytes += self.gradient_residual[name].detach().numpy().tobytes()
        return gradients_bytes
