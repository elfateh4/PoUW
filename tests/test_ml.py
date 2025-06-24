"""
Unit tests for PoUW ML training functionality.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pouw.ml import SimpleMLP, DistributedTrainer, MiniBatch, GradientUpdate


class TestSimpleMLP:
    """Test Simple MLP model"""
    
    def test_model_creation(self):
        """Test model creation"""
        model = SimpleMLP(
            input_size=784,
            hidden_sizes=[128, 64],
            output_size=10
        )
        
        assert isinstance(model, nn.Module)
        
        # Test forward pass
        x = torch.randn(32, 784)
        output = model(x)
        assert output.shape == (32, 10)
    
    def test_get_set_weights(self):
        """Test weight getting and setting"""
        model = SimpleMLP(784, [64], 10)
        
        # Get weights
        weights = model.get_weights()
        assert isinstance(weights, dict)
        assert len(weights) > 0
        
        # Modify weights
        modified_weights = {}
        for name, param in weights.items():
            modified_weights[name] = param + 0.1
        
        # Set modified weights
        model.set_weights(modified_weights)
        
        # Verify weights changed
        new_weights = model.get_weights()
        for name in weights:
            assert not torch.equal(weights[name], new_weights[name])
    
    def test_get_gradients(self):
        """Test gradient extraction"""
        model = SimpleMLP(784, [64], 10)
        
        # Forward and backward pass
        x = torch.randn(32, 784)
        y = torch.randint(0, 10, (32,))
        criterion = nn.CrossEntropyLoss()
        
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        
        # Get gradients
        gradients = model.get_gradients()
        assert isinstance(gradients, dict)
        assert len(gradients) > 0
        
        # Check that gradients are non-zero
        has_nonzero_grad = False
        for grad in gradients.values():
            if torch.any(grad != 0):
                has_nonzero_grad = True
                break
        assert has_nonzero_grad


class TestMiniBatch:
    """Test MiniBatch functionality"""
    
    def test_minibatch_creation(self):
        """Test mini-batch creation"""
        data = np.random.randn(32, 784)
        labels = np.random.randint(0, 10, 32)
        
        batch = MiniBatch(
            batch_id='batch_001',
            data=data,
            labels=labels,
            epoch=1
        )
        
        assert batch.batch_id == 'batch_001'
        assert batch.data.shape == (32, 784)
        assert batch.labels.shape == (32,)
        assert batch.epoch == 1
    
    def test_minibatch_hash(self):
        """Test mini-batch hashing"""
        data = np.random.randn(32, 784)
        labels = np.random.randint(0, 10, 32)
        
        batch1 = MiniBatch('batch_001', data, labels, 1)
        batch2 = MiniBatch('batch_001', data, labels, 1)
        batch3 = MiniBatch('batch_001', data + 0.1, labels, 1)
        
        # Same data should produce same hash
        assert batch1.get_hash() == batch2.get_hash()
        
        # Different data should produce different hash
        assert batch1.get_hash() != batch3.get_hash()
    
    def test_minibatch_size(self):
        """Test mini-batch size calculation"""
        data = np.random.randn(32, 784).astype(np.float32)
        labels = np.random.randint(0, 10, 32).astype(np.int64)
        
        batch = MiniBatch('batch_001', data, labels, 1)
        size = batch.size()
        
        expected_size = data.nbytes + labels.nbytes
        assert size == expected_size


class TestGradientUpdate:
    """Test GradientUpdate functionality"""
    
    def test_gradient_update_creation(self):
        """Test gradient update creation"""
        update = GradientUpdate(
            miner_id='miner_001',
            task_id='task_001',
            iteration=5,
            epoch=1,
            indices=[0, 10, 25],
            values=[0.1, -0.05, 0.2]
        )
        
        assert update.miner_id == 'miner_001'
        assert update.task_id == 'task_001'
        assert update.iteration == 5
        assert update.epoch == 1
        assert update.indices == [0, 10, 25]
        assert update.values == [0.1, -0.05, 0.2]
    
    def test_gradient_update_hash(self):
        """Test gradient update hashing"""
        update1 = GradientUpdate('miner_001', 'task_001', 5, 1, [0, 10], [0.1, -0.05])
        update2 = GradientUpdate('miner_001', 'task_001', 5, 1, [0, 10], [0.1, -0.05])
        update3 = GradientUpdate('miner_001', 'task_001', 6, 1, [0, 10], [0.1, -0.05])
        
        # Same updates should have same hash
        assert update1.get_hash() == update2.get_hash()
        
        # Different updates should have different hash
        assert update1.get_hash() != update3.get_hash()
    
    def test_message_map_conversion(self):
        """Test conversion to message map format"""
        update = GradientUpdate(
            'miner_001', 'task_001', 5, 1, 
            [0, 10, 25], [0.1, -0.05, 0.2]
        )
        
        tau = 0.01
        message_map = update.to_message_map(tau)
        
        assert isinstance(message_map, bytes)
        assert len(message_map) == len(update.indices) * 4  # 4 bytes per index


class TestDistributedTrainer:
    """Test DistributedTrainer functionality"""
    
    def test_trainer_creation(self):
        """Test trainer creation"""
        model = SimpleMLP(784, [64], 10)
        trainer = DistributedTrainer(model, 'task_001', 'miner_001')
        
        assert trainer.task_id == 'task_001'
        assert trainer.miner_id == 'miner_001'
        assert trainer.current_epoch == 0
        assert trainer.current_iteration == 0
        assert len(trainer.gradient_residual) > 0
    
    def test_process_iteration(self):
        """Test processing a training iteration"""
        model = SimpleMLP(784, [64], 10)
        trainer = DistributedTrainer(model, 'task_001', 'miner_001', tau=0.01)
        
        # Create mini-batch
        data = np.random.randn(32, 784).astype(np.float32)
        labels = np.random.randint(0, 10, 32)
        batch = MiniBatch('batch_001', data, labels, 0)
        
        # Setup optimizer and criterion
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Process iteration
        message, metrics = trainer.process_iteration(batch, optimizer, criterion)
        
        # Check message
        assert message.task_id == 'task_001'
        assert message.msg_type == 'IT_RES'
        assert message.iteration == 0
        assert message.batch_hash == batch.get_hash()
        
        # Check metrics
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert isinstance(metrics['loss'], float)
        assert isinstance(metrics['accuracy'], float)
        
        # Check trainer state updated
        assert trainer.current_iteration == 1
    
    def test_peer_updates(self):
        """Test adding and applying peer updates"""
        model = SimpleMLP(784, [64], 10)
        trainer = DistributedTrainer(model, 'task_001', 'miner_001')
        
        # Add peer update
        update = GradientUpdate(
            'miner_002', 'task_001', 0, 0, [0, 10], [0.1, -0.05]
        )
        trainer.add_peer_update(update)
        
        assert len(trainer.peer_updates) == 1
        assert trainer.peer_updates[0] == update
    
    def test_weight_and_gradient_extraction(self):
        """Test extracting weights and gradients for nonce"""
        model = SimpleMLP(784, [64], 10)
        trainer = DistributedTrainer(model, 'task_001', 'miner_001')
        
        # Get weights for nonce
        weights_bytes = trainer.get_model_weights_for_nonce()
        assert isinstance(weights_bytes, bytes)
        assert len(weights_bytes) > 0
        
        # Get gradients for nonce
        gradients_bytes = trainer.get_local_gradients_for_nonce()
        assert isinstance(gradients_bytes, bytes)
        assert len(gradients_bytes) > 0


if __name__ == '__main__':
    pytest.main([__file__])
