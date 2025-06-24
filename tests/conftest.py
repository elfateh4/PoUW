"""
Test configuration and fixtures for PoUW tests.
"""

import pytest
import torch
import numpy as np
from pouw.blockchain import Blockchain, MLTask
from pouw.ml import SimpleMLP, DistributedTrainer, MiniBatch
from pouw.mining import PoUWMiner, PoUWVerifier
from pouw.economics import EconomicSystem, NodeRole
from pouw.network import P2PNode


@pytest.fixture
def blockchain():
    """Create a test blockchain"""
    return Blockchain()


@pytest.fixture
def simple_mlp():
    """Create a simple MLP model for testing"""
    return SimpleMLP(input_size=784, hidden_sizes=[64, 32], output_size=10)


@pytest.fixture
def sample_task():
    """Create a sample ML task"""
    return MLTask(
        task_id='test_task_001',
        model_type='mlp',
        architecture={'input_size': 784, 'hidden_sizes': [64, 32], 'output_size': 10},
        optimizer={'type': 'adam', 'learning_rate': 0.001},
        stopping_criterion={'type': 'max_epochs', 'max_epochs': 10},
        validation_strategy={'type': 'holdout', 'validation_split': 0.2},
        metrics=['accuracy', 'loss'],
        dataset_info={'format': 'MNIST', 'batch_size': 32, 'size': 1000},
        performance_requirements={'gpu': False, 'expected_training_time': 300},
        fee=50.0,
        client_id='test_client_001'
    )


@pytest.fixture
def sample_minibatch():
    """Create a sample mini-batch"""
    data = np.random.randn(32, 784).astype(np.float32)
    labels = np.random.randint(0, 10, 32)
    return MiniBatch(
        batch_id='test_batch_001',
        data=data,
        labels=labels,
        epoch=0
    )


@pytest.fixture
def distributed_trainer(simple_mlp):
    """Create a distributed trainer"""
    return DistributedTrainer(
        model=simple_mlp,
        task_id='test_task_001',
        miner_id='test_miner_001',
        tau=0.01
    )


@pytest.fixture
def pouw_miner():
    """Create a PoUW miner"""
    return PoUWMiner('test_miner_001')


@pytest.fixture
def pouw_verifier():
    """Create a PoUW verifier"""
    return PoUWVerifier()


@pytest.fixture
def economic_system():
    """Create an economic system"""
    return EconomicSystem()


@pytest.fixture
def p2p_node():
    """Create a P2P node"""
    return P2PNode('test_node_001', 'localhost', 9999)


# Test data fixtures
@pytest.fixture
def mnist_like_data():
    """Generate MNIST-like data for testing"""
    X = np.random.randn(1000, 784).astype(np.float32)
    y = np.random.randint(0, 10, 1000)
    return X, y


@pytest.fixture
def sample_weights():
    """Generate sample model weights"""
    return {
        'layer1.weight': torch.randn(64, 784),
        'layer1.bias': torch.randn(64),
        'layer2.weight': torch.randn(32, 64),
        'layer2.bias': torch.randn(32),
        'output.weight': torch.randn(10, 32),
        'output.bias': torch.randn(10)
    }


# Utility functions for tests
def create_test_transaction(version=1, amount=10.0, address='test_user'):
    """Create a test transaction"""
    from pouw.blockchain import Transaction
    return Transaction(
        version=version,
        inputs=[],
        outputs=[{'address': address, 'amount': amount}]
    )


def create_test_block(blockchain, transactions=None, miner_id='test_miner'):
    """Create a test block"""
    if transactions is None:
        transactions = []
    return blockchain.create_block(transactions, miner_id)


# Test configuration
def pytest_configure(config):
    """Configure pytest"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configure torch for testing
    torch.set_default_dtype(torch.float32)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


# Custom test markers
pytest.register_assert_rewrite('tests.test_helpers')  # Enable assertion rewriting for helpers


class TestHelpers:
    """Helper functions for tests"""
    
    @staticmethod
    def assert_valid_hash(hash_string):
        """Assert that a string is a valid SHA-256 hash"""
        assert isinstance(hash_string, str)
        assert len(hash_string) == 64
        assert all(c in '0123456789abcdef' for c in hash_string.lower())
    
    @staticmethod
    def assert_valid_transaction(transaction):
        """Assert that a transaction is valid"""
        assert hasattr(transaction, 'version')
        assert hasattr(transaction, 'inputs')
        assert hasattr(transaction, 'outputs')
        assert hasattr(transaction, 'get_hash')
        assert transaction.version > 0
        assert isinstance(transaction.inputs, list)
        assert isinstance(transaction.outputs, list)
    
    @staticmethod
    def assert_valid_block(block):
        """Assert that a block is valid"""
        assert hasattr(block, 'header')
        assert hasattr(block, 'transactions')
        assert hasattr(block, 'get_hash')
        assert block.header.version > 0
        assert isinstance(block.transactions, list)
        assert len(block.transactions) >= 1  # At least coinbase transaction
    
    @staticmethod
    def assert_valid_ml_metrics(metrics):
        """Assert that ML metrics are valid"""
        assert isinstance(metrics, dict)
        assert 'loss' in metrics or 'accuracy' in metrics
        
        if 'loss' in metrics:
            assert isinstance(metrics['loss'], (int, float))
            assert metrics['loss'] >= 0
        
        if 'accuracy' in metrics:
            assert isinstance(metrics['accuracy'], (int, float))
            assert 0 <= metrics['accuracy'] <= 1
