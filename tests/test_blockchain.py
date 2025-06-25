"""
Unit tests for PoUW blockchain core functionality.
"""

import pytest
import time
from pouw.blockchain import (
    Blockchain,
    Transaction,
    PayForTaskTransaction,
    BuyTicketsTransaction,
    MLTask,
)


class TestTransaction:
    """Test transaction functionality"""

    def test_transaction_creation(self):
        """Test basic transaction creation"""
        tx = Transaction(
            version=1,
            inputs=[{"previous_hash": "abc123", "index": 0}],
            outputs=[{"address": "user1", "amount": 10.0}],
        )

        assert tx.version == 1
        assert len(tx.inputs) == 1
        assert len(tx.outputs) == 1
        assert tx.get_hash() is not None

    def test_pay_for_task_transaction(self):
        """Test PAY_FOR_TASK transaction"""
        task_def = {"model_type": "mlp", "architecture": {"input_size": 784, "output_size": 10}}

        tx = PayForTaskTransaction(
            version=1, inputs=[], outputs=[], task_definition=task_def, fee=50.0
        )

        assert tx.fee == 50.0
        assert tx.task_definition == task_def
        assert tx.op_return is not None

    def test_buy_tickets_transaction(self):
        """Test BUY_TICKETS transaction"""
        tx = BuyTicketsTransaction(
            version=1,
            inputs=[],
            outputs=[],
            role="miner",
            stake_amount=100.0,
            preferences={"gpu": True},
        )

        assert tx.role == "miner"
        assert tx.stake_amount == 100.0
        assert tx.preferences == {"gpu": True}
        assert tx.op_return is not None


class TestMLTask:
    """Test ML task functionality"""

    def test_ml_task_creation(self):
        """Test ML task creation"""
        task = MLTask(
            task_id="task_001",
            model_type="mlp",
            architecture={"input_size": 784, "output_size": 10},
            optimizer={"type": "adam", "lr": 0.001},
            stopping_criterion={"max_epochs": 50},
            validation_strategy={"type": "holdout"},
            metrics=["accuracy", "loss"],
            dataset_info={"size": 60000},
            performance_requirements={"gpu": True},
            fee=100.0,
            client_id="client_001",
        )

        assert task.task_id == "task_001"
        assert task.model_type == "mlp"
        assert task.fee == 100.0
        assert task.client_id == "client_001"

        task_dict = task.to_dict()
        assert isinstance(task_dict, dict)
        assert task_dict["task_id"] == "task_001"


class TestBlockchain:
    """Test blockchain functionality"""

    def test_blockchain_initialization(self):
        """Test blockchain initialization"""
        blockchain = Blockchain()

        assert len(blockchain.chain) == 1  # Genesis block
        assert blockchain.get_chain_length() == 1
        assert blockchain.get_mempool_size() == 0

    def test_add_transaction_to_mempool(self):
        """Test adding transaction to mempool"""
        blockchain = Blockchain()

        tx = Transaction(version=1, inputs=[], outputs=[{"address": "user1", "amount": 10.0}])

        success = blockchain.add_transaction_to_mempool(tx)
        assert success
        assert blockchain.get_mempool_size() == 1

    def test_create_block(self):
        """Test block creation"""
        blockchain = Blockchain()

        tx = Transaction(version=1, inputs=[], outputs=[{"address": "user1", "amount": 10.0}])

        block = blockchain.create_block([tx], "miner1")

        assert block.header.previous_hash == blockchain.get_latest_block().get_hash()
        assert len(block.transactions) == 2  # Coinbase + tx
        assert block.transactions[0].outputs[0]["address"] == "miner1"  # Coinbase

    def test_add_valid_block(self):
        """Test adding valid block to chain"""
        blockchain = Blockchain()

        # Use a very easy difficulty for testing
        blockchain.difficulty_target = (
            0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        )

        tx = Transaction(version=1, inputs=[], outputs=[{"address": "user1", "amount": 10.0}])

        blockchain.add_transaction_to_mempool(tx)
        block = blockchain.create_block([tx], "miner1")

        # Any nonce should work with this easy difficulty
        block.header.nonce = 1

        success = blockchain.add_block(block)
        assert success
        assert blockchain.get_chain_length() == 2
        assert blockchain.get_mempool_size() == 0  # Transaction removed from mempool

    def test_validate_block_previous_hash(self):
        """Test block validation with incorrect previous hash"""
        blockchain = Blockchain()

        block = blockchain.create_block([], "miner1")
        block.header.previous_hash = "invalid_hash"

        success = blockchain.add_block(block)
        assert not success
        assert blockchain.get_chain_length() == 1  # Should still be genesis only


if __name__ == "__main__":
    pytest.main([__file__])
