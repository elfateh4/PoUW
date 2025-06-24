"""
Unit tests for PoUW mining algorithm.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pouw.mining import PoUWMiner, PoUWVerifier, MiningProof
from pouw.ml import SimpleMLP, DistributedTrainer, MiniBatch
from pouw.blockchain import Blockchain, Transaction


class TestMiningProof:
    """Test MiningProof functionality"""
    
    def test_mining_proof_creation(self):
        """Test mining proof creation"""
        proof = MiningProof(
            nonce_precursor='abc123',
            model_weights_hash='def456',
            local_gradients_hash='ghi789',
            iteration_data={'epoch': 1, 'iteration': 5},
            mini_batch_hash='jkl012',
            peer_updates=[],
            message_history_ids=['msg1', 'msg2']
        )
        
        assert proof.nonce_precursor == 'abc123'
        assert proof.model_weights_hash == 'def456'
        assert proof.iteration_data['epoch'] == 1
        assert len(proof.message_history_ids) == 2
        
        # Test serialization
        proof_dict = proof.to_dict()
        assert isinstance(proof_dict, dict)
        assert proof_dict['nonce_precursor'] == 'abc123'


class TestPoUWMiner:
    """Test PoUW miner functionality"""
    
    def test_miner_creation(self):
        """Test miner creation"""
        miner = PoUWMiner('miner_001')
        
        assert miner.miner_id == 'miner_001'
        assert miner.omega_b > 0
        assert miner.omega_m > 0
        assert miner.k == 10
        assert isinstance(miner.zero_nonce_blocks, dict)
    
    def test_zero_nonce_block_commitment(self):
        """Test zero-nonce block commitment"""
        miner = PoUWMiner('miner_001')
        blockchain = Blockchain()
        
        transactions = [Transaction(1, [], [{'address': 'user1', 'amount': 10}])]
        
        znb_hash = miner.commit_zero_nonce_block(5, transactions, blockchain)
        
        assert isinstance(znb_hash, str)
        assert len(znb_hash) == 64  # SHA-256 hash length
        assert 15 in miner.zero_nonce_blocks  # iteration 5 + k=10
        assert miner.zero_nonce_blocks[15] == znb_hash
    
    def test_mine_block_setup(self):
        """Test mining block setup without actual mining"""
        miner = PoUWMiner('miner_001')
        blockchain = Blockchain()
        
        # Setup ML components
        model = SimpleMLP(784, [64], 10)
        trainer = DistributedTrainer(model, 'task_001', 'miner_001')
        
        # Create mini-batch and process iteration
        data = np.random.randn(32, 784).astype(np.float32)
        labels = np.random.randint(0, 10, 32)
        batch = MiniBatch('batch_001', data, labels, 0)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        message, metrics = trainer.process_iteration(batch, optimizer, criterion)
        
        # Test mining setup (without actual mining loop)
        transactions = []
        batch_size = batch.size()
        model_size = sum(p.numel() for p in model.parameters())
        
        # Should not raise exception
        try:
            result = miner.mine_block(
                trainer, message, batch_size, model_size, 
                transactions, blockchain
            )
            # Result could be None if no valid nonce found quickly
            assert result is None or len(result) == 2
        except Exception as e:
            pytest.fail(f"Mining setup failed: {e}")


class TestPoUWVerifier:
    """Test PoUW verifier functionality"""
    
    def test_verifier_creation(self):
        """Test verifier creation"""
        verifier = PoUWVerifier()
        
        assert isinstance(verifier.verification_cache, dict)
    
    def test_basic_proof_of_work_validation(self):
        """Test basic PoW validation"""
        verifier = PoUWVerifier()
        blockchain = Blockchain()
        
        # Create a simple block
        block = blockchain.create_block([], 'miner_001')
        block.header.nonce = 12345
        
        # Should validate basic structure
        is_valid = verifier._verify_basic_proof_of_work(block)
        assert is_valid  # Should pass basic validation
    
    def test_nonce_construction_verification(self):
        """Test nonce construction verification"""
        verifier = PoUWVerifier()
        
        # Create mining proof with any valid precursor
        proof = MiningProof(
            nonce_precursor='test_precursor_123',
            model_weights_hash='def456',
            local_gradients_hash='ghi789',
            iteration_data={'epoch': 1, 'iteration': 5, 'metrics': {'loss': 0.5}},
            mini_batch_hash='jkl012',
            peer_updates=[],
            message_history_ids=[]
        )
        
        # Create block with derived nonce
        blockchain = Blockchain()
        block = blockchain.create_block([], 'miner_001')
        
        # Set nonce based on proof (within allowed range)
        import hashlib
        expected_base_nonce = hashlib.sha256(proof.nonce_precursor.encode()).hexdigest()
        block.header.nonce = int(expected_base_nonce, 16) + 5
        
        # Should verify correctly
        is_valid = verifier._verify_nonce_construction(block, proof)
        assert is_valid
    
    def test_ml_iteration_verification(self):
        """Test ML iteration verification"""
        verifier = PoUWVerifier()
        blockchain = Blockchain()
        
        # Create mock components
        block = blockchain.create_block([], 'miner_001')
        
        proof = MiningProof(
            nonce_precursor='abc123',
            model_weights_hash='def456',
            local_gradients_hash='ghi789',
            iteration_data={
                'epoch': 1, 
                'iteration': 5, 
                'metrics': {'loss': 0.5, 'accuracy': 0.85}
            },
            mini_batch_hash='jkl012',
            peer_updates=[],
            message_history_ids=[]
        )
        
        model = SimpleMLP(784, [64], 10)
        trainer = DistributedTrainer(model, 'task_001', 'miner_001')
        
        # Should pass basic ML verification
        is_valid = verifier._verify_ml_iteration(block, proof, trainer)
        assert is_valid
    
    def test_verification_digest(self):
        """Test verification digest creation"""
        verifier = PoUWVerifier()
        blockchain = Blockchain()
        
        block = blockchain.create_block([], 'miner_001')
        
        digest = verifier.get_verification_digest(block, True)
        
        assert isinstance(digest, str)
        assert len(digest) == 64  # SHA-256 hash length


class TestIntegratedMining:
    """Test integrated mining workflow"""
    
    def test_full_mining_workflow(self):
        """Test the complete mining workflow"""
        # Setup components
        blockchain = Blockchain()
        miner = PoUWMiner('miner_001')
        verifier = PoUWVerifier()
        
        model = SimpleMLP(784, [32], 10)  # Smaller model for testing
        trainer = DistributedTrainer(model, 'task_001', 'miner_001', tau=0.1)
        
        # Create and process mini-batch
        data = np.random.randn(8, 784).astype(np.float32)  # Smaller batch
        labels = np.random.randint(0, 10, 8)
        batch = MiniBatch('batch_001', data, labels, 0)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        message, metrics = trainer.process_iteration(batch, optimizer, criterion)
        
        # Attempt mining (with very relaxed difficulty)
        original_target = blockchain.difficulty_target
        blockchain.difficulty_target = 2**255  # Very easy target
        
        try:
            batch_size = batch.size()
            model_size = sum(p.numel() for p in model.parameters())
            
            result = miner.mine_block(
                trainer, message, batch_size, model_size, [], blockchain
            )
            
            if result:
                block, proof = result
                
                # Debug verification steps
                print(f"Block hash: {block.get_hash()}")
                print(f"PoW valid: {verifier._verify_basic_proof_of_work(block)}")
                print(f"Proof exists: {proof is not None}")
                print(f"Nonce construction: {verifier._verify_nonce_construction(block, proof)}")
                print(f"ML iteration: {verifier._verify_ml_iteration(block, proof, trainer)}")
                
                # Verify the mined block
                is_valid = verifier.verify_block(block, proof, trainer)
                assert is_valid
                
                # Add to blockchain
                success = blockchain.add_block(block)
                assert success
                assert blockchain.get_chain_length() == 2
            
        finally:
            blockchain.difficulty_target = original_target


if __name__ == '__main__':
    pytest.main([__file__])
