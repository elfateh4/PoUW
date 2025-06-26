"""
PoUW Mining Algorithm Implementation.

This module implements the core mining algorithm from the paper,
where nonces are derived from useful ML work.
"""

import hashlib
import json
import time
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

from ..blockchain.core import Block, PoUWBlockHeader, Transaction
from ..ml.training import DistributedTrainer, IterationMessage, GradientUpdate


@dataclass
class MiningProof:
    """Proof data for PoUW mining"""

    nonce_precursor: str
    model_weights_hash: str
    local_gradients_hash: str
    iteration_data: Dict[str, Any]
    mini_batch_hash: str
    peer_updates: List[GradientUpdate]
    message_history_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nonce_precursor": self.nonce_precursor,
            "model_weights_hash": self.model_weights_hash,
            "local_gradients_hash": self.local_gradients_hash,
            "iteration_data": self.iteration_data,
            "mini_batch_hash": self.mini_batch_hash,
            "peer_updates": [update.__dict__ for update in self.peer_updates],
            "message_history_ids": self.message_history_ids,
        }


class PoUWMiner:
    """PoUW Miner implementation following Algorithm 2 from the paper"""

    def __init__(self, miner_id: str, omega_b: float = 1e-6, omega_m: float = 1e-8):
        self.miner_id = miner_id
        self.omega_b = omega_b  # Network coefficient for batch size
        self.omega_m = omega_m  # Network coefficient for model size
        self.zero_nonce_blocks = {}  # k iterations ahead commitments
        self.k = 10  # Number of iterations to commit ahead

    def mine_block(
        self,
        trainer: DistributedTrainer,
        iteration_message: IterationMessage,
        batch_size: int,
        model_size: int,
        transactions: List[Transaction],
        blockchain,
        economic_system=None,
    ) -> Optional[Tuple[Block, MiningProof]]:
        """
        Mine a block using PoUW algorithm (Algorithm 2 from paper).

        Args:
            trainer: The distributed ML trainer
            iteration_message: The IT_RES message from current iteration
            batch_size: Size of mini-batch in bytes
            model_size: Number of weights and biases in model
            transactions: Transactions to include in block
            blockchain: Reference to blockchain for block creation
            economic_system: Optional economic system for dynamic block rewards

        Returns:
            Tuple of (Block, MiningProof) if successful, None otherwise
        """

        # Step 1: Build nonce precursor from ML work
        model_weights = trainer.get_model_weights_for_nonce()
        local_gradients = trainer.get_local_gradients_for_nonce()

        # Combine model state and gradients for nonce precursor
        combined_data = model_weights + local_gradients
        nonce_precursor = hashlib.sha256(combined_data).hexdigest()

        # Step 2: Build nonce from precursor
        base_nonce = hashlib.sha256(nonce_precursor.encode()).hexdigest()
        base_nonce_int = int(base_nonce, 16)

        # Step 3: Calculate allowed number of nonces
        a = int(self.omega_b * batch_size + self.omega_m * model_size)
        a = max(a, 1)  # Ensure at least one nonce attempt

        print(f"Mining with {a} nonce attempts")

        # Step 4: Try mining with each allowed nonce
        for j in range(a):
            current_nonce = base_nonce_int + j

            # Create block with current nonce
            block = self._create_block_with_nonce(
                current_nonce, iteration_message, transactions, blockchain, economic_system
            )

            # Check if block meets difficulty target
            if self._check_proof_of_work(block, blockchain.difficulty_target):
                print(f"Successfully mined block with nonce {current_nonce}")

                # Create mining proof
                mining_proof = MiningProof(
                    nonce_precursor=nonce_precursor,
                    model_weights_hash=hashlib.sha256(model_weights).hexdigest(),
                    local_gradients_hash=hashlib.sha256(local_gradients).hexdigest(),
                    iteration_data={
                        "epoch": iteration_message.epoch,
                        "iteration": iteration_message.iteration,
                        "metrics": iteration_message.metrics,
                    },
                    mini_batch_hash=iteration_message.batch_hash,
                    peer_updates=trainer.peer_updates.copy(),
                    message_history_ids=[msg.get_hash() for msg in trainer.message_history[-10:]],
                )

                block.mining_proof = mining_proof.to_dict()

                # Store mining data for verification
                self._store_mining_data(block, trainer, iteration_message)

                return block, mining_proof

        print("Mining attempt failed - no valid nonce found")
        return None

    def _create_block_with_nonce(
        self,
        nonce: int,
        iteration_message: IterationMessage,
        transactions: List[Transaction],
        blockchain,
        economic_system=None,
    ) -> Block:
        """Create a block with the given nonce"""

        previous_block = blockchain.get_latest_block()

        # Get zero-nonce block if we committed to one k iterations ago
        znb_hash = self._get_zero_nonce_block_hash(iteration_message)

        header = PoUWBlockHeader(
            version=1,
            previous_hash=previous_block.get_hash(),
            merkle_root="",  # Will be calculated
            timestamp=int(time.time()),
            target=blockchain.difficulty_target,
            nonce=nonce,
            ml_task_id=iteration_message.task_id,
            message_history_hash=self._calculate_message_history_hash(iteration_message),
            iteration_message_hash=iteration_message.get_hash(),
            zero_nonce_block_hash=znb_hash,
        )

        # Create coinbase transaction with dynamic block reward
        if economic_system:
            # Calculate block reward based on current supply and halving schedule
            block_height = len(blockchain.chain)
            block_reward = economic_system.calculate_block_reward(block_height)
        else:
            # Fallback to default reward if no economic system provided
            block_reward = 12.5
        
        coinbase_tx = Transaction(
            version=1,
            inputs=[{"previous_hash": "0" * 64, "index": -1}],
            outputs=[{"address": self.miner_id, "amount": block_reward}],
        )

        all_transactions = [coinbase_tx] + transactions

        block = Block(header=header, transactions=all_transactions)
        return block

    def _check_proof_of_work(self, block: Block, difficulty_target: int) -> bool:
        """Check if block hash meets difficulty target"""
        block_hash_int = int(block.get_hash(), 16)
        return block_hash_int < difficulty_target

    def _store_mining_data(
        self, block: Block, trainer: DistributedTrainer, iteration_message: IterationMessage
    ):
        """Store data needed for verification"""
        # In a real implementation, this would store to persistent storage
        # For now, we'll store in memory
        storage_key = block.get_hash()

        mining_data = {
            "model_weights": trainer.model.get_weights(),
            "mini_batch_hash": iteration_message.batch_hash,
            "gradient_residual": trainer.gradient_residual.copy(),
            "peer_updates": trainer.peer_updates.copy(),
            "message_history": trainer.message_history.copy(),
        }

        # Store for later verification
        setattr(self, f"_mining_data_{storage_key}", mining_data)

    def commit_zero_nonce_block(
        self, iteration: int, transactions: List[Transaction], blockchain, economic_system=None
    ) -> str:
        """
        Commit to a zero-nonce block k iterations in advance.
        This prevents miners from manipulating transactions for better hashes.
        """

        # Create block with nonce = 0 and fixed transactions
        previous_block = blockchain.get_latest_block()

        header = PoUWBlockHeader(
            version=1,
            previous_hash=previous_block.get_hash(),
            merkle_root="",
            timestamp=int(time.time()),
            target=blockchain.difficulty_target,
            nonce=0,  # Zero nonce
            ml_task_id="",
            message_history_hash="",
            iteration_message_hash="",
            zero_nonce_block_hash="",
        )

        # Create with fixed transactions and dynamic block reward
        if economic_system:
            # Calculate block reward for future block
            future_block_height = len(blockchain.chain) + self.k
            block_reward = economic_system.calculate_block_reward(future_block_height)
        else:
            # Fallback to default reward
            block_reward = 12.5
            
        coinbase_tx = Transaction(
            version=1,
            inputs=[{"previous_hash": "0" * 64, "index": -1}],
            outputs=[{"address": self.miner_id, "amount": block_reward}],
        )

        all_transactions = [coinbase_tx] + transactions
        znb = Block(header=header, transactions=all_transactions)
        znb_hash = znb.get_hash()

        # Store commitment for future use
        self.zero_nonce_blocks[iteration + self.k] = znb_hash

        return znb_hash

    def _get_zero_nonce_block_hash(self, iteration_message: IterationMessage) -> str:
        """Get the zero-nonce block hash for current iteration"""
        return self.zero_nonce_blocks.get(iteration_message.iteration, "")

    def _calculate_message_history_hash(self, iteration_message: IterationMessage) -> str:
        """Calculate Merkle tree hash of message history"""
        # Simplified - in practice this would be a proper Merkle tree
        message_data = iteration_message.serialize()
        return hashlib.sha256(message_data).hexdigest()


class PoUWVerifier:
    """Verifies PoUW mining proofs by re-running ML iterations"""

    def __init__(self):
        self.verification_cache = {}

    def verify_block(
        self, block: Block, mining_proof: MiningProof, trainer: DistributedTrainer
    ) -> bool:
        """
        Verify a PoUW block by re-running the ML iteration.

        This implements the verification steps from the paper:
        1. Check if block hash meets target
        2. Verify mini-batch exists and should be processed
        3. Verify i+k commitment is valid
        4. Re-run ML iteration and check metrics match
        5. Verify compressed peer messages hash
        6. Reconstruct nonce and compare
        """

        # Step 1: Check basic PoW
        if not self._verify_basic_proof_of_work(block):
            return False

        # Step 2: Verify mining proof structure
        if not mining_proof:
            return False

        # Step 3: Verify nonce construction
        if not self._verify_nonce_construction(block, mining_proof):
            return False

        # Step 4: Verify ML work (would re-run iteration in practice)
        if not self._verify_ml_iteration(block, mining_proof, trainer):
            return False

        return True

    def _verify_basic_proof_of_work(self, block: Block) -> bool:
        """Verify block hash meets difficulty target"""
        # This would check against the actual network difficulty
        # For now, just check that it's a valid hash
        try:
            block_hash = block.get_hash()
            int(block_hash, 16)  # Valid hex
            return True
        except ValueError:
            return False

    def _verify_nonce_construction(self, block: Block, mining_proof: MiningProof) -> bool:
        """Verify nonce was constructed correctly from ML work"""

        # The nonce precursor should be provided in the mining proof
        # In a real implementation, we would reconstruct it from the stored ML data
        # For now, we trust the provided nonce precursor and verify it was used correctly

        # Verify nonce derivation from the precursor
        expected_base_nonce = hashlib.sha256(mining_proof.nonce_precursor.encode()).hexdigest()
        expected_base_nonce_int = int(expected_base_nonce, 16)

        # Check if block nonce is within allowed range
        nonce_diff = block.header.nonce - expected_base_nonce_int
        if nonce_diff < 0 or nonce_diff >= 1000:  # Reasonable limit
            return False

        return True

    def _verify_ml_iteration(
        self, block: Block, mining_proof: MiningProof, trainer: DistributedTrainer
    ) -> bool:
        """Verify the ML iteration was performed correctly"""

        # In a full implementation, this would:
        # 1. Load the exact mini-batch used
        # 2. Set model to the starting state
        # 3. Apply peer updates
        # 4. Re-run forward/backward pass
        # 5. Compare resulting metrics and gradients

        # For this proof-of-concept, we'll do basic checks
        iteration_data = mining_proof.iteration_data

        # Check that metrics are reasonable
        if "loss" in iteration_data["metrics"]:
            loss = iteration_data["metrics"]["loss"]
            if loss < 0 or loss > 1000:  # Unreasonable loss values
                return False

        if "accuracy" in iteration_data["metrics"]:
            accuracy = iteration_data["metrics"]["accuracy"]
            if accuracy < 0 or accuracy > 1:  # Accuracy should be [0,1]
                return False

        return True

    def get_verification_digest(self, block: Block, is_valid: bool) -> str:
        """Create verification digest for the block"""
        digest_data = {
            "block_hash": block.get_hash(),
            "verified_at": int(time.time()),
            "is_valid": is_valid,
            "verifier_id": "verifier_001",  # Would be actual verifier ID
        }

        digest_string = json.dumps(digest_data, sort_keys=True)
        return hashlib.sha256(digest_string.encode()).hexdigest()
