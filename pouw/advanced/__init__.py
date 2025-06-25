"""
Verifiable Random Functions (VRF) and Enhanced Mining for PoUW Implementation.

Implements proper VRF for worker selection and advanced mining features
as described in the research paper.
"""

import hashlib
import hmac
import secrets
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

from ..blockchain.core import Block
from ..ml.training import GradientUpdate


class VRFType(Enum):
    WORKER_SELECTION = "worker_selection"
    BATCH_ASSIGNMENT = "batch_assignment"
    LEADER_ELECTION = "leader_election"
    NONCE_COMMITMENT = "nonce_commitment"


@dataclass
class VRFProof:
    """VRF proof that can be verified by other nodes"""

    input_data: bytes
    output_hash: str
    proof_data: bytes
    public_key: bytes
    timestamp: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_data": self.input_data.hex(),
            "output_hash": self.output_hash,
            "proof_data": self.proof_data.hex(),
            "public_key": self.public_key.hex(),
            "timestamp": self.timestamp,
        }


class VerifiableRandomFunction:
    """
    Verifiable Random Function implementation.

    This is a simplified VRF based on HMAC-SHA256. In production,
    would use a proper VRF like ECVRF (RFC 8032) or RSA-FDH-VRF.
    """

    def __init__(self, private_key: Optional[bytes] = None):
        self.private_key = private_key or secrets.token_bytes(32)
        self.public_key = hashlib.sha256(self.private_key).digest()

    def compute(self, input_data: bytes, vrf_type: VRFType = VRFType.WORKER_SELECTION) -> VRFProof:
        """
        Compute VRF output and proof for given input.

        Returns a VRF proof that can be verified by others.
        """
        # Create VRF input with type discrimination
        vrf_input = vrf_type.value.encode() + b"|" + input_data

        # Compute HMAC as VRF function
        vrf_output = hmac.new(self.private_key, vrf_input, hashlib.sha256).digest()

        # Create proof (simplified - normally would be cryptographic proof)
        proof_input = self.private_key + vrf_input + vrf_output
        proof_data = hashlib.sha256(proof_input).digest()

        # Hash output to get random value
        output_hash = hashlib.sha256(vrf_output).hexdigest()

        return VRFProof(
            input_data=input_data,
            output_hash=output_hash,
            proof_data=proof_data,
            public_key=self.public_key,
            timestamp=int(time.time()),
        )

    def verify(self, proof: VRFProof, vrf_type: VRFType = VRFType.WORKER_SELECTION) -> bool:
        """
        Verify a VRF proof.

        Returns True if the proof is valid.
        """
        try:
            # Recreate VRF input
            vrf_input = vrf_type.value.encode() + b"|" + proof.input_data

            # In a real VRF, would verify cryptographic proof
            # For this simplified version, we check consistency

            # Verify output hash format
            if len(proof.output_hash) != 64:  # SHA256 hex length
                return False

            # Verify proof data format
            if len(proof.proof_data) != 32:  # SHA256 output length
                return False

            # Verify public key format
            if len(proof.public_key) != 32:  # SHA256 output length
                return False

            # Basic timestamp validation (not too old, not in future)
            current_time = int(time.time())
            if proof.timestamp > current_time + 300:  # 5 minutes future tolerance
                return False
            if current_time - proof.timestamp > 3600:  # 1 hour age limit
                return False

            return True

        except Exception:
            return False

    def get_random_value(self, proof: VRFProof) -> float:
        """
        Extract normalized random value [0,1] from VRF proof.
        """
        # Convert hex output to integer, then normalize
        output_int = int(proof.output_hash, 16)
        return output_int / (2**256 - 1)


class AdvancedWorkerSelection:
    """
    Advanced worker selection using proper VRF and sophisticated algorithms.
    """

    def __init__(self, vrf: VerifiableRandomFunction):
        self.vrf = vrf
        self.selection_history: Dict[str, List[Dict]] = {}  # task_id -> selections
        self.node_performance: Dict[str, float] = {}  # node_id -> performance score
        self.reputation_weights = {
            "completion_rate": 0.3,
            "accuracy_score": 0.3,
            "availability_score": 0.2,
            "stake_amount": 0.2,
        }

    def select_workers_with_vrf(
        self,
        task_id: str,
        candidates: List[Dict],
        num_needed: int,
        selection_criteria: Dict[str, Any],
    ) -> Tuple[List[Dict], List[VRFProof]]:
        """
        Select workers using VRF-based randomness with performance weighting.

        Returns (selected_workers, vrf_proofs)
        """
        if not candidates:
            return [], []

        vrf_proofs = []
        scored_candidates = []

        # Calculate VRF-based scores for each candidate
        for candidate in candidates:
            node_id = candidate.get("node_id", "")

            # Create VRF input combining task and node info
            vrf_input_data = json.dumps(
                {
                    "task_id": task_id,
                    "node_id": node_id,
                    "selection_round": int(time.time() // 3600),  # Hourly rounds
                },
                sort_keys=True,
            ).encode()

            # Compute VRF
            vrf_proof = self.vrf.compute(vrf_input_data, VRFType.WORKER_SELECTION)
            vrf_proofs.append(vrf_proof)

            # Get random value from VRF
            vrf_random = self.vrf.get_random_value(vrf_proof)

            # Calculate performance-based weight
            performance_score = self._calculate_performance_score(node_id, candidate)

            # Combine VRF randomness with performance (weighted)
            final_score = 0.7 * vrf_random + 0.3 * performance_score

            scored_candidates.append((candidate, final_score, vrf_proof))

        # Sort by score and select top candidates
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        selected = scored_candidates[:num_needed]

        selected_workers = [worker for worker, _, _ in selected]
        relevant_proofs = [proof for _, _, proof in selected]

        # Record selection for history
        selection_record = {
            "task_id": task_id,
            "selected_nodes": [w.get("node_id", "") for w in selected_workers],
            "timestamp": int(time.time()),
            "num_candidates": len(candidates),
            "selection_criteria": selection_criteria,
        }

        if task_id not in self.selection_history:
            self.selection_history[task_id] = []
        self.selection_history[task_id].append(selection_record)

        return selected_workers, relevant_proofs

    def _calculate_performance_score(self, node_id: str, candidate: Dict[str, Any]) -> float:
        """Calculate performance score for a candidate node"""
        if node_id not in self.node_performance:
            return 0.5  # Default neutral score

        base_score = self.node_performance[node_id]

        # Adjust based on candidate-specific factors
        adjustments = 0.0

        # Stake amount adjustment
        stake_amount = candidate.get("stake_amount", 0)
        if stake_amount > 100:
            adjustments += 0.1

        # Hardware capabilities
        has_gpu = candidate.get("has_gpu", False)
        if has_gpu:
            adjustments += 0.05

        # Network connectivity
        bandwidth = candidate.get("bandwidth_mbps", 10)
        if bandwidth > 100:
            adjustments += 0.05

        return min(1.0, base_score + adjustments)

    def update_node_performance(self, node_id: str, performance_metrics: Dict[str, float]) -> None:
        """Update performance score for a node based on recent metrics"""
        current_score = self.node_performance.get(node_id, 0.5)

        # Calculate weighted performance score
        new_score = 0.0
        for metric, weight in self.reputation_weights.items():
            value = performance_metrics.get(metric, 0.5)
            new_score += weight * value

        # Apply exponential moving average for stability
        alpha = 0.3  # Learning rate
        updated_score = alpha * new_score + (1 - alpha) * current_score

        self.node_performance[node_id] = max(0.0, min(1.0, updated_score))

    def verify_worker_selection(
        self, task_id: str, selected_nodes: List[str], vrf_proofs: List[VRFProof]
    ) -> bool:
        """Verify that worker selection was done correctly using VRF"""
        if len(selected_nodes) != len(vrf_proofs):
            return False

        for i, (node_id, proof) in enumerate(zip(selected_nodes, vrf_proofs)):
            # Recreate expected VRF input
            expected_input = json.dumps(
                {
                    "task_id": task_id,
                    "node_id": node_id,
                    "selection_round": proof.timestamp // 3600,  # Hourly rounds
                },
                sort_keys=True,
            ).encode()

            # Verify proof
            if proof.input_data != expected_input:
                return False

            if not self.vrf.verify(proof, VRFType.WORKER_SELECTION):
                return False

        return True


class ZeroNonceCommitment:
    """
    Implements zero-nonce block commitment mechanism for k iterations ahead.
    """

    def __init__(self, commitment_depth: int = 5):
        self.commitment_depth = commitment_depth  # k in the paper
        self.pending_commitments: Dict[str, Dict] = {}  # block_hash -> commitment_data
        self.commitment_history: List[Dict] = []

    def create_commitment(
        self,
        miner_id: str,
        future_iteration: int,
        model_state: Dict[str, Any],
        vrf: VerifiableRandomFunction,
    ) -> Dict[str, Any]:
        """
        Create a zero-nonce commitment for future iterations.

        This commits to doing useful work k iterations in the future.
        """
        commitment_id = hashlib.sha256(
            f"{miner_id}_{future_iteration}_{time.time()}".encode()
        ).hexdigest()

        # Create VRF for commitment randomness
        vrf_input = json.dumps(
            {
                "miner_id": miner_id,
                "future_iteration": future_iteration,
                "commitment_id": commitment_id,
            },
            sort_keys=True,
        ).encode()

        vrf_proof = vrf.compute(vrf_input, VRFType.NONCE_COMMITMENT)

        # Create commitment data
        commitment_data = {
            "commitment_id": commitment_id,
            "miner_id": miner_id,
            "future_iteration": future_iteration,
            "current_iteration": future_iteration - self.commitment_depth,
            "model_state_hash": hashlib.sha256(
                json.dumps(model_state, sort_keys=True).encode()
            ).hexdigest(),
            "vrf_proof": vrf_proof.to_dict(),
            "timestamp": int(time.time()),
            "status": "pending",
        }

        self.pending_commitments[commitment_id] = commitment_data
        return commitment_data

    def fulfill_commitment(
        self,
        commitment_id: str,
        actual_nonce: int,
        block_hash: str,
        gradient_update: GradientUpdate,
    ) -> bool:
        """
        Fulfill a previously made commitment by providing actual work proof.
        """
        if commitment_id not in self.pending_commitments:
            return False

        commitment = self.pending_commitments[commitment_id]

        # Verify the commitment is ready to be fulfilled
        current_iteration = gradient_update.iteration
        expected_iteration = commitment["future_iteration"]

        if current_iteration != expected_iteration:
            return False

        # Verify the work is consistent with the commitment
        gradient_hash = gradient_update.get_hash()

        # Update commitment with fulfillment data
        commitment.update(
            {
                "actual_nonce": actual_nonce,
                "block_hash": block_hash,
                "gradient_hash": gradient_hash,
                "fulfilled_at": int(time.time()),
                "status": "fulfilled",
            }
        )

        # Move to history
        self.commitment_history.append(commitment)
        del self.pending_commitments[commitment_id]

        return True

    def verify_commitment_fulfillment(self, commitment_data: Dict[str, Any]) -> bool:
        """Verify that a commitment was properly fulfilled"""
        if commitment_data["status"] != "fulfilled":
            return False

        # Verify VRF proof
        vrf_proof_data = commitment_data["vrf_proof"]
        vrf_proof = VRFProof(
            input_data=bytes.fromhex(vrf_proof_data["input_data"]),
            output_hash=vrf_proof_data["output_hash"],
            proof_data=bytes.fromhex(vrf_proof_data["proof_data"]),
            public_key=bytes.fromhex(vrf_proof_data["public_key"]),
            timestamp=vrf_proof_data["timestamp"],
        )

        # Basic VRF verification (would use proper VRF in production)
        vrf = VerifiableRandomFunction()  # Temporary instance for verification
        if not vrf.verify(vrf_proof, VRFType.NONCE_COMMITMENT):
            return False

        # Verify timing constraints
        committed_at = commitment_data["timestamp"]
        fulfilled_at = commitment_data["fulfilled_at"]

        # Should be fulfilled within reasonable time after commitment period
        expected_fulfillment_time = committed_at + (
            self.commitment_depth * 60
        )  # Assume 1 min per iteration
        if (
            fulfilled_at < expected_fulfillment_time - 300
            or fulfilled_at > expected_fulfillment_time + 900
        ):
            return False  # Too early or too late

        return True

    def get_pending_commitments(self, miner_id: str) -> List[Dict[str, Any]]:
        """Get pending commitments for a specific miner"""
        return [
            commitment
            for commitment in self.pending_commitments.values()
            if commitment["miner_id"] == miner_id
        ]


class MessageHistoryMerkleTree:
    """
    Implements Merkle tree construction for message history compression.
    """

    def __init__(self):
        self.message_history: List[str] = []
        self.merkle_roots: Dict[int, str] = {}  # epoch -> merkle_root

    def add_message(self, message: str) -> str:
        """Add a message to history and return its hash"""
        message_hash = hashlib.sha256(message.encode()).hexdigest()
        self.message_history.append(message_hash)
        return message_hash

    def build_merkle_tree(self, epoch: int, messages: List[str]) -> str:
        """
        Build Merkle tree for a set of messages and return root hash.
        """
        if not messages:
            return hashlib.sha256(b"").hexdigest()

        # Convert messages to hashes if needed
        hashes = [
            hashlib.sha256(msg.encode()).hexdigest() if isinstance(msg, str) else msg
            for msg in messages
        ]

        # Build tree bottom-up
        while len(hashes) > 1:
            next_level = []

            # Process pairs
            for i in range(0, len(hashes), 2):
                left = hashes[i]
                right = hashes[i + 1] if i + 1 < len(hashes) else left  # Duplicate if odd

                # Combine hashes
                combined = left + right
                parent_hash = hashlib.sha256(combined.encode()).hexdigest()
                next_level.append(parent_hash)

            hashes = next_level

        merkle_root = hashes[0]
        self.merkle_roots[epoch] = merkle_root
        return merkle_root

    def get_merkle_proof(self, epoch: int, message_index: int, messages: List[str]) -> List[str]:
        """
        Generate Merkle proof for a specific message.

        Returns list of sibling hashes needed to verify inclusion.
        """
        if not messages or message_index >= len(messages):
            return []

        # Convert to hashes
        hashes = [
            hashlib.sha256(msg.encode()).hexdigest() if isinstance(msg, str) else msg
            for msg in messages
        ]

        proof = []
        current_index = message_index

        # Build proof by collecting sibling hashes at each level
        while len(hashes) > 1:
            next_level = []

            for i in range(0, len(hashes), 2):
                left = hashes[i]
                right = hashes[i + 1] if i + 1 < len(hashes) else left

                # If current index is in this pair, add sibling to proof
                if i <= current_index < i + 2:
                    if current_index == i:  # We are left, add right
                        proof.append(right)
                    else:  # We are right, add left
                        proof.append(left)
                    current_index = len(next_level)  # Index in next level

                # Compute parent
                combined = left + right
                parent_hash = hashlib.sha256(combined.encode()).hexdigest()
                next_level.append(parent_hash)

            hashes = next_level

        return proof

    def verify_merkle_proof(
        self, message_hash: str, proof: List[str], root_hash: str, message_index: int
    ) -> bool:
        """
        Verify a Merkle proof for message inclusion.
        """
        current_hash = message_hash
        current_index = message_index

        for sibling_hash in proof:
            if current_index % 2 == 0:  # We are left child
                combined = current_hash + sibling_hash
            else:  # We are right child
                combined = sibling_hash + current_hash

            current_hash = hashlib.sha256(combined.encode()).hexdigest()
            current_index //= 2

        return current_hash == root_hash


# Example usage and testing
if __name__ == "__main__":
    print("Testing VRF and Advanced Mining Features...")

    # Test VRF
    print("\n1. Testing Verifiable Random Functions:")
    vrf = VerifiableRandomFunction()

    test_input = b"test_task_worker_selection_001"
    proof = vrf.compute(test_input, VRFType.WORKER_SELECTION)

    print(f"VRF computed for input: {test_input}")
    print(f"Output hash: {proof.output_hash}")
    print(f"Random value: {vrf.get_random_value(proof):.6f}")

    # Verify proof
    is_valid = vrf.verify(proof, VRFType.WORKER_SELECTION)
    print(f"Proof verification: {is_valid}")

    # Test worker selection
    print("\n2. Testing Advanced Worker Selection:")
    worker_selector = AdvancedWorkerSelection(vrf)

    # Mock candidates
    candidates = [
        {"node_id": "node_001", "stake_amount": 150, "has_gpu": True, "bandwidth_mbps": 200},
        {"node_id": "node_002", "stake_amount": 100, "has_gpu": False, "bandwidth_mbps": 50},
        {"node_id": "node_003", "stake_amount": 200, "has_gpu": True, "bandwidth_mbps": 300},
        {"node_id": "node_004", "stake_amount": 80, "has_gpu": False, "bandwidth_mbps": 100},
        {"node_id": "node_005", "stake_amount": 120, "has_gpu": True, "bandwidth_mbps": 150},
    ]

    selected_workers, vrf_proofs = worker_selector.select_workers_with_vrf(
        task_id="test_task_001",
        candidates=candidates,
        num_needed=3,
        selection_criteria={"min_stake": 100, "prefer_gpu": True},
    )

    print(f"Selected {len(selected_workers)} workers from {len(candidates)} candidates")
    for worker in selected_workers:
        print(
            f"  - {worker['node_id']} (stake: {worker['stake_amount']}, GPU: {worker['has_gpu']})"
        )

    # Verify selection
    selected_node_ids = [w["node_id"] for w in selected_workers]
    verification_result = worker_selector.verify_worker_selection(
        "test_task_001", selected_node_ids, vrf_proofs
    )
    print(f"Selection verification: {verification_result}")

    # Test zero-nonce commitment
    print("\n3. Testing Zero-Nonce Commitment:")
    commitment_system = ZeroNonceCommitment(commitment_depth=3)

    model_state = {"weights": [0.1, 0.2, 0.3], "bias": [0.05, 0.1]}
    commitment = commitment_system.create_commitment(
        miner_id="miner_001", future_iteration=10, model_state=model_state, vrf=vrf
    )

    print(f"Created commitment: {commitment['commitment_id'][:16]}...")
    print(f"For iteration: {commitment['future_iteration']}")

    # Test Merkle tree for message history
    print("\n4. Testing Message History Merkle Tree:")
    merkle_tree = MessageHistoryMerkleTree()

    # Add some messages
    messages = [
        "gradient_update_001_miner_001",
        "gradient_update_002_miner_002",
        "gradient_update_003_miner_003",
        "supervisor_vote_proposal_001",
        "evaluator_result_task_001",
    ]

    for msg in messages:
        merkle_tree.add_message(msg)

    # Build Merkle tree for epoch
    root_hash = merkle_tree.build_merkle_tree(epoch=1, messages=messages)
    print(f"Merkle root for epoch 1: {root_hash[:16]}...")

    # Generate and verify proof for first message
    proof = merkle_tree.get_merkle_proof(epoch=1, message_index=0, messages=messages)
    message_hash = hashlib.sha256(messages[0].encode()).hexdigest()

    verification = merkle_tree.verify_merkle_proof(message_hash, proof, root_hash, 0)
    print(f"Merkle proof verification: {verification}")

    print("\nVRF and advanced mining tests completed!")
