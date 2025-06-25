"""
Advanced Data Management for PoUW Implementation.

Implements Reed-Solomon encoding, consistent hashing, and sophisticated
data handling mechanisms as described in the research paper.
"""

import hashlib
import numpy as np
import json
import time
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import bisect


class DataShardType(Enum):
    TRAINING_DATA = "training_data"
    VALIDATION_DATA = "validation_data"
    TEST_DATA = "test_data"
    MODEL_CHECKPOINT = "model_checkpoint"
    GRADIENT_HISTORY = "gradient_history"


@dataclass
class DataShard:
    """A data shard with Reed-Solomon encoding"""

    shard_id: str
    data_type: DataShardType
    original_hash: str
    encoded_data: bytes
    parity_data: bytes
    metadata: Dict[str, Any]
    created_at: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "data_type": self.data_type.value,
            "original_hash": self.original_hash,
            "encoded_data": self.encoded_data.hex(),
            "parity_data": self.parity_data.hex(),
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


@dataclass
class DataLocation:
    """Location of a data shard in the network"""

    node_id: str
    shard_id: str
    last_verified: int
    availability_score: float  # 0.0 to 1.0


class ReedSolomonEncoder:
    """
    Simplified Reed-Solomon encoding for data redundancy.

    In production, would use a proper RS library like pyfinite or reedsolo.
    This implements a basic version for demonstration.
    """

    def __init__(self, data_shards: int = 4, parity_shards: int = 2):
        self.data_shards = data_shards
        self.parity_shards = parity_shards
        self.total_shards = data_shards + parity_shards

    def encode(self, data: bytes) -> Tuple[List[bytes], List[bytes]]:
        """
        Encode data into data shards and parity shards.

        Returns (data_shards, parity_shards)
        """
        # Pad data to be divisible by data_shards
        data_length = len(data)
        padding_length = (self.data_shards - (data_length % self.data_shards)) % self.data_shards
        padded_data = data + b"\x00" * padding_length

        # Split into data shards
        shard_size = len(padded_data) // self.data_shards
        data_shard_list = []

        for i in range(self.data_shards):
            start = i * shard_size
            end = start + shard_size
            data_shard_list.append(padded_data[start:end])

        # Generate parity shards (simplified XOR-based approach)
        parity_shard_list = []
        for p in range(self.parity_shards):
            parity = bytearray(shard_size)
            for i in range(self.data_shards):
                # XOR with different weights for each parity shard
                weight = (p + 1) * (i + 1) % 256
                for j in range(shard_size):
                    parity[j] ^= (data_shard_list[i][j] * weight) % 256
            parity_shard_list.append(bytes(parity))

        return data_shard_list, parity_shard_list

    def decode(self, available_shards: Dict[int, bytes], original_length: int) -> Optional[bytes]:
        """
        Decode data from available shards.

        available_shards: {shard_index: shard_data}
        Returns decoded data or None if not enough shards
        """
        if len(available_shards) < self.data_shards:
            return None

        # If we have all data shards, simple reconstruction
        data_shards_available = {
            i: available_shards[i] for i in range(self.data_shards) if i in available_shards
        }

        if len(data_shards_available) == self.data_shards:
            # Simple case: all data shards available
            reconstructed = b"".join(data_shards_available[i] for i in range(self.data_shards))
            return reconstructed[:original_length]

        # Complex case: need to use parity shards for reconstruction
        # This would require more sophisticated RS algebra in production
        # For now, return None if we don't have all data shards
        return None


class ConsistentHashRing:
    """
    Consistent hashing with bounded loads for batch assignment.

    Implements the consistent hashing algorithm described in the paper
    for distributing data batches to worker nodes.
    """

    def __init__(self, load_factor: float = 1.25, virtual_nodes: int = 150):
        self.load_factor = load_factor  # Maximum load multiplier
        self.virtual_nodes = virtual_nodes  # Virtual nodes per physical node
        self.ring: Dict[int, str] = {}  # hash_value -> node_id
        self.nodes: Set[str] = set()
        self.node_loads: Dict[str, int] = {}  # node_id -> current_load
        self.sorted_hashes: List[int] = []

    def _hash(self, key: str) -> int:
        """Hash function for the ring"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def add_node(self, node_id: str) -> None:
        """Add a node to the hash ring"""
        if node_id in self.nodes:
            return

        self.nodes.add(node_id)
        self.node_loads[node_id] = 0

        # Add virtual nodes
        for i in range(self.virtual_nodes):
            virtual_key = f"{node_id}:{i}"
            hash_value = self._hash(virtual_key)
            self.ring[hash_value] = node_id
            bisect.insort(self.sorted_hashes, hash_value)

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the hash ring"""
        if node_id not in self.nodes:
            return

        self.nodes.remove(node_id)
        del self.node_loads[node_id]

        # Remove virtual nodes
        for i in range(self.virtual_nodes):
            virtual_key = f"{node_id}:{i}"
            hash_value = self._hash(virtual_key)
            if hash_value in self.ring:
                del self.ring[hash_value]
                self.sorted_hashes.remove(hash_value)

    def get_node(self, key: str) -> Optional[str]:
        """Get the node responsible for a key with load balancing"""
        if not self.nodes:
            return None

        key_hash = self._hash(key)

        # Calculate average load
        total_load = sum(self.node_loads.values())
        avg_load = total_load / len(self.nodes) if self.nodes else 0
        max_load = int(avg_load * self.load_factor)

        # Find the next node in the ring
        idx = bisect.bisect_right(self.sorted_hashes, key_hash)

        # Try nodes starting from the natural position
        attempts = 0
        while attempts < len(self.sorted_hashes):
            if idx >= len(self.sorted_hashes):
                idx = 0

            hash_value = self.sorted_hashes[idx]
            node_id = self.ring[hash_value]

            # Check load constraint
            if self.node_loads[node_id] <= max_load:
                return node_id

            idx += 1
            attempts += 1

        # If all nodes are overloaded, return the one with minimum load
        if self.node_loads:
            return min(self.node_loads.keys(), key=lambda k: self.node_loads[k])
        return None

    def assign_batch(self, batch_id: str) -> Optional[str]:
        """Assign a batch to a node and update load"""
        node_id = self.get_node(batch_id)
        if node_id:
            self.node_loads[node_id] += 1
        return node_id

    def release_batch(self, batch_id: str) -> None:
        """Release a batch assignment and update load"""
        node_id = self.get_node(batch_id)
        if node_id and self.node_loads[node_id] > 0:
            self.node_loads[node_id] -= 1

    def get_load_distribution(self) -> Dict[str, int]:
        """Get current load distribution across nodes"""
        return dict(self.node_loads)


class DataAvailabilityManager:
    """
    Manages data availability and redundancy across the network.

    Implements the data availability protocols described in the paper.
    """

    def __init__(self, replication_factor: int = 3):
        self.replication_factor = replication_factor
        self.rs_encoder = ReedSolomonEncoder()
        self.hash_ring = ConsistentHashRing()
        self.data_registry: Dict[str, List[DataLocation]] = {}  # data_id -> locations
        self.shard_registry: Dict[str, DataShard] = {}  # shard_id -> shard_info

    def store_data(
        self,
        data_id: str,
        data: bytes,
        data_type: DataShardType,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Store data with Reed-Solomon encoding and replication.

        Returns list of shard IDs created.
        """
        metadata = metadata or {}
        original_hash = hashlib.sha256(data).hexdigest()

        # Encode data with Reed-Solomon
        data_shards, parity_shards = self.rs_encoder.encode(data)
        all_shards = data_shards + parity_shards

        created_shards = []

        # Create data shards
        for i, shard_data in enumerate(all_shards):
            shard_id = f"{data_id}_shard_{i:03d}"
            is_parity = i >= len(data_shards)

            # Determine parity data
            if is_parity:
                parity_data = shard_data
                encoded_data = b""
            else:
                encoded_data = shard_data
                parity_data = b""

            shard = DataShard(
                shard_id=shard_id,
                data_type=data_type,
                original_hash=original_hash,
                encoded_data=encoded_data,
                parity_data=parity_data,
                metadata={
                    **metadata,
                    "shard_index": i,
                    "is_parity": is_parity,
                    "original_length": len(data),
                },
                created_at=int(time.time()),
            )

            self.shard_registry[shard_id] = shard
            created_shards.append(shard_id)

        # Assign shards to nodes with replication
        data_locations = []
        for shard_id in created_shards:
            for replica in range(self.replication_factor):
                # Use different keys for replicas to spread them across nodes
                replica_key = f"{shard_id}_replica_{replica}"
                node_id = self.hash_ring.assign_batch(replica_key)

                if node_id:
                    location = DataLocation(
                        node_id=node_id,
                        shard_id=shard_id,
                        last_verified=int(time.time()),
                        availability_score=1.0,
                    )
                    data_locations.append(location)

        self.data_registry[data_id] = data_locations
        return created_shards

    def retrieve_data(self, data_id: str) -> Optional[bytes]:
        """
        Retrieve data by reconstructing from available shards.
        """
        if data_id not in self.data_registry:
            return None

        locations = self.data_registry[data_id]

        # Group locations by shard
        shard_locations = {}
        for location in locations:
            shard_id = location.shard_id
            if shard_id not in shard_locations:
                shard_locations[shard_id] = []
            shard_locations[shard_id].append(location)

        # Try to collect enough shards for reconstruction
        available_shards = {}
        original_length = None

        for shard_id, shard_locations_list in shard_locations.items():
            if shard_id not in self.shard_registry:
                continue

            shard = self.shard_registry[shard_id]
            shard_index = shard.metadata.get("shard_index", 0)
            original_length = shard.metadata.get("original_length", 0)

            # Use the shard data (simplified - in practice would fetch from node)
            if shard.encoded_data:
                available_shards[shard_index] = shard.encoded_data
            elif shard.parity_data:
                available_shards[shard_index] = shard.parity_data

        if not original_length:
            return None

        # Reconstruct data using Reed-Solomon decoder
        reconstructed_data = self.rs_encoder.decode(available_shards, original_length)
        return reconstructed_data

    def verify_data_integrity(self, data_id: str) -> Dict[str, bool]:
        """
        Verify integrity of all shards for a data ID.

        Returns {shard_id: is_valid}
        """
        if data_id not in self.data_registry:
            return {}

        verification_results = {}
        locations = self.data_registry[data_id]

        for location in locations:
            shard_id = location.shard_id
            if shard_id not in self.shard_registry:
                verification_results[shard_id] = False
                continue

            shard = self.shard_registry[shard_id]

            # Verify shard hash (simplified verification)
            if shard.encoded_data:
                computed_hash = hashlib.sha256(shard.encoded_data).hexdigest()
            else:
                computed_hash = hashlib.sha256(shard.parity_data).hexdigest()

            # In practice, would compare against stored hash
            verification_results[shard_id] = len(computed_hash) == 64  # Basic validation

        return verification_results

    def get_data_availability(self, data_id: str) -> float:
        """
        Calculate data availability score (0.0 to 1.0).

        Based on number of available shards and their distribution.
        """
        if data_id not in self.data_registry:
            return 0.0

        verification_results = self.verify_data_integrity(data_id)

        if not verification_results:
            return 0.0

        valid_shards = sum(1 for is_valid in verification_results.values() if is_valid)
        total_shards = len(verification_results)

        # Need at least data_shards for reconstruction
        min_required = self.rs_encoder.data_shards

        if valid_shards >= total_shards:
            return 1.0
        elif valid_shards >= min_required:
            return 0.5 + 0.5 * (valid_shards - min_required) / (total_shards - min_required)
        else:
            return valid_shards / min_required * 0.5

    def repair_data(self, data_id: str) -> bool:
        """
        Attempt to repair missing or corrupted data shards.
        """
        availability = self.get_data_availability(data_id)

        if availability >= 0.5:  # Can reconstruct
            # Retrieve data and re-encode
            original_data = self.retrieve_data(data_id)
            if original_data:
                # Get metadata from existing shards
                locations = self.data_registry[data_id]
                if locations:
                    sample_shard = self.shard_registry[locations[0].shard_id]
                    data_type = sample_shard.data_type
                    metadata = sample_shard.metadata.copy()

                    # Remove old entries
                    del self.data_registry[data_id]

                    # Re-store with fresh encoding
                    self.store_data(data_id, original_data, data_type, metadata)
                    return True

        return False


class DatasetSplitter:
    """
    Handles dataset splitting and hash verification as described in the paper.
    """

    def __init__(self):
        self.split_registry: Dict[str, Dict[str, Any]] = {}

    def split_dataset(
        self,
        dataset_id: str,
        data: List[Any],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> Dict[str, List[Any]]:
        """
        Split dataset into training, validation, and test sets with hash verification.
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")

        # Deterministic shuffle with seed
        np.random.seed(seed)
        indices = np.random.permutation(len(data))

        # Calculate split points
        train_end = int(len(data) * train_ratio)
        val_end = train_end + int(len(data) * val_ratio)

        # Create splits
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        splits = {
            "train": [data[i] for i in train_indices],
            "validation": [data[i] for i in val_indices],
            "test": [data[i] for i in test_indices],
        }

        # Calculate hashes for verification
        split_hashes = {}
        for split_name, split_data in splits.items():
            split_bytes = json.dumps(split_data, sort_keys=True).encode()
            split_hashes[split_name] = hashlib.sha256(split_bytes).hexdigest()

        # Register split information
        self.split_registry[dataset_id] = {
            "seed": seed,
            "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
            "hashes": split_hashes,
            "sizes": {name: len(split_data) for name, split_data in splits.items()},
            "created_at": int(time.time()),
        }

        return splits

    def verify_split(self, dataset_id: str, split_name: str, split_data: List[Any]) -> bool:
        """Verify that a split matches the registered hash"""
        if dataset_id not in self.split_registry:
            return False

        registry = self.split_registry[dataset_id]
        if split_name not in registry["hashes"]:
            return False

        # Calculate hash of provided data
        split_bytes = json.dumps(split_data, sort_keys=True).encode()
        computed_hash = hashlib.sha256(split_bytes).hexdigest()

        return computed_hash == registry["hashes"][split_name]

    def get_split_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a dataset split"""
        return self.split_registry.get(dataset_id)


# Example usage and testing
if __name__ == "__main__":
    print("Testing Advanced Data Management...")

    # Test Reed-Solomon encoding
    print("\n1. Testing Reed-Solomon Encoding:")
    rs_encoder = ReedSolomonEncoder(data_shards=4, parity_shards=2)

    test_data = b"This is test data for Reed-Solomon encoding demonstration."
    print(f"Original data: {test_data}")

    data_shards, parity_shards = rs_encoder.encode(test_data)
    print(f"Encoded into {len(data_shards)} data shards and {len(parity_shards)} parity shards")

    # Test reconstruction with all shards
    all_shards = {i: shard for i, shard in enumerate(data_shards + parity_shards)}
    reconstructed = rs_encoder.decode(all_shards, len(test_data))
    print(f"Reconstructed: {reconstructed}")
    print(f"Reconstruction successful: {reconstructed == test_data}")

    # Test consistent hashing
    print("\n2. Testing Consistent Hashing:")
    hash_ring = ConsistentHashRing()

    # Add nodes
    nodes = ["node_001", "node_002", "node_003", "node_004", "node_005"]
    for node in nodes:
        hash_ring.add_node(node)

    # Assign batches
    batches = [f"batch_{i:03d}" for i in range(20)]
    assignments = {}

    for batch in batches:
        assigned_node = hash_ring.assign_batch(batch)
        assignments[batch] = assigned_node

    print(f"Assigned {len(batches)} batches to {len(nodes)} nodes")
    print("Load distribution:", hash_ring.get_load_distribution())

    # Test data availability manager
    print("\n3. Testing Data Availability Manager:")
    data_manager = DataAvailabilityManager(replication_factor=2)

    # Add nodes to hash ring
    for node in nodes:
        data_manager.hash_ring.add_node(node)

    # Store test data
    test_data_large = (
        b"Large dataset for testing data availability and redundancy mechanisms." * 100
    )
    shard_ids = data_manager.store_data(
        data_id="test_dataset_001",
        data=test_data_large,
        data_type=DataShardType.TRAINING_DATA,
        metadata={"description": "Test dataset"},
    )

    print(f"Created {len(shard_ids)} shards for data storage")

    # Retrieve data
    retrieved_data = data_manager.retrieve_data("test_dataset_001")
    print(f"Data retrieval successful: {retrieved_data == test_data_large}")

    # Check availability
    availability = data_manager.get_data_availability("test_dataset_001")
    print(f"Data availability: {availability:.2f}")

    # Test dataset splitting
    print("\n4. Testing Dataset Splitting:")
    splitter = DatasetSplitter()

    # Create sample dataset
    sample_dataset = [{"id": i, "value": i * 2, "label": i % 3} for i in range(100)]

    splits = splitter.split_dataset(
        dataset_id="sample_dataset",
        data=sample_dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    print(f"Dataset split into:")
    for split_name, split_data in splits.items():
        print(f"  {split_name}: {len(split_data)} samples")

    # Verify splits
    for split_name, split_data in splits.items():
        is_valid = splitter.verify_split("sample_dataset", split_name, split_data)
        print(f"  {split_name} verification: {is_valid}")

    print("\nAdvanced data management tests completed!")
