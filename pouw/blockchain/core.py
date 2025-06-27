"""
Core blockchain data structures for PoUW.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import binascii
from ecdsa import VerifyingKey, SECP256k1, BadSignatureError
from pouw.blockchain.storage import (
    init_db, save_block, load_blocks, save_utxo, load_utxos, delete_utxo
)

MAX_MEMPOOL_SIZE = 10000
MIN_TX_FEE = 0.0001

@dataclass
class Transaction:
    """Base transaction class"""

    version: int
    inputs: List[Dict[str, Any]]
    outputs: List[Dict[str, Any]]
    op_return: Optional[bytes] = None  # 160 bytes for PoUW data
    timestamp: int = field(default_factory=lambda: int(time.time()))
    signature: Optional[bytes] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "type": self.__class__.__name__,
            "version": self.version,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "op_return": self.op_return.hex() if self.op_return else None,
            "timestamp": self.timestamp,
            "signature": self.signature.hex() if self.signature else None,
        }
        # Add subclass fields
        if hasattr(self, "task_definition"):
            d["task_definition"] = getattr(self, "task_definition")
        if hasattr(self, "fee"):
            d["fee"] = getattr(self, "fee")
        if hasattr(self, "role"):
            d["role"] = getattr(self, "role")
        if hasattr(self, "stake_amount"):
            d["stake_amount"] = getattr(self, "stake_amount")
        if hasattr(self, "preferences"):
            d["preferences"] = getattr(self, "preferences")
        return d

    def get_hash(self) -> str:
        """Calculate transaction hash"""
        tx_data = self.to_dict()
        tx_data.pop("signature", None)  # Exclude signature from hash
        tx_string = json.dumps(tx_data, sort_keys=True)
        return hashlib.sha256(tx_string.encode()).hexdigest()


@dataclass
class PayForTaskTransaction(Transaction):
    """Transaction to submit ML task and pay for training"""

    task_definition: Dict[str, Any] = field(default_factory=dict)
    fee: float = 0.0

    def __post_init__(self):
        # Encode task in OP_RETURN
        task_data = json.dumps(
            {"type": "PAY_FOR_TASK", "task": self.task_definition, "fee": self.fee}
        )
        self.op_return = task_data.encode()[:160]  # Limit to 160 bytes


@dataclass
class BuyTicketsTransaction(Transaction):
    """Transaction to stake and register as worker node"""

    role: str = ""  # 'miner', 'supervisor', 'evaluator', 'verifier'
    stake_amount: float = 0.0
    preferences: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        stake_data = json.dumps(
            {
                "type": "BUY_TICKETS",
                "role": self.role,
                "stake": self.stake_amount,
                "preferences": self.preferences,
            }
        )
        self.op_return = stake_data.encode()[:160]


@dataclass
class PoUWBlockHeader:
    """PoUW-specific block header fields"""

    version: int
    previous_hash: str
    merkle_root: str
    timestamp: int
    target: int
    nonce: int

    # PoUW-specific fields
    ml_task_id: Optional[str] = None
    message_history_hash: str = ""
    iteration_message_hash: str = ""
    zero_nonce_block_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "previous_hash": self.previous_hash,
            "merkle_root": self.merkle_root,
            "timestamp": self.timestamp,
            "target": self.target,
            "nonce": self.nonce,
            "ml_task_id": self.ml_task_id,
            "message_history_hash": self.message_history_hash,
            "iteration_message_hash": self.iteration_message_hash,
            "zero_nonce_block_hash": self.zero_nonce_block_hash,
        }

    def get_hash(self) -> str:
        """Calculate block header hash"""
        header_string = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(header_string.encode()).hexdigest()


@dataclass
class Block:
    """PoUW Block structure"""

    header: PoUWBlockHeader
    transactions: List[Transaction] = field(default_factory=list)

    # PoUW-specific mining proof data
    mining_proof: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        # Calculate merkle root from transactions
        if self.transactions:
            tx_hashes = [tx.get_hash() for tx in self.transactions]
            self.header.merkle_root = self._calculate_merkle_root(tx_hashes)
        else:
            self.header.merkle_root = (
                "0" * 64
            )

    def _calculate_merkle_root(self, tx_hashes: List[str]) -> str:
        """Calculate merkle root of transaction hashes"""
        if not tx_hashes:
            return "0" * 64

        if len(tx_hashes) == 1:
            return tx_hashes[0]

        # Build merkle tree
        level = tx_hashes[:]
        while len(level) > 1:
            next_level = []
            for i in range(0, len(level), 2):
                left = level[i]
                right = level[i + 1] if i + 1 < len(level) else level[i]
                combined = left + right
                next_level.append(
                    hashlib.sha256(
                        combined.encode()
                    ).hexdigest()
                )
            level = next_level

        return level[0]

    def get_hash(self) -> str:
        """Get block hash"""
        return self.header.get_hash()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "header": self.header.to_dict(),
            "transactions": [tx.to_dict() for tx in self.transactions],
            "mining_proof": self.mining_proof,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Block":
        header = PoUWBlockHeader(**data["header"])
        txs = []
        for tx in data["transactions"]:
            tx_type = tx.get("type", "Transaction")
            # Remove the 'type' field before creating transaction objects
            tx_data = {k: v for k, v in tx.items() if k != "type"}
            
            if tx_type == "PayForTaskTransaction":
                txs.append(PayForTaskTransaction(**tx_data))
            elif tx_type == "BuyTicketsTransaction":
                txs.append(BuyTicketsTransaction(**tx_data))
            else:
                txs.append(Transaction(**tx_data))
        mining_proof = data.get("mining_proof")
        return Block(header=header, transactions=txs, mining_proof=mining_proof)


@dataclass
class MLTask:
    """Machine Learning task definition"""

    task_id: str
    model_type: str  # 'mlp', 'cnn', etc.
    architecture: Dict[str, Any]
    optimizer: Dict[str, Any]
    stopping_criterion: Dict[str, Any]
    validation_strategy: Dict[str, Any]
    metrics: List[str]
    dataset_info: Dict[str, Any]
    performance_requirements: Dict[str, Any]
    fee: float
    client_id: str
    created_at: int = field(default_factory=lambda: int(time.time()))

    @property
    def complexity_score(self) -> float:
        """Calculate task complexity score based on task parameters"""
        score = 0.5  # Base complexity

        # Architecture complexity
        arch = self.architecture
        if "hidden_sizes" in arch:
            # More layers = higher complexity
            num_layers = len(arch["hidden_sizes"])
            score += min(0.3, num_layers * 0.05)

        if "input_size" in arch and "output_size" in arch:
            # Larger networks = higher complexity
            size_factor = (
                arch["input_size"] + arch["output_size"]
            ) / 1000
            score += min(0.2, size_factor * 0.1)

        # Dataset size complexity
        if "size" in self.dataset_info:
            size_factor = (
                self.dataset_info["size"] / 100000
            )  # Normalize to 100k samples
            score += min(0.2, size_factor * 0.1)

        # Performance requirements complexity
        if "min_accuracy" in self.performance_requirements:
            # Higher accuracy requirements = higher complexity
            acc_requirement = self.performance_requirements["min_accuracy"]
            if acc_requirement > 0.9:
                score += 0.2
            elif acc_requirement > 0.8:
                score += 0.1

        return min(1.0, score)  # Cap at 1.0

    def get_required_miners(self) -> int:
        """Calculate number of miners required based on task complexity"""
        # Base requirement of 1 miner
        required_miners = 1
        
        # Scale with complexity score
        complexity = self.complexity_score
        if complexity > 0.8:
            required_miners = 3
        elif complexity > 0.6:
            required_miners = 2
        
        # Consider dataset size
        if "size" in self.dataset_info:
            size = self.dataset_info["size"]
            if size > 100000:
                required_miners += 1
        
        # Consider performance requirements
        if (
            "gpu" in self.performance_requirements
            and self.performance_requirements["gpu"]
        ):
            required_miners = max(required_miners, 2)  # GPU tasks need at least 2 miners
            
        return min(
            required_miners, 5
        )  # Cap at 5 miners

    def to_dict(self) -> Dict[str, Any]:
        """Convert MLTask to dictionary representation"""
        return {
            "task_id": self.task_id,
            "model_type": self.model_type,
            "architecture": self.architecture,
            "optimizer": self.optimizer,
            "stopping_criterion": self.stopping_criterion,
            "validation_strategy": self.validation_strategy,
            "metrics": self.metrics,
            "dataset_info": self.dataset_info,
            "performance_requirements": self.performance_requirements,
            "fee": self.fee,
            "client_id": self.client_id,
            "created_at": self.created_at,
            "complexity_score": self.complexity_score,
        }


class Blockchain:
    """PoUW Blockchain implementation"""

    def __init__(self):
        init_db()
        self.chain: List[Block] = []
        self.mempool: List[Transaction] = []
        self.utxos: Dict[str, Dict[str, Any]] = {}
        self.active_tasks: Dict[str, MLTask] = {}
        
        # Difficulty adjustment parameters (Bitcoin-style)
        self.difficulty_target = (
            0x00000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        )  # Starting difficulty
        self.target_block_time = 60  # 1 minute target block time
        self.difficulty_adjustment_interval = 144  # Adjust every 144 blocks
        self.max_difficulty_change = 4  # Maximum 4x difficulty change
        # Only enforce maximum difficulty bound (easiest), no minimum (hardest)
        self.max_difficulty_target = (
            0x00FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        )  # Easiest allowed

        # Load chain and UTXOs from DB
        for block_data in load_blocks():
            block = Block.from_dict(block_data)
            self.chain.append(block)
        self.utxos = load_utxos()

        if not self.chain:
            self._create_genesis_block()
        else:
            # Recalculate difficulty based on loaded chain
            self._update_difficulty_if_needed()

    def _create_genesis_block(self):
        """Create the first block in the chain"""
        genesis_header = PoUWBlockHeader(
            version=1,
            previous_hash="0" * 64,
            merkle_root="0" * 64,
            timestamp=int(time.time()),
            target=self.difficulty_target,
            nonce=0,
        )

        genesis_block = Block(header=genesis_header)
        self.chain.append(genesis_block)

    def _update_difficulty_if_needed(self):
        """Check if difficulty adjustment is needed and apply if so"""
        chain_length = len(self.chain)
        
        # Only adjust after we have enough blocks
        if chain_length < self.difficulty_adjustment_interval:
            return
            
        # Check if we're at a difficulty adjustment point
        if chain_length % self.difficulty_adjustment_interval == 0:
            self._adjust_difficulty()

    def _adjust_difficulty(self):
        """Adjust mining difficulty based on block timing (Bitcoin-style)"""
        chain_length = len(self.chain)
        
        if chain_length < self.difficulty_adjustment_interval:
            return
            
        # Get the last adjustment block and current block
        current_block = self.chain[-1]
        adjustment_block = self.chain[-(self.difficulty_adjustment_interval)]
        
        # Calculate actual time taken for last interval
        actual_time = current_block.header.timestamp - adjustment_block.header.timestamp
        expected_time = self.difficulty_adjustment_interval * self.target_block_time
        
        # Calculate adjustment ratio
        time_ratio = actual_time / expected_time
        
        # Limit adjustment to prevent wild swings
        time_ratio = max(
            1.0 / self.max_difficulty_change,
            min(self.max_difficulty_change, time_ratio)
        )
        
        # Calculate new difficulty target
        # If blocks are too fast (time_ratio < 1), make difficulty harder
        # If blocks are too slow (time_ratio > 1), make difficulty easier
        new_target = int(self.difficulty_target * time_ratio)
        
        # Enforce only maximum difficulty bound (easiest), no minimum (hardest)
        new_target = min(self.max_difficulty_target, new_target)
        
        # Log the adjustment
        old_target = self.difficulty_target
        self.difficulty_target = new_target
        
        print(f"🎯 Difficulty Adjustment:")
        print(f"   Blocks: {self.difficulty_adjustment_interval}")
        print(f"   Expected time: {expected_time}s ({expected_time/60:.1f} min)")
        print(f"   Actual time: {actual_time}s ({actual_time/60:.1f} min)")
        print(f"   Time ratio: {time_ratio:.3f}")
        print(f"   Old target: 0x{old_target:064x}")
        print(f"   New target: 0x{new_target:064x}")
        print(f"   Difficulty {'increased' if new_target < old_target else 'decreased'}")

    def get_current_difficulty_info(self) -> Dict[str, Any]:
        """Get current difficulty and mining statistics"""
        chain_length = len(self.chain)
        
        # Calculate blocks until next adjustment
        blocks_until_adjustment = self.difficulty_adjustment_interval - (chain_length % self.difficulty_adjustment_interval)
        
        # Calculate recent average block time
        recent_blocks = min(10, chain_length - 1)
        if recent_blocks > 0:
            time_span = self.chain[-1].header.timestamp - self.chain[-(recent_blocks + 1)].header.timestamp
            avg_block_time = time_span / recent_blocks
        else:
            avg_block_time = self.target_block_time
            
        # Calculate difficulty as a ratio to maximum target
        difficulty_ratio = self.max_difficulty_target / self.difficulty_target
        
        return {
            "current_target": self.difficulty_target,
            "target_hex": f"0x{self.difficulty_target:064x}",
            "difficulty_ratio": difficulty_ratio,
            "target_block_time": self.target_block_time,
            "recent_avg_block_time": avg_block_time,
            "blocks_until_adjustment": blocks_until_adjustment,
            "adjustment_interval": self.difficulty_adjustment_interval,
            "chain_length": chain_length,
            "next_adjustment_at_block": chain_length + blocks_until_adjustment
        }

    def get_latest_block(self) -> Block:
        """Get the most recent block"""
        return self.chain[-1]

    def add_transaction_to_mempool(self, transaction: Transaction):
        """Add transaction to mempool for future inclusion in blocks"""
        # 1. Mempool size limit
        if len(self.mempool) >= MAX_MEMPOOL_SIZE:
            return False
        # 2. Minimum fee check (if present)
        tx_fee = getattr(transaction, "fee", None)
        if tx_fee is not None and tx_fee < MIN_TX_FEE:
            return False
        if self._validate_transaction(transaction):
            self.mempool.append(transaction)
            return True
        return False

    def _validate_transaction(self, transaction: Transaction) -> bool:
        """Validate transaction before adding to mempool"""
        # 1. Duplicate check (already in mempool)
        if transaction.get_hash() in [tx.get_hash() for tx in self.mempool]:
            return False  # Duplicate transaction

        # 2. Signature presence and (stub) verification
        if not transaction.signature:
            return False
        if not self._verify_signature(transaction):
            return False

        # 3. Input checks
        input_sum = 0.0
        seen_inputs = set()
        for inp in transaction.inputs:
            # Must reference a valid UTXO
            prev_hash = inp.get("previous_hash")
            index = inp.get("index")
            if prev_hash is None or index is None:
                return False
            utxo_key = f"{prev_hash}:{index}"
            if utxo_key not in self.utxos:
                return False  # Input not found (not unspent)
            # No double-spending within the transaction
            if utxo_key in seen_inputs:
                return False
            seen_inputs.add(utxo_key)
            # Sum input values
            input_sum += float(self.utxos[utxo_key].get("amount", 0))

        # 4. Output checks
        output_sum = 0.0
        for out in transaction.outputs:
            amount = out.get("amount")
            address = out.get("address")
            if amount is None or address is None:
                return False
            if not isinstance(amount, (int, float)) or amount < 0:
                return False
            output_sum += float(amount)

        # 5. No inflation (inputs >= outputs, allow coinbase exception)
        if transaction.inputs and input_sum < output_sum:
            return False

        return True

    def _validate_transaction_for_block(self, transaction: Transaction) -> bool:
        """Validate transaction for inclusion in a block"""
        # 1. Signature presence and (stub) verification
        if not transaction.signature:
            return False
        if not self._verify_signature(transaction):
            return False

        # 2. Input checks (must be unspent in current UTXO set)
        input_sum = 0.0
        seen_inputs = set()
        for inp in transaction.inputs:
            prev_hash = inp.get("previous_hash")
            index = inp.get("index")
            if prev_hash is None or index is None:
                return False
            utxo_key = f"{prev_hash}:{index}"
            if utxo_key not in self.utxos:
                return False
            if utxo_key in seen_inputs:
                return False
            seen_inputs.add(utxo_key)
            input_sum += float(self.utxos[utxo_key].get("amount", 0))

        # 3. Output checks
        output_sum = 0.0
        for out in transaction.outputs:
            amount = out.get("amount")
            address = out.get("address")
            if amount is None or address is None:
                return False
            if not isinstance(amount, (int, float)) or amount < 0:
                return False
            output_sum += float(amount)

        # 4. No inflation (inputs >= outputs, allow coinbase exception)
        if transaction.inputs and input_sum < output_sum:
            return False

        return True

    def _verify_signature(self, transaction: Transaction) -> bool:
        """Production signature verification using ECDSA (secp256k1)."""
        # The signature should be over the transaction hash (excluding the signature field)
        tx_data = transaction.to_dict()
        tx_data.pop("signature", None)
        tx_string = json.dumps(
            tx_data, sort_keys=True
        )
        tx_hash = hashlib.sha256(tx_string.encode()).digest()
        # The signature is expected to be in DER format
        signature = transaction.signature
        if not signature:
            return False
        # For each input, check the public key
        for inp in transaction.inputs:
            public_key_hex = inp.get("public_key")
            if not public_key_hex:
                return False
            try:
                public_key_bytes = binascii.unhexlify(public_key_hex)
                vk = VerifyingKey.from_string(public_key_bytes, curve=SECP256k1)
                # If signature is hex, decode it
                if isinstance(signature, str):
                    signature_bytes = binascii.unhexlify(signature)
                else:
                    signature_bytes = signature
                if not vk.verify(
                    signature_bytes, tx_hash
                ):
                    return False
            except (binascii.Error, BadSignatureError, Exception):
                return False
        return True

    def create_block(
        self,
        transactions: List[Transaction],
        miner_address: str,
        mining_proof: Optional[Dict[str, Any]] = None,
    ) -> Block:
        """Create a new block with given transactions"""
        previous_block = self.get_latest_block()

        # Create coinbase transaction for miner reward
        coinbase_tx = Transaction(
            version=1,
            inputs=[{"previous_hash": "0" * 64, "index": -1}],
            outputs=[
                {"address": miner_address, "amount": 12.5}
            ],  # Block reward
        )

        all_transactions = [
            coinbase_tx
        ] + transactions

        header = PoUWBlockHeader(
            version=1,
            previous_hash=previous_block.get_hash(),
            merkle_root="",  # Will be calculated in Block.__post_init__
            timestamp=int(time.time()),
            target=self.difficulty_target,
            nonce=0,  # Will be set during mining
        )

        block = Block(header=header, transactions=all_transactions, mining_proof=mining_proof)

        return block

    def add_block(self, block: Block) -> bool:
        """Add a validated block to the chain"""
        if self._validate_block(block):
            self.chain.append(block)
            save_block(block.get_hash(), block.to_dict())
            
            # Check for difficulty adjustment after adding block
            self._update_difficulty_if_needed()
            
            # Remove transactions from mempool
            block_tx_hashes = {
                tx.get_hash() for tx in block.transactions
            }
            self.mempool = [
                tx for tx in self.mempool if tx.get_hash() not in block_tx_hashes
            ]
            # Update UTXO set
            self._update_utxos(block)
            return True
        return False

    def _validate_block(self, block: Block) -> bool:
        """Validate block before adding to chain"""
        # Check if previous hash matches
        if block.header.previous_hash != self.get_latest_block().get_hash():
            return False

        # Check proof of work (for PoUW this involves ML verification)
        if not self._validate_proof_of_work(block):
            return False

        # Validate all transactions (skip coinbase, and use different validation for block context)
        for i, tx in enumerate(block.transactions):
            if i == 0:  # Skip coinbase transaction
                continue
            if not self._validate_transaction_for_block(tx):
                return False

        return True

    def _validate_proof_of_work(self, block: Block) -> bool:
        """Validate the PoUW proof"""
        # Check if block hash meets difficulty target
        block_hash = int(block.get_hash(), 16)
        if block_hash >= self.difficulty_target:
            return False

        # For PoUW, we also need to verify the ML work
        # This would involve re-running the ML iteration
        if block.mining_proof:
            return self._verify_ml_work(block)

        return True

    def _verify_ml_work(self, block: Block) -> bool:
        """Verify the machine learning work done for this block (production logic)."""
        try:
            from pouw.ml.training import IterationMessage
        except ImportError:
            return False
        proof = block.mining_proof
        if (
            not proof
            or "iteration_message" not in proof
            or "model_state_hash" not in proof
        ):
            return False
        # 1. Deserialize the iteration message
        try:
            it_msg = IterationMessage(
                **json.loads(proof["iteration_message"])
            )
        except Exception:
            return False
        # 2. Check model state hash
        if it_msg.model_state_hash != proof["model_state_hash"]:
            return False
        # 3. Dynamically fetch min accuracy from MLTask if possible
        min_acc = 0.8
        ml_task_id = block.header.ml_task_id
        if ml_task_id and ml_task_id in self.active_tasks:
            task = self.active_tasks[ml_task_id]
            perf_req = task.performance_requirements
            if perf_req and "min_accuracy" in perf_req:
                min_acc = perf_req["min_accuracy"]
        if (
            "accuracy" in it_msg.metrics
            and it_msg.metrics["accuracy"] < min_acc
        ):
            return False
        return True

    def _update_utxos(self, block: Block):
        """Update unspent transaction outputs"""
        for tx in block.transactions:
            tx_hash = tx.get_hash()
            # Remove spent outputs
            for inp in tx.inputs:
                if inp.get("previous_hash") != "0" * 64:  # Not coinbase
                    utxo_key = f"{inp['previous_hash']}:{inp['index']}"
                    self.utxos.pop(utxo_key, None)
                    delete_utxo(utxo_key)
            # Add new outputs
            for i, output in enumerate(tx.outputs):
                utxo_key = f"{tx_hash}:{i}"
                self.utxos[utxo_key] = output
                save_utxo(utxo_key, output)

    def get_chain_length(self) -> int:
        """Get the length of the blockchain"""
        return len(
            self.chain
        )

    def get_mempool_size(self) -> int:
        """Get number of transactions in mempool"""
        return len(
            self.mempool
        )
