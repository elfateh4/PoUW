"""
Core blockchain data structures for PoUW.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime


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
        return {
            'version': self.version,
            'inputs': self.inputs,
            'outputs': self.outputs,
            'op_return': self.op_return.hex() if self.op_return else None,
            'timestamp': self.timestamp,
            'signature': self.signature.hex() if self.signature else None
        }
    
    def get_hash(self) -> str:
        """Calculate transaction hash"""
        tx_data = self.to_dict()
        tx_data.pop('signature', None)  # Exclude signature from hash
        tx_string = json.dumps(tx_data, sort_keys=True)
        return hashlib.sha256(tx_string.encode()).hexdigest()


@dataclass
class PayForTaskTransaction(Transaction):
    """Transaction to submit ML task and pay for training"""
    task_definition: Dict[str, Any] = field(default_factory=dict)
    fee: float = 0.0
    
    def __post_init__(self):
        # Encode task in OP_RETURN
        task_data = json.dumps({
            'type': 'PAY_FOR_TASK',
            'task': self.task_definition,
            'fee': self.fee
        })
        self.op_return = task_data.encode()[:160]  # Limit to 160 bytes


@dataclass
class BuyTicketsTransaction(Transaction):
    """Transaction to stake and register as worker node"""
    role: str = ""  # 'miner', 'supervisor', 'evaluator', 'verifier'
    stake_amount: float = 0.0
    preferences: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        stake_data = json.dumps({
            'type': 'BUY_TICKETS',
            'role': self.role,
            'stake': self.stake_amount,
            'preferences': self.preferences
        })
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
            'version': self.version,
            'previous_hash': self.previous_hash,
            'merkle_root': self.merkle_root,
            'timestamp': self.timestamp,
            'target': self.target,
            'nonce': self.nonce,
            'ml_task_id': self.ml_task_id,
            'message_history_hash': self.message_history_hash,
            'iteration_message_hash': self.iteration_message_hash,
            'zero_nonce_block_hash': self.zero_nonce_block_hash
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
            self.header.merkle_root = "0" * 64
    
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
                next_level.append(hashlib.sha256(combined.encode()).hexdigest())
            level = next_level
        
        return level[0]
    
    def get_hash(self) -> str:
        """Get block hash"""
        return self.header.get_hash()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'header': self.header.to_dict(),
            'transactions': [tx.to_dict() for tx in self.transactions],
            'mining_proof': self.mining_proof
        }


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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'model_type': self.model_type,
            'architecture': self.architecture,
            'optimizer': self.optimizer,
            'stopping_criterion': self.stopping_criterion,
            'validation_strategy': self.validation_strategy,
            'metrics': self.metrics,
            'dataset_info': self.dataset_info,
            'performance_requirements': self.performance_requirements,
            'fee': self.fee,
            'client_id': self.client_id,
            'created_at': self.created_at
        }


class Blockchain:
    """PoUW Blockchain implementation"""
    
    def __init__(self):
        self.chain: List[Block] = []
        self.mempool: List[Transaction] = []
        self.utxos: Dict[str, Dict[str, Any]] = {}  # Unspent transaction outputs
        self.active_tasks: Dict[str, MLTask] = {}
        self.difficulty_target = 0x0000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        
        # Create genesis block
        self._create_genesis_block()
    
    def _create_genesis_block(self):
        """Create the first block in the chain"""
        genesis_header = PoUWBlockHeader(
            version=1,
            previous_hash="0" * 64,
            merkle_root="0" * 64,
            timestamp=int(time.time()),
            target=self.difficulty_target,
            nonce=0
        )
        
        genesis_block = Block(header=genesis_header)
        self.chain.append(genesis_block)
    
    def get_latest_block(self) -> Block:
        """Get the most recent block"""
        return self.chain[-1]
    
    def add_transaction_to_mempool(self, transaction: Transaction):
        """Add transaction to mempool for future inclusion in blocks"""
        if self._validate_transaction(transaction):
            self.mempool.append(transaction)
            return True
        return False
    
    def _validate_transaction(self, transaction: Transaction) -> bool:
        """Validate transaction before adding to mempool"""
        # Basic validation - in production this would be more comprehensive
        if transaction.get_hash() in [tx.get_hash() for tx in self.mempool]:
            return False  # Duplicate transaction
        
        # Validate signature, inputs, outputs, etc.
        # For now, we'll assume valid
        return True
    
    def create_block(self, transactions: List[Transaction], 
                    miner_address: str, mining_proof: Optional[Dict[str, Any]] = None) -> Block:
        """Create a new block with given transactions"""
        previous_block = self.get_latest_block()
        
        # Create coinbase transaction for miner reward
        coinbase_tx = Transaction(
            version=1,
            inputs=[{'previous_hash': '0' * 64, 'index': -1}],
            outputs=[{'address': miner_address, 'amount': 12.5}]  # Block reward
        )
        
        all_transactions = [coinbase_tx] + transactions
        
        header = PoUWBlockHeader(
            version=1,
            previous_hash=previous_block.get_hash(),
            merkle_root="",  # Will be calculated in Block.__post_init__
            timestamp=int(time.time()),
            target=self.difficulty_target,
            nonce=0  # Will be set during mining
        )
        
        block = Block(
            header=header,
            transactions=all_transactions,
            mining_proof=mining_proof
        )
        
        return block
    
    def add_block(self, block: Block) -> bool:
        """Add a validated block to the chain"""
        if self._validate_block(block):
            self.chain.append(block)
            
            # Remove transactions from mempool
            block_tx_hashes = {tx.get_hash() for tx in block.transactions}
            self.mempool = [tx for tx in self.mempool 
                          if tx.get_hash() not in block_tx_hashes]
            
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
        """Verify the machine learning work done for this block"""
        # This is where we would re-run the ML iteration
        # For now, we'll assume it's valid
        return True
    
    def _update_utxos(self, block: Block):
        """Update unspent transaction outputs"""
        for tx in block.transactions:
            tx_hash = tx.get_hash()
            
            # Remove spent outputs
            for inp in tx.inputs:
                if inp.get('previous_hash') != '0' * 64:  # Not coinbase
                    utxo_key = f"{inp['previous_hash']}:{inp['index']}"
                    self.utxos.pop(utxo_key, None)
            
            # Add new outputs
            for i, output in enumerate(tx.outputs):
                utxo_key = f"{tx_hash}:{i}"
                self.utxos[utxo_key] = output
    
    def get_chain_length(self) -> int:
        """Get the length of the blockchain"""
        return len(self.chain)
    
    def get_mempool_size(self) -> int:
        """Get number of transactions in mempool"""
        return len(self.mempool)
    
    def _validate_transaction_for_block(self, transaction: Transaction) -> bool:
        """Validate transaction for inclusion in a block"""
        # Basic validation for block context (don't check mempool duplicates)
        # Validate signature, inputs, outputs, etc.
        # For now, we'll assume valid
        return True
