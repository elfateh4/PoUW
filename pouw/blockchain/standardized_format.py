"""
Standardized Transaction Format for PoUW - Research Paper Compliance

This module implements the exact 160-byte OP_RETURN transaction format
as specified in the research paper "A Proof of Useful Work for Artificial
Intelligence on the Blockchain" by Lihu et al.

The standardized format ensures compatibility with the paper specification
while maintaining integration with the existing PoUW blockchain system.
"""

import struct
import hashlib
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

class PoUWOpCode(Enum):
    """PoUW operation codes for structured OP_RETURN data"""
    TASK_SUBMISSION = 0x01
    TASK_RESULT = 0x02
    WORKER_REGISTRATION = 0x03
    CONSENSUS_VOTE = 0x04
    GRADIENT_SHARE = 0x05
    VERIFICATION_PROOF = 0x06
    ECONOMIC_EVENT = 0x07
    NETWORK_STATE = 0x08

@dataclass
class PoUWOpReturnData:
    """Structured PoUW data for OP_RETURN transactions"""
    version: int  # 1 byte
    op_code: PoUWOpCode  # 1 byte
    timestamp: int  # 4 bytes
    node_id_hash: bytes  # 20 bytes (SHA1 of node ID)
    task_id_hash: bytes  # 32 bytes (SHA256 of task ID)
    payload: bytes  # Variable length payload
    checksum: bytes  # 4 bytes (CRC32)
    
    def to_bytes(self) -> bytes:
        """Convert to exactly 160 bytes for OP_RETURN"""
        # Calculate payload size to fit in 160 bytes total
        # Structure: version(1) + op_code(1) + timestamp(4) + node_id_hash(20) + task_id_hash(32) + payload(98) + checksum(4) = 160
        max_payload_size = 98
        
        # Truncate payload if necessary
        payload = self.payload[:max_payload_size]
        # Pad payload to exact size
        payload = payload.ljust(max_payload_size, b'\x00')
        
        # Build the structured data
        data = struct.pack(
            '>B',  # version (1 byte, big-endian)
            self.version
        )
        data += struct.pack(
            '>B',  # op_code (1 byte, big-endian)
            self.op_code.value
        )
        data += struct.pack(
            '>I',  # timestamp (4 bytes, big-endian)
            self.timestamp
        )
        data += self.node_id_hash  # 20 bytes
        data += self.task_id_hash  # 32 bytes
        data += payload  # 98 bytes
        
        # Calculate checksum
        checksum = struct.pack('>I', self._calculate_checksum(data))
        data += checksum  # 4 bytes
        
        assert len(data) == 160, f"OP_RETURN data must be exactly 160 bytes, got {len(data)}"
        return data
    
    def _calculate_checksum(self, data: bytes) -> int:
        """Calculate CRC32 checksum"""
        import zlib
        return zlib.crc32(data) & 0xffffffff
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'PoUWOpReturnData':
        """Parse structured OP_RETURN data from bytes"""
        if len(data) != 160:
            raise ValueError(f"OP_RETURN data must be exactly 160 bytes, got {len(data)}")
        
        # Unpack the structured data
        offset = 0
        
        version = struct.unpack('>B', data[offset:offset+1])[0]
        offset += 1
        
        op_code_val = struct.unpack('>B', data[offset:offset+1])[0]
        op_code = PoUWOpCode(op_code_val)
        offset += 1
        
        timestamp = struct.unpack('>I', data[offset:offset+4])[0]
        offset += 4
        
        node_id_hash = data[offset:offset+20]
        offset += 20
        
        task_id_hash = data[offset:offset+32]
        offset += 32
        
        payload = data[offset:offset+98].rstrip(b'\x00')  # Remove padding
        offset += 98
        
        checksum = struct.unpack('>I', data[offset:offset+4])[0]
        
        # Verify checksum
        data_without_checksum = data[:-4]
        expected_checksum = cls._calculate_checksum_static(data_without_checksum)
        if checksum != expected_checksum:
            raise ValueError(f"Checksum mismatch: expected {expected_checksum}, got {checksum}")
        
        return cls(
            version=version,
            op_code=op_code,
            timestamp=timestamp,
            node_id_hash=node_id_hash,
            task_id_hash=task_id_hash,
            payload=payload,
            checksum=checksum.to_bytes(4, 'big')
        )
    
    @staticmethod
    def _calculate_checksum_static(data: bytes) -> int:
        """Static method for checksum calculation"""
        import zlib
        return zlib.crc32(data) & 0xffffffff

class StandardizedTransactionFormat:
    """Creates standardized PoUW transactions with exact 160-byte OP_RETURN format"""
    
    def __init__(self):
        self.version = 1  # Current PoUW format version
    
    def create_task_submission_transaction(self, 
                                         task_data: Dict[str, Any],
                                         node_id: str,
                                         task_id: str,
                                         inputs: List[Dict[str, Any]],
                                         outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a standardized task submission transaction"""
        # Compress task data for payload
        payload = self._compress_task_data(task_data)
        
        op_return_data = PoUWOpReturnData(
            version=self.version,
            op_code=PoUWOpCode.TASK_SUBMISSION,
            timestamp=int(__import__('time').time()),
            node_id_hash=self._hash_node_id(node_id),
            task_id_hash=self._hash_task_id(task_id),
            payload=payload,
            checksum=b'\x00\x00\x00\x00'  # Will be calculated in to_bytes()
        )
        
        return {
            'version': 1,
            'inputs': inputs,
            'outputs': outputs,
            'op_return': op_return_data.to_bytes(),
            'timestamp': int(__import__('time').time())
        }
    
    def create_worker_registration_transaction(self,
                                             registration_data: Dict[str, Any],
                                             node_id: str,
                                             inputs: List[Dict[str, Any]],
                                             outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a standardized worker registration transaction"""
        payload = self._compress_registration_data(registration_data)
        
        op_return_data = PoUWOpReturnData(
            version=self.version,
            op_code=PoUWOpCode.WORKER_REGISTRATION,
            timestamp=int(__import__('time').time()),
            node_id_hash=self._hash_node_id(node_id),
            task_id_hash=b'\x00' * 32,  # No task ID for registration
            payload=payload,
            checksum=b'\x00\x00\x00\x00'
        )
        
        return {
            'version': 1,
            'inputs': inputs,
            'outputs': outputs,
            'op_return': op_return_data.to_bytes(),
            'timestamp': int(__import__('time').time())
        }
    
    def create_task_result_transaction(self,
                                     result_data: Dict[str, Any],
                                     node_id: str,
                                     task_id: str,
                                     inputs: List[Dict[str, Any]],
                                     outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a standardized task result transaction"""
        payload = self._compress_result_data(result_data)
        
        op_return_data = PoUWOpReturnData(
            version=self.version,
            op_code=PoUWOpCode.TASK_RESULT,
            timestamp=int(__import__('time').time()),
            node_id_hash=self._hash_node_id(node_id),
            task_id_hash=self._hash_task_id(task_id),
            payload=payload,
            checksum=b'\x00\x00\x00\x00'
        )
        
        return {
            'version': 1,
            'inputs': inputs,
            'outputs': outputs,
            'op_return': op_return_data.to_bytes(),
            'timestamp': int(__import__('time').time())
        }
    
    def create_gradient_share_transaction(self,
                                        gradient_data: Dict[str, Any],
                                        node_id: str,
                                        task_id: str,
                                        inputs: List[Dict[str, Any]],
                                        outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a standardized gradient sharing transaction"""
        payload = self._compress_gradient_data(gradient_data)
        
        op_return_data = PoUWOpReturnData(
            version=self.version,
            op_code=PoUWOpCode.GRADIENT_SHARE,
            timestamp=int(__import__('time').time()),
            node_id_hash=self._hash_node_id(node_id),
            task_id_hash=self._hash_task_id(task_id),
            payload=payload,
            checksum=b'\x00\x00\x00\x00'
        )
        
        return {
            'version': 1,
            'inputs': inputs,
            'outputs': outputs,
            'op_return': op_return_data.to_bytes(),
            'timestamp': int(__import__('time').time())
        }
    
    def create_verification_proof_transaction(self,
                                            proof_data: Dict[str, Any],
                                            node_id: str,
                                            task_id: str,
                                            inputs: List[Dict[str, Any]],
                                            outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a standardized verification proof transaction"""
        payload = self._compress_proof_data(proof_data)
        
        op_return_data = PoUWOpReturnData(
            version=self.version,
            op_code=PoUWOpCode.VERIFICATION_PROOF,
            timestamp=int(__import__('time').time()),
            node_id_hash=self._hash_node_id(node_id),
            task_id_hash=self._hash_task_id(task_id),
            payload=payload,
            checksum=b'\x00\x00\x00\x00'
        )
        
        return {
            'version': 1,
            'inputs': inputs,
            'outputs': outputs,
            'op_return': op_return_data.to_bytes(),
            'timestamp': int(__import__('time').time())
        }
    
    def parse_op_return_transaction(self, op_return_bytes: bytes) -> Dict[str, Any]:
        """Parse a standardized OP_RETURN transaction"""
        op_return_data = PoUWOpReturnData.from_bytes(op_return_bytes)
        
        # Decompress payload based on operation type
        if op_return_data.op_code == PoUWOpCode.TASK_SUBMISSION:
            payload_data = self._decompress_task_data(op_return_data.payload)
        elif op_return_data.op_code == PoUWOpCode.WORKER_REGISTRATION:
            payload_data = self._decompress_registration_data(op_return_data.payload)
        elif op_return_data.op_code == PoUWOpCode.TASK_RESULT:
            payload_data = self._decompress_result_data(op_return_data.payload)
        elif op_return_data.op_code == PoUWOpCode.GRADIENT_SHARE:
            payload_data = self._decompress_gradient_data(op_return_data.payload)
        elif op_return_data.op_code == PoUWOpCode.VERIFICATION_PROOF:
            payload_data = self._decompress_proof_data(op_return_data.payload)
        else:
            payload_data = {'raw_payload': op_return_data.payload.hex()}
        
        return {
            'version': op_return_data.version,
            'op_code': op_return_data.op_code.name,
            'timestamp': op_return_data.timestamp,
            'node_id_hash': op_return_data.node_id_hash.hex(),
            'task_id_hash': op_return_data.task_id_hash.hex(),
            'payload_data': payload_data,
            'checksum': op_return_data.checksum.hex()
        }
    
    def _hash_node_id(self, node_id: str) -> bytes:
        """Create 20-byte SHA1 hash of node ID"""
        return hashlib.sha1(node_id.encode()).digest()
    
    def _hash_task_id(self, task_id: str) -> bytes:
        """Create 32-byte SHA256 hash of task ID"""
        return hashlib.sha256(task_id.encode()).digest()
    
    def _compress_task_data(self, task_data: Dict[str, Any]) -> bytes:
        """Compress task data to fit in payload"""
        import zlib
        json_str = json.dumps(task_data, separators=(',', ':'))
        compressed = zlib.compress(json_str.encode(), level=9)
        return compressed[:98]  # Fit in available payload space
    
    def _compress_registration_data(self, registration_data: Dict[str, Any]) -> bytes:
        """Compress registration data to fit in payload"""
        import zlib
        json_str = json.dumps(registration_data, separators=(',', ':'))
        compressed = zlib.compress(json_str.encode(), level=9)
        return compressed[:98]
    
    def _compress_result_data(self, result_data: Dict[str, Any]) -> bytes:
        """Compress result data to fit in payload"""
        import zlib
        json_str = json.dumps(result_data, separators=(',', ':'))
        compressed = zlib.compress(json_str.encode(), level=9)
        return compressed[:98]
    
    def _compress_gradient_data(self, gradient_data: Dict[str, Any]) -> bytes:
        """Compress gradient data to fit in payload"""
        import zlib
        # Use simplified representation for gradients
        simplified = {
            'iteration': gradient_data.get('iteration', 0),
            'model_hash': gradient_data.get('model_hash', '')[:16],  # Truncate hash
            'performance': gradient_data.get('performance_metrics', {})
        }
        json_str = json.dumps(simplified, separators=(',', ':'))
        compressed = zlib.compress(json_str.encode(), level=9)
        return compressed[:98]
    
    def _compress_proof_data(self, proof_data: Dict[str, Any]) -> bytes:
        """Compress verification proof data to fit in payload"""
        import zlib
        # Use simplified representation for proofs
        simplified = {
            'nonce': proof_data.get('nonce', 0),
            'verification_result': proof_data.get('verification_result', False),
            'proof_hash': proof_data.get('proof_hash', '')[:16]  # Truncate hash
        }
        json_str = json.dumps(simplified, separators=(',', ':'))
        compressed = zlib.compress(json_str.encode(), level=9)
        return compressed[:98]
    
    def _decompress_task_data(self, payload: bytes) -> Dict[str, Any]:
        """Decompress task data from payload"""
        try:
            import zlib
            decompressed = zlib.decompress(payload)
            return json.loads(decompressed.decode())
        except:
            return {'error': 'Failed to decompress task data', 'raw': payload.hex()}
    
    def _decompress_registration_data(self, payload: bytes) -> Dict[str, Any]:
        """Decompress registration data from payload"""
        try:
            import zlib
            decompressed = zlib.decompress(payload)
            return json.loads(decompressed.decode())
        except:
            return {'error': 'Failed to decompress registration data', 'raw': payload.hex()}
    
    def _decompress_result_data(self, payload: bytes) -> Dict[str, Any]:
        """Decompress result data from payload"""
        try:
            import zlib
            decompressed = zlib.decompress(payload)
            return json.loads(decompressed.decode())
        except:
            return {'error': 'Failed to decompress result data', 'raw': payload.hex()}
    
    def _decompress_gradient_data(self, payload: bytes) -> Dict[str, Any]:
        """Decompress gradient data from payload"""
        try:
            import zlib
            decompressed = zlib.decompress(payload)
            return json.loads(decompressed.decode())
        except:
            return {'error': 'Failed to decompress gradient data', 'raw': payload.hex()}
    
    def _decompress_proof_data(self, payload: bytes) -> Dict[str, Any]:
        """Decompress proof data from payload"""
        try:
            import zlib
            decompressed = zlib.decompress(payload)
            return json.loads(decompressed.decode())
        except:
            return {'error': 'Failed to decompress proof data', 'raw': payload.hex()}

# Integration helper functions
def create_standardized_pouw_transaction(transaction_type: str, 
                                       data: Dict[str, Any],
                                       node_id: str,
                                       task_id: Optional[str] = None,
                                       inputs: Optional[List[Dict[str, Any]]] = None,
                                       outputs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Convenience function to create standardized PoUW transactions
    
    Args:
        transaction_type: Type of transaction ('task_submission', 'worker_registration', etc.)
        data: Transaction-specific data
        node_id: Node identifier
        task_id: Task identifier (if applicable)
        inputs: Transaction inputs
        outputs: Transaction outputs
    
    Returns:
        Standardized transaction dictionary
    """
    if inputs is None:
        inputs = []
    if outputs is None:
        outputs = []
    
    formatter = StandardizedTransactionFormat()
    
    if transaction_type == 'task_submission':
        return formatter.create_task_submission_transaction(data, node_id, task_id or '', inputs, outputs)
    elif transaction_type == 'worker_registration':
        return formatter.create_worker_registration_transaction(data, node_id, inputs, outputs)
    elif transaction_type == 'task_result':
        return formatter.create_task_result_transaction(data, node_id, task_id or '', inputs, outputs)
    elif transaction_type == 'gradient_share':
        return formatter.create_gradient_share_transaction(data, node_id, task_id or '', inputs, outputs)
    elif transaction_type == 'verification_proof':
        return formatter.create_verification_proof_transaction(data, node_id, task_id or '', inputs, outputs)
    else:
        raise ValueError(f"Unknown transaction type: {transaction_type}")

def parse_standardized_pouw_transaction(op_return_bytes: bytes) -> Dict[str, Any]:
    """
    Convenience function to parse standardized PoUW transactions
    
    Args:
        op_return_bytes: 160-byte OP_RETURN data
    
    Returns:
        Parsed transaction data
    """
    formatter = StandardizedTransactionFormat()
    return formatter.parse_op_return_transaction(op_return_bytes)
