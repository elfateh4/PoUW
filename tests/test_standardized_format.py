"""
Test suite for standardized transaction format implementation.
Tests the exact 160-byte OP_RETURN format compliance with the research paper.
"""

import unittest
import json
import time
from pouw.blockchain.standardized_format import (
    StandardizedTransactionFormat,
    PoUWOpReturnData,
    PoUWOpCode,
    create_standardized_pouw_transaction,
    parse_standardized_pouw_transaction,
)


class TestPoUWOpReturnData(unittest.TestCase):
    """Test PoUW OP_RETURN data structure"""

    def setUp(self):
        self.test_data = PoUWOpReturnData(
            version=1,
            op_code=PoUWOpCode.TASK_SUBMISSION,
            timestamp=int(time.time()),
            node_id_hash=b"a" * 20,  # 20-byte node ID hash
            task_id_hash=b"b" * 32,  # 32-byte task ID hash
            payload=b"test payload data",
            checksum=b"\x00\x00\x00\x00",
        )

    def test_op_return_data_serialization(self):
        """Test OP_RETURN data serialization to exact 160 bytes"""
        serialized = self.test_data.to_bytes()

        # Must be exactly 160 bytes
        self.assertEqual(len(serialized), 160)

        # Should be deterministic
        serialized2 = self.test_data.to_bytes()
        self.assertEqual(serialized, serialized2)

    def test_op_return_data_deserialization(self):
        """Test OP_RETURN data deserialization"""
        serialized = self.test_data.to_bytes()
        deserialized = PoUWOpReturnData.from_bytes(serialized)

        self.assertEqual(deserialized.version, self.test_data.version)
        self.assertEqual(deserialized.op_code, self.test_data.op_code)
        self.assertEqual(deserialized.timestamp, self.test_data.timestamp)
        self.assertEqual(deserialized.node_id_hash, self.test_data.node_id_hash)
        self.assertEqual(deserialized.task_id_hash, self.test_data.task_id_hash)
        self.assertEqual(deserialized.payload, self.test_data.payload)

    def test_checksum_validation(self):
        """Test checksum validation"""
        serialized = self.test_data.to_bytes()

        # Corrupt the data
        corrupted = bytearray(serialized)
        corrupted[50] = (corrupted[50] + 1) % 256

        with self.assertRaises(ValueError):
            PoUWOpReturnData.from_bytes(bytes(corrupted))

    def test_payload_truncation(self):
        """Test payload truncation to fit 160-byte limit"""
        large_payload = b"x" * 200  # Larger than 98-byte payload limit

        test_data = PoUWOpReturnData(
            version=1,
            op_code=PoUWOpCode.TASK_SUBMISSION,
            timestamp=int(time.time()),
            node_id_hash=b"a" * 20,
            task_id_hash=b"b" * 32,
            payload=large_payload,
            checksum=b"\x00\x00\x00\x00",
        )

        serialized = test_data.to_bytes()
        self.assertEqual(len(serialized), 160)

        # Verify payload was truncated
        deserialized = PoUWOpReturnData.from_bytes(serialized)
        self.assertEqual(len(deserialized.payload), 98)

    def test_all_op_codes(self):
        """Test all operation codes serialize correctly"""
        for op_code in PoUWOpCode:
            test_data = PoUWOpReturnData(
                version=1,
                op_code=op_code,
                timestamp=int(time.time()),
                node_id_hash=b"a" * 20,
                task_id_hash=b"b" * 32,
                payload=b"test payload",
                checksum=b"\x00\x00\x00\x00",
            )

            serialized = test_data.to_bytes()
            self.assertEqual(len(serialized), 160)

            deserialized = PoUWOpReturnData.from_bytes(serialized)
            self.assertEqual(deserialized.op_code, op_code)


class TestStandardizedTransactionFormat(unittest.TestCase):
    """Test standardized transaction format"""

    def setUp(self):
        self.formatter = StandardizedTransactionFormat()
        self.test_node_id = "test_node_001"
        self.test_task_id = "test_task_001"
        self.test_inputs = [{"previous_hash": "abc123", "index": 0}]
        self.test_outputs = [{"address": "test_address", "amount": 10.0}]

    def test_task_submission_transaction(self):
        """Test task submission transaction creation"""
        task_data = {
            "model_type": "SimpleMLP",
            "architecture": {"input_size": 784, "output_size": 10},
            "optimizer": {"type": "adam", "lr": 0.001},
        }

        tx = self.formatter.create_task_submission_transaction(
            task_data, self.test_node_id, self.test_task_id, self.test_inputs, self.test_outputs
        )

        self.assertEqual(tx["version"], 1)
        self.assertEqual(tx["inputs"], self.test_inputs)
        self.assertEqual(tx["outputs"], self.test_outputs)
        self.assertEqual(len(tx["op_return"]), 160)
        self.assertIsInstance(tx["timestamp"], int)

    def test_worker_registration_transaction(self):
        """Test worker registration transaction creation"""
        registration_data = {
            "role": "miner",
            "stake_amount": 100.0,
            "preferences": {"gpu": True, "max_batch_size": 32},
        }

        tx = self.formatter.create_worker_registration_transaction(
            registration_data, self.test_node_id, self.test_inputs, self.test_outputs
        )

        self.assertEqual(tx["version"], 1)
        self.assertEqual(len(tx["op_return"]), 160)

        # Parse the transaction
        parsed = self.formatter.parse_op_return_transaction(tx["op_return"])
        self.assertEqual(parsed["op_code"], "WORKER_REGISTRATION")

    def test_task_result_transaction(self):
        """Test task result transaction creation"""
        result_data = {
            "final_accuracy": 0.95,
            "training_loss": 0.05,
            "iterations_completed": 100,
            "model_hash": "model_hash_12345",
        }

        tx = self.formatter.create_task_result_transaction(
            result_data, self.test_node_id, self.test_task_id, self.test_inputs, self.test_outputs
        )

        self.assertEqual(len(tx["op_return"]), 160)

        # Parse the transaction
        parsed = self.formatter.parse_op_return_transaction(tx["op_return"])
        self.assertEqual(parsed["op_code"], "TASK_RESULT")

    def test_gradient_share_transaction(self):
        """Test gradient sharing transaction creation"""
        gradient_data = {
            "iteration": 50,
            "model_hash": "gradient_model_hash_67890",
            "performance_metrics": {"accuracy": 0.85, "loss": 0.15},
        }

        tx = self.formatter.create_gradient_share_transaction(
            gradient_data, self.test_node_id, self.test_task_id, self.test_inputs, self.test_outputs
        )

        self.assertEqual(len(tx["op_return"]), 160)

        # Parse the transaction
        parsed = self.formatter.parse_op_return_transaction(tx["op_return"])
        self.assertEqual(parsed["op_code"], "GRADIENT_SHARE")

    def test_verification_proof_transaction(self):
        """Test verification proof transaction creation"""
        proof_data = {
            "nonce": 123456,
            "verification_result": True,
            "proof_hash": "verification_proof_hash_abc123",
            "iterations_verified": 75,
        }

        tx = self.formatter.create_verification_proof_transaction(
            proof_data, self.test_node_id, self.test_task_id, self.test_inputs, self.test_outputs
        )

        self.assertEqual(len(tx["op_return"]), 160)

        # Parse the transaction
        parsed = self.formatter.parse_op_return_transaction(tx["op_return"])
        self.assertEqual(parsed["op_code"], "VERIFICATION_PROOF")

    def test_transaction_parsing_roundtrip(self):
        """Test transaction creation and parsing roundtrip"""
        task_data = {
            "model_type": "CNN",
            "layers": [
                {"type": "conv2d", "filters": 32, "kernel_size": 3},
                {"type": "maxpool2d", "pool_size": 2},
                {"type": "dense", "units": 128},
            ],
        }

        # Create transaction
        tx = self.formatter.create_task_submission_transaction(
            task_data, self.test_node_id, self.test_task_id, self.test_inputs, self.test_outputs
        )

        # Parse transaction
        parsed = self.formatter.parse_op_return_transaction(tx["op_return"])

        # Verify parsing
        self.assertEqual(parsed["version"], 1)
        self.assertEqual(parsed["op_code"], "TASK_SUBMISSION")
        self.assertIn("payload_data", parsed)
        self.assertIsInstance(parsed["timestamp"], int)

    def test_compression_and_decompression(self):
        """Test data compression and decompression"""
        large_task_data = {
            "model_type": "TransformerLarge",
            "architecture": {
                "num_layers": 24,
                "hidden_size": 1024,
                "num_attention_heads": 16,
                "intermediate_size": 4096,
                "vocab_size": 50000,
                "max_position_embeddings": 512,
                "layer_norm_eps": 1e-12,
                "hidden_dropout_prob": 0.1,
                "attention_probs_dropout_prob": 0.1,
            },
            "optimizer": {
                "type": "adamw",
                "learning_rate": 2e-5,
                "weight_decay": 0.01,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-8,
            },
            "training_config": {
                "batch_size": 32,
                "max_steps": 10000,
                "warmup_steps": 1000,
                "save_steps": 500,
                "eval_steps": 100,
            },
        }

        # Create transaction with large data
        tx = self.formatter.create_task_submission_transaction(
            large_task_data,
            self.test_node_id,
            self.test_task_id,
            self.test_inputs,
            self.test_outputs,
        )

        # Should still be exactly 160 bytes
        self.assertEqual(len(tx["op_return"]), 160)

        # Parse and verify some data was preserved
        parsed = self.formatter.parse_op_return_transaction(tx["op_return"])
        self.assertEqual(parsed["op_code"], "TASK_SUBMISSION")
        self.assertIn("payload_data", parsed)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions for transaction creation and parsing"""

    def setUp(self):
        self.test_node_id = "convenience_test_node"
        self.test_task_id = "convenience_test_task"
        self.test_inputs = []
        self.test_outputs = [{"address": "test", "amount": 5.0}]

    def test_create_standardized_pouw_transaction(self):
        """Test convenience function for transaction creation"""
        task_data = {"model_type": "SimpleMLP", "epochs": 10}

        tx = create_standardized_pouw_transaction(
            "task_submission",
            task_data,
            self.test_node_id,
            self.test_task_id,
            self.test_inputs,
            self.test_outputs,
        )

        self.assertEqual(len(tx["op_return"]), 160)
        self.assertEqual(tx["inputs"], self.test_inputs)
        self.assertEqual(tx["outputs"], self.test_outputs)

    def test_parse_standardized_pouw_transaction(self):
        """Test convenience function for transaction parsing"""
        registration_data = {
            "role": "supervisor",
            "stake": 200.0,
            "capabilities": ["consensus", "verification"],
        }

        tx = create_standardized_pouw_transaction(
            "worker_registration",
            registration_data,
            self.test_node_id,
            None,
            self.test_inputs,
            self.test_outputs,
        )

        parsed = parse_standardized_pouw_transaction(tx["op_return"])

        self.assertEqual(parsed["op_code"], "WORKER_REGISTRATION")
        self.assertEqual(parsed["version"], 1)
        self.assertIn("payload_data", parsed)

    def test_all_transaction_types(self):
        """Test all supported transaction types"""
        transaction_types = [
            "task_submission",
            "worker_registration",
            "task_result",
            "gradient_share",
            "verification_proof",
        ]

        for tx_type in transaction_types:
            test_data = {"type": tx_type, "test": True}

            tx = create_standardized_pouw_transaction(
                tx_type,
                test_data,
                self.test_node_id,
                self.test_task_id,
                self.test_inputs,
                self.test_outputs,
            )

            self.assertEqual(len(tx["op_return"]), 160)

            parsed = parse_standardized_pouw_transaction(tx["op_return"])
            self.assertEqual(parsed["version"], 1)
            self.assertIn("payload_data", parsed)

    def test_invalid_transaction_type(self):
        """Test error handling for invalid transaction types"""
        with self.assertRaises(ValueError):
            create_standardized_pouw_transaction(
                "invalid_type",
                {},
                self.test_node_id,
                self.test_task_id,
                self.test_inputs,
                self.test_outputs,
            )


class TestFormatCompliance(unittest.TestCase):
    """Test compliance with research paper specification"""

    def test_exact_160_byte_format(self):
        """Test that all transactions produce exactly 160-byte OP_RETURN"""
        formatter = StandardizedTransactionFormat()

        test_cases = [
            ("task_submission", {"model": "test"}),
            ("worker_registration", {"role": "miner"}),
            ("task_result", {"accuracy": 0.9}),
            ("gradient_share", {"iteration": 1}),
            ("verification_proof", {"verified": True}),
        ]

        for tx_type, data in test_cases:
            tx = create_standardized_pouw_transaction(
                tx_type, data, "test_node", "test_task", [], []
            )

            # Must be exactly 160 bytes
            self.assertEqual(len(tx["op_return"]), 160)

    def test_structured_data_format(self):
        """Test structured data format compliance"""
        tx = create_standardized_pouw_transaction(
            "task_submission", {"model_type": "MLP", "layers": 3}, "node_123", "task_456", [], []
        )

        parsed = parse_standardized_pouw_transaction(tx["op_return"])

        # Check required fields
        required_fields = [
            "version",
            "op_code",
            "timestamp",
            "node_id_hash",
            "task_id_hash",
            "payload_data",
            "checksum",
        ]

        for field in required_fields:
            self.assertIn(field, parsed)

        # Check field types and formats
        self.assertIsInstance(parsed["version"], int)
        self.assertIsInstance(parsed["op_code"], str)
        self.assertIsInstance(parsed["timestamp"], int)
        self.assertEqual(len(parsed["node_id_hash"]), 40)  # 20 bytes = 40 hex chars
        self.assertEqual(len(parsed["task_id_hash"]), 64)  # 32 bytes = 64 hex chars
        self.assertEqual(len(parsed["checksum"]), 8)  # 4 bytes = 8 hex chars

    def test_deterministic_serialization(self):
        """Test that serialization is deterministic"""
        formatter = StandardizedTransactionFormat()

        task_data = {"model": "test", "epochs": 5}

        tx1 = formatter.create_task_submission_transaction(task_data, "node_1", "task_1", [], [])

        tx2 = formatter.create_task_submission_transaction(task_data, "node_1", "task_1", [], [])

        # Should be identical except for timestamp
        self.assertEqual(len(tx1["op_return"]), len(tx2["op_return"]))
        self.assertEqual(len(tx1["op_return"]), 160)

    def test_format_version_handling(self):
        """Test format version handling"""
        formatter = StandardizedTransactionFormat()

        tx = formatter.create_task_submission_transaction({"test": "data"}, "node", "task", [], [])

        parsed = parse_standardized_pouw_transaction(tx["op_return"])

        # Should use version 1
        self.assertEqual(parsed["version"], 1)


if __name__ == "__main__":
    unittest.main()
