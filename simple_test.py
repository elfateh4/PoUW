#!/usr/bin/env python3

import sys
sys.path.append('/home/elfateh/Projects/PoUW')

print("Starting import...")
from pouw.blockchain.core import Blockchain, Transaction
print("Import successful")

print("Creating blockchain...")
blockchain = Blockchain()
print(f"Blockchain created with {blockchain.get_chain_length()} blocks")

print("Creating transaction...")
tx = Transaction(version=1, inputs=[], outputs=[{'address': 'user1', 'amount': 10.0}])
print(f"Transaction created: {tx.get_hash()}")

print("Adding transaction to mempool...")
result = blockchain.add_transaction_to_mempool(tx)
print(f"Added to mempool: {result}")

print("Creating block...")
block = blockchain.create_block([tx], 'miner1')
print(f"Block created: {block.get_hash()}")

print("Setting easy difficulty...")
blockchain.difficulty_target = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
block.header.nonce = 1

print("Testing validation...")
print(f"PoW valid: {blockchain._validate_proof_of_work(block)}")
print(f"Block valid: {blockchain._validate_block(block)}")

print("Test completed successfully")
