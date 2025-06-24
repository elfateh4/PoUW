#!/usr/bin/env python3

import sys
sys.path.append('/home/elfateh/Projects/PoUW')

from pouw.blockchain import Blockchain, Transaction

try:
    # Test basic blockchain functionality
    blockchain = Blockchain()
    print('Initial chain length:', blockchain.get_chain_length())

    # Create transaction
    tx = Transaction(version=1, inputs=[], outputs=[{'address': 'user1', 'amount': 10.0}])
    print('Transaction hash:', tx.get_hash())

    # Add to mempool
    result = blockchain.add_transaction_to_mempool(tx)
    print('Added to mempool:', result)
    print('Mempool size:', blockchain.get_mempool_size())

    # Create block
    block = blockchain.create_block([tx], 'miner1')
    print('Block created:', block.get_hash())

    # Set easy difficulty and nonce
    blockchain.difficulty_target = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
    block.header.nonce = 1
    print('Block hash meets target:', int(block.get_hash(), 16) < blockchain.difficulty_target)

    # Test individual validations
    print('Previous hash matches:', block.header.previous_hash == blockchain.get_latest_block().get_hash())
    print('PoW valid:', blockchain._validate_proof_of_work(block))
    
    # Test transaction validation
    for i, transaction in enumerate(block.transactions):
        valid = blockchain._validate_transaction(transaction)
        print(f'Transaction {i} valid:', valid)
    
    print('Block valid:', blockchain._validate_block(block))

    # Try to add
    success = blockchain.add_block(block)
    print('Block added successfully:', success)
    print('Final chain length:', blockchain.get_chain_length())
    
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
