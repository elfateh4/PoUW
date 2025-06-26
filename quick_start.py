#!/usr/bin/env python3
"""
Quick start example for PoUW node
"""

import asyncio
from pouw.node import PoUWNode, NodeConfig  
from pouw.economics import NodeRole

async def quick_start():
    # Create a miner node
    config = NodeConfig(
        node_id="quick_miner_001",
        role=NodeRole.MINER,
        host="localhost",
        port=8666,
        initial_stake=100.0
    )
    
    node = PoUWNode("quick_miner_001", NodeRole.MINER, "localhost", 8666, config)
    
    try:
        # Start the node
        await node.start()
        print("✅ Node started!")
        
        # Stake tokens  
        ticket = node.stake_and_register(100.0)
        print("✅ Staked 100 PAI tokens")
        
        # Start mining
        await node.start_mining()
        print("✅ Mining started!")
        
        # Run for 60 seconds
        await asyncio.sleep(60)
        
    finally:
        await node.stop()
        print("✅ Node stopped")

if __name__ == "__main__":
    asyncio.run(quick_start())
