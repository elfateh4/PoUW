#!/usr/bin/env python3
"""
Submit a machine learning task to the PoUW network.
"""

import asyncio
import argparse
import json
from pouw.node import PoUWNode
from pouw.economics import NodeRole


async def main():
    parser = argparse.ArgumentParser(description='Submit ML task to PoUW network')
    parser.add_argument('--node-id', required=True, help='Client node identifier')
    parser.add_argument('--task', required=True, help='Task definition JSON file')
    parser.add_argument('--fee', type=float, required=True, help='Task fee in PAI coins')
    parser.add_argument('--host', default='localhost', help='Host to bind to')
    parser.add_argument('--port', type=int, default=9000, help='Port to bind to')
    parser.add_argument('--bootstrap-peer', action='append', nargs=2,
                       metavar=('HOST', 'PORT'), help='Bootstrap peer (can be repeated)')
    
    args = parser.parse_args()
    
    # Load task definition
    with open(args.task, 'r') as f:
        task_definition = json.load(f)
    
    # Create client node
    node = PoUWNode(args.node_id, NodeRole.PEER, args.host, args.port)
    
    try:
        # Start node
        await node.start()
        
        # Connect to network
        if args.bootstrap_peer:
            bootstrap_peers = [(host, int(port)) for host, port in args.bootstrap_peer]
            await node.connect_to_network(bootstrap_peers)
            await asyncio.sleep(2)  # Allow connections to establish
        
        # Submit task
        task_id = node.submit_ml_task(task_definition, args.fee)
        print(f"Submitted task {task_id} with fee {args.fee} PAI")
        
        # Wait for task completion (simplified)
        print("Waiting for task completion...")
        await asyncio.sleep(60)  # Wait 1 minute
        
        # Check task status
        if task_id in node.economic_system.completed_tasks:
            task_info = node.economic_system.completed_tasks[task_id]
            print(f"Task completed! Rewards distributed: {task_info['rewards']}")
        else:
            print("Task still in progress or failed")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        await node.stop()


if __name__ == '__main__':
    asyncio.run(main())
