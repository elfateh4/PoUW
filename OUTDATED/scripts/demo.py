#!/usr/bin/env python3
"""
Complete PoUW demonstration script.

This script demonstrates a complete PoUW workflow including:
1. Setting up a network with multiple nodes
2. Submitting an ML task
3. Distributed training with mining
4. Verification and reward distribution
"""

import asyncio
import argparse
import logging
import json
import time
from pathlib import Path

from pouw.node import PoUWNode
from pouw.economics import NodeRole
from pouw.blockchain import MLTask


class PoUWDemo:
    """Complete PoUW system demonstration"""
    
    def __init__(self):
        self.nodes = []
        self.client_node = None
        self.task_id = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('PoUWDemo')
    
    async def setup_network(self, num_miners=2, num_supervisors=1):
        """Setup a complete PoUW network"""
        
        self.logger.info("Setting up PoUW network...")
        
        base_port = 8000
        
        # Create client node
        self.client_node = PoUWNode('client_001', NodeRole.PEER, 'localhost', base_port)
        await self.client_node.start()
        self.nodes.append(self.client_node)
        
        # Create miner nodes
        for i in range(num_miners):
            port = base_port + 1 + i
            miner = PoUWNode(f'miner_{i+1}', NodeRole.MINER, 'localhost', port)
            await miner.start()
            
            # Stake and register
            preferences = {
                'model_types': ['mlp', 'cnn'],
                'has_gpu': False,  # CPU-only for demo
                'max_dataset_size': 1000000
            }
            miner.stake_and_register(100.0, preferences)
            miner.is_mining = True
            
            self.nodes.append(miner)
            self.logger.info(f"Started miner {miner.node_id} on port {port}")
        
        # Create supervisor nodes
        for i in range(num_supervisors):
            port = base_port + 1 + num_miners + i
            supervisor = PoUWNode(f'supervisor_{i+1}', NodeRole.SUPERVISOR, 'localhost', port)
            await supervisor.start()
            
            preferences = {
                'storage_capacity': 10000000,
                'bandwidth': 1000000
            }
            supervisor.stake_and_register(50.0, preferences)
            
            self.nodes.append(supervisor)
            self.logger.info(f"Started supervisor {supervisor.node_id} on port {port}")
        
        # Connect all nodes
        await self._connect_all_nodes()
        
        # Wait for network to stabilize
        await asyncio.sleep(3)
        
        self.logger.info(f"Network setup complete with {len(self.nodes)} nodes")
    
    async def _connect_all_nodes(self):
        """Connect all nodes to each other"""
        for i, node in enumerate(self.nodes):
            for j, other_node in enumerate(self.nodes):
                if i != j:
                    await node.p2p_node.connect_to_peer(other_node.host, other_node.port)
                    await asyncio.sleep(0.1)
    
    async def submit_demo_task(self):
        """Submit a demonstration ML task"""
        
        self.logger.info("Submitting demonstration ML task...")
        
        task_definition = {
            "model_type": "mlp",
            "architecture": {
                "input_size": 784,
                "hidden_sizes": [64, 32],
                "output_size": 10
            },
            "optimizer": {
                "type": "adam",
                "learning_rate": 0.001
            },
            "stopping_criterion": {
                "type": "max_epochs",
                "max_epochs": 20
            },
            "validation_strategy": {
                "type": "holdout",
                "validation_split": 0.2
            },
            "metrics": ["accuracy", "loss"],
            "dataset_info": {
                "format": "MNIST",
                "batch_size": 16,  # Small for demo
                "size": 1000
            },
            "performance_requirements": {
                "expected_training_time": 300,
                "gpu": False
            }
        }
        
        self.task_id = self.client_node.submit_ml_task(task_definition, 50.0)
        self.logger.info(f"Submitted task {self.task_id} with fee 50.0 PAI")
        
        return self.task_id
    
    async def monitor_training(self, duration=60):
        """Monitor the training process"""
        
        self.logger.info(f"Monitoring training for {duration} seconds...")
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Print network status
            print("\n" + "="*60)
            print(f"TRAINING PROGRESS ({int(time.time() - start_time)}s)")
            print("="*60)
            
            for node in self.nodes:
                status = node.get_status()
                print(f"{node.node_id:12} | Role: {node.role.value:10} | "
                      f"Height: {status['blockchain_height']:3} | "
                      f"Training: {status['is_training']} | "
                      f"Task: {status['current_task'] or 'None'}")
            
            # Check if task is completed
            if self.task_id and self.client_node:
                if self.task_id in self.client_node.economic_system.completed_tasks:
                    self.logger.info("Task completed!")
                    break
            
            await asyncio.sleep(10)
        
        # Final status
        await self._print_final_results()
    
    async def _print_final_results(self):
        """Print final results of the demonstration"""
        
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        
        if self.task_id and self.client_node:
            if self.task_id in self.client_node.economic_system.completed_tasks:
                task_info = self.client_node.economic_system.completed_tasks[self.task_id]
                
                print(f"Task Status: COMPLETED")
                print(f"Completion Time: {task_info.get('completion_time', 'Unknown')}")
                print(f"Performance Metrics: {task_info.get('performance_metrics', {})}")
                print(f"Rewards Distributed: {task_info.get('rewards', {})}")
                
            elif self.task_id in self.client_node.economic_system.active_tasks:
                print(f"Task Status: IN PROGRESS")
            else:
                print(f"Task Status: UNKNOWN")
        
        # Network statistics
        if self.client_node:
            stats = self.client_node.economic_system.get_network_stats()
            print(f"\nNetwork Statistics:")
            print(f"  Total Tickets: {stats['total_tickets']}")
            print(f"  Active Tasks: {stats['active_tasks']}")
            print(f"  Completed Tasks: {stats['completed_tasks']}")
            print(f"  Current Ticket Price: {stats['current_ticket_price']:.2f} PAI")
        
        # Node reputations
        print(f"\nNode Reputations:")
        for node in self.nodes:
            if node.role in [NodeRole.MINER, NodeRole.SUPERVISOR]:
                rep = node.economic_system.get_node_reputation(node.node_id)
                print(f"  {node.node_id}: Tasks={rep['tasks_completed']}, "
                      f"Rewards={rep['total_rewards']:.2f}, Stake={rep['current_stake']:.2f}")
    
    async def cleanup(self):
        """Cleanup all nodes"""
        self.logger.info("Cleaning up nodes...")
        
        for node in self.nodes:
            await node.stop()
        
        self.logger.info("Cleanup complete")


async def main():
    parser = argparse.ArgumentParser(description='PoUW Complete Demonstration')
    parser.add_argument('--miners', type=int, default=2, help='Number of miner nodes')
    parser.add_argument('--supervisors', type=int, default=1, help='Number of supervisor nodes')
    parser.add_argument('--duration', type=int, default=120, help='Demo duration in seconds')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    demo = PoUWDemo()
    
    try:
        print("Starting PoUW Complete Demonstration")
        print("====================================")
        print(f"Configuration:")
        print(f"  Miners: {args.miners}")
        print(f"  Supervisors: {args.supervisors}")
        print(f"  Duration: {args.duration}s")
        print()
        
        # Setup network
        await demo.setup_network(args.miners, args.supervisors)
        
        # Submit task
        await demo.submit_demo_task()
        
        # Monitor training
        await demo.monitor_training(args.duration)
        
        print("\nDemonstration completed successfully!")
        
    except KeyboardInterrupt:
        print("\nDemonstration interrupted by user")
    except Exception as e:
        print(f"\nDemonstration failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await demo.cleanup()


if __name__ == '__main__':
    asyncio.run(main())
