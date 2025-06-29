#!/usr/bin/env python3
"""
Start a complete PoUW network with multiple nodes.
"""

import asyncio
import argparse
import signal
import sys
import time
from pouw.node import PoUWNode
from pouw.economics.staking import NodeRole  # Import NodeRole from staking module


class NetworkBootstrap:
    """Bootstrap a complete PoUW network"""

    def __init__(self):
        self.nodes = []
        self.running = True

    async def start_network(self, num_miners=3, num_supervisors=2, num_evaluators=1):
        """Start a complete network with all node types"""

        base_port = 8000

        # Start miners
        for i in range(num_miners):
            node = PoUWNode(f"miner_{i}", NodeRole.MINER, "localhost", base_port + i)
            self.nodes.append(node)
            await node.start()

            # Stake and register
            preferences = {
                "model_types": ["mlp", "cnn"],
                "has_gpu": True,
                "max_dataset_size": 1000000,
            }
            node.stake_and_register(100.0, preferences)
            node.is_mining = True

            print(f"Started miner {node.node_id} on port {base_port + i}")

        # Start supervisors
        for i in range(num_supervisors):
            port = base_port + num_miners + i
            node = PoUWNode(f"supervisor_{i}", NodeRole.SUPERVISOR, "localhost", port)
            self.nodes.append(node)
            await node.start()

            preferences = {
                "storage_capacity": 10000000,
                "bandwidth": 1000000,
                "redundancy_scheme": "full_replicas",
            }
            node.stake_and_register(50.0, preferences)

            print(f"Started supervisor {node.node_id} on port {port}")

        # Start evaluators
        for i in range(num_evaluators):
            port = base_port + num_miners + num_supervisors + i
            node = PoUWNode(f"evaluator_{i}", NodeRole.EVALUATOR, "localhost", port)
            self.nodes.append(node)
            await node.start()

            preferences = {
                "evaluation_capacity": 100,
                "specialized_metrics": ["accuracy", "f1_score"],
            }
            node.stake_and_register(30.0, preferences)

            print(f"Started evaluator {node.node_id} on port {port}")

        # Connect all nodes to each other
        await self._connect_network()

        print(f"\nNetwork started with {len(self.nodes)} nodes")
        print("Miners:", [n.node_id for n in self.nodes if n.role == NodeRole.MINER])
        print("Supervisors:", [n.node_id for n in self.nodes if n.role == NodeRole.SUPERVISOR])
        print("Evaluators:", [n.node_id for n in self.nodes if n.role == NodeRole.EVALUATOR])

        return self.nodes

    async def _connect_network(self):
        """Connect all nodes to each other"""
        for i, node in enumerate(self.nodes):
            for j, other_node in enumerate(self.nodes):
                if i != j:
                    success = await node.p2p_node.connect_to_peer(other_node.host, other_node.port)
                    if success:
                        print(f"Connected {node.node_id} -> {other_node.node_id}")
                    await asyncio.sleep(0.1)  # Small delay to avoid overwhelming

    async def submit_test_task(self):
        """Submit a test ML task to the network"""
        # Use the first node as client
        client_node = self.nodes[0]

        task_definition = {
            "model_type": "mlp",
            "architecture": {"input_size": 784, "hidden_sizes": [64, 32], "output_size": 10},
            "optimizer": {"type": "adam", "learning_rate": 0.001},
            "stopping_criterion": {"type": "max_epochs", "max_epochs": 20},
            "validation_strategy": {"type": "holdout", "validation_split": 0.2},
            "metrics": ["accuracy", "loss"],
            "dataset_info": {"format": "MNIST", "batch_size": 32, "size": 1000},
            "performance_requirements": {"expected_training_time": 300, "gpu": False},
        }

        task_id = client_node.submit_ml_task(task_definition, 50.0)
        print(f"\nSubmitted test task: {task_id}")
        return task_id

    async def monitor_network(self):
        """Monitor network activity"""
        while self.running:
            print("\n" + "=" * 60)
            print("NETWORK STATUS")
            print("=" * 60)

            for node in self.nodes:
                status = node.get_status()
                print(
                    f"{node.node_id:12} | Height: {status['blockchain_height']:3} | "
                    f"Peers: {status['peer_count']:2} | Training: {status['is_training']} | "
                    f"Task: {status['current_task'] or 'None'}"
                )

            # Show economic stats
            if self.nodes:
                stats = self.nodes[0].economic_system.get_network_stats()
                print(f"\nEconomic Stats:")
                print(f"  Active Tasks: {stats['active_tasks']}")
                print(f"  Completed Tasks: {stats['completed_tasks']}")
                print(f"  Total Tickets: {stats['total_tickets']}")
                print(f"  Ticket Price: {stats['current_ticket_price']:.2f} PAI")

            await asyncio.sleep(10)

    async def stop_network(self):
        """Stop all nodes"""
        self.running = False
        print("\nStopping network...")

        for node in self.nodes:
            await node.stop()

        print("Network stopped")


async def main():
    parser = argparse.ArgumentParser(description="Bootstrap a complete PoUW network")
    parser.add_argument("--miners", type=int, default=3, help="Number of miner nodes")
    parser.add_argument("--supervisors", type=int, default=2, help="Number of supervisor nodes")
    parser.add_argument("--evaluators", type=int, default=1, help="Number of evaluator nodes")
    parser.add_argument("--submit-task", action="store_true", help="Submit test task after startup")

    args = parser.parse_args()

    network = NetworkBootstrap()

    # Setup graceful shutdown
    def signal_handler(sig, frame):
        print("\nShutting down network...")
        asyncio.create_task(network.stop_network())
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Start network
        await network.start_network(args.miners, args.supervisors, args.evaluators)

        # Wait for network to stabilize
        await asyncio.sleep(5)

        # Submit test task if requested
        if args.submit_task:
            await network.submit_test_task()

        print("\nNetwork is running. Press Ctrl+C to stop")

        # Monitor network
        await network.monitor_network()

    except KeyboardInterrupt:
        await network.stop_network()


if __name__ == "__main__":
    asyncio.run(main())
