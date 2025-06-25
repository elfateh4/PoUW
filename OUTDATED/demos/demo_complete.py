#!/usr/bin/env python3
"""
PoUW System Integration Demo

This script demonstrates the complete Proof of Useful Work blockchain system:
1. Setting up nodes with different roles
2. Submitting ML tasks
3. Mining with useful work
4. Verification and consensus
5. Reward distribution
"""

import asyncio
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

from pouw.blockchain import Blockchain, MLTask, PayForTaskTransaction
from pouw.ml import SimpleMLP, DistributedTrainer, MiniBatch
from pouw.mining import PoUWMiner, PoUWVerifier
from pouw.economics import EconomicSystem, NodeRole
from pouw.network import NetworkMessage


def setup_logging():
    """Setup logging for the demo"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("demo.log"), logging.StreamHandler()],
    )


def create_mnist_task() -> MLTask:
    """Create a sample MNIST-like ML task"""
    return MLTask(
        task_id="mnist_demo_001",
        model_type="mlp",
        architecture={"input_size": 784, "hidden_sizes": [128, 64], "output_size": 10},
        optimizer={"type": "adam", "learning_rate": 0.001, "beta1": 0.9, "beta2": 0.999},
        stopping_criterion={
            "type": "max_epochs",
            "max_epochs": 10,
            "early_stopping": True,
            "patience": 3,
        },
        validation_strategy={"type": "holdout", "validation_split": 0.2},
        metrics=["accuracy", "loss"],
        dataset_info={
            "format": "MNIST",
            "batch_size": 32,
            "training_percent": 0.8,
            "size": 1000,  # Smaller for demo
        },
        performance_requirements={"gpu": False, "min_memory_gb": 2},  # CPU only for demo
        fee=50.0,
        client_id="demo_client",
    )


def generate_synthetic_mnist_batch(batch_size: int = 32) -> MiniBatch:
    """Generate synthetic MNIST-like data for demonstration"""
    data = np.random.randn(batch_size, 784).astype(np.float32) * 0.1
    labels = np.random.randint(0, 10, batch_size)
    return MiniBatch(f"batch_{datetime.now().timestamp()}", data, labels, 0)


class PoUWDemo:
    """Main demo class orchestrating the PoUW system"""

    def __init__(self):
        self.logger = logging.getLogger("PoUWDemo")
        self.blockchain = Blockchain()
        self.economic_system = EconomicSystem()

        # Demo participants
        self.nodes = {}
        self.miners = {}
        self.verifiers = {}

        # Demo state
        self.current_task = None
        self.demo_results = {
            "blocks_mined": 0,
            "transactions_processed": 0,
            "ml_iterations": 0,
            "total_rewards": 0.0,
        }

    def setup_demo_network(self):
        """Setup a demo network with multiple nodes"""
        self.logger.info("Setting up demo network...")

        # Create demo miners with higher omega coefficients for easier mining
        for i in range(3):
            miner_id = f"miner_{i+1:03d}"
            self.miners[miner_id] = PoUWMiner(
                miner_id, omega_b=1e-2, omega_m=1e-2
            )  # Higher coefficients for demo
            self.logger.info(f"Created miner: {miner_id}")

        # Create demo verifiers
        for i in range(2):
            verifier_id = f"verifier_{i+1:03d}"
            self.verifiers[verifier_id] = PoUWVerifier()
            self.logger.info(f"Created verifier: {verifier_id}")

        # Setup economic system with initial stakes
        for miner_id in self.miners:
            success = self.economic_system.buy_ticket(miner_id, NodeRole.MINER, 100.0, {})
            if success:
                self.logger.info(f"Miner {miner_id} bought ticket with 100.0 stake")

        for verifier_id in self.verifiers:
            success = self.economic_system.buy_ticket(verifier_id, NodeRole.VERIFIER, 50.0, {})
            if success:
                self.logger.info(f"Verifier {verifier_id} bought ticket with 50.0 stake")

    def submit_ml_task(self):
        """Submit a machine learning task to the network"""
        self.logger.info("Submitting ML task to network...")

        # Create task
        self.current_task = create_mnist_task()

        # Create payment transaction
        pay_tx = PayForTaskTransaction(
            version=1,
            inputs=[],
            outputs=[{"address": "demo_client", "amount": -50.0}],  # Payment
            task_definition=self.current_task.to_dict(),
            fee=50.0,
        )

        # Add to blockchain
        success = self.blockchain.add_transaction_to_mempool(pay_tx)
        if success:
            self.demo_results["transactions_processed"] += 1
            self.logger.info(f"ML task {self.current_task.task_id} submitted successfully")

            # Register with economic system
            self.economic_system.submit_task(self.current_task)
        else:
            self.logger.error("Failed to submit ML task")

    def simulate_ml_training_and_mining(self, iterations: int = 5):
        """Simulate ML training and mining process"""
        self.logger.info(f"Starting ML training and mining simulation ({iterations} iterations)...")

        if not self.current_task:
            self.logger.error("No active task for training")
            return

        # Create model based on task specification
        arch = self.current_task.architecture
        model = SimpleMLP(arch["input_size"], arch["hidden_sizes"], arch["output_size"])

        # Setup training for each miner
        trainers = {}
        for miner_id in self.miners:
            trainers[miner_id] = DistributedTrainer(
                model=SimpleMLP(arch["input_size"], arch["hidden_sizes"], arch["output_size"]),
                task_id=self.current_task.task_id,
                miner_id=miner_id,
                tau=0.01,
            )

        # Training and mining loop
        for iteration in range(iterations):
            self.logger.info(f"--- Iteration {iteration + 1} ---")

            # Generate training batch
            batch = generate_synthetic_mnist_batch(32)

            # Each miner processes the iteration
            for miner_id, trainer in trainers.items():
                # Setup optimizer and criterion
                optimizer = optim.Adam(trainer.model.parameters(), lr=0.001)
                criterion = nn.CrossEntropyLoss()

                # Process iteration
                message, metrics = trainer.process_iteration(batch, optimizer, criterion)
                self.demo_results["ml_iterations"] += 1

                self.logger.info(
                    f"{miner_id} - Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}"
                )

                # Attempt mining
                miner = self.miners[miner_id]

                # Set easy difficulty for demo
                self.blockchain.difficulty_target = (
                    0x0FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
                )

                # Mine block with more attempts for demo
                batch_size = batch.size()
                model_size = sum(p.numel() for p in trainer.model.parameters())

                result = miner.mine_block(
                    trainer,
                    message,
                    batch_size,
                    model_size,
                    list(self.blockchain.mempool),
                    self.blockchain,
                )

                if result:
                    block, proof = result

                    # Verify with multiple verifiers
                    verification_results = []
                    for verifier_id, verifier in self.verifiers.items():
                        is_valid = verifier.verify_block(block, proof, trainer)
                        verification_results.append(is_valid)
                        self.logger.info(
                            f"{verifier_id} verification: {'VALID' if is_valid else 'INVALID'}"
                        )

                    # Add block if majority verification passes
                    if sum(verification_results) > len(verification_results) // 2:
                        success = self.blockchain.add_block(block)
                        if success:
                            self.demo_results["blocks_mined"] += 1
                            self.logger.info(f"✓ Block mined by {miner_id} and added to chain")

                            # Note: Rewards are handled during final task completion
                            break  # Only one miner wins per iteration
                    else:
                        self.logger.warning(f"Block from {miner_id} failed verification")

            # Simulate peer updates exchange
            self._simulate_peer_updates(trainers)

        self.logger.info(f"Training and mining simulation completed!")

    def _simulate_peer_updates(self, trainers):
        """Simulate exchange of gradient updates between miners"""
        # Collect updates from all trainers
        all_updates = []
        for trainer in trainers.values():
            if trainer.message_history:
                latest_message = trainer.message_history[-1]
                if latest_message.gradient_updates:
                    all_updates.append(latest_message.gradient_updates)

        # Distribute updates to all trainers
        for trainer in trainers.values():
            for update in all_updates:
                if update.miner_id != trainer.miner_id:
                    trainer.add_peer_update(update)

    def complete_task_and_evaluate(self):
        """Complete the ML task and evaluate results"""
        if not self.current_task:
            return

        self.logger.info("Completing ML task and distributing final rewards...")

        # Mark task as complete
        final_models = {
            "best_model": "demo_model_hash"
        }  # In practice, this would be the actual best model
        performance_metrics = {"final_accuracy": 0.85, "final_loss": 0.45}

        rewards = self.economic_system.complete_task(
            self.current_task.task_id, final_models, performance_metrics
        )

        self.demo_results["total_rewards"] = sum(rewards.values())

        self.logger.info(f"Task {self.current_task.task_id} completed successfully")

    def print_demo_results(self):
        """Print comprehensive demo results"""
        print("\n" + "=" * 60)
        print("           PoUW BLOCKCHAIN DEMO RESULTS")
        print("=" * 60)

        print(f"📊 Blockchain Statistics:")
        print(f"   • Chain Length: {self.blockchain.get_chain_length()} blocks")
        print(f"   • Mempool Size: {self.blockchain.get_mempool_size()} transactions")
        print(f"   • Blocks Mined: {self.demo_results['blocks_mined']}")
        print(f"   • Transactions Processed: {self.demo_results['transactions_processed']}")

        print(f"\n🧠 Machine Learning Statistics:")
        print(f"   • ML Iterations: {self.demo_results['ml_iterations']}")
        print(f"   • Active Task: {self.current_task.task_id if self.current_task else 'None'}")

        print(f"\n💰 Economic Statistics:")
        print(f"   • Total Rewards Distributed: {self.demo_results['total_rewards']:.2f}")
        print(f"   • Active Miners: {len(self.miners)}")
        print(f"   • Active Verifiers: {len(self.verifiers)}")

        # Network statistics
        network_stats = self.economic_system.get_network_stats()
        print(f"\n🌐 Network Statistics:")
        print(f"   • Total Tickets: {network_stats['total_tickets']}")
        # Calculate total stake manually
        total_stake = sum(
            ticket.stake_amount for ticket in self.economic_system.stake_pool.tickets.values()
        )
        print(f"   • Total Stake: {total_stake:.2f}")
        print(f"   • Active Tasks: {network_stats['active_tasks']}")
        print(f"   • Completed Tasks: {network_stats['completed_tasks']}")

        print("\n✅ Demo completed successfully!")
        print("=" * 60)


def main():
    """Main demo function"""
    setup_logging()
    logger = logging.getLogger("PoUW Demo")

    print("🚀 Starting PoUW Blockchain Demo...")
    print("This demo showcases the Proof of Useful Work blockchain system.")
    print()

    # Create and run demo
    demo = PoUWDemo()

    try:
        # Phase 1: Setup
        demo.setup_demo_network()

        # Phase 2: Submit task
        demo.submit_ml_task()

        # Phase 3: Training and mining
        demo.simulate_ml_training_and_mining(iterations=3)

        # Phase 4: Complete task
        demo.complete_task_and_evaluate()

        # Phase 5: Results
        demo.print_demo_results()

    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
