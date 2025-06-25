#!/usr/bin/env python3
"""
Enhanced PoUW Demo with Advanced Features.

Demonstrates the complete PoUW system with:
- BLS threshold signatures and DKG
- Gradient poisoning detection
- VRF-based worker selection
- Reed-Solomon data encoding
- Byzantine fault tolerance
- Zero-nonce commitments
- Advanced security features
"""

import asyncio
import time
import logging
import json
from typing import Dict, List
import numpy as np

from pouw.node import PoUWNode
from pouw.economics import NodeRole
from pouw.ml.training import GradientUpdate
from pouw.blockchain.core import MLTask
from pouw.security import AttackType, SecurityAlert
from pouw.advanced import VRFType


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("PoUWAdvancedDemo")


class AdvancedPoUWDemo:
    """Enhanced demo showcasing advanced PoUW features"""

    def __init__(self):
        self.nodes: Dict[str, PoUWNode] = {}
        self.supervisor_nodes: List[str] = []
        self.miner_nodes: List[str] = []
        self.demo_results = {
            "blocks_mined": 0,
            "security_alerts": 0,
            "commitments_created": 0,
            "dkg_completed": False,
            "consensus_transactions": 0,
            "gradient_filtering_events": 0,
        }

    async def setup_network(self):
        """Setup enhanced PoUW network with advanced features"""
        logger.info("🚀 Setting up enhanced PoUW network...")

        # Create supervisor nodes (for DKG and consensus)
        for i in range(3):
            node_id = f"supervisor_{i:03d}"
            node = PoUWNode(node_id, NodeRole.SUPERVISOR, port=8000 + i)
            self.nodes[node_id] = node
            self.supervisor_nodes.append(node_id)
            await node.start()
            logger.info(f"✓ Started supervisor {node_id}")

        # Create miner nodes
        for i in range(5):
            node_id = f"miner_{i:03d}"
            node = PoUWNode(node_id, NodeRole.MINER, port=8010 + i)
            self.nodes[node_id] = node
            self.miner_nodes.append(node_id)
            await node.start()
            logger.info(f"✓ Started miner {node_id}")

        # Create verifier and evaluator nodes
        for role, count in [(NodeRole.VERIFIER, 2), (NodeRole.EVALUATOR, 2)]:
            for i in range(count):
                node_id = f"{role.value}_{i:03d}"
                node = PoUWNode(node_id, role, port=8020 + len(self.nodes))
                self.nodes[node_id] = node
                await node.start()
                logger.info(f"✓ Started {role.value} {node_id}")

        logger.info(f"🌐 Network setup complete: {len(self.nodes)} nodes")

    async def initialize_supervisor_dkg(self):
        """Initialize Distributed Key Generation among supervisors"""
        logger.info("🔐 Initializing Supervisor DKG Protocol...")

        try:
            # Start DKG on all supervisor nodes
            for supervisor_id in self.supervisor_nodes:
                node = self.nodes[supervisor_id]
                await node.initialize_supervisor_network(self.supervisor_nodes)

            # Simulate DKG completion
            await asyncio.sleep(2)

            # Check DKG status
            dkg_successful = 0
            for supervisor_id in self.supervisor_nodes:
                node = self.nodes[supervisor_id]
                if node.dkg and node.dkg.state.value in ["completed", "key_shares_distributed"]:
                    dkg_successful += 1

            if dkg_successful >= 2:  # Majority completed
                self.demo_results["dkg_completed"] = True
                logger.info(
                    f"✅ DKG completed successfully on {dkg_successful}/{len(self.supervisor_nodes)} supervisors"
                )
            else:
                logger.warning(f"⚠️ DKG only completed on {dkg_successful} supervisors")

        except Exception as e:
            logger.error(f"❌ DKG initialization failed: {e}")

    async def stake_and_register_nodes(self):
        """Stake coins and register nodes for participation"""
        logger.info("💰 Staking and registering nodes...")

        staking_config = {
            NodeRole.SUPERVISOR: {"amount": 200.0, "preferences": {"storage_capacity": 1000000}},
            NodeRole.MINER: {
                "amount": 150.0,
                "preferences": {"has_gpu": True, "model_types": ["mlp"]},
            },
            NodeRole.VERIFIER: {"amount": 100.0, "preferences": {"computation_power": "high"}},
            NodeRole.EVALUATOR: {"amount": 120.0, "preferences": {"evaluation_speed": "fast"}},
        }

        for node_id, node in self.nodes.items():
            config = staking_config[node.role]
            try:
                ticket = node.stake_and_register(config["amount"], config["preferences"])
                logger.info(f"✓ {node_id} staked {config['amount']} PAI")
            except Exception as e:
                logger.error(f"❌ Failed to stake {node_id}: {e}")

    async def submit_ml_task_with_advanced_selection(self):
        """Submit ML task using advanced VRF-based worker selection"""
        logger.info("🧠 Submitting ML task with advanced worker selection...")

        # Use first supervisor as client
        client_node = self.nodes[self.supervisor_nodes[0]]

        task_definition = {
            "model_type": "mlp",
            "architecture": {"input_size": 784, "hidden_sizes": [128, 64], "output_size": 10},
            "optimizer": {"type": "adam", "lr": 0.001},
            "stopping_criterion": {"max_iterations": 50},
            "validation_strategy": {"split_ratio": 0.2},
            "metrics": ["accuracy", "loss"],
            "dataset_info": {"name": "mnist", "batch_size": 32},
            "performance_requirements": {"min_stake": 100, "prefer_gpu": True},
        }

        try:
            task_id = client_node.submit_ml_task(task_definition, fee=75.0)
            logger.info(f"✅ ML task submitted: {task_id}")

            # Demonstrate advanced worker selection
            task_obj = (
                client_node.current_task
                if client_node.current_task
                else MLTask(
                    task_id=task_id,
                    model_type=task_definition["model_type"],
                    architecture=task_definition["architecture"],
                    optimizer=task_definition["optimizer"],
                    stopping_criterion=task_definition["stopping_criterion"],
                    validation_strategy=task_definition["validation_strategy"],
                    metrics=task_definition["metrics"],
                    dataset_info=task_definition["dataset_info"],
                    performance_requirements=task_definition["performance_requirements"],
                    fee=75.0,
                    client_id="demo_client",
                )
            )
            selected_workers = await client_node.select_workers_for_task(task_obj)

            logger.info(f"🎯 Advanced worker selection completed:")
            for role, workers in selected_workers.items():
                logger.info(f"  {role.value}: {len(workers)} workers selected")

            return task_id

        except Exception as e:
            logger.error(f"❌ Failed to submit ML task: {e}")
            return None

    async def demonstrate_gradient_poisoning_detection(self):
        """Demonstrate gradient poisoning detection in action"""
        logger.info("🛡️ Demonstrating gradient poisoning detection...")

        # Get a supervisor node for gradient processing
        supervisor_node = self.nodes[self.supervisor_nodes[0]]

        # Create normal gradient updates
        normal_updates = []
        for i, miner_id in enumerate(self.miner_nodes[:3]):
            # Generate normal gradient data using indices and values format
            indices = list(range(20))
            values = [0.1 + np.random.normal(0, 0.01) for _ in range(20)]
            update = GradientUpdate(
                miner_id=miner_id,
                task_id="demo_task",
                iteration=1,
                epoch=1,
                indices=indices,
                values=values,
            )
            normal_updates.append(update)

        # Create poisoned gradient update
        poisoned_indices = list(range(20))
        poisoned_values = [25.0 for _ in range(20)]  # Abnormally large gradients
        poisoned_update = GradientUpdate(
            miner_id="malicious_miner_999",
            task_id="demo_task",
            iteration=1,
            epoch=1,
            indices=poisoned_indices,
            values=poisoned_values,
        )

        all_updates = normal_updates + [poisoned_update]

        try:
            # Process updates through security system
            clean_updates = await supervisor_node.handle_gradient_updates(all_updates)

            alerts_count = len(supervisor_node.security_alerts)
            self.demo_results["security_alerts"] += alerts_count
            self.demo_results["gradient_filtering_events"] += 1

            logger.info(f"🔍 Gradient processing results:")
            logger.info(f"  Original updates: {len(all_updates)}")
            logger.info(f"  Clean updates: {len(clean_updates)}")
            logger.info(f"  Security alerts: {alerts_count}")

            if alerts_count > 0:
                latest_alert = supervisor_node.security_alerts[-1]
                logger.info(
                    f"  Latest alert: {latest_alert.alert_type.value} from {latest_alert.node_id}"
                )

        except Exception as e:
            logger.error(f"❌ Gradient poisoning detection failed: {e}")

    async def demonstrate_zero_nonce_commitments(self):
        """Demonstrate zero-nonce commitment system"""
        logger.info("⏰ Demonstrating zero-nonce commitments...")

        # Use first miner for demonstration
        miner_node = self.nodes[self.miner_nodes[0]]

        try:
            # Create multiple commitments
            for i in range(3):
                model_state = {
                    "weights": [np.random.normal(0, 0.1) for _ in range(10)],
                    "iteration": i,
                    "loss": 0.5 - i * 0.1,
                }

                commitment = miner_node.commitment_system.create_commitment(
                    miner_id=miner_node.node_id,
                    future_iteration=i + 10,
                    model_state=model_state,
                    vrf=miner_node.vrf,
                )

                self.demo_results["commitments_created"] += 1
                logger.info(
                    f"📋 Created commitment {commitment['commitment_id'][:16]}... for iteration {commitment['future_iteration']}"
                )

            # Check pending commitments
            pending = miner_node.commitment_system.get_pending_commitments(miner_node.node_id)
            logger.info(f"✅ {len(pending)} commitments pending for {miner_node.node_id}")

        except Exception as e:
            logger.error(f"❌ Zero-nonce commitment demonstration failed: {e}")

    async def demonstrate_data_management(self):
        """Demonstrate advanced data management features"""
        logger.info("📊 Demonstrating advanced data management...")

        # Use supervisor node for data management
        supervisor_node = self.nodes[self.supervisor_nodes[0]]

        try:
            # Store dataset with Reed-Solomon encoding
            test_dataset = b"Mock ML dataset with features and labels..." * 100
            success = await supervisor_node.store_dataset_securely(
                dataset_id="demo_mnist_dataset",
                data=test_dataset,
                metadata={"description": "Demo MNIST dataset", "samples": 1000},
            )

            if success:
                logger.info(f"💾 Dataset stored securely with Reed-Solomon encoding")

                # Retrieve dataset
                retrieved_data = await supervisor_node.retrieve_dataset("demo_mnist_dataset")
                if retrieved_data and retrieved_data == test_dataset:
                    logger.info(f"✅ Dataset retrieved successfully ({len(retrieved_data)} bytes)")
                else:
                    logger.warning(f"⚠️ Dataset retrieval verification failed")

            # Demonstrate dataset splitting
            sample_data = [
                {"features": np.random.randn(10).tolist(), "label": i % 3} for i in range(100)
            ]
            splits = supervisor_node.dataset_splitter.split_dataset(
                dataset_id="demo_split_dataset",
                data=sample_data,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
            )

            logger.info(f"📈 Dataset split completed:")
            for split_name, split_data in splits.items():
                logger.info(f"  {split_name}: {len(split_data)} samples")

        except Exception as e:
            logger.error(f"❌ Data management demonstration failed: {e}")

    async def run_training_with_security(self, task_id: str):
        """Run training iterations with security monitoring"""
        logger.info("🏃‍♂️ Running training with advanced security monitoring...")

        training_nodes = self.miner_nodes[:3]  # Use 3 miners

        # Start training on selected miners
        for miner_id in training_nodes:
            node = self.nodes[miner_id]
            node.is_training = True

            # Create mock task
            from pouw.blockchain import MLTask

            task = MLTask(
                task_id=task_id or "demo_task_001",
                model_type="mlp",
                architecture={"input_size": 784, "hidden_sizes": [128, 64], "output_size": 10},
                optimizer={"type": "adam", "lr": 0.001},
                stopping_criterion={"max_iterations": 20},
                validation_strategy={"split_ratio": 0.2},
                metrics=["accuracy", "loss"],
                dataset_info={"name": "demo", "batch_size": 32},
                performance_requirements={},
                fee=75.0,
                client_id="demo_client",
            )

            node.current_task = task

            # Start training loop
            asyncio.create_task(node._training_loop(task))

        # Let training run for a while
        logger.info("⏳ Training in progress...")
        await asyncio.sleep(10)

        # Stop training
        for miner_id in training_nodes:
            self.nodes[miner_id].is_training = False

        logger.info("✅ Training phase completed")

    async def demonstrate_byzantine_consensus(self):
        """Demonstrate Byzantine fault tolerant consensus"""
        logger.info("🗳️ Demonstrating Byzantine fault tolerant consensus...")

        try:
            # Create a proposal for supervisor consensus
            proposal_data = {
                "type": "BLACKLIST_NODE",
                "target_node": "malicious_miner_999",
                "reason": "Gradient poisoning detected",
                "evidence_hash": "abc123def456",
                "timestamp": int(time.time()),
            }

            # First supervisor proposes
            proposer = self.nodes[self.supervisor_nodes[0]]
            if proposer.supervisor_consensus:
                proposal_id = proposer.supervisor_consensus.propose_transaction(proposal_data)
                logger.info(f"📝 Proposal created: {proposal_id[:16]}...")

                # Other supervisors vote
                for supervisor_id in self.supervisor_nodes:
                    node = self.nodes[supervisor_id]
                    if node.supervisor_consensus:
                        signature_share = node.supervisor_consensus.sign_transaction(proposal_id)
                        if signature_share:
                            logger.info(f"✍️ {supervisor_id} signed proposal")

                            # Distribute signatures to other supervisors
                            for other_id in self.supervisor_nodes:
                                if other_id != supervisor_id:
                                    other_node = self.nodes[other_id]
                                    if other_node.supervisor_consensus:
                                        other_node.supervisor_consensus.receive_signature_share(
                                            proposal_id, supervisor_id, signature_share
                                        )

                # Check for completed transactions
                completed = proposer.supervisor_consensus.get_completed_transactions()
                if completed:
                    self.demo_results["consensus_transactions"] += len(completed)
                    logger.info(f"✅ Byzantine consensus completed: {len(completed)} transactions")
                else:
                    logger.info("⏳ Consensus still in progress")

        except Exception as e:
            logger.error(f"❌ Byzantine consensus demonstration failed: {e}")

    async def generate_security_report(self):
        """Generate comprehensive security report"""
        logger.info("📋 Generating network security report...")

        # Collect security data from all nodes
        total_alerts = 0
        alert_breakdown = {}
        all_blacklisted = set()

        for node_id, node in self.nodes.items():
            try:
                report = await node.get_network_security_report()
                total_alerts += report.get("total_alerts", 0)

                for alert_type, count in report.get("alert_breakdown", {}).items():
                    alert_breakdown[alert_type] = alert_breakdown.get(alert_type, 0) + count

                blacklisted = report.get("blacklisted_nodes", [])
                all_blacklisted.update(blacklisted)

            except Exception as e:
                logger.error(f"❌ Failed to get security report from {node_id}: {e}")

        logger.info("🔒 Network Security Report:")
        logger.info(f"  Total security alerts: {total_alerts}")
        logger.info(f"  Alert types: {dict(alert_breakdown)}")
        logger.info(f"  Blacklisted nodes: {len(all_blacklisted)}")
        logger.info(f"  Network health: {'Good' if total_alerts < 10 else 'Needs attention'}")

    async def run_demo(self):
        """Run the complete advanced PoUW demonstration"""
        start_time = time.time()

        try:
            logger.info("🎭 Starting Enhanced PoUW Demonstration")
            logger.info("=" * 60)

            # Phase 1: Network setup
            await self.setup_network()
            await asyncio.sleep(1)

            # Phase 2: Advanced cryptographic setup
            await self.initialize_supervisor_dkg()
            await asyncio.sleep(1)

            # Phase 3: Economic participation
            await self.stake_and_register_nodes()
            await asyncio.sleep(1)

            # Phase 4: Advanced task submission
            task_id = await self.submit_ml_task_with_advanced_selection()
            await asyncio.sleep(2)

            # Phase 5: Security demonstrations
            await self.demonstrate_gradient_poisoning_detection()
            await asyncio.sleep(1)

            await self.demonstrate_zero_nonce_commitments()
            await asyncio.sleep(1)

            # Phase 6: Data management
            await self.demonstrate_data_management()
            await asyncio.sleep(1)

            # Phase 7: Training with security
            if task_id:
                await self.run_training_with_security(task_id)
            else:
                logger.warning("⚠️ Skipping training phase - no valid task ID")
            await asyncio.sleep(2)

            # Phase 8: Byzantine consensus
            await self.demonstrate_byzantine_consensus()
            await asyncio.sleep(1)

            # Phase 9: Security reporting
            await self.generate_security_report()

            # Final results
            duration = time.time() - start_time
            logger.info("=" * 60)
            logger.info("🎉 Enhanced PoUW Demonstration Complete!")
            logger.info(f"⏱️ Total duration: {duration:.2f} seconds")
            logger.info("📊 Final Results:")
            logger.info(f"  Nodes deployed: {len(self.nodes)}")
            logger.info(f"  DKG completed: {'✅' if self.demo_results['dkg_completed'] else '❌'}")
            logger.info(f"  Security alerts: {self.demo_results['security_alerts']}")
            logger.info(f"  Commitments created: {self.demo_results['commitments_created']}")
            logger.info(f"  Consensus transactions: {self.demo_results['consensus_transactions']}")
            logger.info(
                f"  Gradient filtering events: {self.demo_results['gradient_filtering_events']}"
            )

            # Node status summary
            logger.info("🏷️ Node Status Summary:")
            for node_id, node in self.nodes.items():
                status = node.get_status()
                logger.info(
                    f"  {node_id}: {status['role']} - Alerts: {status.get('security_alerts_count', 0)}"
                )

        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            raise

        finally:
            # Cleanup
            logger.info("🧹 Cleaning up nodes...")
            for node in self.nodes.values():
                try:
                    await node.stop()
                except Exception as e:
                    logger.error(f"Error stopping node: {e}")


async def main():
    """Main entry point for enhanced demo"""
    demo = AdvancedPoUWDemo()

    try:
        await demo.run_demo()
    except KeyboardInterrupt:
        logger.info("🛑 Demo interrupted by user")
    except Exception as e:
        logger.error(f"❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
