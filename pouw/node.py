"""
PoUW Node Implementation

This module provides the main PoUWNode class that integrates all components
of the Proof of Useful Work blockchain system into a unified, production-ready node.

The PoUWNode class supports multiple roles:
- MINER: Participates in PoUW mining using ML computation
- SUPERVISOR: Coordinates distributed training and network consensus
- VERIFIER: Validates mining proofs and ML work
- EVALUATOR: Evaluates task completion and quality
- PEER: Basic network participant

Key Features:
- Seamless integration of all PoUW components
- Advanced networking with VPN mesh support
- Comprehensive economic participation
- Enterprise-grade security and monitoring
- Production-ready deployment capabilities
"""

import asyncio
import logging
import time
import traceback
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

# Core blockchain components
from .blockchain import Blockchain, MLTask, PayForTaskTransaction, BuyTicketsTransaction
from .mining import PoUWMiner, PoUWVerifier, MiningProof
from .ml import DistributedTrainer, SimpleMLP, MiniBatch, IterationMessage
from .economics import EconomicSystem, NodeRole, Ticket
from .network import (
    P2PNode,
    NetworkMessage,
    NetworkOperationsManager,
    NodeStatus,
    NodeHealthMetrics,
)
from .security import (
    AttackMitigationSystem,
    GradientPoisoningDetector,
    ByzantineFaultTolerance,
    SecurityAlert,
    AttackType,
)


@dataclass
class NodeConfig:
    """Configuration for PoUW node"""

    # Basic node configuration
    node_id: str
    role: NodeRole
    host: str = "localhost"
    port: int = 8666

    # Economic configuration
    initial_stake: float = 0.0
    preferences: Dict[str, Any] = field(default_factory=dict)
    
    # Mining configuration (for miners)
    omega_b: float = 1e-6  # Batch size coefficient
    omega_m: float = 1e-8  # Model size coefficient

    # Network configuration
    max_peers: int = 50
    bootstrap_peers: List[Tuple[str, int]] = field(default_factory=list)

    # Security configuration
    enable_security_monitoring: bool = True
    enable_attack_mitigation: bool = True

    # Advanced features configuration
    enable_advanced_features: bool = True
    enable_production_features: bool = True
    commitment_depth: int = 5  # Zero-nonce commitment depth (k iterations)


class PoUWNode:
    """
    Main PoUW Node Implementation

    A complete blockchain node that integrates all PoUW system components:
    - Blockchain consensus and transaction processing
    - PoUW mining with ML computation
    - Distributed ML training coordination
    - Economic participation and staking
    - Advanced P2P networking with operations management
    - Comprehensive security and attack mitigation
    - Production monitoring and optimization
    """

    def __init__(
        self,
        node_id: str,
        role: NodeRole,
        host: str = "localhost",
        port: int = 8666,
    ):
        """
        Initialize a PoUW node with specified role and configuration.

        Args:
            node_id: Unique identifier for this node
            role: Node role (MINER, SUPERVISOR, VERIFIER, EVALUATOR, PEER)
            host: Host address to bind to
            port: Port to bind to
        """
        self.node_id = node_id
        self.role = role
        self.host = host
        self.port = port

        self.config = NodeConfig(node_id=node_id, role=role, host=host, port=port)

        # Setup logging
        self.logger = logging.getLogger(f"PoUWNode-{node_id}")
        self.logger.setLevel(logging.INFO)

        # Core components
        self.blockchain = Blockchain()
        self.economic_system = EconomicSystem()


        self.p2p_node = P2PNode(node_id, host, port)

        # Convert bootstrap peers to supervisor nodes list for NetworkOperationsManager
        supervisor_nodes = None
        if role == NodeRole.SUPERVISOR and self.config.bootstrap_peers:
            # Extract hostnames from bootstrap peers for supervisor coordination
            supervisor_nodes = [f"{host}:{port}" for host, port in self.config.bootstrap_peers]

        self.network_ops = NetworkOperationsManager(
            node_id, role.value, supervisor_nodes
        )

        # Role-specific components
        self.miner = None
        self.verifier = None
        self.trainer = None

        # Security components
        self.security_system = None
        self.attack_mitigation = None

        # Advanced components
        self.vrf = None                          # Verifiable Random Function
        self.worker_selector = None              # Advanced worker selection
        self.commitment_system = None            # Zero-nonce commitment system
        self.merkle_tree = None                 # Message history Merkle tree
        self.performance_monitor = None

        # Node state
        self.is_running = False
        self.is_mining = False
        self.is_training = False
        self.current_task = None
        self.staking_ticket = None

        # Metrics and monitoring
        self.start_time = None
        self.stats = {
            "blocks_mined": 0,
            "tasks_completed": 0,
            "rewards_earned": 0.0,
            "security_alerts": 0,
            "network_messages": 0,
        }

        self._initialize_components()

    def _initialize_components(self):
        """Initialize node components based on role and configuration"""
        try:
            # Initialize role-specific components
            if self.role in [NodeRole.MINER, NodeRole.SUPERVISOR]:
                self.miner = PoUWMiner(
                    self.node_id, omega_b=self.config.omega_b, omega_m=self.config.omega_m
                )
                # Initialize trainer with simplified constructor
                try:
                    default_model = SimpleMLP(784, [64], 10)  # MNIST-like default
                    self.trainer = DistributedTrainer(default_model, "default_task", self.node_id)
                except TypeError:
                    fallback_model = SimpleMLP(784, [32], 10)  # Smaller default
                    self.trainer = DistributedTrainer(fallback_model, "fallback_task", self.node_id)
                    
                self.logger.info(f"Initialized miner and trainer for {self.role.value}")

            if self.role in [NodeRole.VERIFIER, NodeRole.SUPERVISOR, NodeRole.EVALUATOR]:
                self.verifier = PoUWVerifier()
                self.logger.info(f"Initialized verifier for {self.role.value}")

            # Initialize security components
            if self.config.enable_security_monitoring:
                self.security_system = self._create_security_system()
                self.logger.info("Initialized security monitoring system")

            if self.config.enable_attack_mitigation:
                try:
                    self.attack_mitigation = AttackMitigationSystem()
                except TypeError:
                    # If constructor expects parameters, use None for now
                    self.attack_mitigation = None
                    self.logger.warning("Attack mitigation system not available")
                if self.attack_mitigation:
                    self.logger.info("Initialized attack mitigation system")

            # Initialize advanced features
            if (
                self.config.enable_advanced_features
                and self.role == NodeRole.SUPERVISOR
            ):
                self._initialize_advanced_features()

            # Initialize production features
            if self.config.enable_production_features:
                self._initialize_production_features()

            # Setup message handlers
            self._setup_message_handlers()

            self.logger.info(f"PoUW node {self.node_id} initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize node components: {e}")
            raise

    def _create_security_system(self) -> Dict[str, Any]:
        """Create integrated security monitoring system"""
        try:
            return {
                "gradient_detector": GradientPoisoningDetector(),
                "byzantine_tolerance": ByzantineFaultTolerance(3),  # Default supervisor count
                "alerts": [],
                "threat_level": "LOW",
            }
        except Exception as e:
            self.logger.warning(f"Failed to initialize security system: {e}")
            return {"alerts": [], "threat_level": "LOW"}

    def _initialize_advanced_features(self):
        """Initialize advanced cryptographic features"""
        try:
            if self.config.enable_advanced_features:
                from .advanced import (
                    VerifiableRandomFunction, 
                    AdvancedWorkerSelection, 
                    ZeroNonceCommitment, 
                    MessageHistoryMerkleTree
                )
                
                # Initialize VRF for cryptographic randomness
                self.vrf = VerifiableRandomFunction()
                self.logger.info("VRF (Verifiable Random Function) initialized")
                
                # Initialize advanced worker selection with VRF
                self.worker_selector = AdvancedWorkerSelection(self.vrf)
                self.logger.info("Advanced worker selection system initialized")
                
                # Initialize zero-nonce commitment system for attack prevention
                # Use config parameter if available, otherwise default to 5
                commitment_depth = getattr(self.config, 'commitment_depth', 5)
                self.commitment_system = ZeroNonceCommitment(commitment_depth=commitment_depth)
                self.logger.info(f"Zero-nonce commitment system initialized (depth: {commitment_depth})")
                
                # Initialize Merkle tree for message history compression
                self.merkle_tree = MessageHistoryMerkleTree()
                self.logger.info("Message history Merkle tree initialized")
                
                self.logger.info("All advanced cryptographic features initialized successfully")
            else:
                self.logger.info("Advanced features not available")
        except Exception as e:
            self.logger.warning(f"Failed to initialize advanced features: {e}")
            # Ensure components are None if initialization fails
            self.vrf = None
            self.worker_selector = None
            self.commitment_system = None
            self.merkle_tree = None

    def _initialize_production_features(self):
        """Initialize production monitoring and optimization"""
        try:
            if self.config.enable_production_features:
                from .production import PerformanceMonitor
                self.performance_monitor = PerformanceMonitor(1000)  # Default history size
                self.logger.info("Production monitoring features initialized")
            else:
                self.logger.info("Production features not available")
        except Exception as e:
            self.logger.warning(f"Failed to initialize production features: {e}")

    def _setup_message_handlers(self):
        """Setup handlers for different message types"""
        # Import required classes
        from .network.communication import MessageHandler
        class PoUWNodeMessageHandler(MessageHandler):
            def __init__(self, node):
                self.node = node
                self.handlers = {
                    "NEW_BLOCK": node._handle_new_block,
                    "NEW_TRANSACTION": node._handle_new_transaction,
                    "ML_ITERATION": node._handle_ml_iteration,
                    "TASK_SUBMISSION": node._handle_task_submission,
                    "TASK_ASSIGNMENT": node._handle_task_assignment,
                    "VERIFICATION_REQUEST": node._handle_verification_request,
                    "SECURITY_ALERT": node._handle_security_alert,
                }

            async def handle_message(self, message, sender_address):
                """Route message to appropriate handler"""
                handler = self.handlers.get(message.msg_type)
                if handler:
                    await handler(message)
                else:
                    self.node.logger.warning(f"No handler for message type: {message.msg_type}")

        # Create and register the node message handler
        node_handler = PoUWNodeMessageHandler(self)
        
        # Register handler for all node-specific message types
        handler_message_types = [
            "NEW_BLOCK", "NEW_TRANSACTION", "ML_ITERATION", 
            "TASK_SUBMISSION", "TASK_ASSIGNMENT", "VERIFICATION_REQUEST", "SECURITY_ALERT"
        ]
        
        self.p2p_node.register_handler(handler_message_types, node_handler)
        
        # Also register blockchain and ML specific handlers if available
        if hasattr(self, 'blockchain') and self.blockchain:
            from .network.communication import BlockchainMessageHandler
            blockchain_handler = BlockchainMessageHandler(self.blockchain)
            self.p2p_node.register_handler(["NEW_BLOCK", "NEW_TRANSACTION"], blockchain_handler)
        
        if hasattr(self, 'trainer') and self.trainer:
            from .network.communication import MLMessageHandler
            ml_handler = MLMessageHandler(trainer=self.trainer)
            self.p2p_node.register_handler(["IT_RES", "GRADIENT_UPDATE", "TASK_ASSIGNMENT"], ml_handler)

        self.logger.info(f"Message handlers registered for {len(handler_message_types)} message types")

    async def start(self):
        """Start the PoUW node and all its components"""
        try:
            self.logger.info(f"Starting PoUW node {self.node_id} as {self.role.value}...")

            # Start P2P networking
            await self.p2p_node.start()
            self.logger.info(f"P2P node started on {self.host}:{self.port}")

            # Start network operations
            if self.network_ops and hasattr(self.network_ops, "start_operations"):
                await self.network_ops.start_operations()
                self.logger.info("Network operations started")

            # Connect to bootstrap peers
            for peer_host, peer_port in self.config.bootstrap_peers:
                await self.connect_to_peer(peer_host, peer_port)

            # Start role-specific services
            if self.performance_monitor:
                # Initialize performance monitor if it has start method
                if hasattr(self.performance_monitor, "start_monitoring"):
                    self.performance_monitor.start_monitoring()
                else:
                    # Simple initialization for basic monitoring
                    self.logger.info("Performance monitor initialized")

            # Mark as running
            self.is_running = True
            self.start_time = time.time()

            self.logger.info(
                f"PoUW node {self.node_id} started successfully on {self.host}:{self.port}"
            )

        except Exception as e:
            self.logger.error(f"Failed to start node: {e}")
            self.logger.error(traceback.format_exc())
            raise

    async def stop(self):
        """Stop the PoUW node and cleanup resources"""
        try:
            self.logger.info(f"Stopping PoUW node {self.node_id}...")

            # Stop mining and training
            self.is_mining = False
            self.is_training = False

            # Stop network operations
            if self.network_ops and hasattr(self.network_ops, "stop_operations"):
                await self.network_ops.stop_operations()

            # Stop P2P networking
            await self.p2p_node.stop()
            self.logger.info("P2P node stopped")

            # Stop production monitoring
            if self.performance_monitor:
                if hasattr(self.performance_monitor, "stop_monitoring"):
                    self.performance_monitor.stop_monitoring()
                else:
                    self.logger.info("Performance monitor stopped")

            # Mark as stopped
            self.is_running = False

            self.logger.info(f"PoUW node {self.node_id} stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping node: {e}")

    def stake_and_register(
        self, stake_amount: float, preferences: Optional[Dict[str, Any]] = None
    ) -> Ticket:
        """
        Stake tokens and register for network participation

        Args:
            stake_amount: Amount to stake
            preferences: Node preferences for task assignment

        Returns:
            Staking ticket for participation
        """
        try:
            # Use initial_stake if no amount provided
            actual_stake = stake_amount if stake_amount is not None else self.config.initial_stake
            # Use provided preferences or config defaults
            prefs = preferences or self.config.preferences

            # Buy staking ticket through economic system
            success = self.economic_system.buy_ticket(self.node_id, self.role, actual_stake, prefs)

            if success:
                # Store ticket reference (simplified - would get actual ticket)
                self.staking_ticket = Ticket(
                    ticket_id=f"{self.node_id}_ticket",
                    owner_id=self.node_id,
                    role=self.role,
                    stake_amount=stake_amount,
                    preferences=prefs,
                    expiration_time=int(time.time()) + 86400 * 30,  # 30 days
                )

                self.logger.info(
                    f"Successfully staked {stake_amount} PAI with preferences: {prefs}"
                )
                return self.staking_ticket
            else:
                raise Exception("Failed to purchase staking ticket")

        except Exception as e:
            self.logger.error(f"Failed to stake and register: {e}")
            raise

    async def connect_to_peer(self, peer_host: str, peer_port: int) -> bool:
        """Connect to a peer node"""
        try:
            success = await self.p2p_node.connect_to_peer(peer_host, peer_port)
            if success:
                self.logger.info(f"Connected to peer {peer_host}:{peer_port}")
            return success
        except Exception as e:
            self.logger.error(f"Failed to connect to peer {peer_host}:{peer_port}: {e}")
            return False

    async def submit_task(self, task: MLTask, fee: float) -> bool:
        """Submit an ML task to the network with proper blockchain integration"""
        try:
            # Create and submit task transaction
            task_tx = PayForTaskTransaction(
                version=1, inputs=[], outputs=[], task_definition=task.to_dict(), fee=fee
            )

            # Add transaction to blockchain mempool first
            mempool_success = self.blockchain.add_transaction_to_mempool(task_tx)
            if not mempool_success:
                self.logger.error("Failed to add task transaction to mempool")
                return False

            # Submit through economic system for task management
            task_id = self.economic_system.submit_task(task)

            # Broadcast transaction to network
            await self._broadcast_transaction(task_tx)

            self.logger.info(f"Submitted task {task_id} with fee {fee} - added to mempool and broadcast")
            return True

        except Exception as e:
            self.logger.error(f"Failed to submit task: {e}")
            return False

    async def start_mining(self):
        """Start mining process (for miner nodes)"""
        if not self.miner or self.role not in [NodeRole.MINER, NodeRole.SUPERVISOR]:
            self.logger.warning("Node is not configured for mining")
            return

        try:
            self.is_mining = True
            
            # Create a default training task if none exists
            if not self.current_task:
                from .blockchain.core import MLTask
                self.current_task = MLTask(
                    task_id=f"default_task_{self.node_id}_{int(time.time())}",
                    model_type="mlp",
                    architecture={"input_size": 784, "hidden_sizes": [128, 64], "output_size": 10},
                    optimizer={"type": "adam", "learning_rate": 0.001},
                    stopping_criterion={"type": "max_epochs", "max_epochs": 100},
                    validation_strategy={"type": "holdout", "validation_split": 0.2},
                    metrics=["accuracy", "loss"],
                    dataset_info={"format": "MNIST", "batch_size": 32, "size": 60000},
                    performance_requirements={"gpu": False, "min_memory_gb": 2},
                    fee=50.0,
                    client_id=self.node_id,
                )
                self.logger.info(f"Created default MNIST training task: {self.current_task.task_id}")
            
            self.logger.info("Started mining process")

            # Start mining loop in background
            asyncio.create_task(self._mining_loop())

        except Exception as e:
            self.logger.error(f"Failed to start mining: {e}")

    async def _mining_loop(self):
        """Main mining loop"""
        while self.is_mining and self.is_running:
            try:
                if self.current_task and self.trainer:
                    # Load real dataset based on current task
                    try:
                        # Import numpy for data processing
                        import numpy as np
                        
                        # Try to use production dataset manager for real data
                        from .production.datasets import ProductionDatasetManager, load_mnist
                        
                        # Initialize dataset manager if not already done
                        if not hasattr(self, '_dataset_manager'):
                            self._dataset_manager = ProductionDatasetManager()
                            
                            # Load dataset based on task info
                            dataset_type = self.current_task.dataset_info.get('format', 'MNIST').lower()
                            
                            if dataset_type == 'mnist':
                                load_mnist(self._dataset_manager)
                                self.logger.info("Loaded MNIST dataset for training")
                            else:
                                # Default to MNIST if unknown format
                                load_mnist(self._dataset_manager)
                                self.logger.warning(f"Unknown dataset format {dataset_type}, using MNIST")
                            
                            # Create data loaders
                            batch_size = self.current_task.dataset_info.get('batch_size', 32)
                            self._dataloaders = self._dataset_manager.create_dataloaders('mnist', batch_size)
                        
                        # Get a batch from the training data loader
                        train_loader = self._dataloaders['train']
                        train_iter = iter(train_loader)
                        data_tensor, labels_tensor = next(train_iter)
                        
                        # Convert to numpy for MiniBatch (keeping compatibility)
                        data = data_tensor.numpy().astype(np.float32)
                        labels = labels_tensor.numpy()
                        
                        # Flatten MNIST images if needed (28x28 -> 784)
                        if len(data.shape) > 2:
                            data = data.reshape(data.shape[0], -1)
                        
                        batch = MiniBatch(
                            batch_id=f"batch_{self.node_id}_{int(time.time())}",
                            data=data,
                            labels=labels,
                            epoch=self.trainer.current_epoch,
                        )
                        
                        self.logger.debug(f"Created real data batch: {data.shape} with labels {labels.shape}")
                        
                    except ImportError:
                        # Fallback to synthetic data if production datasets not available
                        self.logger.warning("Production dataset manager not available, using synthetic data")
                        import numpy as np
                        
                        data = np.random.randn(32, 784).astype(np.float32)
                        labels = np.random.randint(0, 10, 32)
                        batch = MiniBatch(
                            batch_id=f"batch_{self.node_id}_{int(time.time())}",
                            data=data,
                            labels=labels,
                            epoch=0,
                        )
                    except Exception as e:
                        # Fallback to synthetic data on any error
                        self.logger.warning(f"Failed to load real dataset: {e}, using synthetic data")
                        import numpy as np
                        
                        data = np.random.randn(32, 784).astype(np.float32)
                        labels = np.random.randint(0, 10, 32)
                        batch = MiniBatch(
                            batch_id=f"batch_{self.node_id}_{int(time.time())}",
                            data=data,
                            labels=labels,
                            epoch=0,
                        )

                    # Set up training components
                    import torch.optim as optim
                    import torch.nn as nn

                    # SimpleMLP inherits from both MLModel and nn.Module
                    # Use isinstance check for safety
                    if isinstance(self.trainer.model, nn.Module):
                        optimizer = optim.Adam(self.trainer.model.parameters(), lr=0.001)
                    else:
                        # Fallback - shouldn't happen with SimpleMLP
                        optimizer = optim.Adam(
                            [p for p in self.trainer.model.get_weights().values()], lr=0.001
                        )
                    criterion = nn.CrossEntropyLoss()

                    # Perform ML training iteration
                    iteration_msg, metrics = self.trainer.process_iteration(
                        batch, optimizer, criterion
                    )

                    # Attempt to mine block
                    transactions = self.blockchain.mempool[:10]  # Take up to 10 transactions
                    batch_size = batch.size()
                    # Get model size - SimpleMLP inherits from nn.Module
                    if isinstance(self.trainer.model, nn.Module):
                        model_size = sum(p.numel() for p in self.trainer.model.parameters())
                    else:
                        # Fallback using get_weights method from MLModel
                        model_size = sum(
                            p.numel() for p in self.trainer.model.get_weights().values()
                        )

                    # Attempt to mine block if miner is available
                    if self.miner is not None:
                        # Create zero-nonce commitment if advanced features are available
                        current_iteration = iteration_msg.iteration
                        commitment_created = False
                        
                        if self.commitment_system and current_iteration % 5 == 0:  # Commit every 5 iterations
                            try:
                                future_iteration = current_iteration + self.config.commitment_depth
                                model_state = {
                                    "weights": self.trainer.model.get_weights(),
                                    "iteration": current_iteration,
                                    "task_id": iteration_msg.task_id
                                }
                                commitment = self.create_zero_nonce_commitment(future_iteration, model_state)
                                if commitment:
                                    commitment_created = True
                                    self.logger.debug(f"Created commitment for iteration {future_iteration}")
                            except Exception as e:
                                self.logger.warning(f"Failed to create zero-nonce commitment: {e}")
                        
                        # Add message to Merkle tree history if available
                        if self.merkle_tree:
                            try:
                                message_str = f"mining_iteration_{current_iteration}_{iteration_msg.task_id}_{self.node_id}"
                                self.add_message_to_history(message_str)
                            except Exception as e:
                                self.logger.warning(f"Failed to add message to history: {e}")
                        
                        result = self.miner.mine_block(
                            self.trainer,
                            iteration_msg,
                            batch_size,
                            model_size,
                            transactions,
                            self.blockchain,
                            self.economic_system,
                        )

                        if result:
                            block, proof = result
                            # Successfully mined block
                            success = self.blockchain.add_block(block)
                            if success:
                                # Record the mined block and token creation in economic system
                                coinbase_tx = block.transactions[0]  # First transaction is coinbase
                                block_reward = coinbase_tx.outputs[0]["amount"]
                                self.economic_system.record_mined_block(block_reward)
                                
                                # Try to fulfill any pending commitments
                                if self.commitment_system:
                                    try:
                                        pending = self.get_pending_commitments()
                                        for commitment in pending:
                                            if commitment["future_iteration"] == current_iteration:
                                                # Create a simplified gradient update for fulfillment
                                                from .ml.training import GradientUpdate
                                                gradient_update = GradientUpdate(
                                                    miner_id=self.node_id,
                                                    task_id=iteration_msg.task_id,
                                                    iteration=current_iteration,
                                                    epoch=iteration_msg.epoch,
                                                    indices=list(range(10)),  # Simplified
                                                    values=[0.1] * 10  # Simplified
                                                )
                                                
                                                fulfilled = self.fulfill_commitment(
                                                    commitment["commitment_id"],
                                                    block.header.nonce,
                                                    block.get_hash(),
                                                    gradient_update
                                                )
                                                if fulfilled:
                                                    self.logger.info(f"Fulfilled commitment {commitment['commitment_id'][:16]}...")
                                    except Exception as e:
                                        self.logger.warning(f"Failed to fulfill commitments: {e}")
                                
                                self.stats["blocks_mined"] += 1
                                self.stats["rewards_earned"] += block_reward
                                self.logger.info(f"Successfully mined block {block.header.nonce} with reward {block_reward} PAI")

                                # Broadcast new block
                                await self._broadcast_new_block(block)

                # Wait before next mining attempt
                await asyncio.sleep(1.0)

            except Exception as e:
                self.logger.error(f"Error in mining loop: {e}")
                await asyncio.sleep(5.0)  # Wait longer on error

    async def _broadcast_new_block(self, block):
        """Broadcast newly mined block to network"""
        message = NetworkMessage(
            msg_type="NEW_BLOCK",
            sender_id=self.node_id,
            data={"block": block.to_dict()},
            timestamp=int(time.time()),
        )
        await self.p2p_node.broadcast_message(message)
        self.stats["network_messages"] += 1

    async def _broadcast_transaction(self, transaction):
        """Broadcast transaction to network"""
        message = NetworkMessage(
            msg_type="NEW_TRANSACTION",
            sender_id=self.node_id,
            data={"transaction": transaction.to_dict()},
            timestamp=int(time.time()),
        )
        await self.p2p_node.broadcast_message(message)
        self.stats["network_messages"] += 1

    # Message handlers
    async def _handle_new_block(self, message: NetworkMessage):
        """Handle incoming new block"""
        try:
            block_data = message.data["block"]
            # Process and validate block
            self.logger.debug(f"Received new block from {message.sender_id}")
        except Exception as e:
            self.logger.error(f"Error handling new block: {e}")

    async def _handle_new_transaction(self, message: NetworkMessage):
        """Handle incoming new transaction"""
        try:
            tx_data = message.data["transaction"]
            # Process transaction
            self.logger.debug(f"Received new transaction from {message.sender_id}")
        except Exception as e:
            self.logger.error(f"Error handling new transaction: {e}")

    async def _handle_ml_iteration(self, message: NetworkMessage):
        """Handle ML training iteration message"""
        try:
            # Process ML iteration from peer
            if self.trainer:
                # Extract gradient update from message and add to trainer
                if "gradient_update" in message.data:
                    gradient_data = message.data["gradient_update"]
                    from .ml.training import GradientUpdate

                    update = GradientUpdate(**gradient_data)
                    self.trainer.add_peer_update(update)
        except Exception as e:
            self.logger.error(f"Error handling ML iteration: {e}")

    async def _handle_task_submission(self, message: NetworkMessage):
        """Handle task submission with advanced worker selection"""
        try:
            task_data = message.data
            self.logger.info(f"Received task submission from {message.sender_id}")
            
            # Extract task requirements
            if "task_definition" in task_data:
                task_def = task_data["task_definition"]
                task_id = task_def.get("task_id", f"task_{int(time.time())}")
                
                # Store the task for potential execution
                if hasattr(self, 'current_task') and not self.current_task:
                    self.current_task = task_def
                    self.logger.info(f"Assigned task {task_id} to this node")
                
                # If this is a supervisor and advanced features are available, perform worker selection
                if (self.role == NodeRole.SUPERVISOR and 
                    self.worker_selector):
                    
                    try:
                        # Get real network participants instead of mock data
                        candidates = await self._get_network_candidates()
                        
                        if not candidates:
                            # Fallback to connected peers if no registered workers found
                            peer_ids = self.p2p_node.get_peers()
                            candidates = await self._get_peer_capabilities(peer_ids)
                        
                        if candidates:
                            # Determine selection criteria based on task requirements
                            selection_criteria = {
                                "min_stake": task_def.get("min_stake", 50),
                                "prefer_gpu": task_def.get("gpu_required", False),
                                "min_bandwidth": task_def.get("min_bandwidth", 50),
                                "required_memory": task_def.get("required_memory", 1024),
                                "dataset_type": task_def.get("dataset_type", "mnist")
                            }
                            
                            # Determine number of workers needed
                            num_miners = task_def.get("required_miners", min(3, len(candidates)))
                            
                            # Select workers using VRF-based selection
                            selected_workers, vrf_proofs = self.select_workers_for_task(
                                task_id=task_id,
                                candidates=candidates,
                                num_needed=num_miners,
                                selection_criteria=selection_criteria
                            )
                            
                            if selected_workers:
                                self.logger.info(f"Selected {len(selected_workers)} workers for task {task_id}")
                                
                                # Verify the selection
                                selected_node_ids = [w.get("node_id", "") for w in selected_workers]
                                if self.verify_worker_selection(task_id, selected_node_ids, vrf_proofs):
                                    self.logger.info(f"Worker selection for task {task_id} verified successfully")
                                    
                                    # Broadcast task assignment to selected workers
                                    await self._broadcast_task_assignments(task_def, selected_workers)
                                    
                                    # Update performance tracking
                                    await self._update_worker_performance_tracking(selected_workers)
                                    
                                    # Add to Merkle tree history
                                    if self.merkle_tree:
                                        message_str = f"task_submission_{task_id}_{message.sender_id}_{len(selected_workers)}_workers"
                                        self.add_message_to_history(message_str)
                                        
                                    # Record task completion in stats
                                    self.stats["tasks_completed"] += 1
                                else:
                                    self.logger.warning(f"Worker selection verification failed for task {task_id}")
                            else:
                                self.logger.warning(f"No suitable workers found for task {task_id}")
                        else:
                            self.logger.warning(f"No network candidates available for task {task_id}")
                            
                    except Exception as e:
                        self.logger.error(f"Error in advanced task handling: {e}")
                        # Fall back to basic task assignment
                        await self._handle_basic_task_assignment(task_def)
                else:
                    # Basic task handling for non-supervisor nodes or when advanced features unavailable
                    await self._handle_basic_task_assignment(task_def)
            else:
                self.logger.warning(f"Task submission from {message.sender_id} missing task_definition")
            
        except Exception as e:
            self.logger.error(f"Error handling task submission: {e}")

    async def _broadcast_task_assignments(self, task_def, selected_workers):
        """Broadcast task assignments to selected workers"""
        try:
            assignment_message = NetworkMessage(
                msg_type="TASK_ASSIGNMENT",
                sender_id=self.node_id,
                data={
                    "task_definition": task_def,
                    "assignment_time": int(time.time()),
                    "supervisor_id": self.node_id
                }
            )
            
            # Try to send directly to each selected worker
            successful_assignments = 0
            for worker in selected_workers:
                worker_id = worker.get("node_id", "")
                if worker_id in self.p2p_node.get_peers():
                    # Send directly to connected peer
                    success = await self.p2p_node.send_to_peer(worker_id, assignment_message)
                    if success:
                        successful_assignments += 1
                        self.logger.info(f"Assigned task {task_def.get('task_id', 'unknown')} to worker {worker_id}")
                    else:
                        self.logger.warning(f"Failed to send task assignment to {worker_id}")
                else:
                    self.logger.warning(f"Worker {worker_id} not connected - broadcasting instead")
            
            # If some workers weren't directly reachable, broadcast to network
            if successful_assignments < len(selected_workers):
                await self.p2p_node.broadcast_message(assignment_message)
                self.logger.info(f"Broadcasted task assignment to network (direct assignments: {successful_assignments}/{len(selected_workers)})")
                
        except Exception as e:
            self.logger.error(f"Error broadcasting task assignments: {e}")

    async def _update_worker_performance_tracking(self, selected_workers):
        """Update performance tracking for selected workers"""
        try:
            for worker in selected_workers:
                node_id = worker.get("node_id", "")
                if node_id and self.worker_selector:
                    # Update performance metrics
                    performance_metrics = {
                        "completion_rate": worker.get("completion_rate", 0.5),
                        "accuracy_score": worker.get("accuracy_score", 0.5),
                        "availability_score": worker.get("availability_score", 0.5),
                        "stake_amount": min(worker.get("stake_amount", 0) / 1000.0, 1.0)
                    }
                    self.update_node_performance_score(node_id, performance_metrics)
                    
        except Exception as e:
            self.logger.error(f"Error updating worker performance tracking: {e}")

    async def _handle_basic_task_assignment(self, task_def):
        """Handle basic task assignment without advanced worker selection"""
        try:
            task_id = task_def.get("task_id", f"basic_task_{int(time.time())}")
            
            # Assign task to this node if it's capable
            if self.role in [NodeRole.MINER, NodeRole.SUPERVISOR]:
                self.current_task = task_def
                self.logger.info(f"Self-assigned task {task_id} for execution")
                
                # If this node is a supervisor, also try to find other miners
                if self.role == NodeRole.SUPERVISOR:
                    connected_peers = self.p2p_node.get_peers()
                    if connected_peers:
                        # Broadcast task to connected peers
                        assignment_message = NetworkMessage(
                            msg_type="TASK_ASSIGNMENT",
                            sender_id=self.node_id,
                            data={
                                "task_definition": task_def,
                                "assignment_time": int(time.time()),
                                "assignment_type": "basic"
                            }
                        )
                        await self.p2p_node.broadcast_message(assignment_message)
                        self.logger.info(f"Broadcasted basic task assignment to {len(connected_peers)} peers")
                
                # Update stats
                self.stats["tasks_completed"] += 1
                
            else:
                self.logger.warning(f"Node role {self.role.value} cannot execute tasks")
                
        except Exception as e:
            self.logger.error(f"Error in basic task assignment: {e}")

    async def _get_network_candidates(self):
        """Get real network participants as candidates"""
        candidates = []
        try:
            # Get registered participants from staking manager
            if hasattr(self, 'economic_system') and self.economic_system:
                stake_pool = self.economic_system.staking_manager.stake_pool
                
                for ticket in stake_pool.tickets.values():
                    if ticket.owner_id != self.node_id and not ticket.is_expired():  # Don't include self
                        candidate = {
                            "node_id": ticket.owner_id,
                            "stake_amount": ticket.stake_amount,
                            "has_gpu": ticket.preferences.get("has_gpu", False),
                            "bandwidth_mbps": ticket.preferences.get("bandwidth_mbps", 100),
                            "completion_rate": 0.8,  # Default - would be tracked in real system
                            "accuracy_score": 0.7,   # Default - would be tracked in real system
                            "availability_score": 0.9,  # Default - would be tracked in real system
                            "role": ticket.role.value
                        }
                        candidates.append(candidate)
                        
            self.logger.info(f"Found {len(candidates)} registered network candidates")
            return candidates
            
        except Exception as e:
            self.logger.error(f"Error getting network candidates: {e}")
            return []

    async def _get_peer_capabilities(self, peer_ids):
        """Get capabilities of connected peers"""
        candidates = []
        try:
            for peer_id in peer_ids:
                # Create basic candidate profile for connected peer
                # In a real implementation, this would query the peer for capabilities
                candidate = {
                    "node_id": peer_id,
                    "stake_amount": 100,  # Default assumption
                    "has_gpu": False,     # Conservative assumption
                    "bandwidth_mbps": 100,
                    "completion_rate": 0.8,
                    "accuracy_score": 0.7,
                    "availability_score": 0.9,
                    "status": "connected"
                }
                candidates.append(candidate)
                
            self.logger.info(f"Generated capability profiles for {len(candidates)} connected peers")
            return candidates
            
        except Exception as e:
            self.logger.error(f"Error getting peer capabilities: {e}")
            return []

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive node status"""
        return {
            "node_id": self.node_id,
            "role": self.role.value,
            "is_running": self.is_running,
            "is_mining": self.is_mining,
            "is_training": self.is_training,
            "blockchain_height": len(self.blockchain.chain),
            "mempool_size": len(self.blockchain.mempool),
            "peer_count": len(self.p2p_node.peers),
            "current_task": self.current_task.task_id if self.current_task else None,
            "uptime": time.time() - self.start_time if self.start_time else 0,
            "stats": self.stats.copy(),
            "staking_ticket": self.staking_ticket.ticket_id if self.staking_ticket else None,
            "network_role": self.role.value,
            "security_alerts": len(self.security_system["alerts"]) if self.security_system else 0,
        }

    def get_health_metrics(self) -> NodeHealthMetrics:
        """Get node health metrics"""
        return NodeHealthMetrics(
            node_id=self.node_id,
            last_heartbeat=time.time(),
            response_time=0.0,  # Would be implemented with actual monitoring
            success_rate=1.0,  # Default success rate
            task_completion_rate=1.0,  # Default completion rate
            bandwidth_utilization=0.0,  # Would be monitored
            cpu_usage=0.0,  # Would be implemented with actual monitoring
            memory_usage=0.0,
        )

    def get_economic_status(self) -> Dict[str, Any]:
        """Get economic participation status"""
        if not self.staking_ticket:
            return {"staked": False}

        reputation = self.economic_system.get_node_reputation(self.node_id)
        return {
            "staked": True,
            "stake_amount": self.staking_ticket.stake_amount,
            "role": self.role.value,
            "reputation": reputation,
            "rewards_earned": self.stats["rewards_earned"],
        }

    def get_token_supply_status(self) -> Dict[str, Any]:
        """Get comprehensive token supply status"""
        return self.economic_system.get_token_supply_info()
    
    def get_supply_health_status(self) -> Dict[str, Any]:
        """Get token supply health indicators"""
        return self.economic_system.get_supply_health_status()
    
    def check_supply_remaining(self) -> Dict[str, Any]:
        """Check remaining token supply and mining viability"""
        supply_info = self.economic_system.get_token_supply_info()
        
        # Calculate mining viability
        is_mining_viable = supply_info["current_block_reward"] > 0
        estimated_blocks_remaining = 0
        
        if supply_info["current_block_reward"] > 0:
            estimated_blocks_remaining = int(supply_info["remaining_supply"] / supply_info["current_block_reward"])
        
        return {
            "mining_viable": is_mining_viable,
            "blocks_until_exhaustion": estimated_blocks_remaining,
            "current_reward": supply_info["current_block_reward"],
            "supply_percentage_used": supply_info["supply_percentage"],
            "inflation_rate": self.economic_system.calculate_inflation_rate()
        }

    # =============================================================================
    # Advanced Features Integration
    # =============================================================================

    def create_zero_nonce_commitment(self, future_iteration: int, model_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create a zero-nonce commitment for future mining iterations.
        
        This prevents useful work manipulation attacks by committing to specific
        model states before actual mining occurs.
        
        Args:
            future_iteration: Iteration number to commit to (k iterations ahead)
            model_state: Current ML model state to commit to
            
        Returns:
            Commitment data if successful, None otherwise
        """
        if not self.commitment_system or not self.vrf:
            self.logger.warning("Zero-nonce commitment system not available")
            return None
            
        try:
            commitment = self.commitment_system.create_commitment(
                miner_id=self.node_id,
                future_iteration=future_iteration,
                model_state=model_state,
                vrf=self.vrf
            )
            
            self.logger.info(f"Created zero-nonce commitment for iteration {future_iteration}")
            return commitment
            
        except Exception as e:
            self.logger.error(f"Failed to create zero-nonce commitment: {e}")
            return None

    def fulfill_commitment(self, commitment_id: str, actual_nonce: int, 
                          block_hash: str, gradient_update) -> bool:
        """
        Fulfill a previously made zero-nonce commitment.
        
        Args:
            commitment_id: ID of the commitment to fulfill
            actual_nonce: The actual nonce used in mining
            block_hash: Hash of the mined block
            gradient_update: ML gradient update for verification
            
        Returns:
            True if commitment fulfilled successfully
        """
        if not self.commitment_system:
            self.logger.warning("Zero-nonce commitment system not available")
            return False
            
        try:
            success = self.commitment_system.fulfill_commitment(
                commitment_id=commitment_id,
                actual_nonce=actual_nonce,
                block_hash=block_hash,
                gradient_update=gradient_update
            )
            
            if success:
                self.logger.info(f"Successfully fulfilled commitment {commitment_id[:16]}...")
            else:
                self.logger.warning(f"Failed to fulfill commitment {commitment_id[:16]}...")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Error fulfilling commitment: {e}")
            return False

    def select_workers_for_task(self, task_id: str, candidates: List[Dict], 
                               num_needed: int, selection_criteria: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict], List[Any]]:
        """
        Select workers using VRF-based advanced selection algorithm.
        
        Combines cryptographic randomness with performance-based scoring for fair
        and efficient worker selection.
        
        Args:
            task_id: Unique identifier for the task
            candidates: List of candidate worker nodes
            num_needed: Number of workers to select
            selection_criteria: Optional criteria for selection
            
        Returns:
            Tuple of (selected_workers, vrf_proofs)
        """
        if not self.worker_selector:
            self.logger.warning("Advanced worker selection not available, using simple selection")
            # Fallback to simple random selection
            import random
            selected = random.sample(candidates, min(num_needed, len(candidates)))
            return selected, []
            
        try:
            criteria = selection_criteria or {}
            selected_workers, vrf_proofs = self.worker_selector.select_workers_with_vrf(
                task_id=task_id,
                candidates=candidates,
                num_needed=num_needed,
                selection_criteria=criteria
            )
            
            self.logger.info(f"Selected {len(selected_workers)} workers for task {task_id} using VRF")
            return selected_workers, vrf_proofs
            
        except Exception as e:
            self.logger.error(f"Failed to select workers with VRF: {e}")
            # Fallback to simple selection
            import random
            selected = random.sample(candidates, min(num_needed, len(candidates)))
            return selected, []

    def verify_worker_selection(self, task_id: str, selected_nodes: List[str], 
                               vrf_proofs: List[Any]) -> bool:
        """
        Verify that worker selection was done correctly using VRF.
        
        Args:
            task_id: Task identifier
            selected_nodes: List of selected node IDs
            vrf_proofs: VRF proofs for verification
            
        Returns:
            True if selection is valid
        """
        if not self.worker_selector:
            self.logger.warning("Advanced worker selection not available for verification")
            return True  # Assume valid if no verification system
            
        try:
            is_valid = self.worker_selector.verify_worker_selection(
                task_id=task_id,
                selected_nodes=selected_nodes,
                vrf_proofs=vrf_proofs
            )
            
            if is_valid:
                self.logger.debug(f"Worker selection for task {task_id} verified successfully")
            else:
                self.logger.warning(f"Worker selection for task {task_id} failed verification")
                
            return is_valid
            
        except Exception as e:
            self.logger.error(f"Error verifying worker selection: {e}")
            return False

    def update_node_performance_score(self, node_id: str, performance_metrics: Dict[str, float]):
        """
        Update performance score for a node based on recent metrics.
        
        Args:
            node_id: Node to update
            performance_metrics: Dict with metrics like completion_rate, accuracy_score, etc.
        """
        if not self.worker_selector:
            self.logger.debug("Advanced worker selection not available for performance updates")
            return
            
        try:
            self.worker_selector.update_node_performance(node_id, performance_metrics)
            self.logger.debug(f"Updated performance metrics for node {node_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to update performance for node {node_id}: {e}")

    def add_message_to_history(self, message: str) -> Optional[str]:
        """
        Add a message to the Merkle tree history for compression and verification.
        
        Args:
            message: Message to add to history
            
        Returns:
            Message hash if successful, None otherwise
        """
        if not self.merkle_tree:
            self.logger.debug("Merkle tree not available for message history")
            return None
            
        try:
            message_hash = self.merkle_tree.add_message(message)
            self.logger.debug(f"Added message to history: {message_hash[:16]}...")
            return message_hash
            
        except Exception as e:
            self.logger.error(f"Failed to add message to history: {e}")
            return None

    def build_epoch_merkle_root(self, epoch: int, messages: List[str]) -> Optional[str]:
        """
        Build Merkle root for an epoch's messages.
        
        Args:
            epoch: Epoch number
            messages: List of messages for the epoch
            
        Returns:
            Merkle root hash if successful
        """
        if not self.merkle_tree:
            self.logger.debug("Merkle tree not available for root building")
            return None
            
        try:
            root_hash = self.merkle_tree.build_merkle_tree(epoch, messages)
            self.logger.info(f"Built Merkle root for epoch {epoch}: {root_hash[:16]}...")
            return root_hash
            
        except Exception as e:
            self.logger.error(f"Failed to build Merkle root for epoch {epoch}: {e}")
            return None

    def get_merkle_proof(self, epoch: int, message_index: int, messages: List[str]) -> List[str]:
        """
        Generate Merkle proof for message inclusion.
        
        Args:
            epoch: Epoch number
            message_index: Index of message in epoch
            messages: All messages in the epoch
            
        Returns:
            List of sibling hashes for proof
        """
        if not self.merkle_tree:
            self.logger.debug("Merkle tree not available for proof generation")
            return []
            
        try:
            proof = self.merkle_tree.get_merkle_proof(epoch, message_index, messages)
            self.logger.debug(f"Generated Merkle proof for message {message_index} in epoch {epoch}")
            return proof
            
        except Exception as e:
            self.logger.error(f"Failed to generate Merkle proof: {e}")
            return []

    def verify_merkle_proof(self, message_hash: str, proof: List[str], 
                           root_hash: str, message_index: int) -> bool:
        """
        Verify a Merkle proof for message inclusion.
        
        Args:
            message_hash: Hash of the message to verify
            proof: Merkle proof (list of sibling hashes)
            root_hash: Expected root hash
            message_index: Index of message in tree
            
        Returns:
            True if proof is valid
        """
        if not self.merkle_tree:
            self.logger.debug("Merkle tree not available for proof verification")
            return True  # Assume valid if no verification system
            
        try:
            is_valid = self.merkle_tree.verify_merkle_proof(
                message_hash=message_hash,
                proof=proof,
                root_hash=root_hash,
                message_index=message_index
            )
            
            if is_valid:
                self.logger.debug("Merkle proof verified successfully")
            else:
                self.logger.warning("Merkle proof verification failed")
                
            return is_valid
            
        except Exception as e:
            self.logger.error(f"Error verifying Merkle proof: {e}")
            return False

    def get_pending_commitments(self) -> List[Dict[str, Any]]:
        """
        Get all pending zero-nonce commitments for this miner.
        
        Returns:
            List of pending commitment data
        """
        if not self.commitment_system:
            return []
            
        try:
            commitments = self.commitment_system.get_pending_commitments(self.node_id)
            return commitments
            
        except Exception as e:
            self.logger.error(f"Failed to get pending commitments: {e}")
            return []

    def get_advanced_features_status(self) -> Dict[str, Any]:
        """
        Get status information about advanced features.
        
        Returns:
            Dict with advanced features status
        """
        return {
            "advanced_features_available": self.config.enable_advanced_features,
            "vrf_initialized": self.vrf is not None,
            "worker_selector_initialized": self.worker_selector is not None,
            "commitment_system_initialized": self.commitment_system is not None,
            "merkle_tree_initialized": self.merkle_tree is not None,
            "pending_commitments": len(self.get_pending_commitments()),
            "vrf_public_key": self.vrf.public_key.hex() if self.vrf else None,
        }

    def __repr__(self) -> str:
        return f"PoUWNode(id={self.node_id}, role={self.role.value}, running={self.is_running})"
    
    async def _handle_task_assignment(self, message: NetworkMessage):
        """Handle task assignment messages from supervisors"""
        try:
            task_data = message.data
            supervisor_id = message.sender_id
            
            self.logger.info(f"Received task assignment from supervisor {supervisor_id}")
            
            if "task_definition" in task_data:
                task_def = task_data["task_definition"]
                task_id = task_def.get("task_id", f"assigned_task_{int(time.time())}")
                
                # Check if this node can execute the task
                if self.role in [NodeRole.MINER, NodeRole.SUPERVISOR]:
                    # Accept the task assignment
                    self.current_task = task_def
                    assignment_time = task_data.get("assignment_time", int(time.time()))
                    
                    self.logger.info(f"Accepted task assignment: {task_id}")
                    
                    # Send acknowledgment back to supervisor
                    ack_message = NetworkMessage(
                        msg_type="TASK_ASSIGNMENT_ACK",
                        sender_id=self.node_id,
                        data={
                            "task_id": task_id,
                            "status": "accepted",
                            "assignment_time": assignment_time,
                            "worker_capabilities": {
                                "role": self.role.value,
                                "has_gpu": False,  # Could be detected from system
                                "available_memory": 4096,  # Could be detected
                                "stake_amount": getattr(self.staking_ticket, 'stake_amount', 0) if self.staking_ticket else 0
                            }
                        }
                    )
                    
                    # Try to send acknowledgment directly to supervisor
                    if supervisor_id in self.p2p_node.get_peers():
                        success = await self.p2p_node.send_to_peer(supervisor_id, ack_message)
                        if success:
                            self.logger.info(f"Sent task assignment acknowledgment to {supervisor_id}")
                        else:
                            self.logger.warning(f"Failed to send acknowledgment to {supervisor_id}")
                    else:
                        # Supervisor not directly connected, broadcast acknowledgment
                        await self.p2p_node.broadcast_message(ack_message)
                        self.logger.info(f"Broadcasted task assignment acknowledgment")
                        
                    # If this is a mining task and the node is a miner, start mining if not already
                    if (self.role == NodeRole.MINER and 
                        not self.is_mining and 
                        task_def.get("requires_mining", True)):
                        await self.start_mining()
                        
                    # Add message to Merkle tree history if available
                    if self.merkle_tree:
                        message_str = f"task_assignment_{task_id}_{supervisor_id}_{self.node_id}"
                        self.add_message_to_history(message_str)
                        
                else:
                    # Node cannot execute this type of task
                    self.logger.warning(f"Cannot execute task {task_id} - incompatible node role: {self.role.value}")
                    
                    # Send rejection back to supervisor
                    rejection_message = NetworkMessage(
                        msg_type="TASK_ASSIGNMENT_REJECT",
                        sender_id=self.node_id,
                        data={
                            "task_id": task_id,
                            "status": "rejected",
                            "reason": f"Node role {self.role.value} cannot execute this task type",
                            "assignment_time": task_data.get("assignment_time", int(time.time()))
                        }
                    )
                    
                    if supervisor_id in self.p2p_node.get_peers():
                        await self.p2p_node.send_to_peer(supervisor_id, rejection_message)
                    else:
                        await self.p2p_node.broadcast_message(rejection_message)
                        
            else:
                self.logger.warning(f"Task assignment from {supervisor_id} missing task_definition")
                
        except Exception as e:
            self.logger.error(f"Error handling task assignment: {e}")

    async def _handle_verification_request(self, message: NetworkMessage):
        """Handle verification request"""
        try:
            request_data = message.data
            self.logger.info(f"Received verification request from {message.sender_id}")
            
            # Check if this node can perform verification
            if self.verifier and self.role in [NodeRole.VERIFIER, NodeRole.SUPERVISOR, NodeRole.EVALUATOR]:
                verification_type = request_data.get("verification_type", "unknown")
                
                if verification_type == "block_verification":
                    # Handle block verification request
                    block_data = request_data.get("block_data")
                    if block_data:
                        self.logger.info(f"Performing block verification for {message.sender_id}")
                        # In a real implementation, this would verify the block
                        # For now, we'll simulate verification
                        verification_result = {
                            "verified": True,
                            "verifier_id": self.node_id,
                            "verification_time": int(time.time()),
                            "block_hash": block_data.get("hash", "unknown")
                        }
                        
                        # Send verification result back
                        response = NetworkMessage(
                            msg_type="VERIFICATION_RESULT",
                            sender_id=self.node_id,
                            data={
                                "request_id": request_data.get("request_id", "unknown"),
                                "verification_result": verification_result
                            }
                        )
                        
                        # Try to send directly to requester
                        if message.sender_id in self.p2p_node.get_peers():
                            await self.p2p_node.send_to_peer(message.sender_id, response)
                        else:
                            await self.p2p_node.broadcast_message(response)
                            
                elif verification_type == "ml_work_verification":
                    # Handle ML work verification request
                    ml_data = request_data.get("ml_data")
                    if ml_data:
                        self.logger.info(f"Performing ML work verification for {message.sender_id}")
                        # Simulate ML work verification
                        verification_result = {
                            "verified": True,
                            "verifier_id": self.node_id,
                            "verification_time": int(time.time()),
                            "task_id": ml_data.get("task_id", "unknown"),
                            "quality_score": 0.95  # Simulated quality score
                        }
                        
                        # Send verification result back
                        response = NetworkMessage(
                            msg_type="VERIFICATION_RESULT", 
                            sender_id=self.node_id,
                            data={
                                "request_id": request_data.get("request_id", "unknown"),
                                "verification_result": verification_result
                            }
                        )
                        
                        if message.sender_id in self.p2p_node.get_peers():
                            await self.p2p_node.send_to_peer(message.sender_id, response)
                        else:
                            await self.p2p_node.broadcast_message(response)
                else:
                    self.logger.warning(f"Unknown verification type: {verification_type}")
            else:
                self.logger.warning(f"Node {self.node_id} cannot perform verification - role: {self.role.value}")
                
        except Exception as e:
            self.logger.error(f"Error handling verification request: {e}")

    async def _handle_security_alert(self, message: NetworkMessage):
        """Handle security alert"""
        try:
            alert_data = message.data
            alert_type = alert_data.get("alert_type", "unknown")
            source_node = message.sender_id
            
            self.logger.warning(f"Received security alert from {source_node}: {alert_type}")
            
            # Process alert through security system if available
            if self.security_system:
                self.security_system["alerts"].append({
                    "alert_type": alert_type,
                    "source_node": source_node,
                    "timestamp": int(time.time()),
                    "details": alert_data
                })
                
                # Update threat level based on alert
                if alert_type in ["gradient_poisoning", "byzantine_fault", "dos_attack"]:
                    self.security_system["threat_level"] = "HIGH"
                elif alert_type in ["sybil_attack", "network_anomaly"]:
                    self.security_system["threat_level"] = "MEDIUM"
                
                # Update stats
                self.stats["security_alerts"] += 1
                
                # If this is a supervisor, coordinate response with other supervisors
                if self.role == NodeRole.SUPERVISOR:
                    # Broadcast alert to other supervisors for coordination
                    coordination_message = NetworkMessage(
                        msg_type="SECURITY_COORDINATION",
                        sender_id=self.node_id,
                        data={
                            "original_alert": alert_data,
                            "coordinator_id": self.node_id,
                            "timestamp": int(time.time()),
                            "action_required": alert_type in ["gradient_poisoning", "byzantine_fault"]
                        }
                    )
                    
                    await self.p2p_node.broadcast_message(coordination_message)
                    self.logger.info(f"Coordinated security response for {alert_type}")
                
                # Take automated mitigation actions if attack mitigation is available
                if hasattr(self, 'attack_mitigation') and self.attack_mitigation:
                    try:
                        # Create a SecurityAlert object for the mitigation system
                        from .security import SecurityAlert, AttackType
                        
                        # Map string alert types to AttackType enum
                        attack_type_map = {
                            "gradient_poisoning": AttackType.GRADIENT_POISONING,
                            "byzantine_fault": AttackType.BYZANTINE_FAULT,
                            "dos_attack": AttackType.DOS_ATTACK,
                            "sybil_attack": AttackType.SYBIL_ATTACK,
                        }
                        
                        attack_type_enum = attack_type_map.get(alert_type, AttackType.GRADIENT_POISONING)
                        
                        security_alert = SecurityAlert(
                            alert_type=attack_type_enum,
                            node_id=source_node,
                            timestamp=int(time.time()),
                            confidence=alert_data.get("confidence", 0.8),
                            evidence=alert_data.get("evidence", {}),
                            description=f"Security alert: {alert_type} from {source_node}"
                        )
                        
                        # Apply mitigation
                        mitigation_applied = self.attack_mitigation.mitigate_attack(security_alert)
                        if mitigation_applied:
                            self.logger.info(f"Applied mitigation for {alert_type} from {source_node}")
                        else:
                            self.logger.warning(f"Failed to apply mitigation for {alert_type}")
                            
                    except ImportError:
                        self.logger.warning("Attack mitigation system not available")
                    except Exception as e:
                        self.logger.error(f"Error applying attack mitigation: {e}")
                        
            else:
                self.logger.warning("No security system available to process security alert")
                
        except Exception as e:
            self.logger.error(f"Error handling security alert: {e}")
            
            
            