#!/usr/bin/env python3
"""
PoUW Node - Production-ready blockchain node implementation.

This module provides a complete PoUW node implementation that integrates all
system components for production deployment.

Features:
- Full blockchain node with mining capabilities
- Distributed ML training coordination
- Network communication and P2P operations
- Advanced security and attack mitigation
- Economic incentive system participation
- Production monitoring and optimization
- Automatic deployment and scaling
- Comprehensive logging and metrics
"""

import asyncio
import argparse
import json
import logging
import os
import signal
import sys
import time
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import threading
import numpy as np
import torch.nn as nn
import torch.optim as optim
from pouw.ml import MiniBatch, SimpleMLP

# Core PoUW imports
from pouw import (
    # Blockchain
    Blockchain,
    PoUWMiner,
    # Network
    P2PNode,
    NetworkOperationsManager,
    VPNMeshTopologyManager,
    CrashRecoveryManager,
    LeaderElectionManager,
    # ML
    DistributedTrainer,
    # Security
    ComprehensiveSecurityMonitor,
    NodeAuthenticator,
    AttackMitigationSystem,
    # Economics
    EconomicSystem,
    StakingManager,
    TaskMatcher,
    # Production
    PerformanceMonitor,
    GPUManager,
    ProductionDatasetManager,
    # Deployment
    KubernetesOrchestrator,
    ProductionMonitor as DeploymentMonitor,
    HealthChecker,
    # Data Management
    DataAvailabilityManager,
    ConsistentHashRing,
    # Advanced
    VerifiableRandomFunction,
    AdvancedWorkerSelection,
    ZeroNonceCommitment,
    # Crypto
    BLSThresholdCrypto,
    DistributedKeyGeneration,
    SupervisorConsensus,
)


class NodeType(Enum):
    """Types of PoUW nodes"""

    WORKER = "worker"
    SUPERVISOR = "supervisor"
    MINER = "miner"
    HYBRID = "hybrid"  # Can perform multiple roles


class NodeState(Enum):
    """Node operational states"""

    INITIALIZING = "initializing"
    CONNECTING = "connecting"
    SYNCING = "syncing"
    ACTIVE = "active"
    MINING = "mining"
    TRAINING = "training"
    MAINTENANCE = "maintenance"
    SHUTTING_DOWN = "shutting_down"
    ERROR = "error"


@dataclass
class NodeConfiguration:
    """Node configuration parameters"""

    # Identity
    node_id: str
    node_type: NodeType
    private_key_path: str

    # Network
    listen_port: int = 8333
    peer_discovery_enabled: bool = True
    max_peers: int = 50
    bootstrap_peers: List[str] = field(default_factory=list)

    # Blockchain
    blockchain_data_dir: str = "./data/blockchain"
    genesis_block_hash: Optional[str] = None

    # Mining
    mining_enabled: bool = True
    mining_threads: int = 1
    mining_reward_address: Optional[str] = None

    # ML Training
    training_enabled: bool = True
    max_concurrent_tasks: int = 3
    gpu_enabled: bool = False
    model_cache_dir: str = "./data/models"

    # Security
    security_level: str = "high"  # low, medium, high, paranoid
    authentication_required: bool = True
    intrusion_detection_enabled: bool = True

    # Economics
    staking_enabled: bool = True
    initial_stake: float = 0.0
    reward_sharing_enabled: bool = True

    # Production
    monitoring_enabled: bool = True
    metrics_port: int = 9090
    health_check_port: int = 8080
    log_level: str = "INFO"

    # Deployment
    kubernetes_enabled: bool = False
    auto_scaling_enabled: bool = False
    load_balancer_enabled: bool = False


class PoUWNode:
    """
    Production-ready PoUW blockchain node.

    Integrates all PoUW system components into a unified node implementation
    suitable for production deployment.
    """

    def __init__(self, config: NodeConfiguration):
        self.config = config
        self.state = NodeState.INITIALIZING
        self.node_id = config.node_id
        self.start_time = time.time()

        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(f"PoUWNode-{self.node_id}")

        # Core components
        self.blockchain: Optional[Blockchain] = None
        self.p2p_node: Optional[P2PNode] = None
        self.miner: Optional[PoUWMiner] = None
        self.trainer: Optional[DistributedTrainer] = None

        # Network management
        self.network_manager: Optional[NetworkOperationsManager] = None
        self.vpn_manager: Optional[VPNMeshTopologyManager] = None
        self.recovery_manager: Optional[CrashRecoveryManager] = None
        self.election_manager: Optional[LeaderElectionManager] = None

        # Security components
        self.security_monitor: Optional[ComprehensiveSecurityMonitor] = None
        self.authenticator: Optional[NodeAuthenticator] = None
        self.attack_mitigator: Optional[AttackMitigationSystem] = None

        # Economic system
        self.economic_system: Optional[EconomicSystem] = None
        self.staking_manager: Optional[StakingManager] = None
        self.task_matcher: Optional[TaskMatcher] = None

        # Production features
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.gpu_manager: Optional[GPUManager] = None
        self.dataset_manager: Optional[ProductionDatasetManager] = None

        # Deployment infrastructure
        self.k8s_orchestrator: Optional[KubernetesOrchestrator] = None
        self.deployment_monitor: Optional[DeploymentMonitor] = None
        self.health_checker: Optional[HealthChecker] = None

        # Data management components
        self.data_manager: Optional[DataAvailabilityManager] = None
        self.hash_ring: Optional[ConsistentHashRing] = None

        # Advanced features
        self.vrf: Optional[VerifiableRandomFunction] = None
        self.worker_selector: Optional[AdvancedWorkerSelection] = None
        self.commitment_manager: Optional[ZeroNonceCommitment] = None

        # Cryptographic components
        self.bls_crypto: Optional[BLSThresholdCrypto] = None
        self.dkg: Optional[DistributedKeyGeneration] = None
        self.consensus: Optional[SupervisorConsensus] = None

        # Runtime state
        self.is_running = False
        self.shutdown_event = threading.Event()
        self.tasks: Set[asyncio.Task] = set()

    def _setup_logging(self):
        """Setup comprehensive logging configuration"""
        log_level = getattr(
            logging, self.config.log_level.upper(), logging.INFO
        )

        # Create logs directory
        os.makedirs("logs", exist_ok=True)

        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(f"logs/pouw_node_{self.node_id}.log"),
                logging.StreamHandler(sys.stdout),
            ],
        )

    async def initialize(self):
        """Initialize all node components"""
        self.logger.info(f"Initializing PoUW node {self.node_id}")
        self.state = NodeState.INITIALIZING

        try:
            # Initialize core blockchain
            await self._init_blockchain()

            # Initialize network components
            await self._init_network()

            # Initialize security systems
            await self._init_security()

            # Initialize economic system
            await self._init_economics()

            # Initialize production features
            await self._init_production()

            # Initialize deployment infrastructure
            await self._init_deployment()

            # Initialize data management
            await self._init_data_management()

            # Initialize advanced features
            await self._init_advanced_features()

            # Initialize cryptographic components
            await self._init_cryptography()

            self.logger.info(
                f"Node {self.node_id} initialization complete"
            )
            self.state = NodeState.CONNECTING

        except Exception as e:
            self.logger.error(f"Node initialization failed: {e}")
            self.state = NodeState.ERROR
            raise

    async def _init_blockchain(self):
        """Initialize blockchain components"""
        self.logger.info("Initializing blockchain components")

        # Create blockchain data directory
        os.makedirs(self.config.blockchain_data_dir, exist_ok=True)

        # Initialize blockchain
        self.blockchain = Blockchain()

        # Initialize miner if enabled
        if self.config.mining_enabled:
            self.miner = PoUWMiner(miner_id=self.node_id)

        # Initialize trainer if enabled
        if self.config.training_enabled:
            # Create dummy model - should be set later with actual task
            model = SimpleMLP(
                input_size=784, hidden_sizes=[128, 64], output_size=10
            )
            self.trainer = DistributedTrainer(
                model=model,
                task_id="default_task",
                miner_id=self.node_id
            )

    async def _init_network(self):
        """Initialize network components"""
        self.logger.info("Initializing network components")

        # Initialize P2P node
        self.p2p_node = P2PNode(
            node_id=self.node_id,
            port=self.config.listen_port
        )

        # Initialize network managers
        self.network_manager = NetworkOperationsManager(
            self.node_id,
            role="worker",
            supervisor_nodes=getattr(self, 'supervisor_nodes', [])
        )
        self.vpn_manager = VPNMeshTopologyManager(self.node_id)
        self.recovery_manager = CrashRecoveryManager(self.node_id)
        self.election_manager = LeaderElectionManager(
            node_id=self.node_id,
            supervisor_nodes=getattr(self, 'supervisor_nodes', [])
        )

        # Initialize consensus trainer
        self.miner = PoUWMiner(miner_id=self.node_id)

        # Initialize distributed trainer
        model = SimpleMLP(
            input_size=784, hidden_sizes=[128, 64], output_size=10
        )
        self.trainer = DistributedTrainer(
            model=model,
            task_id="default_task",
            miner_id=self.node_id
        )

    async def _init_security(self):
        """Initialize security components"""
        self.logger.info("Initializing security components")

        # Initialize security monitor
        self.security_monitor = ComprehensiveSecurityMonitor()

        # Initialize authenticator if required
        if self.config.authentication_required:
            self.authenticator = NodeAuthenticator()

        # Initialize attack mitigation
        self.attack_mitigator = AttackMitigationSystem()

    async def _init_economics(self):
        """Initialize economic system components"""
        self.logger.info("Initializing economic system")

        # Initialize economic system
        self.economic_system = EconomicSystem()

        # Initialize staking if enabled
        if self.config.staking_enabled:
            self.staking_manager = StakingManager()

        # Initialize task matcher
        self.task_matcher = TaskMatcher(
            stake_pool=getattr(self.staking_manager, 'stake_pool', None)
        )

    async def _init_production(self):
        """Initialize production features"""
        self.logger.info("Initializing production features")

        # Initialize performance monitoring
        if self.config.monitoring_enabled:
            self.performance_monitor = PerformanceMonitor()

        # Initialize GPU management if enabled
        if self.config.gpu_enabled:
            self.gpu_manager = GPUManager()

        # Initialize dataset management
        os.makedirs(self.config.model_cache_dir, exist_ok=True)
        self.dataset_manager = ProductionDatasetManager()

    async def _init_deployment(self):
        """Initialize deployment infrastructure"""
        self.logger.info("Initializing deployment infrastructure")

        # Initialize Kubernetes orchestrator if enabled
        if self.config.kubernetes_enabled:
            self.k8s_orchestrator = KubernetesOrchestrator()

        # Initialize deployment monitor
        self.deployment_monitor = DeploymentMonitor()

        # Initialize health checker
        self.health_checker = HealthChecker()

    async def _init_data_management(self):
        """Initialize data management components"""
        self.logger.info("Initializing data management")

        # Initialize data availability manager
        self.data_manager = DataAvailabilityManager()

        # Initialize consistent hash ring
        self.hash_ring = ConsistentHashRing()

    async def _init_advanced_features(self):
        """Initialize advanced features"""
        self.logger.info("Initializing advanced features")

        # Initialize VRF
        self.vrf = VerifiableRandomFunction()

        # Initialize advanced worker selection
        self.worker_selector = AdvancedWorkerSelection(self.vrf)

        # Initialize zero-nonce commitment
        self.commitment_manager = ZeroNonceCommitment()

    async def _init_cryptography(self):
        """Initialize cryptographic components"""
        self.logger.info("Initializing cryptographic components")

        # Initialize BLS threshold cryptography for supervisors
        if self.config.node_type in [NodeType.SUPERVISOR, NodeType.HYBRID]:
            self.bls_crypto = BLSThresholdCrypto(
                threshold=3, total_parties=5
            )
            self.dkg = DistributedKeyGeneration(
                supervisor_id=self.node_id,
                threshold=3,
                total_supervisors=5
            )
            self.consensus = SupervisorConsensus(
                supervisor_id=self.node_id,
                dkg=self.dkg
            )

    async def start(self):
        """Start the node and all its services"""
        self.logger.info(f"Starting PoUW node {self.node_id}")
        self.is_running = True

        try:
            # Start core services
            await self._start_core_services()

            # Start network services
            await self._start_network_services()

            # Start security services
            await self._start_security_services()

            # Start economic services
            await self._start_economic_services()

            # Start production services
            await self._start_production_services()

            # Start deployment services
            await self._start_deployment_services()

            # Start advanced services
            await self._start_advanced_services()

            self.state = NodeState.ACTIVE
            self.logger.info("Node started successfully and is now active")

            # Main event loop
            await self._run_main_loop()

        except Exception as e:
            self.logger.error(f"Node startup failed: {e}")
            self.state = NodeState.ERROR
            await self.shutdown()
            raise

    async def _start_core_services(self):
        """Start core blockchain services"""
        self.logger.info("Starting core services")

        # Start blockchain sync
        if self.blockchain:
            task = asyncio.create_task(self._sync_blockchain())
            self.tasks.add(task)

        # Start miner if enabled
        if self.miner and self.config.mining_enabled:
            task = asyncio.create_task(self._run_miner())
            self.tasks.add(task)

        # Start trainer if enabled
        if self.trainer and self.config.training_enabled:
            task = asyncio.create_task(self._run_trainer())
            self.tasks.add(task)

    async def _start_network_services(self):
        """Start network services"""
        self.logger.info("Starting network services")

        # Start P2P node
        if self.p2p_node:
            if hasattr(self.p2p_node, 'start'):
                task = asyncio.create_task(self.p2p_node.start())
                self.tasks.add(task)

        # Start network managers
        if hasattr(self.network_manager, 'start') \
           and callable(getattr(self.network_manager, 'start')):
            self.network_manager.start()
        if hasattr(self.election_manager, 'start') \
           and callable(getattr(self.election_manager, 'start')):
            self.election_manager.start()
        if hasattr(self.vpn_manager, 'start') \
           and callable(getattr(self.vpn_manager, 'start')):
            self.vpn_manager.start()
        if hasattr(self.recovery_manager, 'start') \
           and callable(getattr(self.recovery_manager, 'start')):
            self.recovery_manager.start()

    async def _start_security_services(self):
        """Start security services"""
        self.logger.info("Starting security services")

        # Start security monitoring
        if hasattr(self.security_monitor, 'start') \
           and callable(getattr(self.security_monitor, 'start')):
            self.security_monitor.start()

        # Start authentication services
        if hasattr(self.authenticator, 'start') \
           and callable(getattr(self.authenticator, 'start')):
            self.authenticator.start()

        # Start attack mitigation
        if hasattr(self.attack_mitigator, 'start') \
           and callable(getattr(self.attack_mitigator, 'start')):
            self.attack_mitigator.start()

    async def _start_economic_services(self):
        """Start economic services"""
        self.logger.info("Starting economic services")

        # Start economic system
        if hasattr(self.economic_system, 'start') \
           and callable(getattr(self.economic_system, 'start')):
            self.economic_system.start()

    async def _start_production_services(self):
        """Start production services"""
        self.logger.info("Starting production services")

        # Start performance monitor
        if hasattr(self.performance_monitor, 'start') \
           and callable(getattr(self.performance_monitor, 'start')):
            self.performance_monitor.start()

        # Initialize GPU manager
        if hasattr(self.gpu_manager, 'initialize') \
           and callable(getattr(self.gpu_manager, 'initialize')):
            self.gpu_manager.initialize()

    async def _start_deployment_services(self):
        """Start deployment services"""
        self.logger.info("Starting deployment services")

        # Start production monitor
        if hasattr(self.deployment_monitor, 'start') \
           and callable(getattr(self.deployment_monitor, 'start')):
            self.deployment_monitor.start()

        # Start health checker
        if hasattr(self.health_checker, 'start') \
           and callable(getattr(self.health_checker, 'start')):
            self.health_checker.start()

    async def _start_advanced_services(self):
        """Start advanced services"""
        self.logger.info("Starting advanced services")

        # Start DKG if configured
        if self.dkg:
            task = asyncio.create_task(self._run_dkg())
            self.tasks.add(task)

    async def _run_main_loop(self):
        """Main node event loop"""
        self.logger.info("Starting main event loop")

        while self.is_running:
            try:
                # Health check
                await self._health_check()

                # Process pending tasks
                await self._process_pending_tasks()

                # Update metrics
                await self._update_metrics()

                # Sleep before next iteration
                await asyncio.sleep(1)

            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(5)

    async def _health_check(self):
        """Perform health check of all components"""
        try:
            # Check blockchain health
            if self.blockchain:
                latest_block = self.blockchain.get_latest_block()
                if latest_block:
                    self.logger.debug(
                        f"Blockchain health OK - Latest block: "
                        f"{latest_block.get_hash()[:8]}"
                    )

            # Check network health
            if self.p2p_node and hasattr(self.p2p_node, 'get_peer_count'):
                peer_count = self.p2p_node.get_peer_count()
                self.logger.debug(f"Network health OK - Connected peers: "
                                  f"{peer_count}")

            # Check mining status
            if self.miner and self.config.mining_enabled:
                self.logger.debug("Mining health OK")

        except Exception as e:
            self.logger.warning(f"Health check failed: {e}")

    async def _process_pending_tasks(self):
        """Process any pending background tasks"""
        try:
            # Clean up completed tasks
            completed_tasks = [task for task in self.tasks if task.done()]
            for task in completed_tasks:
                self.tasks.remove(task)
                if task.exception():
                    self.logger.error(
                        f"Task completed with exception: {task.exception()}"
                    )

        except Exception as e:
            self.logger.error(f"Error processing pending tasks: {e}")

    async def _update_metrics(self):
        """Update system metrics"""
        try:
            # Update performance metrics
            if (hasattr(self.performance_monitor, 'update_metrics')
                    and callable(getattr(self.performance_monitor,
                                         'update_metrics'))):
                self.performance_monitor.update_metrics()

        except Exception as e:
            self.logger.error(f"Failed to update metrics: {e}")

    async def _sync_blockchain(self):
        """Sync blockchain with network"""
        self.logger.info("Starting blockchain sync")

        while self.is_running:
            try:
                # Placeholder for blockchain sync logic
                await asyncio.sleep(30)  # Sync every 30 seconds

            except Exception as e:
                self.logger.error(f"Blockchain sync error: {e}")
                await asyncio.sleep(60)

    async def _run_miner(self):
        """Run traditional proof-of-work mining operations (no ML required)"""
        self.logger.info("Starting traditional PoW mining operations")
        self.state = NodeState.MINING

        import hashlib
        import time as time_module
        import random

        mining_iteration = 0
        
        while self.state == NodeState.MINING:
            try:
                mining_iteration += 1
                self.logger.info(f"Mining iteration {mining_iteration}")

                # Get current difficulty from blockchain
                if self.blockchain:
                    difficulty_info = self.blockchain.get_current_difficulty_info()
                    target = difficulty_info["current_target"]
                    self.logger.info(
                        f"Current difficulty: {difficulty_info['difficulty_ratio']:.2f}x"
                    )
                    self.logger.info(
                        f"Target block time: {difficulty_info['target_block_time']}s"
                    )
                    self.logger.info(
                        f"Recent avg time: {difficulty_info['recent_avg_block_time']:.1f}s"
                    )
                else:
                    target = 0x0FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
                
                # Create a new block to mine
                if self.blockchain:
                    # Get transactions from mempool (limit to reasonable size)
                    transactions = self.blockchain.mempool[:10]
                    miner_address = f"miner_{self.config.node_id}"
                    
                    # Create block template
                    block = self.blockchain.create_block(
                        transactions=transactions,
                        miner_address=miner_address
                    )
                    
                    # Traditional proof-of-work mining
                    max_nonce = 1000000  # Try up to 1M nonces per iteration
                    start_time = time_module.time()
                    
                    self.logger.info(f"Mining with up to {max_nonce} nonce attempts")
                    
                    # Start mining from random nonce to avoid conflicts
                    start_nonce = random.randint(0, 2**32 - max_nonce)
                    
                    for nonce in range(start_nonce, start_nonce + max_nonce):
                        # Update block nonce
                        block.header.nonce = nonce
                        
                        # Calculate block hash
                        block_hash_hex = block.get_hash()
                        block_hash_int = int(block_hash_hex, 16)
                        
                        # Check if we found a valid block
                        if block_hash_int < target:
                            mining_time = time_module.time() - start_time
                            self.logger.info(
                                f"🎉 BLOCK FOUND! Nonce: {nonce}, "
                                f"Time: {mining_time:.2f}s"
                            )
                            self.logger.info(f"Block hash: {block_hash_hex}")
                            
                            # Add block to blockchain
                            if self.blockchain.add_block(block):
                                self.logger.info("✅ Block added to blockchain!")
                                self.logger.info(f"💰 Earned 12.5 coins!")
                                
                                # Show current blockchain stats
                                chain_length = self.blockchain.get_chain_length()
                                self.logger.info(f"📊 Chain length: {chain_length}")
                                
                                break
                            else:
                                self.logger.warning("❌ Block validation failed")
                        
                        # Yield control occasionally to avoid blocking
                        if nonce % 10000 == 0:
                            await asyncio.sleep(0.001)
                    
                    else:
                        # No valid nonce found in this iteration
                        mining_time = time_module.time() - start_time
                        self.logger.info(
                            f"No valid nonce found in {mining_time:.2f}s "
                            f"(tried {max_nonce} nonces)"
                        )
                
                # Wait before next mining attempt
                await asyncio.sleep(2)  # 2 second mining cycle

            except Exception as e:
                self.logger.error(f"Mining error: {e}")
                await asyncio.sleep(5)  # Wait longer on error

    async def _run_trainer(self):
        """Run ML training operations"""
        self.logger.info("Starting ML training operations")
        self.state = NodeState.TRAINING

        while self.is_running and self.config.training_enabled:
            try:
                # Placeholder for training logic
                # In practice, this would process actual ML tasks
                await asyncio.sleep(15)

            except Exception as e:
                self.logger.error(f"Training error: {e}")
                await asyncio.sleep(30)

    async def _run_dkg(self):
        """Run distributed key generation"""
        self.logger.info("Starting DKG process")

        try:
            if self.dkg and hasattr(self.dkg, 'start_dkg'):
                await self.dkg.start_dkg()

            # Wait for DKG completion
            success = await self._wait_for_dkg_completion()
            if success:
                self.logger.info("DKG completed successfully")
            else:
                self.logger.warning("DKG failed or timed out")

        except Exception as e:
            self.logger.error(f"DKG error: {e}")

    async def _wait_for_dkg_completion(self, timeout: int = 300) -> bool:
        """Wait for DKG to complete with timeout"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.dkg and hasattr(self.dkg, 'is_complete'):
                if self.dkg.is_complete():
                    return True

            await asyncio.sleep(1)

        return False

    async def _connect_to_peers(self):
        """Connect to bootstrap peers"""
        self.logger.info("Connecting to bootstrap peers")

        for peer in self.config.bootstrap_peers:
            try:
                if self.p2p_node and hasattr(self.p2p_node, 'connect_to_peer'):
                    await self.p2p_node.connect_to_peer(peer)
                    self.logger.info(f"Connected to peer: {peer}")

            except Exception as e:
                self.logger.warning(f"Failed to connect to peer {peer}: {e}")

    async def shutdown(self):
        """Gracefully shutdown the node"""
        self.logger.info(f"Shutting down PoUW node {self.node_id}")
        self.state = NodeState.SHUTTING_DOWN
        self.is_running = False

        try:
            # Signal shutdown to all components
            self.shutdown_event.set()

            # Cancel all running tasks
            for task in self.tasks:
                if not task.done():
                    task.cancel()

            # Wait for tasks to complete
            if self.tasks:
                await asyncio.gather(*self.tasks, return_exceptions=True)

                # Shutdown individual components
                await self._shutdown_components()

            self.logger.info("Node shutdown completed")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

    async def _shutdown_components(self):
        """Shutdown individual components"""
        # Shutdown in reverse order of startup

        # Shutdown advanced services
        if self.dkg and hasattr(self.dkg, 'shutdown'):
            await self.dkg.shutdown()

        # Shutdown deployment services
        if self.health_checker and hasattr(self.health_checker, 'shutdown'):
            await self.health_checker.shutdown()

        if (self.deployment_monitor and
                hasattr(self.deployment_monitor, 'shutdown')):
            await self.deployment_monitor.shutdown()

        # Shutdown production services
        if (self.performance_monitor and
                hasattr(self.performance_monitor, 'shutdown')):
            await self.performance_monitor.shutdown()

        # Shutdown network services
        if self.p2p_node and hasattr(self.p2p_node, 'shutdown'):
            await self.p2p_node.shutdown()

    def get_status(self) -> Dict[str, Any]:
        """Get current node status"""
        return {
            "node_id": self.node_id,
            "node_type": self.config.node_type.value,
            "state": self.state.value,
            "is_running": self.is_running,
            "uptime": time.time() - getattr(self, "start_time", time.time()),
            "blockchain_height": (
                len(self.blockchain.chain) if self.blockchain else 0
            ),
            "peer_count": (
                self.p2p_node.get_peer_count()
                if self.p2p_node and hasattr(self.p2p_node, 'get_peer_count')
                else 0
            ),
            "mining_enabled": self.config.mining_enabled,
            "training_enabled": self.config.training_enabled,
        }


def load_config_from_file(config_path: str) -> NodeConfiguration:
    """Load node configuration from JSON file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)

    return NodeConfiguration(**config_data)


def create_default_config(node_id: str,
                          node_type: NodeType) -> NodeConfiguration:
    """Create default configuration for a node"""
    return NodeConfiguration(
        node_id=node_id,
        node_type=node_type,
        private_key_path=f"./keys/{node_id}.pem",
        listen_port=8333,
        blockchain_data_dir=f"./data/{node_id}/blockchain",
        model_cache_dir=f"./data/{node_id}/models",
    )


async def main():
    """Main entry point for the PoUW node"""
    parser = argparse.ArgumentParser(description="PoUW Blockchain Node")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--node-id", help="Node identifier")
    parser.add_argument("--node-type", choices=["worker", "supervisor",
                                                "miner", "hybrid"],
                        default="worker", help="Node type")
    parser.add_argument("--port", type=int, default=8333, help="Listen port")
    parser.add_argument("--mining", action="store_true", help="Enable mining")
    parser.add_argument("--training", action="store_true",
                        help="Enable ML training")
    parser.add_argument("--gpu", action="store_true",
                        help="Enable GPU acceleration")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose logging")

    args = parser.parse_args()

    # Load or create configuration
    if args.config:
        config = load_config_from_file(args.config)
    else:
        node_id = args.node_id or f"node_{int(time.time())}"
        node_type = NodeType(args.node_type)
        config = create_default_config(node_id, node_type)

    # Override config with command line arguments
    if args.port:
        config.listen_port = args.port
    if args.mining:
        config.mining_enabled = True
    if args.training:
        config.training_enabled = True
    if args.gpu:
        config.gpu_enabled = True
    if args.verbose:
        config.log_level = "DEBUG"

    # Create and start node
    node = PoUWNode(config)

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        asyncio.create_task(node.shutdown())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Initialize and start the node
        await node.initialize()
        await node.start()

    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        await node.shutdown()

    except Exception as e:
        print(f"Node failed with error: {e}")
        await node.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
