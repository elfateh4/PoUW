"""
Complete PoUW Node Implementation with Advanced Features.

This is the main node class that integrates all PoUW components:
- Blockchain management with BLS threshold signatures
- ML training coordination with gradient poisoning detection
- Mining with useful work and VRF-based selection
- P2P networking with Byzantine fault tolerance
- Economic participation with advanced data management
- Security features with attack mitigation
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim

from .blockchain import Blockchain, MLTask, PayForTaskTransaction, BuyTicketsTransaction
from .ml import SimpleMLP, DistributedTrainer, MiniBatch, GradientUpdate
from .mining import PoUWMiner, PoUWVerifier, MiningProof
from .network import (P2PNode, NetworkMessage, BlockchainMessageHandler, MLMessageHandler, MessageHistory,
                     NetworkOperationsManager)
from .economics import EconomicSystem, NodeRole, Ticket
from .crypto import BLSThresholdCrypto, DistributedKeyGeneration, SupervisorConsensus
from .security import AttackMitigationSystem, SecurityAlert, AttackType
from .data import DataAvailabilityManager, DataShardType, DatasetSplitter
from .advanced import VerifiableRandomFunction, AdvancedWorkerSelection, ZeroNonceCommitment, MessageHistoryMerkleTree


class PoUWNode:
    """Complete PoUW node implementation with advanced features"""
    
    def __init__(self, node_id: str, role: NodeRole, host: str = "localhost", port: int = 8000):
        self.node_id = node_id
        self.role = role
        self.host = host
        self.port = port
        
        # Core components
        self.blockchain = Blockchain()
        self.economic_system = EconomicSystem()
        self.p2p_node = P2PNode(node_id, host, port)
        self.message_history = MessageHistory()
        
        # Advanced security and cryptographic components
        self.vrf = VerifiableRandomFunction()
        self.attack_mitigation = AttackMitigationSystem()
        self.data_manager = DataAvailabilityManager(replication_factor=3)
        self.dataset_splitter = DatasetSplitter()
        
        # Role-specific components
        self.miner = None
        self.verifier = None
        self.trainer = None
        
        # Advanced features for different roles
        self.dkg = None  # Distributed Key Generation for supervisors
        self.supervisor_consensus = None  # BLS threshold signatures
        self.worker_selector = AdvancedWorkerSelection(self.vrf)
        self.commitment_system = ZeroNonceCommitment(commitment_depth=5)
        self.merkle_tree = MessageHistoryMerkleTree()
        
        # Network operations manager ⭐ NEW
        supervisor_nodes = None
        if role == NodeRole.SUPERVISOR:
            supervisor_nodes = []  # Will be populated during network initialization
        self.network_operations = NetworkOperationsManager(
            node_id=node_id, 
            role=role.value,
            supervisor_nodes=supervisor_nodes
        )
        
        # Current state
        self.current_task = None
        self.is_mining = False
        self.is_training = False
        self.security_alerts: List[SecurityAlert] = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"PoUWNode-{node_id}")
        
        self._setup_node()
    
    def _setup_node(self):
        """Setup node based on role with advanced features"""
        
        # Setup message handlers
        blockchain_handler = BlockchainMessageHandler(
            self.blockchain, 
            on_new_block=self._handle_new_block
        )
        
        ml_handler = MLMessageHandler(
            trainer=self.trainer,
            supervisor=self if self.role == NodeRole.SUPERVISOR else None
        )
        
        # Register handlers
        self.p2p_node.register_handler(
            ["NEW_BLOCK", "NEW_TRANSACTION", "REQUEST_BLOCK", "REQUEST_MEMPOOL"],
            blockchain_handler
        )
        
        self.p2p_node.register_handler(
            ["IT_RES", "GRADIENT_UPDATE", "TASK_ASSIGNMENT", "HEARTBEAT", "SECURITY_ALERT"],
            ml_handler
        )
        
        # Setup role-specific components
        if self.role in [NodeRole.MINER, NodeRole.SUPERVISOR]:
            self.miner = PoUWMiner(self.node_id)
        
        if self.role == NodeRole.VERIFIER:
            self.verifier = PoUWVerifier()
        
        # Setup supervisor-specific advanced features
        if self.role == NodeRole.SUPERVISOR:
            self.dkg = DistributedKeyGeneration(
                supervisor_id=self.node_id,
                threshold=3,  # 3-of-5 threshold
                total_supervisors=5
            )
            # Will be initialized after DKG completion
            self.supervisor_consensus = None
        
        # Initialize data manager hash ring
        self.data_manager.hash_ring.add_node(self.node_id)
        
        self.logger.info(f"Node setup complete as {self.role.value} with advanced features")
    
    async def initialize_supervisor_network(self, supervisor_peers: List[str]):
        """Initialize supervisor network with DKG protocol"""
        if self.role != NodeRole.SUPERVISOR or not self.dkg:
            return
        
        try:
            # Start DKG protocol
            commitments, key_shares = self.dkg.start_dkg()
            
            # Distribute key shares to other supervisors
            for peer_id in supervisor_peers:
                if peer_id != self.node_id and peer_id in key_shares:
                    message = NetworkMessage(
                        msg_type="DKG_SHARE",
                        sender_id=self.node_id,
                        data={
                            'key_share': key_shares[peer_id].to_dict(),
                            'commitments': [c.hex() for c in commitments]
                        }
                    )
                    await self.p2p_node.send_to_peer(peer_id, message)
            
            self.logger.info("DKG protocol initiated")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize supervisor network: {e}")
    
    async def handle_gradient_updates(self, gradient_updates: List[GradientUpdate]) -> List[GradientUpdate]:
        """Process gradient updates with security checks"""
        try:
            # Apply gradient poisoning detection
            clean_updates, security_alerts = self.attack_mitigation.process_gradient_updates(gradient_updates)
            
            # Log security alerts
            for alert in security_alerts:
                self.security_alerts.append(alert)
                self.logger.warning(f"Security alert: {alert.alert_type.value} from {alert.node_id}")
                
                # Broadcast security alert to network
                await self._broadcast_security_alert(alert)
            
            # Submit supervisor votes if needed
            if self.role == NodeRole.SUPERVISOR and security_alerts:
                for alert in security_alerts:
                    if alert.confidence > 0.7:  # High confidence threshold
                        proposal_id = f"blacklist_{alert.node_id}_{int(time.time())}"
                        self.attack_mitigation.submit_supervisor_vote(
                            proposal_id, self.node_id, True, alert.evidence
                        )
            
            return clean_updates
            
        except Exception as e:
            self.logger.error(f"Error processing gradient updates: {e}")
            return gradient_updates
    
    async def select_workers_for_task(self, task: MLTask) -> Dict[NodeRole, List[Ticket]]:
        """Select workers using advanced VRF-based selection"""
        try:
            # Get candidates from stake pool
            candidates = []
            for role in [NodeRole.MINER, NodeRole.SUPERVISOR, NodeRole.EVALUATOR]:
                role_tickets = self.economic_system.stake_pool.get_tickets_by_role(role)
                for ticket in role_tickets:
                    candidates.append({
                        'node_id': ticket.owner_id,
                        'role': ticket.role,
                        'stake_amount': ticket.stake_amount,
                        'has_gpu': ticket.preferences.get('has_gpu', False),
                        'bandwidth_mbps': ticket.preferences.get('bandwidth_mbps', 10),
                        'ticket': ticket
                    })
            
            # Filter by role and select using VRF
            selected_workers = {}
            
            for role in [NodeRole.MINER, NodeRole.SUPERVISOR, NodeRole.EVALUATOR]:
                role_candidates = [c for c in candidates if c['role'] == role]
                
                if role_candidates:
                    num_needed = 3 if role == NodeRole.MINER else 2
                    selected, vrf_proofs = self.worker_selector.select_workers_with_vrf(
                        task_id=task.task_id,
                        candidates=role_candidates,
                        num_needed=num_needed,
                        selection_criteria=task.performance_requirements
                    )
                    
                    # Convert back to tickets
                    selected_tickets = [worker['ticket'] for worker in selected]
                    selected_workers[role] = selected_tickets
                    
                    # Store VRF proofs for verification
                    self._store_vrf_proofs(task.task_id, role, vrf_proofs)
            
            return selected_workers
            
        except Exception as e:
            self.logger.error(f"Error selecting workers: {e}")
            return {}
    
    def _store_vrf_proofs(self, task_id: str, role: NodeRole, vrf_proofs: List):
        """Store VRF proofs for later verification"""
        # Implementation would store proofs in a persistent way
        pass
    
    async def start(self):
        """Start the PoUW node"""
        await self.p2p_node.start()
        
        # Start network operations
        await self.network_operations.start_operations()
        
        self.logger.info(f"PoUW node {self.node_id} started with network operations")
        
        # Start background tasks
        asyncio.create_task(self._mining_loop())
        asyncio.create_task(self._maintenance_loop())
    
    async def stop(self):
        """Stop the PoUW node"""
        self.is_mining = False
        self.is_training = False
        
        # Stop network operations
        await self.network_operations.stop_operations()
        
        await self.p2p_node.stop()
        self.logger.info(f"PoUW node {self.node_id} stopped")
    
    async def connect_to_network(self, bootstrap_peers: List[Tuple[str, int]]):
        """Connect to the PoUW network"""
        for host, port in bootstrap_peers:
            success = await self.p2p_node.connect_to_peer(host, port)
            if success:
                self.logger.info(f"Connected to bootstrap peer {host}:{port}")
            else:
                self.logger.warning(f"Failed to connect to {host}:{port}")
    
    def stake_and_register(self, stake_amount: float, preferences: Dict[str, Any]) -> Ticket:
        """Stake coins and register for participation"""
        try:
            ticket = self.economic_system.buy_ticket(
                self.node_id, self.role, stake_amount, preferences
            )
            
            # Create and broadcast BUY_TICKETS transaction
            stake_tx = BuyTicketsTransaction(
                version=1,
                inputs=[],  # Would reference actual UTXOs
                outputs=[],
                role=self.role.value,
                stake_amount=stake_amount,
                preferences=preferences
            )
            
            self.blockchain.add_transaction_to_mempool(stake_tx)
            
            # Broadcast to network
            asyncio.create_task(self._broadcast_transaction(stake_tx))
            
            self.logger.info(f"Staked {stake_amount} PAI coins as {self.role.value}")
            return ticket
            
        except Exception as e:
            self.logger.error(f"Failed to stake: {e}")
            raise
    
    def submit_ml_task(self, task_definition: Dict[str, Any], fee: float) -> str:
        """Submit ML task to network (client role)"""
        
        # Create ML task
        task = MLTask(
            task_id=f"task_{int(time.time())}_{self.node_id}",
            model_type=task_definition.get('model_type', 'mlp'),
            architecture=task_definition.get('architecture', {}),
            optimizer=task_definition.get('optimizer', {}),
            stopping_criterion=task_definition.get('stopping_criterion', {}),
            validation_strategy=task_definition.get('validation_strategy', {}),
            metrics=task_definition.get('metrics', ['accuracy', 'loss']),
            dataset_info=task_definition.get('dataset_info', {}),
            performance_requirements=task_definition.get('performance_requirements', {}),
            fee=fee,
            client_id=self.node_id
        )
        
        # Create and broadcast PAY_FOR_TASK transaction
        task_tx = PayForTaskTransaction(
            version=1,
            inputs=[],  # Would reference actual UTXOs
            outputs=[],
            task_definition=task.to_dict(),
            fee=fee
        )
        
        self.blockchain.add_transaction_to_mempool(task_tx)
        
        # Submit to economic system for worker assignment
        selected_workers = self.economic_system.submit_task(task)
        
        # Broadcast task to network
        asyncio.create_task(self._broadcast_transaction(task_tx))
        asyncio.create_task(self._broadcast_task_assignment(task, selected_workers))
        
        self.logger.info(f"Submitted ML task {task.task_id} with fee {fee}")
        return task.task_id
    
    async def _handle_task_assignment(self, task: MLTask):
        """Handle assignment to ML task"""
        if self.role not in [NodeRole.MINER, NodeRole.SUPERVISOR]:
            return
        
        self.current_task = task
        
        # Setup ML trainer for miners
        if self.role == NodeRole.MINER:
            # Create model based on task specification
            model = self._create_model_from_task(task)
            self.trainer = DistributedTrainer(model, task.task_id, self.node_id)
            
            # Start training
            self.is_training = True
            asyncio.create_task(self._training_loop(task))
        
        self.logger.info(f"Assigned to task {task.task_id} as {self.role.value}")
    
    def _create_model_from_task(self, task: MLTask) -> SimpleMLP:
        """Create ML model from task specification"""
        arch = task.architecture
        
        model = SimpleMLP(
            input_size=arch.get('input_size', 784),
            hidden_sizes=arch.get('hidden_sizes', [128, 64]),
            output_size=arch.get('output_size', 10)
        )
        
        return model
    
    async def _training_loop(self, task: MLTask):
        """Main ML training loop for miners with advanced features"""
        if not self.trainer:
            return
        
        # Setup optimizer and criterion
        optimizer = optim.Adam(self.trainer.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Generate dummy mini-batches (in practice would load real data)
        batch_size = task.dataset_info.get('batch_size', 32)
        input_size = task.architecture.get('input_size', 784)
        output_size = task.architecture.get('output_size', 10)
        
        iteration = 0
        max_iterations = 100  # Task stopping criterion
        
        while self.is_training and iteration < max_iterations:
            try:
                # Create dummy mini-batch
                batch = MiniBatch(
                    batch_id=f"batch_{iteration}",
                    data=torch.randn(batch_size, input_size).numpy(),
                    labels=torch.randint(0, output_size, (batch_size,)).numpy(),
                    epoch=iteration // 10
                )
                
                # Process training iteration
                message, metrics = self.trainer.process_iteration(batch, optimizer, criterion)
                
                # Create zero-nonce commitment for future iterations
                if iteration % 5 == 0:  # Every 5 iterations
                    model_state = {
                        'weights': [p.detach().numpy().tolist() for p in self.trainer.model.parameters()],
                        'iteration': iteration
                    }
                    commitment = self.commitment_system.create_commitment(
                        miner_id=self.node_id,
                        future_iteration=iteration + self.commitment_system.commitment_depth,
                        model_state=model_state,
                        vrf=self.vrf
                    )
                    self.logger.debug(f"Created commitment for iteration {commitment['future_iteration']}")
                
                # Broadcast iteration message
                await self._broadcast_iteration_message(message)
                
                # Record in message history and Merkle tree
                if self.role == NodeRole.SUPERVISOR:
                    self.message_history.record_message(task.task_id, message)
                    message_str = f"gradient_update_{iteration}_{self.node_id}"
                    self.merkle_tree.add_message(message_str)
                
                # Try to mine block after iteration
                if self.miner:
                    await self._attempt_mining_with_advanced_features(message, batch)
                
                iteration += 1
                await asyncio.sleep(1)  # Training interval
                
            except Exception as e:
                self.logger.error(f"Error in training iteration {iteration}: {e}")
                break
        
        # Build Merkle tree for completed epoch
        if self.role == NodeRole.SUPERVISOR and iteration > 0:
            epoch = iteration // 10
            epoch_messages = [f"gradient_update_{i}_{self.node_id}" for i in range(max(0, iteration-10), iteration)]
            merkle_root = self.merkle_tree.build_merkle_tree(epoch, epoch_messages)
            self.logger.info(f"Built Merkle tree for epoch {epoch}: {merkle_root[:16]}...")
        
        self.is_training = False
        self.logger.info(f"Training completed for task {task.task_id}")
    
    async def _attempt_mining_with_advanced_features(self, message, batch):
        """Enhanced mining with commitment fulfillment and security checks"""
        if not self.miner or not self.trainer:
            return
        
        try:
            # Check for commitment fulfillment
            pending_commitments = self.commitment_system.get_pending_commitments(self.node_id)
            current_iteration = message.iteration
            
            for commitment in pending_commitments:
                if commitment['future_iteration'] == current_iteration:
                    # This iteration fulfills a commitment
                    self.logger.info(f"Fulfilling commitment {commitment['commitment_id'][:16]}...")
            
            # Get transactions from mempool (filtered for security)
            all_transactions = self.blockchain.mempool[:10]
            
            # Filter out transactions from blacklisted nodes
            blacklisted_nodes = self.attack_mitigation.get_blacklisted_nodes()
            filtered_transactions = [
                tx for tx in all_transactions 
                if not hasattr(tx, 'sender_id') or tx.sender_id not in blacklisted_nodes
            ]
            
            # Calculate sizes
            batch_size = batch.size()
            model_size = sum(p.numel() for p in self.trainer.model.parameters())
            
            # Attempt mining with security considerations
            result = self.miner.mine_block(
                self.trainer, message, batch_size, model_size,
                filtered_transactions, self.blockchain
            )
            
            if result:
                block, mining_proof = result
                
                # Verify block doesn't contain malicious transactions
                block_is_secure = await self._verify_block_security(block)
                
                if block_is_secure and self.blockchain.add_block(block):
                    self.logger.info(f"Successfully mined secure block {block.get_hash()[:16]}...")
                    
                    # Broadcast new block
                    await self._broadcast_new_block(block)
                    
                    # Fulfill any relevant commitments
                    for commitment in pending_commitments:
                        if commitment['future_iteration'] == current_iteration:
                            gradient_update = GradientUpdate(
                                miner_id=self.node_id,
                                task_id=message.task_id,
                                iteration=current_iteration,
                                batch_id=message.batch_id,
                                model_weights=message.model_weights,
                                gradients=message.gradients
                            )
                            
                            self.commitment_system.fulfill_commitment(
                                commitment['commitment_id'],
                                block.header.nonce,
                                block.get_hash(),
                                gradient_update
                            )
                else:
                    self.logger.warning("Failed to add mined block or block failed security check")
            
        except Exception as e:
            self.logger.error(f"Error during advanced mining: {e}")
    
    async def _verify_block_security(self, block) -> bool:
        """Verify block meets security requirements"""
        try:
            # Check for suspicious patterns in block data
            # This is a simplified security check
            
            # Verify no blacklisted nodes are referenced
            blacklisted_nodes = self.attack_mitigation.get_blacklisted_nodes()
            
            # Check transactions don't involve blacklisted entities
            for tx in block.transactions:
                if hasattr(tx, 'sender_id') and tx.sender_id in blacklisted_nodes:
                    return False
            
            # Additional security checks would go here
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying block security: {e}")
            return False
    
    async def _broadcast_security_alert(self, alert: SecurityAlert):
        """Broadcast security alert to network"""
        message = NetworkMessage(
            msg_type="SECURITY_ALERT",
            sender_id=self.node_id,
            data={"alert": alert.__dict__}
        )
        await self.p2p_node.broadcast_message(message)
    
    async def _mining_loop(self):
        """Background mining loop"""
        while True:
            if self.is_mining and not self.is_training:
                # Mine empty blocks when not training
                try:
                    transactions = self.blockchain.mempool[:10]
                    block = self.blockchain.create_block(transactions, self.node_id)
                    
                    # Simple PoW for non-PoUW blocks
                    nonce = 0
                    while nonce < 1000000:  # Limit attempts
                        block.header.nonce = nonce
                        if int(block.get_hash(), 16) < self.blockchain.difficulty_target:
                            if self.blockchain.add_block(block):
                                await self._broadcast_new_block(block)
                                self.logger.info(f"Mined empty block {block.get_hash()[:16]}...")
                            break
                        nonce += 1
                
                except Exception as e:
                    self.logger.error(f"Error in mining loop: {e}")
            
            await asyncio.sleep(10)  # Mining interval
    
    async def _maintenance_loop(self):
        """Background maintenance tasks"""
        while True:
            try:
                # Clean up expired tickets
                self.economic_system.stake_pool.remove_expired_tickets()
                
                # Log network stats
                stats = self.economic_system.get_network_stats()
                self.logger.debug(f"Network stats: {stats}")
                
            except Exception as e:
                self.logger.error(f"Error in maintenance: {e}")
            
            await asyncio.sleep(60)  # Maintenance interval
    
    async def _handle_new_block(self, block_data: Dict[str, Any]):
        """Handle new block from network"""
        # Validate and add block
        # In practice would reconstruct Block object and validate
        self.logger.info(f"Received new block from network")
    
    async def _broadcast_transaction(self, transaction):
        """Broadcast transaction to network"""
        message = NetworkMessage(
            msg_type="NEW_TRANSACTION",
            sender_id=self.node_id,
            data={"transaction": transaction.to_dict()}
        )
        await self.p2p_node.broadcast_message(message)
    
    async def _broadcast_new_block(self, block):
        """Broadcast new block to network"""
        message = NetworkMessage(
            msg_type="NEW_BLOCK",
            sender_id=self.node_id,
            data={"block": block.to_dict()}
        )
        await self.p2p_node.broadcast_message(message)
    
    async def _broadcast_iteration_message(self, iteration_message):
        """Broadcast ML iteration message"""
        message = NetworkMessage(
            msg_type="IT_RES",
            sender_id=self.node_id,
            data={"iteration_message": iteration_message.__dict__}
        )
        await self.p2p_node.broadcast_message(message)
    
    async def _broadcast_task_assignment(self, task: MLTask, workers: Dict):
        """Broadcast task assignment to selected workers"""
        for role, tickets in workers.items():
            for ticket in tickets:
                message = NetworkMessage(
                    msg_type="TASK_ASSIGNMENT",
                    sender_id=self.node_id,
                    data={"task": task.to_dict()}
                )
                await self.p2p_node.send_to_peer(ticket.owner_id, message)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current node status with advanced features"""
        status = {
            'node_id': self.node_id,
            'role': self.role.value,
            'blockchain_height': self.blockchain.get_chain_length(),
            'mempool_size': self.blockchain.get_mempool_size(),
            'peer_count': self.p2p_node.get_peer_count(),
            'current_task': self.current_task.task_id if self.current_task else None,
            'is_training': self.is_training,
            'is_mining': self.is_mining,
        }
        
        # Add advanced status information
        try:
            status.update({
                'security_alerts_count': len(self.security_alerts),
                'recent_alerts': len([a for a in self.security_alerts if time.time() - a.timestamp < 3600]),
                'blacklisted_nodes': len(self.attack_mitigation.get_blacklisted_nodes()),
                'vrf_public_key': self.vrf.public_key.hex()[:16] + "...",
                'data_availability_score': self._calculate_data_availability(),
                'performance_metrics': self._get_performance_metrics()
            })
            
            # Role-specific status
            if self.role == NodeRole.SUPERVISOR and self.dkg:
                status.update({
                    'dkg_state': self.dkg.state.value,
                    'has_key_share': self.dkg.my_key_share is not None,
                    'pending_consensus_proposals': len(self.supervisor_consensus.pending_transactions) if self.supervisor_consensus else 0
                })
            
            if self.miner:
                status.update({
                    'pending_commitments': len(self.commitment_system.get_pending_commitments(self.node_id)),
                    'fulfilled_commitments': len([c for c in self.commitment_system.commitment_history if c['miner_id'] == self.node_id])
                })
                
        except Exception as e:
            self.logger.error(f"Error getting extended status: {e}")
        
        return status
    
    def _calculate_data_availability(self) -> float:
        """Calculate overall data availability score"""
        try:
            # This would calculate availability across all managed data
            # For now, return a mock score
            return 0.95
        except Exception:
            return 0.0
    
    def _get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for this node"""
        try:
            return {
                'uptime_ratio': 0.99,  # Mock value
                'average_response_time': 0.15,  # Mock value in seconds
                'successful_iterations': 0.98,  # Mock success rate
                'bandwidth_utilization': 0.45  # Mock utilization ratio
            }
        except Exception:
            return {}
    
    async def get_network_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        try:
            report = {
                'timestamp': int(time.time()),
                'node_id': self.node_id,
                'total_alerts': len(self.security_alerts),
                'alert_breakdown': {},
                'blacklisted_nodes': self.attack_mitigation.get_blacklisted_nodes(),
                'suspicious_nodes': self.attack_mitigation.poisoning_detector.get_suspicious_nodes(),
                'network_health': 'good'  # Would be calculated based on metrics
            }
            
            # Breakdown alerts by type
            for alert in self.security_alerts[-100:]:  # Last 100 alerts
                alert_type = alert.alert_type.value
                if alert_type not in report['alert_breakdown']:
                    report['alert_breakdown'][alert_type] = 0
                report['alert_breakdown'][alert_type] += 1
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating security report: {e}")
            return {'error': str(e)}
    
    async def store_dataset_securely(self, dataset_id: str, data: bytes, metadata: Dict[str, Any]) -> bool:
        """Store dataset with Reed-Solomon encoding and distribution"""
        try:
            shard_ids = self.data_manager.store_data(
                data_id=dataset_id,
                data=data,
                data_type=DataShardType.TRAINING_DATA,
                metadata=metadata
            )
            
            self.logger.info(f"Stored dataset {dataset_id} as {len(shard_ids)} shards")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing dataset: {e}")
            return False
    
    async def retrieve_dataset(self, dataset_id: str) -> Optional[bytes]:
        """Retrieve dataset from distributed storage"""
        try:
            data = self.data_manager.retrieve_data(dataset_id)
            if data:
                self.logger.info(f"Retrieved dataset {dataset_id} ({len(data)} bytes)")
            return data
            
        except Exception as e:
            self.logger.error(f"Error retrieving dataset: {e}")
            return None
