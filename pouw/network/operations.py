"""
Network Operations for PoUW.

Implements crash-recovery, worker replacement, leader election,
message history compression, and VPN mesh topology support.
"""

import asyncio
import time
import json
import logging
import hashlib
import zlib
import pickle
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import heapq
from collections import defaultdict, deque

from .communication import NetworkMessage, P2PNode


class NodeStatus(Enum):
    """Node operational status"""
    ONLINE = "online"
    OFFLINE = "offline"
    SUSPECTED = "suspected"
    RECOVERING = "recovering"
    FAILED = "failed"


class LeaderElectionState(Enum):
    """Leader election states"""
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


@dataclass
class NodeHealthMetrics:
    """Health metrics for network nodes"""
    node_id: str
    last_heartbeat: float
    response_time: float
    success_rate: float
    task_completion_rate: float
    bandwidth_utilization: float
    cpu_usage: float
    memory_usage: float
    status: NodeStatus = NodeStatus.ONLINE
    
    def is_healthy(self) -> bool:
        """Check if node is considered healthy"""
        current_time = time.time()
        heartbeat_delay = current_time - self.last_heartbeat
        
        return (
            heartbeat_delay < 30.0 and  # Less than 30 seconds since last heartbeat
            self.success_rate > 0.8 and  # Success rate above 80%
            self.response_time < 5.0 and  # Response time under 5 seconds
            self.cpu_usage < 0.9 and  # CPU usage under 90%
            self.memory_usage < 0.9  # Memory usage under 90%
        )


@dataclass
class CrashRecoveryEvent:
    """Crash recovery event data"""
    node_id: str
    event_type: str  # 'crash_detected', 'recovery_started', 'recovery_completed'
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LeaderElectionVote:
    """Vote in leader election"""
    voter_id: str
    candidate_id: str
    term: int
    timestamp: float


@dataclass
class CompressedMessageBatch:
    """Compressed batch of network messages"""
    batch_id: str
    start_time: float
    end_time: float
    message_count: int
    compressed_data: bytes
    compression_ratio: float


class CrashRecoveryManager:
    """
    Manages crash detection and recovery for network nodes.
    
    Implements sophisticated failure detection, automatic recovery,
    and network resilience mechanisms.
    """
    
    def __init__(self, node_id: str, failure_detector_threshold: float = 10.0):
        self.node_id = node_id
        self.failure_detector_threshold = failure_detector_threshold
        
        # Node health tracking
        self.node_metrics: Dict[str, NodeHealthMetrics] = {}
        self.suspected_nodes: Set[str] = set()
        self.failed_nodes: Set[str] = set()
        self.recovering_nodes: Set[str] = set()
        
        # Recovery event history
        self.recovery_events: List[CrashRecoveryEvent] = []
        self.recovery_strategies: Dict[str, str] = {}  # node_id -> strategy
        
        # Heartbeat management
        self.heartbeat_interval = 5.0  # seconds
        self.heartbeat_timeout = 15.0  # seconds
        self.last_heartbeats: Dict[str, float] = {}
        
        # Failure detection parameters
        self.phi_threshold = 8.0  # Phi accrual failure detector threshold
        self.sampling_window = 100  # Number of samples for phi calculation
        self.heartbeat_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.sampling_window))
        
        self.logger = logging.getLogger(f"CrashRecovery-{node_id}")
    
    def update_node_health(self, node_id: str, metrics: NodeHealthMetrics) -> None:
        """Update health metrics for a node"""
        self.node_metrics[node_id] = metrics
        self.last_heartbeats[node_id] = metrics.last_heartbeat
        
        # Update heartbeat history for phi accrual detector
        current_time = time.time()
        if node_id in self.heartbeat_history:
            last_heartbeat = self.heartbeat_history[node_id][-1] if self.heartbeat_history[node_id] else current_time
            interval = current_time - last_heartbeat
            self.heartbeat_history[node_id].append(interval)
        else:
            self.heartbeat_history[node_id].append(0.0)
        
        # Check if node status changed
        previous_status = self.node_metrics.get(node_id, NodeHealthMetrics(
            node_id=node_id, last_heartbeat=0, response_time=0, success_rate=0,
            task_completion_rate=0, bandwidth_utilization=0, cpu_usage=0, memory_usage=0
        )).status
        
        new_status = self._determine_node_status(node_id, metrics)
        
        if previous_status != new_status:
            self._handle_status_change(node_id, previous_status, new_status)
    
    def _determine_node_status(self, node_id: str, metrics: NodeHealthMetrics) -> NodeStatus:
        """Determine node status based on metrics and phi accrual detector"""
        current_time = time.time()
        
        # Check if node is in recovery
        if node_id in self.recovering_nodes:
            if metrics.is_healthy() and current_time - metrics.last_heartbeat < 10.0:
                return NodeStatus.ONLINE
            else:
                return NodeStatus.RECOVERING
        
        # Check for explicit failure
        if node_id in self.failed_nodes:
            return NodeStatus.FAILED
        
        # Calculate phi value for failure detection
        phi_value = self._calculate_phi_value(node_id)
        
        if phi_value > self.phi_threshold:
            return NodeStatus.OFFLINE
        elif phi_value > self.phi_threshold * 0.5:
            return NodeStatus.SUSPECTED
        elif metrics.is_healthy():
            return NodeStatus.ONLINE
        else:
            return NodeStatus.SUSPECTED
    
    def _calculate_phi_value(self, node_id: str) -> float:
        """Calculate phi value for accrual failure detector"""
        if node_id not in self.heartbeat_history or len(self.heartbeat_history[node_id]) < 2:
            return 0.0
        
        intervals = list(self.heartbeat_history[node_id])
        current_time = time.time()
        last_heartbeat = self.last_heartbeats.get(node_id, current_time)
        time_since_last = current_time - last_heartbeat
        
        # Calculate mean and standard deviation of intervals
        mean_interval = sum(intervals) / len(intervals)
        variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
        std_dev = variance ** 0.5
        
        if std_dev == 0:
            std_dev = 1.0  # Avoid division by zero
        
        # Calculate phi value using normal distribution
        phi = abs(time_since_last - mean_interval) / std_dev
        return phi
    
    def _handle_status_change(self, node_id: str, old_status: NodeStatus, new_status: NodeStatus) -> None:
        """Handle node status changes"""
        self.logger.info(f"Node {node_id} status changed: {old_status.value} -> {new_status.value}")
        
        # Record recovery event
        event = CrashRecoveryEvent(
            node_id=node_id,
            event_type=f"status_change_{new_status.value}",
            timestamp=time.time(),
            metadata={"old_status": old_status.value, "new_status": new_status.value}
        )
        self.recovery_events.append(event)
        
        # Update tracking sets
        if new_status == NodeStatus.SUSPECTED:
            self.suspected_nodes.add(node_id)
        elif new_status == NodeStatus.OFFLINE:
            self.suspected_nodes.discard(node_id)
            self.failed_nodes.add(node_id)
        elif new_status == NodeStatus.RECOVERING:
            self.recovering_nodes.add(node_id)
            self.failed_nodes.discard(node_id)
        elif new_status == NodeStatus.ONLINE:
            self.suspected_nodes.discard(node_id)
            self.failed_nodes.discard(node_id)
            self.recovering_nodes.discard(node_id)
    
    def detect_crashes(self) -> List[str]:
        """Detect crashed nodes and return list of node IDs"""
        current_time = time.time()
        crashed_nodes = []
        
        for node_id, last_heartbeat in self.last_heartbeats.items():
            if current_time - last_heartbeat > self.heartbeat_timeout:
                phi_value = self._calculate_phi_value(node_id)
                
                if phi_value > self.phi_threshold and node_id not in self.failed_nodes:
                    crashed_nodes.append(node_id)
                    self.failed_nodes.add(node_id)
                    
                    # Record crash detection event
                    event = CrashRecoveryEvent(
                        node_id=node_id,
                        event_type="crash_detected",
                        timestamp=current_time,
                        metadata={"phi_value": phi_value, "heartbeat_delay": current_time - last_heartbeat}
                    )
                    self.recovery_events.append(event)
                    
                    self.logger.warning(f"Crash detected for node {node_id} (phi={phi_value:.2f})")
        
        return crashed_nodes
    
    def initiate_recovery(self, node_id: str, recovery_strategy: str = "automatic") -> bool:
        """Initiate recovery for a failed node"""
        if node_id not in self.failed_nodes and node_id not in self.suspected_nodes:
            return False
        
        self.recovering_nodes.add(node_id)
        self.recovery_strategies[node_id] = recovery_strategy
        
        # Record recovery initiation
        event = CrashRecoveryEvent(
            node_id=node_id,
            event_type="recovery_started",
            timestamp=time.time(),
            metadata={"strategy": recovery_strategy}
        )
        self.recovery_events.append(event)
        
        self.logger.info(f"Recovery initiated for node {node_id} with strategy: {recovery_strategy}")
        return True
    
    def get_network_health_summary(self) -> Dict[str, Any]:
        """Get summary of network health status"""
        total_nodes = len(self.node_metrics)
        online_nodes = sum(1 for m in self.node_metrics.values() if m.status == NodeStatus.ONLINE)
        suspected_nodes = len(self.suspected_nodes)
        failed_nodes = len(self.failed_nodes)
        recovering_nodes = len(self.recovering_nodes)
        
        return {
            "total_nodes": total_nodes,
            "online_nodes": online_nodes,
            "suspected_nodes": suspected_nodes,
            "failed_nodes": failed_nodes,
            "recovering_nodes": recovering_nodes,
            "network_health_ratio": online_nodes / max(1, total_nodes),
            "recent_crashes": len([e for e in self.recovery_events 
                                 if e.event_type == "crash_detected" and 
                                 time.time() - e.timestamp < 3600])  # Last hour
        }


class WorkerReplacementManager:
    """
    Manages worker node replacement in case of failures.
    
    Implements dynamic worker allocation, load balancing,
    and seamless task migration.
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.worker_pools: Dict[str, List[str]] = {}  # task_type -> [worker_ids]
        self.worker_assignments: Dict[str, str] = {}  # worker_id -> task_id
        self.backup_workers: Dict[str, List[str]] = {}  # task_id -> [backup_worker_ids]
        self.replacement_history: List[Dict[str, Any]] = []
        
        self.logger = logging.getLogger(f"WorkerReplacement-{node_id}")
    
    def register_worker_pool(self, task_type: str, worker_ids: List[str]) -> None:
        """Register a pool of workers for a specific task type"""
        self.worker_pools[task_type] = worker_ids.copy()
        self.logger.info(f"Registered worker pool for {task_type}: {len(worker_ids)} workers")
    
    def assign_workers_to_task(self, task_id: str, task_type: str, required_workers: int,
                             backup_workers: int = 2) -> Tuple[List[str], List[str]]:
        """Assign workers and backup workers to a task"""
        if task_type not in self.worker_pools:
            raise ValueError(f"No worker pool found for task type: {task_type}")
        
        available_workers = [w for w in self.worker_pools[task_type] 
                           if w not in self.worker_assignments]
        
        if len(available_workers) < required_workers + backup_workers:
            raise ValueError(f"Insufficient workers available. Required: {required_workers + backup_workers}, Available: {len(available_workers)}")
        
        # Assign primary workers
        primary_workers = available_workers[:required_workers]
        backup_worker_list = available_workers[required_workers:required_workers + backup_workers]
        
        # Update assignments
        for worker_id in primary_workers:
            self.worker_assignments[worker_id] = task_id
        
        self.backup_workers[task_id] = backup_worker_list
        
        self.logger.info(f"Assigned {len(primary_workers)} workers and {len(backup_worker_list)} backups to task {task_id}")
        return primary_workers, backup_worker_list
    
    def replace_failed_worker(self, failed_worker_id: str, task_id: str) -> Optional[str]:
        """Replace a failed worker with a backup worker"""
        if task_id not in self.backup_workers or not self.backup_workers[task_id]:
            self.logger.error(f"No backup workers available for task {task_id}")
            return None
        
        # Get replacement worker
        replacement_worker = self.backup_workers[task_id].pop(0)
        
        # Update assignments
        self.worker_assignments[replacement_worker] = task_id
        if failed_worker_id in self.worker_assignments:
            del self.worker_assignments[failed_worker_id]
        
        # Record replacement
        replacement_record = {
            "timestamp": time.time(),
            "task_id": task_id,
            "failed_worker": failed_worker_id,
            "replacement_worker": replacement_worker,
            "remaining_backups": len(self.backup_workers[task_id])
        }
        self.replacement_history.append(replacement_record)
        
        self.logger.info(f"Replaced failed worker {failed_worker_id} with {replacement_worker} for task {task_id}")
        return replacement_worker
    
    def release_task_workers(self, task_id: str) -> None:
        """Release all workers assigned to a completed task"""
        # Release primary workers
        workers_to_release = [w for w, t in self.worker_assignments.items() if t == task_id]
        for worker_id in workers_to_release:
            del self.worker_assignments[worker_id]
        
        # Release backup workers
        if task_id in self.backup_workers:
            del self.backup_workers[task_id]
        
        self.logger.info(f"Released {len(workers_to_release)} workers from completed task {task_id}")
    
    def get_worker_utilization_stats(self) -> Dict[str, Any]:
        """Get worker utilization statistics"""
        total_workers = sum(len(pool) for pool in self.worker_pools.values())
        assigned_workers = len(self.worker_assignments)
        total_backups = sum(len(backups) for backups in self.backup_workers.values())
        
        return {
            "total_workers": total_workers,
            "assigned_workers": assigned_workers,
            "backup_workers": total_backups,
            "utilization_rate": assigned_workers / max(1, total_workers),
            "active_tasks": len(self.backup_workers),
            "recent_replacements": len([r for r in self.replacement_history 
                                      if time.time() - r["timestamp"] < 3600])
        }


class LeaderElectionManager:
    """
    Implements leader election for supervisor nodes using Raft-like algorithm.
    
    Manages leader election, term management, and consensus among supervisors.
    """
    
    def __init__(self, node_id: str, supervisor_nodes: List[str]):
        self.node_id = node_id
        self.supervisor_nodes = supervisor_nodes.copy()
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.state = LeaderElectionState.FOLLOWER
        self.leader_id: Optional[str] = None
        
        # Election timing
        self.election_timeout = 5.0  # seconds
        self.heartbeat_interval = 1.0  # seconds
        self.last_heartbeat = time.time()
        self.election_start_time: Optional[float] = None
        
        # Vote tracking
        self.received_votes: Set[str] = set()
        self.vote_history: List[LeaderElectionVote] = []
        
        # Leader responsibilities
        self.is_leader_active = False
        self.leader_heartbeat_task: Optional[asyncio.Task] = None
        
        self.logger = logging.getLogger(f"LeaderElection-{node_id}")
    
    def start_election(self) -> bool:
        """Start a new leader election"""
        if self.state == LeaderElectionState.LEADER:
            return False  # Already leader
        
        self.current_term += 1
        self.state = LeaderElectionState.CANDIDATE
        self.voted_for = self.node_id
        self.received_votes = {self.node_id}  # Vote for self
        self.election_start_time = time.time()
        
        self.logger.info(f"Starting leader election for term {self.current_term}")
        return True
    
    def request_vote(self, candidate_id: str, candidate_term: int) -> bool:
        """Process a vote request from a candidate"""
        current_time = time.time()
        
        # Reject if candidate term is older
        if candidate_term < self.current_term:
            self.logger.info(f"Rejected vote request from {candidate_id}: term {candidate_term} < {self.current_term}")
            return False
        
        # Update term if candidate has newer term
        if candidate_term > self.current_term:
            self.current_term = candidate_term
            self.voted_for = None
            self.state = LeaderElectionState.FOLLOWER
            self.leader_id = None
        
        # Grant vote if haven't voted in this term or already voted for this candidate
        if self.voted_for is None or self.voted_for == candidate_id:
            self.voted_for = candidate_id
            
            # Record vote
            vote = LeaderElectionVote(
                voter_id=self.node_id,
                candidate_id=candidate_id,
                term=candidate_term,
                timestamp=current_time
            )
            self.vote_history.append(vote)
            
            self.logger.info(f"Granted vote to {candidate_id} for term {candidate_term}")
            return True
        
        self.logger.info(f"Rejected vote request from {candidate_id}: already voted for {self.voted_for}")
        return False
    
    def receive_vote_response(self, voter_id: str, granted: bool) -> bool:
        """Process vote response and check if won election"""
        if self.state != LeaderElectionState.CANDIDATE:
            return False
        
        if granted:
            self.received_votes.add(voter_id)
            self.logger.info(f"Received vote from {voter_id}, total votes: {len(self.received_votes)}")
        
        # Check if won majority
        majority_threshold = len(self.supervisor_nodes) // 2 + 1
        if len(self.received_votes) >= majority_threshold:
            self._become_leader()
            return True
        
        return False
    
    def _become_leader(self) -> None:
        """Transition to leader state"""
        self.state = LeaderElectionState.LEADER
        self.leader_id = self.node_id
        self.is_leader_active = True
        
        self.logger.info(f"Became leader for term {self.current_term}")
        
        # Start sending heartbeats
        if self.leader_heartbeat_task:
            self.leader_heartbeat_task.cancel()
        
        # Note: In a real implementation, this would start an async task
        # self.leader_heartbeat_task = asyncio.create_task(self._send_heartbeats())
    
    def receive_heartbeat(self, leader_id: str, leader_term: int) -> bool:
        """Process heartbeat from leader"""
        current_time = time.time()
        
        # Reject if term is older
        if leader_term < self.current_term:
            return False
        
        # Update term and recognize leader
        if leader_term >= self.current_term:
            self.current_term = leader_term
            self.leader_id = leader_id
            self.state = LeaderElectionState.FOLLOWER
            self.last_heartbeat = current_time
            self.voted_for = None  # Reset vote for new term
        
        return True
    
    def check_election_timeout(self) -> bool:
        """Check if election timeout has occurred"""
        current_time = time.time()
        
        if self.state == LeaderElectionState.LEADER:
            return False  # Leaders don't timeout
        
        time_since_heartbeat = current_time - self.last_heartbeat
        return time_since_heartbeat > self.election_timeout
    
    def step_down(self) -> None:
        """Step down from leadership"""
        if self.state == LeaderElectionState.LEADER:
            self.state = LeaderElectionState.FOLLOWER
            self.is_leader_active = False
            self.leader_id = None
            
            if self.leader_heartbeat_task:
                self.leader_heartbeat_task.cancel()
                self.leader_heartbeat_task = None
            
            self.logger.info(f"Stepped down from leadership")
    
    def get_election_status(self) -> Dict[str, Any]:
        """Get current election status"""
        return {
            "node_id": self.node_id,
            "state": self.state.value,
            "current_term": self.current_term,
            "leader_id": self.leader_id,
            "voted_for": self.voted_for,
            "is_leader": self.state == LeaderElectionState.LEADER,
            "vote_count": len(self.received_votes) if self.state == LeaderElectionState.CANDIDATE else 0,
            "election_in_progress": self.state == LeaderElectionState.CANDIDATE,
            "time_since_last_heartbeat": time.time() - self.last_heartbeat
        }


class MessageHistoryCompressor:
    """
    Manages compression and storage of message history.
    
    Implements efficient compression algorithms, batch processing,
    and archival storage for network messages.
    """
    
    def __init__(self, node_id: str, compression_threshold: int = 1000, 
                 max_batch_size: int = 10000):
        self.node_id = node_id
        self.compression_threshold = compression_threshold
        self.max_batch_size = max_batch_size
        
        # Message storage
        self.message_buffer: List[NetworkMessage] = []
        self.compressed_batches: List[CompressedMessageBatch] = []
        self.compression_stats = {
            "total_messages": 0,
            "total_compressed": 0,
            "total_raw_bytes": 0,
            "total_compressed_bytes": 0,
            "average_compression_ratio": 0.0
        }
        
        # Archival settings
        self.archive_threshold = 100  # Number of compressed batches before archival
        self.archived_batches: List[str] = []  # Paths to archived files
        
        self.logger = logging.getLogger(f"MessageCompressor-{node_id}")
    
    def add_message(self, message: NetworkMessage) -> None:
        """Add a message to the buffer for compression"""
        self.message_buffer.append(message)
        self.compression_stats["total_messages"] += 1
        
        # Trigger compression if threshold reached
        if len(self.message_buffer) >= self.compression_threshold:
            self.compress_message_batch()
    
    def compress_message_batch(self) -> Optional[CompressedMessageBatch]:
        """Compress current message buffer into a batch"""
        if not self.message_buffer:
            return None
        
        start_time = time.time()
        
        # Prepare messages for compression
        messages_data = []
        earliest_time = float('inf')
        latest_time = 0.0
        
        for msg in self.message_buffer:
            msg_dict = {
                "msg_type": msg.msg_type,
                "sender_id": msg.sender_id,
                "timestamp": msg.timestamp,
                "data": msg.data
            }
            messages_data.append(msg_dict)
            earliest_time = min(earliest_time, msg.timestamp)
            latest_time = max(latest_time, msg.timestamp)
        
        # Serialize and compress
        raw_data = pickle.dumps(messages_data)
        compressed_data = zlib.compress(raw_data, level=9)
        
        # Calculate compression ratio
        raw_size = len(raw_data)
        compressed_size = len(compressed_data)
        compression_ratio = compressed_size / raw_size if raw_size > 0 else 1.0
        
        # Create compressed batch
        batch_id = hashlib.sha256(f"{self.node_id}_{start_time}_{len(messages_data)}".encode()).hexdigest()[:16]
        batch = CompressedMessageBatch(
            batch_id=batch_id,
            start_time=earliest_time,
            end_time=latest_time,
            message_count=len(messages_data),
            compressed_data=compressed_data,
            compression_ratio=compression_ratio
        )
        
        self.compressed_batches.append(batch)
        
        # Update statistics
        self.compression_stats["total_compressed"] += len(messages_data)
        self.compression_stats["total_raw_bytes"] += raw_size
        self.compression_stats["total_compressed_bytes"] += compressed_size
        self.compression_stats["average_compression_ratio"] = (
            self.compression_stats["total_compressed_bytes"] / 
            max(1, self.compression_stats["total_raw_bytes"])
        )
        
        # Clear buffer
        self.message_buffer.clear()
        
        self.logger.info(f"Compressed batch {batch_id}: {len(messages_data)} messages, "
                        f"ratio: {compression_ratio:.3f}")
        
        # Check if archival is needed
        if len(self.compressed_batches) >= self.archive_threshold:
            self._archive_old_batches()
        
        return batch
    
    def decompress_batch(self, batch: CompressedMessageBatch) -> List[Dict[str, Any]]:
        """Decompress a message batch"""
        try:
            raw_data = zlib.decompress(batch.compressed_data)
            messages_data = pickle.loads(raw_data)
            return messages_data
        except Exception as e:
            self.logger.error(f"Failed to decompress batch {batch.batch_id}: {e}")
            return []
    
    def search_messages(self, start_time: float, end_time: float, 
                       message_type: Optional[str] = None,
                       sender_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for messages within time range and optional filters"""
        matching_messages = []
        
        # Search in current buffer
        for msg in self.message_buffer:
            if start_time <= msg.timestamp <= end_time:
                if message_type and msg.msg_type != message_type:
                    continue
                if sender_id and msg.sender_id != sender_id:
                    continue
                
                matching_messages.append({
                    "msg_type": msg.msg_type,
                    "sender_id": msg.sender_id,
                    "timestamp": msg.timestamp,
                    "data": msg.data
                })
        
        # Search in compressed batches
        for batch in self.compressed_batches:
            if batch.end_time < start_time or batch.start_time > end_time:
                continue  # Batch outside time range
            
            messages = self.decompress_batch(batch)
            for msg_dict in messages:
                if start_time <= msg_dict["timestamp"] <= end_time:
                    if message_type and msg_dict["msg_type"] != message_type:
                        continue
                    if sender_id and msg_dict["sender_id"] != sender_id:
                        continue
                    
                    matching_messages.append(msg_dict)
        
        return sorted(matching_messages, key=lambda x: x["timestamp"])
    
    def _archive_old_batches(self) -> None:
        """Archive old compressed batches to storage"""
        # In a real implementation, this would write to persistent storage
        # For now, we'll just remove old batches to prevent memory bloat
        batches_to_archive = self.compressed_batches[:-50]  # Keep last 50 batches
        self.compressed_batches = self.compressed_batches[-50:]
        
        for batch in batches_to_archive:
            self.archived_batches.append(f"archived_{batch.batch_id}")
        
        self.logger.info(f"Archived {len(batches_to_archive)} message batches")
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics"""
        stats = self.compression_stats.copy()
        stats.update({
            "current_buffer_size": len(self.message_buffer),
            "compressed_batches": len(self.compressed_batches),
            "archived_batches": len(self.archived_batches),
            "space_saved_bytes": stats["total_raw_bytes"] - stats["total_compressed_bytes"],
            "space_saved_percentage": (1 - stats["average_compression_ratio"]) * 100
        })
        return stats


class VPNMeshTopologyManager:
    """
    Manages VPN mesh topology for secure worker communication.
    
    Implements virtual network overlay, secure tunnels,
    and optimized routing for worker nodes.
    """
    
    def __init__(self, node_id: str, network_cidr: str = "10.0.0.0/16"):
        self.node_id = node_id
        self.network_cidr = network_cidr
        
        # Network topology
        self.mesh_nodes: Dict[str, Dict[str, Any]] = {}  # node_id -> node_info
        self.virtual_ips: Dict[str, str] = {}  # node_id -> virtual_ip
        self.routing_table: Dict[str, List[str]] = {}  # destination -> [path_nodes]
        self.tunnels: Dict[str, Dict[str, Any]] = {}  # (node1, node2) -> tunnel_info
        
        # Network metrics
        self.bandwidth_metrics: Dict[str, float] = {}  # node_id -> bandwidth_mbps
        self.latency_metrics: Dict[Tuple[str, str], float] = {}  # (node1, node2) -> latency_ms
        self.tunnel_health: Dict[str, bool] = {}  # tunnel_id -> is_healthy
        
        # Security settings
        self.encryption_enabled = True
        self.tunnel_protocols = ["wireguard", "openvpn", "ipsec"]
        self.preferred_protocol = "wireguard"
        
        self.logger = logging.getLogger(f"VPNMesh-{node_id}")
    
    def join_mesh(self, node_info: Dict[str, Any]) -> str:
        """Join a node to the VPN mesh network"""
        node_id = node_info["node_id"]
        
        # Assign virtual IP
        virtual_ip = self._assign_virtual_ip(node_id)
        self.virtual_ips[node_id] = virtual_ip
        
        # Store node information
        self.mesh_nodes[node_id] = {
            **node_info,
            "virtual_ip": virtual_ip,
            "join_time": time.time(),
            "status": "active"
        }
        
        # Initialize routing entry
        self.routing_table[node_id] = [node_id]  # Direct route to self
        
        # Update mesh topology
        self._update_mesh_topology()
        
        self.logger.info(f"Node {node_id} joined mesh with virtual IP {virtual_ip}")
        return virtual_ip
    
    def leave_mesh(self, node_id: str) -> bool:
        """Remove a node from the VPN mesh network"""
        if node_id not in self.mesh_nodes:
            return False
        
        # Remove tunnels involving this node
        tunnels_to_remove = [tid for tid in self.tunnels.keys() if node_id in tid]
        for tunnel_id in tunnels_to_remove:
            del self.tunnels[tunnel_id]
        
        # Remove routing entries
        if node_id in self.routing_table:
            del self.routing_table[node_id]
        
        # Update other routing entries
        for dest, path in self.routing_table.items():
            if node_id in path:
                self.routing_table[dest] = [n for n in path if n != node_id]
        
        # Remove node information
        del self.mesh_nodes[node_id]
        if node_id in self.virtual_ips:
            del self.virtual_ips[node_id]
        
        # Update mesh topology
        self._update_mesh_topology()
        
        self.logger.info(f"Node {node_id} left mesh")
        return True
    
    def _assign_virtual_ip(self, node_id: str) -> str:
        """Assign a virtual IP address to a node"""
        # Simple IP assignment based on node count
        # In a real implementation, this would use proper CIDR allocation
        base_ip = "10.0.0."
        used_ips = set(self.virtual_ips.values())
        
        for i in range(2, 255):  # Start from .2, reserve .1 for gateway
            candidate_ip = f"{base_ip}{i}"
            if candidate_ip not in used_ips:
                return candidate_ip
        
        raise ValueError("No available IP addresses in mesh network")
    
    def establish_tunnel(self, remote_node_id: str, protocol: Optional[str] = None) -> bool:
        """Establish a VPN tunnel to a remote node"""
        if remote_node_id not in self.mesh_nodes:
            self.logger.error(f"Cannot establish tunnel: unknown node {remote_node_id}")
            return False
        
        protocol = protocol or self.preferred_protocol
        tunnel_id = f"{self.node_id}_{remote_node_id}"
        
        # Check if tunnel already exists
        if tunnel_id in self.tunnels or f"{remote_node_id}_{self.node_id}" in self.tunnels:
            return True
        
        # Tunnel configuration
        tunnel_config = {
            "protocol": protocol,
            "local_node": self.node_id,
            "remote_node": remote_node_id,
            "local_ip": self.virtual_ips.get(self.node_id),
            "remote_ip": self.virtual_ips.get(remote_node_id),
            "established_time": time.time(),
            "status": "active",
            "encryption": self.encryption_enabled
        }
        
        # Store tunnel information
        self.tunnels[tunnel_id] = tunnel_config
        self.tunnel_health[tunnel_id] = True
        
        # Update routing table
        self._update_routing_for_tunnel(remote_node_id)
        
        self.logger.info(f"Established {protocol} tunnel to {remote_node_id}")
        return True
    
    def _update_mesh_topology(self) -> None:
        """Update mesh topology and routing tables"""
        # Implement shortest path routing using Dijkstra's algorithm
        for destination in self.mesh_nodes.keys():
            if destination == self.node_id:
                continue
            
            # Find shortest path considering latency
            path = self._find_shortest_path(self.node_id, destination)
            if path:
                self.routing_table[destination] = path
    
    def _find_shortest_path(self, source: str, destination: str) -> Optional[List[str]]:
        """Find shortest path between two nodes using Dijkstra's algorithm"""
        # Simplified implementation for demonstration
        # In a real implementation, this would consider actual network topology
        
        if destination in self.mesh_nodes:
            # Direct connection if possible
            tunnel_id = f"{source}_{destination}"
            if tunnel_id in self.tunnels and self.tunnel_health.get(tunnel_id, False):
                return [source, destination]
            
            # Multi-hop routing through other nodes
            for intermediate in self.mesh_nodes.keys():
                if intermediate != source and intermediate != destination:
                    tunnel1 = f"{source}_{intermediate}"
                    tunnel2 = f"{intermediate}_{destination}"
                    
                    if (tunnel1 in self.tunnels and self.tunnel_health.get(tunnel1, False) and
                        tunnel2 in self.tunnels and self.tunnel_health.get(tunnel2, False)):
                        return [source, intermediate, destination]
        
        return None
    
    def _update_routing_for_tunnel(self, remote_node_id: str) -> None:
        """Update routing table when a new tunnel is established"""
        # Direct route through the new tunnel
        self.routing_table[remote_node_id] = [self.node_id, remote_node_id]
        
        # Update routes to other nodes that might benefit from this tunnel
        self._update_mesh_topology()
    
    def monitor_tunnel_health(self) -> Dict[str, Any]:
        """Monitor health of all VPN tunnels"""
        health_report = {
            "total_tunnels": len(self.tunnels),
            "healthy_tunnels": sum(1 for health in self.tunnel_health.values() if health),
            "tunnel_details": {}
        }
        
        for tunnel_id, tunnel_config in self.tunnels.items():
            # Simulate health check (in real implementation, this would ping/test tunnel)
            is_healthy = self.tunnel_health.get(tunnel_id, False)
            
            health_report["tunnel_details"][tunnel_id] = {
                "protocol": tunnel_config["protocol"],
                "remote_node": tunnel_config["remote_node"],
                "status": tunnel_config["status"],
                "healthy": is_healthy,
                "uptime": time.time() - tunnel_config["established_time"]
            }
        
        return health_report
    
    def get_mesh_topology(self) -> Dict[str, Any]:
        """Get current mesh topology information"""
        return {
            "node_id": self.node_id,
            "virtual_ip": self.virtual_ips.get(self.node_id),
            "mesh_nodes": len(self.mesh_nodes),
            "active_tunnels": len(self.tunnels),
            "routing_table": self.routing_table,
            "network_health": {
                "connectivity": len(self.tunnels) / max(1, len(self.mesh_nodes) - 1),
                "average_latency": sum(self.latency_metrics.values()) / max(1, len(self.latency_metrics)),
                "total_bandwidth": sum(self.bandwidth_metrics.values())
            }
        }


class NetworkOperationsManager:
    """
    Unified manager for all network operations.
    
    Integrates crash recovery, worker replacement, leader election,
    message compression, and VPN mesh topology management.
    """
    
    def __init__(self, node_id: str, role: str, supervisor_nodes: Optional[List[str]] = None):
        self.node_id = node_id
        self.role = role
        
        # Initialize all network operations components
        self.crash_recovery = CrashRecoveryManager(node_id)
        self.worker_replacement = WorkerReplacementManager(node_id)
        self.message_compressor = MessageHistoryCompressor(node_id)
        self.vpn_mesh = VPNMeshTopologyManager(node_id)
        
        # Leader election only for supervisors
        self.leader_election = None
        if role == "supervisor" and supervisor_nodes is not None:
            self.leader_election = LeaderElectionManager(node_id, supervisor_nodes)
        
        # Operation scheduling
        self.operation_tasks: List[asyncio.Task] = []
        self.is_running = False
        
        self.logger = logging.getLogger(f"NetworkOps-{node_id}")
    
    async def start_operations(self) -> None:
        """Start all network operations"""
        self.is_running = True
        
        # Start monitoring tasks
        self.operation_tasks.append(
            asyncio.create_task(self._monitoring_loop())
        )
        
        if self.leader_election:
            self.operation_tasks.append(
                asyncio.create_task(self._election_loop())
            )
        
        self.logger.info("Network operations started")
    
    async def stop_operations(self) -> None:
        """Stop all network operations"""
        self.is_running = False
        
        # Cancel all tasks
        for task in self.operation_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.operation_tasks, return_exceptions=True)
        self.operation_tasks.clear()
        
        self.logger.info("Network operations stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop for network operations"""
        while self.is_running:
            try:
                # Check for crashed nodes
                crashed_nodes = self.crash_recovery.detect_crashes()
                for node_id in crashed_nodes:
                    await self._handle_node_crash(node_id)
                
                # Compress messages if needed
                if len(self.message_compressor.message_buffer) >= self.message_compressor.compression_threshold:
                    self.message_compressor.compress_message_batch()
                
                # Monitor VPN tunnel health
                tunnel_health = self.vpn_mesh.monitor_tunnel_health()
                await self._handle_tunnel_issues(tunnel_health)
                
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _election_loop(self) -> None:
        """Election management loop for supervisors"""
        if not self.leader_election:
            return
        
        while self.is_running:
            try:
                # Check for election timeout
                if self.leader_election.check_election_timeout():
                    if self.leader_election.start_election():
                        await self._conduct_election()
                
                await asyncio.sleep(1.0)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Error in election loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _handle_node_crash(self, crashed_node_id: str) -> None:
        """Handle detected node crash"""
        self.logger.warning(f"Handling crash of node {crashed_node_id}")
        
        # Initiate recovery
        self.crash_recovery.initiate_recovery(crashed_node_id)
        
        # Replace workers if needed
        affected_tasks = [task_id for worker_id, task_id in self.worker_replacement.worker_assignments.items() 
                         if worker_id == crashed_node_id]
        
        for task_id in affected_tasks:
            replacement = self.worker_replacement.replace_failed_worker(crashed_node_id, task_id)
            if replacement:
                self.logger.info(f"Replaced crashed worker {crashed_node_id} with {replacement} for task {task_id}")
        
        # Remove from VPN mesh
        self.vpn_mesh.leave_mesh(crashed_node_id)
    
    async def _handle_tunnel_issues(self, tunnel_health: Dict[str, Any]) -> None:
        """Handle VPN tunnel health issues"""
        for tunnel_id, details in tunnel_health.get("tunnel_details", {}).items():
            if not details["healthy"]:
                self.logger.warning(f"Unhealthy tunnel detected: {tunnel_id}")
                # In a real implementation, this would attempt to re-establish the tunnel
    
    async def _conduct_election(self) -> None:
        """Conduct leader election process"""
        if not self.leader_election:
            return
        
        # In a real implementation, this would send vote requests to other supervisors
        # and wait for responses. For demonstration, we'll simulate the process.
        self.logger.info(f"Conducting leader election for term {self.leader_election.current_term}")
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get comprehensive network status"""
        status = {
            "node_id": self.node_id,
            "role": self.role,
            "is_running": self.is_running,
            "crash_recovery": self.crash_recovery.get_network_health_summary(),
            "worker_replacement": self.worker_replacement.get_worker_utilization_stats(),
            "message_compression": self.message_compressor.get_compression_stats(),
            "vpn_mesh": self.vpn_mesh.get_mesh_topology()
        }
        
        if self.leader_election:
            status["leader_election"] = self.leader_election.get_election_status()
        
        return status
