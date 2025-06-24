#!/usr/bin/env python3
"""
Corrected test suite for PoUW Network Operations.

Tests network operation components with the correct interface.
"""

import asyncio
import unittest
import time

from pouw.network.operations import (
    NodeStatus, LeaderElectionState, NodeHealthMetrics,
    CrashRecoveryManager, WorkerReplacementManager, LeaderElectionManager,
    MessageHistoryCompressor, VPNMeshTopologyManager, NetworkOperationsManager
)
from pouw.network.communication import NetworkMessage


class TestCrashRecoveryManager(unittest.TestCase):
    """Test crash recovery functionality"""
    
    def setUp(self):
        self.manager = CrashRecoveryManager("test_node")
    
    def test_health_metrics_update(self):
        """Test updating node health metrics"""
        node_id = "worker_001"
        
        # Create health metrics
        metrics = NodeHealthMetrics(
            node_id=node_id,
            last_heartbeat=time.time(),
            response_time=0.1,
            success_rate=0.95,
            task_completion_rate=0.90,
            bandwidth_utilization=0.5,
            cpu_usage=0.3,
            memory_usage=0.4
        )
        
        # Update health
        self.manager.update_node_health(node_id, metrics)
        
        # Verify storage
        self.assertIn(node_id, self.manager.node_metrics)
        stored_metrics = self.manager.node_metrics[node_id]
        self.assertEqual(stored_metrics.node_id, node_id)
        self.assertEqual(stored_metrics.success_rate, 0.95)
    
    def test_crash_detection(self):
        """Test crash detection mechanism"""
        # Should return empty list when no crashes
        crashes = self.manager.detect_crashes()
        self.assertIsInstance(crashes, list)
    
    def test_recovery_initiation(self):
        """Test recovery process initiation"""
        node_id = "failed_worker"
        
        # Add to failed nodes first
        self.manager.failed_nodes.add(node_id)
        
        # Test recovery initiation
        result = self.manager.initiate_recovery(node_id)
        self.assertTrue(result)
        self.assertIn(node_id, self.manager.recovering_nodes)
    
    def test_health_summary(self):
        """Test network health summary"""
        # Add some metrics
        for i in range(3):
            node_id = f"worker_{i:03d}"
            metrics = NodeHealthMetrics(
                node_id=node_id,
                last_heartbeat=time.time(),
                response_time=0.1,
                success_rate=0.95,
                task_completion_rate=0.90,
                bandwidth_utilization=0.5,
                cpu_usage=0.3,
                memory_usage=0.4
            )
            self.manager.update_node_health(node_id, metrics)
        
        summary = self.manager.get_network_health_summary()
        
        # Verify summary structure
        expected_keys = [
            'total_nodes', 'online_nodes', 'suspected_nodes',
            'failed_nodes', 'recovering_nodes', 'network_health_ratio'
        ]
        for key in expected_keys:
            self.assertIn(key, summary)
        
        self.assertEqual(summary['total_nodes'], 3)


class TestWorkerReplacementManager(unittest.TestCase):
    """Test worker replacement functionality"""
    
    def setUp(self):
        self.manager = WorkerReplacementManager("supervisor_001")
    
    def test_worker_pool_registration(self):
        """Test registering worker pools"""
        task_type = "ml_training"
        workers = ["worker_001", "worker_002", "worker_003"]
        
        self.manager.register_worker_pool(task_type, workers)
        
        self.assertIn(task_type, self.manager.worker_pools)
        self.assertEqual(self.manager.worker_pools[task_type], workers)
    
    def test_worker_assignment(self):
        """Test assigning workers to tasks"""
        # Setup worker pool
        task_type = "ml_training"
        workers = ["worker_001", "worker_002", "worker_003", "worker_004"]
        self.manager.register_worker_pool(task_type, workers)
        
        # Assign workers
        task_id = "task_001"
        primary, backup = self.manager.assign_workers_to_task(
            task_id, task_type, required_workers=1, backup_workers=1
        )
        
        self.assertEqual(len(primary), 1)
        self.assertEqual(len(backup), 1)
        self.assertIn(primary[0], self.manager.worker_assignments)
        self.assertIn(task_id, self.manager.backup_workers)
    
    def test_worker_replacement(self):
        """Test replacing failed workers"""
        # Setup
        task_type = "ml_training"
        workers = ["worker_001", "worker_002", "worker_003"]
        self.manager.register_worker_pool(task_type, workers)
        
        task_id = "task_001"
        primary, backup = self.manager.assign_workers_to_task(
            task_id, task_type, required_workers=1, backup_workers=1
        )
        
        # Replace failed worker
        failed_worker = primary[0]
        replacement = self.manager.replace_failed_worker(failed_worker, task_id)
        
        self.assertIsNotNone(replacement)
        self.assertIn(replacement, self.manager.worker_assignments)
        self.assertNotIn(failed_worker, self.manager.worker_assignments)
    
    def test_utilization_stats(self):
        """Test worker utilization statistics"""
        # Setup worker pool
        task_type = "ml_training"
        workers = ["worker_001", "worker_002", "worker_003"]
        self.manager.register_worker_pool(task_type, workers)
        
        stats = self.manager.get_worker_utilization_stats()
        
        expected_keys = [
            'total_workers', 'assigned_workers', 'backup_workers',
            'utilization_rate', 'active_tasks'
        ]
        for key in expected_keys:
            self.assertIn(key, stats)


class TestLeaderElectionManager(unittest.TestCase):
    """Test leader election functionality"""
    
    def setUp(self):
        supervisors = ["supervisor_001", "supervisor_002", "supervisor_003"]
        self.manager = LeaderElectionManager("supervisor_001", supervisors)
    
    def test_initialization(self):
        """Test manager initialization"""
        self.assertEqual(self.manager.node_id, "supervisor_001")
        self.assertEqual(self.manager.state, LeaderElectionState.FOLLOWER)
        self.assertEqual(self.manager.current_term, 0)
    
    def test_election_start(self):
        """Test starting an election"""
        result = self.manager.start_election()
        
        self.assertTrue(result)
        self.assertEqual(self.manager.state, LeaderElectionState.CANDIDATE)
        self.assertEqual(self.manager.current_term, 1)
        self.assertIn(self.manager.node_id, self.manager.received_votes)
    
    def test_vote_processing(self):
        """Test processing vote requests"""
        # Test granting vote
        result = self.manager.request_vote("supervisor_002", 1)
        self.assertTrue(result)
        self.assertEqual(self.manager.voted_for, "supervisor_002")
        
        # Test rejecting vote for same term
        result = self.manager.request_vote("supervisor_003", 1)
        self.assertFalse(result)
    
    def test_heartbeat_processing(self):
        """Test processing leader heartbeats"""
        result = self.manager.receive_heartbeat("supervisor_002", 2)
        
        self.assertTrue(result)
        self.assertEqual(self.manager.current_term, 2)
        self.assertEqual(self.manager.leader_id, "supervisor_002")
        self.assertEqual(self.manager.state, LeaderElectionState.FOLLOWER)
    
    def test_election_status(self):
        """Test election status reporting"""
        status = self.manager.get_election_status()
        
        expected_keys = [
            'node_id', 'state', 'current_term', 'leader_id',
            'is_leader', 'vote_count'
        ]
        for key in expected_keys:
            self.assertIn(key, status)


class TestMessageHistoryCompressor(unittest.TestCase):
    """Test message compression functionality"""
    
    def setUp(self):
        self.compressor = MessageHistoryCompressor("test_node", compression_threshold=5)
    
    def test_message_addition(self):
        """Test adding messages"""
        message = NetworkMessage(
            msg_type="test_message",
            sender_id="test_sender",
            data={"content": "test data"}
        )
        
        self.compressor.add_message(message)
        self.assertEqual(len(self.compressor.message_buffer), 1)
    
    def test_compression_trigger(self):
        """Test automatic compression when threshold reached"""
        # Add messages to trigger compression
        for i in range(6):  # Threshold is 5
            message = NetworkMessage(
                msg_type="test_message",
                sender_id="test_sender",
                data={"id": i, "content": f"test data {i}"}
            )
            self.compressor.add_message(message)
        
        # Should have triggered compression
        self.assertEqual(len(self.compressor.message_buffer), 1)  # Last message
        self.assertGreater(len(self.compressor.compressed_batches), 0)
    
    def test_compression_stats(self):
        """Test compression statistics"""
        stats = self.compressor.get_compression_stats()
        
        expected_keys = [
            'total_messages', 'compressed_batches', 'current_buffer_size',
            'average_compression_ratio'
        ]
        for key in expected_keys:
            self.assertIn(key, stats)


class TestVPNMeshTopologyManager(unittest.TestCase):
    """Test VPN mesh functionality"""
    
    def setUp(self):
        self.manager = VPNMeshTopologyManager("coordinator_001")
    
    def test_mesh_joining(self):
        """Test nodes joining the mesh"""
        node_info = {
            "node_id": "worker_001",
            "capabilities": ["training", "inference"]
        }
        
        virtual_ip = self.manager.join_mesh(node_info)
        
        self.assertIsNotNone(virtual_ip)
        self.assertIn("worker_001", self.manager.mesh_nodes)
        self.assertIn("worker_001", self.manager.virtual_ips)
    
    def test_mesh_leaving(self):
        """Test nodes leaving the mesh"""
        # First join
        node_info = {"node_id": "worker_001"}
        self.manager.join_mesh(node_info)
        
        # Then leave
        result = self.manager.leave_mesh("worker_001")
        
        self.assertTrue(result)
        self.assertNotIn("worker_001", self.manager.mesh_nodes)
    
    def test_tunnel_establishment(self):
        """Test establishing tunnels"""
        # Add nodes first
        for i in range(2):
            node_info = {"node_id": f"worker_{i:03d}"}
            self.manager.join_mesh(node_info)
        
        # Establish tunnel
        result = self.manager.establish_tunnel("worker_000")
        self.assertTrue(result)
    
    def test_mesh_topology(self):
        """Test getting mesh topology"""
        # Add some nodes
        for i in range(3):
            node_info = {"node_id": f"worker_{i:03d}"}
            self.manager.join_mesh(node_info)
        
        topology = self.manager.get_mesh_topology()
        
        expected_keys = ['node_id', 'mesh_nodes', 'active_tunnels']
        for key in expected_keys:
            self.assertIn(key, topology)
        
        self.assertEqual(topology['mesh_nodes'], 3)


class TestNetworkOperationsManager(unittest.TestCase):
    """Test integrated network operations"""
    
    def setUp(self):
        supervisors = ["supervisor_001", "supervisor_002"]
        self.manager = NetworkOperationsManager(
            "supervisor_001", "supervisor", supervisors
        )
    
    def test_initialization(self):
        """Test manager initialization"""
        self.assertIsNotNone(self.manager.crash_recovery)
        self.assertIsNotNone(self.manager.worker_replacement)
        self.assertIsNotNone(self.manager.message_compressor)
        self.assertIsNotNone(self.manager.vpn_mesh)
        self.assertIsNotNone(self.manager.leader_election)
    
    def test_operations_lifecycle(self):
        """Test starting and stopping operations"""
        async def test():
            await self.manager.start_operations()
            self.assertTrue(self.manager.is_running)
            
            await self.manager.stop_operations()
            self.assertFalse(self.manager.is_running)
        
        asyncio.run(test())


if __name__ == '__main__':
    print("🧪 Running Network Operations Test Suite")
    print("=" * 60)
    
    # Run tests
    unittest.main(verbosity=2, exit=False)
    
    print("\\n" + "=" * 60)
    print("✅ Network Operations Test Suite Complete")
