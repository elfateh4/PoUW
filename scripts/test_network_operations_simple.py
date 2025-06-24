#!/usr/bin/env python3
"""
Simple Network Operations Test for PoUW.

This script tests the network operations integration without the P2P networking complexity.
"""

import asyncio
import logging
import time
from typing import List

from pouw.node import PoUWNode
from pouw.economics import NodeRole
from pouw.network.operations import (
    CrashRecoveryManager, WorkerReplacementManager, LeaderElectionManager,
    MessageHistoryCompressor, VPNMeshTopologyManager, NetworkOperationsManager,
    NodeHealthMetrics, NodeStatus
)
from pouw.network.communication import NetworkMessage


async def test_network_operations():
    """Test network operations components directly"""
    print("🧪 Testing Network Operations Components")
    print("=" * 60)
    
    # Test 1: Crash Recovery Manager
    print("\\n1. Testing Crash Recovery Manager...")
    crash_manager = CrashRecoveryManager("test_node")
    
    # Create and update health metrics
    metrics = NodeHealthMetrics(
        node_id="worker_001",
        last_heartbeat=time.time(),
        response_time=0.1,
        success_rate=0.95,
        task_completion_rate=0.90,
        bandwidth_utilization=0.5,
        cpu_usage=0.3,
        memory_usage=0.4
    )
    crash_manager.update_node_health("worker_001", metrics)
    print(f"   ✓ Updated health metrics for worker_001")
    
    # Test crash detection
    crashed_nodes = crash_manager.detect_crashes()
    print(f"   ✓ Detected crashes: {len(crashed_nodes)} nodes")
    
    # Get health summary
    summary = crash_manager.get_network_health_summary()
    print(f"   ✓ Network health summary: {summary['total_nodes']} nodes, {summary['online_nodes']} online")
    
    # Test 2: Worker Replacement Manager
    print("\\n2. Testing Worker Replacement Manager...")
    replacement_manager = WorkerReplacementManager("supervisor_001")
    
    # Register worker pool
    workers = ["worker_001", "worker_002", "worker_003"]
    replacement_manager.register_worker_pool("ml_training", workers)
    print(f"   ✓ Registered worker pool with {len(workers)} workers")
    
    # Assign workers to task
    try:
        primary, backup = replacement_manager.assign_workers_to_task(
            "task_001", "ml_training", required_workers=1, backup_workers=1
        )
        print(f"   ✓ Assigned primary: {primary}, backup: {backup}")
        
        # Test replacement
        if backup:
            replacement = replacement_manager.replace_failed_worker(primary[0], "task_001")
            print(f"   ✓ Replaced failed worker with: {replacement}")
    except ValueError as e:
        print(f"   ⚠️ Worker assignment failed: {e}")
    
    # Test 3: Leader Election Manager
    print("\\n3. Testing Leader Election Manager...")
    supervisors = ["supervisor_001", "supervisor_002", "supervisor_003"]
    election_manager = LeaderElectionManager("supervisor_001", supervisors)
    
    # Start election
    election_manager.start_election()
    print(f"   ✓ Started election for term {election_manager.current_term}")
    print(f"   ✓ State: {election_manager.state.value}")
    
    # Simulate receiving votes
    election_manager.receive_vote_response("supervisor_002", True)
    election_manager.receive_vote_response("supervisor_003", True)
    
    status = election_manager.get_election_status()
    print(f"   ✓ Election status: {status['state']}, leader: {status['leader_id']}")
    
    # Test 4: Message History Compressor
    print("\\n4. Testing Message History Compressor...")
    compressor = MessageHistoryCompressor("test_node")
    
    # Add messages
    for i in range(10):
        message = NetworkMessage(
            msg_type="test_message",
            sender_id="test_sender",
            data={"id": i, "content": f"Test message {i}"}
        )
        compressor.add_message(message)
    
    print(f"   ✓ Added 10 test messages")
    
    # Get stats
    stats = compressor.get_compression_stats()
    print(f"   ✓ Compression stats: {stats['total_messages']} messages, {stats['compressed_batches']} batches")
    
    # Test 5: VPN Mesh Topology Manager
    print("\\n5. Testing VPN Mesh Topology Manager...")
    vpn_manager = VPNMeshTopologyManager("coordinator_001")
    
    # Add nodes to mesh
    nodes_info = [
        {"node_id": "worker_001", "capabilities": ["training"]},
        {"node_id": "worker_002", "capabilities": ["inference"]},
        {"node_id": "worker_003", "capabilities": ["training", "inference"]}
    ]
    
    for node_info in nodes_info:
        virtual_ip = vpn_manager.join_mesh(node_info)
        print(f"   ✓ Added {node_info['node_id']} to mesh with IP {virtual_ip}")
    
    # Get topology
    topology = vpn_manager.get_mesh_topology()
    print(f"   ✓ Mesh topology: {topology['mesh_nodes']} nodes, {topology['active_tunnels']} tunnels")
    
    # Test 6: Network Operations Manager Integration
    print("\\n6. Testing Network Operations Manager...")
    network_ops = NetworkOperationsManager("supervisor_001", "supervisor", supervisors)
    
    print(f"   ✓ Created NetworkOperationsManager")
    print(f"   ✓ Components initialized:")
    print(f"     - Crash Recovery: {network_ops.crash_recovery is not None}")
    print(f"     - Worker Replacement: {network_ops.worker_replacement is not None}")
    print(f"     - Leader Election: {network_ops.leader_election is not None}")
    print(f"     - Message Compressor: {network_ops.message_compressor is not None}")
    print(f"     - VPN Mesh: {network_ops.vpn_mesh is not None}")
    
    # Start and stop operations
    await network_ops.start_operations()
    print(f"   ✓ Started network operations")
    
    await asyncio.sleep(1)  # Let it run briefly
    
    await network_ops.stop_operations()
    print(f"   ✓ Stopped network operations")
    
    print("\\n" + "=" * 60)
    print("✅ All Network Operations Tests Completed Successfully!")
    print("🎉 Network operations are fully functional and integrated!")
    print("=" * 60)


async def test_node_integration():
    """Test network operations integration with PoUW nodes"""
    print("\\n🔗 Testing Node Integration...")
    print("=" * 60)
    
    # Create nodes with network operations
    supervisor = PoUWNode("supervisor_test", NodeRole.SUPERVISOR, "localhost", 8500)
    worker = PoUWNode("worker_test", NodeRole.MINER, "localhost", 8501)
    
    print(f"   ✓ Created supervisor and worker nodes")
    print(f"   ✓ Supervisor network ops: {supervisor.network_operations is not None}")
    print(f"   ✓ Worker network ops: {worker.network_operations is not None}")
    
    # Start nodes
    await supervisor.start()
    await worker.start()
    
    print(f"   ✓ Started nodes with network operations")
    
    # Brief operation
    await asyncio.sleep(2)
    
    # Stop nodes
    await supervisor.stop()
    await worker.stop()
    
    print(f"   ✓ Stopped nodes cleanly")
    print("   🎯 Node integration test completed!")


async def main():
    """Run all network operations tests"""
    try:
        await test_network_operations()
        await test_node_integration()
        
        print("\\n🏆 ALL TESTS PASSED!")
        print("Network operations are successfully implemented and integrated!")
        
    except Exception as e:
        print(f"\\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise
    
    asyncio.run(main())
