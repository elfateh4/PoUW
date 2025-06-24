#!/usr/bin/env python3
"""
Network Operations Demonstration for PoUW.

This script demonstrates the advanced network operations features:
1. Crash recovery with phi accrual failure detection
2. Worker replacement mechanisms
3. Leader election for supervisors
4. Message history compression
5. VPN mesh topology management
"""

import asyncio
import logging
import time
import random
from typing import List

from pouw.node import PoUWNode
from pouw.economics import NodeRole
from pouw.network.operations import NodeStatus, NodeHealthMetrics


class NetworkOperationsDemo:
    """Demonstration of network operations features"""
    
    def __init__(self):
        self.nodes: List[PoUWNode] = []
        self.supervisor_nodes: List[PoUWNode] = []
        self.worker_nodes: List[PoUWNode] = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("NetworkOpsDemo")
    
    async def setup_network(self):
        """Setup a network to demonstrate network operations"""
        self.logger.info("🚀 Setting up network for operations demonstration...")
        
        base_port = 9000
        
        # Create supervisor nodes (for leader election)
        for i in range(3):
            supervisor = PoUWNode(
                f"supervisor_{i:03d}", 
                NodeRole.SUPERVISOR, 
                "localhost", 
                base_port + i
            )
            
            # Set supervisor nodes for leader election
            supervisor_ids = [f"supervisor_{j:03d}" for j in range(3)]
            if supervisor.network_operations.leader_election:
                supervisor.network_operations.leader_election.supervisor_nodes = supervisor_ids
            
            await supervisor.start()
            self.supervisor_nodes.append(supervisor)
            self.nodes.append(supervisor)
            
            self.logger.info(f"✓ Started supervisor {supervisor.node_id}")
        
        # Create worker nodes (miners)
        for i in range(5):
            worker = PoUWNode(
                f"worker_{i:03d}", 
                NodeRole.MINER, 
                "localhost", 
                base_port + 10 + i
            )
            
            await worker.start()
            self.worker_nodes.append(worker)
            self.nodes.append(worker)
            
            self.logger.info(f"✓ Started worker {worker.node_id}")
        
        # Connect all nodes
        await self._connect_all_nodes()
        
        # Wait for network to stabilize
        await asyncio.sleep(3)
        
        self.logger.info(f"🌐 Network setup complete: {len(self.nodes)} nodes")
    
    async def _connect_all_nodes(self):
        """Connect all nodes to each other"""
        for i, node in enumerate(self.nodes):
            for j, other_node in enumerate(self.nodes):
                if i != j:
                    await node.p2p_node.connect_to_peer(other_node.host, other_node.port)
                    await asyncio.sleep(0.1)
    
    async def demonstrate_crash_recovery(self):
        """Demonstrate crash recovery mechanisms"""
        self.logger.info("\\n" + "="*60)
        self.logger.info("DEMONSTRATING CRASH RECOVERY")
        self.logger.info("="*60)
        
        # Select a worker to simulate crash
        crashed_worker = self.worker_nodes[0]
        
        self.logger.info(f"💥 Simulating crash of {crashed_worker.node_id}...")
        
        # Stop the worker to simulate crash
        await crashed_worker.stop()
        
        # Wait for failure detection
        await asyncio.sleep(10)
        
        # Check if other nodes detected the failure
        for supervisor in self.supervisor_nodes:
            crash_manager = supervisor.network_operations.crash_recovery
            health_summary = crash_manager.get_network_health_summary()
            
            self.logger.info(f"📊 Health summary from {supervisor.node_id}:")
            self.logger.info(f"   Failed nodes: {health_summary['failed_nodes']}")
            self.logger.info(f"   Suspected nodes: {health_summary['suspected_nodes']}")
            self.logger.info(f"   Recovery events: {health_summary['total_recovery_events']}")
        
        # Simulate recovery
        self.logger.info(f"🔄 Attempting recovery of {crashed_worker.node_id}...")
        
        # Restart the crashed worker
        new_worker = PoUWNode(
            crashed_worker.node_id, 
            NodeRole.MINER, 
            crashed_worker.host, 
            crashed_worker.port
        )
        await new_worker.start()
        
        # Reconnect to network
        for node in self.nodes[:-1]:  # Exclude the crashed one
            await new_worker.p2p_node.connect_to_peer(node.host, node.port)
        
        # Replace in our list
        self.worker_nodes[0] = new_worker
        self.nodes[-len(self.worker_nodes)] = new_worker
        
        self.logger.info(f"✅ {crashed_worker.node_id} has been recovered and reconnected")
    
    async def demonstrate_worker_replacement(self):
        """Demonstrate worker replacement mechanisms"""
        self.logger.info("\\n" + "="*60)
        self.logger.info("DEMONSTRATING WORKER REPLACEMENT")
        self.logger.info("="*60)
        
        # Setup some task assignments first
        supervisor = self.supervisor_nodes[0]
        replacement_manager = supervisor.network_operations.worker_replacement
        
        # Register worker pool for task type
        worker_ids = [worker.node_id for worker in self.worker_nodes]
        replacement_manager.register_worker_pool("demo_task_type", worker_ids)
        
        # Assign workers to tasks
        for i in range(3):
            task_id = f"demo_task_{i:03d}"
            try:
                primary_workers, backup_workers = replacement_manager.assign_workers_to_task(
                    task_id, "demo_task_type", required_workers=1, backup_workers=1
                )
                self.logger.info(f"📝 Assigned {primary_workers[0]} to {task_id} (backup: {backup_workers[0] if backup_workers else 'none'})")
            except ValueError as e:
                self.logger.warning(f"Could not assign workers to {task_id}: {e}")
                break
        
        # Show current assignments
        supervisor = self.supervisor_nodes[0]
        replacement_manager = supervisor.network_operations.worker_replacement
        
        self.logger.info("\\n📋 Current worker assignments:")
        for worker_id, task_id in replacement_manager.worker_assignments.items():
            self.logger.info(f"   {worker_id} -> {task_id}")
        
        # Simulate worker failure and replacement
        failed_worker_id = self.worker_nodes[1].node_id
        task_id = replacement_manager.worker_assignments.get(failed_worker_id)
        
        if task_id:
            self.logger.info(f"\\n💥 Worker {failed_worker_id} failed during {task_id}")
            
            # Attempt replacement
            replacement = replacement_manager.replace_failed_worker(failed_worker_id, task_id)
            
            if replacement:
                self.logger.info(f"🔄 Replaced {failed_worker_id} with {replacement}")
            else:
                self.logger.info("❌ No suitable replacement worker available")
        
        # Show updated assignments
        self.logger.info("\\n📋 Updated worker assignments:")
        for worker_id, task_id in replacement_manager.worker_assignments.items():
            self.logger.info(f"   {worker_id} -> {task_id}")
    
    async def demonstrate_leader_election(self):
        """Demonstrate leader election among supervisors"""
        self.logger.info("\\n" + "="*60)
        self.logger.info("DEMONSTRATING LEADER ELECTION")
        self.logger.info("="*60)
        
        # Show initial states
        self.logger.info("📊 Initial supervisor states:")
        for supervisor in self.supervisor_nodes:
            election_manager = supervisor.network_operations.leader_election
            self.logger.info(f"   {supervisor.node_id}: {election_manager.state.value}")
        
        # Trigger election
        self.logger.info("\\n🗳️  Triggering leader election...")
        
        # Start election from first supervisor
        first_supervisor = self.supervisor_nodes[0]
        await first_supervisor.network_operations._conduct_election()
        
        # Wait for election to complete
        await asyncio.sleep(5)
        
        # Show results
        self.logger.info("\\n🏆 Election results:")
        leader_found = False
        for supervisor in self.supervisor_nodes:
            election_manager = supervisor.network_operations.leader_election
            if election_manager:
                state = election_manager.state.value
                term = election_manager.current_term
                
                self.logger.info(f"   {supervisor.node_id}: {state} (term {term})")
                
                if state == "leader":
                    leader_found = True
                    self.logger.info(f"👑 Leader elected: {supervisor.node_id}")
            else:
                self.logger.info(f"   {supervisor.node_id}: no election manager")
        
        if not leader_found:
            self.logger.info("⚠️  No leader elected in this round")
    
    async def demonstrate_message_compression(self):
        """Demonstrate message history compression"""
        self.logger.info("\\n" + "="*60)
        self.logger.info("DEMONSTRATING MESSAGE COMPRESSION")
        self.logger.info("="*60)
        
        # Use first supervisor for demonstration
        supervisor = self.supervisor_nodes[0]
        compressor = supervisor.network_operations.message_compressor
        
        # Generate some sample messages
        from pouw.network.communication import NetworkMessage
        
        sample_messages = []
        for i in range(50):
            message = NetworkMessage(
                msg_type='demo_message',
                sender_id=supervisor.node_id,
                data={
                    'id': i,
                    'content': f"Sample message content {i}" * 10,  # Make it larger
                    'demo_timestamp': time.time() + i
                }
            )
            sample_messages.append(message)
            compressor.add_message(message)
        
        # Show compression stats before
        stats_before = compressor.get_compression_stats()
        self.logger.info(f"📊 Compression stats:")
        self.logger.info(f"   Total messages: {stats_before['total_messages']}")
        self.logger.info(f"   Current buffer: {stats_before['current_buffer_size']}")
        self.logger.info(f"   Compressed batches: {stats_before['compressed_batches']}")
        
        # Force compression by adding more messages to trigger threshold
        for i in range(50, 1050):  # Add enough to trigger compression
            message = NetworkMessage(
                msg_type='demo_message',
                sender_id=supervisor.node_id,
                data={'id': i, 'content': f"Additional message {i}"}
            )
            compressor.add_message(message)
        
        # Show stats after compression
        stats_after = compressor.get_compression_stats()
        self.logger.info(f"\\n📈 After compression:")
        self.logger.info(f"   Total messages: {stats_after['total_messages']}")
        self.logger.info(f"   Current buffer: {stats_after['current_buffer_size']}")
        self.logger.info(f"   Compressed batches: {stats_after['compressed_batches']}")
        if stats_after['average_compression_ratio'] > 0:
            self.logger.info(f"   Compression ratio: {stats_after['average_compression_ratio']:.2f}")
    
    async def demonstrate_vpn_mesh(self):
        """Demonstrate VPN mesh topology management"""
        self.logger.info("\\n" + "="*60)
        self.logger.info("DEMONSTRATING VPN MESH TOPOLOGY")
        self.logger.info("="*60)
        
        # Use supervisor for VPN management
        supervisor = self.supervisor_nodes[0]
        vpn_manager = supervisor.network_operations.vpn_mesh
        
        # Add nodes to mesh
        self.logger.info("🌐 Building VPN mesh topology...")
        
        for worker in self.worker_nodes:
            node_info = {
                "node_id": worker.node_id,
                "physical_ip": worker.host,
                "capabilities": ["training", "inference"]
            }
            
            virtual_ip = vpn_manager.join_mesh(node_info)
            
            if virtual_ip:
                self.logger.info(f"✓ Added {worker.node_id} to VPN mesh with IP {virtual_ip}")
            else:
                self.logger.info(f"❌ Failed to add {worker.node_id} to VPN mesh")
        
        # Show mesh status
        mesh_topology = vpn_manager.get_mesh_topology()
        self.logger.info(f"\\n📊 VPN Mesh Statistics:")
        self.logger.info(f"   Active tunnels: {mesh_topology['active_tunnels']}")
        self.logger.info(f"   Total nodes: {mesh_topology['mesh_nodes']}")
        
        # Simulate tunnel health check
        self.logger.info("\\n🔍 Checking tunnel health...")
        health_report = vpn_manager.monitor_tunnel_health()
        
        self.logger.info(f"   Total tunnels: {health_report['total_tunnels']}")
        self.logger.info(f"   Healthy tunnels: {health_report['healthy_tunnels']}")
        
        for tunnel_id, details in health_report['tunnel_details'].items():
            status = "🟢 Healthy" if details['healthy'] else "🔴 Unhealthy"
            self.logger.info(f"   {tunnel_id}: {status}")
    
    async def run_demonstration(self):
        """Run complete network operations demonstration"""
        try:
            # Setup network
            await self.setup_network()
            
            # Run demonstrations
            await self.demonstrate_leader_election()
            await asyncio.sleep(2)
            
            await self.demonstrate_message_compression()
            await asyncio.sleep(2)
            
            await self.demonstrate_vpn_mesh()
            await asyncio.sleep(2)
            
            await self.demonstrate_worker_replacement()
            await asyncio.sleep(2)
            
            await self.demonstrate_crash_recovery()
            
            self.logger.info("\\n" + "="*60)
            self.logger.info("🎉 NETWORK OPERATIONS DEMONSTRATION COMPLETE")
            self.logger.info("="*60)
            self.logger.info("✅ All network operations features successfully demonstrated!")
            
        except Exception as e:
            self.logger.error(f"❌ Demo failed: {e}")
            raise
        finally:
            # Cleanup
            await self.cleanup()
    
    async def cleanup(self):
        """Cleanup all nodes"""
        self.logger.info("\\n🧹 Cleaning up nodes...")
        for node in self.nodes:
            try:
                await node.stop()
            except Exception as e:
                self.logger.warning(f"Error stopping {node.node_id}: {e}")
        
        self.logger.info("✅ Cleanup complete")


async def main():
    """Main demo function"""
    demo = NetworkOperationsDemo()
    await demo.run_demonstration()


if __name__ == '__main__':
    asyncio.run(main())
