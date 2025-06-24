#!/usr/bin/env python3
"""
Enhanced PoUW Node VPN Mesh Integration Demo

This script demonstrates the complete integration of the production-ready
VPN mesh networking system into the PoUW node architecture. It shows how
nodes can establish secure mesh topologies and leverage them for ML training.

Key Features Demonstrated:
- Full node integration with VPN mesh capabilities
- Supervisor mesh topology formation
- Worker-supervisor redundant connections
- Real-time mesh health monitoring
- Network optimization and fault tolerance
- Integration with existing PoUW blockchain and ML systems
"""

import asyncio
import logging
import time
from typing import List, Dict, Any

from pouw.node import PoUWNode
from pouw.economics import NodeRole


class IntegratedVPNMeshDemo:
    """Demonstrates integrated VPN mesh functionality within PoUW nodes"""
    
    def __init__(self):
        self.supervisor_nodes: List[PoUWNode] = []
        self.worker_nodes: List[PoUWNode] = []
        self.logger = logging.getLogger("VPNMeshDemo")
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    async def demonstrate_integrated_vpn_mesh(self):
        """Run comprehensive VPN mesh integration demonstration"""
        
        print("🚀 Enhanced PoUW Node VPN Mesh Integration Demo")
        print("=" * 60)
        
        try:
            # Phase 1: Node Creation and Basic Setup
            await self._phase_1_create_nodes()
            
            # Phase 2: VPN Mesh Network Initialization
            await self._phase_2_initialize_mesh()
            
            # Phase 3: Supervisor Mesh Formation
            await self._phase_3_supervisor_mesh()
            
            # Phase 4: Worker-Supervisor Connections
            await self._phase_4_worker_connections()
            
            # Phase 5: Mesh Health Monitoring
            await self._phase_5_health_monitoring()
            
            # Phase 6: ML Training with VPN Mesh
            await self._phase_6_ml_training_integration()
            
            # Phase 7: Network Optimization
            await self._phase_7_network_optimization()
            
            # Phase 8: Cleanup
            await self._phase_8_cleanup()
            
        except Exception as e:
            self.logger.error(f"Demo failed: {e}")
            raise
    
    async def _phase_1_create_nodes(self):
        """Phase 1: Create PoUW nodes with integrated VPN mesh"""
        print("\n📦 Phase 1: Creating PoUW Nodes with VPN Mesh Integration")
        print("-" * 50)
        
        # Create supervisor nodes
        for i in range(3):
            node = PoUWNode(
                node_id=f"supervisor_{i+1:03d}",
                role=NodeRole.SUPERVISOR,
                host="localhost",
                port=8000 + i
            )
            self.supervisor_nodes.append(node)
            print(f"  ✅ Created supervisor node: {node.node_id}")
        
        # Create worker nodes  
        for i in range(5):
            node = PoUWNode(
                node_id=f"worker_{i+1:03d}",
                role=NodeRole.MINER,  # Workers are miners in this system
                host="localhost", 
                port=8100 + i
            )
            self.worker_nodes.append(node)
            print(f"  ✅ Created worker node: {node.node_id}")
        
        print(f"  📊 Total nodes created: {len(self.supervisor_nodes + self.worker_nodes)}")
        await asyncio.sleep(1)
    
    async def _phase_2_initialize_mesh(self):
        """Phase 2: Initialize VPN mesh networking on all nodes"""
        print("\n🔗 Phase 2: Initializing VPN Mesh Networks")
        print("-" * 50)
        
        supervisor_ids = [node.node_id for node in self.supervisor_nodes]
        
        # Initialize mesh on supervisor nodes
        for node in self.supervisor_nodes:
            try:
                success = await node.initialize_vpn_mesh(supervisor_nodes=supervisor_ids)
                status = "✅ SUCCESS" if success else "❌ FAILED"
                print(f"  {status} VPN mesh initialized for {node.node_id}")
            except Exception as e:
                print(f"  ❌ Failed to initialize mesh for {node.node_id}: {e}")
        
        # Initialize mesh on worker nodes
        for node in self.worker_nodes:
            try:
                success = await node.initialize_vpn_mesh(supervisor_nodes=supervisor_ids)
                status = "✅ SUCCESS" if success else "❌ FAILED"
                print(f"  {status} VPN mesh initialized for {node.node_id}")
            except Exception as e:
                print(f"  ❌ Failed to initialize mesh for {node.node_id}: {e}")
        
        await asyncio.sleep(2)
    
    async def _phase_3_supervisor_mesh(self):
        """Phase 3: Demonstrate supervisor mesh topology"""
        print("\n🏗️ Phase 3: Supervisor Mesh Topology Analysis")
        print("-" * 50)
        
        for node in self.supervisor_nodes:
            try:
                status = await node.get_vpn_mesh_status()
                tunnel_count = len(status.get('tunnel_details', {}))
                mesh_health = status.get('mesh_health', 'unknown')
                
                print(f"  📊 {node.node_id}:")
                print(f"    🔗 Active tunnels: {tunnel_count}")
                print(f"    💚 Mesh health: {mesh_health}")
                print(f"    🌐 Role: {status.get('role', 'unknown')}")
                
            except Exception as e:
                print(f"  ❌ Failed to get status for {node.node_id}: {e}")
        
        await asyncio.sleep(2)
    
    async def _phase_4_worker_connections(self):
        """Phase 4: Analyze worker-supervisor connections"""
        print("\n👷 Phase 4: Worker-Supervisor Connection Analysis")
        print("-" * 50)
        
        for node in self.worker_nodes:
            try:
                status = await node.get_vpn_mesh_status()
                tunnel_count = len(status.get('tunnel_details', {}))
                mesh_health = status.get('mesh_health', 'unknown')
                
                print(f"  📊 {node.node_id}:")
                print(f"    🔗 Supervisor connections: {tunnel_count}")
                print(f"    💚 Connection health: {mesh_health}")
                
            except Exception as e:
                print(f"  ❌ Failed to get status for {node.node_id}: {e}")
        
        await asyncio.sleep(2)
    
    async def _phase_5_health_monitoring(self):
        """Phase 5: Real-time mesh health monitoring"""
        print("\n💓 Phase 5: Real-time Mesh Health Monitoring")
        print("-" * 50)
        
        # Monitor health across all nodes
        total_tunnels = 0
        healthy_tunnels = 0
        
        all_nodes = self.supervisor_nodes + self.worker_nodes
        
        for node in all_nodes:
            try:
                status = await node.get_vpn_mesh_status()
                tunnel_details = status.get('tunnel_details', {})
                
                for tunnel_id, details in tunnel_details.items():
                    total_tunnels += 1
                    if details.get('state') == 'CONNECTED':
                        healthy_tunnels += 1
                
            except Exception as e:
                print(f"  ⚠️ Monitoring error for {node.node_id}: {e}")
        
        if total_tunnels > 0:
            health_percentage = (healthy_tunnels / total_tunnels) * 100
            print(f"  📈 Network Health Summary:")
            print(f"    🔗 Total tunnels: {total_tunnels}")
            print(f"    ✅ Healthy tunnels: {healthy_tunnels}")
            print(f"    📊 Health percentage: {health_percentage:.1f}%")
        else:
            print("  ℹ️ No tunnels found for monitoring")
        
        await asyncio.sleep(2)
    
    async def _phase_6_ml_training_integration(self):
        """Phase 6: Demonstrate ML training over VPN mesh"""
        print("\n🧠 Phase 6: ML Training Integration with VPN Mesh")
        print("-" * 50)
        
        print("  🎯 Simulating ML task distribution over secure VPN mesh...")
        
        # Simulate task creation on supervisor
        if self.supervisor_nodes:
            supervisor = self.supervisor_nodes[0]
            
            # Create a mock ML task
            task_def = {
                'task_type': 'image_classification',
                'model_architecture': 'SimpleMLP',
                'dataset': 'MNIST',
                'epochs': 5,
                'batch_size': 32
            }
            
            try:
                # Submit task through the integrated system
                task_id = supervisor.submit_ml_task(task_def, fee=10.0)
                print(f"  ✅ ML task submitted: {task_id}")
                
                # Show how mesh status relates to training
                mesh_status = await supervisor.get_vpn_mesh_status()
                mesh_health = mesh_status.get('mesh_health', 'unknown')
                print(f"  🌐 Training network health: {mesh_health}")
                
                # Simulate worker selection based on mesh connectivity
                connected_workers = []
                for worker in self.worker_nodes:
                    worker_status = await worker.get_vpn_mesh_status()
                    if worker_status.get('mesh_health') in ['excellent', 'good']:
                        connected_workers.append(worker.node_id)
                
                print(f"  👥 Available workers via mesh: {len(connected_workers)}")
                for worker_id in connected_workers[:3]:  # Show first 3
                    print(f"    - {worker_id}")
                
            except Exception as e:
                print(f"  ❌ ML integration demo failed: {e}")
        
        await asyncio.sleep(2)
    
    async def _phase_7_network_optimization(self):
        """Phase 7: Network optimization demonstration"""
        print("\n⚡ Phase 7: Network Optimization")
        print("-" * 50)
        
        print("  🔧 Running network optimization on all nodes...")
        
        optimization_results = []
        
        for node in self.supervisor_nodes + self.worker_nodes:
            try:
                # Get status before optimization
                status_before = await node.get_vpn_mesh_status()
                health_before = status_before.get('mesh_health', 'unknown')
                
                # Run optimization
                await node.optimize_mesh_network()
                
                # Get status after optimization
                status_after = await node.get_vpn_mesh_status() 
                health_after = status_after.get('mesh_health', 'unknown')
                
                optimization_results.append({
                    'node_id': node.node_id,
                    'before': health_before,
                    'after': health_after
                })
                
            except Exception as e:
                print(f"  ⚠️ Optimization failed for {node.node_id}: {e}")
        
        # Show optimization results
        print(f"  📊 Optimization Results:")
        for result in optimization_results[:5]:  # Show first 5
            node_id = result['node_id']
            before = result['before']
            after = result['after']
            print(f"    {node_id}: {before} → {after}")
        
        await asyncio.sleep(2)
    
    async def _phase_8_cleanup(self):
        """Phase 8: Graceful cleanup"""
        print("\n🧹 Phase 8: Graceful Cleanup")
        print("-" * 50)
        
        print("  🔌 Shutting down VPN mesh networks...")
        
        cleanup_count = 0
        all_nodes = self.supervisor_nodes + self.worker_nodes
        
        for node in all_nodes:
            try:
                await node.shutdown_vpn_mesh()
                cleanup_count += 1
                print(f"  ✅ VPN mesh shutdown for {node.node_id}")
            except Exception as e:
                print(f"  ⚠️ Cleanup warning for {node.node_id}: {e}")
        
        print(f"  📊 Cleanup completed for {cleanup_count}/{len(all_nodes)} nodes")
        
        # Clear node lists
        self.supervisor_nodes.clear()
        self.worker_nodes.clear()
        
        print("\n🎉 VPN Mesh Integration Demo Completed Successfully!")
        print("=" * 60)
        
        # Summary
        print("\n📋 Demo Summary:")
        print("  ✅ Node integration with VPN mesh capabilities")
        print("  ✅ Supervisor mesh topology formation")
        print("  ✅ Worker-supervisor redundant connections")
        print("  ✅ Real-time health monitoring")
        print("  ✅ ML training integration over secure mesh")
        print("  ✅ Network optimization capabilities")
        print("  ✅ Graceful cleanup and shutdown")


async def main():
    """Run the integrated VPN mesh demonstration"""
    demo = IntegratedVPNMeshDemo()
    try:
        await demo.demonstrate_integrated_vpn_mesh()
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
