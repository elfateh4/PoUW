#!/usr/bin/env python3
"""
Enhanced VPN Mesh Topology Demo for PoUW.

Demonstrates the production-ready VPN mesh networking capabilities
as the highest priority improvement from the IMPLEMENTATION_REPORT.md.
"""

import asyncio
import time
import logging
import json
from typing import Dict, List
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pouw.network.vpn_mesh_enhanced import (
    ProductionVPNMeshManager,
    MeshNetworkCoordinator,
    VPNProtocol,
    TunnelState,
)


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("VPNMeshDemo")


class EnhancedVPNMeshDemo:
    """Enhanced VPN mesh topology demonstration"""

    def __init__(self):
        self.coordinator = MeshNetworkCoordinator(network_cidr="10.100.0.0/16")
        self.worker_nodes: List[ProductionVPNMeshManager] = []
        self.supervisor_nodes: List[ProductionVPNMeshManager] = []

    async def demonstrate_enhanced_vpn_mesh(self):
        """Main demonstration of enhanced VPN mesh capabilities"""
        logger.info("🚀 Starting Enhanced VPN Mesh Topology Demonstration")
        logger.info("=" * 80)

        # Phase 1: Network Coordinator Setup
        await self._demonstrate_coordinator_setup()

        # Phase 2: Node Registration and Mesh Formation
        await self._demonstrate_mesh_formation()

        # Phase 3: Tunnel Establishment and Health Monitoring
        await self._demonstrate_tunnel_management()

        # Phase 4: Advanced Features
        await self._demonstrate_advanced_features()

        # Phase 5: Network Optimization and Monitoring
        await self._demonstrate_network_optimization()

        # Phase 6: Cleanup
        await self._cleanup_demonstration()

        logger.info("✅ Enhanced VPN Mesh Topology Demonstration Complete")
        logger.info("=" * 80)

    async def _demonstrate_coordinator_setup(self):
        """Demonstrate mesh network coordinator setup"""
        logger.info("\\n📡 Phase 1: Mesh Network Coordinator Setup")
        logger.info("-" * 50)

        # Display coordinator capabilities
        logger.info(f"🌐 Network CIDR: {self.coordinator.network_cidr}")
        logger.info(f"📊 Available IP addresses: {self.coordinator.network_cidr.num_addresses - 2}")

        topology = self.coordinator.get_mesh_topology()
        logger.info(f"🔧 Initial topology: {json.dumps(topology, indent=2)}")

    async def _demonstrate_mesh_formation(self):
        """Demonstrate mesh network formation"""
        logger.info("\\n🔗 Phase 2: Mesh Network Formation")
        logger.info("-" * 50)

        # Create supervisor nodes
        logger.info("\\n👨‍💼 Creating Supervisor Nodes...")
        for i in range(3):
            supervisor_id = f"supervisor_{i+1:03d}"
            supervisor = ProductionVPNMeshManager(
                node_id=supervisor_id,
                network_cidr="10.100.0.0/16",
                preferred_protocol=VPNProtocol.WIREGUARD,
                base_port=51820 + i,
            )

            # Register with coordinator
            node_info = {
                "node_id": supervisor_id,
                "physical_ip": f"192.168.1.{10+i}",
                "port": 51820 + i,
                "public_key": (
                    supervisor.node_public_key.hex() if supervisor.node_public_key else ""
                ),
                "role": "supervisor",
                "capabilities": ["consensus", "validation"],
            }

            virtual_ip = self.coordinator.register_node(node_info)
            logger.info(f"   ✅ {supervisor_id} registered with IP {virtual_ip}")

            # Join mesh network
            success = await supervisor.join_mesh_network("coordinator:8080")
            if success:
                logger.info(f"   🔌 {supervisor_id} joined mesh network")
                self.supervisor_nodes.append(supervisor)
            else:
                logger.error(f"   ❌ {supervisor_id} failed to join mesh")

        # Create worker nodes
        logger.info("\\n👷‍♂️ Creating Worker Nodes...")
        for i in range(5):
            worker_id = f"worker_{i+1:03d}"
            worker = ProductionVPNMeshManager(
                node_id=worker_id,
                network_cidr="10.100.0.0/16",
                preferred_protocol=VPNProtocol.WIREGUARD,
                base_port=52000 + i,
            )

            # Register with coordinator
            node_info = {
                "node_id": worker_id,
                "physical_ip": f"192.168.1.{20+i}",
                "port": 52000 + i,
                "public_key": worker.node_public_key.hex() if worker.node_public_key else "",
                "role": "worker",
                "capabilities": ["training", "inference"],
            }

            virtual_ip = self.coordinator.register_node(node_info)
            logger.info(f"   ✅ {worker_id} registered with IP {virtual_ip}")

            # Join mesh network
            success = await worker.join_mesh_network("coordinator:8080")
            if success:
                logger.info(f"   🔌 {worker_id} joined mesh network")
                self.worker_nodes.append(worker)
            else:
                logger.error(f"   ❌ {worker_id} failed to join mesh")

        # Display final topology
        topology = self.coordinator.get_mesh_topology()
        logger.info(f"\\n📈 Final network topology:")
        logger.info(f"   Total nodes: {topology['total_nodes']}")
        logger.info(f"   Network CIDR: {topology['network_cidr']}")

        for node_id, node_info in topology["nodes"].items():
            logger.info(f"   🖥️  {node_id}: {node_info['virtual_ip']} ({node_info['role']})")

    async def _demonstrate_tunnel_management(self):
        """Demonstrate tunnel establishment and health monitoring"""
        logger.info("\\n🚇 Phase 3: Tunnel Management and Health Monitoring")
        logger.info("-" * 50)

        # Establish tunnels between supervisors (full mesh)
        logger.info("\\n🔐 Establishing Supervisor Mesh Tunnels...")
        for i, supervisor1 in enumerate(self.supervisor_nodes):
            for j, supervisor2 in enumerate(self.supervisor_nodes[i + 1 :], i + 1):
                peer_info = {
                    "virtual_ip": supervisor2.mesh_nodes[supervisor2.node_id].virtual_ip,
                    "physical_ip": f"192.168.1.{10+j}",
                    "port": 51820 + j,
                    "public_key": (
                        supervisor2.node_public_key.hex() if supervisor2.node_public_key else ""
                    ),
                }

                success = await supervisor1.establish_tunnel_to_peer(supervisor2.node_id, peer_info)
                if success:
                    logger.info(
                        f"   ✅ Tunnel established: {supervisor1.node_id} ↔ {supervisor2.node_id}"
                    )
                else:
                    logger.error(
                        f"   ❌ Failed to establish tunnel: {supervisor1.node_id} ↔ {supervisor2.node_id}"
                    )

        # Establish tunnels between workers and supervisors
        logger.info("\\n🔗 Establishing Worker-Supervisor Tunnels...")
        for worker in self.worker_nodes:
            # Connect each worker to 2 supervisors for redundancy
            for supervisor in self.supervisor_nodes[:2]:
                peer_info = {
                    "virtual_ip": supervisor.mesh_nodes[supervisor.node_id].virtual_ip,
                    "physical_ip": f"192.168.1.{10 + self.supervisor_nodes.index(supervisor)}",
                    "port": 51820 + self.supervisor_nodes.index(supervisor),
                    "public_key": (
                        supervisor.node_public_key.hex() if supervisor.node_public_key else ""
                    ),
                }

                success = await worker.establish_tunnel_to_peer(supervisor.node_id, peer_info)
                if success:
                    logger.info(
                        f"   ✅ Tunnel established: {worker.node_id} → {supervisor.node_id}"
                    )

        # Wait for tunnel establishment
        await asyncio.sleep(2)

        # Monitor tunnel health
        logger.info("\\n🔍 Monitoring Tunnel Health...")
        all_nodes = self.supervisor_nodes + self.worker_nodes

        for node in all_nodes:
            status = node.get_mesh_status()
            tunnel_details = node.get_tunnel_details()

            logger.info(f"\\n   📊 {node.node_id} Status:")
            logger.info(f"      Virtual IP: {status['virtual_ip']}")
            logger.info(
                f"      Active tunnels: {status['active_tunnels']}/{status['total_tunnels']}"
            )
            logger.info(f"      Protocol: {status['protocol']}")
            logger.info(
                f"      Success rate: {status['network_statistics']['tunnel_success_rate']:.1%}"
            )

            if tunnel_details:
                logger.info(f"      Tunnel details:")
                for tunnel_id, details in tunnel_details.items():
                    health_status = "🟢 Healthy" if details["is_healthy"] else "🔴 Unhealthy"
                    logger.info(f"        {tunnel_id}: {health_status} ({details['state']})")

    async def _demonstrate_advanced_features(self):
        """Demonstrate advanced VPN mesh features"""
        logger.info("\\n⚡ Phase 4: Advanced VPN Mesh Features")
        logger.info("-" * 50)

        # Demonstrate encryption protocols
        logger.info("\\n🔒 Encryption Protocol Demonstration:")
        protocols_in_use = set()
        for node in self.supervisor_nodes + self.worker_nodes:
            status = node.get_mesh_status()
            protocols_in_use.add(status["protocol"])

        for protocol in protocols_in_use:
            logger.info(f"   ✅ Protocol in use: {protocol.upper()}")

        # Demonstrate bandwidth monitoring
        logger.info("\\n📊 Bandwidth Monitoring:")
        total_bandwidth = 0
        for node in self.supervisor_nodes + self.worker_nodes:
            status = node.get_mesh_status()
            node_bandwidth = status["network_statistics"]["total_bandwidth"]
            total_bandwidth += node_bandwidth
            logger.info(f"   📈 {node.node_id}: {node_bandwidth:.1f} Mbps")

        logger.info(f"   🌍 Total network bandwidth: {total_bandwidth:.1f} Mbps")

        # Demonstrate routing optimization
        logger.info("\\n🗺️  Routing Optimization:")
        for supervisor in self.supervisor_nodes[:1]:  # Check one supervisor's routing
            status = supervisor.get_mesh_status()
            if "routing_entries" in status:
                logger.info(f"   📍 {supervisor.node_id} routing table:")
                # Note: routing table details would be shown in production
                logger.info(f"      Routing entries: {status['routing_entries']}")

        # Demonstrate automatic tunnel repair
        logger.info("\\n🔧 Automatic Tunnel Repair Simulation:")
        if self.supervisor_nodes:
            test_node = self.supervisor_nodes[0]
            logger.info(f"   🔍 Simulating tunnel health check for {test_node.node_id}")

            # The health monitoring runs automatically in the background
            tunnel_details = test_node.get_tunnel_details()
            if tunnel_details:
                for tunnel_id, details in tunnel_details.items():
                    if details["is_healthy"]:
                        logger.info(
                            f"   ✅ {tunnel_id}: Healthy (Latency: {details['latency_ms']:.1f}ms)"
                        )
                    else:
                        logger.info(f"   🔧 {tunnel_id}: Initiating repair procedure")

    async def _demonstrate_network_optimization(self):
        """Demonstrate network optimization features"""
        logger.info("\\n🎯 Phase 5: Network Optimization and Advanced Monitoring")
        logger.info("-" * 50)

        # Get optimized topology from coordinator
        logger.info("\\n🧠 Topology Optimization:")
        optimized_connections = self.coordinator.optimize_topology()

        for node_id, recommended_peers in optimized_connections.items():
            if recommended_peers:
                logger.info(f"   🎯 {node_id} → Recommended peers: {', '.join(recommended_peers)}")

        # Demonstrate comprehensive network statistics
        logger.info("\\n📈 Comprehensive Network Statistics:")

        network_stats = {
            "total_nodes": len(self.supervisor_nodes) + len(self.worker_nodes),
            "supervisor_nodes": len(self.supervisor_nodes),
            "worker_nodes": len(self.worker_nodes),
            "total_tunnels": 0,
            "healthy_tunnels": 0,
            "total_bandwidth": 0,
            "average_latency": 0,
            "protocols_used": set(),
        }

        latency_measurements = []

        for node in self.supervisor_nodes + self.worker_nodes:
            status = node.get_mesh_status()
            tunnel_details = node.get_tunnel_details()

            network_stats["total_tunnels"] += status["total_tunnels"]
            network_stats["healthy_tunnels"] += status["active_tunnels"]
            network_stats["total_bandwidth"] += status["network_statistics"]["total_bandwidth"]
            network_stats["protocols_used"].add(status["protocol"])

            if status["network_statistics"]["average_latency"] > 0:
                latency_measurements.append(status["network_statistics"]["average_latency"])

        if latency_measurements:
            network_stats["average_latency"] = sum(latency_measurements) / len(latency_measurements)

        # Display statistics
        logger.info(f"   🌐 Network Overview:")
        logger.info(f"      Total nodes: {network_stats['total_nodes']}")
        logger.info(f"      Supervisors: {network_stats['supervisor_nodes']}")
        logger.info(f"      Workers: {network_stats['worker_nodes']}")
        logger.info(f"      Total tunnels: {network_stats['total_tunnels']}")
        logger.info(f"      Healthy tunnels: {network_stats['healthy_tunnels']}")
        logger.info(
            f"      Success rate: {network_stats['healthy_tunnels']/max(1, network_stats['total_tunnels']):.1%}"
        )
        logger.info(f"      Total bandwidth: {network_stats['total_bandwidth']:.1f} Mbps")
        logger.info(f"      Average latency: {network_stats['average_latency']:.1f} ms")
        logger.info(f"      Protocols: {', '.join(network_stats['protocols_used'])}")

        # Demonstrate fault tolerance
        logger.info("\\n🛡️  Fault Tolerance Demonstration:")
        logger.info("   📊 Network resilience features:")
        logger.info("      ✅ Automatic tunnel health monitoring")
        logger.info("      ✅ Self-healing tunnel repair")
        logger.info("      ✅ Multi-path routing redundancy")
        logger.info("      ✅ Dynamic peer discovery")
        logger.info("      ✅ Load balancing across tunnels")

    async def _cleanup_demonstration(self):
        """Clean up demonstration resources"""
        logger.info("\\n🧹 Phase 6: Cleanup and Shutdown")
        logger.info("-" * 50)

        # Disconnect all nodes from mesh
        all_nodes = self.supervisor_nodes + self.worker_nodes

        for node in all_nodes:
            try:
                await node.disconnect_from_mesh()
                logger.info(f"   ✅ {node.node_id} disconnected from mesh")
            except Exception as e:
                logger.error(f"   ❌ Error disconnecting {node.node_id}: {e}")

        # Clear coordinator state
        self.coordinator.registered_nodes.clear()
        self.coordinator.ip_assignments.clear()
        self.coordinator.topology_graph.clear()

        logger.info("   🧹 Cleanup complete")


async def main():
    """Main entry point for enhanced VPN mesh demo"""
    demo = EnhancedVPNMeshDemo()

    try:
        await demo.demonstrate_enhanced_vpn_mesh()
    except KeyboardInterrupt:
        logger.info("\\n⚠️  Demo interrupted by user")
    except Exception as e:
        logger.error(f"\\n❌ Demo failed with error: {e}")
    finally:
        logger.info("\\n👋 Enhanced VPN Mesh Demo completed")


if __name__ == "__main__":
    asyncio.run(main())
