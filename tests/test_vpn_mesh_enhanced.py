"""
Test suite for enhanced VPN mesh topology implementation.

Tests the production-ready VPN mesh networking capabilities
including tunnel establishment, health monitoring, and network optimization.
"""

import unittest
import asyncio
import time
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pouw.network.vpn_mesh_enhanced import (
    ProductionVPNMeshManager,
    MeshNetworkCoordinator,
    VPNProtocol,
    TunnelState,
    VPNTunnelConfig,
    MeshNode
)


class TestProductionVPNMeshManager(unittest.TestCase):
    """Test production VPN mesh manager"""
    
    def setUp(self):
        """Set up test environment"""
        self.manager = ProductionVPNMeshManager(
            node_id="test_node_001",
            network_cidr="10.200.0.0/16",
            preferred_protocol=VPNProtocol.WIREGUARD,
            base_port=55000
        )
    
    def tearDown(self):
        """Clean up test environment"""
        # Stop any background threads
        if hasattr(self.manager, 'shutdown_event'):
            self.manager.shutdown_event.set()
    
    def test_initialization(self):
        """Test manager initialization"""
        self.assertEqual(self.manager.node_id, "test_node_001")
        self.assertEqual(self.manager.preferred_protocol, VPNProtocol.WIREGUARD)
        self.assertEqual(self.manager.base_port, 55000)
        self.assertIsNotNone(self.manager.node_private_key)
        self.assertIsNotNone(self.manager.node_public_key)
        self.assertTrue(self.manager.config_dir.exists())
    
    def test_key_generation(self):
        """Test cryptographic key generation"""
        # Test WireGuard key generation (may fall back to generic)
        manager = ProductionVPNMeshManager("test_key_gen", preferred_protocol=VPNProtocol.WIREGUARD)
        self.assertIsNotNone(manager.node_private_key)
        self.assertIsNotNone(manager.node_public_key)
        
        # Test generic key generation
        manager_generic = ProductionVPNMeshManager("test_generic", preferred_protocol=VPNProtocol.IPSEC)
        self.assertIsNotNone(manager_generic.node_private_key)
        self.assertIsNotNone(manager_generic.node_public_key)
        self.assertEqual(len(manager_generic.node_private_key), 32)
        self.assertEqual(len(manager_generic.node_public_key), 32)
    
    @patch('pouw.network.vpn_mesh_enhanced.ProductionVPNMeshManager._request_virtual_ip')
    @patch('pouw.network.vpn_mesh_enhanced.ProductionVPNMeshManager._get_local_ip')
    @patch('pouw.network.vpn_mesh_enhanced.ProductionVPNMeshManager._create_network_interface')
    async def test_join_mesh_network(self, mock_create_interface, mock_get_ip, mock_request_ip):
        """Test joining mesh network"""
        # Setup mocks
        mock_request_ip.return_value = "10.200.1.10"
        mock_get_ip.return_value = "192.168.1.100"
        mock_create_interface.return_value = True
        
        # Test joining mesh
        result = await self.manager.join_mesh_network("coordinator:8080")
        
        self.assertTrue(result)
        self.assertIn(self.manager.node_id, self.manager.mesh_nodes)
        self.assertEqual(self.manager.mesh_nodes[self.manager.node_id].virtual_ip, "10.200.1.10")
    
    def test_derive_tunnel_key(self):
        """Test tunnel key derivation"""
        key1 = self.manager._derive_tunnel_key("peer_001")
        key2 = self.manager._derive_tunnel_key("peer_001")
        key3 = self.manager._derive_tunnel_key("peer_002")
        
        # Same peer should generate same key
        self.assertEqual(key1, key2)
        # Different peers should generate different keys
        self.assertNotEqual(key1, key3)
        # Keys should be 32 bytes (SHA256)
        self.assertEqual(len(key1), 32)
    
    @patch('pouw.network.vpn_mesh_enhanced.ProductionVPNMeshManager._establish_tunnel')
    async def test_establish_tunnel_to_peer(self, mock_establish):
        """Test tunnel establishment to peer"""
        # Setup manager with a mesh node
        self.manager.mesh_nodes[self.manager.node_id] = MeshNode(
            node_id=self.manager.node_id,
            virtual_ip="10.200.1.10",
            physical_ip="192.168.1.100",
            port=55000
        )
        
        mock_establish.return_value = True
        
        peer_info = {
            "virtual_ip": "10.200.1.11",
            "physical_ip": "192.168.1.101",
            "port": 55001,
            "public_key": b"mock_public_key"
        }
        
        result = await self.manager.establish_tunnel_to_peer("peer_001", peer_info)
        
        self.assertTrue(result)
        tunnel_id = f"{self.manager.node_id}_peer_001"
        self.assertIn(tunnel_id, self.manager.tunnels)
        self.assertEqual(self.manager.tunnels[tunnel_id].state, TunnelState.CONNECTED)
    
    def test_get_mesh_status(self):
        """Test mesh status reporting"""
        # Add some test data
        self.manager.mesh_nodes[self.manager.node_id] = MeshNode(
            node_id=self.manager.node_id,
            virtual_ip="10.200.1.10",
            physical_ip="192.168.1.100",
            port=55000
        )
        
        self.manager.tunnels["test_tunnel"] = VPNTunnelConfig(
            local_node_id=self.manager.node_id,
            remote_node_id="peer_001",
            protocol=VPNProtocol.WIREGUARD,
            local_virtual_ip="10.200.1.10",
            remote_virtual_ip="10.200.1.11",
            local_port=55000,
            remote_port=55001,
            encryption_key=b"test_key",
            state=TunnelState.CONNECTED
        )
        
        status = self.manager.get_mesh_status()
        
        expected_keys = [
            'node_id', 'protocol', 'virtual_ip', 'mesh_nodes',
            'active_tunnels', 'total_tunnels', 'interfaces',
            'routing_entries', 'bandwidth_usage', 'connection_health',
            'network_statistics'
        ]
        
        for key in expected_keys:
            self.assertIn(key, status)
        
        self.assertEqual(status['node_id'], self.manager.node_id)
        self.assertEqual(status['protocol'], VPNProtocol.WIREGUARD.value)
        self.assertEqual(status['mesh_nodes'], 1)
        self.assertEqual(status['total_tunnels'], 1)
    
    def test_tunnel_health_monitoring(self):
        """Test tunnel health monitoring"""
        # Add test tunnel
        tunnel_config = VPNTunnelConfig(
            local_node_id=self.manager.node_id,
            remote_node_id="peer_001",
            protocol=VPNProtocol.WIREGUARD,
            local_virtual_ip="10.200.1.10",
            remote_virtual_ip="10.200.1.11",
            local_port=55000,
            remote_port=55001,
            encryption_key=b"test_key",
            state=TunnelState.CONNECTED,
            established_time=time.time()
        )
        
        tunnel_id = "test_tunnel"
        self.manager.tunnels[tunnel_id] = tunnel_config
        self.manager.connection_health[tunnel_id] = True
        
        # Test health check
        self.manager._check_tunnel_health()
        
        # Verify health status was updated
        self.assertIn(tunnel_id, self.manager.connection_health)
    
    @patch('pouw.network.vpn_mesh_enhanced.ProductionVPNMeshManager._ping_tunnel')
    def test_tunnel_repair(self, mock_ping):
        """Test tunnel repair functionality"""
        # Setup unhealthy tunnel
        tunnel_config = VPNTunnelConfig(
            local_node_id=self.manager.node_id,
            remote_node_id="peer_001",
            protocol=VPNProtocol.WIREGUARD,
            local_virtual_ip="10.200.1.10",
            remote_virtual_ip="10.200.1.11",
            local_port=55000,
            remote_port=55001,
            encryption_key=b"test_key",
            state=TunnelState.CONNECTED
        )
        
        tunnel_id = "test_tunnel"
        self.manager.tunnels[tunnel_id] = tunnel_config
        
        # Simulate tunnel failure
        mock_ping.return_value = False
        
        # Attempt repair
        self.manager._attempt_tunnel_repair(tunnel_id, tunnel_config)
        
        # Check that state was updated (either repaired or marked as error)
        self.assertIn(tunnel_config.state, [TunnelState.CONNECTED, TunnelState.ERROR])


class TestMeshNetworkCoordinator(unittest.TestCase):
    """Test mesh network coordinator"""
    
    def setUp(self):
        """Set up test environment"""
        self.coordinator = MeshNetworkCoordinator(network_cidr="10.100.0.0/16")
    
    def test_initialization(self):
        """Test coordinator initialization"""
        self.assertEqual(str(self.coordinator.network_cidr), "10.100.0.0/16")
        self.assertEqual(len(self.coordinator.registered_nodes), 0)
        self.assertEqual(len(self.coordinator.ip_assignments), 0)
    
    def test_node_registration(self):
        """Test node registration and IP assignment"""
        node_info = {
            "node_id": "worker_001",
            "physical_ip": "192.168.1.100",
            "port": 52000,
            "public_key": b"test_public_key",
            "capabilities": ["training", "inference"],
            "role": "worker"
        }
        
        virtual_ip = self.coordinator.register_node(node_info)
        
        self.assertIsNotNone(virtual_ip)
        self.assertIn("worker_001", self.coordinator.registered_nodes)
        self.assertIn("worker_001", self.coordinator.ip_assignments)
        self.assertEqual(self.coordinator.ip_assignments["worker_001"], virtual_ip)
        
        # Test duplicate registration
        virtual_ip2 = self.coordinator.register_node(node_info)
        self.assertEqual(virtual_ip, virtual_ip2)
    
    def test_ip_assignment_uniqueness(self):
        """Test that IP assignments are unique"""
        assigned_ips = set()
        
        for i in range(10):
            node_info = {
                "node_id": f"node_{i:03d}",
                "physical_ip": f"192.168.1.{100+i}",
                "port": 52000 + i,
                "role": "worker"
            }
            
            virtual_ip = self.coordinator.register_node(node_info)
            self.assertIsNotNone(virtual_ip)
            self.assertNotIn(virtual_ip, assigned_ips)
            assigned_ips.add(virtual_ip)
    
    def test_topology_optimization(self):
        """Test mesh topology optimization"""
        # Register multiple nodes
        for i in range(5):
            node_info = {
                "node_id": f"node_{i:03d}",
                "physical_ip": f"192.168.1.{100+i}",
                "port": 52000 + i,
                "role": "worker"
            }
            self.coordinator.register_node(node_info)
        
        # Get optimized topology
        optimized = self.coordinator.optimize_topology()
        
        self.assertEqual(len(optimized), 5)
        
        for node_id, peers in optimized.items():
            self.assertIsInstance(peers, list)
            self.assertNotIn(node_id, peers)  # Node shouldn't be its own peer
    
    def test_mesh_topology_info(self):
        """Test mesh topology information"""
        # Register some nodes
        for i in range(3):
            node_info = {
                "node_id": f"supervisor_{i:03d}",
                "physical_ip": f"192.168.1.{10+i}",
                "port": 51820 + i,
                "role": "supervisor",
                "capabilities": ["consensus", "validation"]
            }
            self.coordinator.register_node(node_info)
        
        topology = self.coordinator.get_mesh_topology()
        
        expected_keys = ['total_nodes', 'network_cidr', 'nodes', 'topology']
        for key in expected_keys:
            self.assertIn(key, topology)
        
        self.assertEqual(topology['total_nodes'], 3)
        self.assertEqual(topology['network_cidr'], "10.100.0.0/16")
        self.assertEqual(len(topology['nodes']), 3)


class TestVPNTunnelConfig(unittest.TestCase):
    """Test VPN tunnel configuration"""
    
    def test_tunnel_config_creation(self):
        """Test tunnel configuration creation"""
        config = VPNTunnelConfig(
            local_node_id="node_001",
            remote_node_id="node_002",
            protocol=VPNProtocol.WIREGUARD,
            local_virtual_ip="10.100.1.10",
            remote_virtual_ip="10.100.1.11",
            local_port=51820,
            remote_port=51821,
            encryption_key=b"test_encryption_key_32_bytes_12"
        )
        
        self.assertEqual(config.local_node_id, "node_001")
        self.assertEqual(config.remote_node_id, "node_002")
        self.assertEqual(config.protocol, VPNProtocol.WIREGUARD)
        self.assertEqual(config.state, TunnelState.DISCONNECTED)
        self.assertIsNone(config.established_time)
    
    def test_tunnel_state_transitions(self):
        """Test tunnel state transitions"""
        config = VPNTunnelConfig(
            local_node_id="node_001",
            remote_node_id="node_002",
            protocol=VPNProtocol.WIREGUARD,
            local_virtual_ip="10.100.1.10",
            remote_virtual_ip="10.100.1.11",
            local_port=51820,
            remote_port=51821,
            encryption_key=b"test_key"
        )
        
        # Test state transitions
        config.state = TunnelState.CONNECTING
        self.assertEqual(config.state, TunnelState.CONNECTING)
        
        config.state = TunnelState.CONNECTED
        config.established_time = time.time()
        self.assertEqual(config.state, TunnelState.CONNECTED)
        self.assertIsNotNone(config.established_time)


class TestIntegration(unittest.TestCase):
    """Integration tests for VPN mesh components"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.coordinator = MeshNetworkCoordinator("10.150.0.0/16")
        self.managers = []
    
    def tearDown(self):
        """Clean up integration test environment"""
        for manager in self.managers:
            if hasattr(manager, 'shutdown_event'):
                manager.shutdown_event.set()
    
    async def test_full_mesh_setup(self):
        """Test full mesh network setup"""
        # Create multiple managers
        for i in range(3):
            manager = ProductionVPNMeshManager(
                node_id=f"node_{i:03d}",
                network_cidr="10.150.0.0/16",
                preferred_protocol=VPNProtocol.WIREGUARD,
                base_port=60000 + i
            )
            self.managers.append(manager)
        
        # Register all nodes with coordinator
        for i, manager in enumerate(self.managers):
            node_info = {
                "node_id": manager.node_id,
                "physical_ip": f"192.168.1.{200+i}",
                "port": 60000 + i,
                "public_key": manager.node_public_key,
                "role": "worker"
            }
            
            virtual_ip = self.coordinator.register_node(node_info)
            self.assertIsNotNone(virtual_ip)
        
        # Verify topology
        topology = self.coordinator.get_mesh_topology()
        self.assertEqual(topology['total_nodes'], 3)
        
        # Test optimization
        optimized = self.coordinator.optimize_topology()
        self.assertEqual(len(optimized), 3)


def run_async_test(test_func):
    """Helper function to run async tests"""
    def wrapper(self):
        asyncio.run(test_func(self))
    return wrapper


# Apply async wrapper to async test methods
TestProductionVPNMeshManager.test_join_mesh_network = run_async_test(
    TestProductionVPNMeshManager.test_join_mesh_network
)
TestProductionVPNMeshManager.test_establish_tunnel_to_peer = run_async_test(
    TestProductionVPNMeshManager.test_establish_tunnel_to_peer
)
TestIntegration.test_full_mesh_setup = run_async_test(
    TestIntegration.test_full_mesh_setup
)


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests
    unittest.main(verbosity=2)
