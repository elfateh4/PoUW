#!/usr/bin/env python3
"""
PoUW Network Join Script

Simple script for new devices to easily join the PoUW network.
Automatically detects device capabilities and configures optimal settings.
"""

import asyncio
import argparse
import sys
import os
import platform
import psutil
import json
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from pouw.node import PoUWNode, NodeConfig
    from pouw.economics import NodeRole
    from config import get_config_manager, get_config, get_node_config
except ImportError as e:
    print(f"❌ Error: Could not import PoUW modules: {e}")
    print("Make sure you're in the PoUW directory and dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)


class DeviceDetector:
    """Detect device capabilities and recommend optimal configuration"""
    
    @staticmethod
    def detect_capabilities():
        """Detect device hardware and software capabilities"""
        capabilities = {
            'platform': platform.system(),
            'architecture': platform.machine(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 1),
            'has_gpu': False,
            'gpu_memory': 0,
            'disk_space_gb': round(psutil.disk_usage('/').free / (1024**3), 1),
            'network_interfaces': len(psutil.net_if_addrs()),
            'is_mobile': False,
            'is_laptop': False,
            'is_server': False
        }
        
        # Detect GPU (simplified)
        try:
            import torch
            if torch.cuda.is_available():
                capabilities['has_gpu'] = True
                capabilities['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        except ImportError:
            pass
        
        # Device type detection
        if capabilities['platform'] == 'Darwin':  # macOS
            if capabilities['memory_gb'] <= 8:
                capabilities['is_laptop'] = True
        elif capabilities['platform'] == 'Linux':
            if os.path.exists('/sys/class/power_supply/BAT0'):
                capabilities['is_laptop'] = True
            elif capabilities['cpu_count'] >= 16 and capabilities['memory_gb'] >= 32:
                capabilities['is_server'] = True
        elif capabilities['platform'] == 'Windows':
            if capabilities['memory_gb'] <= 16:
                capabilities['is_laptop'] = True
        
        return capabilities
    
    @staticmethod
    def recommend_role(capabilities):
        """Recommend optimal node role based on capabilities"""
        if capabilities['is_server'] or (capabilities['has_gpu'] and capabilities['memory_gb'] >= 16):
            return NodeRole.SUPERVISOR
        elif capabilities['has_gpu'] and capabilities['memory_gb'] >= 8:
            return NodeRole.MINER
        elif capabilities['memory_gb'] >= 4:
            return NodeRole.VERIFIER
        else:
            return NodeRole.PEER
    
    @staticmethod
    def recommend_stake(capabilities, role):
        """Recommend stake amount based on device capabilities"""
        base_stakes = {
            NodeRole.SUPERVISOR: 500.0,
            NodeRole.MINER: 100.0,
            NodeRole.VERIFIER: 50.0,
            NodeRole.PEER: 10.0
        }
        
        base_stake = base_stakes.get(role, 10.0)
        
        # Adjust based on capabilities
        if capabilities['has_gpu']:
            base_stake *= 2
        if capabilities['memory_gb'] >= 32:
            base_stake *= 1.5
        if capabilities['is_server']:
            base_stake *= 2
            
        return base_stake


class NetworkJoiner:
    """Handle the process of joining the PoUW network with configuration support"""
    
    def __init__(self, config_file=None):
        # Load configuration
        self.config_manager = get_config_manager(env_file=config_file)
        self.config = self.config_manager.get_config()
        self.detector = DeviceDetector()
        self.node = None
        
    def print_banner(self):
        """Print welcome banner"""
        print("=" * 60)
        print("🚀 Welcome to the PoUW Network!")
        print("   Proof of Useful Work - Decentralized ML Computing")
        print("=" * 60)
        print()
    
    def display_capabilities(self, capabilities):
        """Display detected device capabilities"""
        print("🔍 Device Detection Results:")
        print(f"   Platform: {capabilities['platform']} ({capabilities['architecture']})")
        print(f"   CPU Cores: {capabilities['cpu_count']}")
        print(f"   Memory: {capabilities['memory_gb']} GB")
        print(f"   GPU: {'✅ Available' if capabilities['has_gpu'] else '❌ Not detected'}")
        if capabilities['has_gpu']:
            print(f"   GPU Memory: {capabilities['gpu_memory']} GB")
        print(f"   Free Disk: {capabilities['disk_space_gb']} GB")
        
        device_type = "Server" if capabilities['is_server'] else \
                     "Laptop" if capabilities['is_laptop'] else \
                     "Mobile" if capabilities['is_mobile'] else "Desktop"
        print(f"   Device Type: {device_type}")
        print()
    
    def get_user_preferences(self, recommended_role, recommended_stake, bootstrap_peer):
        """Get user preferences for node configuration"""
        print("⚙️  Configuration Options:")
        print(f"   Recommended Role: {recommended_role.value}")
        print(f"   Recommended Stake: {recommended_stake:.1f} PAI tokens")
        print(f"   Bootstrap Peer: {bootstrap_peer}")
        print()
        
        # Get user input
        use_recommended = input("Use recommended settings? (Y/n): ").lower()
        if use_recommended in ['', 'y', 'yes']:
            return recommended_role, recommended_stake, bootstrap_peer
        
        # Custom configuration
        print("\n📝 Custom Configuration:")
        
        # Role selection
        print("Available roles:")
        for i, role in enumerate([NodeRole.PEER, NodeRole.VERIFIER, NodeRole.MINER, NodeRole.SUPERVISOR]):
            print(f"   {i+1}. {role.value}")
        
        while True:
            try:
                role_choice = int(input("Select role (1-4): ")) - 1
                roles = [NodeRole.PEER, NodeRole.VERIFIER, NodeRole.MINER, NodeRole.SUPERVISOR]
                selected_role = roles[role_choice]
                break
            except (ValueError, IndexError):
                print("Invalid choice. Please enter 1-4.")
        
        # Stake amount
        while True:
            try:
                stake_input = input(f"Stake amount (default {recommended_stake:.1f}): ")
                stake = float(stake_input) if stake_input else recommended_stake
                if stake >= 1.0:
                    break
                else:
                    print("Stake must be at least 1.0 PAI")
            except ValueError:
                print("Invalid amount. Please enter a number.")
        
        # Bootstrap peer
        peer_input = input(f"Bootstrap peer (default {bootstrap_peer}): ")
        peer = peer_input if peer_input else bootstrap_peer
        
        return selected_role, stake, peer
    
    async def join_network(self, node_id, role, stake, bootstrap_peer, port=8000):
        """Join the PoUW network with specified configuration"""
        try:
            print(f"🌐 Joining PoUW Network...")
            print(f"   Node ID: {node_id}")
            print(f"   Role: {role.value}")
            print(f"   Stake: {stake:.1f} PAI")
            print(f"   Port: {port}")
            print(f"   Bootstrap: {bootstrap_peer}")
            print()
            
            # Parse bootstrap peer
            if ':' in bootstrap_peer:
                peer_host, peer_port = bootstrap_peer.split(':')
                peer_port = int(peer_port)
            else:
                peer_host = bootstrap_peer
                peer_port = 8000
            
            # Create node configuration
            config = NodeConfig(
                node_id=node_id,
                role=role,
                host="0.0.0.0",
                port=port,
                initial_stake=stake,
                bootstrap_peers=[(peer_host, peer_port)]
            )
            
            # Create and start node
            self.node = PoUWNode(node_id, role, "0.0.0.0", port, config=config)
            await self.node.start()
            
            print("✅ Successfully joined the PoUW network!")
            print()
            
            # Display status
            await self.monitor_node()
            
        except Exception as e:
            print(f"❌ Failed to join network: {e}")
            return False
    
    async def monitor_node(self):
        """Monitor node status and display updates"""
        print("📊 Node Status Monitor (Press Ctrl+C to stop):")
        print()
        
        try:
            while True:
                if self.node is None:
                    print("\r❌ Node not available", end="", flush=True)
                    await asyncio.sleep(5)
                    continue
                    
                status = self.node.get_status()
                peer_count = status.get('peer_count', 0)
                is_running = status.get('is_running', False)
                blockchain_height = status.get('blockchain_height', 0)
                
                print(f"\r🟢 Running | Peers: {peer_count:2d} | Height: {blockchain_height:4d} | "
                      f"Role: {self.node.role.value}", end="", flush=True)
                
                await asyncio.sleep(5)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping node...")
            if self.node:
                await self.node.stop()
            print("✅ Node stopped successfully!")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Join the PoUW Network")
    parser.add_argument("--node-id", help="Unique node identifier")
    parser.add_argument("--role", choices=['PEER', 'VERIFIER', 'MINER', 'SUPERVISOR'], 
                       help="Node role")
    parser.add_argument("--stake", type=float, help="Stake amount")
    parser.add_argument("--bootstrap-peer", default="127.0.0.1:8000",
                       help="Bootstrap peer address (host:port)")
    parser.add_argument("--port", type=int, default=8000, help="Node port")
    parser.add_argument("--auto", action="store_true", 
                       help="Use auto-detected settings without prompts")
    
    args = parser.parse_args()
    
    joiner = NetworkJoiner()
    joiner.print_banner()
    
    # Detect device capabilities
    capabilities = joiner.detector.detect_capabilities()
    joiner.display_capabilities(capabilities)
    
    # Get configuration
    if args.auto:
        # Use auto-detected settings
        role = joiner.detector.recommend_role(capabilities)
        stake = joiner.detector.recommend_stake(capabilities, role)
        bootstrap_peer = args.bootstrap_peer
        node_id = args.node_id or f"auto_{role.value.lower()}_{os.getpid()}"
    else:
        # Interactive configuration
        recommended_role = joiner.detector.recommend_role(capabilities)
        recommended_stake = joiner.detector.recommend_stake(capabilities, recommended_role)
        
        role, stake, bootstrap_peer = joiner.get_user_preferences(
            recommended_role, recommended_stake, args.bootstrap_peer
        )
        
        node_id = args.node_id or input(f"Node ID (default: my_{role.value.lower()}_001): ") or f"my_{role.value.lower()}_001"
    
    # Override with command line arguments
    if args.role:
        role = NodeRole(args.role)
    if args.stake:
        stake = args.stake
    if args.bootstrap_peer != "127.0.0.1:8000":
        bootstrap_peer = args.bootstrap_peer
    
    # Join the network
    await joiner.join_network(node_id, role, stake, bootstrap_peer, args.port)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
