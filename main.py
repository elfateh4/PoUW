#!/usr/bin/env python3
"""
PoUW Main Application Entry Point

This module provides the main entry point for running PoUW nodes in different configurations.
It supports various node types and can be configured via environment variables or command line arguments.
"""

import asyncio
import os
import sys
import signal
import logging
import argparse
from typing import Optional
from pathlib import Path
import json
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pouw.node import PoUWNode, NodeConfig
from pouw.economics import NodeRole

# Simple HTTP server for health checks
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HealthCheckHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for health checks"""
    
    def __init__(self, *args, app_instance=None, **kwargs):
        self.app_instance = app_instance
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/health':
            self.send_health_response()
        elif self.path == '/status':
            self.send_status_response()
        else:
            self.send_error(404, "Not Found")
    
    def send_health_response(self):
        """Send health check response"""
        try:
            health_data = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "service": "pouw-node",
                "version": "1.0.0"
            }
            
            if self.app_instance and self.app_instance.node:
                node_status = self.app_instance.node.get_status()
                health_data.update({
                    "node_id": self.app_instance.node.node_id,
                    "role": self.app_instance.node.role.name,
                    "running": node_status.get("is_running", False),
                    "peers": node_status.get("connected_peers", 0)
                })
            
            response = json.dumps(health_data)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Content-length', str(len(response)))
            self.end_headers()
            self.wfile.write(response.encode())
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            self.send_error(500, "Internal Server Error")
    
    def send_status_response(self):
        """Send detailed status response"""
        try:
            status_data = {
                "timestamp": datetime.now().isoformat(),
                "service": "pouw-node",
                "uptime": "unknown"
            }
            
            if self.app_instance and self.app_instance.node:
                node_status = self.app_instance.node.get_status()
                status_data.update(node_status)
            
            response = json.dumps(status_data, indent=2)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Content-length', str(len(response)))
            self.end_headers()
            self.wfile.write(response.encode())
            
        except Exception as e:
            logger.error(f"Status check error: {e}")
            self.send_error(500, "Internal Server Error")
    
    def log_message(self, format, *args):
        """Override to reduce log noise"""
        return  # Disable default logging


class PoUWApplication:
    """Main PoUW application controller"""
    
    def __init__(self):
        self.node: Optional[PoUWNode] = None
        self.running = False
        self.health_server: Optional[HTTPServer] = None
        self.health_thread: Optional[threading.Thread] = None
        
    def parse_node_role(self, role_str: str) -> NodeRole:
        """Parse node role from string"""
        role_mapping = {
            'MINER': NodeRole.MINER,
            'SUPERVISOR': NodeRole.SUPERVISOR,
            'VERIFIER': NodeRole.VERIFIER,
            'EVALUATOR': NodeRole.EVALUATOR,
            'PEER': NodeRole.PEER
        }
        
        role_str = role_str.upper()
        if role_str not in role_mapping:
            raise ValueError(f"Invalid node role: {role_str}. Valid roles: {list(role_mapping.keys())}")
        
        return role_mapping[role_str]
    
    def parse_bootstrap_peers(self, peers_str: str) -> list:
        """Parse bootstrap peers from comma-separated string"""
        if not peers_str:
            return []
        
        peers = []
        for peer in peers_str.split(','):
            peer = peer.strip()
            if ':' in peer:
                host, port = peer.split(':', 1)
                peers.append((host.strip(), int(port.strip())))
        
        return peers
    
    def create_node_config(self, args) -> NodeConfig:
        """Create node configuration from arguments and environment"""
        
        # Get configuration from environment variables or arguments
        node_id = args.node_id or os.getenv('POUW_NODE_ID', f'node_{args.role.lower()}_{os.getpid()}')
        host = args.host or os.getenv('POUW_HOST', '0.0.0.0')
        port = args.port or int(os.getenv('POUW_PORT', '8000'))
        stake_amount = args.stake or float(os.getenv('POUW_STAKE_AMOUNT', '100.0'))
        max_peers = args.max_peers or int(os.getenv('POUW_MAX_PEERS', '50'))
        
        # Parse bootstrap peers
        bootstrap_peers_str = args.bootstrap_peers or os.getenv('POUW_BOOTSTRAP_PEERS', '')
        bootstrap_peers = self.parse_bootstrap_peers(bootstrap_peers_str)
        
        # Feature flags
        enable_security = args.enable_security or os.getenv('POUW_ENABLE_SECURITY', 'true').lower() == 'true'
        enable_production = args.enable_production or os.getenv('POUW_ENABLE_PRODUCTION_FEATURES', 'false').lower() == 'true'
        enable_advanced = args.enable_advanced or os.getenv('POUW_ENABLE_ADVANCED_FEATURES', 'false').lower() == 'true'
        
        # Mining configuration
        omega_b = float(os.getenv('POUW_MINING_INTENSITY', '0.00001'))
        
        return NodeConfig(
            node_id=node_id,
            role=args.role,
            host=host,
            port=port,
            initial_stake=stake_amount,
            omega_b=omega_b,
            max_peers=max_peers,
            bootstrap_peers=bootstrap_peers,
            enable_security_monitoring=enable_security,
            enable_production_features=enable_production,
            enable_advanced_features=enable_advanced
        )
    
    async def start_node(self, config: NodeConfig):
        """Start the PoUW node"""
        try:
            logger.info(f"Starting PoUW node {config.node_id} as {config.role.value}")
            
            # Create and configure node
            self.node = PoUWNode(
                node_id=config.node_id,
                role=config.role,
                host=config.host,
                port=config.port,
                config=config
            )
            
            # Start the node
            await self.node.start()
            
            # Stake and register if configured
            if config.initial_stake > 0:
                preferences = self.get_node_preferences(config.role)
                ticket = self.node.stake_and_register(config.initial_stake, preferences)
                logger.info(f"Staked {config.initial_stake} PAI with ticket {ticket.ticket_id}")
            
            # Start mining for appropriate node types
            if config.role in [NodeRole.MINER, NodeRole.SUPERVISOR]:
                await self.node.start_mining()
                logger.info("Mining started")
            
            self.running = True
            logger.info(f"PoUW node {config.node_id} started successfully on {config.host}:{config.port}")
            
        except Exception as e:
            logger.error(f"Failed to start node: {e}")
            raise
    
    def get_node_preferences(self, role: NodeRole) -> dict:
        """Get default preferences for node role"""
        base_preferences = {
            'model_types': ['mlp', 'cnn'],
            'has_gpu': False,  # Default to CPU unless GPU is available
            'max_dataset_size': 1000000
        }
        
        if role == NodeRole.MINER:
            base_preferences.update({
                'mining_intensity': float(os.getenv('POUW_MINING_INTENSITY', '0.00001')),
                'preferred_tasks': ['image_classification', 'nlp']
            })
        elif role == NodeRole.SUPERVISOR:
            base_preferences.update({
                'storage_capacity': 10000000,
                'bandwidth': 1000000,
                'redundancy_scheme': 'full_replicas'
            })
        elif role == NodeRole.EVALUATOR:
            base_preferences.update({
                'evaluation_capacity': 100,
                'specialized_metrics': ['accuracy', 'f1_score']
            })
        
        return base_preferences
    
    async def run_forever(self):
        """Run the node indefinitely"""
        try:
            while self.running:
                # Print status periodically
                if self.node:
                    status = self.node.get_status()
                    logger.info(f"Node status - Height: {status.get('blockchain_height', 0)}, "
                               f"Peers: {status.get('peer_count', 0)}, "
                               f"Training: {status.get('is_training', False)}")
                
                await asyncio.sleep(60)  # Status update every minute
                
        except asyncio.CancelledError:
            logger.info("Node operation cancelled")
        except Exception as e:
            logger.error(f"Error during node operation: {e}")
            raise
    
    async def stop_node(self):
        """Stop the PoUW node"""
        if self.node and self.running:
            logger.info("Stopping PoUW node...")
            self.running = False
            await self.node.stop()
            logger.info("PoUW node stopped")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, shutting down...")
            asyncio.create_task(self.stop_node())
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def start_health_server(self, host='0.0.0.0', port=8080):
        """Start the health check HTTP server"""
        if self.health_server:
            logger.warning("Health server is already running")
            return
        
        # Create a handler class that includes the app instance
        class AppHealthHandler(HealthCheckHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, app_instance=self, **kwargs)
        
        try:
            self.health_server = HTTPServer((host, port), AppHealthHandler)
            self.health_thread = threading.Thread(target=self.run_health_server, daemon=True)
            self.health_thread.start()
            
            logger.info(f"Health server started at http://{host}:{port}")
            logger.info(f"Health endpoint: http://{host}:{port}/health")
            logger.info(f"Status endpoint: http://{host}:{port}/status")
        except Exception as e:
            logger.error(f"Failed to start health server: {e}")
    
    def run_health_server(self):
        """Run the health check server loop"""
        try:
            if self.health_server:
                self.health_server.serve_forever()
        except Exception as e:
            logger.error(f"Error in health server: {e}")
    
    def stop_health_server(self):
        """Stop the health check HTTP server"""
        if self.health_server:
            logger.info("Stopping health server...")
            self.health_server.shutdown()
            self.health_server.server_close()
            self.health_server = None
            logger.info("Health server stopped")


async def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description='PoUW Node Application')
    
    # Core configuration
    parser.add_argument('--node-id', type=str, help='Unique node identifier')
    parser.add_argument('--role', type=str, required=True, 
                       choices=['MINER', 'SUPERVISOR', 'VERIFIER', 'EVALUATOR', 'PEER'],
                       help='Node role')
    parser.add_argument('--host', type=str, help='Host address to bind to')
    parser.add_argument('--port', type=int, help='Port to bind to')
    
    # Economic configuration
    parser.add_argument('--stake', type=float, help='Initial stake amount')
    
    # Network configuration
    parser.add_argument('--max-peers', type=int, help='Maximum number of peers')
    parser.add_argument('--bootstrap-peers', type=str, 
                       help='Comma-separated list of bootstrap peers (host:port)')
    
    # Feature flags
    parser.add_argument('--enable-security', action='store_true', 
                       help='Enable security monitoring')
    parser.add_argument('--enable-production', action='store_true',
                       help='Enable production features')
    parser.add_argument('--enable-advanced', action='store_true',
                       help='Enable advanced features')
    
    # Development options
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse role
    try:
        args.role = NodeRole[args.role.upper()]
    except KeyError:
        logger.error(f"Invalid role: {args.role}")
        sys.exit(1)
    
    app = PoUWApplication()
    app.setup_signal_handlers()
    
    try:
        # Create configuration
        config = app.create_node_config(args)
        
        # Start node
        await app.start_node(config)
        
        # Start health server
        app.start_health_server(host=config.host, port=config.port + 1)
        
        # Run forever
        await app.run_forever()
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        await app.stop_node()
        app.stop_health_server()


if __name__ == '__main__':
    # Handle asyncio compatibility
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
