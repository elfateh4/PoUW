#!/usr/bin/env python3
"""
PoUW CLI - Command Line Interface for PoUW Node Management

This module provides a comprehensive command-line interface for managing
PoUW blockchain nodes including starting, stopping, monitoring, and 
configuration.

Usage:
    pouw-cli start --node-id worker-1 --node-type worker
    pouw-cli stop --node-id worker-1
    pouw-cli status --node-id worker-1
    pouw-cli logs --node-id worker-1 --tail 100
    pouw-cli config create --template worker
    pouw-cli interactive  # Enter interactive mode
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
import psutil
import logging
from datetime import datetime
import tarfile
import shutil
import tempfile
import socket
import urllib.request

from .node import PoUWNode, NodeConfiguration


class InteractiveMode:
    """Interactive CLI mode for easier node management"""
    
    def __init__(self, cli_instance):
        self.cli = cli_instance
        self.running = True
    
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def print_header(self):
        """Print the main header"""
        print("=" * 60)
        print("🔗  PoUW Blockchain Node Management - Interactive Mode  🔗")
        print("=" * 60)
        print()
    
    def print_menu(self):
        """Print the main menu"""
        print("📋 Main Menu:")
        print("  1. 🚀 Start Node")
        print("  2. 🛑 Stop Node") 
        print("  3. 🔄 Restart Node")
        print("  4. 📊 Node Status")
        print("  5. 📋 List All Nodes")
        print("  6. 📝 View Logs")
        print("  7. ⚙️  Configuration Management")
        print("  8. 🔧 Advanced Options")
        print("  9. 💰 Wallet & Mining")
        print(" 10. 📦 Import/Export Account")
        print(" 11. 🌐 Peer Management")
        print(" 12. 🧠 ML Task Management")
        print("  0. 🚪 Exit")
        print()
    
    def get_input(self, prompt: str, 
                  default: Optional[str] = None) -> str:
        """Get user input with optional default"""
        if default:
            user_input = input(f"{prompt} (default: {default}): ").strip()
            return user_input if user_input else default
        return input(f"{prompt}: ").strip()
    
    def get_choice(self, prompt: str, choices: List[str], 
                   default: Optional[str] = None) -> str:
        """Get user choice from a list of options"""
        print(f"{prompt}:")
        for i, choice in enumerate(choices, 1):
            print(f"  {i}. {choice}")
        
        while True:
            try:
                if default:
                    choice_input = input(
                        f"Choose (1-{len(choices)}) "
                        f"[default: {default}]: ").strip()
                    if not choice_input:
                        return default
                else:
                    choice_input = input(
                        f"Choose (1-{len(choices)}): ").strip()
                
                choice_num = int(choice_input)
                if 1 <= choice_num <= len(choices):
                    return choices[choice_num - 1]
                else:
                    print("❌ Invalid choice. Please try again.")
            except ValueError:
                print("❌ Please enter a valid number.")
    
    def confirm(self, message: str) -> bool:
        """Get yes/no confirmation"""
        while True:
            response = input(f"{message} (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            print("❌ Please enter 'y' or 'n'")
    
    def pause(self):
        """Pause and wait for user input"""
        input("\nPress Enter to continue...")
    
    def display_nodes_table(self, nodes: List[Dict]):
        """Display nodes in a formatted table"""
        if not nodes:
            print("📭 No nodes configured")
            return
        
        print("📋 Configured Nodes:")
        print("-" * 70)
        print(f"{'Node ID':<20} {'Type':<12} {'Port':<8} {'Status':<10} "
              f"{'PID':<8}")
        print("-" * 70)
        for node in nodes:
            status_emoji = "🟢" if node['status'] == 'running' else "🔴"
            print(f"{node['node_id']:<20} {node['node_type']:<12} "
                  f"{node['port']:<8} {status_emoji}{node['status']:<9} "
                  f"{node['pid'] or '':<8}")
        print("-" * 70)
    
    async def start_node_interactive(self):
        """Interactive node start wizard"""
        print("\n🚀 Start Node Wizard")
        print("-" * 30)
        
        node_id = self.get_input("Enter node ID")
        if not node_id:
            print("❌ Node ID is required")
            return
        
        if self.cli.is_node_running(node_id):
            print(f"⚠️  Node {node_id} is already running")
            return
        
        # Check if config exists
        config = self.cli.load_config(node_id)
        if config:
            print(f"✅ Found existing configuration for {node_id}")
            use_existing = self.confirm("Use existing configuration?")
            if not use_existing:
                config = None
        
        if not config:
            print("\n⚙️  Node Configuration:")
            node_type = self.get_choice(
                "Select node type",
                ["client", "miner", "supervisor", "evaluator", "verifier", "peer"],
                "miner"
            )
            
            port = self.get_input("Port", "8333")
            try:
                port = int(port)
            except ValueError:
                port = 8333
            
            enable_mining = self.confirm("Enable mining?")
            enable_training = self.confirm("Enable ML training?")
            enable_gpu = self.confirm("Enable GPU acceleration?")
            
            config_data = self.cli.create_default_config(
                node_id, node_type,
                port=port, mining=enable_mining,
                training=enable_training, gpu=enable_gpu
            )
            self.cli.save_config(node_id, config_data)
            print("✅ Configuration saved")
        
        daemon_mode = self.confirm("Run in daemon mode (background)?")
        
        print(f"\n🚀 Starting node {node_id}...")
        try:
            success = await self.cli.start_node(node_id, daemon=daemon_mode)
            if success:
                print(f"✅ Node {node_id} started successfully!")
            else:
                print(f"❌ Failed to start node {node_id}")
        except Exception as e:
            print(f"❌ Error starting node: {e}")
    
    def stop_node_interactive(self):
        """Interactive node stop"""
        print("\n🛑 Stop Node")
        print("-" * 20)
        
        nodes = [n for n in self.cli.list_nodes() if n['status'] == 'running']
        if not nodes:
            print("📭 No running nodes to stop")
            return
        
        print("Running nodes:")
        for i, node in enumerate(nodes, 1):
            print(f"  {i}. {node['node_id']} "
                  f"(Type: {node['node_type']}, Port: {node['port']})")
        
        try:
            choice = int(input(f"Select node to stop (1-{len(nodes)}): "))
            if 1 <= choice <= len(nodes):
                node_id = nodes[choice - 1]['node_id']
                force = self.confirm("Force kill (not recommended)?")
                
                print(f"\n🛑 Stopping node {node_id}...")
                success = self.cli.stop_node(node_id, force=force)
                if success:
                    print(f"✅ Node {node_id} stopped successfully!")
                else:
                    print(f"❌ Failed to stop node {node_id}")
            else:
                print("❌ Invalid choice")
        except ValueError:
            print("❌ Please enter a valid number")
    
    def restart_node_interactive(self):
        """Interactive node restart"""
        print("\n🔄 Restart Node")
        print("-" * 20)
        
        nodes = self.cli.list_nodes()
        if not nodes:
            print("📭 No nodes configured")
            return
        
        print("Available nodes:")
        for i, node in enumerate(nodes, 1):
            status_emoji = "🟢" if node['status'] == 'running' else "🔴"
            print(f"  {i}. {node['node_id']} {status_emoji}")
        
        try:
            choice = int(input(f"Select node to restart (1-{len(nodes)}): "))
            if 1 <= choice <= len(nodes):
                node_id = nodes[choice - 1]['node_id']
                
                print(f"\n🔄 Restarting node {node_id}...")
                
                # Handle async restart properly
                try:
                    loop = asyncio.get_running_loop()
                    # Create task to restart the node
                    loop.create_task(self.cli.restart_node_async(node_id))
                    print(f"✅ Node {node_id} restart initiated!")
                    print("💡 Use 'Node Status' to check if restart "
                          "completed.")
                except RuntimeError:
                    # Fallback for environments without running loop
                    success = self.cli.restart_node(node_id)
                    if success:
                        print(f"✅ Node {node_id} restarted successfully!")
                    else:
                        print(f"❌ Failed to restart node {node_id}")
            else:
                print("❌ Invalid choice")
        except ValueError:
            print("❌ Please enter a valid number")
    
    def view_status_interactive(self):
        """Interactive status viewing"""
        print("\n📊 Node Status")
        print("-" * 20)
        
        nodes = self.cli.list_nodes()
        if not nodes:
            print("📭 No nodes configured")
            return
        
        print("Select option:")
        print("  1. Show all nodes")
        print("  2. Show specific node")
        
        try:
            choice = int(input("Choose (1-2): "))
            if choice == 1:
                self.display_nodes_table(nodes)
                
                # Show detailed info for running nodes
                running_nodes = [n for n in nodes if n['status'] == 'running']
                if running_nodes and self.confirm(
                        "\nShow detailed status for running nodes?"):
                    for node in running_nodes:
                        status = self.cli.get_node_status(node['node_id'])
                        print(f"\n📊 {node['node_id']} Details:")
                        if 'cpu_percent' in status:
                            print(f"  💻 CPU: {status['cpu_percent']:.1f}%")
                            mem_mb = (status['memory_info']['rss'] / 
                                     1024 / 1024)
                            print(f"  🧠 Memory: {mem_mb:.1f} MB")
                            print(f"  🔗 Connections: "
                                  f"{status['connections']}")
            
            elif choice == 2:
                print("Available nodes:")
                for i, node in enumerate(nodes, 1):
                    status_emoji = ("🟢" if node['status'] == 'running' 
                                  else "🔴")
                    print(f"  {i}. {node['node_id']} {status_emoji}")
                
                node_choice = int(input(f"Select node (1-{len(nodes)}): "))
                if 1 <= node_choice <= len(nodes):
                    node_id = nodes[node_choice - 1]['node_id']
                    status = self.cli.get_node_status(node_id)
                    
                    print(f"\n📊 Status for {node_id}:")
                    status_icon = ("🟢" if status['status'] == 'running' 
                                 else "🔴")
                    print(f"  Status: {status_icon} {status['status']}")
                    if status['pid']:
                        print(f"  PID: {status['pid']}")
                    if 'cpu_percent' in status:
                        print(f"  CPU: {status['cpu_percent']:.1f}%")
                        mem_mb = (status['memory_info']['rss'] / 
                                 1024 / 1024)
                        print(f"  Memory: {mem_mb:.1f} MB")
                        print(f"  Connections: {status['connections']}")
                    if 'log_file' in status:
                        print(f"  Log file: {status['log_file']}")
                        print(f"  Log size: {status['log_size']} bytes")
                else:
                    print("❌ Invalid choice")
            else:
                print("❌ Invalid choice")
        except ValueError:
            print("❌ Please enter a valid number")
    
    def view_logs_interactive(self):
        """Interactive log viewing"""
        print("\n📝 View Logs")
        print("-" * 20)
        
        nodes = self.cli.list_nodes()
        if not nodes:
            print("📭 No nodes configured")
            return
        
        print("Available nodes:")
        for i, node in enumerate(nodes, 1):
            log_file = self.cli.get_node_log_file(node['node_id'])
            log_exists = "📝" if log_file.exists() else "📭"
            print(f"  {i}. {node['node_id']} {log_exists}")
        
        try:
            choice = int(input(f"Select node (1-{len(nodes)}): "))
            if 1 <= choice <= len(nodes):
                node_id = nodes[choice - 1]['node_id']
                
                lines = self.get_input("Number of lines to show", "50")
                try:
                    lines = int(lines)
                except ValueError:
                    lines = 50
                
                follow = self.confirm("Follow logs in real-time?")
                
                print(f"\n📝 Logs for {node_id}:")
                print("-" * 40)
                self.cli.show_logs(node_id, lines=lines, follow=follow)
            else:
                print("❌ Invalid choice")
        except ValueError:
            print("❌ Please enter a valid number")
    
    def config_management_interactive(self):
        """Interactive configuration management"""
        print("\n⚙️  Configuration Management")
        print("-" * 35)
        
        print("Select action:")
        print("  1. 📝 Create new configuration")
        print("  2. 👁️  Show configuration")
        print("  3. ✏️  Edit configuration")
        print("  4. 🗑️  Delete configuration")
        
        try:
            choice = int(input("Choose (1-4): "))
            
            if choice == 1:
                self.create_config_interactive()
            elif choice == 2:
                self.show_config_interactive()
            elif choice == 3:
                self.edit_config_interactive()
            elif choice == 4:
                self.delete_config_interactive()
            else:
                print("❌ Invalid choice")
        except ValueError:
            print("❌ Please enter a valid number")
    
    def create_config_interactive(self):
        """Interactive configuration creation"""
        print("\n📝 Create Configuration")
        print("-" * 25)
        
        node_id = self.get_input("Enter node ID")
        if not node_id:
            print("❌ Node ID is required")
            return
        
        if self.cli.load_config(node_id):
            if not self.confirm(
                    f"Configuration for {node_id} exists. Overwrite?"):
                return
        
        template = self.get_choice(
            "Select template",
            ["client", "miner", "supervisor", "evaluator", "verifier", "peer"],
            "miner"
        )
        
        port = self.get_input("Port", "8333")
        try:
            port = int(port)
        except ValueError:
            port = 8333
        
        enable_mining = self.confirm("Enable mining?")
        enable_training = self.confirm("Enable ML training?")
        enable_gpu = self.confirm("Enable GPU acceleration?")
        
        config = self.cli.create_default_config(
            node_id, template,
            port=port, mining=enable_mining,
            training=enable_training, gpu=enable_gpu
        )
        
        self.cli.save_config(node_id, config)
        print(f"✅ Configuration created for {node_id}")
    
    def show_config_interactive(self):
        """Interactive configuration display"""
        print("\n👁️  Show Configuration")
        print("-" * 25)
        
        nodes = self.cli.list_nodes()
        if not nodes:
            print("📭 No configurations found")
            return
        
        print("Available configurations:")
        for i, node in enumerate(nodes, 1):
            print(f"  {i}. {node['node_id']}")
        
        try:
            choice = int(input(f"Select configuration (1-{len(nodes)}): "))
            if 1 <= choice <= len(nodes):
                node_id = nodes[choice - 1]['node_id']
                config = self.cli.load_config(node_id)
                
                print(f"\n⚙️  Configuration for {node_id}:")
                print("-" * 40)
                print(json.dumps(config, indent=2))
            else:
                print("❌ Invalid choice")
        except ValueError:
            print("❌ Please enter a valid number")
    
    def edit_config_interactive(self):
        """Interactive configuration editing"""
        print("\n✏️  Edit Configuration")
        print("-" * 25)
        
        nodes = self.cli.list_nodes()
        if not nodes:
            print("📭 No configurations found")
            return
        
        print("Available configurations:")
        for i, node in enumerate(nodes, 1):
            print(f"  {i}. {node['node_id']}")
        
        try:
            choice = int(input(f"Select configuration (1-{len(nodes)}): "))
            if 1 <= choice <= len(nodes):
                node_id = nodes[choice - 1]['node_id']
                config_file = self.cli.get_node_config_file(node_id)
                
                editor = os.environ.get('EDITOR', 'nano')
                print(f"Opening {config_file} with {editor}...")
                subprocess.run([editor, str(config_file)])
                print("✅ Configuration editing completed")
            else:
                print("❌ Invalid choice")
        except ValueError:
            print("❌ Please enter a valid number")
    
    def delete_config_interactive(self):
        """Interactive configuration deletion"""
        print("\n🗑️  Delete Configuration")
        print("-" * 25)
        
        nodes = self.cli.list_nodes()
        if not nodes:
            print("📭 No configurations found")
            return
        
        print("Available configurations:")
        for i, node in enumerate(nodes, 1):
            status_emoji = "🟢" if node['status'] == 'running' else "🔴"
            print(f"  {i}. {node['node_id']} {status_emoji}")
        
        try:
            choice = int(input(f"Select configuration (1-{len(nodes)}): "))
            if 1 <= choice <= len(nodes):
                node = nodes[choice - 1]
                node_id = node['node_id']
                
                if node['status'] == 'running':
                    print(f"⚠️  Node {node_id} is currently running!")
                    if not self.confirm("Stop node and delete configuration?"):
                        return
                    self.cli.stop_node(node_id)
                
                if self.confirm(f"Really delete configuration for {node_id}?"):
                    config_file = self.cli.get_node_config_file(node_id)
                    config_file.unlink()
                    print(f"✅ Configuration for {node_id} deleted")
                else:
                    print("❌ Deletion cancelled")
            else:
                print("❌ Invalid choice")
        except ValueError:
            print("❌ Please enter a valid number")
    
    def advanced_options_interactive(self):
        """Advanced options menu"""
        print("\n🔧 Advanced Options")
        print("-" * 25)
        
        print("Select option:")
        print("  1. 🧹 Clean up old logs")
        print("  2. 🔍 System diagnostics")
        print("  3. 📊 Resource monitoring")
        print("  4. 🔄 Bulk operations")
        
        try:
            choice = int(input("Choose (1-4): "))
            
            if choice == 1:
                self.cleanup_logs_interactive()
            elif choice == 2:
                self.system_diagnostics_interactive()
            elif choice == 3:
                self.resource_monitoring_interactive()
            elif choice == 4:
                self.bulk_operations_interactive()
            else:
                print("❌ Invalid choice")
        except ValueError:
            print("❌ Please enter a valid number")
    
    def cleanup_logs_interactive(self):
        """Interactive log cleanup"""
        print("\n🧹 Clean Up Logs")
        print("-" * 20)
        
        log_files = list(self.cli.logs_dir.glob("*.log"))
        if not log_files:
            print("📭 No log files found")
            return
        
        total_size = sum(f.stat().st_size for f in log_files)
        print(f"Found {len(log_files)} log files "
              f"({total_size / 1024 / 1024:.2f} MB total)")
        
        days = self.get_input("Delete logs older than X days", "7")
        try:
            days = int(days)
        except ValueError:
            days = 7
        
        if self.confirm(f"Delete logs older than {days} days?"):
            import time
            cutoff = time.time() - (days * 24 * 3600)
            deleted = 0
            for log_file in log_files:
                if log_file.stat().st_mtime < cutoff:
                    log_file.unlink()
                    deleted += 1
            print(f"✅ Deleted {deleted} old log files")
    
    def system_diagnostics_interactive(self):
        """System diagnostics"""
        print("\n🔍 System Diagnostics")
        print("-" * 25)
        
        # Check system resources
        print("💻 System Resources:")
        print(f"  CPU Usage: {psutil.cpu_percent()}%")
        print(f"  Memory Usage: {psutil.virtual_memory().percent}%")
        print(f"  Disk Usage: {psutil.disk_usage('/').percent}%")
        
        # Check Python environment
        print("\n🐍 Python Environment:")
        print(f"  Version: {sys.version.split()[0]}")
        print(f"  Executable: {sys.executable}")
        
        # Check PoUW components
        print("\n🔗 PoUW Status:")
        nodes = self.cli.list_nodes()
        running_nodes = [n for n in nodes if n['status'] == 'running']
        print(f"  Total Nodes: {len(nodes)}")
        print(f"  Running Nodes: {len(running_nodes)}")
        
        # Check directories
        print("\n📁 Directories:")
        for name, path in [("Configs", self.cli.configs_dir),
                           ("Logs", self.cli.logs_dir),
                           ("PIDs", self.cli.pids_dir)]:
            exists = "✅" if path.exists() else "❌"
            count = len(list(path.glob("*"))) if path.exists() else 0
            print(f"  {name}: {exists} ({count} files)")
    
    def resource_monitoring_interactive(self):
        """Resource monitoring"""
        print("\n📊 Resource Monitoring")
        print("-" * 25)
        
        running_nodes = [n for n in self.cli.list_nodes() 
                        if n['status'] == 'running']
        if not running_nodes:
            print("📭 No running nodes to monitor")
            return
        
        print("Running nodes:")
        for node in running_nodes:
            status = self.cli.get_node_status(node['node_id'])
            print(f"\n🔗 {node['node_id']}:")
            if 'cpu_percent' in status:
                print(f"  💻 CPU: {status['cpu_percent']:.1f}%")
                mem_mb = status['memory_info']['rss'] / 1024 / 1024
                print(f"  🧠 Memory: {mem_mb:.1f} MB")
                print(f"  🔗 Connections: {status['connections']}")
                # Fix the timestamp conversion issue
                if 'create_time' in status:
                    try:
                        # Handle both timestamp and ISO string formats
                        create_time_val = status['create_time']
                        if isinstance(create_time_val, str):
                            # It's already an ISO string, parse it
                            uptime_dt = datetime.fromisoformat(
                                create_time_val.replace('Z', '+00:00'))
                        else:
                            # It's a timestamp
                            uptime_dt = datetime.fromtimestamp(create_time_val)
                        print(f"  ⏰ Uptime: {uptime_dt}")
                    except (ValueError, TypeError) as e:
                        print(f"  ⏰ Uptime: [Error: {e}]")
    
    def bulk_operations_interactive(self):
        """Bulk operations menu"""
        print("\n🔄 Bulk Operations")
        print("-" * 20)
        
        print("Select operation:")
        print("  1. 🚀 Start all stopped nodes")
        print("  2. 🛑 Stop all running nodes")
        print("  3. 🔄 Restart all nodes")
        print("  4. 📊 Export all configurations")
        
        try:
            choice = int(input("Choose (1-4): "))
            
            if choice == 1:
                # Run the async function properly
                import asyncio
                asyncio.run(self.bulk_start_nodes())
            elif choice == 2:
                self.bulk_stop_nodes()
            elif choice == 3:
                self.bulk_restart_nodes()
            elif choice == 4:
                self.export_configurations()
            else:
                print("❌ Invalid choice")
        except ValueError:
            print("❌ Please enter a valid number")
    
    async def bulk_start_nodes(self):
        """Start all stopped nodes"""
        nodes = self.cli.list_nodes()
        stopped_nodes = [n for n in nodes if n['status'] == 'stopped']
        
        if not stopped_nodes:
            print("📭 No stopped nodes to start")
            return
        
        print(f"Found {len(stopped_nodes)} stopped nodes:")
        for node in stopped_nodes:
            print(f"  - {node['node_id']}")
        
        if self.confirm("Start all stopped nodes?"):
            for node in stopped_nodes:
                print(f"🚀 Starting {node['node_id']}...")
                try:
                    success = await self.cli.start_node(node['node_id'])
                    status = "✅" if success else "❌"
                    print(f"  {status} {node['node_id']}")
                except Exception as e:
                    print(f"  ❌ {node['node_id']}: {e}")
    
    def bulk_stop_nodes(self):
        """Stop all running nodes"""
        nodes = self.cli.list_nodes()
        running_nodes = [n for n in nodes if n['status'] == 'running']
        
        if not running_nodes:
            print("📭 No running nodes to stop")
            return
        
        print(f"Found {len(running_nodes)} running nodes:")
        for node in running_nodes:
            print(f"  - {node['node_id']}")
        
        if self.confirm("Stop all running nodes?"):
            for node in running_nodes:
                print(f"🛑 Stopping {node['node_id']}...")
                success = self.cli.stop_node(node['node_id'])
                status = "✅" if success else "❌"
                print(f"  {status} {node['node_id']}")
    
    def bulk_restart_nodes(self):
        """Restart all nodes"""
        nodes = self.cli.list_nodes()
        
        if not nodes:
            print("📭 No nodes to restart")
            return
        
        print(f"Found {len(nodes)} nodes:")
        for node in nodes:
            status_emoji = "🟢" if node['status'] == 'running' else "🔴"
            print(f"  - {node['node_id']} {status_emoji}")
        
        if self.confirm("Restart all nodes?"):
            for node in nodes:
                print(f"🔄 Restarting {node['node_id']}...")
                success = self.cli.restart_node(node['node_id'])
                status = "✅" if success else "❌"
                print(f"  {status} {node['node_id']}")
    
    def export_configurations(self):
        """Export all configurations"""
        nodes = self.cli.list_nodes()
        
        if not nodes:
            print("📭 No configurations to export")
            return
        
        export_data = {}
        for node in nodes:
            config = self.cli.load_config(node['node_id'])
            if config:
                export_data[node['node_id']] = config
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"pouw_configs_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✅ Exported {len(export_data)} configurations to {filename}")
    
    def show_help(self):
        """Show help information"""
        print("\n❓ Help")
        print("-" * 10)
        print("PoUW Interactive Mode provides an easy-to-use interface")
        print("for managing your PoUW blockchain nodes.")
        print()
        print("🔧 Features:")
        print("  • Start, stop, and restart nodes")
        print("  • Monitor node status and resource usage")
        print("  • View and follow log files")
        print("  • Create and manage configurations")
        print("  • Bulk operations for multiple nodes")
        print("  • System diagnostics and cleanup tools")
        print()
        print("💡 Tips:")
        print("  • Use Ctrl+C to interrupt long-running operations")
        print("  • Log following can be stopped with Ctrl+C")
        print("  • Configuration files are stored in ./configs/")
        print("  • Log files are stored in ./logs/")
        print()
        print("📖 For detailed documentation, see CLI_GUIDE.md")
    
    async def run(self):
        """Run the interactive CLI"""
        while self.running:
            self.clear_screen()
            self.print_header()
            self.print_menu()
            
            choice = input("Enter your choice (0-12): ").strip()
            
            if choice == "0":
                print("👋 Goodbye!")
                self.running = False
            elif choice == "1":
                await self.start_node_interactive()
            elif choice == "2":
                self.stop_node_interactive()
            elif choice == "3":
                self.restart_node_interactive()
            elif choice == "4":
                self.view_status_interactive()
            elif choice == "5":
                nodes = self.cli.list_nodes()
                self.display_nodes_table(nodes)
                self.pause()
            elif choice == "6":
                self.view_logs_interactive()
            elif choice == "7":
                self.config_management_interactive()
            elif choice == "8":
                self.advanced_options_interactive()
            elif choice == "9":
                self.wallet_mining_interactive()
            elif choice == "10":
                self.import_export_interactive()
            elif choice == "11":
                self.peer_management_interactive()
            elif choice == "12":
                self.ml_task_management_interactive()
            else:
                print("❌ Invalid choice. Please try again.")
                time.sleep(1)

    def wallet_mining_interactive(self):
        """Interactive wallet and mining management"""
        while True:
            print("\n💰 Wallet & Mining Management")
            print("-" * 40)
            print("1. 💳 Check Balance")
            print("2. 📍 Show Address") 
            print("3. ⚡ Mining Difficulty") 
            print("4. ⛏️  Mining History")
            print("5. 💸 Create Transaction")
            print("6. 📊 Mining Statistics")
            print("0. 🔙 Back to Main Menu")
            
            choice = input("\nEnter your choice (0-6): ").strip()
            
            if choice == "0":
                break
            elif choice == "1":
                self.check_balance_interactive()
            elif choice == "2":
                self.show_address_interactive()
            elif choice == "3":
                self.check_difficulty_interactive() 
            elif choice == "4":
                self.mining_history_interactive()
            elif choice == "5":
                self.create_transaction_interactive()
            elif choice == "6":
                self.mining_statistics_interactive()
            else:
                print("❌ Invalid choice. Please try again.")
            
            if choice != "0":
                self.pause()
    
    def check_balance_interactive(self):
        """Interactive balance check"""
        print("\n💳 Check Wallet Balance")
        print("-" * 30)
        
        node_id = self.get_input("Node ID to check balance for", "cpu-miner")
        
        try:
            # Load blockchain data from storage
            from pouw.blockchain.core import Blockchain
            from pouw.blockchain.storage import load_blocks
            
            blockchain = Blockchain()
            # Load blocks from database - this returns a list
            blocks_data = load_blocks()
            
            # Count mining rewards for this node
            miner_address = f"miner_{node_id}"
            total_coins = 0.0
            blocks_mined = 0
            mining_history = []
            
            # Process stored blocks (list format)
            for block_data in blocks_data:
                try:
                    # Check coinbase transaction (first transaction)
                    coinbase_tx_data = block_data['transactions'][0]
                    if coinbase_tx_data['outputs'][0]['address'] == miner_address:
                        reward = coinbase_tx_data['outputs'][0]['amount']
                        total_coins += reward
                        blocks_mined += 1
                        mining_history.append({
                            'block_hash': block_data.get('hash', 'unknown'),
                            'nonce': block_data['header']['nonce'],
                            'reward': reward,
                            'timestamp': block_data['header']['timestamp']
                        })
                except (KeyError, IndexError):
                    continue
            
            print(f"💰 Balance for node '{node_id}':")
            print(f"   Total Coins: {total_coins}")
            print(f"   Blocks Mined: {blocks_mined}")
            if blocks_mined > 0:
                avg_reward = total_coins / blocks_mined
                success_rate = (blocks_mined / len(blocks_data)) * 100
                print(f"   Average Reward: {avg_reward:.2f} coins/block")
                print(f"   Mining Success Rate: {success_rate:.1f}%")
            
            # Show recent mining activity
            if mining_history:
                print("\n📊 Recent Mining Activity (last 5 blocks):")
                # Sort by timestamp and get recent
                mining_history.sort(key=lambda x: x['timestamp'], reverse=True)
                recent = mining_history[:5]
                for activity in recent:
                    timestamp = datetime.fromtimestamp(
                        activity['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                    print(f"   Block #{activity['block_hash'][:16]}...: "
                          f"Nonce {activity['nonce']}, "
                          f"Reward {activity['reward']}, "
                          f"Time {timestamp}")
            
        except FileNotFoundError:
            print("❌ Blockchain database not found. Start a node first.")
        except Exception as e:
            print(f"❌ Error loading balance: {e}")

    def show_address_interactive(self):
        """Interactive address display"""
        print("\n📍 Show Wallet Address")
        print("-" * 30)
        
        node_id = self.get_input("Node ID to show address for", "cpu-miner")
        miner_address = f"miner_{node_id}"
        
        print(f"📍 Wallet Address Information:")
        print(f"   Node ID: {node_id}")
        print(f"   Address: {miner_address}")
        print(f"")
        print(f"💡 Usage:")
        print(f"   • To receive coins, share this address: {miner_address}")
        print(f"   • Others can send you coins with:")
        print(f"     ./pouw-cli send --to-address \"{miner_address}\" --amount X.X")
        print(f"   • Check your balance with:")
        print(f"     ./pouw-cli balance --node-id {node_id}")

    def check_difficulty_interactive(self):
        """Interactive difficulty check"""
        print("\n⚡ Mining Difficulty Check")
        print("-" * 35)
        
        node_id = self.get_input("Node ID to check difficulty for", "cpu-miner")
        
        try:
            from pouw.blockchain.core import Blockchain
            from pouw.blockchain.storage import load_blocks
            
            blockchain = Blockchain()
            blocks_list = load_blocks()
            
            difficulty_info = blockchain.get_current_difficulty_info()
            
            print(f"✅ Current Mining Difficulty:")
            print(f"   🎯 Target: {hex(difficulty_info['current_target'])}")
            print(f"   📊 Difficulty: {difficulty_info['difficulty_ratio']:.2f}x")
            print(f"   ⏱️  Target Block Time: {difficulty_info['target_block_time']} seconds")
            print(f"   📈 Recent Average: {difficulty_info['recent_avg_block_time']:.1f}s")
            print(f"   🔢 Total Blocks: {len(blocks_list)}")
            print(f"   🔄 Blocks Until Adjustment: {difficulty_info['blocks_until_adjustment']}")
            
        except Exception as e:
            print(f"❌ Error checking difficulty: {e}")
    
    def mining_history_interactive(self):
        """Interactive mining history view"""
        print("\n⛏️  Mining History")
        print("-" * 25)
        
        node_id = self.get_input("Node ID to check history for", "cpu-miner")
        limit = self.get_input("Number of recent blocks to show", "10")
        
        try:
            limit = int(limit)
            from pouw.blockchain.core import Blockchain
            from pouw.blockchain.storage import load_blocks
            
            blockchain = Blockchain()
            blocks_list = load_blocks()
            
            miner_address = f"miner_{node_id}"
            mined_blocks = []
            
            # Process stored blocks (list format)
            for block_data in blocks_list:
                try:
                    # Check coinbase transaction
                    coinbase_tx_data = block_data['transactions'][0]
                    if coinbase_tx_data['outputs'][0]['address'] == miner_address:
                        reward = coinbase_tx_data['outputs'][0]['amount']
                        mined_blocks.append({
                            'block_hash': block_data.get('hash', 'unknown'),
                            'nonce': block_data['header']['nonce'],
                            'reward': reward,
                            'timestamp': block_data['header']['timestamp']
                        })
                except (KeyError, IndexError):
                    continue
            
            # Sort by timestamp and show most recent
            mined_blocks.sort(key=lambda x: x['timestamp'], reverse=True)
            recent_blocks = mined_blocks[:limit]
            
            print(f"✅ Recent Mining Activity ({len(recent_blocks)} blocks):")
            print(f"{'Hash':<16} {'Nonce':<12} {'Reward':<8} {'Time'}")
            print("-" * 50)
            
            for block_info in recent_blocks:
                timestamp = datetime.fromtimestamp(
                    block_info['timestamp']).strftime('%H:%M:%S')
                hash_short = block_info['block_hash'][:14]
                print(f"{hash_short:<16} {block_info['nonce']:<12} "
                      f"{block_info['reward']:<8} {timestamp}")
            
        except Exception as e:
            print(f"❌ Error viewing mining history: {e}")

    def create_transaction_interactive(self):
        """Interactive transaction creation to send coins"""
        print("\n💸 Create Transaction")
        print("-" * 25)
        
        # Get transaction details from user
        from_node = self.get_input("From node ID", "cpu-miner")
        to_address = self.get_input("To address (recipient)", "")
        amount_str = self.get_input("Amount to send", "1.0")
        
        if not to_address:
            print("❌ Recipient address is required!")
            return
            
        try:
            amount = float(amount_str)
            if amount <= 0:
                print("❌ Amount must be positive!")
                return
        except ValueError:
            print("❌ Invalid amount format!")
            return
        
        try:
            from pouw.blockchain.core import Blockchain, Transaction
            from pouw.blockchain.storage import load_blocks
            from ecdsa import SigningKey, SECP256k1
            import secrets
            import binascii
            import json
            import hashlib
            
            # Initialize blockchain
            blockchain = Blockchain()
            
            # Find available UTXOs for the sender
            sender_address = f"miner_{from_node}"
            available_utxos = []
            utxo_value = 0.0
            
            print(f"\n🔍 Finding available UTXOs for {sender_address}...")
            
            # Search through UTXOs for ones belonging to sender
            for utxo_key, utxo_data in blockchain.utxos.items():
                if utxo_data.get('address') == sender_address:
                    prev_hash, index = utxo_key.split(':')
                    available_utxos.append({
                        'previous_hash': prev_hash,
                        'index': int(index),
                        'amount': utxo_data['amount']
                    })
                    utxo_value += utxo_data['amount']
            
            if not available_utxos:
                print(f"❌ No available UTXOs found for {sender_address}")
                print("   Try mining some blocks first to earn coins.")
                return
            
            if utxo_value < amount:
                print(f"❌ Insufficient balance! Available: {utxo_value}, Requested: {amount}")
                return
            
            print(f"✅ Found {len(available_utxos)} UTXOs with total value: {utxo_value}")
            
            # Select UTXOs to cover the amount (simple greedy selection)
            selected_utxos = []
            selected_value = 0.0
            
            for utxo in available_utxos:
                selected_utxos.append(utxo)
                selected_value += utxo['amount']
                if selected_value >= amount:
                    break
            
            # Calculate change
            change = selected_value - amount
            
            print(f"💰 Transaction Details:")
            print(f"   From: {sender_address}")
            print(f"   To: {to_address}")
            print(f"   Amount: {amount}")
            print(f"   Using {len(selected_utxos)} UTXOs worth: {selected_value}")
            if change > 0:
                print(f"   Change back to sender: {change}")
            
            # Create transaction outputs
            outputs = [
                {"address": to_address, "amount": amount}
            ]
            
            # Add change output if necessary
            if change > 0:
                outputs.append({"address": sender_address, "amount": change})
            
            # Generate a simple private key for signing (in production, this would be stored securely)
            # For demo purposes, we'll generate a deterministic key based on the node
            key_seed = f"pouw_node_{from_node}_private_key".encode()
            private_key_bytes = hashlib.sha256(key_seed).digest()
            signing_key = SigningKey.from_string(private_key_bytes, curve=SECP256k1)
            public_key = signing_key.verifying_key.to_string()
            
            # Add public key to inputs for verification
            inputs_with_pubkey = []
            for utxo in selected_utxos:
                inputs_with_pubkey.append({
                    'previous_hash': utxo['previous_hash'],
                    'index': utxo['index'],
                    'public_key': public_key.hex()
                })
            
            # Create transaction
            transaction = Transaction(
                version=1,
                inputs=inputs_with_pubkey,
                outputs=outputs
            )
            
            # Sign transaction
            tx_data = transaction.to_dict()
            tx_data.pop('signature', None)
            tx_string = json.dumps(tx_data, sort_keys=True)
            tx_hash = hashlib.sha256(tx_string.encode()).digest()
            
            signature = signing_key.sign(tx_hash)
            transaction.signature = signature
            
            # Validate transaction
            print(f"\n🔒 Validating transaction...")
            if blockchain._validate_transaction(transaction):
                print("✅ Transaction validation successful!")
                
                # Add to mempool
                if blockchain.add_transaction_to_mempool(transaction):
                    print("✅ Transaction added to mempool!")
                    print(f"🆔 Transaction ID: {transaction.get_hash()}")
                    print("\n📝 Transaction will be included in the next mined block.")
                else:
                    print("❌ Failed to add transaction to mempool!")
            else:
                print("❌ Transaction validation failed!")
                
        except ImportError:
            print("❌ ECDSA library not available. Install with: pip install ecdsa")
        except Exception as e:
            print(f"❌ Error creating transaction: {e}")
            import traceback
            traceback.print_exc()

    def mining_statistics_interactive(self):
        """Interactive mining statistics"""
        print("\n📊 Mining Statistics")
        print("-" * 25)
        
        node_id = self.get_input("Node ID to show statistics for", "cpu-miner")
        
        try:
            from pouw.blockchain.core import Blockchain
            from pouw.blockchain.storage import load_blocks
            
            blocks_list = load_blocks()
            miner_address = f"miner_{node_id}"
            
            # Collect mining statistics
            total_blocks = len(blocks_list)
            mined_count = 0
            total_rewards = 0.0
            nonce_stats = []
            time_stats = []
            
            for block_data in blocks_list:
                try:
                    coinbase_tx = block_data['transactions'][0]
                    if coinbase_tx['outputs'][0]['address'] == miner_address:
                        mined_count += 1
                        total_rewards += coinbase_tx['outputs'][0]['amount']
                        nonce_stats.append(block_data['header']['nonce'])
                        time_stats.append(block_data['header']['timestamp'])
                except (KeyError, IndexError):
                    continue
            
            print(f"📊 Mining Statistics for '{node_id}':")
            print(f"   Total Blocks in Chain: {total_blocks}")
            print(f"   Blocks Mined: {mined_count}")
            print(f"   Success Rate: {(mined_count/total_blocks)*100:.1f}%")
            print(f"   Total Rewards: {total_rewards} coins")
            
            if nonce_stats:
                avg_nonce = sum(nonce_stats) / len(nonce_stats)
                max_nonce = max(nonce_stats)
                min_nonce = min(nonce_stats)
                print(f"   Average Nonce: {avg_nonce:.0f}")
                print(f"   Max Nonce: {max_nonce}")
                print(f"   Min Nonce: {min_nonce}")
            
            if len(time_stats) > 1:
                intervals = []
                for i in range(1, len(time_stats)):
                    intervals.append(time_stats[i] - time_stats[i-1])
                if intervals:
                    avg_interval = sum(intervals) / len(intervals)
                    print(f"   Avg Mining Interval: {avg_interval:.1f}s")
            
        except Exception as e:
            print(f"❌ Error loading statistics: {e}")

    def import_export_interactive(self):
        """Interactive import/export management"""
        while True:
            print("\n📦 Import/Export Account Management")
            print("-" * 45)
            print("1. 📤 Export Account")
            print("2. 📥 Import Account")
            print("3. 📋 List Archives")
            print("4. 🧹 Clean Old Archives")
            print("0. 🔙 Back to Main Menu")
            
            choice = input("\nEnter your choice (0-4): ").strip()
            
            if choice == "0":
                break
            elif choice == "1":
                self.export_account_interactive()
            elif choice == "2":
                self.import_account_interactive()
            elif choice == "3":
                self.list_archives_interactive()
            elif choice == "4":
                self.clean_archives_interactive()
            else:
                print("❌ Invalid choice. Please try again.")
            
            if choice != "0":
                self.pause()

    def export_account_interactive(self):
        """Interactive account export"""
        print("\n📤 Export Account")
        print("-" * 20)
        
        node_id = self.get_input("Node ID to export", "cpu-miner")
        
        try:
            archive_path = self.export_account(node_id)
            archive_size = Path(archive_path).stat().st_size / (1024 * 1024)
            
            print(f"✅ Account exported successfully!")
            print(f"📦 Archive: {archive_path}")
            print(f"📏 Size: {archive_size:.2f} MB")
            print(f"💡 Transfer this file and import with:")
            print(f"   pouw-cli import {Path(archive_path).name}")
            
        except Exception as e:
            print(f"❌ Export failed: {e}")

    def import_account_interactive(self):
        """Interactive account import"""
        print("\n📥 Import Account")
        print("-" * 20)
        
        archive_path = self.get_input("Archive path (.tar.gz)")
        
        if not archive_path:
            print("❌ Archive path is required")
            return
        
        try:
            node_id = self.import_account(archive_path)
            print(f"✅ Account imported successfully!")
            print(f"🎯 Node ID: {node_id}")
            print(f"📚 Start with: pouw-cli start --node-id {node_id}")
            
        except Exception as e:
            print(f"❌ Import failed: {e}")

    def list_archives_interactive(self):
        """List available archive files"""
        print("\n📋 Available Archives")
        print("-" * 25)
        
        backup_dir = Path("account_backup")
        if not backup_dir.exists():
            print("📭 No backup directory found")
            return
        
        archives = list(backup_dir.glob("*.tar.gz"))
        if not archives:
            print("📭 No archive files found")
            return
        
        print(f"Found {len(archives)} archive files:")
        for archive in sorted(archives):
            size_mb = archive.stat().st_size / (1024 * 1024)
            mod_time = datetime.fromtimestamp(
                archive.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
            print(f"   📦 {archive.name} ({size_mb:.2f} MB, {mod_time})")

    def clean_archives_interactive(self):
        """Clean old archive files"""
        print("\n🧹 Clean Old Archives")
        print("-" * 25)
        
        backup_dir = Path("account_backup")
        if not backup_dir.exists():
            print("📭 No backup directory found")
            return
        
        archives = list(backup_dir.glob("*.tar.gz"))
        if not archives:
            print("📭 No archive files to clean")
            return
        
        days = self.get_input("Delete archives older than X days", "30")
        try:
            days = int(days)
        except ValueError:
            days = 30
        
        cutoff_time = time.time() - (days * 24 * 3600)
        old_archives = [a for a in archives if a.stat().st_mtime < cutoff_time]
        
        if not old_archives:
            print(f"📭 No archives older than {days} days found")
            return
        
        print(f"Found {len(old_archives)} old archives:")
        for archive in old_archives:
            mod_time = datetime.fromtimestamp(
                archive.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
            print(f"   📦 {archive.name} ({mod_time})")
        
        if self.confirm(f"Delete {len(old_archives)} old archives?"):
            for archive in old_archives:
                archive.unlink()
            print(f"✅ Deleted {len(old_archives)} old archives")

    def peer_management_interactive(self):
        """Interactive peer management"""
        while True:
            print("\n🌐 Peer Management")
            print("-" * 25)
            print("1. 📋 List Peers (by Node)")
            print("2. ➕ Add Peer (to Node)")
            print("3. ➖ Remove Peer (from Node)")
            print("4. 🔗 Connect to Peers (Node)")
            print("5. 📊 Peer Status (Node)")
            print("6. 🔗 Direct Connect (IP:Port)")
            print("7. 🔊 Show Listening Nodes")
            print("8. 🌐 Node Network Info")
            print("0. 🔙 Back to Main Menu")
            
            choice = input("\nEnter your choice (0-8): ").strip()
            
            if choice == "0":
                break
            elif choice == "1":
                self.list_peers_interactive()
            elif choice == "2":
                self.add_peer_interactive()
            elif choice == "3":
                self.remove_peer_interactive()
            elif choice == "4":
                self.connect_peers_interactive()
            elif choice == "5":
                self.peer_status_interactive()
            elif choice == "6":
                self.direct_connect_interactive()
            elif choice == "7":
                self.listen_nodes_interactive()
            elif choice == "8":
                self.node_info_interactive()
            else:
                print("❌ Invalid choice. Please try again.")
            
            if choice != "0":
                self.pause()

    def direct_connect_interactive(self):
        """Interactive direct peer connection using IP and port"""
        print("\n🔗 Direct Peer Connection")
        print("-" * 30)
        
        peer_address = self.get_input("Peer IP address or hostname", "localhost")
        peer_port = self.get_input("Peer port", "8333")
        
        try:
            peer_port = int(peer_port)
        except ValueError:
            print("❌ Invalid port number")
            return
        
        class MockArgs:
            def __init__(self, address, port):
                self.address = address
                self.port = port
        
        args = MockArgs(peer_address, peer_port)
        self.cli.cmd_connect(args)

    def listen_nodes_interactive(self):
        """Interactive listening nodes display"""
        print("\n🔊 Listening Nodes")
        print("-" * 20)
        
        class MockArgs:
            def __init__(self):
                self.port = None
        
        args = MockArgs()
        self.cli.cmd_listen(args)

    def node_info_interactive(self):
        """Interactive node network information"""
        print("\n🌐 Node Network Information")
        print("-" * 35)
        
        node_id = self.get_input("Node ID to show network info for", "cpu-miner")
        
        class MockArgs:
            def __init__(self, node_id):
                self.node_id = node_id
        
        args = MockArgs(node_id)
        self.cli.cmd_node_info(args)

    def list_peers_interactive(self):
        """Interactive peer listing"""
        print("\n📋 List Configured Peers")
        print("-" * 30)
        
        node_id = self.get_input("Node ID to list peers for", "cpu-miner")
        
        # Use the existing cmd_list_peers method
        class MockArgs:
            def __init__(self, node_id):
                self.node_id = node_id
        
        args = MockArgs(node_id)
        self.cli.cmd_list_peers(args)

    def add_peer_interactive(self):
        """Interactive peer addition"""
        print("\n➕ Add Peer to Bootstrap List")
        print("-" * 35)
        
        node_id = self.get_input("Node ID", "cpu-miner")
        peer_address = self.get_input("Peer address (IP or hostname)", "192.168.1.100")
        peer_port_str = self.get_input("Peer port", "8333")
        
        try:
            peer_port = int(peer_port_str)
            
            class MockArgs:
                def __init__(self, node_id, peer_address, peer_port):
                    self.node_id = node_id
                    self.peer_address = peer_address
                    self.peer_port = peer_port
            
            args = MockArgs(node_id, peer_address, peer_port)
            self.cli.cmd_add_peer(args)
            
        except ValueError:
            print("❌ Invalid port number")

    def remove_peer_interactive(self):
        """Interactive peer removal"""
        print("\n➖ Remove Peer from Bootstrap List")
        print("-" * 40)
        
        node_id = self.get_input("Node ID", "cpu-miner")
        
        # First show current peers
        print("\nCurrent peers:")
        try:
            config_file = Path(f"./configs/{node_id}.json")
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                bootstrap_peers = config.get('bootstrap_peers', [])
                if bootstrap_peers:
                    for i, peer in enumerate(bootstrap_peers, 1):
                        print(f"   {i}. {peer}")
                else:
                    print("   No peers configured")
                    return
            else:
                print(f"❌ Configuration file not found")
                return
        except Exception as e:
            print(f"❌ Error reading configuration: {e}")
            return
        
        peer_address = self.get_input("Peer address to remove", "")
        peer_port_str = self.get_input("Peer port to remove", "8333")
        
        if not peer_address:
            print("❌ Peer address is required")
            return
        
        try:
            peer_port = int(peer_port_str)
            
            class MockArgs:
                def __init__(self, node_id, peer_address, peer_port):
                    self.node_id = node_id
                    self.peer_address = peer_address
                    self.peer_port = peer_port
            
            args = MockArgs(node_id, peer_address, peer_port)
            self.cli.cmd_remove_peer(args)
            
        except ValueError:
            print("❌ Invalid port number")

    def connect_peers_interactive(self):
        """Interactive peer connection"""
        print("\n🔄 Connect to Peers")
        print("-" * 25)
        
        node_id = self.get_input("Node ID", "cpu-miner")
        
        print("Connect to:")
        print("1. All bootstrap peers")
        print("2. Specific peer")
        
        choice = input("Enter choice (1-2): ").strip()
        
        if choice == "1":
            # Connect to all bootstrap peers
            class MockArgs:
                def __init__(self, node_id):
                    self.node_id = node_id
            
            args = MockArgs(node_id)
            self.cli.cmd_connect_peers(args)
            
        elif choice == "2":
            # Connect to specific peer
            peer_address = self.get_input("Peer address", "192.168.1.100")
            peer_port_str = self.get_input("Peer port", "8333")
            
            try:
                peer_port = int(peer_port_str)
                
                class MockArgs:
                    def __init__(self, node_id, peer_address, peer_port):
                        self.node_id = node_id
                        self.peer_address = peer_address
                        self.peer_port = peer_port
                
                args = MockArgs(node_id, peer_address, peer_port)
                self.cli.cmd_connect_peers(args)
                
            except ValueError:
                print("❌ Invalid port number")
        else:
            print("❌ Invalid choice")

    def peer_status_interactive(self):
        """Interactive peer status check"""
        print("\n📊 Peer Connection Status")
        print("-" * 35)
        
        node_id = self.get_input("Node ID to check status for", "cpu-miner")
        
        class MockArgs:
            def __init__(self, node_id):
                self.node_id = node_id
        
        args = MockArgs(node_id)
        self.cli.cmd_peer_status(args)

    def ml_task_management_interactive(self):
        """Interactive ML task management"""
        while True:
            self.clear_screen()
            self.print_header()
            
            print("\n🧠 ML Task Management")
            print("=" * 40)
            print("  1. 📤 Submit New ML Task")
            print("  2. 📋 List Submitted Tasks")
            print("  3. 📊 Task Status & Results")
            print("  4. 📄 View Task Templates")
            print("  0. 🔙 Back to Main Menu")
            
            choice = input("\nEnter your choice (0-4): ").strip()
            
            if choice == "0":
                break
            elif choice == "1":
                self.submit_task_interactive()
            elif choice == "2":
                self.list_tasks_interactive()
            elif choice == "3":
                self.task_status_interactive()
            elif choice == "4":
                self.task_templates_interactive()
            else:
                print("❌ Invalid choice. Please try again.")
                self.pause()

    def submit_task_interactive(self):
        """Interactive ML task submission"""
        print("\n🧠 Submit ML Task")
        print("-" * 30)
        
        # Select node
        nodes = [n for n in self.cli.list_nodes() if n['status'] == 'running']
        if not nodes:
            print("❌ No running nodes available")
            print("Start a node first using the Start Node option")
            self.pause()
            return
        
        print("Available nodes:")
        for i, node in enumerate(nodes, 1):
            print(f"  {i}. {node['node_id']} (Type: {node['node_type']}, Port: {node['port']})")
        
        try:
            choice = int(input(f"Select node (1-{len(nodes)}): "))
            if not (1 <= choice <= len(nodes)):
                print("❌ Invalid choice")
                self.pause()
                return
            node_id = nodes[choice - 1]['node_id']
        except ValueError:
            print("❌ Please enter a valid number")
            self.pause()
            return
        
        # Task configuration
        print(f"\n📋 Task Configuration for {node_id}:")
        
        # Option to use existing task file or create new
        use_file = self.confirm("Do you have an existing task definition file?")
        task_file = None
        
        if use_file:
            task_file = self.get_input("Enter path to task file", "examples/mnist_task.json")
            if not Path(task_file).exists():
                print(f"❌ File not found: {task_file}")
                self.pause()
                return
        
        # Task parameters
        print("\n⚙️ Task Parameters:")
        fee = self.get_input("Task fee in PAI", "50.0")
        try:
            fee = float(fee)
        except ValueError:
            fee = 50.0
        
        if not use_file:
            epochs = self.get_input("Max training epochs", "20")
            try:
                epochs = int(epochs)
            except ValueError:
                epochs = 20
            
            batch_size = self.get_input("Batch size", "32")
            try:
                batch_size = int(batch_size)
            except ValueError:
                batch_size = 32
            
            dataset_size = self.get_input("Dataset size", "60000")
            try:
                dataset_size = int(dataset_size)
            except ValueError:
                dataset_size = 60000
            
            gpu = self.confirm("Require GPU acceleration?")
            min_accuracy = self.get_input("Minimum accuracy required", "0.8")
            try:
                min_accuracy = float(min_accuracy)
            except ValueError:
                min_accuracy = 0.8
        else:
            epochs = batch_size = dataset_size = None
            gpu = False
            min_accuracy = None
        
        # Confirmation
        print(f"\n📝 Task Summary:")
        print(f"  Node: {node_id}")
        print(f"  Fee: {fee} PAI")
        if task_file:
            print(f"  Task File: {task_file}")
        else:
            print(f"  Epochs: {epochs}")
            print(f"  Batch Size: {batch_size}")
            print(f"  Dataset Size: {dataset_size:,}")
            print(f"  GPU Required: {gpu}")
            print(f"  Min Accuracy: {min_accuracy}")
        
        if not self.confirm("Submit this ML task?"):
            print("❌ Task submission cancelled")
            self.pause()
            return
        
        # Create mock args and submit
        class MockArgs:
            def __init__(self):
                self.node_id = node_id
                self.task_file = task_file if use_file else None
                self.fee = fee
                self.epochs = epochs if not use_file else None
                self.batch_size = batch_size if not use_file else None
                self.dataset_size = dataset_size if not use_file else None
                self.gpu = gpu
                self.min_accuracy = min_accuracy if not use_file else None
        
        args = MockArgs()
        print(f"\n🚀 Submitting ML task...")
        self.cli.cmd_submit_task(args)
        self.pause()

    def list_tasks_interactive(self):
        """List submitted ML tasks"""
        print("\n📋 Submitted ML Tasks")
        print("-" * 30)
        
        # List data directories to find submitted tasks
        import os
        import glob
        
        task_files = []
        if os.path.exists("data"):
            task_files = glob.glob("data/*/submitted_task_*.json")
        
        if not task_files:
            print("📭 No submitted tasks found")
            self.pause()
            return
        
        print("📋 Found ML Tasks:")
        print("-" * 50)
        
        for task_file in task_files:
            try:
                with open(task_file, 'r') as f:
                    task_data = json.load(f)
                
                task_info = task_data['task']
                submission_time = task_data['submission_time']
                status = task_data.get('status', 'unknown')
                
                # Extract info
                node_id = task_file.split('/')[-2]  # Extract from path
                task_id = task_info['task_id']
                model_type = task_info['model_type']
                fee = task_info['fee']
                
                print(f"🔹 Task ID: {task_id}")
                print(f"   Node: {node_id}")
                print(f"   Model: {model_type.upper()}")
                print(f"   Fee: {fee} PAI")
                print(f"   Status: {status}")
                print(f"   Submitted: {submission_time}")
                print("-" * 30)
                
            except Exception as e:
                print(f"❌ Error reading {task_file}: {e}")
        
        self.pause()

    def task_status_interactive(self):
        """Show task status and results"""
        print("\n📊 Task Status & Results")
        print("-" * 30)
        print("🚧 Feature coming soon!")
        print("This will show real-time task progress, training metrics,")
        print("and final results when tasks complete.")
        self.pause()

    def task_templates_interactive(self):
        """Show available task templates"""
        print("\n📄 ML Task Templates")
        print("-" * 30)
        
        print("📋 Available Templates:")
        print("  1. 🧠 MNIST Handwritten Digit Recognition")
        print("     - Model: Multi-Layer Perceptron (MLP)")
        print("     - Dataset: MNIST (60,000 samples)")
        print("     - Task: Image classification (10 classes)")
        print("     - File: examples/mnist_task.json")
        print()
        print("  2. 🔍 Custom Template")
        print("     - Create your own task definition")
        print("     - Specify model architecture")
        print("     - Define training parameters")
        print()
        
        choice = self.get_choice("Select template to view details", 
                               ["MNIST Template", "Custom Template Guide", "Back"])
        
        if choice == "MNIST Template":
            print("\n📄 MNIST Task Template:")
            template_path = "examples/mnist_task.json"
            if Path(template_path).exists():
                try:
                    with open(template_path, 'r') as f:
                        template = json.load(f)
                    print(json.dumps(template, indent=2))
                except Exception as e:
                    print(f"❌ Error reading template: {e}")
            else:
                print("❌ Template file not found")
        
        elif choice == "Custom Template Guide":
            print("\n📋 Custom Task Template Guide:")
            print("""
Required fields in JSON task definition:
- model_type: "mlp", "cnn", "lstm", etc.
- architecture: Model layer specifications
- optimizer: Training optimizer settings
- stopping_criterion: When to stop training
- validation_strategy: How to validate model
- metrics: What to measure (accuracy, loss, etc.)
- dataset_info: Dataset format and parameters
- performance_requirements: GPU, accuracy minimums

Example structure:
{
  "model_type": "mlp",
  "architecture": {
    "input_size": 784,
    "hidden_sizes": [128, 64],
    "output_size": 10
  },
  "optimizer": {
    "type": "adam",
    "learning_rate": 0.001
  },
  // ... more fields
}
            """)
        
        self.pause()


class PoUWCLI:
    """Main PoUW CLI class for node management"""
    
    def __init__(self):
        self.configs_dir = Path("configs")
        self.logs_dir = Path("logs") 
        self.pids_dir = Path("pids")
        self.logger = logging.getLogger("PoUW-CLI")
        
        # Create necessary directories
        for directory in [self.configs_dir, self.logs_dir, self.pids_dir]:
            directory.mkdir(exist_ok=True)
    
    def setup_logging(self, verbose: bool = False):
        """Setup logging configuration"""
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def get_node_pid_file(self, node_id: str) -> Path:
        """Get path to node PID file"""
        return self.pids_dir / f"{node_id}.pid"
    
    def get_node_config_file(self, node_id: str) -> Path:
        """Get path to node configuration file"""
        return self.configs_dir / f"{node_id}.json"
    
    def get_node_log_file(self, node_id: str) -> Path:
        """Get path to node log file"""
        return self.logs_dir / f"{node_id}.log"
    
    def is_node_running(self, node_id: str) -> bool:
        """Check if a node is currently running"""
        pid_file = self.get_node_pid_file(node_id)
        if not pid_file.exists():
            return False
        
        try:
            with open(pid_file) as f:
                pid = int(f.read().strip())
            # Check if process exists
            return psutil.pid_exists(pid)
        except (ValueError, FileNotFoundError):
            return False
    
    def get_node_pid(self, node_id: str) -> Optional[int]:
        """Get PID of running node"""
        pid_file = self.get_node_pid_file(node_id)
        if not pid_file.exists():
            return None
        
        try:
            with open(pid_file) as f:
                pid = int(f.read().strip())
            if psutil.pid_exists(pid):
                return pid
        except (ValueError, FileNotFoundError):
            pass
        
        return None
    
    def save_node_pid(self, node_id: str, pid: int):
        """Save node PID to file"""
        pid_file = self.get_node_pid_file(node_id)
        with open(pid_file, 'w') as f:
            f.write(str(pid))
    
    def remove_node_pid(self, node_id: str):
        """Remove node PID file"""
        pid_file = self.get_node_pid_file(node_id)
        pid_file.unlink(missing_ok=True)
    
    def create_default_config(self, node_id: str, node_type: str, **kwargs) -> Dict[str, Any]:
        """Create default configuration for a node"""
        config = {
            "node_id": node_id,
            "node_type": node_type,
            "network": {
                "port": kwargs.get("port", 8333),
                "bootstrap_peers": kwargs.get("bootstrap_peers", []),
            },
            "mining": {
                "enabled": kwargs.get("mining", False),
                "gpu_enabled": kwargs.get("gpu", False),
                "threads": kwargs.get("threads", 1),
            },
            "training": {
                "enabled": kwargs.get("training", False),
                "batch_size": 32,
                "learning_rate": 0.001,
            },
            "logging": {
                "level": kwargs.get("log_level", "INFO"),
                "file": str(self.get_node_log_file(node_id)),
            },
        }
        return config
    
    def save_config(self, node_id: str, config: Dict[str, Any]):
        """Save node configuration"""
        config_file = self.get_node_config_file(node_id)
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_config(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Load node configuration"""
        config_file = self.get_node_config_file(node_id)
        if not config_file.exists():
            return None
        
        try:
            with open(config_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None
    
    def list_nodes(self) -> List[Dict[str, Any]]:
        """List all configured nodes"""
        nodes = []
        for config_file in self.configs_dir.glob("*.json"):
            node_id = config_file.stem
            config = self.load_config(node_id)
            if config:
                status = "running" if self.is_node_running(node_id) else "stopped"
                pid = self.get_node_pid(node_id)
                nodes.append({
                    "node_id": node_id,
                    "node_type": config.get("node_type", "unknown"),
                    "port": config.get("network", {}).get("port", 8333),
                    "status": status,
                    "pid": pid,
                })
        return nodes
    
    async def start_node(self, node_id: str, config_path: Optional[str] = None, 
                        daemon: bool = True, **kwargs):
        """Start a PoUW node"""
        if self.is_node_running(node_id):
            print(f"Node {node_id} is already running")
            return False
        
        # Load or create configuration
        if config_path:
            with open(config_path) as f:
                config = json.load(f)
        else:
            config = self.load_config(node_id)
            if not config:
                config = self.create_default_config(node_id, **kwargs)
                self.save_config(node_id, config)
        
        try:
            # Import and create node
            node_config = NodeConfiguration(**config)
            node = PoUWNode(node_config)
            
            if daemon:
                # Start in background
                import multiprocessing
                process = multiprocessing.Process(target=node.start)
                process.start()
                self.save_node_pid(node_id, process.pid)
                print(f"Node {node_id} started with PID {process.pid}")
            else:
                # Start in foreground
                await node.start()
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to start node {node_id}: {e}")
            return False
    
    def stop_node(self, node_id: str, force: bool = False) -> bool:
        """Stop a PoUW node"""
        pid = self.get_node_pid(node_id)
        if not pid:
            print(f"Node {node_id} is not running")
            return False
        
        try:
            process = psutil.Process(pid)
            if force:
                process.kill()
            else:
                process.terminate()
            
            # Wait for process to stop
            try:
                process.wait(timeout=10)
            except psutil.TimeoutExpired:
                if not force:
                    process.kill()
                    process.wait(timeout=5)
            
            self.remove_node_pid(node_id)
            print(f"Node {node_id} stopped")
            return True
        except psutil.NoSuchProcess:
            self.remove_node_pid(node_id)
            print(f"Node {node_id} was already stopped")
            return True
        except Exception as e:
            self.logger.error(f"Failed to stop node {node_id}: {e}")
            return False
    
    def restart_node(self, node_id: str, **kwargs) -> bool:
        """Restart a PoUW node (synchronous version)"""
        self.stop_node(node_id)
        time.sleep(2)  # Brief pause
        
        # Use create_task to run the async method in the current loop
        try:
            loop = asyncio.get_running_loop()
            # Create a task and run it synchronously
            task = loop.create_task(self.start_node(node_id, **kwargs))
            return True  # Return immediately - the task will run
        except RuntimeError:
            # No running loop, use asyncio.run
            return asyncio.run(self.start_node(node_id, **kwargs))
    
    async def restart_node_async(self, node_id: str, **kwargs) -> bool:
        """Restart a PoUW node (async version)"""
        self.stop_node(node_id)
        await asyncio.sleep(2)  # Brief pause
        await self.start_node(node_id, **kwargs)
        return True
    
    def get_node_status(self, node_id: str) -> Dict[str, Any]:
        """Get detailed status of a node"""
        status = {
            "node_id": node_id,
            "status": "stopped",
            "pid": None,
        }
        
        pid = self.get_node_pid(node_id)
        if pid:
            try:
                process = psutil.Process(pid)
                status.update({
                    "status": "running",
                    "pid": pid,
                    "cpu_percent": process.cpu_percent(),
                    "memory_info": process.memory_info()._asdict(),
                    "create_time": process.create_time(),
                })
            except psutil.NoSuchProcess:
                self.remove_node_pid(node_id)
        
        return status
    
    def show_logs(self, node_id: str, lines: int = 50, follow: bool = False):
        """Show node logs"""
        log_file = self.get_node_log_file(node_id)
        if not log_file.exists():
            print(f"No logs found for node {node_id}")
            return
        
        if follow:
            # Use tail -f equivalent
            subprocess.run(["tail", "-f", str(log_file)])
        else:
            # Show last N lines
            try:
                with open(log_file) as f:
                    log_lines = f.readlines()
                    for line in log_lines[-lines:]:
                        print(line.rstrip())
            except Exception as e:
                print(f"Error reading logs: {e}")
    
    def cmd_difficulty(self, args):
        """Show current mining difficulty and blockchain stats"""
        try:
            from pouw.blockchain.core import Blockchain
            from pouw.blockchain.storage import load_blocks
            
            blockchain = Blockchain()
            blocks_list = load_blocks()
            
            difficulty_info = blockchain.get_current_difficulty_info()
            
            print(f"⚡ Mining Difficulty Information:")
            print(f"   Current Target: {hex(difficulty_info['current_target'])}")
            print(f"   Difficulty Multiplier: {difficulty_info['difficulty_ratio']:.2f}x")
            print(f"   Target Block Time: {difficulty_info['target_block_time']} seconds")
            print(f"   Recent Average: {difficulty_info['recent_avg_block_time']:.1f}s")
            print(f"   Total Blocks: {len(blocks_list)}")
            print(f"   Blocks Until Adjustment: {difficulty_info['blocks_until_adjustment']}")
            
        except Exception as e:
            print(f"❌ Error loading difficulty: {e}")
    
    def cmd_balance(self, args):
        """Show wallet balance and mining earnings"""
        try:
            from pouw.blockchain.core import Blockchain
            from pouw.blockchain.storage import load_blocks
            
            # Load blockchain data from database
            blockchain = Blockchain()
            blocks_data = load_blocks()
            
            # Count mining rewards for this node
            node_id = args.node_id
            miner_address = f"miner_{node_id}"
            total_coins = 0.0
            blocks_mined = 0
            mining_history = []
            
            # Process stored blocks (list format)
            for block_data in blocks_data:
                try:
                    # Check coinbase transaction
                    coinbase_tx_data = block_data['transactions'][0]
                    if coinbase_tx_data['outputs'][0]['address'] == miner_address:
                        reward = coinbase_tx_data['outputs'][0]['amount']
                        total_coins += reward
                        blocks_mined += 1
                        mining_history.append({
                            'block_hash': block_data.get('hash', 'unknown'),
                            'nonce': block_data['header']['nonce'],
                            'reward': reward,
                            'timestamp': block_data['header']['timestamp']
                        })
                except (KeyError, IndexError):
                    continue
            
            print(f"💰 Balance for node '{node_id}':")
            print(f"   Total Coins: {total_coins}")
            print(f"   Blocks Mined: {blocks_mined}")
            if blocks_mined > 0:
                avg_reward = total_coins / blocks_mined
                success_rate = (blocks_mined / len(blocks_data)) * 100
                print(f"   Average Reward: {avg_reward:.2f} coins/block")
                print(f"   Mining Success Rate: {success_rate:.1f}%")
            
            # Show recent mining activity
            if mining_history:
                print(f"\n📊 Recent Mining Activity (last 5 blocks):")
                # Sort by timestamp and get recent
                mining_history.sort(key=lambda x: x['timestamp'], reverse=True)
                recent = mining_history[:5]
                for activity in recent:
                    timestamp = datetime.fromtimestamp(activity['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                    print(f"   Block: {activity['block_hash'][:16]}..., "
                          f"Nonce {activity['nonce']}, "
                          f"Reward {activity['reward']}, "
                          f"Time {timestamp}")
            
        except FileNotFoundError:
            print("❌ Blockchain database not found. Start a node first.")
        except Exception as e:
            print(f"❌ Error loading balance: {e}")

    def export_account(self, node_id: str) -> str:
        """Export account data to a compressed archive"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_name = f"pouw_account_{node_id}_{timestamp}.tar.gz"
        
        # Create backup directory if it doesn't exist
        backup_dir = Path("account_backup")
        backup_dir.mkdir(exist_ok=True)
        
        archive_path = backup_dir / archive_name
        
        # Create temporary directory for staging files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            account_dir = temp_path / "account_data"
            account_dir.mkdir()
            
            # Files to include in export
            files_to_export = [
                ("blockchain.db", "blockchain.db"),
                (f"configs/{node_id}.json", "config.json"),
                (f"data/{node_id}", "node_data"),
                ("keys", "keys"),
            ]
            
            exported_files = []
            for src_path, dest_name in files_to_export:
                src = Path(src_path)
                dest = account_dir / dest_name
                
                if src.exists():
                    if src.is_file():
                        shutil.copy2(src, dest)
                        exported_files.append(dest_name)
                    elif src.is_dir():
                        shutil.copytree(src, dest, dirs_exist_ok=True)
                        exported_files.append(f"{dest_name}/ (directory)")
            
            # Create account info file
            account_info = {
                "node_id": node_id,
                "export_timestamp": timestamp,
                "exported_files": exported_files,
                "pouw_version": "1.0.0",
                "export_type": "full_account"
            }
            
            with open(account_dir / "account_info.json", 'w') as f:
                json.dump(account_info, f, indent=2)
            
            # Create compressed archive
            with tarfile.open(archive_path, 'w:gz') as tar:
                tar.add(account_dir, arcname=".", recursive=True)
        
        return str(archive_path)
    
    def import_account(self, archive_path: str) -> str:
        """Import account data from a compressed archive"""
        archive_path = Path(archive_path)
        
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")
        
        # Create temporary directory for extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Extract archive
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(temp_path)
            
            # Read account info
            account_info_path = temp_path / "account_info.json"
            if account_info_path.exists():
                with open(account_info_path) as f:
                    account_info = json.load(f)
                node_id = account_info.get("node_id", "imported_node")
            else:
                node_id = f"imported_node_{int(time.time())}"
            
            # Import files
            import_mappings = [
                ("blockchain.db", "blockchain.db"),
                ("config.json", f"configs/{node_id}.json"),
                ("node_data", f"data/{node_id}"),
                ("keys", "keys"),
            ]
            
            for src_name, dest_path in import_mappings:
                src = temp_path / src_name
                dest = Path(dest_path)
                
                if src.exists():
                    # Create parent directories
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    
                    if src.is_file():
                        shutil.copy2(src, dest)
                    elif src.is_dir():
                        if dest.exists():
                            shutil.rmtree(dest)
                        shutil.copytree(src, dest)
        
        return node_id

    def cmd_send_transaction(self, args):
        """Send coins to another address"""
        try:
            from pouw.blockchain.core import Blockchain, Transaction
            from ecdsa import SigningKey, SECP256k1
            import json
            import hashlib
            
            # Initialize blockchain
            blockchain = Blockchain()
            
            # Find available UTXOs for the sender
            sender_address = f"miner_{args.from_node}"
            available_utxos = []
            utxo_value = 0.0
            
            print(f"🔍 Finding available UTXOs for {sender_address}...")
            
            # Search through UTXOs for ones belonging to sender
            for utxo_key, utxo_data in blockchain.utxos.items():
                if utxo_data.get('address') == sender_address:
                    prev_hash, index = utxo_key.split(':')
                    available_utxos.append({
                        'previous_hash': prev_hash,
                        'index': int(index),
                        'amount': utxo_data['amount']
                    })
                    utxo_value += utxo_data['amount']
            
            if not available_utxos:
                print(f"❌ No available UTXOs found for {sender_address}")
                print("   Try mining some blocks first to earn coins.")
                return
            
            if utxo_value < args.amount:
                print(f"❌ Insufficient balance! Available: {utxo_value}, Requested: {args.amount}")
                return
            
            print(f"✅ Found {len(available_utxos)} UTXOs with total value: {utxo_value}")
            
            # Select UTXOs to cover the amount (simple greedy selection)
            selected_utxos = []
            selected_value = 0.0
            
            for utxo in available_utxos:
                selected_utxos.append(utxo)
                selected_value += utxo['amount']
                if selected_value >= args.amount:
                    break
            
            # Calculate change
            change = selected_value - args.amount
            
            print(f"💰 Transaction Details:")
            print(f"   From: {sender_address}")
            print(f"   To: {args.to_address}")
            print(f"   Amount: {args.amount}")
            print(f"   Using {len(selected_utxos)} UTXOs worth: {selected_value}")
            if change > 0:
                print(f"   Change back to sender: {change}")
            
            # Create transaction outputs
            outputs = [
                {"address": args.to_address, "amount": args.amount}
            ]
            
            # Add change output if necessary
            if change > 0:
                outputs.append({"address": sender_address, "amount": change})
            
            # Generate a simple private key for signing
            key_seed = f"pouw_node_{args.from_node}_private_key".encode()
            private_key_bytes = hashlib.sha256(key_seed).digest()
            signing_key = SigningKey.from_string(private_key_bytes, curve=SECP256k1)
            public_key = signing_key.verifying_key.to_string()
            
            # Add public key to inputs for verification
            inputs_with_pubkey = []
            for utxo in selected_utxos:
                inputs_with_pubkey.append({
                    'previous_hash': utxo['previous_hash'],
                    'index': utxo['index'],
                    'public_key': public_key.hex()
                })
            
            # Create transaction
            transaction = Transaction(
                version=1,
                inputs=inputs_with_pubkey,
                outputs=outputs
            )
            
            # Sign transaction
            tx_data = transaction.to_dict()
            tx_data.pop('signature', None)
            tx_string = json.dumps(tx_data, sort_keys=True)
            tx_hash = hashlib.sha256(tx_string.encode()).digest()
            
            signature = signing_key.sign(tx_hash)
            transaction.signature = signature
            
            # Validate and add to mempool
            print(f"\n🔒 Validating transaction...")
            if blockchain._validate_transaction(transaction):
                print("✅ Transaction validation successful!")
                
                if blockchain.add_transaction_to_mempool(transaction):
                    print("✅ Transaction added to mempool!")
                    print(f"🆔 Transaction ID: {transaction.get_hash()}")
                    print("\n📝 Transaction will be included in the next mined block.")
                else:
                    print("❌ Failed to add transaction to mempool!")
            else:
                print("❌ Transaction validation failed!")
                
        except ImportError:
            print("❌ ECDSA library not available. Install with: pip install ecdsa")
        except Exception as e:
            print(f"❌ Error creating transaction: {e}")

    def cmd_address(self, args):
        """Show wallet address for a node"""
        node_id = args.node_id
        miner_address = f"miner_{node_id}"
        
        print(f"📍 Wallet Address Information:")
        print(f"   Node ID: {node_id}")
        print(f"   Address: {miner_address}")
        print(f"")
        print(f"💡 Usage:")
        print(f"   • To receive coins, share this address: {miner_address}")
        print(f"   • Others can send you coins with:")
        print(f"     ./pouw-cli send --to-address \"{miner_address}\" --amount X.X")
        print(f"   • Check your balance with:")
        print(f"     ./pouw-cli balance --node-id {node_id}")

    def cmd_add_peer(self, args):
        """Add a peer to the bootstrap peers list for a node"""
        node_id = args.node_id
        peer_address = args.peer_address
        peer_port = args.peer_port
        
        try:
            # Load node configuration
            config_file = Path(f"./configs/{node_id}.json")
            if not config_file.exists():
                print(f"❌ Configuration file not found for node {node_id}")
                print(f"   Expected: {config_file}")
                return
            
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Initialize bootstrap_peers if not present
            if 'bootstrap_peers' not in config:
                config['bootstrap_peers'] = []
            
            # Add peer if not already present
            peer_entry = f"{peer_address}:{peer_port}"
            if peer_entry not in config['bootstrap_peers']:
                config['bootstrap_peers'].append(peer_entry)
                
                # Save updated configuration
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=2)
                
                print(f"✅ Added peer {peer_entry} to {node_id}")
                print(f"   Total peers: {len(config['bootstrap_peers'])}")
                print(f"   Restart the node for changes to take effect")
            else:
                print(f"⚠️  Peer {peer_entry} already exists for {node_id}")
                
        except Exception as e:
            print(f"❌ Error adding peer: {e}")

    def cmd_remove_peer(self, args):
        """Remove a peer from the bootstrap peers list"""
        node_id = args.node_id
        peer_address = args.peer_address
        peer_port = args.peer_port
        
        try:
            # Load node configuration
            config_file = Path(f"./configs/{node_id}.json")
            if not config_file.exists():
                print(f"❌ Configuration file not found for node {node_id}")
                return
            
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            if 'bootstrap_peers' not in config:
                config['bootstrap_peers'] = []
            
            # Remove peer
            peer_entry = f"{peer_address}:{peer_port}"
            if peer_entry in config['bootstrap_peers']:
                config['bootstrap_peers'].remove(peer_entry)
                
                # Save updated configuration
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=2)
                
                print(f"✅ Removed peer {peer_entry} from {node_id}")
                print(f"   Total peers: {len(config['bootstrap_peers'])}")
            else:
                print(f"⚠️  Peer {peer_entry} not found for {node_id}")
                
        except Exception as e:
            print(f"❌ Error removing peer: {e}")

    def cmd_list_peers(self, args):
        """List configured and connected peers for a node"""
        node_id = args.node_id
        
        try:
            # Load node configuration to show bootstrap peers
            config_file = Path(f"./configs/{node_id}.json")
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                bootstrap_peers = config.get('bootstrap_peers', [])
                
                print(f"📋 Peer Information for Node: {node_id}")
                print("=" * 50)
                
                print(f"\n🔗 Bootstrap Peers ({len(bootstrap_peers)}):")
                if bootstrap_peers:
                    for i, peer in enumerate(bootstrap_peers, 1):
                        print(f"   {i}. {peer}")
                else:
                    print("   No bootstrap peers configured")
                
                print(f"\n⚙️  Node Configuration:")
                print(f"   • Listen Port: {config.get('listen_port', 'N/A')}")
                print(f"   • Max Peers: {config.get('max_peers', 'N/A')}")
                print(f"   • Peer Discovery: {config.get('peer_discovery_enabled', 'N/A')}")
                
            else:
                print(f"❌ Configuration file not found for node {node_id}")
                print(f"   Expected: {config_file}")
                
        except Exception as e:
            print(f"❌ Error listing peers: {e}")

    def cmd_connect_peers(self, args):
        """Connect to peers in real-time (requires running node)"""
        node_id = args.node_id
        peer_address = args.peer_address if hasattr(args, 'peer_address') else None
        peer_port = args.peer_port if hasattr(args, 'peer_port') else None
        
        try:
            # Check if node is running
            pid = self.get_node_pid(node_id)
            if not pid:
                print(f"❌ Node {node_id} is not running")
                print(f"   Start the node first: ./pouw-cli start --node-id {node_id}")
                return
            
            print(f"🔄 Attempting to connect node {node_id} to peers...")
            
            if peer_address and peer_port:
                # Connect to specific peer
                print(f"   Connecting to {peer_address}:{peer_port}...")
                # Note: This would require implementing a live connection API
                print(f"⚠️  Live peer connection API not implemented yet")
                print(f"   To connect peers:")
                print(f"   1. Add peer: ./pouw-cli add-peer --node-id {node_id} --peer-address {peer_address} --peer-port {peer_port}")
                print(f"   2. Restart node: ./pouw-cli restart --node-id {node_id}")
            else:
                # Connect to all bootstrap peers
                config_file = Path(f"./configs/{node_id}.json")
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    bootstrap_peers = config.get('bootstrap_peers', [])
                    if bootstrap_peers:
                        print(f"   Attempting to connect to {len(bootstrap_peers)} bootstrap peers...")
                        print(f"⚠️  Live peer connection API not implemented yet")
                        print(f"   Restart the node to connect to bootstrap peers")
                    else:
                        print(f"   No bootstrap peers configured")
                else:
                    print(f"❌ Configuration file not found")
                
        except Exception as e:
            print(f"❌ Error connecting to peers: {e}")

    def cmd_peer_status(self, args):
        """Show peer connection status for a running node"""
        node_id = args.node_id
        
        try:
            # Check if node is running
            pid = self.get_node_pid(node_id)
            if not pid:
                print(f"❌ Node {node_id} is not running")
                print(f"   Start the node first: ./pouw-cli start --node-id {node_id}")
                return
            
            print(f"📊 Peer Status for Node: {node_id}")
            print("=" * 40)
            print(f"   PID: {pid}")
            print(f"   Status: Running")
            
            # Load configuration for reference
            config_file = Path(f"./configs/{node_id}.json")
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                print(f"   Listen Port: {config.get('listen_port', 'N/A')}")
                print(f"   Max Peers: {config.get('max_peers', 'N/A')}")
                print(f"   Bootstrap Peers: {len(config.get('bootstrap_peers', []))}")
            
            print(f"\n⚠️  Live peer status monitoring not implemented yet")
            print(f"   Check log file: {self.get_node_log_file(node_id)}")
            
        except Exception as e:
            print(f"❌ Error checking peer status: {e}")

    def cmd_connect(self, args):
        """Connect to a peer using IP and port directly"""
        peer_address = args.address
        peer_port = args.port
        
        print(f"🔗 Testing connection to peer at {peer_address}:{peer_port}...")
        
        # Test basic TCP connection
        import socket
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((peer_address, peer_port))
            sock.close()
            
            if result == 0:
                print(f"✅ Port {peer_port} is open on {peer_address}")
                print(f"✅ TCP connection successful!")
                print(f"💡 This confirms the port is accessible.")
                print(f"   For full PoUW peer testing, start a node and check logs.")
                return True
            else:
                print(f"❌ Connection failed - port {peer_port} is not accessible")
                self._show_connection_troubleshooting(peer_address, peer_port)
                return False
                
        except socket.gaierror as e:
            print(f"❌ DNS resolution failed: {e}")
            self._show_connection_troubleshooting(peer_address, peer_port)
            return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            self._show_connection_troubleshooting(peer_address, peer_port)
            return False
    
    def _show_connection_troubleshooting(self, peer_address, peer_port):
        """Show troubleshooting tips for connection issues"""
        print(f"\n💡 Troubleshooting tips:")
        print(f"   • Check if peer node is running on {peer_address}:{peer_port}")
        print(f"   • Verify firewall settings allow connections to port {peer_port}")
        print(f"   • Ensure peer address is reachable from your network")
        print(f"   • Try: nc -zv {peer_address} {peer_port}")
        print(f"   • Or: telnet {peer_address} {peer_port}")

    def cmd_listen(self, args):
        """Show listening information for all nodes or start listening on a port"""
        port = args.port if hasattr(args, 'port') and args.port else None
        
        if port:
            # Start listening on specified port
            print(f"🔊 Starting listener on port {port}...")
            try:
                # This would start a simple peer listener
                print(f"⚠️  Standalone listener not implemented yet")
                print(f"   Use: ./pouw-cli start --node-id <name> --port {port}")
            except Exception as e:
                print(f"❌ Error starting listener: {e}")
        else:
            # Show listening information for all nodes
            print(f"🔊 Node Listening Information")
            print("=" * 50)
            
            nodes = self.list_nodes()
            if not nodes:
                print("No nodes configured")
                return
            
            print(f"{'Node ID':<20} {'Status':<10} {'Listen Port':<12} {'Address':<15}")
            print("-" * 70)
            
            for node in nodes:
                node_id = node['node_id']
                status = node['status']
                port = node['port']
                
                # Try to get local IP
                try:
                    import socket
                    hostname = socket.gethostname()
                    local_ip = socket.gethostbyname(hostname)
                except:
                    local_ip = "localhost"
                
                print(f"{node_id:<20} {status:<10} {port:<12} {local_ip}")
            
            print(f"\n💡 Connection examples:")
            print(f"   • From same machine: ./pouw-cli connect --address localhost --port 8333")
            print(f"   • From other device: ./pouw-cli connect --address {local_ip} --port 8333")

    def cmd_node_info(self, args):
        """Show detailed network information for a node"""
        nodes = self.list_nodes()
        
        # Find the requested node
        target_node = None
        for node in nodes:
            if node['node_id'] == args.node_id:
                target_node = node
                break
        
        if not target_node:
            print(f"❌ Node '{args.node_id}' not found")
            available_nodes = [node['node_id'] for node in nodes]
            if available_nodes:
                print(f"Available nodes: {', '.join(available_nodes)}")
            return
        
        print(f"\n🔍 Network Information for Node: {args.node_id}")
        print("=" * 60)
        
        # Show basic node info
        status_emoji = "🟢" if target_node['status'] == 'running' else "🔴"
        print(f"Status: {status_emoji} {target_node['status']}")
        if target_node['pid']:
            print(f"PID: {target_node['pid']}")
        
        # Network information
        print(f"\n🌐 Network Details:")
        
        # Get local IP
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
        except:
            local_ip = "localhost"
        
        # Get external IP
        external_ip = "unknown"
        try:
            # Try multiple external IP services
            services = [
                "https://api.ipify.org?format=json",
                "https://ipapi.co/json/",
                "https://httpbin.org/ip"
            ]
            
            for service_url in services:
                try:
                    with urllib.request.urlopen(service_url, timeout=5) as response:
                        data = json.loads(response.read().decode())
                        
                        # Different services return IP in different fields
                        if 'ip' in data:
                            external_ip = data['ip']
                            break
                        elif 'origin' in data:  # httpbin.org format
                            external_ip = data['origin']
                            break
                except:
                    continue
                    
            # Fallback: if all services fail, use local detection but mark it clearly
            if external_ip == "unknown":
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    s.connect(("8.8.8.8", 80))
                    external_ip = f"{s.getsockname()[0]} (local)"
                    
        except Exception as e:
            # Final fallback
            external_ip = "unavailable"
        
        # Get hostname
        hostname = socket.gethostname()
        
        print(f"  📍 Local IP: {local_ip}")
        print(f"  🌍 External IP: {external_ip}")
        print(f"  🏠 Hostname: {hostname}")
        
        # Show configured ports (from default config)
        default_ports = [8333, 8334, 8335, 8336, 8337]
        print(f"  🔌 Default Ports: {', '.join(map(str, default_ports))}")
        
        print(f"\n💡 Connection Examples:")
        print(f"  Connect to this node:")
        print(f"    python3 pouw-cli connect --address {local_ip} --port 8333")
        print(f"    python3 pouw-cli connect --address {hostname} --port 8333")
        if external_ip != "unknown" and external_ip != local_ip:
            print(f"    python3 pouw-cli connect --address {external_ip} --port 8333")

    def cmd_submit_task(self, args):
        """Submit an ML task to the PoUW network"""
        import json
        import uuid
        
        try:
            # Validate node exists and is running
            if not self.is_node_running(args.node_id):
                print(f"❌ Node '{args.node_id}' is not running")
                print("Start the node first using: python3 pouw-cli start --node-id <node-id>")
                return
            
            # Load task definition
            if hasattr(args, 'task_file') and args.task_file:
                try:
                    with open(args.task_file, 'r') as f:
                        task_definition = json.load(f)
                    print(f"📄 Loaded task definition from: {args.task_file}")
                except FileNotFoundError:
                    print(f"❌ Task file not found: {args.task_file}")
                    return
                except json.JSONDecodeError as e:
                    print(f"❌ Invalid JSON in task file: {e}")
                    return
            else:
                # Create a default MNIST task
                task_definition = {
                    "model_type": "mlp",
                    "architecture": {
                        "input_size": 784,
                        "hidden_sizes": [128, 64],
                        "output_size": 10
                    },
                    "optimizer": {
                        "type": "adam",
                        "learning_rate": 0.001,
                        "beta1": 0.9,
                        "beta2": 0.999
                    },
                    "stopping_criterion": {
                        "type": "max_epochs",
                        "max_epochs": getattr(args, 'epochs', 20),
                        "early_stopping": True,
                        "patience": 5
                    },
                    "validation_strategy": {
                        "type": "holdout",
                        "validation_split": 0.2
                    },
                    "metrics": ["accuracy", "loss"],
                    "dataset_info": {
                        "format": "MNIST",
                        "batch_size": getattr(args, 'batch_size', 32),
                        "training_percent": 0.8,
                        "size": getattr(args, 'dataset_size', 60000)
                    },
                    "performance_requirements": {
                        "gpu": getattr(args, 'gpu', False),
                        "min_accuracy": getattr(args, 'min_accuracy', 0.8),
                        "expected_training_time": 3600
                    }
                }
                print("📋 Using default MNIST task definition")
            
            # Generate unique task ID
            task_id = f"task_{uuid.uuid4().hex[:8]}"
            
            # Import required classes
            try:
                from pouw.blockchain.core import MLTask, PayForTaskTransaction
            except ImportError as e:
                print(f"❌ Failed to import PoUW components: {e}")
                print("Make sure PoUW is properly installed")
                return
            
            # Create MLTask object
            ml_task = MLTask(
                task_id=task_id,
                model_type=task_definition["model_type"],
                architecture=task_definition["architecture"],
                optimizer=task_definition["optimizer"],
                stopping_criterion=task_definition["stopping_criterion"],
                validation_strategy=task_definition["validation_strategy"],
                metrics=task_definition["metrics"],
                dataset_info=task_definition["dataset_info"],
                performance_requirements=task_definition["performance_requirements"],
                fee=args.fee,
                client_id=args.node_id
            )
            
            print(f"\n🧠 ML Task Details:")
            print(f"  Task ID: {task_id}")
            print(f"  Model: {task_definition['model_type'].upper()}")
            print(f"  Architecture: {task_definition['architecture']}")
            print(f"  Dataset: {task_definition['dataset_info']['format']}")
            print(f"  Dataset Size: {task_definition['dataset_info']['size']:,} samples")
            print(f"  Batch Size: {task_definition['dataset_info']['batch_size']}")
            print(f"  Max Epochs: {task_definition['stopping_criterion']['max_epochs']}")
            print(f"  GPU Required: {task_definition['performance_requirements']['gpu']}")
            print(f"  Fee: {args.fee} PAI")
            print(f"  Complexity Score: {ml_task.complexity_score:.2f}")
            print(f"  Required Miners: {ml_task.get_required_miners()}")
            
            # Create payment transaction
            pay_tx = PayForTaskTransaction(
                version=1,
                inputs=[],  # Simplified for CLI demo
                outputs=[{"address": args.node_id, "amount": -args.fee}],
                task_definition=ml_task.to_dict(),
                fee=args.fee
            )
            
            print(f"\n💳 Transaction Created:")
            print(f"  Type: PayForTaskTransaction")
            print(f"  Fee: {args.fee} PAI")
            print(f"  Transaction Hash: {pay_tx.get_hash()}")
            
            # For now, we'll save the task to a file since we don't have a full node implementation
            # In a full implementation, this would be submitted to the blockchain
            task_file_path = f"data/{args.node_id}/submitted_task_{task_id}.json"
            
            # Ensure directory exists
            import os
            os.makedirs(f"data/{args.node_id}", exist_ok=True)
            
            # Save task details
            task_submission = {
                "task": ml_task.to_dict(),
                "transaction": pay_tx.to_dict(),
                "submission_time": pay_tx.timestamp,
                "status": "submitted"
            }
            
            with open(task_file_path, 'w') as f:
                json.dump(task_submission, f, indent=2)
            
            print(f"\n✅ ML Task Successfully Submitted!")
            print(f"📁 Task saved to: {task_file_path}")
            print(f"🔍 Task ID: {task_id}")
            print(f"💰 Fee: {args.fee} PAI")
            
            # Show what happens next
            print(f"\n📋 Next Steps:")
            print(f"  1. Worker nodes will bid on this task")
            print(f"  2. Economic system will select optimal workers")
            print(f"  3. Training will begin on selected miners")
            print(f"  4. Results will be verified and rewards distributed")
            
            # Show monitoring command
            print(f"\n🔍 Monitor task status:")
            print(f"  python3 pouw-cli task-status --node-id {args.node_id} --task-id {task_id}")
            
        except Exception as e:
            print(f"❌ Error submitting ML task: {e}")
            import traceback
            print(f"Details: {traceback.format_exc()}")


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser"""
    parser = argparse.ArgumentParser(
        prog="pouw-cli",
        description="PoUW Blockchain Node Management CLI"
    )
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output")
    
    subparsers = parser.add_subparsers(dest="command", 
                                      help="Available commands")
    
    # Interactive command
    subparsers.add_parser("interactive", 
                         help="Enter interactive mode")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start a PoUW node")
    start_parser.add_argument("--node-id", required=True, 
                             help="Node identifier")
    start_parser.add_argument("--config", help="Configuration file path")
    start_parser.add_argument("--node-type", 
                             choices=["client", "miner", "supervisor", 
                                     "evaluator", "verifier", "peer"],
                             default="miner", help="Node type")
    start_parser.add_argument("--port", type=int, default=8333, 
                             help="Listen port")
    start_parser.add_argument("--mining", action="store_true", 
                             help="Enable mining")
    start_parser.add_argument("--training", action="store_true", 
                             help="Enable training")
    start_parser.add_argument("--gpu", action="store_true", 
                             help="Enable GPU")
    start_parser.add_argument("--daemon", action="store_true", default=True, 
                             help="Run as daemon")
    start_parser.add_argument("--foreground", action="store_true", 
                             help="Run in foreground")
    start_parser.add_argument("--bootstrap-peers", nargs="*", default=[], 
                             help="Bootstrap peers")
    start_parser.add_argument("--log-level", 
                             choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                             default="INFO", help="Log level")
    
    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop a PoUW node")
    stop_parser.add_argument("--node-id", required=True, 
                            help="Node identifier")
    stop_parser.add_argument("--force", action="store_true", 
                            help="Force kill the node")
    
    # Restart command
    restart_parser = subparsers.add_parser("restart", 
                                          help="Restart a PoUW node")
    restart_parser.add_argument("--node-id", required=True, 
                               help="Node identifier")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show node status")
    status_parser.add_argument("--node-id", 
                              help="Node identifier "
                              "(show all if not specified)")
    status_parser.add_argument("--json", action="store_true", 
                              help="Output as JSON")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all nodes")
    list_parser.add_argument("--json", action="store_true", 
                            help="Output as JSON")
    
    # Logs command
    logs_parser = subparsers.add_parser("logs", help="Show node logs")
    logs_parser.add_argument("--node-id", required=True, 
                            help="Node identifier")
    logs_parser.add_argument("--lines", "-n", type=int, default=50, 
                            help="Number of lines to show")
    logs_parser.add_argument("--follow", "-f", action="store_true", 
                            help="Follow logs")
    
    # Config command
    config_parser = subparsers.add_parser("config", 
                                         help="Configuration management")
    config_subparsers = config_parser.add_subparsers(dest="config_action")
    
    # Config create
    create_config_parser = config_subparsers.add_parser(
        "create", help="Create configuration")
    create_config_parser.add_argument("--node-id", required=True, 
                                     help="Node identifier")
    create_config_parser.add_argument(
        "--template", 
        choices=["client", "miner", "supervisor", "evaluator", "verifier", "peer"],
        default="miner", 
        help="Configuration template")
    create_config_parser.add_argument("--port", type=int, default=8333, 
                                     help="Listen port")
    create_config_parser.add_argument("--mining", action="store_true", 
                                     help="Enable mining")
    create_config_parser.add_argument("--training", action="store_true", 
                                     help="Enable training")
    create_config_parser.add_argument("--gpu", action="store_true", 
                                     help="Enable GPU")
    
    # Config show
    show_config_parser = config_subparsers.add_parser(
        "show", help="Show configuration")
    show_config_parser.add_argument("--node-id", required=True, 
                                   help="Node identifier")
    
    # Config edit
    edit_config_parser = config_subparsers.add_parser(
        "edit", help="Edit configuration")
    edit_config_parser.add_argument("--node-id", required=True, 
                                   help="Node identifier")
    
    # Difficulty command
    parser_difficulty = subparsers.add_parser(
        'difficulty', 
        help='Show current mining difficulty and blockchain stats'
    )
    parser_difficulty.add_argument(
        '--node-id', 
        help='Node ID to get difficulty from',
        default='cpu-miner'
    )
    
    # Balance command
    parser_balance = subparsers.add_parser(
        'balance', 
        help='Show wallet balance and mining earnings'
    )
    parser_balance.add_argument(
        '--node-id', 
        help='Node ID to get balance from',
        default='cpu-miner'
    )
    
    # Address command
    parser_address = subparsers.add_parser(
        'address', 
        help='Show wallet address for receiving coins'
    )
    parser_address.add_argument(
        '--node-id', 
        help='Node ID to get address for',
        default='cpu-miner'
    )
    
    # Export command
    parser_export = subparsers.add_parser(
        'export', 
        help='Export account data to archive'
    )
    parser_export.add_argument(
        '--node-id', 
        help='Node ID to export',
        default='cpu-miner'
    )
    parser_export.add_argument(
        '--output', 
        help='Output archive path (optional)'
    )
    
    # Import command
    parser_import = subparsers.add_parser(
        'import', 
        help='Import account data from archive'
    )
    parser_import.add_argument(
        'archive_path', 
        help='Path to account archive (.tar.gz)'
    )
    parser_import.add_argument(
        '--node-id', 
        help='Override node ID for imported account'
    )
    
    # Transaction commands
    send_parser = subparsers.add_parser("send", help="Send coins to another address")
    send_parser.add_argument("--from-node", default="cpu-miner", help="Node ID to send from")
    send_parser.add_argument("--to-address", required=True, help="Recipient address")
    send_parser.add_argument("--amount", type=float, required=True, help="Amount to send")
    send_parser.set_defaults(func="cmd_send_transaction")
    
    # Add-peer command
    add_peer_parser = subparsers.add_parser("add-peer", help="Add a peer to the bootstrap peers list")
    add_peer_parser.add_argument("--node-id", required=True, help="Node ID")
    add_peer_parser.add_argument("--peer-address", required=True, help="Peer address")
    add_peer_parser.add_argument("--peer-port", type=int, required=True, help="Peer port")
    add_peer_parser.set_defaults(func="cmd_add_peer")
    
    # Remove-peer command
    remove_peer_parser = subparsers.add_parser("remove-peer", help="Remove a peer from the bootstrap peers list")
    remove_peer_parser.add_argument("--node-id", required=True, help="Node ID")
    remove_peer_parser.add_argument("--peer-address", required=True, help="Peer address")
    remove_peer_parser.add_argument("--peer-port", type=int, required=True, help="Peer port")
    remove_peer_parser.set_defaults(func="cmd_remove_peer")
    
    # List-peers command
    list_peers_parser = subparsers.add_parser("list-peers", help="List configured and connected peers for a node")
    list_peers_parser.add_argument("--node-id", required=True, help="Node ID")
    list_peers_parser.set_defaults(func="cmd_list_peers")
    
    # Connect-peers command
    connect_peers_parser = subparsers.add_parser("connect-peers", help="Connect to peers in real-time (requires running node)")
    connect_peers_parser.add_argument("--node-id", required=True, help="Node ID")
    connect_peers_parser.add_argument("--peer-address", help="Peer address (optional)")
    connect_peers_parser.add_argument("--peer-port", type=int, help="Peer port (optional)")
    connect_peers_parser.set_defaults(func="cmd_connect_peers")
    
    # Peer-status command
    peer_status_parser = subparsers.add_parser("peer-status", help="Show peer connection status for a running node")
    peer_status_parser.add_argument("--node-id", required=True, help="Node ID")
    peer_status_parser.set_defaults(func="cmd_peer_status")
    
    # Connect command (IP & port only)
    connect_parser = subparsers.add_parser("connect", help="Connect to a peer using IP address and port")
    connect_parser.add_argument("--address", required=True, help="Peer IP address or hostname")
    connect_parser.add_argument("--port", type=int, required=True, help="Peer port number")
    connect_parser.set_defaults(func="cmd_connect")
    
    # Listen command
    listen_parser = subparsers.add_parser("listen", help="Show listening nodes or start listener on port")
    listen_parser.add_argument("--port", type=int, help="Port to listen on (optional)")
    listen_parser.set_defaults(func="cmd_listen")
    
    # Node-info command  
    node_info_parser = subparsers.add_parser("node-info", help="Show IP and port information for a node")
    node_info_parser.add_argument("--node-id", required=True, help="Node ID")
    node_info_parser.set_defaults(func="cmd_node_info")
    
    # Submit-task command
    submit_task_parser = subparsers.add_parser("submit-task", help="Submit an ML task to the PoUW network")
    submit_task_parser.add_argument("--node-id", required=True, help="Node ID to submit task from")
    submit_task_parser.add_argument("--task-file", help="Path to JSON task definition file")
    submit_task_parser.add_argument("--fee", type=float, default=50.0, help="Task fee in PAI coins (default: 50.0)")
    submit_task_parser.add_argument("--epochs", type=int, help="Max training epochs (default: 20)")
    submit_task_parser.add_argument("--batch-size", type=int, help="Training batch size (default: 32)")
    submit_task_parser.add_argument("--dataset-size", type=int, help="Dataset size (default: 60000)")
    submit_task_parser.add_argument("--gpu", action="store_true", help="Require GPU for training")
    submit_task_parser.add_argument("--min-accuracy", type=float, help="Minimum required accuracy (default: 0.8)")
    submit_task_parser.set_defaults(func="cmd_submit_task")
    
    return parser


async def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = PoUWCLI()
    cli.setup_logging(args.verbose)
    
    try:
        if args.command == "interactive":
            # Enter interactive mode
            interactive = InteractiveMode(cli)
            await interactive.run()
        
        elif args.command == "start":
            daemon = (not args.foreground if hasattr(args, 'foreground') 
                     else args.daemon)
            success = await cli.start_node(
                args.node_id,
                config_path=getattr(args, 'config', None),
                daemon=daemon,
                node_type=args.node_type,
                port=args.port,
                mining=args.mining,
                training=args.training,
                gpu=args.gpu,
                bootstrap_peers=args.bootstrap_peers,
                log_level=args.log_level,
            )
            sys.exit(0 if success else 1)
        
        elif args.command == "stop":
            success = cli.stop_node(args.node_id, force=args.force)
            sys.exit(0 if success else 1)
        
        elif args.command == "restart":
            success = cli.restart_node(args.node_id)
            sys.exit(0 if success else 1)
        
        elif args.command == "status":
            if args.node_id:
                status = cli.get_node_status(args.node_id)
                if args.json:
                    print(json.dumps(status, indent=2))
                else:
                    print(f"Node: {status['node_id']}")
                    print(f"Status: {status['status']}")
                    if status['pid']:
                        print(f"PID: {status['pid']}")
                        if 'cpu_percent' in status:
                            print(f"CPU: {status['cpu_percent']:.1f}%")
                            mem_mb = (status['memory_info']['rss'] / 
                                     1024 / 1024)
                            print(f"Memory: {mem_mb:.1f} MB")
            else:
                nodes = cli.list_nodes()
                if args.json:
                    print(json.dumps(nodes, indent=2))
                else:
                    if not nodes:
                        print("No nodes configured")
                    else:
                        print(f"{'Node ID':<20} {'Type':<12} {'Port':<8} "
                              f"{'Status':<10} {'PID':<8}")
                        print("-" * 65)
                        for node in nodes:
                            print(f"{node['node_id']:<20} "
                                  f"{node['node_type']:<12} "
                                  f"{node['port']:<8} {node['status']:<10} "
                                  f"{node['pid'] or '':<8}")
        
        elif args.command == "list":
            nodes = cli.list_nodes()
            if args.json:
                print(json.dumps(nodes, indent=2))
            else:
                if not nodes:
                    print("No nodes configured")
                else:
                    print(f"{'Node ID':<20} {'Type':<12} {'Port':<8} "
                          f"{'Status':<10} {'PID':<8}")
                    print("-" * 65)
                    for node in nodes:
                        print(f"{node['node_id']:<20} "
                              f"{node['node_type']:<12} "
                              f"{node['port']:<8} {node['status']:<10} "
                              f"{node['pid'] or '':<8}")
        
        elif args.command == "logs":
            cli.show_logs(args.node_id, lines=args.lines, follow=args.follow)
        
        elif args.command == "config":
            if args.config_action == "create":
                config = cli.create_default_config(
                    args.node_id,
                    args.template,
                    port=args.port,
                    mining=args.mining,
                    training=args.training,
                    gpu=args.gpu,
                )
                cli.save_config(args.node_id, config)
                print(f"Configuration created for node {args.node_id}")
            
            elif args.config_action == "show":
                config = cli.load_config(args.node_id)
                if config:
                    print(json.dumps(config, indent=2))
                else:
                    print(f"No configuration found for node {args.node_id}")
            
            elif args.config_action == "edit":
                config_file = cli.get_node_config_file(args.node_id)
                if config_file.exists():
                    editor = os.environ.get('EDITOR', 'nano')
                    subprocess.run([editor, str(config_file)])
                else:
                    print(f"No configuration found for node {args.node_id}")
        
        elif args.command == "difficulty":
            cli.cmd_difficulty(args)
        
        elif args.command == "balance":
            cli.cmd_balance(args)
        
        elif args.command == "address":
            cli.cmd_address(args)
        
        elif args.command == "export":
            try:
                if hasattr(args, 'output') and args.output:
                    # Custom output path specified
                    archive_path = cli.export_account(args.node_id)
                    custom_path = Path(args.output)
                    shutil.move(archive_path, custom_path)
                    archive_path = str(custom_path)
                else:
                    archive_path = cli.export_account(args.node_id)
                
                archive_size = Path(archive_path).stat().st_size / (1024 * 1024)
                print(f"✅ Account exported successfully!")
                print(f"📦 Archive: {archive_path}")
                print(f"📏 Size: {archive_size:.2f} MB")
                print(f"💡 Transfer this file to new device and use: pouw-cli import {Path(archive_path).name}")
            except Exception as e:
                print(f"❌ Export failed: {e}")
                sys.exit(1)
        
        elif args.command == "import":
            try:
                if hasattr(args, 'node_id') and args.node_id:
                    # Node ID override specified, modify archive
                    original_node_id = cli.import_account(args.archive_path)
                    # TODO: Implement node ID override logic
                    node_id = args.node_id
                    print(f"✅ Account imported with node ID: {node_id}")
                else:
                    node_id = cli.import_account(args.archive_path)
                    print(f"✅ Account imported successfully!")
                    print(f"🎯 Node ID: {node_id}")
                
                print(f"📚 Start the node with: pouw-cli start --node-id {node_id}")
                print(f"💰 Check balance with: pouw-cli balance --node-id {node_id}")
            except Exception as e:
                print(f"❌ Import failed: {e}")
                sys.exit(1)
        
        elif args.command == "send":
            cli.cmd_send_transaction(args)
        
        elif args.command == "add-peer":
            cli.cmd_add_peer(args)
        
        elif args.command == "remove-peer":
            cli.cmd_remove_peer(args)
        
        elif args.command == "list-peers":
            cli.cmd_list_peers(args)
        
        elif args.command == "connect-peers":
            cli.cmd_connect_peers(args)
        
        elif args.command == "peer-status":
            cli.cmd_peer_status(args)
        
        elif args.command == "connect":
            cli.cmd_connect(args)
        
        elif args.command == "listen":
            cli.cmd_listen(args)
        
        elif args.command == "node-info":
            cli.cmd_node_info(args)
        
        elif args.command == "submit-task":
            cli.cmd_submit_task(args)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled")
        sys.exit(1)
    except Exception as e:
        cli.logger.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 