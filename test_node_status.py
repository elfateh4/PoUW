#!/usr/bin/env python3
"""
Test script to demonstrate node status functionality
"""

import asyncio
import sys
import os
import time

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pouw.cli import PoUWCLI

async def test_node_status():
    """Test node status functionality"""
    print("🧪 Testing Node Status Functionality")
    print("=" * 50)
    
    cli = PoUWCLI()
    
    # Show current status
    print("\n📋 Current node status:")
    nodes = cli.list_nodes()
    for node in nodes:
        status_emoji = "🟢" if node['status'] == 'running' else "🔴"
        print(f"  {node['node_id']}: {status_emoji} {node['status']} (PID: {node['pid'] or 'None'})")
    
    # Check PID files
    print("\n📁 PID files in pids directory:")
    pids_dir = cli.pids_dir
    if pids_dir.exists():
        pid_files = list(pids_dir.glob("*.pid"))
        if pid_files:
            for pid_file in pid_files:
                try:
                    with open(pid_file) as f:
                        pid = f.read().strip()
                    print(f"  {pid_file.name}: PID {pid}")
                except Exception as e:
                    print(f"  {pid_file.name}: Error reading - {e}")
        else:
            print("  No PID files found")
    else:
        print("  Pids directory doesn't exist")
    
    # Test starting a node (this will fail but show the process)
    print("\n🚀 Testing node start process (will fail but show PID creation):")
    try:
        # Try to start a test node
        success = await cli.start_node("test-node", daemon=True, node_type="miner")
        if success:
            print("✅ Test node started successfully")
            time.sleep(2)  # Wait a moment
            
            # Check status again
            print("\n📋 Status after starting test node:")
            nodes = cli.list_nodes()
            for node in nodes:
                status_emoji = "🟢" if node['status'] == 'running' else "🔴"
                print(f"  {node['node_id']}: {status_emoji} {node['status']} (PID: {node['pid'] or 'None'})")
            
            # Stop the test node
            print("\n🛑 Stopping test node:")
            cli.stop_node("test-node")
            
        else:
            print("❌ Test node failed to start (expected)")
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
    
    print("\n💡 Explanation:")
    print("  • All nodes show as 'stopped' because no actual PoUW nodes are running")
    print("  • Only CLI interactive sessions are running (not PoUW nodes)")
    print("  • To see 'running' status, you need to start actual PoUW nodes")
    print("  • Use 'Start Node' option to start a node and see the status change")

if __name__ == "__main__":
    asyncio.run(test_node_status()) 