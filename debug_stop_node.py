#!/usr/bin/env python3
"""
Debug script to test PoUW node stop functionality
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the pouw directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from pouw.cli import PoUWCLI

async def test_stop_node():
    """Test the stop node functionality"""
    print("🔍 Testing PoUW Node Stop Functionality")
    print("=" * 50)
    
    cli = PoUWCLI()
    
    # List all nodes
    print("\n📋 Current nodes:")
    nodes = cli.list_nodes()
    if not nodes:
        print("No nodes found")
        return
    
    for node in nodes:
        print(f"  - {node['node_id']} ({node['node_type']}) - {node['status']} (PID: {node['pid']})")
    
    # Check running nodes
    running_nodes = [n for n in nodes if n['status'] == 'running']
    if not running_nodes:
        print("\n❌ No running nodes to stop")
        return
    
    print(f"\n🟢 Found {len(running_nodes)} running nodes:")
    for node in running_nodes:
        print(f"  - {node['node_id']} (PID: {node['pid']})")
    
    # Test stopping each running node
    for node in running_nodes:
        node_id = node['node_id']
        pid = node['pid']
        
        print(f"\n🛑 Testing stop for node: {node_id}")
        print(f"   PID: {pid}")
        
        # Check if process actually exists
        import psutil
        try:
            process = psutil.Process(pid)
            print(f"   Process name: {process.name()}")
            print(f"   Process status: {process.status()}")
        except psutil.NoSuchProcess:
            print(f"   ❌ Process {pid} does not exist!")
            print(f"   Cleaning up stale PID file...")
            cli.remove_node_pid(node_id)
            continue
        
        # Try to stop the node
        print(f"   Attempting to stop...")
        success = cli.stop_node(node_id, force=False)
        
        if success:
            print(f"   ✅ Successfully stopped {node_id}")
        else:
            print(f"   ❌ Failed to stop {node_id}")
            
            # Try force stop
            print(f"   🔨 Trying force stop...")
            success = cli.stop_node(node_id, force=True)
            if success:
                print(f"   ✅ Force stopped {node_id}")
            else:
                print(f"   ❌ Force stop also failed for {node_id}")
    
    # Final status check
    print(f"\n📊 Final status:")
    final_nodes = cli.list_nodes()
    for node in final_nodes:
        print(f"  - {node['node_id']} - {node['status']} (PID: {node['pid']})")

if __name__ == "__main__":
    asyncio.run(test_stop_node()) 