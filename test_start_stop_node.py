#!/usr/bin/env python3
"""
Test script to start and stop a PoUW node
"""

import asyncio
import sys
import time
from pathlib import Path

# Add the pouw directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from pouw.cli import PoUWCLI

async def test_start_stop_cycle():
    """Test the complete start/stop cycle"""
    print("🔄 Testing PoUW Node Start/Stop Cycle")
    print("=" * 50)
    
    cli = PoUWCLI()
    test_node_id = "test_node_001"
    
    # Clean up any existing test node
    print(f"\n🧹 Cleaning up existing test node: {test_node_id}")
    if cli.is_node_running(test_node_id):
        cli.stop_node(test_node_id, force=True)
    
    # Create and start a test node
    print(f"\n🚀 Starting test node: {test_node_id}")
    try:
        success = await cli.start_node(
            test_node_id,
            node_type="miner",
            port=8334,
            mining=False,
            training=False,
            daemon=True
        )
        
        if success:
            print(f"✅ Node {test_node_id} started successfully")
            
            # Wait a moment for the node to fully start
            print("⏳ Waiting 3 seconds for node to stabilize...")
            await asyncio.sleep(3)
            
            # Check if node is running
            if cli.is_node_running(test_node_id):
                print(f"✅ Node {test_node_id} is confirmed running")
                
                # Get node status
                status = cli.get_node_status(test_node_id)
                print(f"📊 Node status: {status}")
                
                # Now try to stop the node
                print(f"\n🛑 Stopping node: {test_node_id}")
                stop_success = cli.stop_node(test_node_id, force=False)
                
                if stop_success:
                    print(f"✅ Node {test_node_id} stopped successfully")
                    
                    # Wait a moment and check final status
                    await asyncio.sleep(1)
                    final_status = cli.get_node_status(test_node_id)
                    print(f"📊 Final status: {final_status}")
                    
                    if not cli.is_node_running(test_node_id):
                        print("✅ Node stop confirmed - not running")
                    else:
                        print("❌ Node is still running after stop")
                        
                else:
                    print(f"❌ Failed to stop node {test_node_id}")
                    
                    # Try force stop
                    print("🔨 Trying force stop...")
                    force_success = cli.stop_node(test_node_id, force=True)
                    if force_success:
                        print(f"✅ Force stopped {test_node_id}")
                    else:
                        print(f"❌ Force stop also failed")
            else:
                print(f"❌ Node {test_node_id} failed to start properly")
        else:
            print(f"❌ Failed to start node {test_node_id}")
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        print(f"\n🧹 Final cleanup...")
        if cli.is_node_running(test_node_id):
            cli.stop_node(test_node_id, force=True)

if __name__ == "__main__":
    asyncio.run(test_start_stop_cycle()) 