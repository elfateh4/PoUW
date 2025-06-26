#!/usr/bin/env python3

import asyncio
import argparse
import signal
import sys
from pouw.node import PoUWNode, NodeConfig
from pouw.economics import NodeRole


async def main():
    parser = argparse.ArgumentParser(description="Start a PoUW node")
    parser.add_argument("--node-id", default="my_node_001", help="Unique node identifier")
    parser.add_argument("--role", 
                        choices=["MINER", "SUPERVISOR", "VERIFIER", "EVALUATOR", "PEER"], 
                        default="MINER", 
                        help="Node role")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8666, help="Port to bind to")
    parser.add_argument("--stake", type=float, default=100.0, help="Stake amount")
    parser.add_argument("--bootstrap-peer", 
                        default="localhost:8000", 
                        help="Bootstrap peer (host:port)")

    args = parser.parse_args()

    # Parse role - convert uppercase to lowercase for NodeRole enum
    role_mapping = {
        "MINER": NodeRole.MINER,
        "SUPERVISOR": NodeRole.SUPERVISOR,
        "VERIFIER": NodeRole.VERIFIER,
        "EVALUATOR": NodeRole.EVALUATOR,
        "PEER": NodeRole.PEER
    }
    role = role_mapping[args.role]
    
    # Parse bootstrap peer
    if ":" in args.bootstrap_peer:
        peer_host, peer_port = args.bootstrap_peer.split(":")
        bootstrap_peers = [(peer_host, int(peer_port))]
    else:
        bootstrap_peers = []

    # Create node configuration
    config = NodeConfig(
        node_id=args.node_id,
        role=role,
        host=args.host,
        port=args.port,
        initial_stake=args.stake,
        bootstrap_peers=bootstrap_peers
    )

    # Create and start node
    print(f"🚀 Starting PoUW {role.value} node: {args.node_id}")
    print(f"   Host: {args.host}:{args.port}")
    print(f"   Stake: {args.stake} PAI")
    print(f"   Bootstrap: {args.bootstrap_peer}")
    print()

    node = PoUWNode(args.node_id, role, args.host, args.port)

    # Setup graceful shutdown
    def signal_handler(sig, frame):
        print("\n🛑 Received shutdown signal...")
        node.is_running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Start the node
        await node.start()
        print("✅ Node started successfully!")

        # Stake and register
        ticket = node.stake_and_register(args.stake)
        print(f"✅ Staked {args.stake} PAI tokens")

        # Start mining if it's a miner
        if role in [NodeRole.MINER, NodeRole.SUPERVISOR]:
            await node.start_mining()
            print("✅ Mining started!")

        # Keep running
        print("🔄 Node is running... Press Ctrl+C to stop")
        while node.is_running:
            # Print status every 30 seconds
            status = node.get_status()
            stats = status.get('stats', {})
            peer_count = status.get('peer_count', 0)
            blockchain_height = status.get('blockchain_height', 0)
            print(f"📊 Status: {stats.get('blocks_mined', 0)} blocks | {stats.get('tasks_completed', 0)} tasks | {peer_count} peers | Height: {blockchain_height}")
            await asyncio.sleep(30)

    except KeyboardInterrupt:
        print("\n🛑 Received keyboard interrupt...")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🔄 Stopping node...")
        await node.stop()
        print("✅ Node stopped successfully!")


if __name__ == "__main__":
    asyncio.run(main())
