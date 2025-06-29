#!/usr/bin/env python3
"""
Start a PoUW miner node.
"""

import asyncio
import argparse
import signal
import sys
from pouw.node import PoUWNode
from pouw.economics.staking import NodeRole


async def main():
    parser = argparse.ArgumentParser(description="Start a PoUW miner node")
    parser.add_argument("--node-id", required=True, help="Unique node identifier")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--stake", type=float, default=100.0, help="Stake amount")
    parser.add_argument(
        "--bootstrap-peer",
        action="append",
        nargs=2,
        metavar=("HOST", "PORT"),
        help="Bootstrap peer (can be repeated)",
    )

    args = parser.parse_args()

    # Create miner node
    node = PoUWNode(args.node_id, NodeRole.MINER, args.host, args.port)

    # Setup graceful shutdown
    def signal_handler(sig, frame):
        print(f"\nShutting down node {args.node_id}...")
        asyncio.create_task(node.stop())
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Start node
        await node.start()

        # Connect to bootstrap peers
        if args.bootstrap_peer:
            bootstrap_peers = [(host, int(port)) for host, port in args.bootstrap_peer]
            await node.connect_to_network(bootstrap_peers)

        # Stake and register
        preferences = {"model_types": ["mlp", "cnn"], "has_gpu": True, "max_dataset_size": 1000000}

        ticket = node.stake_and_register(args.stake, preferences)
        print(f"Staked {args.stake} PAI with ticket {ticket.ticket_id}")

        # Enable mining
        node.is_mining = True

        print(f"Miner node {args.node_id} running on {args.host}:{args.port}")
        print("Press Ctrl+C to stop")

        # Keep running
        while True:
            # Print status every 30 seconds
            await asyncio.sleep(30)
            status = node.get_status()
            print(
                f"Status: Height={status['blockchain_height']}, "
                f"Peers={status['peer_count']}, Training={status['is_training']}"
            )

    except KeyboardInterrupt:
        print(f"\nShutting down node {args.node_id}...")
        await node.stop()


if __name__ == "__main__":
    asyncio.run(main())
