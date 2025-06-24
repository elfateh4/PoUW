# PoUW Node - Quick Start Guide

A unified blockchain node implementation for the Proof of Useful Work (PoUW) network that integrates machine learning computation, economic participation, and distributed consensus.

## Features

🔗 **Blockchain Integration** - Complete blockchain node with consensus participation  
🤖 **ML-Powered Mining** - Proof of Useful Work using machine learning computation  
💰 **Economic Participation** - Staking, rewards, and reputation system  
🌐 **P2P Networking** - Advanced peer-to-peer communication and operations  
🔒 **Enterprise Security** - Attack detection, mitigation, and Byzantine fault tolerance  
📊 **Production Ready** - Performance monitoring, health metrics, and optimization

## Supported Node Roles

- **MINER** - Participates in PoUW mining using ML computation
- **SUPERVISOR** - Coordinates distributed training and network consensus
- **VERIFIER** - Validates mining proofs and ML work quality
- **EVALUATOR** - Evaluates task completion and quality metrics
- **PEER** - Basic network participant for message relay

## Quick Start

### 1. Basic Miner Setup

```python
import asyncio
from pouw.node import PoUWNode, NodeConfig
from pouw.economics import NodeRole

async def run_miner():
    # Create a miner node
    node = PoUWNode(
        node_id="miner_001",
        role=NodeRole.MINER,
        host="0.0.0.0",
        port=8001
    )

    # Start the node
    await node.start()

    # Stake tokens and start mining
    ticket = node.stake_and_register(100.0)
    await node.start_mining()

    # Monitor status
    while True:
        status = node.get_status()
        print(f"Blocks mined: {status['stats']['blocks_mined']}")
        await asyncio.sleep(60)

asyncio.run(run_miner())
```

### 2. Advanced Configuration

```python
from pouw.node import NodeConfig

# Create advanced configuration
config = NodeConfig(
    node_id="supervisor_001",
    role=NodeRole.SUPERVISOR,
    host="0.0.0.0",
    port=8002,
    initial_stake=500.0,
    omega_b=1e-5,  # Mining intensity
    omega_m=1e-7,  # Model complexity
    max_peers=100,
    bootstrap_peers=[("bootstrap.pouw.network", 8000)],
    preferences={"task_types": ["image_classification", "nlp"]}
)

node = PoUWNode(
    node_id="supervisor_001",
    role=NodeRole.SUPERVISOR,
    config=config
)
```

### 3. Task Submission

```python
from pouw.blockchain import MLTask

async def submit_task():
    client = PoUWNode("client_001", NodeRole.PEER)
    await client.start()

    # Create ML task
    task = MLTask(
        task_id="classification_001",
        task_type="image_classification",
        dataset_hash="0x123...",
        model_requirements={
            "architecture": "CNN",
            "min_accuracy": 0.9
        },
        reward_amount=50.0
    )

    # Submit with fee
    success = await client.submit_task(task, fee=5.0)
    print(f"Task submitted: {success}")
```

## Architecture

```
┌─────────────────────┐
│     PoUWNode        │
├─────────────────────┤
│ • Blockchain        │  ← Consensus & transactions
│ • Mining            │  ← PoUW with ML computation
│ • ML Training       │  ← Distributed learning
│ • Economics         │  ← Staking & rewards
│ • Networking        │  ← P2P communication
│ • Security          │  ← Attack mitigation
│ • Monitoring        │  ← Performance tracking
└─────────────────────┘
```

## Configuration Options

| Parameter         | Description                         | Default     |
| ----------------- | ----------------------------------- | ----------- |
| `node_id`         | Unique node identifier              | Required    |
| `role`            | Node role (MINER, SUPERVISOR, etc.) | Required    |
| `host`            | Network host address                | "localhost" |
| `port`            | Network port                        | 8000        |
| `initial_stake`   | Initial staking amount (PAI)        | 100.0       |
| `omega_b`         | Batch size coefficient              | 1e-6        |
| `omega_m`         | Model size coefficient              | 1e-8        |
| `max_peers`       | Maximum peer connections            | 50          |
| `bootstrap_peers` | Bootstrap peer addresses            | []          |

## API Overview

### Core Methods

```python
# Lifecycle
await node.start()                           # Start node
await node.stop()                            # Stop node

# Economic participation
ticket = node.stake_and_register(amount)     # Stake tokens
status = node.get_economic_status()          # Get economic status

# Mining operations
await node.start_mining()                    # Start mining (miners only)

# Network operations
await node.connect_to_peer(host, port)       # Connect to peer
await node.submit_task(task, fee)            # Submit ML task

# Monitoring
status = node.get_status()                   # Get node status
health = node.get_health_metrics()           # Get health metrics
```

## Monitoring and Status

### Node Status

```python
status = node.get_status()
print(f"Role: {status['role']}")
print(f"Running: {status['is_running']}")
print(f"Mining: {status['is_mining']}")
print(f"Blockchain height: {status['blockchain_height']}")
print(f"Peer count: {status['peer_count']}")
print(f"Blocks mined: {status['stats']['blocks_mined']}")
```

### Health Metrics

```python
health = node.get_health_metrics()
print(f"Response time: {health.response_time}ms")
print(f"Success rate: {health.success_rate:.2%}")
print(f"CPU usage: {health.cpu_usage:.1f}%")
```

## Security Features

- **Gradient Poisoning Detection** - Monitors for malicious ML updates
- **Byzantine Fault Tolerance** - Handles up to 1/3 malicious nodes
- **Attack Mitigation** - Automated response to detected threats
- **Secure Communication** - Encrypted peer-to-peer messaging
- **Economic Security** - Stake-based incentives for honest behavior

## Production Deployment

### Docker Setup

```dockerfile
FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "-m", "pouw.node"]
```

### Environment Variables

```bash
export POUW_NODE_ID="prod_miner_001"
export POUW_ROLE="MINER"
export POUW_HOST="0.0.0.0"
export POUW_PORT="8001"
export POUW_STAKE_AMOUNT="1000"
export POUW_BOOTSTRAP_PEERS="node1.pouw.network:8000,node2.pouw.network:8000"
```

### Systemd Service

```ini
[Unit]
Description=PoUW Node
After=network.target

[Service]
Type=simple
User=pouw
WorkingDirectory=/opt/pouw
ExecStart=/usr/bin/python3 -m pouw.node
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

## Performance Tuning

### Mining Optimization

```python
# Increase mining intensity for powerful hardware
config.omega_b = 1e-4  # Larger batches
config.omega_m = 1e-6  # Larger models

# Decrease for lower-end hardware
config.omega_b = 1e-7  # Smaller batches
config.omega_m = 1e-9  # Smaller models
```

### Network Optimization

```python
# High-performance networking
config.max_peers = 200
config.enable_production_features = True

# Resource-constrained environments
config.max_peers = 20
config.enable_advanced_features = False
```

## Troubleshooting

### Common Issues

**Node won't start**

```
Check logs for component initialization errors
Verify configuration parameters
Ensure dependencies are installed
```

**Mining not working**

```
Verify node role is MINER or SUPERVISOR
Check staking ticket is valid
Monitor for security alerts
```

**Network connectivity issues**

```
Check firewall settings
Verify bootstrap peer addresses
Monitor peer connection status
```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

node = PoUWNode("debug_node", NodeRole.MINER)
# Detailed logs will be available
```

## Documentation

- **[Complete Documentation](docs/node_documentation.md)** - Comprehensive usage guide
- **[API Reference](docs/node_api_reference.md)** - Technical API documentation
- **[Architecture Guide](docs/architecture.md)** - System design and components

## Support

- **Issues**: Report bugs and feature requests on GitHub
- **Discord**: Join the PoUW community discord
- **Email**: Contact support@pouw.network

## License

This project is licensed under the MIT License - see the LICENSE file for details.
