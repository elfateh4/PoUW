# PoUW Node Implementation Documentation

## Overview

The `PoUWNode` class is the main entry point for participating in the Proof of Useful Work (PoUW) blockchain network. It provides a unified interface that integrates all system components including blockchain consensus, machine learning computation, economic participation, networking, and security features.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Node Roles](#node-roles)
3. [Configuration](#configuration)
4. [API Reference](#api-reference)
5. [Usage Examples](#usage-examples)
6. [Component Integration](#component-integration)
7. [Error Handling](#error-handling)
8. [Performance Considerations](#performance-considerations)
9. [Security Features](#security-features)
10. [Troubleshooting](#troubleshooting)

## Architecture Overview

The PoUW node integrates multiple subsystems:

```
┌─────────────────┐
│   PoUWNode      │
├─────────────────┤
│ • Blockchain    │
│ • Mining        │
│ • ML Training   │
│ • Economics     │
│ • Networking    │
│ • Security      │
│ • Monitoring    │
└─────────────────┘
```

### Core Components

- **Blockchain**: Manages the distributed ledger and consensus
- **Mining**: Performs Proof of Useful Work mining using ML computation
- **ML Training**: Coordinates distributed machine learning tasks
- **Economics**: Handles staking, rewards, and economic incentives
- **Networking**: P2P communication and network operations
- **Security**: Attack detection and mitigation systems
- **Monitoring**: Performance tracking and health metrics

## Node Roles

The PoUW network supports multiple node roles:

### MINER

- Participates in PoUW mining using ML computation
- Trains ML models and attempts to mine blocks
- Requires significant computational resources

### SUPERVISOR

- Coordinates distributed training across multiple nodes
- Manages task distribution and aggregation
- Combines mining and coordination capabilities

### VERIFIER

- Validates mining proofs and ML work quality
- Ensures network integrity and security
- Lightweight computational requirements

### EVALUATOR

- Evaluates task completion and quality metrics
- Provides feedback on ML model performance
- Supports network quality assurance

### PEER

- Basic network participant
- Relays messages and maintains network connectivity
- Minimal computational requirements

## Configuration

### NodeConfig Class

```python
@dataclass
class NodeConfig:
    # Basic node configuration
    node_id: str
    role: NodeRole
    host: str = "localhost"
    port: int = 8000

    # Economic configuration
    initial_stake: float = 100.0
    preferences: Dict[str, Any] = field(default_factory=dict)

    # Mining configuration (for miners)
    omega_b: float = 1e-6  # Batch size coefficient
    omega_m: float = 1e-8  # Model size coefficient

    # Network configuration
    max_peers: int = 50
    bootstrap_peers: List[Tuple[str, int]] = field(default_factory=list)

    # Security configuration
    enable_security_monitoring: bool = True
    enable_attack_mitigation: bool = True

    # Advanced features
    enable_advanced_features: bool = True
    enable_production_features: bool = True
```

### Configuration Examples

#### Basic Miner Configuration

```python
config = NodeConfig(
    node_id="miner_001",
    role=NodeRole.MINER,
    host="0.0.0.0",
    port=8001,
    initial_stake=500.0,
    omega_b=1e-5,  # Higher batch size coefficient
    omega_m=1e-7,  # Higher model size coefficient
    bootstrap_peers=[("bootstrap.pouw.network", 8000)]
)
```

#### Supervisor Configuration

```python
config = NodeConfig(
    node_id="supervisor_001",
    role=NodeRole.SUPERVISOR,
    host="0.0.0.0",
    port=8002,
    initial_stake=1000.0,
    max_peers=100,  # More peers for coordination
    preferences={"task_types": ["image_classification", "nlp"]},
    enable_advanced_features=True
)
```

## API Reference

### Constructor

```python
PoUWNode(node_id: str, role: NodeRole, host: str = "localhost",
         port: int = 8000, config: Optional[NodeConfig] = None)
```

**Parameters:**

- `node_id`: Unique identifier for the node
- `role`: Node role from `NodeRole` enum
- `host`: Host address to bind to (default: "localhost")
- `port`: Port to bind to (default: 8000)
- `config`: Optional detailed configuration

### Core Methods

#### Node Lifecycle

```python
async def start() -> None
```

Starts the node and all its components.

```python
async def stop() -> None
```

Stops the node and cleans up resources.

#### Economic Participation

```python
def stake_and_register(stake_amount: float,
                      preferences: Optional[Dict[str, Any]] = None) -> Ticket
```

Stakes tokens and registers for network participation.

**Parameters:**

- `stake_amount`: Amount of PAI tokens to stake
- `preferences`: Optional node preferences for task assignment

**Returns:** Staking ticket for participation

#### Network Operations

```python
async def connect_to_peer(peer_host: str, peer_port: int) -> bool
```

Connects to a peer node.

```python
async def submit_task(task: MLTask, fee: float) -> bool
```

Submits an ML task to the network.

#### Mining Operations

```python
async def start_mining() -> None
```

Starts the mining process (for miner and supervisor nodes).

#### Status and Monitoring

```python
def get_status() -> Dict[str, Any]
```

Returns comprehensive node status including:

- Node identification and role
- Runtime status (running, mining, training)
- Blockchain metrics (height, mempool size)
- Network metrics (peer count)
- Performance statistics

```python
def get_health_metrics() -> NodeHealthMetrics
```

Returns detailed health metrics for monitoring.

```python
def get_economic_status() -> Dict[str, Any]
```

Returns economic participation status including staking and rewards.

## Usage Examples

### Basic Miner Setup

```python
import asyncio
from pouw.node import PoUWNode, NodeConfig
from pouw.economics import NodeRole

async def run_miner():
    # Create configuration
    config = NodeConfig(
        node_id="miner_001",
        role=NodeRole.MINER,
        host="0.0.0.0",
        port=8001,
        initial_stake=100.0,
        bootstrap_peers=[("127.0.0.1", 8000)]
    )

    # Create and start node
    node = PoUWNode(
        node_id="miner_001",
        role=NodeRole.MINER,
        host="0.0.0.0",
        port=8001,
        config=config
    )

    try:
        # Start the node
        await node.start()

        # Stake tokens and register
        ticket = node.stake_and_register(100.0, {"gpu_available": True})
        print(f"Staking ticket: {ticket.ticket_id}")

        # Start mining
        await node.start_mining()

        # Run for some time
        await asyncio.sleep(3600)  # Run for 1 hour

    finally:
        await node.stop()

# Run the miner
asyncio.run(run_miner())
```

### Supervisor Node Setup

```python
async def run_supervisor():
    config = NodeConfig(
        node_id="supervisor_001",
        role=NodeRole.SUPERVISOR,
        host="0.0.0.0",
        port=8002,
        initial_stake=500.0,
        max_peers=50,
        preferences={"coordination": True, "task_types": ["classification"]}
    )

    node = PoUWNode(
        node_id="supervisor_001",
        role=NodeRole.SUPERVISOR,
        config=config
    )

    await node.start()

    # Stake with supervisor preferences
    ticket = node.stake_and_register(500.0, {
        "coordination_capacity": 100,
        "preferred_algorithms": ["federated_learning"],
        "quality_threshold": 0.95
    })

    # Connect to other supervisors
    await node.connect_to_peer("supervisor_002.pouw.network", 8002)

    # Start mining and coordination
    await node.start_mining()

    # Monitor status
    while True:
        status = node.get_status()
        print(f"Supervisor status: {status}")
        await asyncio.sleep(60)

asyncio.run(run_supervisor())
```

### Task Submission Client

```python
from pouw.blockchain import MLTask

async def submit_training_task():
    # Create a client node
    client = PoUWNode("client_001", NodeRole.PEER)
    await client.start()

    # Create ML task
    task = MLTask(
        task_id="image_classification_001",
        task_type="image_classification",
        dataset_hash="0x123...",
        model_requirements={
            "architecture": "CNN",
            "min_accuracy": 0.9,
            "max_epochs": 100
        },
        reward_amount=50.0,
        deadline=int(time.time()) + 86400  # 24 hours
    )

    # Submit task with fee
    success = await client.submit_task(task, fee=5.0)
    if success:
        print(f"Task {task.task_id} submitted successfully")

    await client.stop()

asyncio.run(submit_training_task())
```

### Multi-Node Network Setup

```python
async def setup_test_network():
    """Setup a small test network with multiple node types"""

    # Bootstrap node (supervisor)
    bootstrap = PoUWNode(
        "bootstrap_001",
        NodeRole.SUPERVISOR,
        host="0.0.0.0",
        port=8000
    )

    # Miner nodes
    miners = []
    for i in range(3):
        config = NodeConfig(
            node_id=f"miner_{i:03d}",
            role=NodeRole.MINER,
            port=8001 + i,
            bootstrap_peers=[("127.0.0.1", 8000)]
        )
        miner = PoUWNode(f"miner_{i:03d}", NodeRole.MINER, port=8001 + i, config=config)
        miners.append(miner)

    # Verifier node
    verifier = PoUWNode(
        "verifier_001",
        NodeRole.VERIFIER,
        port=8010,
        config=NodeConfig(
            node_id="verifier_001",
            role=NodeRole.VERIFIER,
            port=8010,
            bootstrap_peers=[("127.0.0.1", 8000)]
        )
    )

    # Start all nodes
    await bootstrap.start()
    bootstrap.stake_and_register(1000.0)
    await bootstrap.start_mining()

    for miner in miners:
        await miner.start()
        miner.stake_and_register(100.0)
        await miner.start_mining()

    await verifier.start()
    verifier.stake_and_register(50.0)

    print("Test network started successfully")

    # Monitor network for a while
    for _ in range(60):  # 60 seconds
        bootstrap_status = bootstrap.get_status()
        print(f"Network height: {bootstrap_status['blockchain_height']}")
        await asyncio.sleep(1)

    # Cleanup
    await bootstrap.stop()
    for miner in miners:
        await miner.stop()
    await verifier.stop()

asyncio.run(setup_test_network())
```

## Component Integration

### Blockchain Integration

The node integrates with the blockchain through:

- **Transaction Processing**: Handles ML task transactions and staking
- **Block Mining**: Uses ML computation for proof-of-work
- **Consensus Participation**: Validates and propagates blocks

### ML Training Integration

ML components are integrated via:

- **Distributed Training**: Coordinates training across multiple nodes
- **Model Management**: Handles model updates and synchronization
- **Quality Assessment**: Evaluates training progress and model quality

### Economic System Integration

Economic participation includes:

- **Staking Mechanism**: Stake PAI tokens for network participation
- **Reward Distribution**: Earn rewards for successful mining and training
- **Reputation System**: Build reputation through quality contributions

### Network Operations

Network functionality provides:

- **P2P Communication**: Message passing between nodes
- **Peer Discovery**: Find and connect to other network participants
- **Network Health**: Monitor connection quality and network status

## Error Handling

The node implements comprehensive error handling:

### Initialization Errors

- Component initialization failures are logged and handled gracefully
- Missing optional components don't prevent node startup
- Configuration validation ensures proper setup

### Runtime Errors

- Network disconnections are handled with automatic reconnection
- Mining errors don't stop the node operation
- ML training errors are isolated and logged

### Security Errors

- Attack detection triggers appropriate mitigation measures
- Security alerts are logged and can trigger automated responses
- Byzantine fault tolerance handles malicious node behavior

## Performance Considerations

### Resource Management

- Mining intensity can be adjusted via omega coefficients
- Memory usage is monitored and controlled
- CPU usage is balanced across components

### Network Optimization

- Peer connections are managed efficiently
- Message broadcasting is optimized for network topology
- Bandwidth usage is monitored and controlled

### Scalability

- Node supports configurable peer limits
- Component initialization scales with available resources
- Monitoring overhead is minimized

## Security Features

### Attack Mitigation

- **Gradient Poisoning Detection**: Monitors for malicious ML updates
- **Byzantine Fault Tolerance**: Handles up to 1/3 malicious nodes
- **Network Security**: Detects and mitigates network-level attacks

### Data Protection

- **Secure Communication**: Encrypted peer-to-peer messaging
- **Data Integrity**: Cryptographic verification of ML data
- **Privacy Protection**: Differential privacy for sensitive data

### Access Control

- **Stake-based Participation**: Economic incentives for honest behavior
- **Role-based Permissions**: Different capabilities for different roles
- **Reputation System**: Track and reward honest participants

## Troubleshooting

### Common Issues

#### Node Won't Start

```
ERROR: Failed to initialize node components
```

**Solution**: Check that all required dependencies are installed and configuration is valid.

#### Mining Not Starting

```
WARNING: Node is not configured for mining
```

**Solution**: Ensure node role is MINER or SUPERVISOR and staking ticket is valid.

#### Peer Connection Failures

```
ERROR: Failed to connect to peer
```

**Solution**: Check network connectivity and ensure peer addresses are correct.

#### Security Alerts

```
WARNING: Security alert: gradient_poisoning_detected
```

**Solution**: Monitor for malicious peers and consider adjusting security thresholds.

### Logging Configuration

Enable detailed logging for debugging:

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create node with debug logging
node = PoUWNode("debug_node", NodeRole.MINER)
```

### Health Monitoring

Regularly check node health:

```python
async def monitor_node_health(node):
    while True:
        status = node.get_status()
        health = node.get_health_metrics()

        if not status['is_running']:
            print("WARNING: Node is not running")

        if status['security_alerts'] > 0:
            print(f"WARNING: {status['security_alerts']} security alerts")

        print(f"Node health: CPU {health.cpu_usage:.1f}%, Memory {health.memory_usage:.1f}%")

        await asyncio.sleep(30)  # Check every 30 seconds
```

### Performance Optimization

Monitor and optimize performance:

```python
def optimize_mining_parameters(node):
    """Adjust mining parameters based on performance"""
    status = node.get_status()

    if status['stats']['blocks_mined'] < 1:
        # Increase mining intensity
        node.config.omega_b *= 1.1
        node.config.omega_m *= 1.1
        print("Increased mining intensity")

    elif status['stats']['blocks_mined'] > 10:
        # Decrease mining intensity to save resources
        node.config.omega_b *= 0.9
        node.config.omega_m *= 0.9
        print("Decreased mining intensity")
```

## Advanced Usage

### Custom Security Policies

```python
def setup_custom_security(node):
    """Setup custom security policies"""
    if node.security_system:
        # Adjust security thresholds
        node.security_system['byzantine_tolerance'].threshold = 0.25
        node.security_system['gradient_detector'].sensitivity = 0.8

        # Custom alert handler
        async def custom_alert_handler(alert):
            if alert['type'] == 'gradient_poisoning':
                # Take immediate action
                await node.disconnect_peer(alert['source_peer'])
                print(f"Disconnected malicious peer: {alert['source_peer']}")

        node.security_system['custom_handler'] = custom_alert_handler
```

### Production Deployment

```python
async def production_deployment():
    """Production-ready node deployment"""

    # Production configuration
    config = NodeConfig(
        node_id="prod_miner_001",
        role=NodeRole.MINER,
        host="0.0.0.0",
        port=8001,
        initial_stake=1000.0,
        max_peers=100,
        enable_security_monitoring=True,
        enable_attack_mitigation=True,
        enable_production_features=True,
        bootstrap_peers=[
            ("node1.pouw.network", 8000),
            ("node2.pouw.network", 8000),
            ("node3.pouw.network", 8000)
        ]
    )

    node = PoUWNode("prod_miner_001", NodeRole.MINER, config=config)

    try:
        await node.start()
        ticket = node.stake_and_register(1000.0, {
            "gpu_memory": "32GB",
            "cpu_cores": 64,
            "bandwidth": "10Gbps",
            "uptime_guarantee": 0.99
        })

        await node.start_mining()

        # Production monitoring loop
        while True:
            status = node.get_status()
            health = node.get_health_metrics()

            # Log metrics to monitoring system
            print(f"Production metrics: {status}, {health}")

            # Check for issues
            if health.success_rate < 0.95:
                print("WARNING: Success rate below threshold")

            if status['security_alerts'] > 0:
                print("ALERT: Security issues detected")

            await asyncio.sleep(60)

    except Exception as e:
        print(f"Production error: {e}")
        # Implement production error handling

    finally:
        await node.stop()

asyncio.run(production_deployment())
```

This comprehensive documentation provides everything needed to understand, configure, and operate PoUW nodes in various scenarios from development to production deployment.
