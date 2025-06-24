# PoUW Node API Reference

## Module: `pouw.node`

### Classes

#### `NodeConfig`

Configuration dataclass for PoUW node initialization.

```python
@dataclass
class NodeConfig:
    node_id: str
    role: NodeRole
    host: str = "localhost"
    port: int = 8000
    initial_stake: float = 100.0
    preferences: Dict[str, Any] = field(default_factory=dict)
    omega_b: float = 1e-6
    omega_m: float = 1e-8
    max_peers: int = 50
    bootstrap_peers: List[Tuple[str, int]] = field(default_factory=list)
    enable_security_monitoring: bool = True
    enable_attack_mitigation: bool = True
    enable_advanced_features: bool = True
    enable_production_features: bool = True
```

**Fields:**

- `node_id`: Unique node identifier
- `role`: Node role from `NodeRole` enum
- `host`: Network host address
- `port`: Network port number
- `initial_stake`: Initial staking amount in PAI tokens
- `preferences`: Node-specific preferences for task assignment
- `omega_b`: Batch size coefficient for mining difficulty
- `omega_m`: Model size coefficient for mining difficulty
- `max_peers`: Maximum number of peer connections
- `bootstrap_peers`: List of bootstrap peer addresses
- `enable_security_monitoring`: Enable security monitoring features
- `enable_attack_mitigation`: Enable attack mitigation systems
- `enable_advanced_features`: Enable advanced cryptographic features
- `enable_production_features`: Enable production monitoring

#### `PoUWNode`

Main PoUW node implementation that integrates all system components.

```python
class PoUWNode:
    def __init__(self, node_id: str, role: NodeRole, host: str = "localhost",
                 port: int = 8000, config: Optional[NodeConfig] = None)
```

**Constructor Parameters:**

- `node_id`: Unique identifier for this node
- `role`: Node role (MINER, SUPERVISOR, VERIFIER, EVALUATOR, PEER)
- `host`: Host address to bind to
- `port`: Port to bind to
- `config`: Optional detailed configuration

**Attributes:**

##### Core Components

- `blockchain: Blockchain` - Blockchain instance
- `economic_system: EconomicSystem` - Economic system for staking and rewards
- `p2p_node: P2PNode` - P2P networking component
- `network_ops: NetworkOperationsManager` - Network operations manager

##### Role-Specific Components

- `miner: Optional[PoUWMiner]` - Mining component (for miners/supervisors)
- `verifier: Optional[PoUWVerifier]` - Verification component (for verifiers/supervisors/evaluators)
- `trainer: Optional[DistributedTrainer]` - ML training component (for miners/supervisors)

##### Security Components

- `security_system: Optional[Dict[str, Any]]` - Security monitoring system
- `attack_mitigation: Optional[AttackMitigationSystem]` - Attack mitigation system

##### Advanced Components

- `performance_monitor: Optional[PerformanceMonitor]` - Performance monitoring
- `dkg_participant: Optional[Any]` - Distributed key generation participant
- `advanced_consensus: Optional[Any]` - Advanced consensus mechanisms

##### State Variables

- `is_running: bool` - Node running status
- `is_mining: bool` - Mining status
- `is_training: bool` - Training status
- `current_task: Optional[MLTask]` - Current ML task
- `staking_ticket: Optional[Ticket]` - Current staking ticket
- `start_time: Optional[float]` - Node start timestamp
- `stats: Dict[str, Any]` - Performance statistics

### Methods

#### Lifecycle Management

##### `async def start() -> None`

Starts the PoUW node and all its components.

**Process:**

1. Starts network operations
2. Connects to bootstrap peers
3. Initializes performance monitoring
4. Sets running state

**Raises:**

- `Exception`: If node startup fails

##### `async def stop() -> None`

Stops the PoUW node and cleans up resources.

**Process:**

1. Stops mining and training
2. Stops network operations
3. Stops performance monitoring
4. Cleans up resources

#### Economic Participation

##### `def stake_and_register(stake_amount: float, preferences: Optional[Dict[str, Any]] = None) -> Ticket`

Stakes tokens and registers for network participation.

**Parameters:**

- `stake_amount`: Amount of PAI tokens to stake
- `preferences`: Optional node preferences for task assignment

**Returns:**

- `Ticket`: Staking ticket for network participation

**Process:**

1. Creates ticket through economic system
2. Stores ticket reference
3. Logs successful staking

**Raises:**

- `Exception`: If staking fails

#### Network Operations

##### `async def connect_to_peer(peer_host: str, peer_port: int) -> bool`

Connects to a peer node.

**Parameters:**

- `peer_host`: Peer hostname or IP address
- `peer_port`: Peer port number

**Returns:**

- `bool`: True if connection successful, False otherwise

##### `async def submit_task(task: MLTask, fee: float) -> bool`

Submits an ML task to the network.

**Parameters:**

- `task`: ML task to submit
- `fee`: Transaction fee in PAI tokens

**Returns:**

- `bool`: True if submission successful, False otherwise

**Process:**

1. Creates PayForTaskTransaction
2. Submits through economic system
3. Broadcasts transaction to network

#### Mining Operations

##### `async def start_mining() -> None`

Starts mining process for miner and supervisor nodes.

**Preconditions:**

- Node role must be MINER or SUPERVISOR
- Miner component must be initialized

**Process:**

1. Sets mining flag
2. Starts background mining loop
3. Logs mining start

##### `async def _mining_loop() -> None` (Private)

Main mining loop that runs continuously while mining is enabled.

**Process:**

1. Creates training batch with synthetic data
2. Sets up optimizer and loss function
3. Performs ML training iteration
4. Attempts to mine block using ML computation
5. Broadcasts successful blocks
6. Waits before next iteration

**Error Handling:**

- Catches and logs mining errors
- Continues operation with longer wait on errors

#### Status and Monitoring

##### `def get_status() -> Dict[str, Any]`

Returns comprehensive node status.

**Returns:**

```python
{
    'node_id': str,
    'role': str,
    'is_running': bool,
    'is_mining': bool,
    'is_training': bool,
    'blockchain_height': int,
    'mempool_size': int,
    'peer_count': int,
    'current_task': Optional[str],
    'uptime': float,
    'stats': Dict[str, Any],
    'staking_ticket': Optional[str],
    'network_role': str,
    'security_alerts': int
}
```

##### `def get_health_metrics() -> NodeHealthMetrics`

Returns detailed health metrics for monitoring.

**Returns:**

- `NodeHealthMetrics`: Comprehensive health information

##### `def get_economic_status() -> Dict[str, Any]`

Returns economic participation status.

**Returns:**

```python
{
    'staked': bool,
    'stake_amount': Optional[float],
    'role': str,
    'reputation': float,
    'rewards_earned': float
}
```

#### Message Handlers (Private)

##### `async def _handle_new_block(message: NetworkMessage) -> None`

Handles incoming new block messages.

##### `async def _handle_new_transaction(message: NetworkMessage) -> None`

Handles incoming new transaction messages.

##### `async def _handle_ml_iteration(message: NetworkMessage) -> None`

Handles ML training iteration messages from peers.

##### `async def _handle_task_submission(message: NetworkMessage) -> None`

Handles task submission messages.

##### `async def _handle_verification_request(message: NetworkMessage) -> None`

Handles verification request messages.

##### `async def _handle_security_alert(message: NetworkMessage) -> None`

Handles security alert messages.

#### Utility Methods (Private)

##### `def _initialize_components() -> None`

Initializes node components based on role and configuration.

##### `def _create_security_system() -> Dict[str, Any]`

Creates integrated security monitoring system.

##### `def _initialize_advanced_features() -> None`

Initializes advanced cryptographic features.

##### `def _initialize_production_features() -> None`

Initializes production monitoring and optimization.

##### `def _setup_message_handlers() -> None`

Sets up handlers for different message types.

##### `async def _broadcast_new_block(block) -> None`

Broadcasts newly mined block to network.

##### `async def _broadcast_transaction(transaction) -> None`

Broadcasts transaction to network.

### Dependencies

#### Core Dependencies

```python
# Blockchain components
from .blockchain import Blockchain, MLTask, PayForTaskTransaction, BuyTicketsTransaction
from .mining import PoUWMiner, PoUWVerifier, MiningProof
from .ml import DistributedTrainer, SimpleMLP, MiniBatch, IterationMessage
from .economics import EconomicSystem, NodeRole, Ticket
from .network import P2PNode, NetworkMessage, NetworkOperationsManager, NodeStatus, NodeHealthMetrics
from .security import AttackMitigationSystem, GradientPoisoningDetector, ByzantineFaultTolerance, SecurityAlert, AttackType
```

#### Optional Dependencies

```python
# Advanced features (optional)
from .advanced import AdvancedWorkerSelection, ZeroNonceCommitment

# Production features (optional)
from .production import PerformanceMonitor
```

### Error Handling

The node implementation includes comprehensive error handling:

#### Initialization Errors

- Component initialization failures are caught and logged
- Missing optional components don't prevent node startup
- Graceful degradation when advanced features are unavailable

#### Runtime Errors

- Network errors are handled with reconnection attempts
- Mining errors don't terminate node operation
- ML training errors are isolated and logged

#### Security Errors

- Attack detection triggers mitigation measures
- Security alerts are logged and propagated
- Byzantine faults are tolerated up to system limits

### Threading and Concurrency

The node uses asyncio for concurrent operations:

- **Main Event Loop**: Handles network I/O and coordination
- **Mining Loop**: Runs in background task during mining
- **Network Operations**: Asynchronous message handling
- **Component Integration**: Thread-safe component interaction

### Performance Characteristics

#### Memory Usage

- Configurable based on model size and batch size
- Efficient memory management for ML operations
- Monitoring and alerts for memory pressure

#### CPU Usage

- Adjustable mining intensity via omega coefficients
- Balanced between mining and network operations
- Performance monitoring and optimization

#### Network Usage

- Efficient P2P message propagation
- Configurable peer limits
- Bandwidth monitoring and control

### Security Considerations

#### Attack Resistance

- Gradient poisoning detection for ML components
- Byzantine fault tolerance for network consensus
- Economic incentives for honest behavior

#### Data Protection

- Secure communication protocols
- Cryptographic verification of ML data
- Privacy-preserving ML techniques

#### Access Control

- Stake-based participation requirements
- Role-based capability restrictions
- Reputation-based trust management

### Integration Points

#### Blockchain Integration

- Transaction creation and validation
- Block mining with ML computation
- Consensus participation

#### ML Framework Integration

- PyTorch integration for model training
- Distributed training coordination
- Model synchronization and aggregation

#### Economic System Integration

- PAI token staking and rewards
- Task assignment and completion
- Reputation and quality metrics

#### Network Protocol Integration

- P2P message passing
- Peer discovery and management
- Network health monitoring

### Example Usage Patterns

#### Basic Node Setup

```python
node = PoUWNode("node_001", NodeRole.MINER)
await node.start()
ticket = node.stake_and_register(100.0)
await node.start_mining()
```

#### Advanced Configuration

```python
config = NodeConfig(
    node_id="advanced_node",
    role=NodeRole.SUPERVISOR,
    omega_b=1e-5,
    enable_advanced_features=True,
    bootstrap_peers=[("bootstrap.example.com", 8000)]
)
node = PoUWNode("advanced_node", NodeRole.SUPERVISOR, config=config)
```

#### Production Deployment

```python
config = NodeConfig(
    node_id="prod_node",
    role=NodeRole.MINER,
    host="0.0.0.0",
    port=8001,
    max_peers=100,
    enable_production_features=True
)
node = PoUWNode("prod_node", NodeRole.MINER, config=config)
await node.start()
```

This API reference provides complete technical documentation for developers implementing and integrating with the PoUW node system.
