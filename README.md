# PoUW: Proof of Useful Work Blockchain

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-comprehensive-brightgreen.svg)](tests/)
[![CLI](https://img.shields.io/badge/CLI-interactive-green.svg)](pouw-cli)

A revolutionary blockchain implementation that combines **distributed machine learning** with **cryptocurrency mining**, creating a system where computational work contributes to both network security and artificial intelligence advancement.

## Key Features

- **ML-Powered Mining**: Mine blocks using machine learning computation instead of wasteful hash calculations
- **Distributed AI Training**: Coordinate federated learning across a decentralized network
- **Economic Incentives**: Stake-based participation with rewards for quality contributions
- **Enterprise Security**: Advanced attack detection, Byzantine fault tolerance, and gradient poisoning protection
- **High Performance**: Optimized for production deployment with GPU acceleration and monitoring
- **Interactive CLI**: Comprehensive command-line interface for easy node management
- **Multi-Role Support**: Worker, supervisor, miner, and hybrid nodes working together

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             PoUW Network                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │ Blockchain Layer│  │   ML Layer      │  │ Economic Layer  │             │
│  │ • Consensus     │  │ • Fed Learning  │  │ • Staking       │             │
│  │ • Transactions  │  │ • Model Sync    │  │ • Rewards       │             │
│  │ • Block Mining  │  │ • Quality Eval  │  │ • Reputation    │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │ Network Layer   │  │ Security Layer  │  │ Production      │             │
│  │ • P2P Protocol  │  │ • Attack Det.   │  │ • Monitoring    │             │
│  │ • VPN Mesh      │  │ • BFT           │  │ • GPU Support   │             │
│  │ • Load Balance  │  │ • Encryption    │  │ • Analytics     │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │ Data Management │  │ Advanced Crypto │  │ Infrastructure  │             │
│  │ • Sharding      │  │ • BLS Threshold │  │ • Kubernetes    │             │
│  │ • Reed-Solomon  │  │ • VRF           │  │ • CI/CD         │             │
│  │ • Consistency   │  │ • DKG           │  │ • Docker        │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- **Python 3.9+** (Python 3.11+ recommended)
- **PyTorch** (for ML computation)
- **4GB+ RAM** (16GB+ recommended for mining)
- **Multi-core CPU** (GPU optional but recommended)

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/your-org/PoUW.git
cd PoUW
```

#### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate     # On Windows
```

#### 3. Install Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# Verify installation
python -c "import pouw; print('PoUW installed successfully!')"
```

#### 4. Make CLI Executable

```bash
# Make the CLI script executable
chmod +x pouw-cli

# Test CLI
./pouw-cli --help
```

## Interactive CLI Usage

The easiest way to get started is using the interactive CLI:

```bash
# Start interactive mode
./pouw-cli interactive

# Or use specific commands
./pouw-cli start --node-id my-worker --node-type worker
./pouw-cli status --node-id my-worker
./pouw-cli logs --node-id my-worker
```

### CLI Features

- **Node Management**: Start, stop, restart nodes
- **Monitoring**: Real-time status, logs, metrics
- **Wallet Operations**: Balance checking, transactions
- **Peer Management**: Connect to network, manage peers
- **ML Tasks**: Submit and manage machine learning tasks
- **Configuration**: Create, edit, manage node configs
- **Account Management**: Import/export node accounts

## Usage Examples

### Start Your First Node

#### Using Interactive CLI (Recommended)

```bash
./pouw-cli interactive
# Select "1. Start Node"
# Follow the setup wizard
```

#### Using Direct Commands

```bash
# Create a worker node
./pouw-cli start --node-id worker-1 --node-type worker --port 8333

# Create a miner node
./pouw-cli start --node-id miner-1 --node-type miner --port 8334 --enable-mining

# Check status
./pouw-cli status --node-id worker-1
```

### Python API Usage

```python
import asyncio
from pouw.node import PoUWNode, NodeConfiguration, NodeType

async def start_worker_node():
    # Create configuration
    config = NodeConfiguration(
        node_id="my-worker",
        node_type=NodeType.WORKER,
        listen_port=8333,
        training_enabled=True,
        gpu_enabled=True  # Enable if you have GPU
    )
    
    # Create and start node
    node = PoUWNode(config)
    await node.initialize()
    await node.start()
    
    print(f" Worker node {config.node_id} started!")
    
    # Keep running
    try:
        while True:
            status = node.get_status()
            print(f"Node status: {status['state']}")
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        await node.shutdown()

# Run the node
asyncio.run(start_worker_node())
```

### Multi-Node Network Setup

```python
async def setup_network():
    # Start supervisor node
    supervisor_config = NodeConfiguration(
        node_id="supervisor-1",
        node_type=NodeType.SUPERVISOR,
        listen_port=8000,
        staking_enabled=True,
        initial_stake=1000.0
    )
    supervisor = PoUWNode(supervisor_config)
    await supervisor.initialize()
    await supervisor.start()
    
    # Start worker nodes
    workers = []
    for i in range(3):
        worker_config = NodeConfiguration(
            node_id=f"worker-{i+1}",
            node_type=NodeType.WORKER,
            listen_port=8001 + i,
            bootstrap_peers=[f"127.0.0.1:8000"],
            training_enabled=True,
            gpu_enabled=True
        )
        worker = PoUWNode(worker_config)
        await worker.initialize()
        await worker.start()
        workers.append(worker)
    
    print("Network started: 1 supervisor + 3 workers")
```

## Production Deployment

### Docker Deployment

```bash
# Build image
docker build -t pouw-node .

# Run worker node
docker run -d \
  --name pouw-worker \
  -p 8333:8333 \
  -e POUW_NODE_ID=prod-worker-1 \
  -e POUW_NODE_TYPE=worker \
  -e POUW_ENABLE_GPU=true \
  pouw-node

# Run miner node
docker run -d \
  --name pouw-miner \
  -p 8334:8334 \
  -e POUW_NODE_ID=prod-miner-1 \
  -e POUW_NODE_TYPE=miner \
  -e POUW_ENABLE_MINING=true \
  pouw-node
```

### Environment Variables

```bash
# Core configuration
export POUW_NODE_ID="prod-node-001"
export POUW_NODE_TYPE="worker"          # worker, supervisor, miner, hybrid
export POUW_LISTEN_PORT="8333"

# Network settings
export POUW_MAX_PEERS="100"
export POUW_BOOTSTRAP_PEERS="node1.pouw.network:8000,node2.pouw.network:8000"

# ML settings
export POUW_ENABLE_TRAINING="true"
export POUW_ENABLE_GPU="true"
export POUW_MAX_CONCURRENT_TASKS="3"

# Mining settings
export POUW_ENABLE_MINING="true"
export POUW_MINING_THREADS="4"

# Security settings
export POUW_SECURITY_LEVEL="high"
export POUW_ENABLE_INTRUSION_DETECTION="true"

# Production settings
export POUW_ENABLE_MONITORING="true"
export POUW_LOG_LEVEL="INFO"
```

## Development Setup

### Development Environment

```bash
# Clone repository
git clone https://github.com/your-org/PoUW.git
cd PoUW

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (including dev dependencies)
pip install -r requirements.txt

# Install development tools
pip install black pylint mypy pytest pytest-cov pytest-asyncio
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=pouw --cov-report=html

# Run specific test categories
python -m pytest tests/test_blockchain.py -v     # Blockchain tests
python -m pytest tests/test_mining.py -v            # ML tests
python -m pytest tests/test_economics.py -v     # Economic system tests
python -m pytest tests/test_network_operations.py -v  # Network communication tests
python -m pytest tests/test_enhanced_security.py -v  # Security framework tests
```

### Code Quality

```bash
# Format code
python -m black pouw/ --line-length=100

# Lint code
python -m pylint pouw/

# Type checking
python -m mypy pouw/

# Security scan
python -m bandit -r pouw/
```

## Monitoring & Management

### Node Status Monitoring

```bash
# CLI monitoring
./pouw-cli status --node-id my-node

# Interactive monitoring
./pouw-cli interactive
# Select "4. Node Status"
```

### Health Metrics

```python
# Get comprehensive status via Python API
status = node.get_status()
print(f"""
Node Status:
- ID: {status['node_id']}
- Type: {status['node_type']}
- State: {status['state']}
- Uptime: {status['uptime']}
- Peer Count: {status['peer_count']}
- Active Tasks: {status['active_tasks']}
""")
```

### Log Management

```bash
# View recent logs
./pouw-cli logs --node-id my-node --lines 100

# Follow logs in real-time
./pouw-cli logs --node-id my-node --follow

# View logs by level
./pouw-cli logs --node-id my-node --level ERROR
```

## Advanced Configuration

### High-Performance Mining Setup

```python
config = NodeConfiguration(
    node_id="performance-miner",
    node_type=NodeType.MINER,
    listen_port=8333,
    mining_enabled=True,
    mining_threads=8,        # Use all CPU cores
    gpu_enabled=True,        # Enable GPU acceleration
    max_concurrent_tasks=5,  # Handle multiple tasks
    security_level="high",
    monitoring_enabled=True
)
```

### Secure Network Configuration

```python
config = NodeConfiguration(
    node_id="secure-node",
    node_type=NodeType.SUPERVISOR,
    listen_port=8000,
    security_level="paranoid",
    authentication_required=True,
    intrusion_detection_enabled=True,
    staking_enabled=True,
    initial_stake=10000.0
)
```

## Documentation

### Core Documentation

- **[System Overview](pouw/POUW_SYSTEM_OVERVIEW_REPORT.md)** - Comprehensive system architecture
- **[CLI Guide](CLI_GUIDE.md)** - Complete CLI usage guide
- **[Deployment Guide](deployment%20docs/DEPLOYMENT.md)** - Production deployment
- **[Network Participation](deployment%20docs/HOW_TO_JOIN_NETWORK.md)** - Join the network

### Component Documentation

- **[Blockchain](pouw/blockchain/)** - Consensus and transaction processing
- **[Mining](pouw/mining/)** - PoUW mining algorithms
- **[ML Training](pouw/ml/)** - Distributed machine learning
- **[Economics](pouw/economics/)** - Staking and reward systems
- **[Networking](pouw/network/)** - P2P communication and VPN mesh
- **[Security](pouw/security/)** - Attack detection and mitigation
- **[Production](pouw/production/)** - Monitoring and optimization

## 🔍 Project Structure

```
PoUW/
├── 📁 pouw/                          # Core implementation
│   ├── 🚀 node.py                    # Main node implementation (34KB)
│   ├── 🎮 cli.py                     # Interactive CLI (135KB)
│   ├── 📁 blockchain/                # Blockchain consensus & transactions
│   ├── 📁 mining/                    # PoUW mining algorithms
│   ├── 📁 ml/                        # Machine learning components
│   ├── 📁 economics/                 # Economic systems & staking
│   ├── 📁 network/                   # P2P networking & VPN mesh
│   ├── 📁 security/                  # Security & attack mitigation
│   ├── 📁 production/                # Production monitoring & GPU
│   ├── 📁 deployment/                # Kubernetes & CI/CD
│   ├── 📁 data/                      # Data management & sharding
│   ├── 📁 advanced/                  # Advanced cryptographic features
│   └── 📁 crypto/                    # Cryptographic primitives
│
├── 📁 tests/                         # Comprehensive test suites
│   ├── 🧪 test_blockchain.py         # Blockchain functionality tests
│   ├── 🧪 test_mining.py             # Mining algorithm tests
│   ├── 🧪 test_ml.py                 # ML training tests
│   ├── 🧪 test_economics.py          # Economic system tests
│   ├── 🧪 test_network_operations.py # Network communication tests
│   ├── 🧪 test_enhanced_security.py  # Security framework tests
│   └── 🧪 test_*.py                  # Additional component tests
│
├── 📁 deployment docs/               # Deployment documentation
│   ├── 📖 DEPLOYMENT.md              # Production deployment guide
│   ├── 📖 HOW_TO_JOIN_NETWORK.md     # Network participation guide
│   └── 📖 *.md                       # Additional deployment docs
│
├── 📁 data/                          # Node data directories
├── 📁 logs/                          # Application logs
├── 📁 cache/                         # System cache
├── 🎮 pouw-cli                       # CLI executable script
├── 📋 requirements.txt               # Python dependencies
├── 🐳 .dockerignore                  # Docker build exclusions
└── 📖 README.md                      # This file
```

## Important Notes

### System Requirements

**Minimum Requirements:**
- Python 3.9+
- 4GB RAM
- 2 CPU cores
- 10GB disk space
- 10Mbps network

**Recommended for Production:**
- Python 3.11+
- 16GB+ RAM
- 8+ CPU cores (or GPU)
- 100GB+ disk space
- 100Mbps+ network

### Security Considerations

- Always use strong, unique node IDs
- Enable authentication in production environments
- Use VPN mesh networking for sensitive deployments
- Regularly monitor security logs and alerts
- Keep software updated to latest versions

## Contributing

### Getting Started

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**: Follow the coding standards
4. **Run tests**: `python -m pytest tests/ -v`
5. **Submit pull request**: With clear description

### Coding Standards

```bash
# Format code before committing
python -m black pouw/ --line-length=100

# Ensure type hints
python -m mypy pouw/

# Lint code
python -m pylint pouw/

# Run security scan
python -m bandit -r pouw/

# Run all tests with coverage
python -m pytest tests/ --cov=pouw --cov-report=html
```

### Development Workflow

```bash
# Set up development environment
git clone https://github.com/your-org/PoUW.git
cd PoUW
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Make changes and test
python -m pytest tests/ -v

# Format and lint
python -m black pouw/
python -m pylint pouw/

# Submit changes
git add .
git commit -m "feat: add amazing feature"
git push origin feature/amazing-feature
```

## Support & Community

### Getting Help

- **Documentation**: Comprehensive docs in the repository
- **Issues**: Report bugs on GitHub Issues

### Common Issues & Solutions

**Node won't start:**
```bash
# Check dependencies
pip list | grep torch
python -c "import pouw; print('OK')"

# Check configuration
./pouw-cli config show --node-id your-node
```

**Mining not working:**
```bash
# Check node type and configuration
./pouw-cli status --node-id your-node

# Verify mining is enabled
./pouw-cli config show --node-id your-node | grep mining
```

**Network connectivity issues:**
```bash
# Check peer connections
./pouw-cli peer-status --node-id your-node

# Test connectivity
./pouw-cli connect --address peer-ip --port peer-port
```
