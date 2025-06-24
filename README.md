# PoUW: Proof of Useful Work Blockchain

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](tests/)
[![Documentation](https://img.shields.io/badge/docs-comprehensive-brightgreen.svg)](pouw/node%20docs/)

A revolutionary blockchain implementation that combines **distributed machine learning** with **cryptocurrency mining**, creating a system where computational work contributes to both network security and artificial intelligence advancement.

## 🌟 Key Features

- **🤖 ML-Powered Mining**: Mine blocks using machine learning computation instead of wasteful hash calculations
- **🌐 Distributed AI Training**: Coordinate federated learning across a decentralized network
- **💰 Economic Incentives**: Stake-based participation with rewards for quality contributions
- **🔒 Enterprise Security**: Advanced attack detection, Byzantine fault tolerance, and gradient poisoning protection
- **⚡ High Performance**: Optimized for production deployment with comprehensive monitoring
- **🔧 Multi-Role Support**: Miners, supervisors, verifiers, evaluators, and peers working together

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    PoUW Network                             │
├─────────────────────────────────────────────────────────────┤
│  Blockchain Layer    │  ML Layer         │  Economic Layer  │
│  • Consensus         │  • Fed Learning   │  • Staking       │
│  • Transactions      │  • Model Sync     │  • Rewards       │
│  • Block Mining      │  • Quality Eval   │  • Reputation    │
├─────────────────────────────────────────────────────────────┤
│  Network Layer       │  Security Layer   │  Production      │
│  • P2P Protocol      │  • Attack Det.    │  • Monitoring    │
│  • Mesh Networking   │  • BFT            │  • Optimization  │
│  • Load Balancing    │  • Encryption     │  • Analytics     │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+**
- **PyTorch** (for ML computation)
- **NumPy** (for numerical operations)
- **4GB+ RAM** (8GB+ recommended for mining)
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
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/macOS
# or
venv\Scripts\activate     # On Windows
```

#### 3. Install Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# Verify installation
python -c "import pouw; print('PoUW installed successfully!')"
```

#### 4. Run Basic Test

```bash
# Run the node test suite
python test_node.py

# Expected output: "🎉 All tests passed!"
```

### 🎯 Your First PoUW Node

#### Start a Miner Node

```python
import asyncio
from pouw.node import PoUWNode, NodeConfig
from pouw.economics import NodeRole

async def start_miner():
    # Create miner configuration
    config = NodeConfig(
        node_id="my_miner",
        role=NodeRole.MINER,
        host="0.0.0.0",
        port=8001,
        initial_stake=100.0
    )

    # Create and start node
    node = PoUWNode("my_miner", NodeRole.MINER, config=config)
    await node.start()

    # Stake tokens and start mining
    ticket = node.stake_and_register(100.0)
    await node.start_mining()

    print(f"🎉 Miner started! Ticket ID: {ticket.ticket_id}")

    # Keep running
    while True:
        status = node.get_status()
        print(f"Blocks mined: {status['stats']['blocks_mined']}")
        await asyncio.sleep(60)

# Run the miner
asyncio.run(start_miner())
```

#### Using VS Code Tasks (Recommended)

```bash
# Use pre-configured VS Code tasks
Ctrl+Shift+P -> "Tasks: Run Task"

# Available tasks:
- "Start Miner Node"      # Launch a miner
- "Submit Test Task"      # Submit ML task
- "Run Complete Demo"     # Full system demo
- "Run All Tests"         # Test suite
```

## 🎮 Usage Examples

### Multi-Node Network Setup

```python
# scripts/start_network.py
async def setup_network():
    # Bootstrap supervisor
    supervisor = PoUWNode("supervisor", NodeRole.SUPERVISOR, port=8000)
    await supervisor.start()
    supervisor.stake_and_register(1000.0)

    # Worker miners
    miners = []
    for i in range(3):
        config = NodeConfig(
            node_id=f"miner_{i}",
            role=NodeRole.MINER,
            port=8001 + i,
            bootstrap_peers=[("127.0.0.1", 8000)]
        )
        miner = PoUWNode(f"miner_{i}", NodeRole.MINER, config=config)
        await miner.start()
        miner.stake_and_register(100.0)
        await miner.start_mining()
        miners.append(miner)

    print("🌐 Network started with 1 supervisor + 3 miners")
```

### Task Submission

```python
from pouw.blockchain import MLTask

async def submit_training_task():
    # Create client node
    client = PoUWNode("client", NodeRole.PEER)
    await client.start()

    # Define ML task
    task = MLTask(
        task_id="image_classification_001",
        task_type="image_classification",
        dataset_hash="0x123abc...",
        model_requirements={
            "architecture": "CNN",
            "min_accuracy": 0.90,
            "max_epochs": 100
        },
        reward_amount=50.0
    )

    # Submit with fee
    success = await client.submit_task(task, fee=5.0)
    print(f"Task submitted: {success}")
```

## 🏭 Production Deployment

### Docker Deployment

```bash
# Build image
docker build -t pouw-node .

# Run miner
docker run -d \
  --name pouw-miner \
  -p 8001:8001 \
  -e POUW_ROLE=MINER \
  -e POUW_NODE_ID=prod_miner_001 \
  -e POUW_STAKE_AMOUNT=1000 \
  pouw-node
```

### Kubernetes Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/deployment.yaml

# Scale miners
kubectl scale deployment pouw-miners --replicas=5

# Check status
kubectl get pods -l app=pouw-node
```

### Environment Variables

```bash
# Core configuration
export POUW_NODE_ID="prod_node_001"
export POUW_ROLE="MINER"              # MINER, SUPERVISOR, VERIFIER, EVALUATOR, PEER
export POUW_HOST="0.0.0.0"
export POUW_PORT="8001"

# Economic settings
export POUW_STAKE_AMOUNT="1000"
export POUW_MINING_INTENSITY="0.00001"

# Network settings
export POUW_MAX_PEERS="100"
export POUW_BOOTSTRAP_PEERS="node1.pouw.network:8000,node2.pouw.network:8000"

# Feature flags
export POUW_ENABLE_SECURITY="true"
export POUW_ENABLE_PRODUCTION_FEATURES="true"
export POUW_ENABLE_ADVANCED_FEATURES="true"
```

## 🔧 Development Setup

### Development Environment

```bash
# Install development dependencies
pip install -r requirements.txt
pip install black pylint mypy pytest pytest-cov

# Set up pre-commit hooks (optional)
pre-commit install
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=pouw --cov-report=html

# Run specific test categories
python -m pytest tests/test_blockchain.py -v    # Blockchain tests
python -m pytest tests/test_ml.py -v           # ML tests
python -m pytest tests/test_mining.py -v       # Mining tests
python -m pytest tests/test_economics.py -v    # Economics tests
```

### Code Quality

```bash
# Format code
python -m black . --line-length=100

# Lint code
python -m pylint pouw/

# Type checking
python -m mypy pouw/

# Clean cache
find . -type d -name "__pycache__" -exec rm -rf {} +
```

## 📊 Monitoring & Analytics

### Node Status Monitoring

```python
# Get comprehensive status
status = node.get_status()
print(f"""
Node Status:
- ID: {status['node_id']}
- Role: {status['role']}
- Running: {status['is_running']}
- Mining: {status['is_mining']}
- Blockchain Height: {status['blockchain_height']}
- Peer Count: {status['peer_count']}
- Blocks Mined: {status['stats']['blocks_mined']}
- Rewards Earned: {status['stats']['rewards_earned']} PAI
""")
```

### Health Metrics

```python
# Monitor node health
health = node.get_health_metrics()
print(f"""
Health Metrics:
- Response Time: {health.response_time}ms
- Success Rate: {health.success_rate:.2%}
- CPU Usage: {health.cpu_usage:.1f}%
- Memory Usage: {health.memory_usage:.1f}%
""")
```

### Economic Status

```python
# Check economic participation
economic = node.get_economic_status()
print(f"""
Economic Status:
- Staked: {economic['staked']}
- Stake Amount: {economic.get('stake_amount', 0)} PAI
- Reputation: {economic['reputation']:.2f}
- Total Rewards: {economic['rewards_earned']} PAI
""")
```

## 🛠️ Advanced Configuration

### Mining Optimization

```python
# High-performance mining setup
config = NodeConfig(
    node_id="performance_miner",
    role=NodeRole.MINER,
    omega_b=1e-4,      # Larger batch sizes
    omega_m=1e-6,      # Larger models
    max_peers=200,     # More network connections
    enable_production_features=True
)
```

### Security Hardening

```python
# Maximum security configuration
config = NodeConfig(
    node_id="secure_node",
    role=NodeRole.SUPERVISOR,
    enable_security_monitoring=True,
    enable_attack_mitigation=True,
    preferences={
        "security_level": "maximum",
        "byzantine_tolerance": 0.33,
        "gradient_detection_sensitivity": 0.9
    }
)
```

### Network Optimization

```python
# Network-optimized setup
config = NodeConfig(
    node_id="network_node",
    role=NodeRole.VERIFIER,
    max_peers=500,
    bootstrap_peers=[
        ("primary.pouw.network", 8000),
        ("secondary.pouw.network", 8000),
        ("tertiary.pouw.network", 8000)
    ],
    preferences={
        "bandwidth_limit": "100Mbps",
        "connection_timeout": 30,
        "heartbeat_interval": 10
    }
)
```

## 📚 Documentation

- **[Node Documentation](pouw/node%20docs/node_documentation.md)** - Comprehensive usage guide
- **[API Reference](pouw/node%20docs/node_api_reference.md)** - Technical API documentation
- **[Quick Start Guide](pouw/node%20docs/node_README.md)** - Getting started quickly
- **[Implementation Details](pouw/node%20docs/node_implementation_details.md)** - Technical deep dive

### Component Documentation

- **[Blockchain](pouw/blockchain/)** - Consensus and transaction processing
- **[Mining](pouw/mining/)** - PoUW mining algorithms
- **[ML Training](pouw/ml/)** - Distributed machine learning
- **[Economics](pouw/economics/)** - Staking and reward systems
- **[Networking](pouw/network/)** - P2P communication
- **[Security](pouw/security/)** - Attack detection and mitigation

## 🔍 Project Structure

```
PoUW/
├── 📁 pouw/                          # Core implementation
│   ├── 🚀 node.py                    # Main node implementation
│   ├── 📁 blockchain/                # Blockchain consensus
│   ├── 📁 mining/                    # PoUW mining algorithms
│   ├── 📁 ml/                        # Machine learning components
│   ├── 📁 economics/                 # Economic systems
│   ├── 📁 network/                   # P2P networking
│   ├── 📁 security/                  # Security & attack mitigation
│   └── 📁 production/                # Production monitoring
│
├── 📁 tests/                         # Test suites
│   ├── 🧪 test_blockchain.py         # Blockchain tests
│   ├── 🧪 test_mining.py             # Mining tests
│   ├── 🧪 test_ml.py                 # ML tests
│   └── 🧪 test_*.py                  # Other component tests
│
├── 📁 OUTDATED/                      # Historical reference files
│   ├── 📁 scripts/                   # Legacy scripts
│   ├── 📁 demos/                     # Old demonstrations
│   └── 📁 debugs/                    # Debug utilities
│
├── 📁 k8s/                           # Kubernetes deployment
├── 📁 jenkins/                       # CI/CD pipeline
├── 🐳 docker-compose.yml             # Docker setup
├── 📋 requirements.txt               # Dependencies
├── 🧪 test_node.py                   # Main test suite
└── 📖 README.md                      # This file
```

## ⚠️ Important Notes

### Historical Files

The `OUTDATED/` directory contains legacy scripts, demos, and documentation that may not work with the current implementation. These are kept for reference only.

**Current/Working Files:**

- ✅ `pouw/node.py` - Main implementation
- ✅ `test_node.py` - Working test suite
- ✅ `pouw/node docs/` - Current documentation
- ✅ VS Code tasks - Pre-configured commands

**Outdated/Reference Files:**

- ⚠️ `OUTDATED/scripts/` - Legacy scripts (may have broken APIs)
- ⚠️ `OUTDATED/demos/` - Old demonstrations (dependency issues)
- ⚠️ `OUTDATED/docs/` - Historical documentation

### Performance Requirements

**Minimum Requirements:**

- Python 3.9+
- 4GB RAM
- 2 CPU cores
- 10GB disk space
- 10Mbps network

**Recommended for Mining:**

- Python 3.11+
- 16GB+ RAM
- 8+ CPU cores (or GPU)
- 50GB+ disk space
- 100Mbps+ network

## 🤝 Contributing

### Getting Started

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**: Follow the coding standards
4. **Run tests**: `python -m pytest tests/ -v`
5. **Submit pull request**: With clear description

### Coding Standards

```bash
# Format code before committing
python -m black . --line-length=100

# Ensure type hints
python -m mypy pouw/

# Lint code
python -m pylint pouw/

# Run all tests
python -m pytest tests/ --cov=pouw
```

### Development Workflow

```bash
# Set up development environment
git clone https://github.com/your-org/PoUW.git
cd PoUW
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Make changes and test
python test_node.py
python -m pytest tests/ -v

# Submit changes
git add .
git commit -m "feat: add amazing feature"
git push origin feature/amazing-feature
```

## 🆘 Support & Community

### Getting Help

- **📖 Documentation**: Check the comprehensive docs in `pouw/node docs/`
- **🐛 Issues**: Report bugs on GitHub Issues
- **💬 Discussions**: Join GitHub Discussions for questions
- **📧 Email**: Contact team@pouw.network for enterprise support

### Common Issues & Solutions

**Node won't start:**

```bash
# Check dependencies
pip list | grep torch
python -c "import pouw; print('OK')"

# Check configuration
python test_node.py
```

**Mining not working:**

```bash
# Verify node role and staking
# Check logs for mining errors
# Ensure sufficient resources
```

**Network connectivity issues:**

```bash
# Check firewall settings
# Verify bootstrap peers
# Test network connectivity
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎯 Roadmap

### Current Status (v1.0)

- ✅ Complete PoUW node implementation
- ✅ Multi-role support (Miner, Supervisor, Verifier, Evaluator, Peer)
- ✅ ML-powered mining with PyTorch integration
- ✅ Economic staking and reward systems
- ✅ Advanced security and attack mitigation
- ✅ Production-ready monitoring and optimization
- ✅ Comprehensive documentation and testing

### Coming Soon (v1.1)

- 🔮 Enhanced consensus algorithms
- 🔮 Mobile/lightweight node support
- 🔮 Advanced ML model architectures
- 🔮 Cross-chain interoperability
- 🔮 Web dashboard for monitoring
- 🔮 Plugin architecture for extensions

### Future Vision (v2.0+)

- 🌟 Quantum-resistant cryptography
- 🌟 Advanced privacy-preserving ML
- 🌟 Decentralized governance system
- 🌟 Enterprise SaaS platform
- 🌟 Mobile app ecosystem
- 🌟 Global network scaling

## 🏆 Acknowledgments

- **PyTorch Team** - For the excellent ML framework
- **Bitcoin/Ethereum Communities** - For blockchain inspiration
- **Federated Learning Research** - For distributed ML concepts
- **Open Source Community** - For tools and libraries used

---

**⭐ Star this repository if you find PoUW useful!**

**🚀 Ready to revolutionize blockchain with useful computation? [Get started now](#-quick-start)!**
