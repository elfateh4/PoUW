# Proof of Useful Work (PoUW) Implementation

A blockchain implementation that replaces Bitcoin's wasteful proof-of-work with useful machine learning training work.

## Overview

This implementation is based on the research paper "A Proof of Useful Work for Artificial Intelligence on the Blockchain" by Lihu et al. The system enables:

- **Distributed machine learning training** on blockchain
- **Mining through useful ML computation** instead of wasteful hashing
- **Verification of ML work** through deterministic replay
- **Economic incentives** for honest participation
- **Decentralized AI** using blockchain security

## Architecture

### Core Components

1. **Blockchain Core** (`pouw/blockchain/`)
   - Block structure with PoUW extensions
   - Transaction types (PAY_FOR_TASK, BUY_TICKETS, etc.)
   - Chain validation and consensus

2. **ML Training System** (`pouw/ml/`)
   - Distributed neural network training
   - Data parallelism with gradient sharing
   - Model coordination and synchronization

3. **Mining Algorithm** (`pouw/mining/`)
   - Nonce generation from ML work
   - Zero-nonce block commitments
   - PoUW mining procedure

4. **Verification System** (`pouw/verification/`)
   - ML iteration replay for verification
   - Block validation with PoUW proof
   - Verifier selection and coordination

5. **Communication** (`pouw/network/`)
   - P2P networking for blockchain
   - Fast messaging for ML coordination
   - Message history recording

6. **Economic System** (`pouw/economics/`)
   - Staking mechanism
   - Reward distribution
   - Task matching and assignment

## Quick Start

### Installation

```bash
# Clone and setup
git clone <repository>
cd PoUW
./setup.sh
source venv/bin/activate
```

### Run Complete Demo

```bash
# Start a complete PoUW network demonstration
python scripts/demo.py --miners 3 --supervisors 2 --duration 180
```

This will:
1. Start multiple PoUW nodes (miners, supervisors)
2. Submit a sample ML task (MNIST-like classification)
3. Show distributed training with useful work mining
4. Display real-time progress and final results

### Run Tests

```bash
# Run comprehensive test suite
python tests/run_tests.py
```

## Detailed Usage

### 1. Start Individual Nodes

#### Miner Node
```bash
python scripts/start_miner.py \
    --node-id miner_001 \
    --port 8000 \
    --stake 100.0
```

#### Supervisor Node
```bash
python scripts/start_supervisor.py \
    --node-id supervisor_001 \
    --port 8001 \
    --stake 50.0
```

### 2. Submit ML Tasks

```bash
python scripts/submit_task.py \
    --node-id client_001 \
    --task examples/mnist_task.json \
    --fee 50.0 \
    --bootstrap-peer localhost 8000
```

### 3. Start Complete Network

```bash
python scripts/start_network.py \
    --miners 3 \
    --supervisors 2 \
    --evaluators 1 \
    --submit-task
```

## Key Concepts

### Node Roles

- **Miners**: Perform ML training and mine blocks with nonces derived from ML work
- **Supervisors**: Record message history and guard against malicious behavior  
- **Evaluators**: Test final models and distribute rewards
- **Verifiers**: Validate blocks by re-running ML iterations
- **Clients**: Submit ML tasks and pay for training

### PoUW Mining Process

1. **ML Iteration**: Miner performs one training iteration (forward/backward pass)
2. **Nonce Generation**: Derive nonce from model weights and gradients
3. **Block Mining**: Attempt to mine block with limited nonce range
4. **Verification**: Verifiers re-run ML iteration to validate work

### Economic Incentives

- **Staking**: Nodes stake PAI coins to participate
- **Rewards**: Based on ML performance and honest behavior
- **Punishment**: Malicious actors lose their stake

## Configuration Examples

### MNIST Classification Task
```json
{
  "model_type": "mlp",
  "architecture": {
    "input_size": 784,
    "hidden_sizes": [128, 64],
    "output_size": 10
  },
  "optimizer": {
    "type": "adam",
    "learning_rate": 0.001
  },
  "stopping_criterion": {
    "type": "max_epochs",
    "max_epochs": 50
  },
  "metrics": ["accuracy", "loss"],
  "dataset_info": {
    "batch_size": 32,
    "size": 60000
  },
  "performance_requirements": {
    "gpu": false,
    "min_accuracy": 0.95
  }
}
```

### Node Preferences
```python
# Miner preferences
miner_preferences = {
    'model_types': ['mlp', 'cnn'],
    'has_gpu': True,
    'max_dataset_size': 1000000
}

# Supervisor preferences  
supervisor_preferences = {
    'storage_capacity': 10000000,
    'bandwidth': 1000000,
    'redundancy_scheme': 'full_replicas'
}
```

## Implementation Status

- [x] **Core Blockchain** - Complete block/transaction structure
- [x] **PoUW Mining** - Nonce generation from ML work
- [x] **ML Training** - Distributed training coordination
- [x] **Network Communication** - P2P messaging system
- [x] **Economic System** - Staking and reward distribution
- [x] **Verification** - ML work validation
- [x] **Demo & Tests** - Complete test suite and demonstrations
- [ ] **Production Features** - Advanced security, optimizations
- [ ] **Real Datasets** - Integration with actual ML datasets
- [ ] **GPU Support** - CUDA acceleration for training

## Performance Characteristics

### Theoretical Analysis (from paper)
- **Energy Efficiency**: 10x more efficient than Bitcoin mining
- **Profitability**: ~200% ROI vs 18% for Bitcoin mining
- **Client Savings**: ~30% cheaper than cloud ML services

### Actual Implementation
- **Throughput**: ~1-10 blocks/minute (adjustable difficulty)
- **Training Speed**: Depends on model size and network latency
- **Verification Time**: 2-5x training time for iteration replay

## Security Considerations

### Attack Mitigation
- **Pre-trained Models**: Requires iteration-by-iteration verification
- **Sybil Attacks**: VRF-based random worker selection
- **Byzantine Actors**: Stake-based punishment system
- **DoS Attacks**: Pause/resume mechanisms with heartbeats

### Trust Model
- Assumes majority of participants are honest
- Economic incentives align with network security
- Cryptographic verification of all ML work

## Development

### Project Structure
```
PoUW/
├── pouw/                 # Main package
│   ├── blockchain/       # Blockchain core
│   ├── ml/              # ML training system
│   ├── mining/          # PoUW mining algorithm
│   ├── network/         # P2P communication
│   ├── economics/       # Economic system
│   └── node.py          # Complete node implementation
├── scripts/             # Executable scripts
├── tests/               # Test suite
├── examples/            # Configuration examples
└── docs/                # Documentation
```

### Adding New Features

1. **New ML Models**: Extend `MLModel` base class in `pouw/ml/training.py`
2. **Mining Algorithms**: Modify `PoUWMiner` in `pouw/mining/algorithm.py`
3. **Economic Mechanisms**: Update `EconomicSystem` in `pouw/economics/system.py`
4. **Network Protocols**: Extend message handlers in `pouw/network/communication.py`

### Testing

```bash
# Run specific test modules
python -m pytest tests/test_blockchain.py -v
python -m pytest tests/test_ml.py -v
python -m pytest tests/test_mining.py -v
python -m pytest tests/test_economics.py -v

# Run with coverage
python -m pytest tests/ --cov=pouw --cov-report=html
```

## Troubleshooting

### Common Issues

1. **Port Conflicts**: Change default ports in scripts
2. **Memory Usage**: Reduce model size or batch size for testing
3. **Network Connectivity**: Check firewall settings for P2P connections
4. **Dependencies**: Ensure PyTorch and other requirements are installed

### Debug Mode

```bash
# Enable verbose logging
python scripts/demo.py --verbose

# Check individual components
python -c "from pouw.blockchain import Blockchain; print('Blockchain OK')"
python -c "from pouw.ml import SimpleMLP; print('ML OK')"
python -c "from pouw.mining import PoUWMiner; print('Mining OK')"
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Ensure all tests pass (`python tests/run_tests.py`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open Pull Request

## References

- **Original Paper**: [arXiv:2001.09244v1](https://arxiv.org/abs/2001.09244) - "A Proof of Useful Work for Artificial Intelligence on the Blockchain"
- **Project PAI**: https://projectpai.com
- **Related Work**: Permacoin, Gridcoin, SingularityNET

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original research by Lihu et al.
- Project PAI for the theoretical foundation
- PyTorch team for the ML framework
- Bitcoin developers for blockchain inspiration
