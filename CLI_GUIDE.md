# PoUW CLI - Command Line Interface Guide

The PoUW CLI is a comprehensive command-line tool for managing PoUW blockchain nodes. It provides easy-to-use commands for starting, stopping, monitoring, and configuring nodes in your PoUW network.

## Installation and Setup

### Prerequisites
- Python 3.8+
- Virtual environment activated
- PoUW project dependencies installed

### Making CLI Executable
```bash
chmod +x pouw-cli
```

## Usage

The CLI can be used in three ways:

1. **Direct executable** (recommended):
   ```bash
   ./pouw-cli [command] [options]
   ```

2. **Python module**:
   ```bash
   python -m pouw.cli [command] [options]
   ```

3. **Interactive mode** (new!):
   ```bash
   ./pouw-cli interactive
   ```

## Interactive Mode 🎮

The interactive mode provides a user-friendly menu-driven interface that makes node management easier without needing to remember command syntax.

### Entering Interactive Mode

```bash
./pouw-cli interactive
```

### Interactive Features

The interactive mode includes:

- **🚀 Node Management**: Start, stop, restart nodes with guided wizards
- **📊 Status Monitoring**: Real-time node status with resource usage
- **📝 Log Viewing**: Browse and follow logs with easy navigation
- **⚙️ Configuration**: Create, edit, and manage configurations visually
- **🔧 Advanced Tools**: System diagnostics, cleanup, bulk operations
- **📋 Node Overview**: Table-formatted node listings with status indicators

### Interactive Menu Structure

```
🔗  PoUW Blockchain Node Management - Interactive Mode  🔗
============================================================

📋 Main Menu:
  1. 🚀 Start Node
  2. 🛑 Stop Node
  3. 🔄 Restart Node
  4. 📊 Node Status
  5. 📋 List All Nodes
  6. 📝 View Logs
  7. ⚙️  Configuration Management
  8. 🔧 Advanced Options
  9. ❓ Help
  0. 🚪 Exit
```

### Start Node Wizard

The interactive start wizard guides you through:
1. Node ID specification
2. Configuration selection (existing or new)
3. Node type selection (worker, supervisor, miner, hybrid)
4. Port configuration
5. Feature toggles (mining, training, GPU)
6. Daemon vs foreground mode selection

### Configuration Management

The configuration menu provides:
- **📝 Create**: Guided configuration creation with templates
- **👁️ Show**: Pretty-printed configuration viewing
- **✏️ Edit**: Direct file editing with your preferred editor
- **🗑️ Delete**: Safe configuration deletion with confirmations

### Advanced Options

Advanced features include:
- **🧹 Log Cleanup**: Remove old log files to free disk space
- **🔍 System Diagnostics**: Check system resources and PoUW status
- **📊 Resource Monitoring**: Monitor CPU, memory, and connections
- **🔄 Bulk Operations**: Start/stop/restart multiple nodes at once

### Tips for Interactive Mode

- Use **Ctrl+C** to cancel operations or return to main menu
- All operations include confirmation prompts for safety
- Status indicators: 🟢 (running), 🔴 (stopped)
- Log following can be interrupted with **Ctrl+C**
- Configuration files are automatically validated

### 9. ML Task Management

Submit and manage machine learning tasks on the PoUW network:

#### Submit ML Task

Submit an ML task to the network for distributed training:

```bash
# Submit task with default MNIST settings
./pouw-cli submit-task --node-id worker-1 --fee 50.0

# Submit task with custom parameters
./pouw-cli submit-task --node-id worker-1 \
  --fee 75.0 \
  --epochs 30 \
  --batch-size 64 \
  --dataset-size 50000 \
  --gpu \
  --min-accuracy 0.9

# Submit task using custom task definition file
./pouw-cli submit-task --node-id worker-1 \
  --task-file examples/custom_task.json \
  --fee 100.0
```

**Options:**
- `--node-id`: Node ID to submit task from (required)
- `--task-file`: Path to JSON task definition file
- `--fee`: Task fee in PAI coins (default: 50.0)
- `--epochs`: Max training epochs (default: 20)
- `--batch-size`: Training batch size (default: 32)
- `--dataset-size`: Dataset size (default: 60000)
- `--gpu`: Require GPU for training
- `--min-accuracy`: Minimum required accuracy (default: 0.8)

#### Interactive ML Task Management

The interactive mode provides a comprehensive ML task management interface:

```bash
./pouw-cli interactive
# Select option 12: 🧠 ML Task Management
```

**Interactive Features:**
- **📤 Submit New ML Task**: Guided task submission wizard
- **📋 List Submitted Tasks**: View all submitted tasks with status
- **📊 Task Status & Results**: Monitor task progress and results
- **📄 View Task Templates**: Browse available task templates

#### Task Definition Files

Create custom ML tasks using JSON definition files:

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
    "learning_rate": 0.001,
    "beta1": 0.9,
    "beta2": 0.999
  },
  "stopping_criterion": {
    "type": "max_epochs",
    "max_epochs": 20,
    "early_stopping": true,
    "patience": 5
  },
  "validation_strategy": {
    "type": "holdout",
    "validation_split": 0.2
  },
  "metrics": ["accuracy", "loss"],
  "dataset_info": {
    "format": "MNIST",
    "batch_size": 32,
    "training_percent": 0.8,
    "size": 60000
  },
  "performance_requirements": {
    "gpu": false,
    "min_accuracy": 0.8,
    "expected_training_time": 3600
  }
}
```

#### Supported Model Types

- **MLP**: Multi-Layer Perceptron for general classification
- **CNN**: Convolutional Neural Network for image processing
- **LSTM**: Long Short-Term Memory for sequence data
- **Custom**: User-defined architectures

#### Task Lifecycle

1. **Submission**: Task is submitted to the network with fee payment
2. **Bidding**: Worker nodes bid on the task based on requirements
3. **Selection**: Economic system selects optimal workers
4. **Training**: Selected miners perform distributed training
5. **Verification**: Results are verified and consensus reached
6. **Completion**: Rewards distributed, model weights returned

#### Example Task Workflows

**Quick MNIST Task:**
```bash
# Start a worker node
./pouw-cli start --node-id my-worker --training --gpu

# Submit simple MNIST task
./pouw-cli submit-task --node-id my-worker --fee 25.0 --epochs 10
```

**Advanced Custom Task:**
```bash
# Create custom task definition
cat > my_task.json << EOF
{
  "model_type": "cnn",
  "architecture": {
    "conv_layers": [32, 64],
    "fc_layers": [128, 10]
  },
  "dataset_info": {
    "format": "CIFAR10",
    "batch_size": 128
  },
  "performance_requirements": {
    "gpu": true,
    "min_accuracy": 0.85
  }
}
EOF

# Submit the task
./pouw-cli submit-task --node-id my-worker \
  --task-file my_task.json \
  --fee 100.0
```

**Monitor Task Progress:**
```bash
# List all submitted tasks
./pouw-cli interactive
# Select: ML Task Management → List Submitted Tasks

# Check specific task status
# (Feature coming soon: task-status command)
```

## Node Types

### Worker Node
- Participates in ML training tasks
- Processes distributed computations
- Default configuration for general use

### Supervisor Node
- Manages and coordinates worker nodes
- Validates training results
- Handles task distribution

### Miner Node
- Focuses on blockchain mining
- Validates transactions
- Maintains blockchain consensus

### Hybrid Node
- Combines multiple functionalities
- Can mine, train, and supervise
- Resource-intensive but versatile

## Configuration Templates

### Worker Node Template
```json
{
  "node_type": "worker",
  "mining_enabled": false,
  "training_enabled": true,
  "max_concurrent_tasks": 3,
  "gpu_enabled": false
}
```

### Miner Node Template
```json
{
  "node_type": "miner",
  "mining_enabled": true,
  "mining_threads": 4,
  "training_enabled": false,
  "gpu_enabled": true
}
```

### Supervisor Node Template
```json
{
  "node_type": "supervisor",
  "mining_enabled": false,
  "training_enabled": true,
  "max_concurrent_tasks": 5,
  "authentication_required": true
}
```

## File Structure

The CLI creates and manages several directories:

```
.
├── configs/          # Node configuration files
│   ├── worker-1.json
│   └── miner-1.json
├── logs/            # Node log files
│   ├── pouw_node_worker-1.log
│   └── pouw_node_miner-1.log
├── pids/            # Process ID files
│   ├── worker-1.pid
│   └── miner-1.pid
└── data/            # Node data directories
    ├── worker-1/
    │   ├── blockchain/
    │   └── models/
    └── miner-1/
        ├── blockchain/
        └── models/
```

## Example Workflows

### Quick Start with Interactive Mode

1. **Enter interactive mode:**
   ```bash
   ./pouw-cli interactive
   ```

2. **Create and start your first node:**
   - Choose option 1 (🚀 Start Node)
   - Enter node ID: `my-first-node`
   - Select node type: `worker`
   - Configure settings through the wizard
   - Start in daemon mode

3. **Monitor your node:**
   - Choose option 4 (📊 Node Status)
   - View real-time resource usage

4. **View logs:**
   - Choose option 6 (📝 View Logs)
   - Follow logs in real-time

### Setting Up a Development Network (Interactive)

1. **Enter interactive mode:**
   ```bash
   ./pouw-cli interactive
   ```

2. **Create supervisor node:**
   - Use Start Node wizard
   - Set node type to "supervisor"
   - Configure on port 8333

3. **Create worker nodes:**
   - Repeat start wizard for multiple workers
   - Use different ports (8334, 8335, etc.)

4. **Monitor the network:**
   - Use "List All Nodes" to see overview
   - Use "Node Status" for detailed monitoring

### Setting Up a Development Network (Command Line)

1. **Create a supervisor node:**
   ```bash
   ./pouw-cli config create --node-id supervisor-1 --template supervisor
   ./pouw-cli start --node-id supervisor-1 --port 8333
   ```

2. **Create worker nodes:**
   ```bash
   ./pouw-cli start --node-id worker-1 --node-type worker --port 8334 --training
   ./pouw-cli start --node-id worker-2 --node-type worker --port 8335 --training
   ```

3. **Create a miner node:**
   ```bash
   ./pouw-cli start --node-id miner-1 --node-type miner --port 8336 --mining --gpu
   ```

4. **Monitor the network:**
   ```bash
   ./pouw-cli list
   ./pouw-cli logs --node-id supervisor-1 --follow
   ```

### Production Deployment

1. **Create production configurations:**
   ```bash
   ./pouw-cli config create --node-id prod-supervisor --template supervisor --port 8333
   ./pouw-cli config create --node-id prod-worker-1 --template worker --port 8334
   ./pouw-cli config create --node-id prod-miner-1 --template miner --port 8335
   ```

2. **Start nodes in daemon mode:**
   ```bash
   ./pouw-cli start --node-id prod-supervisor
   ./pouw-cli start --node-id prod-worker-1
   ./pouw-cli start --node-id prod-miner-1
   ```

3. **Monitor status:**
   ```bash
   ./pouw-cli status --json | jq '.'
   ```

### Debugging and Troubleshooting

#### Using Interactive Mode
1. Enter interactive mode: `./pouw-cli interactive`
2. Choose "Advanced Options" → "System Diagnostics"
3. Check resource usage and system status
4. Use "Configuration Management" to verify settings
5. Use "View Logs" with follow mode for real-time debugging

#### Using Command Line
1. **Check node status:**
   ```bash
   ./pouw-cli status --node-id worker-1
   ```

2. **View recent logs:**
   ```bash
   ./pouw-cli logs --node-id worker-1 --lines 100
   ```

3. **Restart problematic node:**
   ```bash
   ./pouw-cli restart --node-id worker-1
   ```

4. **Run in foreground for debugging:**
   ```bash
   ./pouw-cli stop --node-id worker-1
   ./pouw-cli start --node-id worker-1 --foreground --log-level DEBUG
   ```

### Bulk Operations (Interactive Mode)

1. **Enter interactive mode and go to Advanced Options**
2. **Choose Bulk Operations:**
   - Start all stopped nodes
   - Stop all running nodes  
   - Restart all nodes
   - Export all configurations

## Advanced Usage

### Environment Variables

- `EDITOR`: Default text editor for config editing (default: nano)
- `POUW_LOG_LEVEL`: Global log level override
- `POUW_DATA_DIR`: Custom data directory

### Custom Bootstrap Peers

```bash
./pouw-cli start --node-id worker-1 \
  --bootstrap-peers "192.168.1.100:8333" "192.168.1.101:8333"
```

### Resource Monitoring

Monitor node resource usage:

```bash
# CPU and memory usage
./pouw-cli status --node-id worker-1 | grep -E "(CPU|Memory)"

# Connection count
./pouw-cli status --node-id worker-1 --json | jq '.connections'
```

## Troubleshooting

### Common Issues

1. **Node fails to start:**
   - Check configuration file validity
   - Verify port availability
   - Check log files for errors

2. **Permission errors:**
   - Ensure CLI script is executable
   - Check file permissions in data directories

3. **Import errors:**
   - Verify virtual environment is activated
   - Check that all dependencies are installed

### Getting Help

```bash
# General help
./pouw-cli --help

# Command-specific help
./pouw-cli start --help
./pouw-cli config --help
```

### Verbose Output

Add `--verbose` flag for detailed output:

```bash
./pouw-cli --verbose start --node-id worker-1
```

## Integration

### Systemd Integration

Create a systemd service for production deployment:

```ini
[Unit]
Description=PoUW Node %i
After=network.target

[Service]
Type=forking
User=pouw
WorkingDirectory=/path/to/pouw
ExecStart=/path/to/pouw/pouw-cli start --node-id %i
ExecStop=/path/to/pouw/pouw-cli stop --node-id %i
Restart=always

[Install]
WantedBy=multi-user.target
```

### Docker Integration

Use the CLI in Docker containers:

```dockerfile
FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["./pouw-cli", "start", "--node-id", "container-node"]
```

### Monitoring Integration

Integrate with monitoring systems:

```bash
# Prometheus metrics endpoint
curl http://localhost:9090/metrics

# Health check endpoint
curl http://localhost:8080/health
```

This CLI provides a complete solution for managing PoUW nodes in development, testing, and production environments. 