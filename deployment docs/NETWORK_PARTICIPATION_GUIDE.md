# PoUW Network Participation Guide

## 🌐 How Other Devices Can Join the PoUW Network

### Node Types Available

Your PoUW network supports multiple node types as defined in the PoUW paper:

#### 1. **Client Nodes**

- **Purpose**: Submit ML tasks and pay for training services
- **Requirements**: PAI tokens for task fees, task definition
- **Ideal for**: ML researchers, companies, individuals needing model training

#### 2. **Miner Nodes (Workers)**

- **Purpose**: Perform ML training tasks and mine blocks with useful work
- **Requirements**: GPU/CPU for ML computations, stake tokens
- **Ideal for**: Gaming PCs, workstations, dedicated mining rigs

#### 3. **Supervisor Nodes**

- **Purpose**: Record message history, coordinate tasks, validate work, manage network
- **Requirements**: Reliable internet, moderate compute, higher stake
- **Ideal for**: Servers, always-on devices, enterprise nodes

#### 4. **Evaluator Nodes**

- **Purpose**: Test final models, select best model, distribute client fees
- **Requirements**: ML expertise, evaluation datasets, stake tokens
- **Ideal for**: Academic institutions, ML researchers

#### 5. **Verifier Nodes**

- **Purpose**: Validate ML model outputs and training results, verify blocks
- **Requirements**: Moderate compute, network bandwidth, stake tokens
- **Ideal for**: Edge devices, validation services

#### 6. **Peer Nodes**

- **Purpose**: Support network infrastructure, data relay, regular transactions
- **Requirements**: Network bandwidth, storage
- **Ideal for**: Home routers, IoT devices, mobile phones

---

## 🚀 Quick Start for New Devices

### Method 1: Direct Connection (Recommended)

```bash
# On the new device, install PoUW
git clone https://github.com/YOUR_USERNAME/PoUW.git
cd PoUW
pip install -r requirements.txt

# Start as a miner node connecting to your VPS
python main.py \
  --role miner \
  --node-id "home_miner_001" \
  --port 8000 \
  --stake 100.0 \
  --bootstrap-peers "YOUR_VPS_IP:8000"
```

### Method 2: Docker Container

```bash
# Pull and run PoUW node
docker run -d \
  --name pouw-miner \
  -p 8000:8000 \
  -e NODE_ROLE=miner \
  -e NODE_ID=docker_miner_001 \
  -e STAKE=100.0 \
  -e BOOTSTRAP_PEERS="YOUR_VPS_IP:8000" \
  your-registry/pouw:latest
```

### Method 3: Python Script

```python
#!/usr/bin/env python3
"""
Simple script to join PoUW network
"""
import asyncio
from pouw.node import PoUWNode, NodeConfig
from pouw.economics import NodeRole

async def join_network():
    # Create node configuration
    config = NodeConfig(
        node_id="my_device_001",
        role=NodeRole.MINER,
        host="0.0.0.0",  # Listen on all interfaces
        port=8000,
        initial_stake=100.0,
        bootstrap_peers=[("YOUR_VPS_IP", 8000)]  # Your Hostinger VPS
    )

    # Create and start node
    node = PoUWNode("my_device_001", NodeRole.MINER, config=config)
    await node.start()

    print(f"Node {node.node_id} joined network!")
    print("Press Ctrl+C to stop")

    # Keep running
    try:
        while True:
            await asyncio.sleep(10)
            status = node.get_status()
            print(f"Status: Connected peers: {status.get('peer_count', 0)}")
    except KeyboardInterrupt:
        await node.stop()

if __name__ == "__main__":
    asyncio.run(join_network())
```

---

## 🔧 Device-Specific Configurations

### Desktop/Laptop Computers

```python
# High-performance miner configuration
config = NodeConfig(
    node_id="desktop_miner_001",
    role=NodeRole.MINER,
    initial_stake=200.0,
    preferences={
        'model_types': ['cnn', 'transformer', 'mlp'],
        'has_gpu': True,
        'gpu_memory': '8GB',
        'max_dataset_size': 10000000,
        'training_batch_size': 128
    }
)
```

### Raspberry Pi / Edge Devices

```python
# Lightweight configuration for resource-constrained devices
config = NodeConfig(
    node_id="pi_node_001",
    role=NodeRole.PEER,
    initial_stake=10.0,
    preferences={
        'lightweight_mode': True,
        'max_dataset_size': 10000,
        'cpu_only': True,
        'low_power_mode': True
    }
)
```

### Mobile Devices

```python
# Mobile-optimized configuration
config = NodeConfig(
    node_id="mobile_node_001",
    role=NodeRole.VERIFIER,
    initial_stake=5.0,
    preferences={
        'mobile_optimized': True,
        'battery_aware': True,
        'data_saver_mode': True,
        'background_processing': True
    }
)
```

### Cloud Instances

```python
# Cloud instance configuration
config = NodeConfig(
    node_id="aws_node_001",
    role=NodeRole.SUPERVISOR,
    initial_stake=500.0,
    preferences={
        'cloud_provider': 'aws',
        'instance_type': 'p3.2xlarge',
        'auto_scaling': True,
        'high_availability': True
    }
)
```

---

## 🌍 Network Discovery Methods

### 1. Bootstrap Peers (Primary Method)

```python
# Connect to known network entry points
bootstrap_peers = [
    ("your-vps-domain.com", 8000),    # Your Hostinger VPS
    ("backup-node.example.com", 8000), # Backup entry point
    ("192.168.1.100", 8000)           # Local network node
]
```

### 2. Dynamic Discovery (Advanced)

```python
# DNS-based discovery
def discover_network_peers():
    import socket
    peers = []

    # DNS TXT record lookup for network peers
    try:
        txt_records = socket.getaddrinfo('_pouw._tcp.your-domain.com', None)
        for record in txt_records:
            peers.append((record[4][0], 8000))
    except:
        pass

    return peers
```

### 3. Local Network Discovery

```python
# mDNS/Bonjour discovery for local peers
def discover_local_peers():
    import socket
    peers = []

    # Broadcast discovery on local network
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    # Send discovery broadcast
    discovery_msg = "POUW_DISCOVERY"
    sock.sendto(discovery_msg.encode(), ('255.255.255.255', 8001))

    # Listen for responses (simplified)
    return peers
```

---

## 🔐 Security & Authentication

### Device Authentication

```python
# Generate device credentials
from pouw.crypto import generate_keypair

private_key, public_key = generate_keypair()

config = NodeConfig(
    node_id="authenticated_device_001",
    private_key=private_key,
    public_key=public_key,
    enable_encryption=True,
    require_authentication=True
)
```

### Network Security

```python
# Secure connection configuration
config = NodeConfig(
    node_id="secure_node_001",
    security_settings={
        'tls_enabled': True,
        'certificate_path': '/path/to/cert.pem',
        'allowed_peers': ['trusted_node_001', 'supervisor_001'],
        'firewall_rules': {
            'allow_ports': [8000, 8001, 8002],
            'block_ips': ['malicious.ip.address']
        }
    }
)
```

---

## 📱 Platform-Specific Instructions

### Windows

```powershell
# Install Python and dependencies
winget install Python.Python.3.11
pip install -r requirements.txt

# Run as Windows Service
python -m pip install pywin32
python scripts/install_windows_service.py

# Start PoUW node service
net start PoUWNode
```

### macOS

```bash
# Using Homebrew
brew install python@3.11
pip3 install -r requirements.txt

# Create launch daemon
sudo cp scripts/com.pouw.node.plist /Library/LaunchDaemons/
sudo launchctl load /Library/LaunchDaemons/com.pouw.node.plist
```

### Linux

```bash
# Install dependencies
sudo apt update
sudo apt install python3 python3-pip
pip3 install -r requirements.txt

# Create systemd service
sudo cp scripts/pouw-node.service /etc/systemd/system/
sudo systemctl enable pouw-node
sudo systemctl start pouw-node
```

### Android (Termux)

```bash
# Install Termux, then:
pkg update
pkg install python
pip install -r requirements-mobile.txt

# Run lightweight node
python main.py --role PEER --lightweight-mode
```

### iOS (iSH Shell)

```bash
# Install iSH app, then:
apk add python3 py3-pip
pip3 install -r requirements-ios.txt

# Run verification node
python3 main.py --role VERIFIER --mobile-optimized
```

---

## 🔧 Configuration Templates

### Home Network Setup

```yaml
# home-network.yml
network_config:
  gateway_node:
    role: SUPERVISOR
    stake: 100.0
    public_ip: true

  worker_nodes:
    - device: "gaming-pc"
      role: MINER
      stake: 200.0
      gpu: true

    - device: "laptop"
      role: VERIFIER
      stake: 50.0

    - device: "raspberry-pi"
      role: PEER
      stake: 10.0

  network_settings:
    local_discovery: true
    port_range: [8000, 8010]
    bandwidth_limit: "100Mbps"
```

### Enterprise Deployment

```yaml
# enterprise.yml
network_config:
  data_center:
    supervisor_count: 3
    miner_count: 10
    verifier_count: 5

  edge_locations:
    - location: "office-ny"
      nodes: 5
      role: MINER

    - location: "office-sf"
      nodes: 3
      role: VERIFIER

  security:
    vpn_mesh: true
    encryption: "AES-256"
    authentication: "certificate"
```

---

## 📊 Monitoring & Management

### Device Status Dashboard

```python
# Simple monitoring script
async def monitor_network_devices():
    while True:
        devices = await discover_network_devices()

        for device in devices:
            status = await device.get_health_status()
            print(f"{device.node_id}: {status['health']} - "
                  f"Tasks: {status['active_tasks']}")

        await asyncio.sleep(30)
```

### Resource Management

```python
# Auto-adjust based on device capabilities
def optimize_for_device(node):
    if node.is_mobile():
        node.set_power_mode('battery_saver')
        node.set_task_limit(1)
    elif node.has_gpu():
        node.set_task_types(['cnn', 'transformer'])
        node.set_batch_size(128)
    else:
        node.set_task_types(['mlp', 'simple'])
        node.set_batch_size(32)
```

---

## 🚀 Getting Started Checklist

### For New Device Owners:

1. **Install PoUW software** on your device
2. **Configure node role** based on device capabilities
3. **Set bootstrap peers** to your VPS: `YOUR_VPS_IP:8000`
4. **Choose stake amount** (start with minimum: 10-100 tokens)
5. **Start the node** and verify connection
6. **Monitor performance** and adjust settings

### For Network Administrators:

1. **Document network entry points** (your VPS details)
2. **Set up monitoring** for new node connections
3. **Configure load balancing** for high node counts
4. **Implement security policies** for device authentication
5. **Create device-specific configs** for different platforms
6. **Set up automated deployment** scripts

---

## 🤝 Community & Support

- **Network Status**: Monitor at `http://YOUR_VPS_IP/status`
- **Node Discovery**: Use bootstrap peer `YOUR_VPS_IP:8000`
- **Documentation**: Full API docs in `/docs`
- **Troubleshooting**: Check logs in `/logs` directory

Join the network today and start contributing to useful AI computation! 🚀
