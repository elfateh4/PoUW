# 🚀 PoUW Deployment Checklist

## ✅ Pre-Deployment Checklist

### 1. Local Testing

- [ ] Run `python main.py --help` to verify main application works
- [ ] Run `python join_network.py --help` to verify join script works
- [ ] Test locally: `python main.py --role SUPERVISOR --port 8000`
- [ ] Run test suite: `./test-deployment.sh`

### 2. VPS Preparation

- [ ] VPS instance running (Ubuntu 20.04+ recommended)
- [ ] SSH access configured
- [ ] Domain name pointed to VPS IP (**required for SSL certificates**)
- [ ] DNS propagation completed (check with `dig +short your-domain.com`)
- [ ] Email address for Let's Encrypt notifications
- [ ] Firewall ports opened: 22 (SSH), 80 (HTTP), 443 (HTTPS), 8000 (PoUW)

### 3. GitHub Repository

- [ ] Code pushed to GitHub repository
- [ ] GitHub Actions secrets configured:
  - `VPS_HOST`: Your VPS IP address
  - `VPS_USERNAME`: SSH username (usually 'root')
  - `VPS_SSH_KEY`: Your private SSH key
  - `VPS_PORT`: SSH port (usually 22)

---

## 🚀 Deployment Steps

### Method 1: Automated Deployment with SSL

```bash
# Quick deployment with SSL certificates
curl -fsSL https://raw.githubusercontent.com/YOUR_USERNAME/PoUW/main/deploy.sh -o deploy.sh
chmod +x deploy.sh

# Deploy with SSL automation
sudo ./deploy.sh -d your-domain.com -e your-email@example.com

# OR deploy with staging certificates (for testing)
sudo ./deploy.sh -d your-domain.com -e your-email@example.com --ssl-staging

# OR deploy without SSL (development only)
sudo ./deploy.sh --skip-ssl
```

### Method 2: Manual SSL Setup

```bash
# Standard deployment first
curl -fsSL https://raw.githubusercontent.com/YOUR_USERNAME/PoUW/main/deploy.sh -o deploy.sh
chmod +x deploy.sh
sudo ./deploy.sh --skip-ssl

# Then setup SSL manually
cd /opt/pouw
./scripts/ssl-setup.sh -d your-domain.com -e your-email@example.com
```

### Step 2: Verify SSL Configuration

```bash
# Comprehensive SSL verification
cd /opt/pouw
./scripts/ssl-verify.sh -d your-domain.com -v

# Check certificate status
./scripts/ssl-monitor.sh -d your-domain.com --monitor
```

### Step 3: Start Services

```bash
cd /opt/pouw
docker-compose -f docker-compose.production.yml up -d
```

### Step 4: Verify Deployment

```bash
# Test health endpoint (HTTPS if SSL configured)
curl https://your-domain.com/health
# OR for non-SSL
curl http://YOUR_VPS_IP/health

# Check container status
docker-compose -f docker-compose.production.yml ps

# View logs
docker-compose -f docker-compose.production.yml logs -f
```

---

## 🧪 Testing Your Deployment

### Automated Testing

```bash
# Run comprehensive tests
./test-deployment.sh --host YOUR_VPS_IP

# Test specific components
curl http://YOUR_VPS_IP:8000/health
curl http://YOUR_VPS_IP:8000/status
```

### Manual Testing

#### 1. Health Checks

- [ ] Health endpoint: `http://YOUR_VPS_IP/health`
- [ ] Status endpoint: `http://YOUR_VPS_IP/status`
- [ ] Dashboard: `http://YOUR_VPS_IP:8080` (if running)

#### 2. Network Connectivity

- [ ] Port 8000 accessible from external networks
- [ ] WebSocket connections work
- [ ] Firewall properly configured

#### 3. Container Health

```bash
docker ps
docker logs pouw-production
docker exec pouw-production python -c "from pouw.node import PoUWNode; print('Import successful')"
```

---

## 👥 Adding Devices to Network

### For New Users

#### Method 1: Simple Join (Recommended)

```bash
# On any device with Python
curl -fsSL https://raw.githubusercontent.com/YOUR_USERNAME/PoUW/main/join_network.py -o join_network.py
python join_network.py --bootstrap-peer YOUR_VPS_IP:8000 --auto
```

#### Method 2: Manual Setup

```bash
git clone https://github.com/YOUR_USERNAME/PoUW.git
cd PoUW
pip install -r requirements.txt
python main.py --role MINER --bootstrap-peers YOUR_VPS_IP:8000
```

#### Method 3: Docker

```bash
docker run -d \
  --name my-pouw-node \
  -p 8001:8000 \
  -e BOOTSTRAP_PEERS="YOUR_VPS_IP:8000" \
  -e NODE_ROLE=MINER \
  your-registry/pouw:latest
```

### Device-Specific Configurations

#### Gaming PC / High-Performance Desktop

```bash
python main.py \
  --role MINER \
  --node-id "gaming_pc_001" \
  --stake 200.0 \
  --bootstrap-peers YOUR_VPS_IP:8000
```

#### Laptop / Mobile Device

```bash
python main.py \
  --role VERIFIER \
  --node-id "laptop_001" \
  --stake 50.0 \
  --bootstrap-peers YOUR_VPS_IP:8000
```

#### Raspberry Pi / IoT Device

```bash
python main.py \
  --role PEER \
  --node-id "pi_001" \
  --stake 10.0 \
  --bootstrap-peers YOUR_VPS_IP:8000
```

#### Cloud Instance

```bash
python main.py \
  --role SUPERVISOR \
  --node-id "aws_node_001" \
  --stake 500.0 \
  --bootstrap-peers YOUR_VPS_IP:8000
```

---

## 📊 Monitoring & Management

### Dashboard Setup

```bash
# Start dashboard on VPS
cd /opt/pouw
python dashboard.py

# Access at: http://YOUR_VPS_IP:8080
```

### Monitoring Commands

```bash
# View network status
curl http://YOUR_VPS_IP:8000/status | jq

# Check connected nodes
curl http://YOUR_VPS_IP:8080/api/nodes | jq

# Monitor container logs
docker-compose logs -f pouw-app
```

### Performance Monitoring

```bash
# System resources
htop
docker stats

# Network connections
netstat -tulpn | grep 8000
ss -tulpn | grep 8000

# Disk usage
df -h
du -sh /opt/pouw/data/
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Connection Refused

```bash
# Check if service is running
docker ps | grep pouw

# Check port binding
netstat -tulpn | grep 8000

# Check firewall
sudo ufw status
sudo ufw allow 8000/tcp
```

#### 2. SSL Certificate Issues

```bash
# Renew certificate
sudo certbot renew

# Check certificate status
sudo certbot certificates
```

#### 3. Container Won't Start

```bash
# Check logs
docker logs pouw-production

# Check configuration
docker-compose config

# Rebuild container
docker-compose build --no-cache
```

#### 4. Python Import Errors

```bash
# Install missing dependencies
pip install -r requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

### Recovery Commands

```bash
# Restart all services
docker-compose -f docker-compose.production.yml restart

# Rebuild from scratch
docker-compose -f docker-compose.production.yml down
docker-compose -f docker-compose.production.yml up -d --build

# Reset data (WARNING: destructive)
docker-compose down -v
rm -rf data/ logs/ cache/
docker-compose up -d
```

---

## 📈 Scaling Your Network

### Multiple Bootstrap Nodes

1. Deploy additional VPS instances
2. Configure load balancer
3. Use DNS round-robin for discovery

### Geographic Distribution

1. Deploy nodes in different regions
2. Configure region-aware routing
3. Set up cross-region VPN mesh

### Performance Optimization

```bash
# Increase container resources
docker-compose -f docker-compose.production.yml up -d --scale pouw-app=3

# Configure nginx load balancing
# Edit nginx/nginx.conf to add multiple upstream servers
```

---

## 🛡️ Security Best Practices

### Network Security

- [ ] Use HTTPS with valid SSL certificates
- [ ] Configure proper firewall rules
- [ ] Regular security updates
- [ ] Monitor for suspicious activity

### Access Control

- [ ] Disable root SSH login
- [ ] Use SSH keys only
- [ ] Configure fail2ban
- [ ] Regular password/key rotation

### Container Security

- [ ] Run containers as non-root user
- [ ] Regular image updates
- [ ] Scan for vulnerabilities
- [ ] Limit container permissions

---

## 📞 Support & Resources

### Documentation

- [Main Documentation](README.md)
- [Network Participation Guide](NETWORK_PARTICIPATION_GUIDE.md)
- [Quick Reference](QUICK_REFERENCE.md)

### Health Checks

- Health: `http://YOUR_VPS_IP/health`
- Status: `http://YOUR_VPS_IP/status`
- Dashboard: `http://YOUR_VPS_IP:8080`

### Useful Commands

```bash
# Check deployment status
./test-deployment.sh --host YOUR_VPS_IP

# Monitor network activity
curl -s http://YOUR_VPS_IP:8080/api/nodes | jq '.nodes | length'

# Get network statistics
curl -s http://YOUR_VPS_IP:8000/status | jq '.peer_count'
```

---

## 🎯 Success Criteria

Your deployment is successful when:

✅ **Health endpoints respond** (`/health`, `/status`)  
✅ **Containers are running** and healthy  
✅ **Network ports are accessible** from external devices  
✅ **SSL certificates work** (if using HTTPS)  
✅ **Dashboard shows network activity** (if running)  
✅ **New devices can join** using bootstrap peer  
✅ **Monitoring systems work** and show metrics

---

## 🚀 You're Ready!

Your PoUW network is now deployed and ready for others to join. Share your VPS details with the community and watch your decentralized ML network grow!

**Bootstrap Peer**: `YOUR_VPS_IP:8000`  
**Dashboard**: `http://YOUR_VPS_IP:8080`  
**Join Command**: `python join_network.py --bootstrap-peer YOUR_VPS_IP:8000`

Happy distributed computing! 🎉
