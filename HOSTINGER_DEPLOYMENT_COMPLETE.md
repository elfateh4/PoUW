# PoUW Hostinger VPS Deployment - Complete Guide

## 🚀 Quick Deployment Summary

This guide provides everything needed to deploy your PoUW (Proof of Useful Work) blockchain application to a Hostinger VPS using modern DevOps practices.

### What We've Created

1. **🐳 Docker Configuration**

   - `Dockerfile` - Production-ready container
   - `docker-compose.yml` - Multi-node development setup
   - `docker-compose.production.yml` - Production deployment
   - `.dockerignore` - Optimized build context

2. **⚙️ GitHub Actions CI/CD**

   - `.github/workflows/deploy.yml` - Automated deployment pipeline
   - Tests, builds, pushes to registry, deploys to VPS
   - Health checks and notifications

3. **🌐 Nginx Configuration**

   - `nginx/nginx.conf` - Reverse proxy with SSL termination
   - Load balancing, rate limiting, security headers
   - Health check integration

4. **📜 Deployment Scripts**

   - `deploy.sh` - Automated VPS setup script
   - System dependencies, Docker installation
   - SSL certificates, firewall, monitoring

5. **📊 Health Monitoring**
   - HTTP health endpoints in `main.py`
   - Docker health checks
   - System monitoring setup

## 🎯 Deployment Steps

### 1. Prerequisites

- Hostinger VPS with Ubuntu 20.04+
- Domain name pointed to VPS IP
- GitHub repository with your code
- SSH access to VPS

### 2. VPS Setup

**Option A: Automated Setup**

```bash
# On your VPS
curl -fsSL https://raw.githubusercontent.com/YOUR_USERNAME/PoUW/main/deploy.sh -o deploy.sh
chmod +x deploy.sh
sudo ./deploy.sh
```

**Option B: Manual Setup**

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Create app directory
sudo mkdir -p /opt/pouw
cd /opt/pouw

# Clone repository
git clone https://github.com/YOUR_USERNAME/PoUW.git .
```

### 3. GitHub Configuration

Add these secrets in your GitHub repository settings:

| Secret Name         | Description                    | Example                               |
| ------------------- | ------------------------------ | ------------------------------------- |
| `VPS_HOST`          | Your VPS IP or domain          | `123.456.789.0`                       |
| `VPS_USERNAME`      | SSH username                   | `root`                                |
| `VPS_SSH_KEY`       | Private SSH key                | `-----BEGIN OPENSSH PRIVATE KEY-----` |
| `VPS_PORT`          | SSH port (optional)            | `22`                                  |
| `SLACK_WEBHOOK_URL` | Slack notifications (optional) | `https://hooks.slack.com/...`         |

### 4. SSL Certificate Setup

**Let's Encrypt (Recommended)**

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

**Self-Signed (Testing)**

```bash
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /opt/pouw/nginx/ssl/privkey.pem \
  -out /opt/pouw/nginx/ssl/fullchain.pem
```

### 5. Deploy Application

**Automatic Deployment**

- Push to main branch triggers GitHub Actions
- Tests → Build → Deploy → Health Check

**Manual Deployment**

```bash
cd /opt/pouw
docker-compose -f docker-compose.production.yml up -d
```

## 🔧 Configuration

### Environment Variables

Create `/opt/pouw/.env.production`:

```bash
# Node Configuration
NODE_ROLE=SUPERVISOR
NODE_ID=vps_supervisor_001
PORT=8000
LOG_LEVEL=INFO
ENVIRONMENT=production

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret-here

# Health Check
HEALTH_CHECK_HOST=0.0.0.0
HEALTH_CHECK_PORT=8080
```

### Nginx Configuration

Update `nginx/nginx.conf`:

- Replace `your-domain.com` with your actual domain
- Configure SSL certificate paths
- Adjust rate limiting if needed

### Docker Compose Services

**Production Services:**

- `pouw-app`: Main application container
- `nginx`: Reverse proxy with SSL
- `watchtower`: Automatic updates

**Development Services:**

- `supervisor`: Blockchain supervisor node
- `miner1`, `miner2`: Mining nodes
- `verifier`: Verification node

## 📈 Monitoring & Maintenance

### Health Checks

```bash
# Application health
curl https://your-domain.com/health

# Detailed status
curl https://your-domain.com/status

# Service status
docker-compose -f docker-compose.production.yml ps
```

### Logs

```bash
# Application logs
docker-compose -f docker-compose.production.yml logs pouw-app

# All services
docker-compose -f docker-compose.production.yml logs

# Follow live logs
docker-compose -f docker-compose.production.yml logs -f
```

### Backups

```bash
# Manual backup
cd /opt/pouw
./backup.sh

# Schedule automatic backups
crontab -e
# Add: 0 2 * * * /opt/pouw/backup.sh
```

### Updates

```bash
# Automatic via GitHub Actions (recommended)
git push origin main

# Manual update
cd /opt/pouw
git pull origin main
docker-compose -f docker-compose.production.yml pull
docker-compose -f docker-compose.production.yml up -d
```

## 🛡️ Security

### Firewall Setup

```bash
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### SSH Hardening

```bash
# Disable root login (recommended)
sudo nano /etc/ssh/sshd_config
# Set: PermitRootLogin no
# Set: PasswordAuthentication no

sudo systemctl restart ssh
```

### Docker Security

- Containers run as non-root user
- Resource limits configured
- Regular image updates via Watchtower

## 🚨 Troubleshooting

### Common Issues

**Port 80/443 already in use**

```bash
sudo netstat -tulpn | grep :80
sudo systemctl stop apache2  # If Apache is running
```

**Docker permission denied**

```bash
sudo usermod -aG docker $USER
# Log out and log back in
```

**SSL certificate renewal**

```bash
sudo certbot renew --dry-run
```

**Application won't start**

```bash
docker-compose -f docker-compose.production.yml logs pouw-app
```

### Performance Tuning

**Increase Docker resources**

```json
// /etc/docker/daemon.json
{
  "storage-driver": "overlay2",
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
```

**Optimize Nginx**

```nginx
# nginx.conf
events {
    worker_connections 2048;
}
```

## 📋 File Structure Summary

```
/opt/pouw/
├── docker-compose.production.yml  # Production services
├── docker-compose.yml             # Development services
├── Dockerfile                     # Application container
├── .dockerignore                  # Build optimization
├── deploy.sh                      # Automated setup script
├── main.py                        # Application with health endpoints
├── requirements.txt               # Python dependencies
├── nginx/
│   ├── nginx.conf                 # Reverse proxy config
│   └── ssl/                       # SSL certificates
├── .github/workflows/
│   └── deploy.yml                 # CI/CD pipeline
└── logs/                          # Application logs
```

## 🎉 Success Verification

After deployment, verify everything is working:

1. **Application Health**: `curl https://your-domain.com/health`
2. **SSL Certificate**: Check browser for green lock
3. **Node Status**: `curl https://your-domain.com/status`
4. **Docker Services**: `docker-compose ps`
5. **Logs**: Check for any errors in application logs

## 🆘 Support

If you encounter issues:

1. Check application logs: `docker-compose logs pouw-app`
2. Verify GitHub Actions: Check workflow results
3. Test health endpoints: `curl http://localhost:8080/health`
4. Review nginx configuration: `nginx -t`
5. Check firewall: `sudo ufw status`

## 📚 Next Steps

1. **Custom Domain**: Point your domain to the VPS IP
2. **Monitoring**: Set up Prometheus/Grafana for advanced monitoring
3. **Scaling**: Add more nodes using the multi-node docker-compose.yml
4. **Security**: Implement additional security measures
5. **Backup Strategy**: Set up automated offsite backups

Your PoUW blockchain application is now production-ready on Hostinger VPS! 🚀
