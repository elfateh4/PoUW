# PoUW Hostinger VPS Deployment - Quick Reference

## 🚀 One-Line Deployment

```bash
curl -fsSL https://raw.githubusercontent.com/YOUR_USERNAME/PoUW/main/deploy.sh | sudo bash
```

## 📋 GitHub Secrets Required

```bash
VPS_HOST=your.vps.ip.address
VPS_USERNAME=root
VPS_SSH_KEY="-----BEGIN OPENSSH PRIVATE KEY-----..."
VPS_PORT=22
SLACK_WEBHOOK_URL=https://hooks.slack.com/... (optional)
```

## 🐳 Docker Commands

```bash
# Development (multi-node)
docker-compose up -d

# Production (single optimized service)
docker-compose -f docker-compose.production.yml up -d

# View logs
docker-compose logs -f pouw-app

# Restart service
docker-compose restart pouw-app

# Update and restart
docker-compose pull && docker-compose up -d
```

## 🌐 Nginx Commands

```bash
# Test configuration
nginx -t

# Reload configuration
nginx -s reload

# View access logs
tail -f /var/log/nginx/access.log
```

## 🔐 SSL Setup

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal test
sudo certbot renew --dry-run
```

## 📊 Health Checks

```bash
# Application health
curl http://localhost:8080/health
curl https://your-domain.com/health

# Detailed status
curl http://localhost:8080/status
curl https://your-domain.com/status

# Docker health
docker ps
docker-compose ps
```

## 🔧 Environment Variables

```bash
# Set in .env.production or docker-compose
NODE_ROLE=SUPERVISOR|MINER|VERIFIER
NODE_ID=unique_node_identifier
PORT=8000
LOG_LEVEL=INFO|DEBUG|WARNING|ERROR
HEALTH_CHECK_PORT=8080
```

## 📁 Key Files

```
📁 /opt/pouw/
├── 🐳 Dockerfile
├── 🐳 docker-compose.production.yml
├── 🌐 nginx/nginx.conf
├── 📜 deploy.sh
├── 🐍 main.py
├── 📦 requirements.txt
├── 🔧 .env.production
└── 📊 logs/
```

## 🚨 Troubleshooting Quick Fixes

```bash
# Port conflicts
sudo netstat -tulpn | grep :80
sudo systemctl stop apache2

# Docker permissions
sudo usermod -aG docker $USER
# Log out and back in

# Service restart
docker-compose restart
systemctl restart nginx

# Check logs
docker logs container_name
journalctl -u nginx
```

## 🔄 Update Process

```bash
# Automatic (recommended)
git push origin main  # Triggers GitHub Actions

# Manual
cd /opt/pouw
git pull origin main
docker-compose -f docker-compose.production.yml pull
docker-compose -f docker-compose.production.yml up -d
```

## 📱 Monitoring Commands

```bash
# System resources
htop
df -h
free -m

# Docker stats
docker stats

# Network connections
netstat -tulpn
ss -tulpn

# Service status
systemctl status docker
systemctl status nginx
```

## 🛡️ Security Checklist

- [ ] SSH key authentication enabled
- [ ] Password authentication disabled
- [ ] Firewall configured (ufw)
- [ ] SSL certificates installed
- [ ] Regular updates scheduled
- [ ] Backup strategy implemented
- [ ] Log rotation configured

## 📞 Support Commands

```bash
# Get deployment script
curl -O https://raw.githubusercontent.com/YOUR_USERNAME/PoUW/main/deploy.sh

# Test deployment config
./test-deployment.sh

# View application logs
docker-compose logs --tail=100 pouw-app

# Check GitHub Actions
# Visit: https://github.com/YOUR_USERNAME/PoUW/actions
```

## 🎯 Performance Tuning

```bash
# Docker resource limits (docker-compose.yml)
deploy:
  resources:
    limits:
      memory: 2G
      cpus: '1.0'

# Nginx worker processes (nginx.conf)
worker_processes auto;
worker_connections 2048;

# Docker log rotation (/etc/docker/daemon.json)
{
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
```

---

**🚀 Happy Deploying!**

For detailed instructions, see `HOSTINGER_DEPLOYMENT_COMPLETE.md`
