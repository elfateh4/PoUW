# PoUW Deployment Guide for Hostinger VPS

This guide provides complete instructions for deploying the PoUW (Proof of Useful Work) application to a Hostinger VPS using Docker, Docker Compose, and GitHub Actions.

## Prerequisites

- Hostinger VPS with Ubuntu 20.04+ or similar Linux distribution
- Domain name pointed to your VPS IP
- SSH access to your VPS
- GitHub repository with your PoUW code

## Quick Setup

### 1. VPS Preparation

Connect to your VPS via SSH:

```bash
ssh root@your-vps-ip
```

Download and run the deployment script:

```bash
curl -fsSL https://raw.githubusercontent.com/YOUR_USERNAME/PoUW/main/deploy.sh -o deploy.sh
chmod +x deploy.sh
sudo ./deploy.sh
```

### 2. GitHub Secrets Configuration

In your GitHub repository, go to Settings > Secrets and Variables > Actions, and add these secrets:

- `VPS_HOST`: Your VPS IP address or domain
- `VPS_USERNAME`: SSH username (usually 'root' for Hostinger)
- `VPS_SSH_KEY`: Your private SSH key
- `VPS_PORT`: SSH port (usually 22)
- `GITHUB_TOKEN`: Automatically provided by GitHub
- `SLACK_WEBHOOK_URL`: (Optional) For deployment notifications

### 3. Domain Configuration

Update the nginx configuration with your domain:

```bash
cd /opt/pouw
nano nginx/nginx.conf
# Replace 'your-domain.com' with your actual domain
```

### 4. SSL Certificate Setup

Install Certbot and get SSL certificate:

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## Manual Deployment

### Local Development

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/PoUW.git
cd PoUW
```

2. Run locally with Docker Compose:

```bash
docker-compose up -d
```

3. Access the application at `http://localhost:8000`

### Production Deployment

1. Build and deploy to production:

```bash
docker-compose -f docker-compose.production.yml up -d
```

2. Monitor the deployment:

```bash
docker-compose -f docker-compose.production.yml logs -f
```

## Configuration

### Environment Variables

The application uses these environment variables (configured in `.env.production`):

- `NODE_ROLE`: Type of node (SUPERVISOR, MINER, VERIFIER)
- `NODE_ID`: Unique identifier for the node
- `PORT`: Port number for the application
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `ENVIRONMENT`: Environment type (development, production)

### Docker Compose Services

#### Development (`docker-compose.yml`)

- `supervisor`: Main supervisor node
- `miner1`, `miner2`: Mining nodes
- `verifier`: Verification node
- `nginx`: Load balancer/reverse proxy

#### Production (`docker-compose.production.yml`)

- `pouw-app`: Main application container
- `nginx`: Reverse proxy with SSL termination
- `watchtower`: Automatic container updates

## Monitoring and Maintenance

### Health Checks

Check application health:

```bash
curl http://your-domain.com/health
```

### View Logs

```bash
# Application logs
docker-compose -f docker-compose.production.yml logs pouw-app

# Nginx logs
docker-compose -f docker-compose.production.yml logs nginx

# All services
docker-compose -f docker-compose.production.yml logs
```

### Backup

The deployment script creates an automatic backup script. Run manually:

```bash
cd /opt/pouw
./backup.sh
```

Schedule automatic backups:

```bash
crontab -e
# Add: 0 2 * * * /opt/pouw/backup.sh
```

### Updates

Updates are handled automatically via GitHub Actions when you push to the main branch. Manual update:

```bash
cd /opt/pouw
git pull origin main
docker-compose -f docker-compose.production.yml pull
docker-compose -f docker-compose.production.yml up -d
```

## Troubleshooting

### Common Issues

1. **Port 80/443 already in use**:

   ```bash
   sudo netstat -tulpn | grep :80
   sudo systemctl stop apache2  # If Apache is running
   ```

2. **Docker permission denied**:

   ```bash
   sudo usermod -aG docker $USER
   # Log out and log back in
   ```

3. **SSL certificate issues**:

   ```bash
   sudo certbot renew --dry-run
   ```

4. **Application won't start**:
   ```bash
   docker-compose -f docker-compose.production.yml logs pouw-app
   ```

### Performance Optimization

1. **Increase Docker resources**:

   ```bash
   # Edit /etc/docker/daemon.json
   {
     "storage-driver": "overlay2",
     "log-driver": "json-file",
     "log-opts": {
       "max-size": "10m",
       "max-file": "3"
     }
   }
   ```

2. **Optimize Nginx**:
   ```bash
   # Increase worker connections in nginx.conf
   events {
       worker_connections 2048;
   }
   ```

## Security Considerations

1. **Firewall Configuration**:

   ```bash
   sudo ufw allow ssh
   sudo ufw allow 80/tcp
   sudo ufw allow 443/tcp
   sudo ufw enable
   ```

2. **SSH Security**:

   - Disable root login
   - Use SSH keys only
   - Change default SSH port

3. **Docker Security**:
   - Run containers as non-root user
   - Use official base images
   - Regularly update images

## Monitoring Setup

### Basic Monitoring

1. **System monitoring with htop**:

   ```bash
   sudo apt install htop
   ```

2. **Docker stats**:
   ```bash
   docker stats
   ```

### Advanced Monitoring (Optional)

1. **Prometheus + Grafana**:

   ```bash
   # Add to docker-compose.production.yml
   prometheus:
     image: prom/prometheus
     ports:
       - "9090:9090"

   grafana:
     image: grafana/grafana
     ports:
       - "3000:3000"
   ```

## Support

For issues and questions:

- Check the application logs
- Review the GitHub Actions workflow results
- Create an issue in the GitHub repository

## License

This deployment configuration is part of the PoUW project and follows the same license terms.
