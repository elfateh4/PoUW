#!/bin/bash

# PoUW VPS Deployment Script for Hostinger
# This script sets up the production environment on your VPS

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting PoUW deployment on Hostinger VPS...${NC}"

# Configuration
APP_DIR="/opt/pouw"
DOCKER_COMPOSE_FILE="docker-compose.production.yml"
BACKUP_DIR="/opt/pouw-backups"

# Function to log messages
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Check if running as root or with sudo
if [[ $EUID -eq 0 ]]; then
    SUDO=""
else
    SUDO="sudo"
fi

# Update system packages
log "Updating system packages..."
$SUDO apt-get update
$SUDO apt-get upgrade -y

# Install required packages
log "Installing required packages..."
$SUDO apt-get install -y \
    curl \
    git \
    unzip \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release

# Install Docker if not already installed
if ! command -v docker &> /dev/null; then
    log "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    $SUDO sh get-docker.sh
    $SUDO usermod -aG docker $USER
    rm get-docker.sh
else
    log "Docker is already installed"
fi

# Install Docker Compose if not already installed
if ! command -v docker-compose &> /dev/null; then
    log "Installing Docker Compose..."
    $SUDO curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    $SUDO chmod +x /usr/local/bin/docker-compose
else
    log "Docker Compose is already installed"
fi

# Create application directory
log "Setting up application directory..."
$SUDO mkdir -p $APP_DIR
$SUDO mkdir -p $BACKUP_DIR
$SUDO mkdir -p $APP_DIR/nginx/ssl
$SUDO mkdir -p $APP_DIR/logs
$SUDO mkdir -p $APP_DIR/data
$SUDO mkdir -p $APP_DIR/cache

# Set permissions
$SUDO chown -R $USER:$USER $APP_DIR
$SUDO chown -R $USER:$USER $BACKUP_DIR

# Clone or update repository
cd $APP_DIR
if [ -d ".git" ]; then
    log "Updating existing repository..."
    git pull origin main
else
    log "Cloning repository..."
    git clone https://github.com/YOUR_USERNAME/PoUW.git .
fi

# Create environment file
log "Creating environment configuration..."
cat > .env.production << EOF
# PoUW Production Environment Configuration
NODE_ROLE=SUPERVISOR
NODE_ID=vps_supervisor_$(date +%s)
PORT=8000
LOG_LEVEL=INFO
ENVIRONMENT=production
PYTHONUNBUFFERED=1

# Security
SECRET_KEY=$(openssl rand -hex 32)
JWT_SECRET=$(openssl rand -hex 32)

# Database (if needed)
# DATABASE_URL=postgresql://user:password@localhost:5432/pouw

# External services
# REDIS_URL=redis://localhost:6379

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
EOF

# Set up SSL certificates (Let's Encrypt)
log "Setting up SSL certificates..."
if [ ! -f "nginx/ssl/fullchain.pem" ]; then
    warn "SSL certificates not found. Setting up self-signed certificates for testing..."
    $SUDO openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout nginx/ssl/privkey.pem \
        -out nginx/ssl/fullchain.pem \
        -subj "/C=US/ST=State/L=City/O=Organization/OU=OrgUnit/CN=localhost"
    
    echo "To set up Let's Encrypt certificates, run:"
    echo "sudo certbot --nginx -d your-domain.com"
fi

# Create backup script
log "Creating backup script..."
cat > backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/opt/pouw-backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="pouw_backup_$DATE"

# Create backup directory
mkdir -p "$BACKUP_DIR/$BACKUP_NAME"

# Backup application data
docker-compose -f docker-compose.production.yml exec -T pouw-app tar czf - /app/data /app/logs | cat > "$BACKUP_DIR/$BACKUP_NAME/app_data.tar.gz"

# Backup configuration
cp -r nginx "$BACKUP_DIR/$BACKUP_NAME/"
cp .env.production "$BACKUP_DIR/$BACKUP_NAME/"
cp docker-compose.production.yml "$BACKUP_DIR/$BACKUP_NAME/"

# Compress backup
cd "$BACKUP_DIR"
tar czf "$BACKUP_NAME.tar.gz" "$BACKUP_NAME"
rm -rf "$BACKUP_NAME"

# Keep only last 7 backups
ls -t *.tar.gz | tail -n +8 | xargs -r rm

echo "Backup completed: $BACKUP_DIR/$BACKUP_NAME.tar.gz"
EOF

chmod +x backup.sh

# Create systemd service for monitoring
log "Creating systemd service..."
$SUDO tee /etc/systemd/system/pouw.service > /dev/null << EOF
[Unit]
Description=PoUW Application
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$APP_DIR
ExecStart=/usr/local/bin/docker-compose -f docker-compose.production.yml up -d
ExecStop=/usr/local/bin/docker-compose -f docker-compose.production.yml down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
$SUDO systemctl daemon-reload
$SUDO systemctl enable pouw.service

# Set up log rotation
log "Setting up log rotation..."
$SUDO tee /etc/logrotate.d/pouw > /dev/null << EOF
$APP_DIR/logs/*.log {
    daily
    missingok
    rotate 7
    compress
    notifempty
    create 644 $USER $USER
    postrotate
        docker-compose -f $APP_DIR/docker-compose.production.yml restart pouw-app
    endscript
}
EOF

# Set up firewall (UFW)
log "Configuring firewall..."
if command -v ufw &> /dev/null; then
    $SUDO ufw allow ssh
    $SUDO ufw allow 80/tcp
    $SUDO ufw allow 443/tcp
    $SUDO ufw --force enable
fi

# Build and start services
log "Building and starting services..."
docker-compose -f $DOCKER_COMPOSE_FILE build
docker-compose -f $DOCKER_COMPOSE_FILE up -d

# Wait for services to start
log "Waiting for services to start..."
sleep 30

# Health check
log "Performing health check..."
if curl -f http://localhost/health; then
    log "✅ Deployment successful! PoUW is running."
else
    warn "Health check failed. Check the logs:"
    docker-compose -f $DOCKER_COMPOSE_FILE logs --tail=50
fi

# Display status
log "Service status:"
docker-compose -f $DOCKER_COMPOSE_FILE ps

log "📋 Post-deployment steps:"
echo "1. Update nginx/nginx.conf with your domain name"
echo "2. Set up Let's Encrypt: sudo certbot --nginx -d your-domain.com"
echo "3. Configure DNS to point to this server"
echo "4. Set up monitoring and alerting"
echo "5. Schedule regular backups: crontab -e"
echo "   Add: 0 2 * * * /opt/pouw/backup.sh"

log "🚀 PoUW deployment completed successfully!"
