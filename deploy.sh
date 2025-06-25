#!/bin/bash

# PoUW VPS Deployment Script for Hostinger
# This script sets up the production environment on your VPS with SSL automation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting PoUW deployment on Hostinger VPS...${NC}"

# Validate configuration before deployment
validate_configuration

# Configuration - Load from environment or use defaults
APP_DIR="/opt/pouw"
DOCKER_COMPOSE_FILE="docker-compose.production.yml"
BACKUP_DIR="/opt/pouw-backups"
CONFIG_ENV_FILE=""
DOMAIN=""
EMAIL=""
SKIP_SSL="false"
SSL_STAGING="false"

# Load configuration from Python config system if available
load_config_from_python() {
    if command -v python3 &> /dev/null && [ -f "config.py" ]; then
        log "Loading configuration from config.py..."
        
        # Extract configuration variables
        eval $(python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from config import get_config_manager
    config = get_config_manager(environment='production').get_config()
    print(f'DOMAIN=\"{config.deployment.domain}\"')
    print(f'EMAIL=\"{config.deployment.email}\"')
    print(f'VPS_IP=\"{config.deployment.vps_ip}\"')
    print(f'ENABLE_SSL=\"{str(config.deployment.enable_ssl).lower()}\"')
    print(f'SSL_STAGING=\"{str(config.deployment.ssl_staging).lower()}\"')
    print(f'GITHUB_REPO=\"{config.deployment.github_repo}\"')
    print(f'NODE_ID=\"{config.node.node_id}\"')
    print(f'NODE_ROLE=\"{config.node.role}\"')
    print(f'NODE_PORT=\"{config.node.port}\"')
except Exception as e:
    print(f'# Config loading failed: {e}', file=sys.stderr)
")
        
        if [ "$ENABLE_SSL" = "true" ] && [ "$SKIP_SSL" != "true" ]; then
            log "SSL enabled in configuration"
        fi
    fi
}

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --domain DOMAIN       Domain name for SSL certificate"
    echo "  -e, --email EMAIL         Email address for Let's Encrypt notifications"
    echo "  -c, --config CONFIG_FILE  Use specific environment config file"
    echo "  --skip-ssl                Skip SSL certificate setup"
    echo "  --ssl-staging             Use Let's Encrypt staging environment"
    echo "  --help                    Display this help message"
    echo ""
    echo "Environment config files:"
    echo "  .env.production          Production configuration"
    echo "  .env.development         Development configuration"
    echo "  .env                     Default configuration"
    echo ""
    echo "Examples:"
    echo "  $0 -d api.pouw.network -e admin@pouw.network"
    echo "  $0 -c .env.production"
    echo "  $0 --skip-ssl  # Deploy without SSL"
    echo "  $0 -d test.pouw.network -e admin@pouw.network --ssl-staging"
}

# Validate configuration before deployment
validate_configuration() {
    log "Validating configuration..."
    
    if [ -f "scripts/validate_config.py" ]; then
        if python3 scripts/validate_config.py --environment production; then
            log "✅ Configuration validation passed"
        else
            error "❌ Configuration validation failed"
            error "Please fix configuration issues before deploying"
            echo "Run: python3 scripts/validate_config.py --environment production --fix-suggestions"
            exit 1
        fi
    else
        warn "Configuration validator not found, skipping validation"
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--domain)
            DOMAIN="$2"
            shift 2
            ;;
        -e|--email)
            EMAIL="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_ENV_FILE="$2"
            shift 2
            ;;
        --skip-ssl)
            SKIP_SSL="true"
            shift
            ;;
        --ssl-staging)
            SSL_STAGING="true"
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            warn "Unknown option: $1"
            shift
            ;;
    esac
done

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
    lsb-release \
    nginx \
    certbot \
    python3-certbot-nginx

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
setup_ssl_certificates() {
    if [ "$SKIP_SSL" = "true" ]; then
        log "Skipping SSL certificate setup..."
        setup_self_signed_certificates
        return
    fi
    
    if [ -z "$DOMAIN" ]; then
        warn "No domain specified. Setting up self-signed certificates..."
        setup_self_signed_certificates
        return
    fi
    
    if [ -z "$EMAIL" ]; then
        warn "No email specified for Let's Encrypt. Please provide email with -e option"
        read -p "Enter email address for Let's Encrypt notifications: " EMAIL
        if [ -z "$EMAIL" ]; then
            warn "No email provided. Setting up self-signed certificates..."
            setup_self_signed_certificates
            return
        fi
    fi
    
    log "Setting up SSL certificates for domain: $DOMAIN"
    
    # Make sure ssl setup script is executable
    chmod +x scripts/ssl-setup.sh
    
    # Build SSL setup command
    SSL_CMD="./scripts/ssl-setup.sh -d $DOMAIN -e $EMAIL"
    
    if [ "$SSL_STAGING" = "true" ]; then
        SSL_CMD="$SSL_CMD --staging"
        log "Using Let's Encrypt staging environment"
    fi
    
    # Run SSL setup
    if $SSL_CMD; then
        log "SSL certificates configured successfully!"
    else
        warn "SSL certificate setup failed. Falling back to self-signed certificates..."
        setup_self_signed_certificates
    fi
}

setup_self_signed_certificates() {
    log "Setting up self-signed certificates for testing..."
    
    $SUDO mkdir -p /etc/nginx/ssl
    
    if [ ! -f "/etc/nginx/ssl/fullchain.pem" ]; then
        $SUDO openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout /etc/nginx/ssl/privkey.pem \
            -out /etc/nginx/ssl/fullchain.pem \
            -subj "/C=US/ST=State/L=City/O=PoUW/OU=Development/CN=localhost"
        
        log "Self-signed certificates created"
    else
        log "Self-signed certificates already exist"
    fi
    
    # Create basic nginx configuration for self-signed certificates
    cat > nginx/nginx.conf << 'EOF'
# PoUW Development Nginx Configuration (Self-Signed SSL)
upstream pouw_backend {
    server pouw-app:8000;
    keepalive 32;
}

# Rate limiting
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

server {
    listen 80;
    server_name _;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name _;

    # Self-signed SSL configuration
    ssl_certificate /etc/nginx/ssl/fullchain.pem;
    ssl_certificate_key /etc/nginx/ssl/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;

    # API endpoints
    location /api/ {
        limit_req zone=api burst=20 nodelay;
        proxy_pass http://pouw_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Health check
    location /health {
        proxy_pass http://pouw_backend/health;
        proxy_set_header Host $host;
    }

    # Dashboard
    location / {
        proxy_pass http://pouw_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF
}

setup_ssl_certificates

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
