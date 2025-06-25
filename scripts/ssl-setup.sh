#!/bin/bash

# PoUW SSL Certificate Management Script
# Automated Certbot SSL certificate setup with SSH verification

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to load configuration from Python config system
load_config_from_python() {
    if command -v python3 &> /dev/null && [ -f "../config.py" ]; then
        log "Loading SSL configuration from config.py..."
        
        # Extract SSL configuration variables
        eval $(python3 -c "
import sys
sys.path.insert(0, '..')
try:
    from config import get_config_manager
    config = get_config_manager(environment='production').get_config()
    print(f'DOMAIN=\"{config.deployment.domain}\"')
    print(f'EMAIL=\"{config.deployment.email}\"')
    print(f'ENABLE_SSL=\"{str(config.deployment.enable_ssl).lower()}\"')
    print(f'SSL_STAGING=\"{str(config.deployment.ssl_staging).lower()}\"')
    print(f'SSL_CERT_PATH=\"{config.ssl.cert_path if hasattr(config, \"ssl\") else \"/etc/nginx/ssl/fullchain.pem\"}\"')
    print(f'SSL_KEY_PATH=\"{config.ssl.key_path if hasattr(config, \"ssl\") else \"/etc/nginx/ssl/privkey.pem\"}\"')
except Exception as e:
    print(f'# SSL config loading failed: {e}', file=sys.stderr)
" 2>/dev/null)
        
        if [ "$ENABLE_SSL" = "true" ]; then
            log "SSL enabled in configuration"
            if [ -z "$DOMAIN" ] || [ "$DOMAIN" = "localhost" ]; then
                warn "Domain not configured for SSL, please set POUW_DOMAIN"
            fi
        fi
    fi
}

# Configuration
DOMAIN=""
EMAIL=""
WEBROOT_PATH="/var/www/certbot"
NGINX_CONF_PATH="/etc/nginx/sites-available/pouw"
SSL_DIR="/etc/nginx/ssl"
CERTBOT_DIR="/etc/letsencrypt"

# Load configuration before parsing arguments
load_config_from_python

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

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Check if running as root or with sudo
if [[ $EUID -eq 0 ]]; then
    SUDO=""
else
    SUDO="sudo"
fi

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --domain DOMAIN     Domain name for SSL certificate"
    echo "  -e, --email EMAIL       Email address for Let's Encrypt notifications"
    echo "  -w, --webroot PATH      Webroot path for verification (default: /var/www/certbot)"
    echo "  --dry-run              Test certificate generation without creating actual certificates"
    echo "  --force-renewal        Force certificate renewal even if not expired"
    echo "  --staging              Use Let's Encrypt staging environment for testing"
    echo "  --dns-verification     Use DNS verification instead of HTTP"
    echo "  --help                 Display this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -d example.com -e admin@example.com"
    echo "  $0 -d api.pouw.network -e cert@pouw.network --staging"
    echo "  $0 --force-renewal -d example.com -e admin@example.com"
}

# Function to check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Certbot is installed
    if ! command -v certbot &> /dev/null; then
        log "Installing Certbot..."
        $SUDO apt-get update
        $SUDO apt-get install -y certbot python3-certbot-nginx
    fi
    
    # Check if nginx is installed
    if ! command -v nginx &> /dev/null; then
        error "Nginx is not installed. Please install nginx first."
    fi
    
    # Check domain parameter
    if [ -z "$DOMAIN" ]; then
        error "Domain name is required. Use -d or --domain option."
    fi
    
    # Check email parameter
    if [ -z "$EMAIL" ]; then
        error "Email address is required. Use -e or --email option."
    fi
    
    # Validate domain format
    if [[ ! "$DOMAIN" =~ ^[a-zA-Z0-9][a-zA-Z0-9-]{0,61}[a-zA-Z0-9]\.[a-zA-Z]{2,}$ ]]; then
        if [[ ! "$DOMAIN" =~ ^[a-zA-Z0-9-]{1,63}\.[a-zA-Z]{2,}$ ]]; then
            warn "Domain format might be invalid: $DOMAIN"
        fi
    fi
    
    # Validate email format
    if [[ ! "$EMAIL" =~ ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$ ]]; then
        error "Invalid email format: $EMAIL"
    fi
}

# Function to setup webroot directory
setup_webroot() {
    log "Setting up webroot directory..."
    
    $SUDO mkdir -p "$WEBROOT_PATH"
    $SUDO chown -R www-data:www-data "$WEBROOT_PATH"
    $SUDO chmod -R 755 "$WEBROOT_PATH"
    
    # Create test file to verify webroot access
    echo "Certbot verification test" | $SUDO tee "$WEBROOT_PATH/test.txt" > /dev/null
}

# Function to configure nginx for HTTP verification
configure_nginx_http_verification() {
    log "Configuring Nginx for HTTP verification..."
    
    # Create temporary nginx configuration for verification
    cat << EOF | $SUDO tee /etc/nginx/sites-available/certbot-verification
server {
    listen 80;
    server_name $DOMAIN www.$DOMAIN;
    
    location /.well-known/acme-challenge/ {
        root $WEBROOT_PATH;
        try_files \$uri =404;
    }
    
    location / {
        return 301 https://\$server_name\$request_uri;
    }
}
EOF
    
    # Enable the verification site
    $SUDO ln -sf /etc/nginx/sites-available/certbot-verification /etc/nginx/sites-enabled/
    
    # Test nginx configuration
    $SUDO nginx -t
    
    # Reload nginx
    $SUDO systemctl reload nginx
}

# Function to test domain accessibility
test_domain_accessibility() {
    log "Testing domain accessibility..."
    
    # Test if domain resolves to current server
    DOMAIN_IP=$(dig +short "$DOMAIN" @8.8.8.8)
    SERVER_IP=$(curl -s http://checkip.amazonaws.com/)
    
    if [ "$DOMAIN_IP" != "$SERVER_IP" ]; then
        warn "Domain $DOMAIN resolves to $DOMAIN_IP but server IP is $SERVER_IP"
        warn "This might cause certificate verification to fail"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            error "Aborting due to DNS mismatch"
        fi
    else
        log "Domain DNS verification passed"
    fi
    
    # Test HTTP accessibility
    if curl -s -o /dev/null -w "%{http_code}" "http://$DOMAIN/.well-known/acme-challenge/test" | grep -q "404"; then
        log "HTTP verification endpoint is accessible"
    else
        warn "HTTP verification endpoint might not be accessible"
    fi
}

# Function to generate SSL certificate
generate_certificate() {
    log "Generating SSL certificate for $DOMAIN..."
    
    # Build certbot command
    CERTBOT_CMD="certbot certonly"
    CERTBOT_CMD="$CERTBOT_CMD --webroot"
    CERTBOT_CMD="$CERTBOT_CMD --webroot-path=$WEBROOT_PATH"
    CERTBOT_CMD="$CERTBOT_CMD --email $EMAIL"
    CERTBOT_CMD="$CERTBOT_CMD --agree-tos"
    CERTBOT_CMD="$CERTBOT_CMD --no-eff-email"
    CERTBOT_CMD="$CERTBOT_CMD --domains $DOMAIN,www.$DOMAIN"
    
    # Add staging flag if specified
    if [ "$STAGING" = "true" ]; then
        CERTBOT_CMD="$CERTBOT_CMD --staging"
        log "Using Let's Encrypt staging environment"
    fi
    
    # Add dry-run flag if specified
    if [ "$DRY_RUN" = "true" ]; then
        CERTBOT_CMD="$CERTBOT_CMD --dry-run"
        log "Running in dry-run mode"
    fi
    
    # Add force renewal if specified
    if [ "$FORCE_RENEWAL" = "true" ]; then
        CERTBOT_CMD="$CERTBOT_CMD --force-renewal"
        log "Forcing certificate renewal"
    fi
    
    # Execute certbot command
    log "Executing: $SUDO $CERTBOT_CMD"
    if $SUDO $CERTBOT_CMD; then
        if [ "$DRY_RUN" != "true" ]; then
            log "SSL certificate generated successfully!"
        else
            log "Dry run completed successfully!"
        fi
    else
        error "Failed to generate SSL certificate"
    fi
}

# Function to configure nginx with SSL
configure_nginx_ssl() {
    if [ "$DRY_RUN" = "true" ]; then
        log "Skipping nginx SSL configuration (dry-run mode)"
        return
    fi
    
    log "Configuring Nginx with SSL..."
    
    # Create SSL directory
    $SUDO mkdir -p "$SSL_DIR"
    
    # Copy certificates to nginx ssl directory
    $SUDO cp "$CERTBOT_DIR/live/$DOMAIN/fullchain.pem" "$SSL_DIR/"
    $SUDO cp "$CERTBOT_DIR/live/$DOMAIN/privkey.pem" "$SSL_DIR/"
    
    # Update nginx configuration
    cat << EOF | $SUDO tee "$NGINX_CONF_PATH"
# PoUW Production Nginx Configuration with SSL
upstream pouw_backend {
    server pouw-app:8000;
    keepalive 32;
}

# Rate limiting
limit_req_zone \$binary_remote_addr zone=api:10m rate=10r/s;
limit_req_zone \$binary_remote_addr zone=auth:10m rate=5r/s;

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name $DOMAIN www.$DOMAIN;
    
    location /.well-known/acme-challenge/ {
        root $WEBROOT_PATH;
        try_files \$uri =404;
    }
    
    location / {
        return 301 https://\$server_name\$request_uri;
    }
}

# HTTPS configuration
server {
    listen 443 ssl http2;
    server_name $DOMAIN www.$DOMAIN;

    # SSL configuration
    ssl_certificate $SSL_DIR/fullchain.pem;
    ssl_certificate_key $SSL_DIR/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-SHA256:ECDHE-RSA-AES256-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # API endpoints
    location /api/ {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://pouw_backend;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://pouw_backend/health;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # Dashboard
    location / {
        proxy_pass http://pouw_backend;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF
    
    # Test nginx configuration
    if $SUDO nginx -t; then
        log "Nginx configuration test passed"
        
        # Reload nginx
        $SUDO systemctl reload nginx
        log "Nginx reloaded with SSL configuration"
    else
        error "Nginx configuration test failed"
    fi
}

# Function to setup automatic renewal
setup_auto_renewal() {
    if [ "$DRY_RUN" = "true" ]; then
        log "Skipping auto-renewal setup (dry-run mode)"
        return
    fi
    
    log "Setting up automatic certificate renewal..."
    
    # Create renewal hook script
    cat << 'EOF' | $SUDO tee /etc/letsencrypt/renewal-hooks/deploy/pouw-renewal.sh
#!/bin/bash

# PoUW SSL Certificate Renewal Hook
# This script runs after successful certificate renewal

DOMAIN="$RENEWED_DOMAINS"
SSL_DIR="/etc/nginx/ssl"
CERTBOT_DIR="/etc/letsencrypt"

# Copy new certificates to nginx directory
cp "$CERTBOT_DIR/live/$DOMAIN/fullchain.pem" "$SSL_DIR/"
cp "$CERTBOT_DIR/live/$DOMAIN/privkey.pem" "$SSL_DIR/"

# Reload nginx
systemctl reload nginx

# Log renewal
echo "$(date): SSL certificate renewed for $DOMAIN" >> /var/log/pouw-ssl-renewal.log
EOF
    
    $SUDO chmod +x /etc/letsencrypt/renewal-hooks/deploy/pouw-renewal.sh
    
    # Test automatic renewal
    log "Testing automatic renewal..."
    if $SUDO certbot renew --dry-run; then
        log "Automatic renewal test passed"
    else
        warn "Automatic renewal test failed"
    fi
    
    # Check if cron job already exists
    if $SUDO crontab -l 2>/dev/null | grep -q "certbot renew"; then
        log "Certbot renewal cron job already exists"
    else
        # Add cron job for automatic renewal
        (
            $SUDO crontab -l 2>/dev/null || true
            echo "0 12 * * * /usr/bin/certbot renew --quiet"
        ) | $SUDO crontab -
        log "Added automatic renewal cron job"
    fi
}

# Function to verify SSL certificate
verify_ssl_certificate() {
    if [ "$DRY_RUN" = "true" ]; then
        log "Skipping SSL verification (dry-run mode)"
        return
    fi
    
    log "Verifying SSL certificate..."
    
    # Check certificate validity
    if echo | openssl s_client -servername "$DOMAIN" -connect "$DOMAIN:443" 2>/dev/null | openssl x509 -noout -dates; then
        log "SSL certificate is valid"
    else
        warn "SSL certificate verification failed"
    fi
    
    # Test HTTPS connectivity
    if curl -s -I "https://$DOMAIN/health" | grep -q "200 OK"; then
        log "HTTPS endpoint is accessible"
    else
        warn "HTTPS endpoint test failed"
    fi
}

# Function to display certificate information
show_certificate_info() {
    if [ "$DRY_RUN" = "true" ]; then
        return
    fi
    
    log "Certificate Information:"
    echo
    $SUDO certbot certificates | grep -A 10 "$DOMAIN" || true
    echo
    
    if [ -f "$CERTBOT_DIR/live/$DOMAIN/fullchain.pem" ]; then
        echo "Certificate details:"
        openssl x509 -in "$CERTBOT_DIR/live/$DOMAIN/fullchain.pem" -text -noout | grep -E "(Subject:|Issuer:|Not Before:|Not After :)"
    fi
}

# Function to cleanup temporary files
cleanup() {
    log "Cleaning up temporary files..."
    
    # Remove verification site if it exists
    if [ -f "/etc/nginx/sites-enabled/certbot-verification" ]; then
        $SUDO rm -f /etc/nginx/sites-enabled/certbot-verification
        $SUDO systemctl reload nginx
    fi
    
    # Remove test file
    if [ -f "$WEBROOT_PATH/test.txt" ]; then
        $SUDO rm -f "$WEBROOT_PATH/test.txt"
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
        -w|--webroot)
            WEBROOT_PATH="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --force-renewal)
            FORCE_RENEWAL="true"
            shift
            ;;
        --staging)
            STAGING="true"
            shift
            ;;
        --dns-verification)
            DNS_VERIFICATION="true"
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Main execution flow
main() {
    log "Starting SSL certificate setup for PoUW..."
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    check_prerequisites
    setup_webroot
    configure_nginx_http_verification
    test_domain_accessibility
    generate_certificate
    configure_nginx_ssl
    setup_auto_renewal
    verify_ssl_certificate
    show_certificate_info
    
    log "SSL certificate setup completed successfully!"
    echo
    info "Your PoUW deployment is now accessible at: https://$DOMAIN"
    info "Certificate will auto-renew before expiration"
    echo
}

# Execute main function
main "$@"
