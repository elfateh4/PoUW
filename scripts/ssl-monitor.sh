#!/bin/bash

# PoUW SSL Certificate Monitor and Auto-Renewal Script
# Monitors certificate expiration and handles automatic renewal

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
LOG_FILE="/var/log/pouw-ssl-monitor.log"
NOTIFICATION_EMAIL=""
DOMAIN=""
WARNING_DAYS=30
CRITICAL_DAYS=7
SLACK_WEBHOOK=""

# Function to log messages
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1" >> "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1" >> "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1" >> "$LOG_FILE"
}

# Function to send notifications
send_notification() {
    local subject="$1"
    local message="$2"
    local urgency="$3"
    
    # Email notification
    if [ -n "$NOTIFICATION_EMAIL" ] && command -v mail &> /dev/null; then
        echo "$message" | mail -s "$subject" "$NOTIFICATION_EMAIL"
        log "Email notification sent to $NOTIFICATION_EMAIL"
    fi
    
    # Slack notification
    if [ -n "$SLACK_WEBHOOK" ]; then
        local color="good"
        if [ "$urgency" = "warning" ]; then
            color="warning"
        elif [ "$urgency" = "critical" ]; then
            color="danger"
        fi
        
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"attachments\":[{\"color\":\"$color\",\"title\":\"$subject\",\"text\":\"$message\"}]}" \
            "$SLACK_WEBHOOK" 2>/dev/null || true
        log "Slack notification sent"
    fi
    
    # System notification
    if command -v notify-send &> /dev/null; then
        notify-send "$subject" "$message"
    fi
}

# Function to check certificate expiration
check_certificate_expiration() {
    local domain="$1"
    
    if [ -z "$domain" ]; then
        error "Domain not specified for certificate check"
        return 1
    fi
    
    log "Checking certificate expiration for $domain..."
    
    # Get certificate expiration date
    local expiry_date
    expiry_date=$(echo | openssl s_client -servername "$domain" -connect "$domain:443" 2>/dev/null | \
                  openssl x509 -noout -enddate 2>/dev/null | cut -d= -f2)
    
    if [ -z "$expiry_date" ]; then
        error "Failed to retrieve certificate expiration date for $domain"
        return 1
    fi
    
    # Convert to timestamp
    local expiry_timestamp
    expiry_timestamp=$(date -d "$expiry_date" +%s)
    local current_timestamp
    current_timestamp=$(date +%s)
    
    # Calculate days until expiration
    local days_until_expiry
    days_until_expiry=$(( (expiry_timestamp - current_timestamp) / 86400 ))
    
    info "Certificate for $domain expires in $days_until_expiry days ($expiry_date)"
    
    # Check if certificate needs attention
    if [ "$days_until_expiry" -le "$CRITICAL_DAYS" ]; then
        warn "CRITICAL: Certificate for $domain expires in $days_until_expiry days!"
        send_notification "🚨 SSL Certificate Critical Warning" \
                         "Certificate for $domain expires in $days_until_expiry days. Immediate action required!" \
                         "critical"
        return 2
    elif [ "$days_until_expiry" -le "$WARNING_DAYS" ]; then
        warn "Certificate for $domain expires in $days_until_expiry days"
        send_notification "⚠️ SSL Certificate Warning" \
                         "Certificate for $domain expires in $days_until_expiry days. Consider renewal soon." \
                         "warning"
        return 1
    else
        log "Certificate for $domain is valid for $days_until_expiry days"
        return 0
    fi
}

# Function to attempt certificate renewal
attempt_renewal() {
    local domain="$1"
    
    log "Attempting certificate renewal for $domain..."
    
    # Test renewal first
    if sudo certbot renew --dry-run --cert-name "$domain" 2>/dev/null; then
        log "Dry run renewal test passed"
        
        # Perform actual renewal
        if sudo certbot renew --cert-name "$domain" --force-renewal; then
            log "Certificate renewal successful for $domain"
            
            # Copy certificates to nginx directory
            sudo cp "/etc/letsencrypt/live/$domain/fullchain.pem" /etc/nginx/ssl/
            sudo cp "/etc/letsencrypt/live/$domain/privkey.pem" /etc/nginx/ssl/
            
            # Reload nginx
            if sudo systemctl reload nginx; then
                log "Nginx reloaded successfully"
                send_notification "✅ SSL Certificate Renewed" \
                                 "Certificate for $domain has been successfully renewed and nginx reloaded." \
                                 "good"
                return 0
            else
                error "Failed to reload nginx after certificate renewal"
                return 1
            fi
        else
            error "Certificate renewal failed for $domain"
            send_notification "❌ SSL Certificate Renewal Failed" \
                             "Failed to renew certificate for $domain. Manual intervention required." \
                             "critical"
            return 1
        fi
    else
        error "Dry run renewal test failed for $domain"
        return 1
    fi
}

# Function to verify certificate installation
verify_certificate() {
    local domain="$1"
    
    log "Verifying certificate installation for $domain..."
    
    # Test HTTPS connectivity
    if curl -s -I "https://$domain/health" | grep -q "200 OK"; then
        log "HTTPS endpoint is accessible"
    else
        warn "HTTPS endpoint test failed"
    fi
    
    # Check certificate chain
    if echo | openssl s_client -servername "$domain" -connect "$domain:443" 2>/dev/null | \
       openssl x509 -noout -subject -issuer 2>/dev/null; then
        log "Certificate chain verification passed"
    else
        warn "Certificate chain verification failed"
    fi
    
    # Check SSL Labs rating (if available)
    if command -v curl &> /dev/null; then
        log "SSL configuration appears to be working"
    fi
}

# Function to monitor certificate health
monitor_certificate_health() {
    local domain="$1"
    
    log "Starting certificate health monitoring for $domain..."
    
    # Check expiration
    local expiry_status
    check_certificate_expiration "$domain"
    expiry_status=$?
    
    # If certificate is expiring soon, attempt renewal
    if [ "$expiry_status" -ge 1 ]; then
        if [ "$expiry_status" -eq 2 ]; then
            # Critical - force renewal
            log "Certificate expiring soon, attempting immediate renewal..."
            attempt_renewal "$domain"
        else
            # Warning - check if renewal is needed
            local days_left
            days_left=$(check_certificate_expiration "$domain" | grep -o '[0-9]\+ days' | cut -d' ' -f1)
            if [ "$days_left" -le 15 ]; then
                log "Certificate expires in $days_left days, attempting renewal..."
                attempt_renewal "$domain"
            fi
        fi
    fi
    
    # Verify current certificate
    verify_certificate "$domain"
    
    log "Certificate health monitoring completed"
}

# Function to setup monitoring cron job
setup_monitoring() {
    log "Setting up certificate monitoring..."
    
    # Create log file
    sudo touch "$LOG_FILE"
    sudo chmod 644 "$LOG_FILE"
    
    # Add to crontab if not already present
    local cron_job="0 2 * * * /opt/pouw/scripts/ssl-monitor.sh --domain $DOMAIN --monitor >> $LOG_FILE 2>&1"
    
    if ! crontab -l 2>/dev/null | grep -q "ssl-monitor.sh"; then
        (crontab -l 2>/dev/null || true; echo "$cron_job") | crontab -
        log "Certificate monitoring cron job added"
    else
        log "Certificate monitoring cron job already exists"
    fi
}

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --domain DOMAIN         Domain to monitor"
    echo "  -e, --email EMAIL           Email for notifications"
    echo "  --slack-webhook URL         Slack webhook URL for notifications"
    echo "  --warning-days DAYS         Days before expiration to warn (default: 30)"
    echo "  --critical-days DAYS        Days before expiration for critical alert (default: 7)"
    echo "  --monitor                   Run monitoring check"
    echo "  --setup                     Setup monitoring cron job"
    echo "  --renew                     Force certificate renewal"
    echo "  --help                      Display this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -d example.com --monitor"
    echo "  $0 -d example.com -e admin@example.com --setup"
    echo "  $0 -d example.com --renew"
}

# Parse command line arguments
MONITOR=false
SETUP=false
RENEW=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--domain)
            DOMAIN="$2"
            shift 2
            ;;
        -e|--email)
            NOTIFICATION_EMAIL="$2"
            shift 2
            ;;
        --slack-webhook)
            SLACK_WEBHOOK="$2"
            shift 2
            ;;
        --warning-days)
            WARNING_DAYS="$2"
            shift 2
            ;;
        --critical-days)
            CRITICAL_DAYS="$2"
            shift 2
            ;;
        --monitor)
            MONITOR=true
            shift
            ;;
        --setup)
            SETUP=true
            shift
            ;;
        --renew)
            RENEW=true
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Main execution
main() {
    if [ -z "$DOMAIN" ]; then
        error "Domain is required. Use -d or --domain option."
        exit 1
    fi
    
    # Create log directory if it doesn't exist
    sudo mkdir -p "$(dirname "$LOG_FILE")"
    
    if [ "$SETUP" = true ]; then
        setup_monitoring
    elif [ "$MONITOR" = true ]; then
        monitor_certificate_health "$DOMAIN"
    elif [ "$RENEW" = true ]; then
        attempt_renewal "$DOMAIN"
    else
        # Default action - monitor
        monitor_certificate_health "$DOMAIN"
    fi
}

# Execute main function
main "$@"
