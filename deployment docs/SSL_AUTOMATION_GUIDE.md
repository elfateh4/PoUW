# PoUW SSL Certificate Automation Guide

This guide covers the automated SSL certificate management system for your PoUW blockchain deployment, including Certbot integration, automatic renewal, and comprehensive verification.

## 🔐 Overview

The SSL automation system provides:

- **Automated certificate generation** with Let's Encrypt
- **Domain verification** via HTTP challenge
- **Automatic renewal** with cron jobs
- **Health monitoring** and alerting
- **Comprehensive verification** and security testing
- **Fallback to self-signed** certificates for development

## 📋 Prerequisites

### System Requirements

- Ubuntu/Debian-based VPS (tested on Hostinger)
- Nginx web server
- Root or sudo access
- Valid domain name pointing to your VPS

### DNS Configuration

Your domain must be properly configured to point to your VPS:

```bash
# Check DNS resolution
dig +short your-domain.com A
# Should return your VPS IP address
```

## 🚀 Quick Start

### 1. Deploy with SSL Automation

```bash
# Deploy with automatic SSL setup
./deploy.sh -d your-domain.com -e your-email@example.com

# Deploy with staging certificates (for testing)
./deploy.sh -d your-domain.com -e your-email@example.com --ssl-staging

# Deploy without SSL (development)
./deploy.sh --skip-ssl
```

### 2. Manual SSL Setup

```bash
# Setup SSL certificates manually
./scripts/ssl-setup.sh -d your-domain.com -e your-email@example.com

# Test with staging environment
./scripts/ssl-setup.sh -d your-domain.com -e your-email@example.com --staging

# Dry run (test without creating certificates)
./scripts/ssl-setup.sh -d your-domain.com -e your-email@example.com --dry-run
```

### 3. Verify SSL Configuration

```bash
# Comprehensive SSL verification
./scripts/ssl-verify.sh -d your-domain.com

# Verbose verification with detailed output
./scripts/ssl-verify.sh -d your-domain.com -v
```

## 🔧 SSL Setup Script (`ssl-setup.sh`)

### Features

- **Automatic Certbot installation** and configuration
- **HTTP challenge verification** via webroot
- **Nginx configuration** with SSL best practices
- **Certificate chain validation**
- **Automatic renewal setup**
- **Staging environment support** for testing

### Usage Examples

```bash
# Basic setup
./scripts/ssl-setup.sh -d api.pouw.network -e admin@pouw.network

# Staging environment (for testing)
./scripts/ssl-setup.sh -d test.pouw.network -e admin@pouw.network --staging

# Force renewal of existing certificate
./scripts/ssl-setup.sh -d api.pouw.network -e admin@pouw.network --force-renewal

# Dry run (test configuration)
./scripts/ssl-setup.sh -d api.pouw.network -e admin@pouw.network --dry-run

# Custom webroot path
./scripts/ssl-setup.sh -d api.pouw.network -e admin@pouw.network -w /var/www/custom
```

### Configuration Options

| Option               | Description                           | Default            |
| -------------------- | ------------------------------------- | ------------------ |
| `-d, --domain`       | Domain name for certificate           | Required           |
| `-e, --email`        | Email for Let's Encrypt notifications | Required           |
| `-w, --webroot`      | Webroot path for verification         | `/var/www/certbot` |
| `--dry-run`          | Test without creating certificates    | False              |
| `--force-renewal`    | Force certificate renewal             | False              |
| `--staging`          | Use staging environment               | False              |
| `--dns-verification` | Use DNS verification                  | False              |

## 📊 SSL Monitoring (`ssl-monitor.sh`)

### Features

- **Certificate expiration monitoring**
- **Automatic renewal attempts**
- **Email and Slack notifications**
- **Health verification**
- **Cron job integration**

### Setup Monitoring

```bash
# Setup automatic monitoring
./scripts/ssl-monitor.sh -d your-domain.com --setup

# Setup with email notifications
./scripts/ssl-monitor.sh -d your-domain.com -e admin@example.com --setup

# Setup with Slack notifications
./scripts/ssl-monitor.sh -d your-domain.com --slack-webhook "https://hooks.slack.com/..." --setup
```

### Manual Monitoring

```bash
# Check certificate status
./scripts/ssl-monitor.sh -d your-domain.com --monitor

# Force certificate renewal
./scripts/ssl-monitor.sh -d your-domain.com --renew

# Custom warning thresholds
./scripts/ssl-monitor.sh -d your-domain.com --warning-days 45 --critical-days 14 --monitor
```

### Monitoring Configuration

| Option            | Description                    | Default  |
| ----------------- | ------------------------------ | -------- |
| `-d, --domain`    | Domain to monitor              | Required |
| `-e, --email`     | Email for notifications        | None     |
| `--slack-webhook` | Slack webhook URL              | None     |
| `--warning-days`  | Days before expiration to warn | 30       |
| `--critical-days` | Days for critical alert        | 7        |
| `--monitor`       | Run monitoring check           | False    |
| `--setup`         | Setup cron job                 | False    |
| `--renew`         | Force renewal                  | False    |

## 🔍 SSL Verification (`ssl-verify.sh`)

### Features

- **DNS resolution verification**
- **TLS connectivity testing**
- **Certificate validity checks**
- **Certificate chain verification**
- **TLS configuration analysis**
- **Security headers inspection**
- **HTTP to HTTPS redirect testing**

### Usage Examples

```bash
# Basic verification
./scripts/ssl-verify.sh -d your-domain.com

# Verbose output with detailed information
./scripts/ssl-verify.sh -d your-domain.com -v

# Custom port and timeout
./scripts/ssl-verify.sh -d your-domain.com -p 443 -t 15
```

### Verification Checks

1. **DNS Resolution** - Confirms domain resolves correctly
2. **Connectivity** - Tests TCP and TLS connections
3. **Certificate Validity** - Checks dates and issuer
4. **Certificate Chain** - Verifies trust chain
5. **TLS Configuration** - Tests protocol versions and ciphers
6. **Security Headers** - Checks for security best practices
7. **HTTP Redirect** - Ensures HTTP redirects to HTTPS

## 🔄 Automatic Renewal

### Cron Job Setup

The system automatically sets up cron jobs for certificate renewal:

```bash
# Check current cron jobs
crontab -l

# Manual cron job entry (if needed)
0 12 * * * /usr/bin/certbot renew --quiet
0 2 * * * /opt/pouw/scripts/ssl-monitor.sh --domain your-domain.com --monitor
```

### Renewal Process

1. **Daily check** at 2 AM for certificate expiration
2. **Automatic renewal** when certificate expires within 30 days
3. **Nginx reload** after successful renewal
4. **Notification** sent on success or failure
5. **Health verification** after renewal

### Manual Renewal

```bash
# Test renewal (dry run)
sudo certbot renew --dry-run

# Force renewal
sudo certbot renew --force-renewal

# Renew specific certificate
sudo certbot renew --cert-name your-domain.com
```

## 🔧 Nginx Configuration

### SSL Configuration Template

The system generates optimized Nginx configurations:

```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com www.your-domain.com;

    # SSL Configuration
    ssl_certificate /etc/nginx/ssl/fullchain.pem;
    ssl_certificate_key /etc/nginx/ssl/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";

    # Application proxy
    location / {
        proxy_pass http://pouw_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# HTTP to HTTPS redirect
server {
    listen 80;
    server_name your-domain.com www.your-domain.com;

    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }

    location / {
        return 301 https://$server_name$request_uri;
    }
}
```

## 📱 Notifications

### Email Notifications

Configure email notifications for certificate events:

```bash
# Install mail utilities
sudo apt-get install mailutils

# Configure with Gmail SMTP (example)
sudo dpkg-reconfigure postfix
```

### Slack Notifications

Set up Slack webhook for team notifications:

1. Create a Slack app and incoming webhook
2. Use webhook URL with monitoring script:

```bash
./scripts/ssl-monitor.sh -d your-domain.com --slack-webhook "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
```

## 🐛 Troubleshooting

### Common Issues

#### 1. DNS Resolution Problems

```bash
# Check DNS propagation
dig +short your-domain.com A
nslookup your-domain.com

# Wait for DNS propagation (can take up to 24 hours)
```

#### 2. Port 80/443 Not Accessible

```bash
# Check firewall rules
sudo ufw status
sudo ufw allow 80
sudo ufw allow 443

# Check if ports are in use
sudo netstat -tlnp | grep ':80\|:443'
```

#### 3. Certbot Rate Limits

```bash
# Use staging environment for testing
./scripts/ssl-setup.sh -d your-domain.com -e your-email@example.com --staging

# Check rate limit status
curl -s "https://crt.sh/?q=your-domain.com" | jq '.[] | select(.name_value=="your-domain.com")'
```

#### 4. Certificate Validation Failures

```bash
# Check webroot accessibility
curl http://your-domain.com/.well-known/acme-challenge/test

# Verify nginx configuration
sudo nginx -t
sudo systemctl reload nginx
```

#### 5. Renewal Failures

```bash
# Check renewal logs
sudo cat /var/log/letsencrypt/letsencrypt.log

# Test renewal manually
sudo certbot renew --dry-run --verbose

# Force renewal if needed
sudo certbot renew --force-renewal
```

### Log Files

Important log files for debugging:

```bash
# Certbot logs
/var/log/letsencrypt/letsencrypt.log

# SSL monitoring logs
/var/log/pouw-ssl-monitor.log

# SSL renewal logs
/var/log/pouw-ssl-renewal.log

# Nginx logs
/var/log/nginx/access.log
/var/log/nginx/error.log
```

## 🔒 Security Best Practices

### Certificate Security

1. **Use strong key sizes** (2048-bit RSA minimum)
2. **Enable HSTS** (HTTP Strict Transport Security)
3. **Implement certificate pinning** for critical applications
4. **Monitor certificate transparency** logs
5. **Regular security audits** with SSL Labs or similar tools

### Automated Security

```bash
# Run security audit
./scripts/ssl-verify.sh -d your-domain.com -v

# Check SSL Labs rating
curl -s "https://api.ssllabs.com/api/v3/analyze?host=your-domain.com" | jq '.status'

# Monitor certificate transparency
curl -s "https://crt.sh/?q=your-domain.com&output=json" | jq '.[0]'
```

## 📊 Monitoring Dashboard

### Certificate Status API

The system provides API endpoints for monitoring:

```bash
# Check certificate status
curl https://your-domain.com/api/ssl/status

# Get certificate details
curl https://your-domain.com/api/ssl/certificate

# Health check
curl https://your-domain.com/health
```

### Grafana Integration

For advanced monitoring, integrate with Grafana:

1. **Prometheus metrics** for certificate expiration
2. **Alerting rules** for renewal failures
3. **Dashboard panels** for SSL health visualization

## 🚀 Production Deployment

### Complete Deployment Example

```bash
# 1. Deploy PoUW with SSL
./deploy.sh -d api.pouw.network -e admin@pouw.network

# 2. Verify SSL configuration
./scripts/ssl-verify.sh -d api.pouw.network -v

# 3. Setup monitoring
./scripts/ssl-monitor.sh -d api.pouw.network -e admin@pouw.network --setup

# 4. Test the deployment
curl -I https://api.pouw.network/health

# 5. Check certificate status
openssl s_client -connect api.pouw.network:443 -servername api.pouw.network < /dev/null 2>/dev/null | openssl x509 -noout -dates
```

### Post-Deployment Checklist

- [ ] SSL certificate is valid and trusted
- [ ] HTTP redirects to HTTPS
- [ ] Security headers are present
- [ ] Automatic renewal is configured
- [ ] Monitoring is active
- [ ] Backup procedures are in place
- [ ] Team is notified of SSL status

## 📚 Additional Resources

- [Let's Encrypt Documentation](https://letsencrypt.org/docs/)
- [SSL Labs SSL Server Test](https://www.ssllabs.com/ssltest/)
- [Mozilla SSL Configuration Generator](https://ssl-config.mozilla.org/)
- [Certificate Transparency Logs](https://crt.sh/)

## 🆘 Support

For issues with SSL automation:

1. **Check logs** in `/var/log/pouw-ssl-*`
2. **Run verification** with `ssl-verify.sh -v`
3. **Test manually** with `certbot renew --dry-run`
4. **Review configuration** in `/etc/nginx/sites-available/`

---

**Note**: Always test SSL configurations in staging environment before production deployment.
