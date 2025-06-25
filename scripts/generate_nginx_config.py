#!/usr/bin/env python3
"""
Nginx Configuration Generator for PoUW

This script generates nginx configuration dynamically based on the
centralized configuration system.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import get_config_manager

def generate_nginx_config(environment='production'):
    """Generate nginx configuration based on environment settings"""
    
    config_manager = get_config_manager(environment=environment)
    config = config_manager.get_config()
    
    # Get configuration values
    domain = config.deployment.domain
    enable_ssl = config.deployment.enable_ssl
    node_port = config.node.port
    dashboard_port = config.monitoring.dashboard_port
    vps_ip = config.deployment.vps_ip
    
    # Generate nginx configuration
    nginx_config = f"""events {{
    worker_connections 1024;
}}

http {{
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;

    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';

    access_log /var/log/nginx/access.log main;
    error_log /var/log/nginx/error.log;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/json application/javascript application/xml+rss application/atom+xml image/svg+xml;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=dashboard:10m rate=5r/s;

    # Upstream for PoUW application
    upstream pouw_backend {{
        server pouw-app:{node_port};
        keepalive 32;
    }}

    # Upstream for PoUW dashboard
    upstream pouw_dashboard {{
        server pouw-dashboard:{dashboard_port};
        keepalive 16;
    }}

    # Health check endpoint (no rate limiting)
    map $request_uri $rate_limit_key {{
        ~^/health$ "";
        ~^/status$ "";
        default $binary_remote_addr;
    }}
"""

    if enable_ssl:
        nginx_config += f"""
    # HTTP server (redirect to HTTPS)
    server {{
        listen 80;
        server_name {domain};
        
        # Allow Let's Encrypt challenges
        location /.well-known/acme-challenge/ {{
            root /var/www/certbot;
        }}
        
        # Health checks (no redirect for Docker health checks)
        location /health {{
            proxy_pass http://pouw_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }}
        
        # Redirect all other traffic to HTTPS
        location / {{
            return 301 https://$server_name$request_uri;
        }}
    }}

    # HTTPS server
    server {{
        listen 443 ssl http2;
        server_name {domain};

        # SSL configuration
        ssl_certificate /etc/nginx/ssl/live/{domain}/fullchain.pem;
        ssl_certificate_key /etc/nginx/ssl/live/{domain}/privkey.pem;
        
        # Modern SSL configuration
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;
        
        # HSTS
        add_header Strict-Transport-Security "max-age=63072000" always;
        
        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Referrer-Policy "strict-origin-when-cross-origin";
"""
    else:
        nginx_config += f"""
    # HTTP server (no SSL)
    server {{
        listen 80;
        server_name {domain} {vps_ip};
"""

    # Add common location blocks
    nginx_config += f"""
        # Health check endpoint
        location /health {{
            proxy_pass http://pouw_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            access_log off;
        }}

        # Status endpoint
        location /status {{
            proxy_pass http://pouw_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }}

        # API endpoints
        location /api/ {{
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://pouw_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            
            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }}

        # Dashboard (served by separate service)
        location /dashboard/ {{
            limit_req zone=dashboard burst=10 nodelay;
            
            rewrite ^/dashboard/(.*)$ /$1 break;
            proxy_pass http://pouw_dashboard;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support for dashboard
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }}

        # WebSocket endpoint for dashboard
        location /ws {{
            proxy_pass http://pouw_dashboard;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }}

        # Default route to dashboard
        location / {{
            limit_req zone=dashboard burst=10 nodelay;
            
            proxy_pass http://pouw_dashboard;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }}
    }}
}}
"""
    
    return nginx_config

def main():
    """Main function to generate nginx configuration"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate nginx configuration for PoUW')
    parser.add_argument('--environment', '-e', default='production',
                       choices=['development', 'production'],
                       help='Environment configuration to use')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file path (default: stdout)')
    
    args = parser.parse_args()
    
    try:
        config_text = generate_nginx_config(args.environment)
        
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(config_text)
            print(f"✅ Nginx configuration written to {output_path}")
        else:
            print(config_text)
            
    except Exception as e:
        print(f"❌ Error generating nginx configuration: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
