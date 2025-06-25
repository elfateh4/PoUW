#!/usr/bin/env python3
"""
PoUW Environment Configuration Generator

This script helps generate and customize environment configuration files
for different deployment environments.
"""

import sys
import os
import secrets
from pathlib import Path
from typing import Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def generate_secret_key(length: int = 64) -> str:
    """Generate a secure random secret key"""
    return secrets.token_urlsafe(length)


def get_environment_defaults(environment: str) -> Dict[str, Any]:
    """Get default values for environment variables based on environment type"""

    common_defaults = {
        # Environment
        "POUW_ENVIRONMENT": environment,
        # Node Configuration
        "POUW_NODE_ID": f"{environment}_supervisor_001",
        "POUW_NODE_ROLE": "SUPERVISOR",
        "POUW_NODE_HOST": "0.0.0.0",
        "POUW_NODE_PORT": "8000",
        "POUW_MAX_PEERS": "20" if environment == "development" else "100",
        "POUW_BOOTSTRAP_PEERS": "",
        "POUW_NETWORK_ID": f"pouw_{environment}",
        "POUW_P2P_PORT": "8001",
        # Economic Configuration
        "POUW_INITIAL_STAKE": "100.0" if environment == "development" else "2000.0",
        "POUW_MINING_INTENSITY": "0.00001",
        "POUW_ENABLE_MINING": "true",
        "POUW_MINING_ALGORITHM": "useful_work",
        # Security Configuration
        "POUW_SECURITY_SECRET_KEY": generate_secret_key(),
        "POUW_SECURITY_JWT_SECRET": generate_secret_key(),
        "POUW_SECURITY_ENCRYPTION_KEY": generate_secret_key(32),
        "POUW_ENABLE_SECURITY": "true",
        "POUW_ENABLE_ATTACK_MITIGATION": "true",
        "POUW_ENABLE_PRODUCTION_FEATURES": "true" if environment == "production" else "false",
        "POUW_ENABLE_ADVANCED_FEATURES": "false",
        # Database Configuration
        "POUW_DATABASE_ENABLE": "false",
        "POUW_DATABASE_HOST": "localhost",
        "POUW_DATABASE_PORT": "5432",
        "POUW_DATABASE_NAME": "pouw",
        "POUW_DATABASE_USER": "pouw",
        "POUW_DATABASE_PASSWORD": "",
        "POUW_DATABASE_SSL": "require" if environment == "production" else "prefer",
        # Monitoring Configuration
        "POUW_LOG_LEVEL": "INFO",
        "POUW_LOG_FILE": f"/app/logs/pouw_{environment}.log",
        "POUW_ENABLE_METRICS": "true",
        "POUW_METRICS_PORT": "9090",
        "POUW_DASHBOARD_HOST": "0.0.0.0",
        "POUW_DASHBOARD_PORT": "8080",
        "POUW_DASHBOARD_URL": "http://localhost:8080",
        "POUW_DASHBOARD_WORKERS": "1" if environment == "development" else "4",
        "POUW_ENABLE_ACCESS_LOGS": "false" if environment == "development" else "true",
        "POUW_HEALTH_CHECK_PORT": "8001",
        "POUW_HEALTH_CHECK_INTERVAL": "30",
        # Notification Configuration
        "POUW_ENABLE_NOTIFICATIONS": "false",
        "POUW_NOTIFICATION_EMAIL": "",
        "POUW_NOTIFICATION_SLACK_WEBHOOK": "",
        "POUW_NOTIFICATION_TELEGRAM_TOKEN": "",
        "POUW_NOTIFICATION_TELEGRAM_CHAT_ID": "",
        "POUW_SMTP_HOST": "",
        "POUW_SMTP_PORT": "587",
        "POUW_SMTP_USERNAME": "",
        "POUW_SMTP_PASSWORD": "",
        "POUW_SMTP_TLS": "true",
        # Development Configuration
        "POUW_DEBUG": "true" if environment == "development" else "false",
        "POUW_ENABLE_HOT_RELOAD": "true" if environment == "development" else "false",
        "POUW_ENABLE_PROFILING": "false",
        "POUW_TEST_MODE": "false",
        # Backup Configuration
        "POUW_ENABLE_BACKUP": "true" if environment == "production" else "false",
        "POUW_BACKUP_INTERVAL": "86400",  # 24 hours
        "POUW_BACKUP_RETENTION": "30",  # 30 days
        "POUW_BACKUP_S3_BUCKET": "",
        "POUW_BACKUP_S3_REGION": "us-east-1",
        "POUW_BACKUP_S3_ACCESS_KEY": "",
        "POUW_BACKUP_S3_SECRET_KEY": "",
        # ML Configuration
        "POUW_ML_ENABLE_GPU": "false",
        "POUW_ML_MAX_MODELS": "10",
        "POUW_ML_MODEL_CACHE_SIZE": "1000000000",  # 1GB
        "POUW_ML_TRAINING_TIMEOUT": "3600",  # 1 hour
        "POUW_ML_ENABLE_DISTRIBUTED": "false",
        # Blockchain Configuration
        "POUW_BLOCKCHAIN_DIFFICULTY": "4",
        "POUW_BLOCKCHAIN_BLOCK_TIME": "600",  # 10 minutes
        "POUW_BLOCKCHAIN_MAX_BLOCK_SIZE": "1048576",  # 1MB
        "POUW_BLOCKCHAIN_ENABLE_SHARDING": "false",
        # Network Performance Configuration
        "POUW_NETWORK_TIMEOUT": "30",
        "POUW_NETWORK_RETRY_ATTEMPTS": "3",
        "POUW_NETWORK_KEEP_ALIVE": "true",
        "POUW_NETWORK_COMPRESSION": "true",
        "POUW_NETWORK_BUFFER_SIZE": "65536",  # 64KB
        # Docker Configuration
        "POUW_DOCKER_ENABLE": "true",
        "POUW_DOCKER_RESTART_POLICY": "unless-stopped",
        "POUW_DOCKER_MEMORY_LIMIT": "2g" if environment == "development" else "8g",
        "POUW_DOCKER_CPU_LIMIT": "2" if environment == "development" else "4",
        # Firewall Configuration
        "POUW_FIREWALL_ENABLE": "true" if environment == "production" else "false",
        "POUW_FIREWALL_ALLOWED_PORTS": "22,80,443,8000,8080",
        "POUW_FIREWALL_RATE_LIMIT": "true" if environment == "production" else "false",
    }

    if environment == "development":
        # Development-specific overrides
        common_defaults.update(
            {
                "POUW_DOMAIN": "localhost",
                "POUW_EMAIL": "dev@localhost",
                "POUW_VPS_IP": "127.0.0.1",
                "POUW_ENABLE_SSL": "false",
                "POUW_SSL_STAGING": "true",
                "POUW_GITHUB_REPO": "https://github.com/YOUR_USERNAME/PoUW.git",
                "POUW_GITHUB_USERNAME": "YOUR_USERNAME",
                "POUW_GITHUB_TOKEN": "DEVELOPMENT_GITHUB_TOKEN",
                "POUW_DEPLOYMENT_BRANCH": "development",
                "POUW_DOCKER_REGISTRY": "docker.io",
                "POUW_DOCKER_USERNAME": "YOUR_DOCKER_USERNAME",
                "POUW_DOCKER_PASSWORD": "YOUR_DOCKER_PASSWORD",
                "POUW_DOCKER_IMAGE_NAME": "pouw-dev",
            }
        )
    else:  # production
        common_defaults.update(
            {
                "POUW_DOMAIN": "api.pouw.network",
                "POUW_EMAIL": "admin@pouw.network",
                "POUW_VPS_IP": "YOUR_HOSTINGER_VPS_IP",
                "POUW_ENABLE_SSL": "true",
                "POUW_SSL_STAGING": "false",
                "POUW_FORCE_SSL_RENEWAL": "false",
                "POUW_VPS_USER": "root",
                "POUW_VPS_SSH_KEY_PATH": "~/.ssh/pouw_production",
                "POUW_VPS_SSH_PORT": "22",
                "POUW_GITHUB_REPO": "https://github.com/YOUR_USERNAME/PoUW.git",
                "POUW_GITHUB_USERNAME": "YOUR_USERNAME",
                "POUW_GITHUB_TOKEN": "PRODUCTION_GITHUB_TOKEN",
                "POUW_DEPLOYMENT_BRANCH": "main",
                "POUW_DOCKER_REGISTRY": "docker.io",
                "POUW_DOCKER_USERNAME": "YOUR_DOCKER_USERNAME",
                "POUW_DOCKER_PASSWORD": "YOUR_DOCKER_PASSWORD",
                "POUW_DOCKER_IMAGE_NAME": "pouw",
            }
        )

    return common_defaults


def generate_env_file(environment: str, output_file: Path = None, overrides: Dict[str, str] = None) -> str:  # type: ignore
    """Generate environment file content"""

    if output_file is None:
        output_file = project_root / f".env.{environment}"

    defaults = get_environment_defaults(environment)

    # Apply overrides
    if overrides:
        defaults.update(overrides)

    # Create file content with organized sections
    content = f"""# PoUW {environment.title()} Environment Configuration
# This file contains {environment}-specific settings

# ==============================================
# DEPLOYMENT CONFIGURATION
# ==============================================

# Domain and SSL Configuration
POUW_DOMAIN={defaults['POUW_DOMAIN']}
POUW_EMAIL={defaults['POUW_EMAIL']}
POUW_ENABLE_SSL={defaults['POUW_ENABLE_SSL']}
POUW_SSL_STAGING={defaults['POUW_SSL_STAGING']}
"""

    if environment == "production":
        content += f"""POUW_FORCE_SSL_RENEWAL={defaults['POUW_FORCE_SSL_RENEWAL']}

# VPS and Server Configuration
POUW_VPS_IP={defaults['POUW_VPS_IP']}
POUW_VPS_USER={defaults['POUW_VPS_USER']}
POUW_VPS_SSH_KEY_PATH={defaults['POUW_VPS_SSH_KEY_PATH']}
POUW_VPS_SSH_PORT={defaults['POUW_VPS_SSH_PORT']}
"""

    content += f"""
# Repository Configuration
POUW_GITHUB_REPO={defaults['POUW_GITHUB_REPO']}
POUW_GITHUB_USERNAME={defaults['POUW_GITHUB_USERNAME']}
POUW_GITHUB_TOKEN={defaults['POUW_GITHUB_TOKEN']}
POUW_DEPLOYMENT_BRANCH={defaults['POUW_DEPLOYMENT_BRANCH']}

# Docker Registry
POUW_DOCKER_REGISTRY={defaults['POUW_DOCKER_REGISTRY']}
POUW_DOCKER_USERNAME={defaults['POUW_DOCKER_USERNAME']}
POUW_DOCKER_PASSWORD={defaults['POUW_DOCKER_PASSWORD']}
POUW_DOCKER_IMAGE_NAME={defaults['POUW_DOCKER_IMAGE_NAME']}

# ==============================================
# NODE CONFIGURATION
# ==============================================

# Node Identity
POUW_NODE_ID={defaults['POUW_NODE_ID']}
POUW_NODE_ROLE={defaults['POUW_NODE_ROLE']}
POUW_NODE_HOST={defaults['POUW_NODE_HOST']}
POUW_NODE_PORT={defaults['POUW_NODE_PORT']}

# Network Configuration
POUW_MAX_PEERS={defaults['POUW_MAX_PEERS']}
POUW_BOOTSTRAP_PEERS={defaults['POUW_BOOTSTRAP_PEERS']}
POUW_NETWORK_ID={defaults['POUW_NETWORK_ID']}
POUW_P2P_PORT={defaults['POUW_P2P_PORT']}

# Economic Configuration
POUW_INITIAL_STAKE={defaults['POUW_INITIAL_STAKE']}
POUW_MINING_INTENSITY={defaults['POUW_MINING_INTENSITY']}
POUW_ENABLE_MINING={defaults['POUW_ENABLE_MINING']}
POUW_MINING_ALGORITHM={defaults['POUW_MINING_ALGORITHM']}

# ==============================================
# SECURITY CONFIGURATION
# ==============================================

# Security Keys (Generated automatically - DO NOT SHARE)
POUW_SECURITY_SECRET_KEY={defaults['POUW_SECURITY_SECRET_KEY']}
POUW_SECURITY_JWT_SECRET={defaults['POUW_SECURITY_JWT_SECRET']}
POUW_SECURITY_ENCRYPTION_KEY={defaults['POUW_SECURITY_ENCRYPTION_KEY']}

# Security Features
POUW_ENABLE_SECURITY={defaults['POUW_ENABLE_SECURITY']}
POUW_ENABLE_ATTACK_MITIGATION={defaults['POUW_ENABLE_ATTACK_MITIGATION']}
POUW_ENABLE_PRODUCTION_FEATURES={defaults['POUW_ENABLE_PRODUCTION_FEATURES']}
POUW_ENABLE_ADVANCED_FEATURES={defaults['POUW_ENABLE_ADVANCED_FEATURES']}

# ==============================================
# DATABASE CONFIGURATION
# ==============================================

POUW_DATABASE_ENABLE={defaults['POUW_DATABASE_ENABLE']}
POUW_DATABASE_HOST={defaults['POUW_DATABASE_HOST']}
POUW_DATABASE_PORT={defaults['POUW_DATABASE_PORT']}
POUW_DATABASE_NAME={defaults['POUW_DATABASE_NAME']}
POUW_DATABASE_USER={defaults['POUW_DATABASE_USER']}
POUW_DATABASE_PASSWORD={defaults['POUW_DATABASE_PASSWORD']}
POUW_DATABASE_SSL={defaults['POUW_DATABASE_SSL']}

# ==============================================
# MONITORING CONFIGURATION
# ==============================================

# Logging
POUW_LOG_LEVEL={defaults['POUW_LOG_LEVEL']}
POUW_LOG_FILE={defaults['POUW_LOG_FILE']}

# Metrics and Monitoring
POUW_ENABLE_METRICS={defaults['POUW_ENABLE_METRICS']}
POUW_METRICS_PORT={defaults['POUW_METRICS_PORT']}

# Dashboard Configuration
POUW_DASHBOARD_HOST={defaults['POUW_DASHBOARD_HOST']}
POUW_DASHBOARD_PORT={defaults['POUW_DASHBOARD_PORT']}
POUW_DASHBOARD_URL={defaults['POUW_DASHBOARD_URL']}
POUW_DASHBOARD_WORKERS={defaults['POUW_DASHBOARD_WORKERS']}
POUW_ENABLE_ACCESS_LOGS={defaults['POUW_ENABLE_ACCESS_LOGS']}

# Health Checks
POUW_HEALTH_CHECK_PORT={defaults['POUW_HEALTH_CHECK_PORT']}
POUW_HEALTH_CHECK_INTERVAL={defaults['POUW_HEALTH_CHECK_INTERVAL']}

# ==============================================
# NOTIFICATION CONFIGURATION
# ==============================================

POUW_ENABLE_NOTIFICATIONS={defaults['POUW_ENABLE_NOTIFICATIONS']}
POUW_NOTIFICATION_EMAIL={defaults['POUW_NOTIFICATION_EMAIL']}
POUW_NOTIFICATION_SLACK_WEBHOOK={defaults['POUW_NOTIFICATION_SLACK_WEBHOOK']}
POUW_NOTIFICATION_TELEGRAM_TOKEN={defaults['POUW_NOTIFICATION_TELEGRAM_TOKEN']}
POUW_NOTIFICATION_TELEGRAM_CHAT_ID={defaults['POUW_NOTIFICATION_TELEGRAM_CHAT_ID']}

# SMTP Configuration
POUW_SMTP_HOST={defaults['POUW_SMTP_HOST']}
POUW_SMTP_PORT={defaults['POUW_SMTP_PORT']}
POUW_SMTP_USERNAME={defaults['POUW_SMTP_USERNAME']}
POUW_SMTP_PASSWORD={defaults['POUW_SMTP_PASSWORD']}
POUW_SMTP_TLS={defaults['POUW_SMTP_TLS']}

# ==============================================
# DEVELOPMENT CONFIGURATION
# ==============================================

POUW_ENVIRONMENT={defaults['POUW_ENVIRONMENT']}
POUW_DEBUG={defaults['POUW_DEBUG']}
POUW_ENABLE_HOT_RELOAD={defaults['POUW_ENABLE_HOT_RELOAD']}
POUW_ENABLE_PROFILING={defaults['POUW_ENABLE_PROFILING']}
POUW_TEST_MODE={defaults['POUW_TEST_MODE']}

# ==============================================
# BACKUP CONFIGURATION
# ==============================================

POUW_ENABLE_BACKUP={defaults['POUW_ENABLE_BACKUP']}
POUW_BACKUP_INTERVAL={defaults['POUW_BACKUP_INTERVAL']}
POUW_BACKUP_RETENTION={defaults['POUW_BACKUP_RETENTION']}
POUW_BACKUP_S3_BUCKET={defaults['POUW_BACKUP_S3_BUCKET']}
POUW_BACKUP_S3_REGION={defaults['POUW_BACKUP_S3_REGION']}
POUW_BACKUP_S3_ACCESS_KEY={defaults['POUW_BACKUP_S3_ACCESS_KEY']}
POUW_BACKUP_S3_SECRET_KEY={defaults['POUW_BACKUP_S3_SECRET_KEY']}

# ==============================================
# ML CONFIGURATION
# ==============================================

POUW_ML_ENABLE_GPU={defaults['POUW_ML_ENABLE_GPU']}
POUW_ML_MAX_MODELS={defaults['POUW_ML_MAX_MODELS']}
POUW_ML_MODEL_CACHE_SIZE={defaults['POUW_ML_MODEL_CACHE_SIZE']}
POUW_ML_TRAINING_TIMEOUT={defaults['POUW_ML_TRAINING_TIMEOUT']}
POUW_ML_ENABLE_DISTRIBUTED={defaults['POUW_ML_ENABLE_DISTRIBUTED']}

# ==============================================
# BLOCKCHAIN CONFIGURATION
# ==============================================

POUW_BLOCKCHAIN_DIFFICULTY={defaults['POUW_BLOCKCHAIN_DIFFICULTY']}
POUW_BLOCKCHAIN_BLOCK_TIME={defaults['POUW_BLOCKCHAIN_BLOCK_TIME']}
POUW_BLOCKCHAIN_MAX_BLOCK_SIZE={defaults['POUW_BLOCKCHAIN_MAX_BLOCK_SIZE']}
POUW_BLOCKCHAIN_ENABLE_SHARDING={defaults['POUW_BLOCKCHAIN_ENABLE_SHARDING']}

# ==============================================
# NETWORK PERFORMANCE CONFIGURATION
# ==============================================

POUW_NETWORK_TIMEOUT={defaults['POUW_NETWORK_TIMEOUT']}
POUW_NETWORK_RETRY_ATTEMPTS={defaults['POUW_NETWORK_RETRY_ATTEMPTS']}
POUW_NETWORK_KEEP_ALIVE={defaults['POUW_NETWORK_KEEP_ALIVE']}
POUW_NETWORK_COMPRESSION={defaults['POUW_NETWORK_COMPRESSION']}
POUW_NETWORK_BUFFER_SIZE={defaults['POUW_NETWORK_BUFFER_SIZE']}

# ==============================================
# DOCKER CONFIGURATION
# ==============================================

POUW_DOCKER_ENABLE={defaults['POUW_DOCKER_ENABLE']}
POUW_DOCKER_RESTART_POLICY={defaults['POUW_DOCKER_RESTART_POLICY']}
POUW_DOCKER_MEMORY_LIMIT={defaults['POUW_DOCKER_MEMORY_LIMIT']}
POUW_DOCKER_CPU_LIMIT={defaults['POUW_DOCKER_CPU_LIMIT']}

# ==============================================
# FIREWALL CONFIGURATION
# ==============================================

POUW_FIREWALL_ENABLE={defaults['POUW_FIREWALL_ENABLE']}
POUW_FIREWALL_ALLOWED_PORTS={defaults['POUW_FIREWALL_ALLOWED_PORTS']}
POUW_FIREWALL_RATE_LIMIT={defaults['POUW_FIREWALL_RATE_LIMIT']}
"""

    return content


def main():
    """Main function to generate environment files"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate PoUW environment configuration files")
    parser.add_argument(
        "--environment",
        "-e",
        default="production",
        choices=["development", "production"],
        help="Environment to generate configuration for",
    )
    parser.add_argument(
        "--output", "-o", type=str, help="Output file path (default: .env.{environment})"
    )
    parser.add_argument(
        "--force", "-f", action="store_true", help="Overwrite existing file if it exists"
    )
    parser.add_argument("--domain", type=str, help="Domain name (overrides default)")
    parser.add_argument("--email", type=str, help="Email address (overrides default)")
    parser.add_argument("--vps-ip", type=str, help="VPS IP address (overrides default)")

    args = parser.parse_args()

    # Determine output file
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = project_root / f".env.{args.environment}"

    # Check if file exists
    if output_file.exists() and not args.force:
        print(f"❌ File {output_file} already exists. Use --force to overwrite.")
        sys.exit(1)

    # Prepare overrides
    overrides = {}
    if args.domain:
        overrides["POUW_DOMAIN"] = args.domain
    if args.email:
        overrides["POUW_EMAIL"] = args.email
    if args.vps_ip:
        overrides["POUW_VPS_IP"] = args.vps_ip

    # Generate content
    content = generate_env_file(args.environment, output_file, overrides)

    # Write file
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            f.write(content)

        print(f"✅ Generated {args.environment} environment file: {output_file}")
        print()
        print("📋 Next steps:")
        print(f"   1. Edit {output_file} with your actual values")
        print(f"   2. Replace placeholder values (YOUR_*, etc.)")
        print(f"   3. Set your domain, email, and VPS IP")
        print(f"   4. Run validation: python3 scripts/validate_config.py -e {args.environment}")

        if args.environment == "production":
            print()
            print("🔒 Security notes:")
            print("   • Secret keys have been generated automatically")
            print("   • Keep this file secure and never commit secrets to version control")
            print("   • Consider using a secrets management system for production")

    except Exception as e:
        print(f"❌ Error writing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
