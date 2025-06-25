#!/usr/bin/env python3
"""
PoUW Configuration Validation Script

This script validates the configuration system and ensures all required
environment variables are properly set for deployment.
"""

import sys
import os
from pathlib import Path
from typing import List, Tuple

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import get_config_manager, ConfigValidationError


def validate_configuration(environment: str = "production") -> Tuple[bool, List[str]]:
    """
    Validate configuration for the specified environment

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    try:
        # Load configuration
        config_manager = get_config_manager(environment=environment)
        config = config_manager.get_config()

        print(f"🔍 Validating {environment} configuration...")
        print(f"📁 Configuration file: {config_manager.config_file}")
        print()

        # Validate deployment configuration
        print("🚀 Deployment Configuration:")
        if not config.deployment.domain or config.deployment.domain == "localhost":
            errors.append("POUW_DOMAIN must be set to a valid domain name")
            print("  ❌ Domain: Not configured")
        else:
            print(f"  ✅ Domain: {config.deployment.domain}")

        if not config.deployment.email or "@" not in config.deployment.email:
            errors.append("POUW_EMAIL must be set to a valid email address")
            print("  ❌ Email: Not configured")
        else:
            print(f"  ✅ Email: {config.deployment.email}")

        if not config.deployment.vps_ip or config.deployment.vps_ip == "YOUR_HOSTINGER_VPS_IP":
            errors.append("POUW_VPS_IP must be set to your actual VPS IP address")
            print("  ❌ VPS IP: Not configured")
        else:
            print(f"  ✅ VPS IP: {config.deployment.vps_ip}")

        if not config.deployment.github_repo or "YOUR_USERNAME" in config.deployment.github_repo:
            errors.append("POUW_GITHUB_REPO must be set to your actual GitHub repository")
            print("  ❌ GitHub Repo: Not configured")
        else:
            print(f"  ✅ GitHub Repo: {config.deployment.github_repo}")

        print()

        # Validate node configuration
        print("🌐 Node Configuration:")
        if not config.node.node_id:
            errors.append("POUW_NODE_ID must be set")
            print("  ❌ Node ID: Not configured")
        else:
            print(f"  ✅ Node ID: {config.node.node_id}")

        if not config.node.role:
            errors.append("POUW_NODE_ROLE must be set")
            print("  ❌ Node Role: Not configured")
        else:
            print(f"  ✅ Node Role: {config.node.role}")

        if config.node.port < 1024 or config.node.port > 65535:
            errors.append("POUW_NODE_PORT must be between 1024 and 65535")
            print(f"  ❌ Node Port: {config.node.port} (invalid range)")
        else:
            print(f"  ✅ Node Port: {config.node.port}")

        print()

        # Validate security configuration
        print("🔒 Security Configuration:")
        if not config.security.secret_key or len(config.security.secret_key) < 32:
            errors.append("POUW_SECURITY_SECRET_KEY must be at least 32 characters long")
            print("  ❌ Secret Key: Too short or not set")
        else:
            print(
                f"  ✅ Secret Key: {'*' * len(config.security.secret_key)} ({len(config.security.secret_key)} chars)"
            )

        if not config.security.jwt_secret or len(config.security.jwt_secret) < 32:
            errors.append("POUW_SECURITY_JWT_SECRET must be at least 32 characters long")
            print("  ❌ JWT Secret: Too short or not set")
        else:
            print(
                f"  ✅ JWT Secret: {'*' * len(config.security.jwt_secret)} ({len(config.security.jwt_secret)} chars)"
            )

        print()

        # Validate database configuration (if enabled)
        print("🗄️ Database Configuration:")
        if config.database.enable_database:
            if not config.database.host:
                errors.append("POUW_DATABASE_HOST must be set when database is enabled")
                print("  ❌ Host: Not configured")
            else:
                print(f"  ✅ Host: {config.database.host}")

            if not config.database.name:
                errors.append("POUW_DATABASE_NAME must be set when database is enabled")
                print("  ❌ Name: Not configured")
            else:
                print(f"  ✅ Name: {config.database.name}")

            if not config.database.user:
                errors.append("POUW_DATABASE_USER must be set when database is enabled")
                print("  ❌ User: Not configured")
            else:
                print(f"  ✅ User: {config.database.user}")

            if not config.database.password:
                errors.append("POUW_DATABASE_PASSWORD should be set when database is enabled")
                print("  ⚠️ Password: Not set (may be intentional for local dev)")
            else:
                print(f"  ✅ Password: {'*' * len(config.database.password)}")
        else:
            print("  ℹ️ Database disabled")

        print()

        # Validate SSL configuration (if enabled)
        print("🔐 SSL Configuration:")
        if config.deployment.enable_ssl:
            if not config.deployment.domain or config.deployment.domain == "localhost":
                errors.append("Valid domain required for SSL")
                print("  ❌ SSL Domain: Not configured")
            else:
                print(f"  ✅ SSL Domain: {config.deployment.domain}")

            if config.deployment.ssl_staging:
                print("  ⚠️ SSL Staging: Enabled (test certificates)")
            else:
                print("  ✅ SSL Production: Enabled")
        else:
            print("  ℹ️ SSL disabled")

        print()

        # Validate monitoring configuration
        print("📊 Monitoring Configuration:")
        print(f"  ✅ Dashboard Host: {config.monitoring.dashboard_host}")
        print(f"  ✅ Dashboard Port: {config.monitoring.dashboard_port}")
        print(f"  ✅ Log Level: {config.monitoring.log_level}")

        if config.monitoring.dashboard_url:
            print(f"  ✅ Dashboard URL: {config.monitoring.dashboard_url}")
        else:
            print("  ⚠️ Dashboard URL: Not configured")

        print()

        # Summary
        if errors:
            print("❌ Configuration validation failed!")
            print(f"Found {len(errors)} error(s):")
            for i, error in enumerate(errors, 1):
                print(f"  {i}. {error}")
            return False, errors
        else:
            print("✅ Configuration validation passed!")
            print("All required settings are properly configured.")
            return True, []

    except ConfigValidationError as e:
        errors.append(f"Configuration validation error: {e}")
        print(f"❌ Configuration validation failed: {e}")
        return False, errors
    except Exception as e:
        errors.append(f"Unexpected error: {e}")
        print(f"❌ Unexpected error during validation: {e}")
        return False, errors


def check_environment_file(environment: str) -> bool:
    """Check if environment file exists and is readable"""
    env_file = project_root / f".env.{environment}"

    if not env_file.exists():
        print(f"❌ Environment file not found: {env_file}")
        return False

    if not env_file.is_file():
        print(f"❌ Environment file is not a regular file: {env_file}")
        return False

    try:
        with open(env_file, "r") as f:
            content = f.read()
            if len(content.strip()) == 0:
                print(f"⚠️ Environment file is empty: {env_file}")
                return False
    except PermissionError:
        print(f"❌ Cannot read environment file (permission denied): {env_file}")
        return False
    except Exception as e:
        print(f"❌ Error reading environment file: {e}")
        return False

    print(f"✅ Environment file found: {env_file}")
    return True


def main():
    """Main validation function"""
    import argparse

    parser = argparse.ArgumentParser(description="Validate PoUW configuration")
    parser.add_argument(
        "--environment",
        "-e",
        default="production",
        choices=["development", "production"],
        help="Environment to validate",
    )
    parser.add_argument(
        "--fix-suggestions",
        action="store_true",
        help="Show suggestions for fixing configuration issues",
    )

    args = parser.parse_args()

    print("🔧 PoUW Configuration Validator")
    print("=" * 50)
    print()

    # Check environment file
    if not check_environment_file(args.environment):
        print()
        print("💡 To create the environment file, copy from template:")
        print(f"   cp .env.template .env.{args.environment}")
        sys.exit(1)

    print()

    # Validate configuration
    is_valid, errors = validate_configuration(args.environment)

    if not is_valid and args.fix_suggestions:
        print()
        print("💡 Fix Suggestions:")
        print("-" * 20)

        suggestion_map = {
            "POUW_DOMAIN": "Set your actual domain: POUW_DOMAIN=api.yourdomain.com",
            "POUW_EMAIL": "Set your email: POUW_EMAIL=admin@yourdomain.com",
            "POUW_VPS_IP": "Set your VPS IP: POUW_VPS_IP=123.456.789.012",
            "POUW_GITHUB_REPO": "Set your GitHub repo: POUW_GITHUB_REPO=https://github.com/username/PoUW.git",
            "POUW_SECURITY_SECRET_KEY": 'Generate key: python -c "import secrets; print(secrets.token_urlsafe(64))"',
            "POUW_SECURITY_JWT_SECRET": 'Generate JWT secret: python -c "import secrets; print(secrets.token_urlsafe(64))"',
        }

        for error in errors:
            for key, suggestion in suggestion_map.items():
                if key.replace("POUW_", "").lower() in error.lower():
                    print(f"  • {suggestion}")
                    break

    print()
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
