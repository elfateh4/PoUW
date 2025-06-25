#!/usr/bin/env python3
"""
PoUW Configuration Management System

This module provides centralized configuration management for the PoUW project.
It handles loading environment variables, validation, and providing default values.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union
import json
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class DeploymentConfig:
    """Deployment-related configuration"""

    domain: str = "localhost"
    email: str = "admin@localhost"
    enable_ssl: bool = False
    ssl_staging: bool = True
    vps_ip: str = "127.0.0.1"
    vps_user: str = "root"
    vps_ssh_key_path: str = "~/.ssh/id_rsa"
    vps_ssh_port: int = 22
    github_repo: str = ""
    github_username: str = ""
    github_token: str = ""
    deployment_branch: str = "main"


@dataclass
class NodeConfig:
    """Node-related configuration"""

    node_id: str = "default_node_001"
    role: str = "MINER"
    host: str = "0.0.0.0"
    port: int = 8000
    max_peers: int = 50
    bootstrap_peers: str = "localhost:8000"
    network_id: str = "pouw_devnet"
    p2p_port: int = 8001
    initial_stake: float = 100.0
    mining_intensity: float = 0.00001
    stake_amount: float = 100.0


@dataclass
class SecurityConfig:
    """Security-related configuration"""

    secret_key: str = "dev_secret_key"
    jwt_secret: str = "dev_jwt_secret"
    encryption_key: str = "dev_encryption_key"
    enable_security: bool = True
    enable_attack_mitigation: bool = True
    enable_advanced_features: bool = False
    enable_production_features: bool = False
    api_rate_limit: int = 100
    api_rate_window: int = 3600
    api_key: str = "dev_api_key"


@dataclass
class DatabaseConfig:
    """Database-related configuration"""

    database_url: str = "sqlite:///pouw.db"
    database_host: str = "localhost"
    database_port: int = 5432
    database_name: str = "pouw"
    database_user: str = "pouw_user"
    database_password: str = "password"
    redis_url: str = "redis://localhost:6379/0"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""


@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration"""

    log_level: str = "INFO"
    log_file: str = "logs/pouw.log"
    log_max_size: str = "100MB"
    log_backup_count: int = 5
    enable_metrics: bool = True
    metrics_port: int = 9090
    metrics_path: str = "/metrics"
    dashboard_url: str = "http://localhost:8080"
    dashboard_port: int = 8080
    dashboard_host: str = "0.0.0.0"
    dashboard_enabled: bool = True
    health_check_port: int = 8080
    health_check_interval: int = 30
    health_check_timeout: int = 10


@dataclass
class NotificationConfig:
    """Notification configuration"""

    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    notification_email: str = ""
    slack_webhook_url: str = ""
    slack_channel: str = "#pouw-alerts"
    discord_webhook_url: str = ""


@dataclass
class PoUWConfig:
    """Complete PoUW configuration"""

    environment: str = "development"
    debug: bool = True
    testing: bool = False
    development: bool = True

    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    node: NodeConfig = field(default_factory=NodeConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    notification: NotificationConfig = field(default_factory=NotificationConfig)


class ConfigManager:
    """Configuration manager for PoUW"""

    def __init__(self, env_file: Optional[str] = None, environment: Optional[str] = None):
        """
        Initialize configuration manager

        Args:
            env_file: Path to environment file (optional)
            environment: Environment name (development, production, etc.)
        """
        self.project_root = Path(__file__).parent
        self.environment = environment or os.getenv("POUW_ENVIRONMENT", "development")
        self.config = PoUWConfig()

        # Load configuration
        self._load_configuration(env_file)

    def _load_configuration(self, env_file: Optional[str] = None):
        """Load configuration from environment files and variables"""

        # Determine environment file to load
        if env_file:
            env_files = [env_file]
        else:
            env_files = [f".env.{self.environment}", ".env", ".env.template"]

        # Load environment files
        for env_file_path in env_files:
            full_path = self.project_root / env_file_path
            if full_path.exists():
                self._load_env_file(full_path)
                logger.info(f"Loaded configuration from {env_file_path}")
                break
        else:
            logger.warning("No environment file found, using defaults")

        # Load from environment variables
        self._load_from_env()

        # Validate configuration
        self._validate_configuration()

    def _load_env_file(self, file_path: Path):
        """Load environment variables from file"""
        try:
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip("\"'")
                        os.environ[key] = value
        except Exception as e:
            logger.error(f"Error loading environment file {file_path}: {e}")

    def _load_from_env(self):
        """Load configuration from environment variables"""

        # General configuration
        self.config.environment = os.getenv("POUW_ENVIRONMENT", "development")
        self.config.debug = self._get_bool("POUW_DEBUG", True)
        self.config.testing = self._get_bool("POUW_TESTING", False)
        self.config.development = self._get_bool("POUW_DEVELOPMENT", True)

        # Deployment configuration
        self.config.deployment.domain = os.getenv("POUW_DOMAIN", "localhost")
        self.config.deployment.email = os.getenv("POUW_EMAIL", "admin@localhost")
        self.config.deployment.enable_ssl = self._get_bool("POUW_ENABLE_SSL", False)
        self.config.deployment.ssl_staging = self._get_bool("POUW_SSL_STAGING", True)
        self.config.deployment.vps_ip = os.getenv("POUW_VPS_IP", "127.0.0.1")
        self.config.deployment.vps_user = os.getenv("POUW_VPS_USER", "root")
        self.config.deployment.vps_ssh_key_path = os.getenv(
            "POUW_VPS_SSH_KEY_PATH", "~/.ssh/id_rsa"
        )
        self.config.deployment.vps_ssh_port = self._get_int("POUW_VPS_SSH_PORT", 22)
        self.config.deployment.github_repo = os.getenv("POUW_GITHUB_REPO", "")
        self.config.deployment.github_username = os.getenv("POUW_GITHUB_USERNAME", "")
        self.config.deployment.github_token = os.getenv("POUW_GITHUB_TOKEN", "")
        self.config.deployment.deployment_branch = os.getenv("POUW_DEPLOYMENT_BRANCH", "main")

        # Node configuration
        self.config.node.node_id = os.getenv("POUW_NODE_ID", "default_node_001")
        self.config.node.role = os.getenv("POUW_NODE_ROLE", "MINER")
        self.config.node.host = os.getenv("POUW_NODE_HOST", "0.0.0.0")
        self.config.node.port = self._get_int("POUW_NODE_PORT", 8000)
        self.config.node.max_peers = self._get_int("POUW_MAX_PEERS", 50)
        self.config.node.bootstrap_peers = os.getenv("POUW_BOOTSTRAP_PEERS", "localhost:8000")
        self.config.node.network_id = os.getenv("POUW_NETWORK_ID", "pouw_devnet")
        self.config.node.p2p_port = self._get_int("POUW_P2P_PORT", 8001)
        self.config.node.initial_stake = self._get_float("POUW_INITIAL_STAKE", 100.0)
        self.config.node.mining_intensity = self._get_float("POUW_MINING_INTENSITY", 0.00001)
        self.config.node.stake_amount = self._get_float("POUW_STAKE_AMOUNT", 100.0)

        # Security configuration
        self.config.security.secret_key = os.getenv("POUW_SECRET_KEY", "dev_secret_key")
        self.config.security.jwt_secret = os.getenv("POUW_JWT_SECRET", "dev_jwt_secret")
        self.config.security.encryption_key = os.getenv("POUW_ENCRYPTION_KEY", "dev_encryption_key")
        self.config.security.enable_security = self._get_bool("POUW_ENABLE_SECURITY", True)
        self.config.security.enable_attack_mitigation = self._get_bool(
            "POUW_ENABLE_ATTACK_MITIGATION", True
        )
        self.config.security.enable_advanced_features = self._get_bool(
            "POUW_ENABLE_ADVANCED_FEATURES", False
        )
        self.config.security.enable_production_features = self._get_bool(
            "POUW_ENABLE_PRODUCTION_FEATURES", False
        )
        self.config.security.api_rate_limit = self._get_int("POUW_API_RATE_LIMIT", 100)
        self.config.security.api_rate_window = self._get_int("POUW_API_RATE_WINDOW", 3600)
        self.config.security.api_key = os.getenv("POUW_API_KEY", "dev_api_key")

        # Database configuration
        self.config.database.database_url = os.getenv("POUW_DATABASE_URL", "sqlite:///pouw.db")
        self.config.database.database_host = os.getenv("POUW_DATABASE_HOST", "localhost")
        self.config.database.database_port = self._get_int("POUW_DATABASE_PORT", 5432)
        self.config.database.database_name = os.getenv("POUW_DATABASE_NAME", "pouw")
        self.config.database.database_user = os.getenv("POUW_DATABASE_USER", "pouw_user")
        self.config.database.database_password = os.getenv("POUW_DATABASE_PASSWORD", "password")
        self.config.database.redis_url = os.getenv("POUW_REDIS_URL", "redis://localhost:6379/0")
        self.config.database.redis_host = os.getenv("POUW_REDIS_HOST", "localhost")
        self.config.database.redis_port = self._get_int("POUW_REDIS_PORT", 6379)
        self.config.database.redis_password = os.getenv("POUW_REDIS_PASSWORD", "")

        # Monitoring configuration
        self.config.monitoring.log_level = os.getenv("POUW_LOG_LEVEL", "INFO")
        self.config.monitoring.log_file = os.getenv("POUW_LOG_FILE", "logs/pouw.log")
        self.config.monitoring.log_max_size = os.getenv("POUW_LOG_MAX_SIZE", "100MB")
        self.config.monitoring.log_backup_count = self._get_int("POUW_LOG_BACKUP_COUNT", 5)
        self.config.monitoring.enable_metrics = self._get_bool("POUW_ENABLE_METRICS", True)
        self.config.monitoring.metrics_port = self._get_int("POUW_METRICS_PORT", 9090)
        self.config.monitoring.metrics_path = os.getenv("POUW_METRICS_PATH", "/metrics")
        self.config.monitoring.dashboard_url = os.getenv(
            "POUW_DASHBOARD_URL", "http://localhost:8080"
        )
        self.config.monitoring.dashboard_port = self._get_int("POUW_DASHBOARD_PORT", 8080)
        self.config.monitoring.dashboard_host = os.getenv("POUW_DASHBOARD_HOST", "0.0.0.0")
        self.config.monitoring.dashboard_enabled = self._get_bool("POUW_DASHBOARD_ENABLED", True)
        self.config.monitoring.health_check_port = self._get_int("POUW_HEALTH_CHECK_PORT", 8080)
        self.config.monitoring.health_check_interval = self._get_int(
            "POUW_HEALTH_CHECK_INTERVAL", 30
        )
        self.config.monitoring.health_check_timeout = self._get_int("POUW_HEALTH_CHECK_TIMEOUT", 10)

        # Notification configuration
        self.config.notification.smtp_host = os.getenv("POUW_SMTP_HOST", "")
        self.config.notification.smtp_port = self._get_int("POUW_SMTP_PORT", 587)
        self.config.notification.smtp_user = os.getenv("POUW_SMTP_USER", "")
        self.config.notification.smtp_password = os.getenv("POUW_SMTP_PASSWORD", "")
        self.config.notification.notification_email = os.getenv("POUW_NOTIFICATION_EMAIL", "")
        self.config.notification.slack_webhook_url = os.getenv("POUW_SLACK_WEBHOOK_URL", "")
        self.config.notification.slack_channel = os.getenv("POUW_SLACK_CHANNEL", "#pouw-alerts")
        self.config.notification.discord_webhook_url = os.getenv("POUW_DISCORD_WEBHOOK_URL", "")

    def _get_bool(self, key: str, default: bool) -> bool:
        """Get boolean value from environment"""
        value = os.getenv(key, str(default)).lower()
        return value in ("true", "1", "yes", "on")

    def _get_int(self, key: str, default: int) -> int:
        """Get integer value from environment"""
        try:
            return int(os.getenv(key, str(default)))
        except ValueError:
            logger.warning(f"Invalid integer value for {key}, using default: {default}")
            return default

    def _get_float(self, key: str, default: float) -> float:
        """Get float value from environment"""
        try:
            return float(os.getenv(key, str(default)))
        except ValueError:
            logger.warning(f"Invalid float value for {key}, using default: {default}")
            return default

    def _validate_configuration(self):
        """Validate configuration values"""

        # Validate required fields for production
        if self.config.environment == "production":
            required_fields = [
                ("POUW_DOMAIN", self.config.deployment.domain),
                ("POUW_EMAIL", self.config.deployment.email),
                ("POUW_VPS_IP", self.config.deployment.vps_ip),
                ("POUW_SECRET_KEY", self.config.security.secret_key),
                ("POUW_JWT_SECRET", self.config.security.jwt_secret),
            ]

            for field_name, field_value in required_fields:
                if not field_value or field_value in [
                    "your-domain.com",
                    "admin@localhost",
                    "dev_secret_key",
                ]:
                    logger.error(f"Production configuration requires valid {field_name}")
                    raise ValueError(f"Invalid production configuration: {field_name}")

        # Validate port ranges
        ports = [
            self.config.node.port,
            self.config.node.p2p_port,
            self.config.monitoring.metrics_port,
            self.config.monitoring.dashboard_port,
            self.config.monitoring.health_check_port,
        ]

        for port in ports:
            if not 1 <= port <= 65535:
                raise ValueError(f"Invalid port number: {port}")

    def get_config(self) -> PoUWConfig:
        """Get the complete configuration"""
        return self.config

    def get_deployment_config(self) -> DeploymentConfig:
        """Get deployment configuration"""
        return self.config.deployment

    def get_node_config(self) -> NodeConfig:
        """Get node configuration"""
        return self.config.node

    def get_security_config(self) -> SecurityConfig:
        """Get security configuration"""
        return self.config.security

    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration"""
        return self.config.database

    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration"""
        return self.config.monitoring

    def get_notification_config(self) -> NotificationConfig:
        """Get notification configuration"""
        return self.config.notification

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "environment": self.config.environment,
            "debug": self.config.debug,
            "testing": self.config.testing,
            "development": self.config.development,
            "deployment": self.config.deployment.__dict__,
            "node": self.config.node.__dict__,
            "security": self.config.security.__dict__,
            "database": self.config.database.__dict__,
            "monitoring": self.config.monitoring.__dict__,
            "notification": self.config.notification.__dict__,
        }

    def to_json(self) -> str:
        """Convert configuration to JSON"""
        return json.dumps(self.to_dict(), indent=2)

    def print_config(self):
        """Print current configuration"""
        print("=" * 50)
        print(f"PoUW Configuration ({self.config.environment})")
        print("=" * 50)
        print(self.to_json())
        print("=" * 50)


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(
    env_file: Optional[str] = None, environment: Optional[str] = None
) -> ConfigManager:
    """Get or create global configuration manager"""
    global _config_manager

    if _config_manager is None:
        _config_manager = ConfigManager(env_file, environment)

    return _config_manager


def get_config() -> PoUWConfig:
    """Get the global configuration"""
    return get_config_manager().get_config()


def get_deployment_config() -> DeploymentConfig:
    """Get deployment configuration"""
    return get_config_manager().get_deployment_config()


def get_node_config() -> NodeConfig:
    """Get node configuration"""
    return get_config_manager().get_node_config()


def get_security_config() -> SecurityConfig:
    """Get security configuration"""
    return get_config_manager().get_security_config()


def get_database_config() -> DatabaseConfig:
    """Get database configuration"""
    return get_config_manager().get_database_config()


def get_monitoring_config() -> MonitoringConfig:
    """Get monitoring configuration"""
    return get_config_manager().get_monitoring_config()


def get_notification_config() -> NotificationConfig:
    """Get notification configuration"""
    return get_config_manager().get_notification_config()


if __name__ == "__main__":
    """CLI for configuration management"""
    import argparse

    parser = argparse.ArgumentParser(description="PoUW Configuration Manager")
    parser.add_argument("--environment", "-e", help="Environment (development, production, etc.)")
    parser.add_argument("--env-file", "-f", help="Environment file path")
    parser.add_argument("--print", "-p", action="store_true", help="Print current configuration")
    parser.add_argument("--validate", "-v", action="store_true", help="Validate configuration")
    parser.add_argument(
        "--generate-keys", "-g", action="store_true", help="Generate new security keys"
    )

    args = parser.parse_args()

    try:
        config_manager = ConfigManager(args.env_file, args.environment)

        if args.print:
            config_manager.print_config()

        if args.validate:
            print("✓ Configuration is valid")

        if args.generate_keys:
            import secrets

            print("Generated security keys:")
            print(f"SECRET_KEY={secrets.token_hex(32)}")
            print(f"JWT_SECRET={secrets.token_hex(32)}")
            print(f"ENCRYPTION_KEY={secrets.token_hex(32)}")
            print(f"API_KEY={secrets.token_hex(16)}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
