"""
Infrastructure Automation Module

Provides load balancing, auto-scaling, infrastructure as code (IaC),
and deployment automation capabilities for enterprise PoUW deployments.
"""

import asyncio
import json
import yaml
import logging
import hashlib
import subprocess
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import tempfile
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Load balancing strategy enumeration"""

    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    IP_HASH = "ip_hash"
    LEAST_RESPONSE_TIME = "least_response_time"


class ScalingDirection(Enum):
    """Scaling direction enumeration"""

    UP = "up"
    DOWN = "down"
    MAINTAIN = "maintain"


@dataclass
class LoadBalancerConfig:
    """Load balancer configuration"""

    name: str
    strategy: LoadBalancingStrategy
    backend_servers: List[Dict[str, Any]]
    health_check: Dict[str, Any] = field(default_factory=dict)
    ssl_config: Optional[Dict[str, Any]] = None
    rate_limiting: Optional[Dict[str, Any]] = None
    session_persistence: bool = False

    def to_nginx_config(self) -> str:
        """Generate nginx configuration"""
        upstream_name = f"pouw_{self.name}"

        # Build upstream block
        upstream_block = f"upstream {upstream_name} {{\n"

        if self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            upstream_block += "    least_conn;\n"
        elif self.strategy == LoadBalancingStrategy.IP_HASH:
            upstream_block += "    ip_hash;\n"

        for server in self.backend_servers:
            weight = server.get("weight", 1)
            max_fails = server.get("max_fails", 3)
            fail_timeout = server.get("fail_timeout", "30s")

            upstream_block += f"    server {server['host']}:{server['port']} "
            upstream_block += (
                f"weight={weight} max_fails={max_fails} fail_timeout={fail_timeout};\n"
            )

        upstream_block += "}\n\n"

        # Build server block
        server_block = "server {\n"
        server_block += "    listen 80;\n"

        if self.ssl_config:
            server_block += "    listen 443 ssl;\n"
            server_block += f"    ssl_certificate {self.ssl_config['cert_path']};\n"
            server_block += f"    ssl_certificate_key {self.ssl_config['key_path']};\n"

        server_block += f"    server_name {self.name}.pouw.local;\n\n"

        # Health check location
        if self.health_check:
            server_block += "    location /health {\n"
            server_block += f"        proxy_pass http://{upstream_name}/health;\n"
            server_block += "        proxy_set_header Host $host;\n"
            server_block += "        proxy_set_header X-Real-IP $remote_addr;\n"
            server_block += "    }\n\n"

        # Main proxy location
        server_block += "    location / {\n"
        server_block += f"        proxy_pass http://{upstream_name};\n"
        server_block += "        proxy_set_header Host $host;\n"
        server_block += "        proxy_set_header X-Real-IP $remote_addr;\n"
        server_block += "        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;\n"
        server_block += "        proxy_set_header X-Forwarded-Proto $scheme;\n"

        if self.rate_limiting:
            server_block += f"        limit_req zone={self.rate_limiting['zone']} "
            server_block += f"burst={self.rate_limiting['burst']} nodelay;\n"

        server_block += "    }\n"
        server_block += "}\n"

        return upstream_block + server_block


@dataclass
class AutoScalingRule:
    """Auto-scaling rule configuration"""

    metric_name: str
    threshold_up: float
    threshold_down: float
    min_replicas: int
    max_replicas: int
    scale_up_cooldown: timedelta = field(default=timedelta(minutes=5))
    scale_down_cooldown: timedelta = field(default=timedelta(minutes=10))
    evaluation_period: timedelta = field(default=timedelta(minutes=2))


@dataclass
class ScalingDecision:
    """Auto-scaling decision result"""

    component: str
    current_replicas: int
    target_replicas: int
    direction: ScalingDirection
    reason: str
    metric_value: float
    timestamp: datetime = field(default_factory=datetime.now)


class LoadBalancer:
    """Advanced load balancer with multiple strategies"""

    def __init__(self, config_dir: str = "/etc/nginx/sites-available"):
        """Initialize load balancer"""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.balancer_configs: Dict[str, LoadBalancerConfig] = {}

    async def deploy_load_balancer(self, config: LoadBalancerConfig) -> bool:
        """Deploy load balancer configuration"""
        try:
            # Generate nginx configuration
            nginx_config = config.to_nginx_config()

            # Write configuration file
            config_file = self.config_dir / f"pouw-{config.name}"
            with open(config_file, "w") as f:
                f.write(nginx_config)

            # Enable site
            enabled_dir = Path("/etc/nginx/sites-enabled")
            enabled_dir.mkdir(parents=True, exist_ok=True)

            symlink_path = enabled_dir / f"pouw-{config.name}"
            if symlink_path.exists():
                symlink_path.unlink()

            symlink_path.symlink_to(config_file)

            # Test nginx configuration
            test_result = await self._test_nginx_config()
            if not test_result:
                logger.error(f"Invalid nginx configuration for {config.name}")
                return False

            # Reload nginx
            reload_result = await self._reload_nginx()
            if reload_result:
                self.balancer_configs[config.name] = config
                logger.info(f"Successfully deployed load balancer: {config.name}")
                return True
            else:
                logger.error(f"Failed to reload nginx for {config.name}")
                return False

        except Exception as e:
            logger.error(f"Error deploying load balancer {config.name}: {e}")
            return False

    async def _test_nginx_config(self) -> bool:
        """Test nginx configuration validity"""
        try:
            result = await asyncio.create_subprocess_exec(
                "nginx", "-t", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await result.communicate()
            return result.returncode == 0

        except Exception as e:
            logger.error(f"Error testing nginx config: {e}")
            return False

    async def _reload_nginx(self) -> bool:
        """Reload nginx configuration"""
        try:
            result = await asyncio.create_subprocess_exec(
                "systemctl",
                "reload",
                "nginx",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await result.communicate()
            return result.returncode == 0

        except Exception as e:
            logger.error(f"Error reloading nginx: {e}")
            return False

    async def update_backend_servers(
        self, balancer_name: str, backend_servers: List[Dict[str, Any]]
    ) -> bool:
        """Update backend servers for a load balancer"""
        if balancer_name not in self.balancer_configs:
            logger.error(f"Load balancer {balancer_name} not found")
            return False

        config = self.balancer_configs[balancer_name]
        config.backend_servers = backend_servers

        return await self.deploy_load_balancer(config)

    async def get_balancer_status(self, balancer_name: str) -> Dict[str, Any]:
        """Get load balancer status and statistics"""
        if balancer_name not in self.balancer_configs:
            return {"error": "Load balancer not found"}

        config = self.balancer_configs[balancer_name]

        # Check backend server health
        backend_health = []
        for server in config.backend_servers:
            health = await self._check_backend_health(server)
            backend_health.append(
                {
                    "server": f"{server['host']}:{server['port']}",
                    "healthy": health,
                    "weight": server.get("weight", 1),
                }
            )

        return {
            "name": balancer_name,
            "strategy": config.strategy.value,
            "backend_count": len(config.backend_servers),
            "backend_health": backend_health,
            "healthy_backends": sum(1 for h in backend_health if h["healthy"]),
            "session_persistence": config.session_persistence,
        }

    async def _check_backend_health(self, server: Dict[str, Any]) -> bool:
        """Check health of a backend server"""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                url = f"http://{server['host']}:{server['port']}/health"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    return response.status == 200

        except Exception:
            return False


class AutoScaler:
    """Intelligent auto-scaling system"""

    def __init__(self, kubernetes_orchestrator):
        """Initialize auto-scaler"""
        self.orchestrator = kubernetes_orchestrator
        self.scaling_rules: Dict[str, AutoScalingRule] = {}
        self.scaling_history: List[ScalingDecision] = []
        self.last_scaling_action: Dict[str, datetime] = {}
        self.running = False
        self.scaling_task: Optional[asyncio.Task] = None

    def add_scaling_rule(self, component: str, rule: AutoScalingRule):
        """Add auto-scaling rule for a component"""
        self.scaling_rules[component] = rule
        logger.info(f"Added scaling rule for {component}")

    async def start_auto_scaling(self, check_interval: int = 60):
        """Start auto-scaling monitoring"""
        if self.running:
            return

        self.running = True
        self.scaling_task = asyncio.create_task(self._scaling_loop(check_interval))
        logger.info("Started auto-scaling")

    async def stop_auto_scaling(self):
        """Stop auto-scaling monitoring"""
        self.running = False
        if self.scaling_task:
            self.scaling_task.cancel()
            try:
                await self.scaling_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped auto-scaling")

    async def _scaling_loop(self, check_interval: int):
        """Main auto-scaling loop"""
        while self.running:
            try:
                await self._evaluate_scaling_rules()
                await asyncio.sleep(check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-scaling loop: {e}")
                await asyncio.sleep(30)

    async def _evaluate_scaling_rules(self):
        """Evaluate all scaling rules and make decisions"""
        deployment_status = await self.orchestrator.get_deployment_status()

        for component, rule in self.scaling_rules.items():
            try:
                decision = await self._evaluate_component_scaling(
                    component, rule, deployment_status
                )
                if decision and decision.direction != ScalingDirection.MAINTAIN:
                    await self._execute_scaling_decision(decision)

            except Exception as e:
                logger.error(f"Error evaluating scaling for {component}: {e}")

    async def _evaluate_component_scaling(
        self, component: str, rule: AutoScalingRule, deployment_status: Dict[str, Any]
    ) -> Optional[ScalingDecision]:
        """Evaluate scaling for a specific component"""
        if component not in deployment_status:
            logger.warning(f"Component {component} not found in deployment status")
            return None

        current_replicas = deployment_status[component].get("replicas", 0)

        # Get metric value (placeholder - would integrate with actual metrics)
        metric_value = await self._get_metric_value(component, rule.metric_name)
        if metric_value is None:
            return None

        # Check cooldown periods
        last_action = self.last_scaling_action.get(component)
        now = datetime.now()

        if last_action:
            time_since_last = now - last_action
            if time_since_last < rule.scale_up_cooldown:
                return None

        # Determine scaling direction
        target_replicas = current_replicas
        direction = ScalingDirection.MAINTAIN
        reason = "Metric within acceptable range"

        if metric_value > rule.threshold_up and current_replicas < rule.max_replicas:
            target_replicas = min(current_replicas + 1, rule.max_replicas)
            direction = ScalingDirection.UP
            reason = (
                f"{rule.metric_name} ({metric_value:.2f}) above threshold ({rule.threshold_up})"
            )

        elif metric_value < rule.threshold_down and current_replicas > rule.min_replicas:
            target_replicas = max(current_replicas - 1, rule.min_replicas)
            direction = ScalingDirection.DOWN
            reason = (
                f"{rule.metric_name} ({metric_value:.2f}) below threshold ({rule.threshold_down})"
            )

        return ScalingDecision(
            component=component,
            current_replicas=current_replicas,
            target_replicas=target_replicas,
            direction=direction,
            reason=reason,
            metric_value=metric_value,
        )

    async def _get_metric_value(self, component: str, metric_name: str) -> Optional[float]:
        """Get metric value for component (placeholder implementation)"""
        # This would integrate with actual metrics collection
        # For demonstration, return a simulated value
        import random

        return random.uniform(0, 100)

    async def _execute_scaling_decision(self, decision: ScalingDecision):
        """Execute a scaling decision"""
        try:
            success = await self.orchestrator.scale_deployment(
                decision.component, decision.target_replicas
            )

            if success:
                self.last_scaling_action[decision.component] = decision.timestamp
                self.scaling_history.append(decision)
                logger.info(
                    f"Scaled {decision.component} from {decision.current_replicas} "
                    f"to {decision.target_replicas} replicas. Reason: {decision.reason}"
                )
            else:
                logger.error(f"Failed to scale {decision.component}")

        except Exception as e:
            logger.error(f"Error executing scaling decision for {decision.component}: {e}")

    def get_scaling_history(
        self, component: Optional[str] = None, time_range: Optional[timedelta] = None
    ) -> List[ScalingDecision]:
        """Get scaling history with optional filtering"""
        history = self.scaling_history

        if component:
            history = [d for d in history if d.component == component]

        if time_range:
            cutoff_time = datetime.now() - time_range
            history = [d for d in history if d.timestamp >= cutoff_time]

        return sorted(history, key=lambda d: d.timestamp, reverse=True)


class InfrastructureAsCode:
    """Infrastructure as Code management"""

    def __init__(self, project_dir: str = "/opt/pouw/infrastructure"):
        """Initialize IaC manager"""
        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.project_dir / "terraform.tfstate"
        self.config_hash: Optional[str] = None

    def generate_terraform_config(self, deployment_config: Dict[str, Any]) -> str:
        """Generate Terraform configuration for PoUW deployment"""
        terraform_config = """
# PoUW Infrastructure Configuration
# Generated automatically - do not edit manually

terraform {
  required_version = ">= 1.0"
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
  }
}

provider "kubernetes" {
  config_path = "~/.kube/config"
}

provider "helm" {
  kubernetes {
    config_path = "~/.kube/config"
  }
}

# Namespace
resource "kubernetes_namespace" "pouw_system" {
  metadata {
    name = "pouw-system"
    labels = {
      app         = "pouw"
      environment = "production"
      managed-by  = "terraform"
    }
  }
}

# ConfigMap for PoUW configuration
resource "kubernetes_config_map" "pouw_config" {
  metadata {
    name      = "pouw-config"
    namespace = kubernetes_namespace.pouw_system.metadata[0].name
  }
  
  data = {
"""

        # Add configuration data
        for key, value in deployment_config.get("config", {}).items():
            terraform_config += f'    "{key}" = "{value}"\n'

        terraform_config += """  }
}

# Secret for sensitive configuration
resource "kubernetes_secret" "pouw_secrets" {
  metadata {
    name      = "pouw-secrets"
    namespace = kubernetes_namespace.pouw_system.metadata[0].name
  }
  
  type = "Opaque"
  
  data = {
"""

        # Add secret data
        for key, value in deployment_config.get("secrets", {}).items():
            terraform_config += f'    "{key}" = base64encode("{value}")\n'

        terraform_config += """  }
}

# Persistent Volume for blockchain data
resource "kubernetes_persistent_volume_claim" "blockchain_data" {
  metadata {
    name      = "blockchain-data"
    namespace = kubernetes_namespace.pouw_system.metadata[0].name
  }
  
  spec {
    access_modes = ["ReadWriteOnce"]
    resources {
      requests = {
        storage = "100Gi"
      }
    }
  }
}

# Monitoring stack using Helm
resource "helm_release" "prometheus" {
  name       = "prometheus"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  namespace  = kubernetes_namespace.pouw_system.metadata[0].name
  
  values = [
    yamlencode({
      prometheus = {
        prometheusSpec = {
          storageSpec = {
            volumeClaimTemplate = {
              spec = {
                storageClassName = "standard"
                accessModes      = ["ReadWriteOnce"]
                resources = {
                  requests = {
                    storage = "50Gi"
                  }
                }
              }
            }
          }
        }
      }
      grafana = {
        adminPassword = "admin"
        persistence = {
          enabled = true
          size    = "10Gi"
        }
      }
    })
  ]
}

# Load balancer service
resource "kubernetes_service" "load_balancer" {
  metadata {
    name      = "pouw-load-balancer"
    namespace = kubernetes_namespace.pouw_system.metadata[0].name
  }
  
  spec {
    type = "LoadBalancer"
    
    selector = {
      app = "pouw"
    }
    
    port {
      name        = "http"
      port        = 80
      target_port = 8080
      protocol    = "TCP"
    }
    
    port {
      name        = "https"
      port        = 443
      target_port = 8443
      protocol    = "TCP"
    }
  }
}

# Outputs
output "namespace" {
  value = kubernetes_namespace.pouw_system.metadata[0].name
}

output "load_balancer_ip" {
  value = kubernetes_service.load_balancer.status[0].load_balancer[0].ingress[0].ip
}
"""

        return terraform_config

    async def deploy_infrastructure(self, deployment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy infrastructure using Terraform"""
        try:
            # Generate Terraform configuration
            terraform_config = self.generate_terraform_config(deployment_config)

            # Calculate configuration hash
            config_hash = hashlib.sha256(terraform_config.encode()).hexdigest()[:8]

            # Write configuration file
            config_file = self.project_dir / "main.tf"
            with open(config_file, "w") as f:
                f.write(terraform_config)

            # Initialize Terraform
            init_result = await self._run_terraform_command(["init"])
            if not init_result["success"]:
                return {
                    "success": False,
                    "error": "Terraform initialization failed",
                    "details": init_result,
                }

            # Plan deployment
            plan_result = await self._run_terraform_command(["plan", "-out=tfplan"])
            if not plan_result["success"]:
                return {
                    "success": False,
                    "error": "Terraform planning failed",
                    "details": plan_result,
                }

            # Apply deployment
            apply_result = await self._run_terraform_command(["apply", "-auto-approve", "tfplan"])

            if apply_result["success"]:
                self.config_hash = config_hash

                # Get outputs
                output_result = await self._run_terraform_command(["output", "-json"])
                outputs = {}
                if output_result["success"]:
                    try:
                        outputs = json.loads(output_result["stdout"])
                    except json.JSONDecodeError:
                        pass

                return {
                    "success": True,
                    "config_hash": config_hash,
                    "outputs": outputs,
                    "deployment_time": datetime.now().isoformat(),
                }
            else:
                return {
                    "success": False,
                    "error": "Terraform apply failed",
                    "details": apply_result,
                }

        except Exception as e:
            logger.error(f"Error deploying infrastructure: {e}")
            return {"success": False, "error": str(e)}

    async def _run_terraform_command(self, args: List[str]) -> Dict[str, Any]:
        """Run Terraform command and return result"""
        try:
            cmd = ["terraform"] + args

            result = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.project_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await result.communicate()

            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": stdout.decode() if stdout else "",
                "stderr": stderr.decode() if stderr else "",
                "command": " ".join(cmd),
            }

        except Exception as e:
            return {"success": False, "error": str(e), "command": " ".join(args)}

    async def destroy_infrastructure(self) -> Dict[str, Any]:
        """Destroy infrastructure"""
        try:
            result = await self._run_terraform_command(["destroy", "-auto-approve"])

            if result["success"]:
                # Clean up state file
                if self.state_file.exists():
                    self.state_file.unlink()

                self.config_hash = None

                return {"success": True, "message": "Infrastructure destroyed successfully"}
            else:
                return {"success": False, "error": "Terraform destroy failed", "details": result}

        except Exception as e:
            logger.error(f"Error destroying infrastructure: {e}")
            return {"success": False, "error": str(e)}


class DeploymentAutomation:
    """End-to-end deployment automation"""

    def __init__(self, namespace: str = "pouw-system"):
        """Initialize deployment automation"""
        self.namespace = namespace
        self.iac = InfrastructureAsCode()
        self.load_balancer = LoadBalancer()

    async def deploy_complete_stack(self, deployment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy complete PoUW stack with automation"""
        deployment_result = {
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "infrastructure": {},
            "load_balancer": {},
            "error": None,
        }

        try:
            logger.info("Starting complete stack deployment...")

            # Deploy infrastructure
            logger.info("Deploying infrastructure...")
            infra_result = await self.iac.deploy_infrastructure(deployment_config)
            deployment_result["infrastructure"] = infra_result

            if not infra_result["success"]:
                deployment_result["error"] = "Infrastructure deployment failed"
                return deployment_result

            # Deploy load balancer
            if "load_balancer" in deployment_config:
                logger.info("Deploying load balancer...")
                lb_config = LoadBalancerConfig(**deployment_config["load_balancer"])
                lb_success = await self.load_balancer.deploy_load_balancer(lb_config)
                deployment_result["load_balancer"] = {
                    "success": lb_success,
                    "config": deployment_config["load_balancer"],
                }

                if not lb_success:
                    logger.warning("Load balancer deployment failed, but continuing...")

            deployment_result["success"] = True
            logger.info("Complete stack deployment successful!")

        except Exception as e:
            logger.error(f"Error in complete stack deployment: {e}")
            deployment_result["error"] = str(e)

        return deployment_result


class ConfigurationManager:
    """Configuration management for PoUW deployments"""

    def __init__(self, config_dir: str = "/etc/pouw"):
        """Initialize configuration manager"""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.configs: Dict[str, Dict[str, Any]] = {}

    def load_configuration(self, config_name: str) -> Optional[Dict[str, Any]]:
        """Load configuration from file"""
        try:
            config_file = self.config_dir / f"{config_name}.yaml"
            if config_file.exists():
                with open(config_file, "r") as f:
                    config = yaml.safe_load(f)
                    self.configs[config_name] = config
                    return config
            return None

        except Exception as e:
            logger.error(f"Error loading configuration {config_name}: {e}")
            return None

    def save_configuration(self, config_name: str, config: Dict[str, Any]) -> bool:
        """Save configuration to file"""
        try:
            config_file = self.config_dir / f"{config_name}.yaml"
            with open(config_file, "w") as f:
                yaml.dump(config, f, default_flow_style=False)

            self.configs[config_name] = config
            logger.info(f"Saved configuration: {config_name}")
            return True

        except Exception as e:
            logger.error(f"Error saving configuration {config_name}: {e}")
            return False

    def get_default_deployment_config(self) -> Dict[str, Any]:
        """Get default deployment configuration"""
        return {
            "config": {
                "POUW_NETWORK_ID": "1337",
                "POUW_LOG_LEVEL": "info",
                "POUW_METRICS_ENABLED": "true",
                "POUW_VPN_ENABLED": "true",
            },
            "secrets": {"POUW_SECRET_KEY": "REDACTED", "POUW_DB_PASSWORD": "REDACTED"},
            "load_balancer": {
                "name": "pouw-api",
                "strategy": "round_robin",
                "backend_servers": [
                    {"host": "10.0.1.10", "port": 8080, "weight": 1},
                    {"host": "10.0.1.11", "port": 8080, "weight": 1},
                    {"host": "10.0.1.12", "port": 8080, "weight": 1},
                ],
                "health_check": {"path": "/health", "interval": 30, "timeout": 5},
            },
            "scaling": {"min_replicas": 2, "max_replicas": 10, "target_cpu_utilization": 70},
        }


class ResourceManager:
    """Resource management and optimization"""

    def __init__(self):
        """Initialize resource manager"""
        self.resource_usage: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    async def collect_resource_usage(self) -> Dict[str, Any]:
        """Collect current resource usage"""
        usage = {
            "timestamp": datetime.now().isoformat(),
            "cpu": {
                "percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count(),
                "freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            },
            "memory": psutil.virtual_memory()._asdict(),
            "disk": psutil.disk_usage("/")._asdict(),
            "network": psutil.net_io_counters()._asdict(),
        }

        # Store for trend analysis
        self.resource_usage["system"].append(usage)

        # Keep only recent data (last 1000 entries)
        if len(self.resource_usage["system"]) > 1000:
            self.resource_usage["system"] = self.resource_usage["system"][-1000:]

        return usage

    def get_resource_recommendations(self) -> List[str]:
        """Get resource optimization recommendations"""
        recommendations = []

        if not self.resource_usage["system"]:
            return ["No resource usage data available"]

        recent_usage = self.resource_usage["system"][-10:]  # Last 10 samples

        # CPU recommendations
        avg_cpu = sum(u["cpu"]["percent"] for u in recent_usage) / len(recent_usage)
        if avg_cpu > 80:
            recommendations.append(
                "High CPU usage detected. Consider scaling up or optimizing workloads."
            )
        elif avg_cpu < 20:
            recommendations.append("Low CPU usage detected. Consider scaling down to reduce costs.")

        # Memory recommendations
        avg_memory = sum(u["memory"]["percent"] for u in recent_usage) / len(recent_usage)
        if avg_memory > 85:
            recommendations.append(
                "High memory usage detected. Consider increasing memory or optimizing applications."
            )
        elif avg_memory < 30:
            recommendations.append(
                "Low memory usage detected. Consider reducing memory allocation."
            )

        # Disk recommendations
        latest_usage = recent_usage[-1]
        disk_percent = (latest_usage["disk"]["used"] / latest_usage["disk"]["total"]) * 100
        if disk_percent > 85:
            recommendations.append(
                "High disk usage detected. Consider adding storage or cleaning up old data."
            )

        if not recommendations:
            recommendations.append("Resource usage is within optimal ranges.")

        return recommendations
