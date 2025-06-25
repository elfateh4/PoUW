"""
Kubernetes Orchestration Module

Provides Kubernetes-based deployment and orchestration capabilities for the PoUW system.
Includes container management, service orchestration, and deployment automation.
"""

import asyncio
import json
import yaml
import subprocess
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import tempfile
import os

logger = logging.getLogger(__name__)


class DeploymentStatus(Enum):
    """Deployment status enumeration"""

    PENDING = "pending"
    DEPLOYING = "deploying"
    RUNNING = "running"
    SCALING = "scaling"
    UPDATING = "updating"
    FAILING = "failing"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ContainerConfiguration:
    """Container configuration for PoUW components"""

    name: str
    image: str
    tag: str = "latest"
    resources: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    ports: List[int] = field(default_factory=list)
    volumes: List[Dict[str, str]] = field(default_factory=list)
    command: Optional[List[str]] = None
    args: Optional[List[str]] = None
    health_check: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        config = {
            "name": self.name,
            "image": f"{self.image}:{self.tag}",
            "resources": self.resources
            or {
                "requests": {"memory": "512Mi", "cpu": "500m"},
                "limits": {"memory": "2Gi", "cpu": "2000m"},
            },
            "env": [{"name": k, "value": v} for k, v in self.environment.items()],
            "ports": [{"containerPort": port} for port in self.ports],
        }

        if self.command:
            config["command"] = self.command
        if self.args:
            config["args"] = self.args
        if self.volumes:
            config["volumeMounts"] = self.volumes
        if self.health_check:
            config.update(self.health_check)

        return config


@dataclass
class ServiceConfiguration:
    """Service configuration for Kubernetes services"""

    name: str
    selector: Dict[str, str]
    ports: List[Dict[str, Any]]
    service_type: str = "ClusterIP"
    external_traffic_policy: str = "Cluster"
    session_affinity: str = "None"
    annotations: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to Kubernetes service manifest"""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": self.name, "annotations": self.annotations},
            "spec": {
                "selector": self.selector,
                "ports": self.ports,
                "type": self.service_type,
                "externalTrafficPolicy": self.external_traffic_policy,
                "sessionAffinity": self.session_affinity,
            },
        }


class KubernetesOrchestrator:
    """Advanced Kubernetes orchestration for PoUW components"""

    def __init__(self, namespace: str = "pouw-system", kubeconfig: Optional[str] = None):
        """Initialize Kubernetes orchestrator"""
        self.namespace = namespace
        self.kubeconfig = kubeconfig
        self.kubectl_cmd = self._build_kubectl_command()

    def _build_kubectl_command(self) -> List[str]:
        """Build kubectl command with proper configuration"""
        cmd = ["kubectl"]
        if self.kubeconfig:
            cmd.extend(["--kubeconfig", self.kubeconfig])
        cmd.extend(["--namespace", self.namespace])
        return cmd

    async def create_namespace(self) -> bool:
        """Create namespace for PoUW deployment"""
        try:
            namespace_manifest = {
                "apiVersion": "v1",
                "kind": "Namespace",
                "metadata": {
                    "name": self.namespace,
                    "labels": {"app": "pouw", "environment": "production"},
                },
            }

            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                yaml.dump(namespace_manifest, f)
                temp_file = f.name

            try:
                cmd = ["kubectl", "apply", "-f", temp_file]
                if self.kubeconfig:
                    cmd = ["kubectl", "--kubeconfig", self.kubeconfig, "apply", "-f", temp_file]

                result = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )

                stdout, stderr = await result.communicate()

                if result.returncode == 0:
                    logger.info(f"Successfully created namespace: {self.namespace}")
                    return True
                else:
                    logger.error(f"Failed to create namespace: {stderr.decode()}")
                    return False

            finally:
                os.unlink(temp_file)

        except Exception as e:
            logger.error(f"Error creating namespace: {e}")
            return False

    async def deploy_pouw_components(
        self, components: Dict[str, ContainerConfiguration]
    ) -> Dict[str, DeploymentStatus]:
        """Deploy all PoUW components to Kubernetes"""
        deployment_status = {}

        # Create namespace first
        await self.create_namespace()

        for component_name, config in components.items():
            try:
                status = await self._deploy_component(component_name, config)
                deployment_status[component_name] = status
                logger.info(f"Deployed {component_name}: {status.value}")

            except Exception as e:
                logger.error(f"Failed to deploy {component_name}: {e}")
                deployment_status[component_name] = DeploymentStatus.ERROR

        return deployment_status

    async def _deploy_component(
        self, component_name: str, config: ContainerConfiguration
    ) -> DeploymentStatus:
        """Deploy individual PoUW component"""
        deployment_manifest = self._create_deployment_manifest(component_name, config)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(deployment_manifest, f)
            temp_file = f.name

        try:
            cmd = self.kubectl_cmd + ["apply", "-f", temp_file]

            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                # Wait for deployment to be ready
                await self._wait_for_deployment(component_name)
                return DeploymentStatus.RUNNING
            else:
                logger.error(f"Deployment failed for {component_name}: {stderr.decode()}")
                return DeploymentStatus.ERROR

        finally:
            os.unlink(temp_file)

    def _create_deployment_manifest(
        self, component_name: str, config: ContainerConfiguration
    ) -> Dict[str, Any]:
        """Create Kubernetes deployment manifest"""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"pouw-{component_name}",
                "namespace": self.namespace,
                "labels": {"app": "pouw", "component": component_name, "version": "v1.0.0"},
            },
            "spec": {
                "replicas": 1,
                "selector": {"matchLabels": {"app": "pouw", "component": component_name}},
                "template": {
                    "metadata": {"labels": {"app": "pouw", "component": component_name}},
                    "spec": {"containers": [config.to_dict()], "restartPolicy": "Always"},
                },
            },
        }

    async def _wait_for_deployment(self, component_name: str, timeout: int = 300) -> bool:
        """Wait for deployment to be ready"""
        cmd = self.kubectl_cmd + [
            "rollout",
            "status",
            f"deployment/pouw-{component_name}",
            f"--timeout={timeout}s",
        ]

        try:
            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await result.communicate()
            return result.returncode == 0

        except Exception as e:
            logger.error(f"Error waiting for deployment {component_name}: {e}")
            return False

    async def create_services(self, services: Dict[str, ServiceConfiguration]) -> Dict[str, bool]:
        """Create Kubernetes services for PoUW components"""
        service_status = {}

        for service_name, config in services.items():
            try:
                success = await self._create_service(config)
                service_status[service_name] = success
                logger.info(f"Created service {service_name}: {'success' if success else 'failed'}")

            except Exception as e:
                logger.error(f"Failed to create service {service_name}: {e}")
                service_status[service_name] = False

        return service_status

    async def _create_service(self, config: ServiceConfiguration) -> bool:
        """Create individual Kubernetes service"""
        service_manifest = config.to_dict()
        service_manifest["metadata"]["namespace"] = self.namespace

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(service_manifest, f)
            temp_file = f.name

        try:
            cmd = self.kubectl_cmd + ["apply", "-f", temp_file]

            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await result.communicate()
            return result.returncode == 0

        finally:
            os.unlink(temp_file)

    async def scale_deployment(self, component_name: str, replicas: int) -> bool:
        """Scale PoUW component deployment"""
        cmd = self.kubectl_cmd + [
            "scale",
            f"deployment/pouw-{component_name}",
            f"--replicas={replicas}",
        ]

        try:
            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                logger.info(f"Scaled {component_name} to {replicas} replicas")
                return True
            else:
                logger.error(f"Failed to scale {component_name}: {stderr.decode()}")
                return False

        except Exception as e:
            logger.error(f"Error scaling {component_name}: {e}")
            return False

    async def get_deployment_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all PoUW deployments"""
        cmd = self.kubectl_cmd + ["get", "deployments", "-o", "json"]

        try:
            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                data = json.loads(stdout.decode())
                status = {}

                for item in data.get("items", []):
                    name = item["metadata"]["name"]
                    if name.startswith("pouw-"):
                        component_name = name[5:]  # Remove 'pouw-' prefix
                        status[component_name] = {
                            "ready_replicas": item["status"].get("readyReplicas", 0),
                            "replicas": item["status"].get("replicas", 0),
                            "updated_replicas": item["status"].get("updatedReplicas", 0),
                            "available_replicas": item["status"].get("availableReplicas", 0),
                        }

                return status
            else:
                logger.error(f"Failed to get deployment status: {stderr.decode()}")
                return {}

        except Exception as e:
            logger.error(f"Error getting deployment status: {e}")
            return {}

    async def delete_deployment(self, component_name: str) -> bool:
        """Delete PoUW component deployment"""
        cmd = self.kubectl_cmd + ["delete", "deployment", f"pouw-{component_name}"]

        try:
            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                logger.info(f"Deleted deployment: pouw-{component_name}")
                return True
            else:
                logger.error(f"Failed to delete deployment {component_name}: {stderr.decode()}")
                return False

        except Exception as e:
            logger.error(f"Error deleting deployment {component_name}: {e}")
            return False


class PoUWDeploymentManager:
    """High-level PoUW deployment management"""

    def __init__(self, namespace: str = "pouw-system", kubeconfig: Optional[str] = None):
        """Initialize deployment manager"""
        self.orchestrator = KubernetesOrchestrator(namespace, kubeconfig)
        self.namespace = namespace

    def get_default_configurations(self) -> Dict[str, ContainerConfiguration]:
        """Get default container configurations for PoUW components"""
        return {
            "blockchain-node": ContainerConfiguration(
                name="blockchain-node",
                image="pouw/blockchain-node",
                tag="v1.0.0",
                ports=[8545, 8546, 30303],
                environment={
                    "POUW_NETWORK_ID": "1337",
                    "POUW_ROLE": "miner",
                    "POUW_LOG_LEVEL": "info",
                },
                resources={
                    "requests": {"memory": "1Gi", "cpu": "1000m"},
                    "limits": {"memory": "4Gi", "cpu": "4000m"},
                },
                health_check={
                    "livenessProbe": {
                        "httpGet": {"path": "/health", "port": 8545},
                        "initialDelaySeconds": 30,
                        "periodSeconds": 10,
                    },
                    "readinessProbe": {
                        "httpGet": {"path": "/ready", "port": 8545},
                        "initialDelaySeconds": 5,
                        "periodSeconds": 5,
                    },
                },
            ),
            "ml-trainer": ContainerConfiguration(
                name="ml-trainer",
                image="pouw/ml-trainer",
                tag="v1.0.0",
                ports=[8080],
                environment={
                    "POUW_TRAINING_MODE": "distributed",
                    "POUW_GPU_ENABLED": "true",
                    "POUW_BATCH_SIZE": "32",
                },
                resources={
                    "requests": {"memory": "2Gi", "cpu": "2000m", "nvidia.com/gpu": "1"},
                    "limits": {"memory": "8Gi", "cpu": "8000m", "nvidia.com/gpu": "1"},
                },
                health_check={
                    "livenessProbe": {
                        "httpGet": {"path": "/health", "port": 8080},
                        "initialDelaySeconds": 60,
                        "periodSeconds": 30,
                    }
                },
            ),
            "vpn-mesh": ContainerConfiguration(
                name="vpn-mesh",
                image="pouw/vpn-mesh",
                tag="v1.0.0",
                ports=[1194, 8081],
                environment={"POUW_VPN_MODE": "mesh", "POUW_ENCRYPTION": "aes-256-cbc"},
                resources={
                    "requests": {"memory": "512Mi", "cpu": "500m"},
                    "limits": {"memory": "2Gi", "cpu": "2000m"},
                },
                health_check={
                    "livenessProbe": {
                        "tcpSocket": {"port": 1194},
                        "initialDelaySeconds": 10,
                        "periodSeconds": 10,
                    }
                },
            ),
            "monitoring": ContainerConfiguration(
                name="monitoring",
                image="pouw/monitoring",
                tag="v1.0.0",
                ports=[3000, 9090],
                environment={"POUW_METRICS_RETENTION": "7d", "POUW_ALERT_ENABLED": "true"},
                resources={
                    "requests": {"memory": "1Gi", "cpu": "500m"},
                    "limits": {"memory": "4Gi", "cpu": "2000m"},
                },
            ),
        }

    def get_default_services(self) -> Dict[str, ServiceConfiguration]:
        """Get default service configurations for PoUW components"""
        return {
            "blockchain-api": ServiceConfiguration(
                name="pouw-blockchain-api",
                selector={"app": "pouw", "component": "blockchain-node"},
                ports=[
                    {"name": "rpc", "port": 8545, "targetPort": 8545, "protocol": "TCP"},
                    {"name": "ws", "port": 8546, "targetPort": 8546, "protocol": "TCP"},
                ],
                service_type="LoadBalancer",
            ),
            "ml-trainer-api": ServiceConfiguration(
                name="pouw-ml-trainer-api",
                selector={"app": "pouw", "component": "ml-trainer"},
                ports=[{"name": "api", "port": 8080, "targetPort": 8080, "protocol": "TCP"}],
                service_type="ClusterIP",
            ),
            "vpn-mesh-service": ServiceConfiguration(
                name="pouw-vpn-mesh",
                selector={"app": "pouw", "component": "vpn-mesh"},
                ports=[
                    {"name": "vpn", "port": 1194, "targetPort": 1194, "protocol": "UDP"},
                    {"name": "api", "port": 8081, "targetPort": 8081, "protocol": "TCP"},
                ],
                service_type="LoadBalancer",
            ),
            "monitoring-dashboard": ServiceConfiguration(
                name="pouw-monitoring",
                selector={"app": "pouw", "component": "monitoring"},
                ports=[
                    {"name": "grafana", "port": 3000, "targetPort": 3000, "protocol": "TCP"},
                    {"name": "prometheus", "port": 9090, "targetPort": 9090, "protocol": "TCP"},
                ],
                service_type="LoadBalancer",
            ),
        }

    async def deploy_full_stack(self) -> Dict[str, Any]:
        """Deploy complete PoUW stack to Kubernetes"""
        logger.info("Starting full PoUW stack deployment...")

        # Get default configurations
        components = self.get_default_configurations()
        services = self.get_default_services()

        deployment_results = {
            "namespace_created": False,
            "components_deployed": {},
            "services_created": {},
            "deployment_time": None,
            "status": "pending",
        }

        start_time = datetime.now()

        try:
            # Deploy components
            component_status = await self.orchestrator.deploy_pouw_components(components)
            deployment_results["components_deployed"] = component_status

            # Create services
            service_status = await self.orchestrator.create_services(services)
            deployment_results["services_created"] = service_status

            # Check overall success
            all_components_ok = all(
                status == DeploymentStatus.RUNNING for status in component_status.values()
            )
            all_services_ok = all(service_status.values())

            if all_components_ok and all_services_ok:
                deployment_results["status"] = "success"
                logger.info("PoUW stack deployment completed successfully")
            else:
                deployment_results["status"] = "partial_success"
                logger.warning("PoUW stack deployment completed with some failures")

        except Exception as e:
            deployment_results["status"] = "failed"
            logger.error(f"PoUW stack deployment failed: {e}")

        finally:
            end_time = datetime.now()
            deployment_results["deployment_time"] = (end_time - start_time).total_seconds()

        return deployment_results

    async def scale_components(self, scaling_config: Dict[str, int]) -> Dict[str, bool]:
        """Scale PoUW components based on configuration"""
        scaling_results = {}

        for component, replicas in scaling_config.items():
            try:
                success = await self.orchestrator.scale_deployment(component, replicas)
                scaling_results[component] = success
                logger.info(
                    f"Scaled {component} to {replicas} replicas: {'success' if success else 'failed'}"
                )

            except Exception as e:
                logger.error(f"Error scaling {component}: {e}")
                scaling_results[component] = False

        return scaling_results

    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status"""
        deployment_status = await self.orchestrator.get_deployment_status()

        cluster_status = {
            "namespace": self.namespace,
            "deployments": deployment_status,
            "total_components": len(deployment_status),
            "healthy_components": sum(
                1 for status in deployment_status.values() if status.get("ready_replicas", 0) > 0
            ),
            "timestamp": datetime.now().isoformat(),
        }

        # Calculate overall health
        if cluster_status["total_components"] == 0:
            cluster_status["health"] = "unknown"
        elif cluster_status["healthy_components"] == cluster_status["total_components"]:
            cluster_status["health"] = "healthy"
        elif cluster_status["healthy_components"] > 0:
            cluster_status["health"] = "degraded"
        else:
            cluster_status["health"] = "unhealthy"

        return cluster_status

    async def cleanup_deployment(self) -> Dict[str, bool]:
        """Clean up PoUW deployment"""
        logger.info("Starting PoUW deployment cleanup...")

        cleanup_results = {}
        components = self.get_default_configurations()

        for component_name in components.keys():
            try:
                success = await self.orchestrator.delete_deployment(component_name)
                cleanup_results[component_name] = success

            except Exception as e:
                logger.error(f"Error cleaning up {component_name}: {e}")
                cleanup_results[component_name] = False

        return cleanup_results
