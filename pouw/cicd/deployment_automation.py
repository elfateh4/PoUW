"""
Deployment Automation Module for PoUW CI/CD Pipeline

This module provides comprehensive deployment automation capabilities including:
- Deployment pipeline management and orchestration
- Release management and versioning
- Environment management (dev, staging, production)
- Blue-green and canary deployment strategies
- Rollback and disaster recovery automation
- Infrastructure as Code (IaC) management
"""

import asyncio
import json
import subprocess
import time
import yaml
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import tempfile
import shutil

import docker
from kubernetes import client, config as k8s_config
import boto3
from azure.identity import DefaultAzureCredential
from azure.mgmt.containerinstance import ContainerInstanceManagementClient


class DeploymentStrategy(Enum):
    """Deployment strategies supported by the automation framework."""
    ROLLING_UPDATE = "rolling_update"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    RECREATE = "recreate"
    A_B_TESTING = "a_b_testing"


class Environment(Enum):
    """Target deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"
    PREVIEW = "preview"


class DeploymentStatus(Enum):
    """Deployment execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


class PlatformType(Enum):
    """Supported deployment platforms."""
    KUBERNETES = "kubernetes"
    DOCKER = "docker"
    AWS_ECS = "aws_ecs"
    AWS_LAMBDA = "aws_lambda"
    AZURE_CONTAINER = "azure_container"
    GOOGLE_CLOUD_RUN = "google_cloud_run"
    HEROKU = "heroku"
    BARE_METAL = "bare_metal"


@dataclass
class DeploymentConfiguration:
    """Configuration for deployment execution."""
    name: str
    environment: Environment
    strategy: DeploymentStrategy
    platform: PlatformType
    image_tag: str
    replicas: int = 3
    cpu_request: str = "100m"
    memory_request: str = "128Mi"
    cpu_limit: str = "500m"
    memory_limit: str = "512Mi"
    health_check_path: str = "/health"
    timeout: int = 600  # seconds
    rollback_on_failure: bool = True
    enable_monitoring: bool = True
    enable_logging: bool = True
    env_vars: Dict[str, str] = field(default_factory=dict)
    secrets: Dict[str, str] = field(default_factory=dict)
    volumes: List[Dict[str, Any]] = field(default_factory=list)
    ingress_rules: List[Dict[str, Any]] = field(default_factory=list)
    auto_scaling: Optional[Dict[str, Any]] = None
    annotations: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class DeploymentResult:
    """Result of deployment execution."""
    deployment_id: str
    name: str
    environment: Environment
    strategy: DeploymentStrategy
    status: DeploymentStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: float = 0.0
    endpoints: List[str] = field(default_factory=list)
    logs: str = ""
    error_message: str = ""
    rollback_info: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReleaseInfo:
    """Information about a software release."""
    version: str
    tag: str
    branch: str
    commit_hash: str
    release_notes: str = ""
    artifacts: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    is_prerelease: bool = False
    deployments: List[DeploymentResult] = field(default_factory=list)


class DeploymentPipelineManager:
    """
    Manages deployment pipelines across different environments and platforms.
    
    Provides capabilities for:
    - Multi-environment deployment orchestration
    - Different deployment strategies
    - Platform-agnostic deployment automation
    - Rollback and disaster recovery
    """

    def __init__(self, project_root: str = "/home/elfateh/Projects/PoUW"):
        self.project_root = Path(project_root)
        self.docker_client = docker.from_env()
        self.deployments: Dict[str, DeploymentResult] = {}
        self.releases: Dict[str, ReleaseInfo] = {}
        
        # Initialize cloud clients
        self._init_cloud_clients()
        
        # Load Kubernetes config
        try:
            k8s_config.load_incluster_config()
        except:
            try:
                k8s_config.load_kube_config()
            except:
                print("Warning: Kubernetes config not available")

    def _init_cloud_clients(self):
        """Initialize cloud platform clients."""
        try:
            # AWS clients
            self.aws_session = boto3.Session()
            self.ecs_client = self.aws_session.client('ecs')
            self.lambda_client = self.aws_session.client('lambda')
            self.ecr_client = self.aws_session.client('ecr')
        except:
            print("Warning: AWS credentials not available")
        
        try:
            # Azure clients
            self.azure_credential = DefaultAzureCredential()
            # Initialize other Azure clients as needed
        except:
            print("Warning: Azure credentials not available")

    async def deploy(self, config: DeploymentConfiguration) -> DeploymentResult:
        """Execute a deployment with the given configuration."""
        deployment_id = f"{config.name}_{config.environment.value}_{int(time.time())}"
        
        print(f"Starting deployment: {deployment_id}")
        
        result = DeploymentResult(
            deployment_id=deployment_id,
            name=config.name,
            environment=config.environment,
            strategy=config.strategy,
            status=DeploymentStatus.PENDING,
            start_time=datetime.now()
        )
        
        try:
            result.status = DeploymentStatus.IN_PROGRESS
            
            # Execute deployment based on platform
            if config.platform == PlatformType.KUBERNETES:
                await self._deploy_to_kubernetes(config, result)
            elif config.platform == PlatformType.DOCKER:
                await self._deploy_to_docker(config, result)
            elif config.platform == PlatformType.AWS_ECS:
                await self._deploy_to_aws_ecs(config, result)
            elif config.platform == PlatformType.AWS_LAMBDA:
                await self._deploy_to_aws_lambda(config, result)
            elif config.platform == PlatformType.AZURE_CONTAINER:
                await self._deploy_to_azure_container(config, result)
            else:
                raise ValueError(f"Unsupported platform: {config.platform}")
            
            result.status = DeploymentStatus.DEPLOYED
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()
            
        except Exception as e:
            result.status = DeploymentStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()
            
            # Attempt rollback if enabled
            if config.rollback_on_failure:
                await self._rollback_deployment(config, result)
        
        # Store deployment result
        self.deployments[deployment_id] = result
        
        print(f"Deployment completed: {deployment_id} - Status: {result.status.value}")
        return result

    async def _deploy_to_kubernetes(self, config: DeploymentConfiguration, result: DeploymentResult):
        """Deploy to Kubernetes cluster."""
        # Generate Kubernetes manifests
        manifests = self._generate_k8s_manifests(config)
        
        # Apply manifests based on strategy
        if config.strategy == DeploymentStrategy.ROLLING_UPDATE:
            await self._k8s_rolling_update(manifests, config, result)
        elif config.strategy == DeploymentStrategy.BLUE_GREEN:
            await self._k8s_blue_green_deployment(manifests, config, result)
        elif config.strategy == DeploymentStrategy.CANARY:
            await self._k8s_canary_deployment(manifests, config, result)
        else:
            await self._k8s_rolling_update(manifests, config, result)
        
        # Get service endpoints
        result.endpoints = await self._get_k8s_endpoints(config)

    def _generate_k8s_manifests(self, config: DeploymentConfiguration) -> Dict[str, Any]:
        """Generate Kubernetes deployment manifests."""
        namespace = f"pouw-{config.environment.value}"
        
        # Deployment manifest
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": config.name,
                "namespace": namespace,
                "labels": {
                    "app": config.name,
                    "environment": config.environment.value,
                    **config.labels
                },
                "annotations": config.annotations
            },
            "spec": {
                "replicas": config.replicas,
                "selector": {
                    "matchLabels": {
                        "app": config.name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": config.name,
                            "environment": config.environment.value
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": config.name,
                            "image": config.image_tag,
                            "ports": [{"containerPort": 8000}],
                            "resources": {
                                "requests": {
                                    "cpu": config.cpu_request,
                                    "memory": config.memory_request
                                },
                                "limits": {
                                    "cpu": config.cpu_limit,
                                    "memory": config.memory_limit
                                }
                            },
                            "env": [
                                {"name": key, "value": value}
                                for key, value in config.env_vars.items()
                            ],
                            "livenessProbe": {
                                "httpGet": {
                                    "path": config.health_check_path,
                                    "port": 8000
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": config.health_check_path,
                                    "port": 8000
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            }
                        }]
                    }
                }
            }
        }
        
        # Service manifest
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{config.name}-service",
                "namespace": namespace,
                "labels": {
                    "app": config.name
                }
            },
            "spec": {
                "selector": {
                    "app": config.name
                },
                "ports": [{
                    "port": 80,
                    "targetPort": 8000,
                    "protocol": "TCP"
                }],
                "type": "ClusterIP"
            }
        }
        
        # HPA manifest (if auto-scaling enabled)
        hpa = None
        if config.auto_scaling:
            hpa = {
                "apiVersion": "autoscaling/v2",
                "kind": "HorizontalPodAutoscaler",
                "metadata": {
                    "name": f"{config.name}-hpa",
                    "namespace": namespace
                },
                "spec": {
                    "scaleTargetRef": {
                        "apiVersion": "apps/v1",
                        "kind": "Deployment",
                        "name": config.name
                    },
                    "minReplicas": config.auto_scaling.get("min_replicas", 2),
                    "maxReplicas": config.auto_scaling.get("max_replicas", 10),
                    "metrics": [{
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": config.auto_scaling.get("cpu_target", 70)
                            }
                        }
                    }]
                }
            }
        
        manifests = {
            "deployment": deployment,
            "service": service
        }
        
        if hpa:
            manifests["hpa"] = hpa
        
        return manifests

    async def _k8s_rolling_update(self, manifests: Dict[str, Any], config: DeploymentConfiguration, result: DeploymentResult):
        """Perform rolling update deployment to Kubernetes."""
        namespace = f"pouw-{config.environment.value}"
        
        # Create namespace if it doesn't exist
        await self._create_k8s_namespace(namespace)
        
        # Apply manifests
        for name, manifest in manifests.items():
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(manifest, f)
                temp_file = f.name
            
            try:
                cmd = ["kubectl", "apply", "-f", temp_file]
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()
                
                if proc.returncode != 0:
                    raise Exception(f"kubectl apply failed: {stderr.decode()}")
                
                result.logs += f"Applied {name}: {stdout.decode()}\n"
                
            finally:
                Path(temp_file).unlink()
        
        # Wait for rollout to complete
        cmd = ["kubectl", "rollout", "status", f"deployment/{config.name}", "-n", namespace, f"--timeout={config.timeout}s"]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            raise Exception(f"Rollout failed: {stderr.decode()}")
        
        result.logs += f"Rollout status: {stdout.decode()}\n"

    async def _k8s_blue_green_deployment(self, manifests: Dict[str, Any], config: DeploymentConfiguration, result: DeploymentResult):
        """Perform blue-green deployment to Kubernetes."""
        namespace = f"pouw-{config.environment.value}"
        
        # Modify deployment name for blue-green
        current_color = await self._get_current_k8s_color(config.name, namespace)
        new_color = "green" if current_color == "blue" else "blue"
        
        # Update deployment name with color
        manifests["deployment"]["metadata"]["name"] = f"{config.name}-{new_color}"
        manifests["deployment"]["metadata"]["labels"]["color"] = new_color
        manifests["deployment"]["spec"]["template"]["metadata"]["labels"]["color"] = new_color
        
        # Deploy new version
        await self._k8s_rolling_update(manifests, config, result)
        
        # Wait for new deployment to be ready
        await asyncio.sleep(30)
        
        # Switch service to new deployment
        service_patch = {
            "spec": {
                "selector": {
                    "app": config.name,
                    "color": new_color
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(service_patch, f)
            temp_file = f.name
        
        try:
            cmd = ["kubectl", "patch", "service", f"{config.name}-service", "-n", namespace, "--patch-file", temp_file]
            proc = await asyncio.create_subprocess_exec(*cmd)
            await proc.communicate()
            
            if proc.returncode == 0:
                # Clean up old deployment
                old_deployment = f"{config.name}-{current_color}"
                cmd = ["kubectl", "delete", "deployment", old_deployment, "-n", namespace]
                await asyncio.create_subprocess_exec(*cmd)
                
        finally:
            Path(temp_file).unlink()

    async def _k8s_canary_deployment(self, manifests: Dict[str, Any], config: DeploymentConfiguration, result: DeploymentResult):
        """Perform canary deployment to Kubernetes."""
        namespace = f"pouw-{config.environment.value}"
        
        # Deploy canary version with reduced replicas
        canary_replicas = max(1, config.replicas // 4)  # 25% traffic
        
        manifests["deployment"]["metadata"]["name"] = f"{config.name}-canary"
        manifests["deployment"]["spec"]["replicas"] = canary_replicas
        manifests["deployment"]["metadata"]["labels"]["version"] = "canary"
        manifests["deployment"]["spec"]["template"]["metadata"]["labels"]["version"] = "canary"
        
        # Deploy canary
        await self._k8s_rolling_update(manifests, config, result)
        
        # Monitor canary metrics (simplified)
        await asyncio.sleep(60)  # Monitor for 1 minute
        
        # If canary is healthy, promote to full deployment
        # This is a simplified version - in practice you'd check metrics
        canary_healthy = True  # Check actual health metrics here
        
        if canary_healthy:
            # Scale up canary to full replicas
            cmd = ["kubectl", "scale", "deployment", f"{config.name}-canary", f"--replicas={config.replicas}", "-n", namespace]
            await asyncio.create_subprocess_exec(*cmd)
            
            # Remove old deployment
            cmd = ["kubectl", "delete", "deployment", config.name, "-n", namespace, "--ignore-not-found=true"]
            await asyncio.create_subprocess_exec(*cmd)
            
            # Rename canary to main deployment
            cmd = ["kubectl", "patch", "deployment", f"{config.name}-canary", "-n", namespace, "--type=json", 
                   f"-p=[{{\"op\": \"replace\", \"path\": \"/metadata/name\", \"value\": \"{config.name}\"}}]"]
            await asyncio.create_subprocess_exec(*cmd)
        else:
            # Rollback canary
            cmd = ["kubectl", "delete", "deployment", f"{config.name}-canary", "-n", namespace]
            await asyncio.create_subprocess_exec(*cmd)
            raise Exception("Canary deployment failed health checks")

    async def _create_k8s_namespace(self, namespace: str):
        """Create Kubernetes namespace if it doesn't exist."""
        cmd = ["kubectl", "create", "namespace", namespace]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await proc.communicate()
        # Ignore error if namespace already exists

    async def _get_current_k8s_color(self, app_name: str, namespace: str) -> str:
        """Get current color for blue-green deployment."""
        cmd = ["kubectl", "get", "deployment", "-n", namespace, "-l", f"app={app_name}", "-o", "jsonpath={.items[0].metadata.labels.color}"]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        
        current_color = stdout.decode().strip()
        return current_color if current_color in ["blue", "green"] else "blue"

    async def _get_k8s_endpoints(self, config: DeploymentConfiguration) -> List[str]:
        """Get Kubernetes service endpoints."""
        namespace = f"pouw-{config.environment.value}"
        
        cmd = ["kubectl", "get", "service", f"{config.name}-service", "-n", namespace, "-o", "jsonpath={.status.loadBalancer.ingress[0].ip}"]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        
        ip = stdout.decode().strip()
        if ip:
            return [f"http://{ip}"]
        else:
            # Try to get cluster IP
            cmd = ["kubectl", "get", "service", f"{config.name}-service", "-n", namespace, "-o", "jsonpath={.spec.clusterIP}"]
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE)
            stdout, _ = await proc.communicate()
            cluster_ip = stdout.decode().strip()
            return [f"http://{cluster_ip}"] if cluster_ip else []

    async def _deploy_to_docker(self, config: DeploymentConfiguration, result: DeploymentResult):
        """Deploy to Docker (standalone or Swarm)."""
        container_name = f"{config.name}_{config.environment.value}"
        
        # Stop and remove existing container
        try:
            existing_container = self.docker_client.containers.get(container_name)
            existing_container.stop()
            existing_container.remove()
        except docker.errors.NotFound:
            pass
        
        # Create and start new container
        container = self.docker_client.containers.run(
            config.image_tag,
            name=container_name,
            environment=config.env_vars,
            ports={"8000/tcp": None},  # Random port
            detach=True,
            restart_policy={"Name": "unless-stopped"}
        )
        
        # Wait for container to be ready
        await asyncio.sleep(10)
        container.reload()
        
        if container.status != "running":
            raise Exception(f"Container failed to start: {container.status}")
        
        # Get container port mapping
        port_mapping = container.attrs["NetworkSettings"]["Ports"]["8000/tcp"]
        if port_mapping:
            host_port = port_mapping[0]["HostPort"]
            result.endpoints = [f"http://localhost:{host_port}"]
        
        result.logs = container.logs().decode()

    async def _deploy_to_aws_ecs(self, config: DeploymentConfiguration, result: DeploymentResult):
        """Deploy to AWS ECS."""
        cluster_name = f"pouw-{config.environment.value}"
        service_name = f"{config.name}-service"
        
        # Define task definition
        task_definition = {
            "family": config.name,
            "networkMode": "awsvpc",
            "requiresCompatibilities": ["FARGATE"],
            "cpu": "256",
            "memory": "512",
            "executionRoleArn": "arn:aws:iam::YOUR_ACCOUNT:role/ecsTaskExecutionRole",
            "containerDefinitions": [{
                "name": config.name,
                "image": config.image_tag,
                "portMappings": [{
                    "containerPort": 8000,
                    "protocol": "tcp"
                }],
                "environment": [
                    {"name": key, "value": value}
                    for key, value in config.env_vars.items()
                ],
                "logConfiguration": {
                    "logDriver": "awslogs",
                    "options": {
                        "awslogs-group": f"/ecs/{config.name}",
                        "awslogs-region": "us-east-1",
                        "awslogs-stream-prefix": "ecs"
                    }
                }
            }]
        }
        
        # Register task definition
        response = self.ecs_client.register_task_definition(**task_definition)
        task_def_arn = response["taskDefinition"]["taskDefinitionArn"]
        
        # Update or create service
        try:
            self.ecs_client.update_service(
                cluster=cluster_name,
                service=service_name,
                taskDefinition=task_def_arn,
                desiredCount=config.replicas
            )
        except self.ecs_client.exceptions.ServiceNotFoundException:
            # Create new service
            self.ecs_client.create_service(
                cluster=cluster_name,
                serviceName=service_name,
                taskDefinition=task_def_arn,
                desiredCount=config.replicas,
                launchType="FARGATE",
                networkConfiguration={
                    "awsvpcConfiguration": {
                        "subnets": ["subnet-12345"],  # Replace with actual subnet IDs
                        "securityGroups": ["sg-12345"],  # Replace with actual security group
                        "assignPublicIp": "ENABLED"
                    }
                }
            )
        
        # Wait for deployment to stabilize
        waiter = self.ecs_client.get_waiter('services_stable')
        waiter.wait(
            cluster=cluster_name,
            services=[service_name],
            WaiterConfig={'maxAttempts': 30, 'delay': 30}
        )

    async def _deploy_to_aws_lambda(self, config: DeploymentConfiguration, result: DeploymentResult):
        """Deploy to AWS Lambda."""
        function_name = f"{config.name}-{config.environment.value}"
        
        # Check if function exists
        try:
            self.lambda_client.get_function(FunctionName=function_name)
            # Update existing function
            self.lambda_client.update_function_code(
                FunctionName=function_name,
                ImageUri=config.image_tag
            )
            
            # Update configuration
            self.lambda_client.update_function_configuration(
                FunctionName=function_name,
                Environment={"Variables": config.env_vars},
                Timeout=config.timeout
            )
            
        except self.lambda_client.exceptions.ResourceNotFoundException:
            # Create new function
            self.lambda_client.create_function(
                FunctionName=function_name,
                Role="arn:aws:iam::YOUR_ACCOUNT:role/lambda-execution-role",
                Code={"ImageUri": config.image_tag},
                PackageType="Image",
                Environment={"Variables": config.env_vars},
                Timeout=config.timeout
            )
        
        # Get function URL if it exists
        try:
            url_config = self.lambda_client.get_function_url_config(FunctionName=function_name)
            result.endpoints = [url_config["FunctionUrl"]]
        except:
            pass

    async def _deploy_to_azure_container(self, config: DeploymentConfiguration, result: DeploymentResult):
        """Deploy to Azure Container Instances."""
        # This is a simplified implementation
        # In practice, you'd use the Azure SDK properly
        
        resource_group = f"pouw-{config.environment.value}"
        container_name = f"{config.name}-{config.environment.value}"
        
        # Azure CLI deployment (simplified)
        cmd = [
            "az", "container", "create",
            "--resource-group", resource_group,
            "--name", container_name,
            "--image", config.image_tag,
            "--cpu", "1",
            "--memory", "1",
            "--restart-policy", "Always",
            "--ports", "80"
        ]
        
        # Add environment variables
        for key, value in config.env_vars.items():
            cmd.extend(["--environment-variables", f"{key}={value}"])
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            raise Exception(f"Azure deployment failed: {stderr.decode()}")
        
        result.logs = stdout.decode()

    async def _rollback_deployment(self, config: DeploymentConfiguration, result: DeploymentResult):
        """Perform deployment rollback."""
        print(f"Attempting rollback for deployment: {result.deployment_id}")
        
        try:
            if config.platform == PlatformType.KUBERNETES:
                namespace = f"pouw-{config.environment.value}"
                cmd = ["kubectl", "rollout", "undo", f"deployment/{config.name}", "-n", namespace]
                proc = await asyncio.create_subprocess_exec(*cmd)
                await proc.communicate()
                
                if proc.returncode == 0:
                    result.status = DeploymentStatus.ROLLED_BACK
                    result.rollback_info = {"method": "kubernetes_rollout_undo"}
            
            elif config.platform == PlatformType.DOCKER:
                # For Docker, we'd need to keep track of previous image tags
                # This is a simplified version
                result.rollback_info = {"method": "docker_previous_image"}
            
        except Exception as e:
            result.error_message += f"\nRollback failed: {str(e)}"

    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentResult]:
        """Get the status of a specific deployment."""
        return self.deployments.get(deployment_id)

    def list_deployments(self, environment: Optional[Environment] = None) -> List[DeploymentResult]:
        """List all deployments, optionally filtered by environment."""
        deployments = list(self.deployments.values())
        
        if environment:
            deployments = [d for d in deployments if d.environment == environment]
        
        return sorted(deployments, key=lambda x: x.start_time, reverse=True)

    async def cleanup_old_deployments(self, environment: Environment, keep_count: int = 5):
        """Clean up old deployments, keeping only the specified number."""
        deployments = self.list_deployments(environment)
        
        if len(deployments) > keep_count:
            old_deployments = deployments[keep_count:]
            
            for deployment in old_deployments:
                print(f"Cleaning up old deployment: {deployment.deployment_id}")
                # Implementation depends on platform
                # For now, just remove from our tracking
                del self.deployments[deployment.deployment_id]


class ReleaseManager:
    """
    Manages software releases including versioning, tagging, and release automation.
    """

    def __init__(self, project_root: str = "/home/elfateh/Projects/PoUW"):
        self.project_root = Path(project_root)
        self.releases: Dict[str, ReleaseInfo] = {}

    async def create_release(self, version: str, branch: str = "main", 
                           release_notes: str = "", is_prerelease: bool = False) -> ReleaseInfo:
        """Create a new software release."""
        print(f"Creating release: {version}")
        
        # Get current commit hash
        proc = await asyncio.create_subprocess_exec(
            "git", "rev-parse", "HEAD",
            cwd=self.project_root,
            stdout=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        commit_hash = stdout.decode().strip()
        
        # Create git tag
        tag = f"v{version}"
        await asyncio.create_subprocess_exec(
            "git", "tag", "-a", tag, "-m", f"Release {version}",
            cwd=self.project_root
        )
        
        # Build release artifacts
        artifacts = await self._build_release_artifacts(version)
        
        release = ReleaseInfo(
            version=version,
            tag=tag,
            branch=branch,
            commit_hash=commit_hash,
            release_notes=release_notes,
            artifacts=artifacts,
            is_prerelease=is_prerelease
        )
        
        self.releases[version] = release
        
        print(f"Release created: {version}")
        return release

    async def _build_release_artifacts(self, version: str) -> List[str]:
        """Build release artifacts."""
        artifacts = []
        
        # Build Docker images
        image_tag = f"pouw:{version}"
        proc = await asyncio.create_subprocess_exec(
            "docker", "build", "-t", image_tag, ".",
            cwd=self.project_root
        )
        await proc.communicate()
        
        if proc.returncode == 0:
            artifacts.append(image_tag)
        
        # Create source archive
        archive_name = f"pouw-{version}.tar.gz"
        proc = await asyncio.create_subprocess_exec(
            "git", "archive", "--format=tar.gz", f"--output={archive_name}", "HEAD",
            cwd=self.project_root
        )
        await proc.communicate()
        
        if proc.returncode == 0:
            artifacts.append(archive_name)
        
        return artifacts

    async def deploy_release(self, version: str, environment: Environment, 
                           deployment_manager: DeploymentPipelineManager) -> DeploymentResult:
        """Deploy a specific release to an environment."""
        release = self.releases.get(version)
        if not release:
            raise ValueError(f"Release {version} not found")
        
        # Find Docker image artifact
        image_tag = None
        for artifact in release.artifacts:
            if artifact.startswith("pouw:"):
                image_tag = artifact
                break
        
        if not image_tag:
            raise ValueError(f"No Docker image found for release {version}")
        
        # Create deployment configuration
        config = DeploymentConfiguration(
            name="pouw",
            environment=environment,
            strategy=DeploymentStrategy.ROLLING_UPDATE,
            platform=PlatformType.KUBERNETES,
            image_tag=image_tag,
            labels={"version": version, "release": "true"}
        )
        
        # Execute deployment
        result = await deployment_manager.deploy(config)
        
        # Track deployment in release
        release.deployments.append(result)
        
        return result

    def get_release(self, version: str) -> Optional[ReleaseInfo]:
        """Get information about a specific release."""
        return self.releases.get(version)

    def list_releases(self, include_prereleases: bool = True) -> List[ReleaseInfo]:
        """List all releases."""
        releases = list(self.releases.values())
        
        if not include_prereleases:
            releases = [r for r in releases if not r.is_prerelease]
        
        return sorted(releases, key=lambda x: x.created_at, reverse=True)

    def get_latest_release(self, include_prereleases: bool = False) -> Optional[ReleaseInfo]:
        """Get the latest release."""
        releases = self.list_releases(include_prereleases)
        return releases[0] if releases else None


# Example usage and predefined configurations
class PoUWDeploymentConfigurations:
    """Predefined deployment configurations for PoUW system."""

    @staticmethod
    def development() -> DeploymentConfiguration:
        """Development environment configuration."""
        return DeploymentConfiguration(
            name="pouw",
            environment=Environment.DEVELOPMENT,
            strategy=DeploymentStrategy.RECREATE,
            platform=PlatformType.KUBERNETES,
            image_tag="pouw:latest",
            replicas=1,
            cpu_request="50m",
            memory_request="64Mi",
            cpu_limit="200m",
            memory_limit="256Mi"
        )

    @staticmethod
    def staging() -> DeploymentConfiguration:
        """Staging environment configuration."""
        return DeploymentConfiguration(
            name="pouw",
            environment=Environment.STAGING,
            strategy=DeploymentStrategy.ROLLING_UPDATE,
            platform=PlatformType.KUBERNETES,
            image_tag="pouw:staging",
            replicas=2,
            auto_scaling={
                "min_replicas": 2,
                "max_replicas": 5,
                "cpu_target": 70
            }
        )

    @staticmethod
    def production() -> DeploymentConfiguration:
        """Production environment configuration."""
        return DeploymentConfiguration(
            name="pouw",
            environment=Environment.PRODUCTION,
            strategy=DeploymentStrategy.BLUE_GREEN,
            platform=PlatformType.KUBERNETES,
            image_tag="pouw:v1.0.0",
            replicas=5,
            auto_scaling={
                "min_replicas": 5,
                "max_replicas": 20,
                "cpu_target": 60
            },
            enable_monitoring=True,
            enable_logging=True
        )


# Example usage
async def main():
    """Example usage of the deployment automation system."""
    # Initialize managers
    deployment_manager = DeploymentPipelineManager()
    release_manager = ReleaseManager()
    
    # Create a release
    release = await release_manager.create_release(
        version="1.0.0",
        release_notes="Initial production release"
    )
    
    print(f"Created release: {release.version}")
    
    # Deploy to staging
    staging_config = PoUWDeploymentConfigurations.staging()
    staging_config.image_tag = f"pouw:{release.version}"
    
    staging_result = await deployment_manager.deploy(staging_config)
    print(f"Staging deployment: {staging_result.status.value}")
    
    # If staging is successful, deploy to production
    if staging_result.status == DeploymentStatus.DEPLOYED:
        prod_config = PoUWDeploymentConfigurations.production()
        prod_config.image_tag = f"pouw:{release.version}"
        
        prod_result = await deployment_manager.deploy(prod_config)
        print(f"Production deployment: {prod_result.status.value}")


if __name__ == "__main__":
    asyncio.run(main())
