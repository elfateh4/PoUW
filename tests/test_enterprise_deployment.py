"""
Comprehensive test suite for Enterprise Deployment Infrastructure.

Tests Kubernetes orchestration, production monitoring, load balancing,
auto-scaling, and infrastructure automation capabilities.
"""

import pytest
import asyncio
import tempfile
import yaml
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from pouw.deployment.kubernetes import (
    KubernetesOrchestrator,
    PoUWDeploymentManager,
    ContainerConfiguration,
    ServiceConfiguration,
    DeploymentStatus
)
from pouw.deployment.monitoring import (
    ProductionMonitor,
    MetricsCollector,
    AlertingSystem,
    LoggingManager,
    HealthChecker,
    PerformanceAnalyzer,
    Metric,
    Alert,
    AlertSeverity,
    MetricType,
    HealthStatus
)
from pouw.deployment.infrastructure import (
    LoadBalancer,
    AutoScaler,
    InfrastructureAsCode,
    DeploymentAutomation,
    ConfigurationManager,
    ResourceManager,
    LoadBalancerConfig,
    LoadBalancingStrategy,
    AutoScalingRule,
    ScalingDirection
)


class TestKubernetesOrchestrator:
    """Test Kubernetes orchestration capabilities"""
    
    @pytest.fixture
    def orchestrator(self):
        """Create Kubernetes orchestrator for testing"""
        return KubernetesOrchestrator(namespace="test-pouw", kubeconfig=None)
    
    @pytest.fixture
    def container_config(self):
        """Create test container configuration"""
        return ContainerConfiguration(
            name="test-container",
            image="pouw/test",
            tag="v1.0.0",
            ports=[8080, 8443],
            environment={"TEST_VAR": "test_value"},
            resources={
                'requests': {'memory': '512Mi', 'cpu': '500m'},
                'limits': {'memory': '1Gi', 'cpu': '1000m'}
            }
        )
    
    @pytest.fixture
    def service_config(self):
        """Create test service configuration"""
        return ServiceConfiguration(
            name="test-service",
            selector={"app": "pouw", "component": "test"},
            ports=[
                {"name": "http", "port": 80, "targetPort": 8080, "protocol": "TCP"}
            ],
            service_type="ClusterIP"
        )
    
    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization"""
        assert orchestrator.namespace == "test-pouw"
        assert orchestrator.kubeconfig is None
        assert "kubectl" in orchestrator.kubectl_cmd
        assert "--namespace" in orchestrator.kubectl_cmd
        assert "test-pouw" in orchestrator.kubectl_cmd
    
    def test_container_configuration_to_dict(self, container_config):
        """Test container configuration conversion"""
        config_dict = container_config.to_dict()
        
        assert config_dict["name"] == "test-container"
        assert config_dict["image"] == "pouw/test:v1.0.0"
        assert config_dict["ports"] == [{"containerPort": 8080}, {"containerPort": 8443}]
        assert len(config_dict["env"]) == 1
        assert config_dict["env"][0] == {"name": "TEST_VAR", "value": "test_value"}
        assert "resources" in config_dict
    
    def test_service_configuration_to_dict(self, service_config):
        """Test service configuration conversion"""
        service_dict = service_config.to_dict()
        
        assert service_dict["apiVersion"] == "v1"
        assert service_dict["kind"] == "Service"
        assert service_dict["metadata"]["name"] == "test-service"
        assert service_dict["spec"]["selector"] == {"app": "pouw", "component": "test"}
        assert service_dict["spec"]["type"] == "ClusterIP"
    
    @pytest.mark.asyncio
    async def test_deployment_manifest_creation(self, orchestrator, container_config):
        """Test deployment manifest creation"""
        manifest = orchestrator._create_deployment_manifest("test-component", container_config)
        
        assert manifest["apiVersion"] == "apps/v1"
        assert manifest["kind"] == "Deployment"
        assert manifest["metadata"]["name"] == "pouw-test-component"
        assert manifest["metadata"]["namespace"] == "test-pouw"
        assert manifest["spec"]["replicas"] == 1
        assert len(manifest["spec"]["template"]["spec"]["containers"]) == 1
    
    @pytest.mark.asyncio
    async def test_kubectl_command_building(self, orchestrator):
        """Test kubectl command building"""
        cmd = orchestrator._build_kubectl_command()
        
        assert cmd[0] == "kubectl"
        assert "--namespace" in cmd
        assert "test-pouw" in cmd
    
    @pytest.mark.asyncio
    @patch('asyncio.create_subprocess_exec')
    async def test_create_namespace(self, mock_subprocess, orchestrator):
        """Test namespace creation"""
        # Mock successful subprocess execution
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"namespace/test-pouw created", b"")
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process
        
        result = await orchestrator.create_namespace()
        
        assert result is True
        mock_subprocess.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('asyncio.create_subprocess_exec')
    async def test_deployment_status_retrieval(self, mock_subprocess, orchestrator):
        """Test deployment status retrieval"""
        # Mock kubectl get deployments response
        deployments_data = {
            "items": [
                {
                    "metadata": {"name": "pouw-test-component"},
                    "status": {
                        "readyReplicas": 1,
                        "replicas": 1,
                        "updatedReplicas": 1,
                        "availableReplicas": 1
                    }
                }
            ]
        }
        
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (json.dumps(deployments_data).encode(), b"")
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process
        
        status = await orchestrator.get_deployment_status()
        
        assert "test-component" in status
        assert status["test-component"]["ready_replicas"] == 1
        assert status["test-component"]["replicas"] == 1


class TestPoUWDeploymentManager:
    """Test PoUW deployment management"""
    
    @pytest.fixture
    def deployment_manager(self):
        """Create deployment manager for testing"""
        return PoUWDeploymentManager(namespace="test-pouw")
    
    def test_deployment_manager_initialization(self, deployment_manager):
        """Test deployment manager initialization"""
        assert deployment_manager.namespace == "test-pouw"
        assert deployment_manager.orchestrator.namespace == "test-pouw"
    
    def test_default_configurations(self, deployment_manager):
        """Test default component configurations"""
        configs = deployment_manager.get_default_configurations()
        
        expected_components = ['blockchain-node', 'ml-trainer', 'vpn-mesh', 'monitoring']
        assert all(component in configs for component in expected_components)
        
        # Test blockchain-node configuration
        blockchain_config = configs['blockchain-node']
        assert blockchain_config.name == 'blockchain-node'
        assert blockchain_config.image == 'pouw/blockchain-node'
        assert 8545 in blockchain_config.ports
        assert blockchain_config.environment['POUW_NETWORK_ID'] == '1337'
    
    def test_default_services(self, deployment_manager):
        """Test default service configurations"""
        services = deployment_manager.get_default_services()
        
        expected_services = ['blockchain-api', 'ml-trainer-api', 'vpn-mesh-service', 'monitoring-dashboard']
        assert all(service in services for service in expected_services)
        
        # Test blockchain-api service
        blockchain_service = services['blockchain-api']
        assert blockchain_service.name == 'pouw-blockchain-api'
        assert blockchain_service.service_type == 'LoadBalancer'
        assert len(blockchain_service.ports) == 2
    
    @pytest.mark.asyncio
    @patch.object(KubernetesOrchestrator, 'deploy_pouw_components')
    @patch.object(KubernetesOrchestrator, 'create_services')
    async def test_full_stack_deployment(self, mock_create_services, mock_deploy_components, deployment_manager):
        """Test full stack deployment"""
        # Mock successful deployments
        mock_deploy_components.return_value = {
            'blockchain-node': DeploymentStatus.RUNNING,
            'ml-trainer': DeploymentStatus.RUNNING,
            'vpn-mesh': DeploymentStatus.RUNNING,
            'monitoring': DeploymentStatus.RUNNING
        }
        mock_create_services.return_value = {
            'blockchain-api': True,
            'ml-trainer-api': True,
            'vpn-mesh-service': True,
            'monitoring-dashboard': True
        }
        
        result = await deployment_manager.deploy_full_stack()
        
        assert result['status'] == 'success'
        assert len(result['components_deployed']) == 4
        assert len(result['services_created']) == 4
        assert all(status == DeploymentStatus.RUNNING for status in result['components_deployed'].values())
        assert all(result['services_created'].values())
        assert result['deployment_time'] > 0


class TestMetricsCollector:
    """Test metrics collection system"""
    
    @pytest.fixture
    def metrics_collector(self):
        """Create metrics collector for testing"""
        return MetricsCollector(collection_interval=1)
    
    def test_metrics_collector_initialization(self, metrics_collector):
        """Test metrics collector initialization"""
        assert metrics_collector.collection_interval == 1
        assert len(metrics_collector.metrics) == 0
        assert len(metrics_collector.metric_handlers) == 0
        assert not metrics_collector.running
    
    def test_metric_handler_registration(self, metrics_collector):
        """Test metric handler registration"""
        def test_handler():
            return 42.0
        
        metrics_collector.register_metric_handler("test_metric", test_handler)
        
        assert "test_metric" in metrics_collector.metric_handlers
        assert metrics_collector.metric_handlers["test_metric"] == test_handler
    
    def test_metric_addition(self, metrics_collector):
        """Test manual metric addition"""
        metric = Metric("test_metric", 25.5, MetricType.GAUGE)
        metrics_collector.add_metric(metric)
        
        assert len(metrics_collector.metrics["test_metric"]) == 1
        assert metrics_collector.metrics["test_metric"][0].value == 25.5
    
    def test_latest_metrics_retrieval(self, metrics_collector):
        """Test latest metrics retrieval"""
        # Add some test metrics
        metric1 = Metric("cpu_usage", 50.0, MetricType.GAUGE)
        metric2 = Metric("memory_usage", 75.0, MetricType.GAUGE)
        
        metrics_collector.add_metric(metric1)
        metrics_collector.add_metric(metric2)
        
        latest = metrics_collector.get_latest_metrics()
        
        assert len(latest) == 2
        assert latest["cpu_usage"].value == 50.0
        assert latest["memory_usage"].value == 75.0
    
    def test_metrics_filtering_by_time_range(self, metrics_collector):
        """Test metrics filtering by time range"""
        # Add old metric
        old_metric = Metric("test_metric", 10.0, MetricType.GAUGE,
                           timestamp=datetime.now() - timedelta(hours=2))
        metrics_collector.add_metric(old_metric)
        
        # Add recent metric
        recent_metric = Metric("test_metric", 20.0, MetricType.GAUGE)
        metrics_collector.add_metric(recent_metric)
        
        # Get metrics from last hour
        recent_metrics = metrics_collector.get_metrics(
            metric_name="test_metric",
            time_range=timedelta(hours=1)
        )
        
        assert len(recent_metrics["test_metric"]) == 1
        assert recent_metrics["test_metric"][0].value == 20.0
    
    @pytest.mark.asyncio
    async def test_collection_lifecycle(self, metrics_collector):
        """Test metrics collection start/stop"""
        assert not metrics_collector.running
        
        await metrics_collector.start_collection()
        assert metrics_collector.running
        assert metrics_collector.collection_task is not None
        
        await asyncio.sleep(0.1)  # Brief pause
        
        await metrics_collector.stop_collection()
        assert not metrics_collector.running


class TestAlertingSystem:
    """Test alerting system"""
    
    @pytest.fixture
    def alerting_system(self):
        """Create alerting system for testing"""
        return AlertingSystem()
    
    @pytest.fixture
    def test_alert(self):
        """Create test alert"""
        return Alert(
            id="test_alert",
            severity=AlertSeverity.WARNING,
            message="Test alert message",
            component="test_component"
        )
    
    def test_alerting_system_initialization(self, alerting_system):
        """Test alerting system initialization"""
        assert len(alerting_system.alert_handlers) == 0
        assert len(alerting_system.active_alerts) == 0
        assert len(alerting_system.alert_history) == 0
        assert len(alerting_system.alert_rules) == 0
        assert not alerting_system.running
    
    def test_alert_handler_registration(self, alerting_system):
        """Test alert handler registration"""
        handler_called = False
        
        def test_handler(alert):
            nonlocal handler_called
            handler_called = True
        
        alerting_system.add_alert_handler(test_handler)
        assert len(alerting_system.alert_handlers) == 1
    
    def test_alert_rule_registration(self, alerting_system):
        """Test alert rule registration"""
        def test_rule(metrics):
            return None
        
        alerting_system.add_alert_rule(test_rule)
        assert len(alerting_system.alert_rules) == 1
    
    @pytest.mark.asyncio
    async def test_alert_handling(self, alerting_system, test_alert):
        """Test alert handling"""
        handler_called = False
        received_alert = None
        
        def test_handler(alert):
            nonlocal handler_called, received_alert
            handler_called = True
            received_alert = alert
        
        alerting_system.add_alert_handler(test_handler)
        await alerting_system._handle_alert(test_alert)
        
        assert handler_called
        assert received_alert == test_alert
        assert test_alert.id in alerting_system.active_alerts
        assert test_alert in alerting_system.alert_history
    
    def test_alert_resolution(self, alerting_system, test_alert):
        """Test alert resolution"""
        alerting_system.active_alerts[test_alert.id] = test_alert
        
        alerting_system.resolve_alert(test_alert.id)
        
        assert alerting_system.active_alerts[test_alert.id].resolved
        assert alerting_system.active_alerts[test_alert.id].resolved_at is not None
    
    def test_active_alerts_retrieval(self, alerting_system):
        """Test active alerts retrieval"""
        # Create test alerts with different severities
        warning_alert = Alert("warning_1", AlertSeverity.WARNING, "Warning message", "component1")
        error_alert = Alert("error_1", AlertSeverity.ERROR, "Error message", "component2")
        
        alerting_system.active_alerts[warning_alert.id] = warning_alert
        alerting_system.active_alerts[error_alert.id] = error_alert
        
        # Get all active alerts
        all_alerts = alerting_system.get_active_alerts()
        assert len(all_alerts) == 2
        
        # Get only error alerts
        error_alerts = alerting_system.get_active_alerts(severity=AlertSeverity.ERROR)
        assert len(error_alerts) == 1
        assert error_alerts[0].severity == AlertSeverity.ERROR


class TestLoadBalancer:
    """Test load balancer functionality"""
    
    @pytest.fixture
    def load_balancer(self):
        """Create load balancer for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield LoadBalancer(config_dir=temp_dir)
    
    @pytest.fixture
    def lb_config(self):
        """Create load balancer configuration"""
        return LoadBalancerConfig(
            name="test-lb",
            strategy=LoadBalancingStrategy.ROUND_ROBIN,
            backend_servers=[
                {"host": "10.0.1.10", "port": 8080, "weight": 1},
                {"host": "10.0.1.11", "port": 8080, "weight": 2}
            ],
            health_check={"path": "/health", "interval": 30}
        )
    
    def test_load_balancer_initialization(self, load_balancer):
        """Test load balancer initialization"""
        assert load_balancer.config_dir.exists()
        assert len(load_balancer.balancer_configs) == 0
    
    def test_nginx_config_generation(self, lb_config):
        """Test nginx configuration generation"""
        nginx_config = lb_config.to_nginx_config()
        
        assert "upstream pouw_test-lb" in nginx_config
        assert "server 10.0.1.10:8080" in nginx_config
        assert "server 10.0.1.11:8080" in nginx_config
        assert "weight=1" in nginx_config
        assert "weight=2" in nginx_config
        assert "location /health" in nginx_config
        assert "proxy_pass http://pouw_test-lb" in nginx_config
    
    def test_weighted_round_robin_config(self):
        """Test weighted round robin configuration"""
        config = LoadBalancerConfig(
            name="weighted-lb",
            strategy=LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN,
            backend_servers=[
                {"host": "10.0.1.10", "port": 8080, "weight": 3}
            ]
        )
        
        nginx_config = config.to_nginx_config()
        assert "weight=3" in nginx_config
    
    def test_least_connections_config(self):
        """Test least connections configuration"""
        config = LoadBalancerConfig(
            name="lc-lb",
            strategy=LoadBalancingStrategy.LEAST_CONNECTIONS,
            backend_servers=[
                {"host": "10.0.1.10", "port": 8080}
            ]
        )
        
        nginx_config = config.to_nginx_config()
        assert "least_conn;" in nginx_config
    
    def test_ssl_configuration(self):
        """Test SSL configuration"""
        config = LoadBalancerConfig(
            name="ssl-lb",
            strategy=LoadBalancingStrategy.ROUND_ROBIN,
            backend_servers=[{"host": "10.0.1.10", "port": 8080}],
            ssl_config={
                "cert_path": "/etc/ssl/cert.pem",
                "key_path": "/etc/ssl/key.pem"
            }
        )
        
        nginx_config = config.to_nginx_config()
        assert "listen 443 ssl;" in nginx_config
        assert "ssl_certificate /etc/ssl/cert.pem;" in nginx_config
        assert "ssl_certificate_key /etc/ssl/key.pem;" in nginx_config


class TestAutoScaler:
    """Test auto-scaling functionality"""
    
    @pytest.fixture
    def auto_scaler(self):
        """Create auto-scaler for testing"""
        mock_orchestrator = MagicMock()
        return AutoScaler(mock_orchestrator)
    
    @pytest.fixture
    def scaling_rule(self):
        """Create auto-scaling rule"""
        return AutoScalingRule(
            metric_name="cpu_usage",
            threshold_up=80.0,
            threshold_down=20.0,
            min_replicas=2,
            max_replicas=10
        )
    
    def test_auto_scaler_initialization(self, auto_scaler):
        """Test auto-scaler initialization"""
        assert len(auto_scaler.scaling_rules) == 0
        assert len(auto_scaler.scaling_history) == 0
        assert len(auto_scaler.last_scaling_action) == 0
        assert not auto_scaler.running
    
    def test_scaling_rule_addition(self, auto_scaler, scaling_rule):
        """Test scaling rule addition"""
        auto_scaler.add_scaling_rule("test-component", scaling_rule)
        
        assert "test-component" in auto_scaler.scaling_rules
        assert auto_scaler.scaling_rules["test-component"] == scaling_rule
    
    @pytest.mark.asyncio
    async def test_scaling_evaluation_scale_up(self, auto_scaler, scaling_rule):
        """Test scaling evaluation for scale up"""
        auto_scaler.add_scaling_rule("test-component", scaling_rule)
        
        # Mock high CPU usage
        auto_scaler._get_metric_value = AsyncMock(return_value=85.0)
        
        deployment_status = {
            "test-component": {"replicas": 3, "ready_replicas": 3}
        }
        
        decision = await auto_scaler._evaluate_component_scaling(
            "test-component", scaling_rule, deployment_status
        )
        
        assert decision is not None
        assert decision.direction == ScalingDirection.UP
        assert decision.target_replicas == 4
        assert decision.current_replicas == 3
        assert decision.metric_value == 85.0
    
    @pytest.mark.asyncio
    async def test_scaling_evaluation_scale_down(self, auto_scaler, scaling_rule):
        """Test scaling evaluation for scale down"""
        auto_scaler.add_scaling_rule("test-component", scaling_rule)
        
        # Mock low CPU usage
        auto_scaler._get_metric_value = AsyncMock(return_value=15.0)
        
        deployment_status = {
            "test-component": {"replicas": 5, "ready_replicas": 5}
        }
        
        decision = await auto_scaler._evaluate_component_scaling(
            "test-component", scaling_rule, deployment_status
        )
        
        assert decision is not None
        assert decision.direction == ScalingDirection.DOWN
        assert decision.target_replicas == 4
        assert decision.current_replicas == 5
        assert decision.metric_value == 15.0
    
    @pytest.mark.asyncio
    async def test_scaling_evaluation_maintain(self, auto_scaler, scaling_rule):
        """Test scaling evaluation for maintain"""
        auto_scaler.add_scaling_rule("test-component", scaling_rule)
        
        # Mock moderate CPU usage
        auto_scaler._get_metric_value = AsyncMock(return_value=50.0)
        
        deployment_status = {
            "test-component": {"replicas": 3, "ready_replicas": 3}
        }
        
        decision = await auto_scaler._evaluate_component_scaling(
            "test-component", scaling_rule, deployment_status
        )
        
        assert decision is not None
        assert decision.direction == ScalingDirection.MAINTAIN
        assert decision.target_replicas == 3
        assert decision.current_replicas == 3
    
    def test_scaling_history_retrieval(self, auto_scaler):
        """Test scaling history retrieval"""
        from pouw.deployment.infrastructure import ScalingDecision
        
        # Add test scaling decisions
        decision1 = ScalingDecision("component1", 2, 3, ScalingDirection.UP, "High CPU", 85.0)
        decision2 = ScalingDecision("component2", 4, 3, ScalingDirection.DOWN, "Low CPU", 15.0)
        
        auto_scaler.scaling_history = [decision1, decision2]
        
        # Get all history
        all_history = auto_scaler.get_scaling_history()
        assert len(all_history) == 2
        
        # Get history for specific component
        component1_history = auto_scaler.get_scaling_history(component="component1")
        assert len(component1_history) == 1
        assert component1_history[0].component == "component1"


class TestInfrastructureAsCode:
    """Test Infrastructure as Code functionality"""
    
    @pytest.fixture
    def iac(self):
        """Create IaC manager for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield InfrastructureAsCode(project_dir=temp_dir)
    
    @pytest.fixture
    def deployment_config(self):
        """Create deployment configuration"""
        return {
            "config": {
                "POUW_NETWORK_ID": "1337",
                "POUW_LOG_LEVEL": "info"
            },
            "secrets": {
                "POUW_SECRET_KEY": "REDACTED"
            }
        }
    
    def test_iac_initialization(self, iac):
        """Test IaC initialization"""
        assert iac.project_dir.exists()
        assert iac.config_hash is None
    
    def test_terraform_config_generation(self, iac, deployment_config):
        """Test Terraform configuration generation"""
        terraform_config = iac.generate_terraform_config(deployment_config)
        
        assert "terraform {" in terraform_config
        assert "kubernetes_namespace" in terraform_config
        assert "kubernetes_config_map" in terraform_config
        assert "kubernetes_secret" in terraform_config
        assert "helm_release" in terraform_config
        assert "POUW_NETWORK_ID" in terraform_config
        assert "POUW_LOG_LEVEL" in terraform_config
        assert "REDACTED" in terraform_config
    
    def test_config_map_generation(self, iac, deployment_config):
        """Test ConfigMap generation in Terraform config"""
        terraform_config = iac.generate_terraform_config(deployment_config)
        
        assert 'resource "kubernetes_config_map" "pouw_config"' in terraform_config
        assert '"POUW_NETWORK_ID" = "1337"' in terraform_config
        assert '"POUW_LOG_LEVEL" = "info"' in terraform_config
    
    def test_secret_generation(self, iac, deployment_config):
        """Test Secret generation in Terraform config"""
        terraform_config = iac.generate_terraform_config(deployment_config)
        
        assert 'resource "kubernetes_secret" "pouw_secrets"' in terraform_config
        assert '"POUW_SECRET_KEY" = base64encode("REDACTED")' in terraform_config
    
    def test_monitoring_stack_generation(self, iac, deployment_config):
        """Test monitoring stack generation"""
        terraform_config = iac.generate_terraform_config(deployment_config)
        
        assert 'resource "helm_release" "prometheus"' in terraform_config
        assert "kube-prometheus-stack" in terraform_config
        assert "grafana" in terraform_config


class TestProductionMonitor:
    """Test production monitoring system"""
    
    @pytest.fixture
    def production_monitor(self):
        """Create production monitor for testing"""
        return ProductionMonitor(namespace="test-pouw")
    
    def test_production_monitor_initialization(self, production_monitor):
        """Test production monitor initialization"""
        assert production_monitor.namespace == "test-pouw"
        assert production_monitor.metrics_collector is not None
        assert production_monitor.alerting_system is not None
        assert production_monitor.logging_manager is not None
        assert production_monitor.health_checker is not None
        assert production_monitor.performance_analyzer is not None
    
    def test_default_alert_rules_setup(self, production_monitor):
        """Test default alert rules setup"""
        # Should have default rules for CPU, memory, and disk
        assert len(production_monitor.alerting_system.alert_rules) >= 3
    
    def test_default_health_checks_setup(self, production_monitor):
        """Test default health checks setup"""
        # Should have at least system health check
        assert len(production_monitor.health_checker.health_checks) >= 1
        assert "system" in production_monitor.health_checker.health_checks
    
    def test_monitoring_dashboard_data(self, production_monitor):
        """Test monitoring dashboard data generation"""
        # Add some test data
        test_metric = Metric("test_cpu", 50.0, MetricType.GAUGE)
        production_monitor.metrics_collector.add_metric(test_metric)
        
        dashboard_data = production_monitor.get_monitoring_dashboard()
        
        assert "timestamp" in dashboard_data
        assert "namespace" in dashboard_data
        assert "metrics" in dashboard_data
        assert "alerts" in dashboard_data
        assert "health" in dashboard_data
        assert "performance" in dashboard_data
        assert "logs" in dashboard_data
        
        assert dashboard_data["namespace"] == "test-pouw"
        assert "test_cpu" in dashboard_data["metrics"]


class TestPerformanceOptimization:
    """Test performance optimization and analysis"""
    
    @pytest.fixture
    def performance_analyzer(self):
        """Create performance analyzer for testing"""
        return PerformanceAnalyzer()
    
    def test_performance_analyzer_initialization(self, performance_analyzer):
        """Test performance analyzer initialization"""
        assert len(performance_analyzer.performance_data) == 0
        assert len(performance_analyzer.analysis_rules) == 0
    
    def test_performance_metric_recording(self, performance_analyzer):
        """Test performance metric recording"""
        performance_analyzer.record_performance_metric("response_time", 250.5, {"endpoint": "/api/health"})
        
        assert len(performance_analyzer.performance_data["response_time"]) == 1
        data_point = performance_analyzer.performance_data["response_time"][0]
        assert data_point["value"] == 250.5
        assert data_point["metadata"]["endpoint"] == "/api/health"
    
    def test_analysis_rule_addition(self, performance_analyzer):
        """Test analysis rule addition"""
        def test_rule(data):
            return {"recommendations": ["Test recommendation"]}
        
        performance_analyzer.add_analysis_rule(test_rule)
        assert len(performance_analyzer.analysis_rules) == 1
    
    def test_performance_analysis(self, performance_analyzer):
        """Test performance analysis execution"""
        # Add test data
        performance_analyzer.record_performance_metric("cpu_usage", 75.0)
        performance_analyzer.record_performance_metric("memory_usage", 60.0)
        
        # Add test rule
        def test_rule(data):
            recommendations = []
            if "cpu_usage" in data and data["cpu_usage"]["avg"] > 70:
                recommendations.append("Consider scaling up due to high CPU usage")
            return {"recommendations": recommendations}
        
        performance_analyzer.add_analysis_rule(test_rule)
        
        analysis = performance_analyzer.analyze_performance()
        
        assert "analysis_time" in analysis
        assert "metrics_analyzed" in analysis
        assert "aggregated_data" in analysis
        assert "recommendations" in analysis
        assert analysis["metrics_analyzed"] == 2
        assert len(analysis["recommendations"]) > 0


# Integration tests for complete workflow
class TestDeploymentWorkflow:
    """Test complete deployment workflow"""
    
    @pytest.mark.asyncio
    async def test_complete_deployment_workflow(self):
        """Test complete deployment workflow integration"""
        # This test would require actual Kubernetes cluster
        # For now, we'll test the configuration and workflow setup
        
        # Initialize components
        deployment_manager = PoUWDeploymentManager(namespace="test-workflow")
        config_manager = ConfigurationManager()
        
        # Get default configuration
        config = config_manager.get_default_deployment_config()
        
        assert config is not None
        assert "config" in config
        assert "secrets" in config
        assert "load_balancer" in config
        assert "scaling" in config
        
        # Verify configuration structure
        assert config["config"]["POUW_NETWORK_ID"] == "1337"
        assert config["load_balancer"]["name"] == "pouw-api"
        assert config["scaling"]["min_replicas"] == 2


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmarks for deployment infrastructure"""
    
    @pytest.mark.asyncio
    async def test_metrics_collection_performance(self):
        """Test metrics collection performance"""
        collector = MetricsCollector(collection_interval=1)
        
        # Time metrics addition
        import time
        start_time = time.time()
        
        for i in range(1000):
            metric = Metric(f"test_metric_{i % 10}", float(i), MetricType.GAUGE)
            collector.add_metric(metric)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should handle 1000 metrics quickly
        assert duration < 1.0, f"Metrics addition took {duration:.3f}s, expected < 1.0s"
        
        # Verify all metrics were added
        assert len(collector.metrics) == 10  # 10 unique metric names
        
        # Test retrieval performance
        start_time = time.time()
        latest_metrics = collector.get_latest_metrics()
        end_time = time.time()
        retrieval_duration = end_time - start_time
        
        assert retrieval_duration < 0.1, f"Metrics retrieval took {retrieval_duration:.3f}s"
        assert len(latest_metrics) == 10
    
    def test_load_balancer_config_generation_performance(self):
        """Test load balancer configuration generation performance"""
        # Create large backend server list
        backend_servers = []
        for i in range(100):
            backend_servers.append({
                "host": f"10.0.{i // 254}.{i % 254}",
                "port": 8080,
                "weight": 1
            })
        
        config = LoadBalancerConfig(
            name="large-lb",
            strategy=LoadBalancingStrategy.ROUND_ROBIN,
            backend_servers=backend_servers
        )
        
        # Time configuration generation
        import time
        start_time = time.time()
        nginx_config = config.to_nginx_config()
        end_time = time.time()
        duration = end_time - start_time
        
        # Should generate config quickly even with many backends
        assert duration < 0.5, f"Config generation took {duration:.3f}s, expected < 0.5s"
        assert len(nginx_config) > 1000  # Should be substantial config
        assert nginx_config.count("server 10.0.") == 100  # All backends included


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([__file__, "-v", "--tb=short"])
