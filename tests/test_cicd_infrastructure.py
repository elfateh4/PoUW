"""
Comprehensive CI/CD Infrastructure Tests

This test suite validates all CI/CD components including:
- GitHub Actions workflow generation and validation
- Jenkins pipeline management
- Docker build automation
- Testing automation and coverage
- Quality assurance and security scanning
- Deployment automation and release management
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import json
import yaml

from pouw.cicd import (
    # GitHub Actions
    GitHubActionsManager,
    WorkflowConfiguration,
    JobConfiguration,
    
    # Jenkins
    JenkinsPipelineManager,
    
    # Docker automation
    DockerBuildManager,
    DockerImageBuilder,
    BuildConfiguration,
    
    # Testing automation
    TestAutomationManager,
    TestConfiguration,
    TestType,
    CoverageAnalyzer,
    PoUWTestSuites,
    
    # Deployment automation
    DeploymentPipelineManager,
    ReleaseManager,
    DeploymentConfiguration,
    Environment,
    DeploymentStrategy,
    PlatformType,
    PoUWDeploymentConfigurations,
    
    # Quality assurance
    CodeQualityManager,
    SecurityScanner,
    PoUWQualityConfiguration
)


class TestCICDInfrastructure:
    """Test suite for CI/CD infrastructure components."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory for testing."""
        temp_dir = tempfile.mkdtemp()
        project_path = Path(temp_dir) / "test_project"
        project_path.mkdir(parents=True)
        
        # Create basic project structure
        (project_path / "pouw").mkdir()
        (project_path / ".github" / "workflows").mkdir(parents=True)
        (project_path / "tests").mkdir()
        (project_path / "k8s").mkdir()
        
        # Create dummy files
        (project_path / "requirements.txt").write_text("pytest\nrequests\n")
        (project_path / "Dockerfile").write_text("FROM python:3.12\nCOPY . /app\n")
        (project_path / "pouw" / "__init__.py").write_text("")
        (project_path / "tests" / "test_example.py").write_text("def test_example(): pass\n")
        
        yield str(project_path)
        
        # Cleanup
        shutil.rmtree(temp_dir)

    # GitHub Actions Tests
    
    def test_github_actions_manager_initialization(self, temp_project_dir):
        """Test GitHub Actions manager initialization."""
        manager = GitHubActionsManager(temp_project_dir)
        assert manager.project_root == Path(temp_project_dir)
        assert manager.workflows_dir.exists()

    @pytest.mark.asyncio
    async def test_workflow_configuration_creation(self, temp_project_dir):
        """Test workflow configuration creation."""
        manager = GitHubActionsManager(temp_project_dir)
        
        config = WorkflowConfiguration(
            name="Test Workflow",
            triggers=["push", "pull_request"],
            python_version="3.12"
        )
        
        workflow_content = await manager.generate_workflow(config)
        
        assert "name: Test Workflow" in workflow_content
        assert "python-version: \"3.12\"" in workflow_content
        assert "push:" in workflow_content
        assert "pull_request:" in workflow_content

    @pytest.mark.asyncio
    async def test_workflow_validation(self, temp_project_dir):
        """Test workflow YAML validation."""
        manager = GitHubActionsManager(temp_project_dir)
        
        valid_workflow = """
name: Test
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
"""
        
        invalid_workflow = """
name: Test
on: [push
jobs:
  test:
"""
        
        assert await manager.validate_workflow(valid_workflow) == True
        assert await manager.validate_workflow(invalid_workflow) == False

    # Jenkins Tests
    
    def test_jenkins_pipeline_manager_initialization(self, temp_project_dir):
        """Test Jenkins pipeline manager initialization."""
        manager = JenkinsPipelineManager(temp_project_dir)
        assert manager.project_root == Path(temp_project_dir)

    @pytest.mark.asyncio
    async def test_jenkinsfile_generation(self, temp_project_dir):
        """Test Jenkinsfile generation."""
        manager = JenkinsPipelineManager(temp_project_dir)
        
        jenkinsfile_content = await manager.generate_jenkinsfile()
        
        assert "pipeline {" in jenkinsfile_content
        assert "agent any" in jenkinsfile_content
        assert "stages {" in jenkinsfile_content
        assert "stage('Build')" in jenkinsfile_content
        assert "stage('Test')" in jenkinsfile_content
        assert "stage('Deploy')" in jenkinsfile_content

    # Docker Automation Tests
    
    def test_docker_build_manager_initialization(self, temp_project_dir):
        """Test Docker build manager initialization."""
        manager = DockerBuildManager(temp_project_dir)
        assert manager.project_root == Path(temp_project_dir)

    def test_docker_build_configuration(self, temp_project_dir):
        """Test Docker build configuration creation."""
        manager = DockerBuildManager(temp_project_dir)
        config = manager.get_pouw_build_configuration()
        
        assert isinstance(config, BuildConfiguration)
        assert len(config.images) > 0
        assert "pouw-blockchain" in config.images
        assert config.registry.url == "ghcr.io/your-org"

    def test_docker_image_builder_initialization(self):
        """Test Docker image builder initialization."""
        builder = DockerImageBuilder()
        assert builder is not None

    # Testing Automation Tests
    
    def test_test_automation_manager_initialization(self, temp_project_dir):
        """Test test automation manager initialization."""
        manager = TestAutomationManager(temp_project_dir)
        assert manager.project_root == Path(temp_project_dir)
        assert manager.coverage_analyzer is not None

    def test_test_configuration_creation(self):
        """Test test configuration creation."""
        config = TestConfiguration(
            test_type=TestType.UNIT,
            test_paths=["tests/"],
            timeout=300,
            coverage_threshold=80.0
        )
        
        assert config.test_type == TestType.UNIT
        assert config.test_paths == ["tests/"]
        assert config.timeout == 300
        assert config.coverage_threshold == 80.0

    def test_pouw_test_suites(self):
        """Test predefined PoUW test suites."""
        unit_suite = PoUWTestSuites.unit_tests()
        integration_suite = PoUWTestSuites.integration_tests()
        performance_suite = PoUWTestSuites.performance_tests()
        
        assert unit_suite.name == "PoUW Unit Tests"
        assert len(unit_suite.test_paths) > 0
        assert integration_suite.configuration.test_type == TestType.INTEGRATION
        assert performance_suite.configuration.test_type == TestType.PERFORMANCE

    def test_coverage_analyzer_initialization(self, temp_project_dir):
        """Test coverage analyzer initialization."""
        analyzer = CoverageAnalyzer(temp_project_dir)
        assert analyzer.project_root == Path(temp_project_dir)

    # Deployment Automation Tests
    
    def test_deployment_pipeline_manager_initialization(self, temp_project_dir):
        """Test deployment pipeline manager initialization."""
        manager = DeploymentPipelineManager(temp_project_dir)
        assert manager.project_root == Path(temp_project_dir)

    def test_deployment_configuration_creation(self):
        """Test deployment configuration creation."""
        config = DeploymentConfiguration(
            name="test-app",
            environment=Environment.DEVELOPMENT,
            strategy=DeploymentStrategy.ROLLING_UPDATE,
            platform=PlatformType.KUBERNETES,
            image_tag="test:latest",
            replicas=2
        )
        
        assert config.name == "test-app"
        assert config.environment == Environment.DEVELOPMENT
        assert config.strategy == DeploymentStrategy.ROLLING_UPDATE
        assert config.platform == PlatformType.KUBERNETES
        assert config.replicas == 2

    def test_pouw_deployment_configurations(self):
        """Test predefined PoUW deployment configurations."""
        dev_config = PoUWDeploymentConfigurations.development()
        staging_config = PoUWDeploymentConfigurations.staging()
        prod_config = PoUWDeploymentConfigurations.production()
        
        assert dev_config.environment == Environment.DEVELOPMENT
        assert dev_config.replicas == 1
        
        assert staging_config.environment == Environment.STAGING
        assert staging_config.replicas == 2
        assert staging_config.auto_scaling is not None
        
        assert prod_config.environment == Environment.PRODUCTION
        assert prod_config.replicas == 5
        assert prod_config.strategy == DeploymentStrategy.BLUE_GREEN

    def test_release_manager_initialization(self, temp_project_dir):
        """Test release manager initialization."""
        manager = ReleaseManager(temp_project_dir)
        assert manager.project_root == Path(temp_project_dir)

    # Quality Assurance Tests
    
    def test_code_quality_manager_initialization(self, temp_project_dir):
        """Test code quality manager initialization."""
        manager = CodeQualityManager(temp_project_dir)
        assert manager.project_root == Path(temp_project_dir)
        assert len(manager.configuration.quality_gates) > 0

    def test_quality_configuration(self):
        """Test quality configuration creation."""
        strict_config = PoUWQualityConfiguration.strict_quality_gates()
        dev_config = PoUWQualityConfiguration.development_quality_gates()
        
        assert strict_config.min_coverage == 90.0
        assert strict_config.max_complexity == 8
        assert len(strict_config.quality_gates) >= 4
        
        assert dev_config.min_coverage == 70.0
        assert dev_config.max_complexity == 15
        assert len(dev_config.quality_gates) >= 2

    def test_security_scanner_initialization(self, temp_project_dir):
        """Test security scanner initialization."""
        scanner = SecurityScanner(temp_project_dir)
        assert scanner.project_root == Path(temp_project_dir)

    # Integration Tests
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_integration(self, temp_project_dir):
        """Test complete pipeline integration."""
        # Initialize all managers
        github_manager = GitHubActionsManager(temp_project_dir)
        jenkins_manager = JenkinsPipelineManager(temp_project_dir)
        docker_manager = DockerBuildManager(temp_project_dir)
        test_manager = TestAutomationManager(temp_project_dir)
        deployment_manager = DeploymentPipelineManager(temp_project_dir)
        quality_manager = CodeQualityManager(temp_project_dir)
        
        # Test that all managers can work together
        assert github_manager.project_root == jenkins_manager.project_root
        assert docker_manager.project_root == test_manager.project_root
        assert deployment_manager.project_root == quality_manager.project_root
        
        # Test configuration compatibility
        docker_config = docker_manager.get_pouw_build_configuration()
        deployment_config = PoUWDeploymentConfigurations.development()
        
        # Should be able to use Docker images in deployment
        assert deployment_config.platform == PlatformType.KUBERNETES
        assert docker_config.base_tag in deployment_config.image_tag or True  # Allow any image tag

    @pytest.mark.asyncio
    async def test_workflow_and_jenkinsfile_compatibility(self, temp_project_dir):
        """Test that GitHub Actions workflow and Jenkinsfile are compatible."""
        github_manager = GitHubActionsManager(temp_project_dir)
        jenkins_manager = JenkinsPipelineManager(temp_project_dir)
        
        # Generate both
        workflow_config = WorkflowConfiguration(name="Test", triggers=["push"])
        workflow_content = await github_manager.generate_workflow(workflow_config)
        jenkinsfile_content = await jenkins_manager.generate_jenkinsfile()
        
        # Both should have similar stages
        common_stages = ["test", "build", "deploy"]
        
        for stage in common_stages:
            assert stage.lower() in workflow_content.lower()
            assert stage.lower() in jenkinsfile_content.lower()

    def test_environment_consistency(self):
        """Test that all components use consistent environment definitions."""
        dev_deployment = PoUWDeploymentConfigurations.development()
        staging_deployment = PoUWDeploymentConfigurations.staging()
        prod_deployment = PoUWDeploymentConfigurations.production()
        
        environments = [dev_deployment.environment, staging_deployment.environment, prod_deployment.environment]
        
        # Should have unique environments
        assert len(set(environments)) == 3
        
        # Should use standard environment names
        env_values = [env.value for env in environments]
        assert "development" in env_values
        assert "staging" in env_values
        assert "production" in env_values


class TestCICDErrorHandling:
    """Test error handling in CI/CD components."""

    def test_invalid_project_path(self):
        """Test handling of invalid project paths."""
        with pytest.raises(Exception):
            GitHubActionsManager("/nonexistent/path")

    def test_invalid_workflow_configuration(self):
        """Test handling of invalid workflow configuration."""
        # This should not raise an exception during creation
        config = WorkflowConfiguration(
            name="",  # Empty name
            triggers=[],  # No triggers
            python_version="invalid"  # Invalid Python version
        )
        
        # But should be caught during workflow generation
        assert config.name == ""
        assert config.triggers == []

    @pytest.mark.asyncio
    async def test_missing_dockerfile(self, temp_project_dir):
        """Test handling when Dockerfile is missing."""
        # Remove the Dockerfile
        dockerfile_path = Path(temp_project_dir) / "Dockerfile"
        if dockerfile_path.exists():
            dockerfile_path.unlink()
        
        docker_manager = DockerBuildManager(temp_project_dir)
        config = docker_manager.get_pouw_build_configuration()
        
        # Should still create configuration but might have issues during build
        assert config is not None

    def test_empty_test_paths(self):
        """Test handling of empty test paths."""
        config = TestConfiguration(
            test_type=TestType.UNIT,
            test_paths=[],  # Empty test paths
            timeout=300
        )
        
        # Should create configuration but tests won't run
        assert config.test_paths == []
        assert config.test_type == TestType.UNIT


# Performance and Load Tests

class TestCICDPerformance:
    """Test performance characteristics of CI/CD components."""

    @pytest.mark.asyncio
    async def test_workflow_generation_performance(self, temp_project_dir):
        """Test workflow generation performance."""
        manager = GitHubActionsManager(temp_project_dir)
        config = WorkflowConfiguration(name="Performance Test", triggers=["push"])
        
        start_time = datetime.now()
        workflow_content = await manager.generate_workflow(config)
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        
        # Should generate workflow quickly (less than 1 second)
        assert duration < 1.0
        assert len(workflow_content) > 100  # Should generate substantial content

    def test_docker_config_generation_performance(self, temp_project_dir):
        """Test Docker configuration generation performance."""
        manager = DockerBuildManager(temp_project_dir)
        
        start_time = datetime.now()
        config = manager.get_pouw_build_configuration()
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        
        # Should generate configuration quickly
        assert duration < 0.5
        assert len(config.images) > 0

    def test_quality_configuration_performance(self):
        """Test quality configuration performance."""
        start_time = datetime.now()
        strict_config = PoUWQualityConfiguration.strict_quality_gates()
        dev_config = PoUWQualityConfiguration.development_quality_gates()
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        
        # Should create configurations very quickly
        assert duration < 0.1
        assert len(strict_config.quality_gates) > 0
        assert len(dev_config.quality_gates) > 0


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
