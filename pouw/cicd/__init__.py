"""
CI/CD Pipeline Integration Package

This package provides comprehensive CI/CD pipeline capabilities for the PoUW system,
including automated builds, testing, deployment, and release management.
"""

from .github_actions import (
    GitHubActionsManager,
    WorkflowConfiguration,
    JobConfiguration,
    StepConfiguration,
    TriggerConfiguration
)

from .jenkins import (
    JenkinsPipelineManager,
    JenkinsfileGenerator,
    PipelineStage,
    PipelineConfiguration
)

from .docker_automation import (
    DockerBuildManager,
    DockerImageBuilder,
    ContainerRegistry,
    ImageConfiguration,
    BuildConfiguration
)

from .testing_automation import (
    TestAutomationManager,
    TestSuite,
    CoverageAnalyzer,
    TestConfiguration,
    TestResult,
    TestSuiteResult,
    TestType,
    PoUWTestSuites
)

from .deployment_automation import (
    DeploymentPipelineManager,
    ReleaseManager,
    DeploymentConfiguration,
    DeploymentResult,
    ReleaseInfo,
    DeploymentStrategy,
    Environment,
    PlatformType,
    PoUWDeploymentConfigurations
)

from .quality_assurance import (
    CodeQualityManager,
    SecurityScanner,
    QualityReport,
    QualityMetrics,
    QualityIssue,
    QualityConfiguration,
    QualityGateRule,
    PoUWQualityConfiguration
)

__all__ = [
    # GitHub Actions
    'GitHubActionsManager',
    'WorkflowConfiguration',
    'JobConfiguration',
    'StepConfiguration',
    'TriggerConfiguration',
    
    # Jenkins
    'JenkinsPipelineManager',
    'JenkinsfileGenerator',
    'PipelineStage',
    'PipelineConfiguration',
    
    # Docker automation
    'DockerBuildManager',
    'DockerImageBuilder',
    'ContainerRegistry',
    'ImageConfiguration',
    'BuildConfiguration',
    
    # Testing automation
    'TestAutomationManager',
    'TestSuite',
    'CoverageAnalyzer',
    'TestConfiguration',
    'TestResult',
    'TestSuiteResult',
    'TestType',
    'PoUWTestSuites',
    
    # Deployment automation
    'DeploymentPipelineManager',
    'ReleaseManager',
    'DeploymentConfiguration',
    'DeploymentResult',
    'ReleaseInfo',
    'DeploymentStrategy',
    'Environment',
    'PlatformType',
    'PoUWDeploymentConfigurations',
    
    # Quality assurance
    'CodeQualityManager',
    'SecurityScanner',
    'QualityReport',
    'QualityMetrics',
    'QualityIssue',
    'QualityConfiguration',
    'QualityGateRule',
    'PoUWQualityConfiguration'
]
