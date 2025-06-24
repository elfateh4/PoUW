"""
PoUW (Proof of Useful Work) Implementation

A blockchain that replaces wasteful mining with useful machine learning work.
This system provides enterprise-grade deployment infrastructure with 
Kubernetes orchestration, production monitoring, and auto-scaling capabilities.
"""

__version__ = "1.0.0"
__author__ = "PoUW Implementation Team"

# Import deployment infrastructure components
try:
    from .deployment import (
        KubernetesOrchestrator,
        PoUWDeploymentManager,
        ProductionMonitor,
        LoadBalancer,
        AutoScaler,
        InfrastructureAsCode,
        DeploymentAutomation,
        ConfigurationManager,
        ResourceManager
    )
    
    __all__ = [
        'KubernetesOrchestrator',
        'PoUWDeploymentManager', 
        'ProductionMonitor',
        'LoadBalancer',
        'AutoScaler',
        'InfrastructureAsCode',
        'DeploymentAutomation',
        'ConfigurationManager',
        'ResourceManager'
    ]
    
except ImportError:
    # Graceful fallback if deployment dependencies are not available
    __all__ = []
