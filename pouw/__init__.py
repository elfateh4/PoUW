"""
PoUW (Proof of Useful Work) - Production-ready blockchain implementation.

This package provides a comprehensive implementation of the Proof of Useful Work
consensus mechanism for distributed machine learning tasks.

Features:
- Blockchain core functionality with standardized transaction format
- Distributed ML training and verification
- VPN mesh networking and P2P communication
- Advanced security and byzantine fault tolerance
- Economic incentive system with staking and rewards
- Production monitoring and GPU acceleration
- Enterprise deployment with Kubernetes support
- CI/CD pipeline integration
- Comprehensive data management with Reed-Solomon encoding
- Cryptographic primitives including BLS threshold signatures
"""

# Core blockchain functionality
from .blockchain import (
    Transaction,
    PayForTaskTransaction,
    BuyTicketsTransaction,
    PoUWBlockHeader,
    Block,
    MLTask,
    Blockchain,
    StandardizedTransactionFormat,
    PoUWOpReturnData,
    PoUWOpCode,
    create_standardized_pouw_transaction,
    parse_standardized_pouw_transaction,
)

# Network and communication
from .network import (
    NetworkMessage,
    MessageHandler,
    BlockchainMessageHandler,
    MLMessageHandler,
    P2PNode,
    MessageHistory,
    NodeStatus,
    LeaderElectionState,
    NodeHealthMetrics,
    CrashRecoveryEvent,
    LeaderElectionVote,
    CompressedMessageBatch,
    CrashRecoveryManager,
    WorkerReplacementManager,
    LeaderElectionManager,
    MessageHistoryCompressor,
    VPNMeshTopologyManager,
    NetworkOperationsManager,
)

# Machine learning functionality
from .ml import (
    MiniBatch,
    GradientUpdate,
    IterationMessage,
    MLModel,
    SimpleMLP,
    DistributedTrainer,
)

# Mining and consensus
from .mining import (
    MiningProof,
    PoUWMiner,
    PoUWVerifier,
)

# Security framework
from .security import (
    AttackType,
    SecurityAlert,
    GradientPoisoningDetector,
    ByzantineFaultTolerance,
    AttackMitigationSystem,
    BehavioralAnomalyDetector,
    NodeAuthenticator,
    NetworkIntrusionDetector,
    ComprehensiveSecurityMonitor,
    SecurityEvent,
    SecurityLevel,
    AnomalyType,
    NodeBehaviorProfile,
)

# Economic system
from .economics import (
    NodeRole,
    Ticket,
    StakePool,
    StakingManager,
    TaskMatcher,
    RewardScheme,
    RewardDistributor,
    DynamicPricingEngine,
    MarketCondition,
    MarketMetrics,
    EconomicSystem,
)

# Production features
from .production import (
    ProductionDatasetManager,
    DatasetMetadata,
    PerformanceMonitor,
    PerformanceProfiler,
    OptimizationManager,
    PerformanceMetrics,
    SystemHealth,
    monitor_mining_performance,
    monitor_training_performance,
    monitor_verification_performance,
    GPUManager,
    GPUAcceleratedTrainer,
    GPUAcceleratedMiner,
    GPUMemoryManager,
    LargeModelArchitectures,
    LargeModelManager,
    ModelConfig,
    ModelArchitectures,
    CrossValidationManager,
    HyperparameterOptimizer,
    ModelArchitectureConfig,
    ValidationResults,
    HyperparameterConfig,
)

# Deployment infrastructure
from .deployment import (
    KubernetesOrchestrator,
    PoUWDeploymentManager,
    ContainerConfiguration,
    ServiceConfiguration,
    DeploymentStatus,
    ProductionMonitor,
    MetricsCollector,
    AlertingSystem,
    LoggingManager,
    HealthChecker,
    PerformanceAnalyzer,
    LoadBalancer,
    AutoScaler,
    InfrastructureAsCode,
    DeploymentAutomation,
    ConfigurationManager,
    ResourceManager,
)

# CI/CD pipeline
from .cicd import (
    GitHubActionsManager,
    WorkflowConfiguration,
    JobConfiguration,
    StepConfiguration,
    TriggerConfiguration,
    JenkinsPipelineManager,
    JenkinsfileGenerator,
    PipelineStage,
    PipelineConfiguration,
    DockerBuildManager,
    DockerImageBuilder,
    ContainerRegistry,
    ImageConfiguration,
    BuildConfiguration,
    TestAutomationManager,
    TestSuite,
    CoverageAnalyzer,
    TestConfiguration,
    TestResult,
    TestSuiteResult,
    TestType,
    PoUWTestSuites,
    DeploymentPipelineManager,
    ReleaseManager,
    DeploymentConfiguration,
    DeploymentResult,
    ReleaseInfo,
    DeploymentStrategy,
    Environment,
    PlatformType,
    PoUWDeploymentConfigurations,
    CodeQualityManager,
    SecurityScanner,
    QualityReport,
    QualityMetrics,
    QualityIssue,
    QualityConfiguration,
    QualityGateRule,
    PoUWQualityConfiguration,
)

# Data management
from .data import (
    DataShardType,
    DataShard,
    DataLocation,
    ReedSolomonEncoder,
    ConsistentHashRing,
    DataAvailabilityManager,
    DatasetSplitter,
)

# Advanced features
from .advanced import (
    VRFType,
    VRFProof,
    VerifiableRandomFunction,
    AdvancedWorkerSelection,
    ZeroNonceCommitment,
    MessageHistoryMerkleTree,
)

# Cryptographic primitives
from .crypto import (
    DKGState,
    BLSKeyShare,
    DKGComplaint,
    ThresholdSignature,
    BLSThresholdCrypto,
    DistributedKeyGeneration,
    SupervisorConsensus,
)

# Version and metadata
__version__ = "1.0.0"
__author__ = "PoUW Development Team"
__license__ = "MIT"
__description__ = (
    "Proof of Useful Work blockchain for distributed machine learning"
)

# All exports
__all__ = [
    # Blockchain core
    "Transaction",
    "PayForTaskTransaction",
    "BuyTicketsTransaction",
    "PoUWBlockHeader",
    "Block",
    "MLTask",
    "Blockchain",
    "StandardizedTransactionFormat",
    "PoUWOpReturnData",
    "PoUWOpCode",
    "create_standardized_pouw_transaction",
    "parse_standardized_pouw_transaction",
    
    # Network
    "NetworkMessage",
    "MessageHandler",
    "BlockchainMessageHandler",
    "MLMessageHandler",
    "P2PNode",
    "MessageHistory",
    "NodeStatus",
    "LeaderElectionState",
    "NodeHealthMetrics",
    "CrashRecoveryEvent",
    "LeaderElectionVote",
    "CompressedMessageBatch",
    "CrashRecoveryManager",
    "WorkerReplacementManager",
    "LeaderElectionManager",
    "MessageHistoryCompressor",
    "VPNMeshTopologyManager",
    "NetworkOperationsManager",
    
    # ML
    "MiniBatch",
    "GradientUpdate",
    "IterationMessage",
    "MLModel",
    "SimpleMLP",
    "DistributedTrainer",
    
    # Mining
    "MiningProof",
    "PoUWMiner",
    "PoUWVerifier",
    
    # Security
    "AttackType",
    "SecurityAlert",
    "GradientPoisoningDetector",
    "ByzantineFaultTolerance",
    "AttackMitigationSystem",
    "BehavioralAnomalyDetector",
    "NodeAuthenticator",
    "NetworkIntrusionDetector",
    "ComprehensiveSecurityMonitor",
    "SecurityEvent",
    "SecurityLevel",
    "AnomalyType",
    "NodeBehaviorProfile",
    
    # Economics
    "NodeRole",
    "Ticket",
    "StakePool",
    "StakingManager",
    "TaskMatcher",
    "RewardScheme",
    "RewardDistributor",
    "DynamicPricingEngine",
    "MarketCondition",
    "MarketMetrics",
    "EconomicSystem",
    
    # Production
    "ProductionDatasetManager",
    "DatasetMetadata",
    "PerformanceMonitor",
    "PerformanceProfiler",
    "OptimizationManager",
    "PerformanceMetrics",
    "SystemHealth",
    "monitor_mining_performance",
    "monitor_training_performance",
    "monitor_verification_performance",
    "GPUManager",
    "GPUAcceleratedTrainer",
    "GPUAcceleratedMiner",
    "GPUMemoryManager",
    "LargeModelArchitectures",
    "LargeModelManager",
    "ModelConfig",
    "ModelArchitectures",
    "CrossValidationManager",
    "HyperparameterOptimizer",
    "ModelArchitectureConfig",
    "ValidationResults",
    "HyperparameterConfig",
    
    # Deployment
    "KubernetesOrchestrator",
    "PoUWDeploymentManager",
    "ContainerConfiguration",
    "ServiceConfiguration",
    "DeploymentStatus",
    "ProductionMonitor",
    "MetricsCollector",
    "AlertingSystem",
    "LoggingManager",
    "HealthChecker",
    "PerformanceAnalyzer",
    "LoadBalancer",
    "AutoScaler",
    "InfrastructureAsCode",
    "DeploymentAutomation",
    "ConfigurationManager",
    "ResourceManager",
    
    # CI/CD
    "GitHubActionsManager",
    "WorkflowConfiguration",
    "JobConfiguration",
    "StepConfiguration",
    "TriggerConfiguration",
    "JenkinsPipelineManager",
    "JenkinsfileGenerator",
    "PipelineStage",
    "PipelineConfiguration",
    "DockerBuildManager",
    "DockerImageBuilder",
    "ContainerRegistry",
    "ImageConfiguration",
    "BuildConfiguration",
    "TestAutomationManager",
    "TestSuite",
    "CoverageAnalyzer",
    "TestConfiguration",
    "TestResult",
    "TestSuiteResult",
    "TestType",
    "PoUWTestSuites",
    "DeploymentPipelineManager",
    "ReleaseManager",
    "DeploymentConfiguration",
    "DeploymentResult",
    "ReleaseInfo",
    "DeploymentStrategy",
    "Environment",
    "PlatformType",
    "PoUWDeploymentConfigurations",
    "CodeQualityManager",
    "SecurityScanner",
    "QualityReport",
    "QualityMetrics",
    "QualityIssue",
    "QualityConfiguration",
    "QualityGateRule",
    "PoUWQualityConfiguration",
    
    # Data management
    "DataShardType",
    "DataShard",
    "DataLocation",
    "ReedSolomonEncoder",
    "ConsistentHashRing",
    "DataAvailabilityManager",
    "DatasetSplitter",
    
    # Advanced features
    "VRFType",
    "VRFProof",
    "VerifiableRandomFunction",
    "AdvancedWorkerSelection",
    "ZeroNonceCommitment",
    "MessageHistoryMerkleTree",
    
    # Cryptography
    "DKGState",
    "BLSKeyShare",
    "DKGComplaint",
    "ThresholdSignature",
    "BLSThresholdCrypto",
    "DistributedKeyGeneration",
    "SupervisorConsensus",
]
