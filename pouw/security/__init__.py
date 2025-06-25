"""
Security Module for PoUW Implementation.

This module provides comprehensive security features organized into focused components:
- Gradient poisoning protection (gradient_protection.py)
- Byzantine fault tolerance (byzantine_tolerance.py)
- Attack mitigation systems (attack_mitigation.py)
- Behavioral anomaly detection (anomaly_detection.py)
- Node authentication (authentication.py)
- Network intrusion detection (intrusion_detection.py)
- Comprehensive security monitoring (security_monitoring.py)
"""

from typing import Dict, Any
from dataclasses import dataclass
from enum import Enum


class AttackType(Enum):
    """Types of attacks that can be detected"""

    GRADIENT_POISONING = "gradient_poisoning"
    BYZANTINE_FAULT = "byzantine_fault"
    SYBIL_ATTACK = "sybil_attack"
    DOS_ATTACK = "dos_attack"
    MODEL_INVERSION = "model_inversion"
    MEMBERSHIP_INFERENCE = "membership_inference"


@dataclass
class SecurityAlert:
    """Security alert for detected attacks or anomalies"""

    alert_type: AttackType
    node_id: str
    timestamp: int
    confidence: float
    evidence: Dict[str, Any]
    description: str


# Import refactored security components
from .gradient_protection import GradientPoisoningDetector
from .byzantine_tolerance import ByzantineFaultTolerance
from .attack_mitigation import AttackMitigationSystem

# Import advanced security features
try:
    from .anomaly_detection import (
        BehavioralAnomalyDetector,
        SecurityEvent,
        SecurityLevel,
        AnomalyType,
        NodeBehaviorProfile,
    )
    from .authentication import NodeAuthenticator
    from .intrusion_detection import NetworkIntrusionDetector
    from .security_monitoring import ComprehensiveSecurityMonitor

    __all__ = [
        # Core security types
        "AttackType",
        "SecurityAlert",
        # Basic security components
        "GradientPoisoningDetector",
        "ByzantineFaultTolerance",
        "AttackMitigationSystem",
        # Advanced security components (renamed from "enhanced")
        "BehavioralAnomalyDetector",  # was AdvancedAnomalyDetector
        "NodeAuthenticator",  # was AdvancedAuthentication
        "NetworkIntrusionDetector",  # was IntrusionDetectionSystem
        "ComprehensiveSecurityMonitor",  # was SecurityMonitor
        # Security event types
        "SecurityEvent",
        "SecurityLevel",
        "AnomalyType",
        "NodeBehaviorProfile",
    ]

except ImportError as e:
    # Fallback to basic security components only
    __all__ = [
        "AttackType",
        "SecurityAlert",
        "GradientPoisoningDetector",
        "ByzantineFaultTolerance",
        "AttackMitigationSystem",
    ]


# Backward compatibility aliases (for existing code)
try:
    # Provide aliases for old naming
    AdvancedAnomalyDetector = BehavioralAnomalyDetector
    AdvancedAuthentication = NodeAuthenticator
    IntrusionDetectionSystem = NetworkIntrusionDetector
    SecurityMonitor = ComprehensiveSecurityMonitor

    # Add aliases to exports
    __all__.extend(
        [
            "AdvancedAnomalyDetector",
            "AdvancedAuthentication",
            "IntrusionDetectionSystem",
            "SecurityMonitor",
        ]
    )

except NameError:
    # Advanced components not available
    pass
