"""
Behavioral Anomaly Detection Module for PoUW Security.

This module provides advanced behavioral anomaly detection including:
- Computational pattern analysis
- Network communication monitoring
- Gradient pattern detection
- Temporal behavior analysis
"""

import time
import hashlib
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging

from ..ml.training import GradientUpdate


class SecurityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyType(Enum):
    BEHAVIORAL = "behavioral"
    STATISTICAL = "statistical"
    TEMPORAL = "temporal"
    NETWORK = "network"
    COMPUTATIONAL = "computational"


@dataclass
class SecurityEvent:
    """Security event for behavioral monitoring"""

    event_id: str
    event_type: str
    node_id: str
    timestamp: int
    severity: SecurityLevel
    anomaly_type: Optional[AnomalyType] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0


@dataclass
class NodeBehaviorProfile:
    """Behavioral profile of a node for anomaly detection"""

    node_id: str
    creation_time: int

    # Computational patterns
    avg_computation_time: float = 0.0
    computation_variance: float = 0.0
    recent_computation_times: deque = field(default_factory=lambda: deque(maxlen=100))

    # Network patterns
    avg_message_frequency: float = 0.0
    message_frequency_variance: float = 0.0
    recent_message_times: deque = field(default_factory=lambda: deque(maxlen=100))

    # Gradient patterns
    gradient_norms: deque = field(default_factory=lambda: deque(maxlen=50))
    gradient_patterns: Dict[str, float] = field(default_factory=dict)

    # Trust metrics
    trust_score: float = 1.0
    reputation_score: float = 1.0
    verified_contributions: int = 0
    failed_verifications: int = 0


class BehavioralAnomalyDetector:
    """Advanced behavioral anomaly detection using statistical analysis"""

    def __init__(self, sensitivity: float = 0.8):
        self.sensitivity = sensitivity
        self.node_profiles: Dict[str, NodeBehaviorProfile] = {}
        self.global_baselines: Dict[str, float] = {}
        self.security_events: List[SecurityEvent] = []
        self.logger = logging.getLogger(__name__)

        # Adaptive thresholds
        self.anomaly_thresholds = {
            AnomalyType.BEHAVIORAL: 2.5,
            AnomalyType.STATISTICAL: 3.0,
            AnomalyType.TEMPORAL: 2.0,
            AnomalyType.NETWORK: 2.5,
            AnomalyType.COMPUTATIONAL: 3.5,
        }

    def register_node(self, node_id: str) -> None:
        """Register a new node for behavioral monitoring"""
        if node_id not in self.node_profiles:
            self.node_profiles[node_id] = NodeBehaviorProfile(
                node_id=node_id, creation_time=int(time.time())
            )
            self.logger.info(f"Registered node {node_id} for behavioral analysis")

    def update_computation_metrics(
        self, node_id: str, computation_time: float
    ) -> Optional[SecurityEvent]:
        """Update computation metrics and detect computational anomalies"""
        if node_id not in self.node_profiles:
            self.register_node(node_id)

        profile = self.node_profiles[node_id]
        profile.recent_computation_times.append(computation_time)

        # Update running statistics
        times = list(profile.recent_computation_times)
        if len(times) >= 10:
            # Use all times except the current one for baseline statistics
            baseline_times = (
                times[:-1] if len(times) > 10 else times[:-1] if len(times) == 10 else times
            )

            if len(baseline_times) >= 5:  # Need at least 5 baseline measurements
                profile.avg_computation_time = float(np.mean(baseline_times))
                profile.computation_variance = float(np.var(baseline_times))

                # Detect computational anomalies
                variance_sqrt = np.sqrt(profile.computation_variance) + 1e-6
                z_score = abs(computation_time - profile.avg_computation_time) / variance_sqrt

                threshold = self.anomaly_thresholds[AnomalyType.COMPUTATIONAL]
                if z_score > threshold:
                    return self._create_security_event(
                        node_id=node_id,
                        event_type="computational_anomaly",
                        severity=(SecurityLevel.HIGH if z_score > 5 else SecurityLevel.MEDIUM),
                        anomaly_type=AnomalyType.COMPUTATIONAL,
                        metadata={
                            "computation_time": computation_time,
                            "avg_time": profile.avg_computation_time,
                            "z_score": float(z_score),
                            "threshold": threshold,
                        },
                        confidence_score=min(z_score / 10.0, 1.0),
                    )

        return None

    def update_network_metrics(
        self, node_id: str, message_timestamp: float
    ) -> Optional[SecurityEvent]:
        """Update network communication metrics and detect network anomalies"""
        if node_id not in self.node_profiles:
            self.register_node(node_id)

        profile = self.node_profiles[node_id]
        profile.recent_message_times.append(message_timestamp)

        # Calculate message frequency
        if len(profile.recent_message_times) >= 2:
            intervals = []
            times = list(profile.recent_message_times)
            for i in range(1, len(times)):
                intervals.append(times[i] - times[i - 1])

            if len(intervals) >= 10:
                avg_interval = np.mean(intervals)
                profile.avg_message_frequency = float(1.0 / (avg_interval + 1e-6))
                profile.message_frequency_variance = float(np.var(intervals))

                # Detect frequency anomalies
                current_interval = times[-1] - times[-2]
                z_score = abs(current_interval - avg_interval) / (
                    np.sqrt(profile.message_frequency_variance) + 1e-6
                )

                if z_score > self.anomaly_thresholds[AnomalyType.NETWORK]:
                    return self._create_security_event(
                        node_id=node_id,
                        event_type="network_frequency_anomaly",
                        severity=SecurityLevel.MEDIUM,
                        anomaly_type=AnomalyType.NETWORK,
                        metadata={
                            "current_interval": current_interval,
                            "avg_interval": avg_interval,
                            "z_score": float(z_score),
                        },
                        confidence_score=min(z_score / 5.0, 1.0),
                    )

        return None

    def analyze_gradient_pattern(
        self, node_id: str, gradient_update: GradientUpdate
    ) -> Optional[SecurityEvent]:
        """Analyze gradient patterns for statistical anomalies"""
        if node_id not in self.node_profiles:
            self.register_node(node_id)

        profile = self.node_profiles[node_id]

        # Calculate gradient norm
        gradient_norm = np.sqrt(sum(v * v for v in gradient_update.values))
        profile.gradient_norms.append(gradient_norm)

        # Analyze gradient patterns
        if len(profile.gradient_norms) >= 10:
            norms = list(profile.gradient_norms)
            avg_norm = np.mean(norms)
            norm_variance = np.var(norms)

            # Statistical anomaly detection
            z_score = abs(gradient_norm - avg_norm) / (np.sqrt(norm_variance) + 1e-6)

            if z_score > self.anomaly_thresholds[AnomalyType.STATISTICAL]:
                return self._create_security_event(
                    node_id=node_id,
                    event_type="gradient_statistical_anomaly",
                    severity=SecurityLevel.HIGH,
                    anomaly_type=AnomalyType.STATISTICAL,
                    metadata={
                        "gradient_norm": float(gradient_norm),
                        "avg_norm": float(avg_norm),
                        "z_score": float(z_score),
                        "iteration": gradient_update.iteration,
                    },
                    confidence_score=min(z_score / 5.0, 1.0),
                )

        return None

    def detect_temporal_anomalies(
        self, node_id: str, current_time: float
    ) -> Optional[SecurityEvent]:
        """Detect temporal anomalies in node behavior patterns"""
        if node_id not in self.node_profiles:
            return None

        profile = self.node_profiles[node_id]

        # Check for unusual timing patterns
        if len(profile.recent_message_times) >= 5:
            times = list(profile.recent_message_times)
            intervals = [times[i] - times[i - 1] for i in range(1, len(times))]

            # Detect burst patterns (many messages in short time)
            # Check the most recent intervals (at least 4 to detect burst)
            if len(intervals) >= 4:
                recent_intervals = intervals[-4:]  # Last 4 intervals
                if all(interval < 1 for interval in recent_intervals):  # Less than 1 second apart
                    return self._create_security_event(
                        node_id=node_id,
                        event_type="message_burst_detected",
                        severity=SecurityLevel.MEDIUM,
                        anomaly_type=AnomalyType.TEMPORAL,
                        metadata={
                            "burst_intervals": recent_intervals,
                            "message_count": len(recent_intervals)
                            + 1,  # +1 for the actual message count
                        },
                        confidence_score=0.8,
                    )

        return None

    def get_node_risk_score(self, node_id: str) -> float:
        """Calculate overall risk score for a node based on behavioral analysis"""
        if node_id not in self.node_profiles:
            return 0.0

        profile = self.node_profiles[node_id]

        # Base score from trust and reputation
        base_score = (profile.trust_score + profile.reputation_score) / 2.0

        # Adjust based on recent security events
        recent_events = [
            event
            for event in self.security_events
            if event.node_id == node_id and event.timestamp > (time.time() - 3600)  # Last hour
        ]

        risk_penalty = 0.0
        for event in recent_events:
            if event.severity == SecurityLevel.CRITICAL:
                risk_penalty += 0.5
            elif event.severity == SecurityLevel.HIGH:
                risk_penalty += 0.3
            elif event.severity == SecurityLevel.MEDIUM:
                risk_penalty += 0.1

        return max(0.0, base_score - risk_penalty)

    def get_node_profile(self, node_id: str) -> Optional[NodeBehaviorProfile]:
        """Get the behavioral profile for a specific node"""
        return self.node_profiles.get(node_id)

    def get_all_security_events(self) -> List[SecurityEvent]:
        """Get all recorded security events"""
        return self.security_events.copy()

    def get_recent_security_events(self, time_window: int = 3600) -> List[SecurityEvent]:
        """Get security events from the specified time window (default: 1 hour)"""
        current_time = time.time()
        return [
            event for event in self.security_events if current_time - event.timestamp <= time_window
        ]

    def clear_security_events(self) -> None:
        """Clear all security events (for testing or maintenance)"""
        self.security_events.clear()

    def _create_security_event(
        self,
        node_id: str,
        event_type: str,
        severity: SecurityLevel,
        anomaly_type: AnomalyType,
        metadata: Dict[str, Any],
        confidence_score: float,
    ) -> SecurityEvent:
        """Create a security event for behavioral anomalies"""
        event = SecurityEvent(
            event_id=hashlib.sha256(f"{node_id}{event_type}{time.time()}".encode()).hexdigest()[
                :16
            ],
            event_type=event_type,
            node_id=node_id,
            timestamp=int(time.time()),
            severity=severity,
            anomaly_type=anomaly_type,
            metadata=metadata,
            confidence_score=confidence_score,
        )

        self.security_events.append(event)
        self.logger.warning(f"Behavioral anomaly detected: {event.event_type} for node {node_id}")

        return event
