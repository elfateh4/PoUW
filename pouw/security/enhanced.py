"""
Enhanced Security Features for PoUW Implementation.

This module provides advanced security features that complement the existing
security system with anomaly detection, advanced authentication, intrusion
detection, and comprehensive security monitoring.
"""

import time
import hashlib
import hmac
import secrets
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import logging

from ..ml.training import GradientUpdate
from . import SecurityAlert, AttackType


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
    """Security event for enhanced monitoring"""

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


class AdvancedAnomalyDetector:
    """Advanced anomaly detection using machine learning and statistical methods"""

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
        """Register a new node for monitoring"""
        if node_id not in self.node_profiles:
            self.node_profiles[node_id] = NodeBehaviorProfile(
                node_id=node_id, creation_time=int(time.time())
            )
            self.logger.info(f"Registered node {node_id} for anomaly detection")

    def update_computation_metrics(
        self, node_id: str, computation_time: float
    ) -> Optional[SecurityEvent]:
        """Update computation metrics and detect anomalies"""
        if node_id not in self.node_profiles:
            self.register_node(node_id)

        profile = self.node_profiles[node_id]
        profile.recent_computation_times.append(computation_time)

        # Update running statistics
        times = list(profile.recent_computation_times)
        if len(times) >= 10:
            # Use all times except the current one for baseline statistics
            baseline_times = times[:-1] if len(times) > 10 else times[:-1] if len(times) == 10 else times
            
            if len(baseline_times) >= 5:  # Need at least 5 baseline measurements
                profile.avg_computation_time = float(np.mean(baseline_times))
                profile.computation_variance = float(np.var(baseline_times))

                # Detect computational anomalies
                variance_sqrt = np.sqrt(profile.computation_variance) + 1e-6
                z_score = (
                    abs(computation_time - profile.avg_computation_time) / variance_sqrt
                )

                threshold = self.anomaly_thresholds[AnomalyType.COMPUTATIONAL]
                if z_score > threshold:
                    return self._create_security_event(
                        node_id=node_id,
                        event_type="computational_anomaly",
                        severity=(
                            SecurityLevel.HIGH if z_score > 5 else SecurityLevel.MEDIUM
                        ),
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
        """Update network communication metrics and detect anomalies"""
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
        """Analyze gradient patterns for anomalies"""
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
        """Detect temporal anomalies in node behavior"""
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
                if all(
                    interval < 1 for interval in recent_intervals
                ):  # Less than 1 second apart
                    return self._create_security_event(
                        node_id=node_id,
                        event_type="message_burst_detected",
                        severity=SecurityLevel.MEDIUM,
                        anomaly_type=AnomalyType.TEMPORAL,
                        metadata={
                            "burst_intervals": recent_intervals,
                            "message_count": len(recent_intervals) + 1,  # +1 for the actual message count
                        },
                        confidence_score=0.8,
                    )

        return None

    def _create_security_event(
        self,
        node_id: str,
        event_type: str,
        severity: SecurityLevel,
        anomaly_type: AnomalyType,
        metadata: Dict[str, Any],
        confidence_score: float,
    ) -> SecurityEvent:
        """Create a security event"""
        event = SecurityEvent(
            event_id=hashlib.sha256(
                f"{node_id}{event_type}{time.time()}".encode()
            ).hexdigest()[:16],
            event_type=event_type,
            node_id=node_id,
            timestamp=int(time.time()),
            severity=severity,
            anomaly_type=anomaly_type,
            metadata=metadata,
            confidence_score=confidence_score,
        )

        self.security_events.append(event)
        self.logger.warning(
            f"Security event detected: {event.event_type} for node {node_id}"
        )

        return event

    def get_node_risk_score(self, node_id: str) -> float:
        """Calculate overall risk score for a node"""
        if node_id not in self.node_profiles:
            return 0.0

        profile = self.node_profiles[node_id]

        # Base score from trust and reputation
        base_score = (profile.trust_score + profile.reputation_score) / 2.0

        # Adjust based on recent security events
        recent_events = [
            event
            for event in self.security_events
            if event.node_id == node_id
            and event.timestamp > (time.time() - 3600)  # Last hour
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


class AdvancedAuthentication:
    """Advanced authentication and authorization system"""

    def __init__(self):
        self.node_credentials: Dict[str, Dict[str, Any]] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.authentication_events: List[SecurityEvent] = []
        self.rate_limits: Dict[str, List[int]] = defaultdict(list)
        self.logger = logging.getLogger(__name__)

    def register_node_credentials(
        self,
        node_id: str,
        public_key: bytes,
        capabilities: List[str],
        stake_amount: float,
    ) -> str:
        """Register node credentials with capabilities"""
        credential_hash = hashlib.sha256(
            f"{node_id}{public_key.hex()}{time.time()}".encode()
        ).hexdigest()

        self.node_credentials[node_id] = {
            "credential_hash": credential_hash,
            "public_key": public_key,
            "capabilities": capabilities,
            "stake_amount": stake_amount,
            "registration_time": int(time.time()),
            "last_authentication": None,
            "failed_attempts": 0,
        }

        self.logger.info(
            f"Registered credentials for node {node_id} with capabilities {capabilities}"
        )
        return credential_hash

    def authenticate_node(
        self, node_id: str, signature: bytes, challenge: bytes
    ) -> Tuple[bool, Optional[str]]:
        """Authenticate node using digital signature"""
        if node_id not in self.node_credentials:
            return False, "Node not registered"

        # Check rate limiting
        current_time = int(time.time())
        if not self._check_rate_limit(node_id, current_time):
            return False, "Rate limit exceeded"

        credentials = self.node_credentials[node_id]

        # Verify signature (simplified - in production use proper cryptographic verification)
        expected_signature = hmac.new(
            credentials["public_key"], challenge, hashlib.sha256
        ).digest()

        if hmac.compare_digest(signature, expected_signature):
            # Successful authentication
            session_token = self._create_session(node_id)
            credentials["last_authentication"] = current_time
            credentials["failed_attempts"] = 0

            self.logger.info(f"Successfully authenticated node {node_id}")
            return True, session_token
        else:
            # Failed authentication
            credentials["failed_attempts"] += 1

            if credentials["failed_attempts"] >= 5:
                self._create_authentication_event(
                    node_id, "repeated_auth_failures", SecurityLevel.HIGH
                )

            self.logger.warning(f"Authentication failed for node {node_id}")
            return False, "Authentication failed"

    def authorize_action(self, node_id: str, session_token: str, action: str) -> bool:
        """Authorize a specific action for a node"""
        if session_token not in self.active_sessions:
            return False

        session = self.active_sessions[session_token]
        if session["node_id"] != node_id:
            return False

        # Check session validity
        if session["expires_at"] < time.time():
            del self.active_sessions[session_token]
            return False

        # Check capabilities
        credentials = self.node_credentials[node_id]
        required_capability = self._get_required_capability(action)

        if (
            required_capability
            and required_capability not in credentials["capabilities"]
        ):
            self._create_authentication_event(
                node_id, "unauthorized_action_attempt", SecurityLevel.MEDIUM
            )
            return False

        return True

    def _check_rate_limit(self, node_id: str, current_time: int) -> bool:
        """Check if node is within rate limits"""
        # Allow 10 authentication attempts per minute
        window = 60
        max_attempts = 10

        attempts = self.rate_limits[node_id]

        # Remove old attempts outside the window
        attempts[:] = [t for t in attempts if current_time - t < window]

        if len(attempts) >= max_attempts:
            return False

        attempts.append(current_time)
        return True

    def _create_session(self, node_id: str) -> str:
        """Create authentication session"""
        session_token = secrets.token_hex(32)

        self.active_sessions[session_token] = {
            "node_id": node_id,
            "created_at": time.time(),
            "expires_at": time.time() + 3600,  # 1 hour
            "last_activity": time.time(),
        }

        return session_token

    def _get_required_capability(self, action: str) -> Optional[str]:
        """Get required capability for an action"""
        capability_map = {
            "mine_block": "mining",
            "submit_gradient": "training",
            "supervise_task": "supervision",
            "evaluate_model": "evaluation",
            "verify_block": "verification",
        }
        return capability_map.get(action)

    def _create_authentication_event(
        self, node_id: str, event_type: str, severity: SecurityLevel
    ):
        """Create authentication-related security event"""
        event = SecurityEvent(
            event_id=hashlib.sha256(
                f"{node_id}{event_type}{time.time()}".encode()
            ).hexdigest()[:16],
            event_type=event_type,
            node_id=node_id,
            timestamp=int(time.time()),
            severity=severity,
            metadata={"authentication_module": True},
        )

        self.authentication_events.append(event)


class IntrusionDetectionSystem:
    """Intrusion detection system for the PoUW network"""

    def __init__(self):
        self.known_attack_patterns: Dict[str, Dict[str, Any]] = {}
        self.detection_rules: List[Dict[str, Any]] = []
        self.alert_history: List[SecurityAlert] = []
        self.node_connections: Dict[str, Set[str]] = defaultdict(set)
        self.logger = logging.getLogger(__name__)

        self._initialize_detection_rules()

    def _initialize_detection_rules(self):
        """Initialize intrusion detection rules"""
        self.detection_rules = [
            {
                "name": "gradient_manipulation",
                "description": "Detect gradient manipulation attacks",
                "pattern": "abnormal_gradient_values",
                "threshold": 5,
                "severity": SecurityLevel.HIGH,
            },
            {
                "name": "consensus_disruption",
                "description": "Detect consensus disruption attempts",
                "pattern": "repeated_minority_votes",
                "threshold": 3,
                "severity": SecurityLevel.CRITICAL,
            },
            {
                "name": "resource_exhaustion",
                "description": "Detect resource exhaustion attacks",
                "pattern": "excessive_computation_requests",
                "threshold": 10,
                "severity": SecurityLevel.HIGH,
            },
            {
                "name": "network_flooding",
                "description": "Detect network flooding attacks",
                "pattern": "high_message_frequency",
                "threshold": 100,
                "severity": SecurityLevel.MEDIUM,
            },
        ]

    def analyze_network_behavior(
        self, node_id: str, connections: List[str], message_count: int, time_window: int
    ) -> List[SecurityAlert]:
        """Analyze network behavior for intrusion patterns"""
        alerts = []

        # Update node connections
        self.node_connections[node_id] = set(connections)

        # Check for network flooding
        if (
            message_count > 100 and time_window < 60
        ):  # More than 100 messages per minute
            alert = SecurityAlert(
                alert_type=AttackType.DOS_ATTACK,
                node_id=node_id,
                timestamp=int(time.time()),
                confidence=0.8,
                evidence={
                    "message_count": message_count,
                    "time_window": time_window,
                    "rate": message_count / time_window,
                },
                description=f"Potential network flooding detected from {node_id}",
            )
            alerts.append(alert)

        # Check for abnormal connection patterns
        if len(connections) > 50:  # Unusually high number of connections
            alert = SecurityAlert(
                alert_type=AttackType.SYBIL_ATTACK,
                node_id=node_id,
                timestamp=int(time.time()),
                confidence=0.7,
                evidence={
                    "connection_count": len(connections),
                    "connections": connections[:10],  # First 10 for evidence
                },
                description=f"Potential Sybil attack detected - unusual connection pattern from {node_id}",
            )
            alerts.append(alert)

        return alerts

    def detect_coordination_attacks(
        self, participating_nodes: List[str], behavior_data: Dict[str, Any]
    ) -> List[SecurityAlert]:
        """Detect coordinated attacks across multiple nodes"""
        alerts = []

        # Look for synchronized behavior patterns
        if len(participating_nodes) >= 3:
            # Check for synchronized gradient submissions
            submission_times = behavior_data.get("gradient_submission_times", {})

            if len(submission_times) >= 3:
                times = [submission_times.get(node, 0) for node in participating_nodes]
                time_variance = np.var(times)

                # If submissions are too synchronized (within 1 second)
                if time_variance < 1.0:
                    alert = SecurityAlert(
                        alert_type=AttackType.BYZANTINE_FAULT,
                        node_id=",".join(participating_nodes),
                        timestamp=int(time.time()),
                        confidence=0.9,
                        evidence={
                            "coordinated_nodes": participating_nodes,
                            "time_variance": float(time_variance),
                            "submission_times": submission_times,
                        },
                        description="Coordinated attack detected - synchronized behavior",
                    )
                    alerts.append(alert)

        return alerts

    def update_attack_patterns(self, alert: SecurityAlert):
        """Update known attack patterns based on detected alerts"""
        pattern_key = f"{alert.alert_type.value}_{alert.node_id}"

        if pattern_key not in self.known_attack_patterns:
            self.known_attack_patterns[pattern_key] = {
                "first_seen": alert.timestamp,
                "count": 0,
                "last_seen": alert.timestamp,
                "evidence_samples": [],
            }

        pattern = self.known_attack_patterns[pattern_key]
        pattern["count"] += 1
        pattern["last_seen"] = alert.timestamp
        pattern["evidence_samples"].append(alert.evidence)

        # Keep only recent evidence samples
        if len(pattern["evidence_samples"]) > 10:
            pattern["evidence_samples"] = pattern["evidence_samples"][-10:]


class SecurityMonitor:
    """Comprehensive security monitoring and alerting system"""

    def __init__(self):
        self.anomaly_detector = AdvancedAnomalyDetector()
        self.authentication = AdvancedAuthentication()
        self.intrusion_detection = IntrusionDetectionSystem()

        self.all_security_events: List[SecurityEvent] = []
        self.security_metrics: Dict[str, float] = {}
        self.monitoring_active = False
        self.logger = logging.getLogger(__name__)

    def start_monitoring(self):
        """Start security monitoring"""
        self.monitoring_active = True
        self.logger.info("Security monitoring started")

    def stop_monitoring(self):
        """Stop security monitoring"""
        self.monitoring_active = False
        self.logger.info("Security monitoring stopped")

    def process_node_activity(
        self, node_id: str, activity_data: Dict[str, Any]
    ) -> List[SecurityEvent]:
        """Process node activity and generate security events"""
        if not self.monitoring_active:
            return []

        events = []

        # Anomaly detection
        if "computation_time" in activity_data:
            event = self.anomaly_detector.update_computation_metrics(
                node_id, activity_data["computation_time"]
            )
            if event:
                events.append(event)

        if "message_timestamp" in activity_data:
            event = self.anomaly_detector.update_network_metrics(
                node_id, activity_data["message_timestamp"]
            )
            if event:
                events.append(event)

        if "gradient_update" in activity_data:
            event = self.anomaly_detector.analyze_gradient_pattern(
                node_id, activity_data["gradient_update"]
            )
            if event:
                events.append(event)

        # Temporal anomaly detection
        event = self.anomaly_detector.detect_temporal_anomalies(
            node_id, int(time.time())
        )
        if event:
            events.append(event)

        # Store events
        self.all_security_events.extend(events)

        return events

    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive security dashboard data"""
        current_time = int(time.time())

        # Recent events (last hour)
        recent_events = [
            event
            for event in self.all_security_events
            if current_time - event.timestamp < 3600
        ]

        # Security metrics
        total_nodes = len(self.anomaly_detector.node_profiles)
        high_risk_nodes = len(
            [
                node_id
                for node_id in self.anomaly_detector.node_profiles
                if self.anomaly_detector.get_node_risk_score(node_id) < 0.5
            ]
        )

        # Event counts by severity
        severity_counts = {level.value: 0 for level in SecurityLevel}
        for event in recent_events:
            severity_counts[event.severity.value] += 1

        return {
            "monitoring_status": "active" if self.monitoring_active else "inactive",
            "total_nodes_monitored": total_nodes,
            "high_risk_nodes": high_risk_nodes,
            "recent_events_count": len(recent_events),
            "events_by_severity": severity_counts,
            "security_metrics": self.security_metrics,
            "last_updated": current_time,
        }

    def generate_security_report(self, time_period: int = 3600) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        current_time = int(time.time())
        start_time = current_time - time_period

        # Filter events in time period
        period_events = [
            event
            for event in self.all_security_events
            if start_time <= event.timestamp <= current_time
        ]

        # Analyze trends
        anomaly_trends = defaultdict(int)
        node_risk_distribution = {}

        for event in period_events:
            if event.anomaly_type:
                anomaly_trends[event.anomaly_type.value] += 1

        for node_id in self.anomaly_detector.node_profiles:
            risk_score = self.anomaly_detector.get_node_risk_score(node_id)
            risk_category = (
                "low" if risk_score > 0.7 else "medium" if risk_score > 0.3 else "high"
            )
            node_risk_distribution[risk_category] = (
                node_risk_distribution.get(risk_category, 0) + 1
            )

        return {
            "report_period": time_period,
            "total_events": len(period_events),
            "anomaly_trends": dict(anomaly_trends),
            "node_risk_distribution": node_risk_distribution,
            "authentication_failures": len(self.authentication.authentication_events),
            "intrusion_attempts": len(self.intrusion_detection.alert_history),
            "recommendations": self._generate_security_recommendations(),
        }

    def _generate_security_recommendations(self) -> List[str]:
        """Generate security recommendations based on current state"""
        recommendations = []

        # Check for high-risk nodes
        high_risk_count = len(
            [
                node_id
                for node_id in self.anomaly_detector.node_profiles
                if self.anomaly_detector.get_node_risk_score(node_id) < 0.3
            ]
        )

        if high_risk_count > 0:
            recommendations.append(
                f"Review {high_risk_count} high-risk nodes - consider additional verification"
            )

        # Check for recent critical events
        recent_critical = [
            event
            for event in self.all_security_events
            if event.severity == SecurityLevel.CRITICAL
            and event.timestamp > (time.time() - 1800)  # Last 30 minutes
        ]

        if recent_critical:
            recommendations.append(
                f"Address {len(recent_critical)} critical security events immediately"
            )

        # Check authentication patterns
        failed_auths = len(
            [
                event
                for event in self.authentication.authentication_events
                if event.event_type == "repeated_auth_failures"
            ]
        )

        if failed_auths > 5:
            recommendations.append(
                "High number of authentication failures - review access controls"
            )

        return recommendations
