"""
Comprehensive Security Monitoring Module for PoUW Security.

This module provides comprehensive security monitoring and alerting including:
- Centralized security event coordination
- Real-time security dashboard
- Security metrics and reporting
- Automated threat response coordination
"""

import time
from typing import Dict, List, Any
from collections import defaultdict
import logging

from .anomaly_detection import BehavioralAnomalyDetector, SecurityEvent, SecurityLevel
from .authentication import NodeAuthenticator
from .intrusion_detection import NetworkIntrusionDetector


class ComprehensiveSecurityMonitor:
    """Comprehensive security monitoring and alerting system"""

    def __init__(self):
        # Initialize security subsystems
        self.anomaly_detector = BehavioralAnomalyDetector()
        self.authentication = NodeAuthenticator()
        self.intrusion_detection = NetworkIntrusionDetector()

        # Central monitoring state
        self.all_security_events: List[SecurityEvent] = []
        self.security_metrics: Dict[str, float] = {}
        self.monitoring_active = False
        self.alert_thresholds: Dict[str, int] = {
            "critical_events_per_hour": 5,
            "high_risk_nodes_threshold": 3,
            "failed_auth_threshold": 10,
        }
        self.logger = logging.getLogger(__name__)

    def start_monitoring(self):
        """Start comprehensive security monitoring"""
        self.monitoring_active = True
        self.logger.info("Comprehensive security monitoring started")

    def stop_monitoring(self):
        """Stop security monitoring"""
        self.monitoring_active = False
        self.logger.info("Security monitoring stopped")

    def process_node_activity(
        self, node_id: str, activity_data: Dict[str, Any]
    ) -> List[SecurityEvent]:
        """Process node activity through all security subsystems"""
        if not self.monitoring_active:
            return []

        events = []

        # Behavioral anomaly detection
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
        event = self.anomaly_detector.detect_temporal_anomalies(node_id, time.time())
        if event:
            events.append(event)

        # Network intrusion detection
        if "network_connections" in activity_data and "message_count" in activity_data:
            intrusion_alerts = self.intrusion_detection.analyze_network_behavior(
                node_id,
                activity_data["network_connections"],
                activity_data["message_count"],
                activity_data.get("time_window", 60),
            )
            # Convert SecurityAlerts to SecurityEvents
            for alert in intrusion_alerts:
                event = SecurityEvent(
                    event_id=f"intrusion_{alert.node_id}_{alert.timestamp}",
                    event_type=f"intrusion_{alert.alert_type.value}",
                    node_id=alert.node_id,
                    timestamp=alert.timestamp,
                    severity=SecurityLevel.HIGH,  # Map from alert confidence
                    metadata=alert.evidence,
                    confidence_score=alert.confidence,
                )
                events.append(event)

        # Store all events centrally
        self.all_security_events.extend(events)

        # Update security metrics
        self._update_security_metrics()

        # Check for automatic response triggers
        self._check_automatic_responses(events)

        return events

    def process_authentication_request(
        self, node_id: str, signature: bytes, challenge: bytes
    ) -> tuple[bool, str | None]:
        """Process authentication request through security monitoring"""
        success, result = self.authentication.authenticate_node(node_id, signature, challenge)

        # Log authentication events
        if not success:
            auth_event = SecurityEvent(
                event_id=f"auth_fail_{node_id}_{int(time.time())}",
                event_type="authentication_failure",
                node_id=node_id,
                timestamp=int(time.time()),
                severity=SecurityLevel.MEDIUM,
                metadata={"reason": result or "Unknown authentication failure"},
                confidence_score=1.0,
            )
            self.all_security_events.append(auth_event)

        return success, result

    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive real-time security dashboard data"""
        current_time = int(time.time())

        # Recent events (last hour)
        recent_events = [
            event for event in self.all_security_events if current_time - event.timestamp < 3600
        ]

        # Node risk analysis
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

        # Authentication statistics
        auth_stats = self.authentication.get_authentication_statistics()

        # Network intrusion statistics
        network_stats = self.intrusion_detection.get_network_statistics()

        return {
            "monitoring_status": "active" if self.monitoring_active else "inactive",
            "timestamp": current_time,
            "node_security": {
                "total_monitored": total_nodes,
                "high_risk_count": high_risk_nodes,
                "risk_threshold_exceeded": high_risk_nodes
                > self.alert_thresholds["high_risk_nodes_threshold"],
            },
            "recent_activity": {
                "events_count": len(recent_events),
                "events_by_severity": severity_counts,
                "critical_alerts": severity_counts["critical"]
                > self.alert_thresholds["critical_events_per_hour"],
            },
            "authentication": auth_stats,
            "network_security": network_stats,
            "security_metrics": self.security_metrics,
            "system_health": self._calculate_system_health(),
        }

    def generate_security_report(self, time_period: int = 3600) -> Dict[str, Any]:
        """Generate comprehensive security report for specified time period"""
        current_time = int(time.time())
        start_time = current_time - time_period

        # Filter events in time period
        period_events = [
            event
            for event in self.all_security_events
            if start_time <= event.timestamp <= current_time
        ]

        # Analyze trends and patterns
        anomaly_trends = defaultdict(int)
        node_risk_distribution = {"low": 0, "medium": 0, "high": 0}
        event_timeline = self._create_event_timeline(period_events)

        for event in period_events:
            if event.anomaly_type:
                anomaly_trends[event.anomaly_type.value] += 1

        # Calculate node risk distribution
        for node_id in self.anomaly_detector.node_profiles:
            risk_score = self.anomaly_detector.get_node_risk_score(node_id)
            if risk_score > 0.7:
                node_risk_distribution["low"] += 1
            elif risk_score > 0.3:
                node_risk_distribution["medium"] += 1
            else:
                node_risk_distribution["high"] += 1

        # Security assessment
        security_assessment = self._assess_security_posture(period_events)

        return {
            "report_metadata": {
                "period_hours": time_period // 3600,
                "start_time": start_time,
                "end_time": current_time,
                "generated_at": current_time,
            },
            "executive_summary": {
                "total_events": len(period_events),
                "security_level": security_assessment["overall_level"],
                "critical_issues": security_assessment["critical_issues"],
                "improvement_trend": security_assessment["trend"],
            },
            "detailed_analysis": {
                "anomaly_trends": dict(anomaly_trends),
                "node_risk_distribution": node_risk_distribution,
                "event_timeline": event_timeline,
                "authentication_analysis": self._analyze_authentication_patterns(),
                "network_threat_analysis": self._analyze_network_threats(),
            },
            "recommendations": self._generate_security_recommendations(),
            "metrics": self.security_metrics,
        }

    def get_node_security_profile(self, node_id: str) -> Dict[str, Any]:
        """Get comprehensive security profile for a specific node"""
        # Behavioral profile
        behavior_profile = self.anomaly_detector.get_node_profile(node_id)
        risk_score = self.anomaly_detector.get_node_risk_score(node_id)

        # Authentication history
        capabilities = self.authentication.get_node_capabilities(node_id)

        # Threat assessment
        threat_level = self.intrusion_detection.get_node_threat_level(node_id)

        # Recent events
        recent_events = [
            event
            for event in self.all_security_events
            if event.node_id == node_id and time.time() - event.timestamp <= 3600
        ]

        return {
            "node_id": node_id,
            "risk_assessment": {
                "risk_score": risk_score,
                "threat_level": threat_level,
                "risk_category": (
                    "high" if risk_score < 0.3 else "medium" if risk_score < 0.7 else "low"
                ),
            },
            "behavioral_profile": {
                "trust_score": behavior_profile.trust_score if behavior_profile else 1.0,
                "reputation_score": behavior_profile.reputation_score if behavior_profile else 1.0,
                "monitored_since": behavior_profile.creation_time if behavior_profile else None,
            },
            "authentication": {
                "capabilities": capabilities or [],
                "registered": capabilities is not None,
            },
            "recent_activity": {
                "event_count": len(recent_events),
                "severity_breakdown": self._categorize_events_by_severity(recent_events),
            },
        }

    def _update_security_metrics(self):
        """Update internal security metrics"""
        current_time = time.time()

        # Calculate metrics
        recent_events = [
            event for event in self.all_security_events if current_time - event.timestamp <= 3600
        ]

        self.security_metrics.update(
            {
                "events_per_hour": len(recent_events),
                "average_risk_score": self._calculate_average_risk_score(),
                "authentication_success_rate": self._calculate_auth_success_rate(),
                "network_health_score": self._calculate_network_health(),
                "last_update": current_time,
            }
        )

    def _check_automatic_responses(self, events: List[SecurityEvent]):
        """Check if automatic security responses should be triggered"""
        for event in events:
            if event.severity == SecurityLevel.CRITICAL:
                self.logger.critical(
                    f"Critical security event detected: {event.event_type} from {event.node_id}"
                )
                # In a real system, this would trigger automatic mitigation
            elif event.severity == SecurityLevel.HIGH and event.confidence_score > 0.8:
                self.logger.warning(
                    f"High confidence threat detected: {event.event_type} from {event.node_id}"
                )

    def _calculate_system_health(self) -> str:
        """Calculate overall system security health"""
        current_time = time.time()

        # Count recent critical events
        recent_critical = len(
            [
                event
                for event in self.all_security_events
                if event.severity == SecurityLevel.CRITICAL
                and current_time - event.timestamp <= 3600
            ]
        )

        # Count high-risk nodes
        high_risk_nodes = len(
            [
                node_id
                for node_id in self.anomaly_detector.node_profiles
                if self.anomaly_detector.get_node_risk_score(node_id) < 0.3
            ]
        )

        if recent_critical > 5 or high_risk_nodes > 5:
            return "CRITICAL"
        elif recent_critical > 2 or high_risk_nodes > 2:
            return "WARNING"
        elif recent_critical > 0 or high_risk_nodes > 0:
            return "CAUTION"
        else:
            return "HEALTHY"

    def _generate_security_recommendations(self) -> List[str]:
        """Generate actionable security recommendations"""
        recommendations = []
        current_time = time.time()

        # Check for high-risk nodes
        high_risk_nodes = [
            node_id
            for node_id in self.anomaly_detector.node_profiles
            if self.anomaly_detector.get_node_risk_score(node_id) < 0.3
        ]

        if high_risk_nodes:
            recommendations.append(
                f"Review {len(high_risk_nodes)} high-risk nodes: {', '.join(high_risk_nodes[:5])}{'...' if len(high_risk_nodes) > 5 else ''}"
            )

        # Check for recent critical events
        recent_critical = [
            event
            for event in self.all_security_events
            if event.severity == SecurityLevel.CRITICAL and current_time - event.timestamp <= 1800
        ]

        if recent_critical:
            recommendations.append(
                f"Address {len(recent_critical)} critical security events immediately"
            )

        # Check authentication patterns
        auth_stats = self.authentication.get_authentication_statistics()
        if auth_stats.get("failed_authentications", 0) > 10:
            recommendations.append(
                "High number of authentication failures - review access controls and implement additional verification"
            )

        # Check network security
        network_stats = self.intrusion_detection.get_network_statistics()
        if network_stats.get("recent_alerts", 0) > 5:
            recommendations.append(
                "Multiple network intrusion attempts detected - enhance network monitoring and filtering"
            )

        if not recommendations:
            recommendations.append("Security posture is good - continue monitoring")

        return recommendations

    def _assess_security_posture(self, events: List[SecurityEvent]) -> Dict[str, Any]:
        """Assess overall security posture based on events"""
        critical_count = len([e for e in events if e.severity == SecurityLevel.CRITICAL])
        high_count = len([e for e in events if e.severity == SecurityLevel.HIGH])

        if critical_count > 5:
            level = "CRITICAL"
        elif critical_count > 0 or high_count > 10:
            level = "HIGH"
        elif high_count > 5:
            level = "MEDIUM"
        else:
            level = "LOW"

        return {
            "overall_level": level,
            "critical_issues": critical_count,
            "trend": "improving" if len(events) < len(self.all_security_events) // 2 else "stable",
        }

    def _create_event_timeline(self, events: List[SecurityEvent]) -> List[Dict[str, Any]]:
        """Create timeline of security events"""
        timeline = []
        for event in sorted(events, key=lambda x: x.timestamp):
            timeline.append(
                {
                    "timestamp": event.timestamp,
                    "event_type": event.event_type,
                    "severity": event.severity.value,
                    "node_id": event.node_id,
                }
            )
        return timeline[-20:]  # Return last 20 events

    def _analyze_authentication_patterns(self) -> Dict[str, Any]:
        """Analyze authentication patterns for threats"""
        stats = self.authentication.get_authentication_statistics()
        return {
            "total_registered_nodes": stats.get("total_registered_nodes", 0),
            "active_sessions": stats.get("active_sessions", 0),
            "failed_attempts": stats.get("failed_authentications", 0),
            "suspicious_patterns": stats.get("failed_authentications", 0) > 5,
        }

    def _analyze_network_threats(self) -> Dict[str, Any]:
        """Analyze network-level threats"""
        stats = self.intrusion_detection.get_network_statistics()
        return {
            "monitored_nodes": stats.get("total_monitored_nodes", 0),
            "recent_alerts": stats.get("recent_alerts", 0),
            "attack_types_detected": stats.get("attack_types_seen", 0),
            "threat_level": (
                "high"
                if stats.get("recent_alerts", 0) > 5
                else "medium" if stats.get("recent_alerts", 0) > 0 else "low"
            ),
        }

    def _categorize_events_by_severity(self, events: List[SecurityEvent]) -> Dict[str, int]:
        """Categorize events by severity level"""
        categorization = {level.value: 0 for level in SecurityLevel}
        for event in events:
            categorization[event.severity.value] += 1
        return categorization

    def _calculate_average_risk_score(self) -> float:
        """Calculate average risk score across all nodes"""
        if not self.anomaly_detector.node_profiles:
            return 1.0

        scores = [
            self.anomaly_detector.get_node_risk_score(node_id)
            for node_id in self.anomaly_detector.node_profiles
        ]
        return sum(scores) / len(scores)

    def _calculate_auth_success_rate(self) -> float:
        """Calculate authentication success rate"""
        stats = self.authentication.get_authentication_statistics()
        total_attempts = stats.get("total_registered_nodes", 0)
        failed_attempts = stats.get("failed_authentications", 0)

        if total_attempts == 0:
            return 1.0

        return max(0.0, 1.0 - (failed_attempts / total_attempts))

    def _calculate_network_health(self) -> float:
        """Calculate network security health score"""
        stats = self.intrusion_detection.get_network_statistics()
        recent_alerts = stats.get("recent_alerts", 0)

        # Simple scoring: 1.0 - (alerts / max_expected_alerts)
        return max(0.0, 1.0 - (recent_alerts / 10.0))

    def clear_all_data(self) -> None:
        """Clear all security monitoring data (for testing)"""
        self.all_security_events.clear()
        self.security_metrics.clear()
        self.anomaly_detector.clear_security_events()
        self.authentication.reset_authentication_state()
        self.intrusion_detection.clear_alert_history()
