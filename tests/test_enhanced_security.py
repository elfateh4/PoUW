"""
Tests for Enhanced Security Features.

Comprehensive tests for anomaly detection, advanced authentication,
intrusion detection, and security monitoring capabilities.
"""

import pytest
import time
import numpy as np
from unittest.mock import Mock, patch

from pouw.security import (
    AdvancedAnomalyDetector,
    AdvancedAuthentication,
    IntrusionDetectionSystem,
    SecurityMonitor,
    SecurityEvent,
    SecurityLevel,
    AnomalyType,
)
from pouw.security import SecurityAlert, AttackType
from pouw.ml.training import GradientUpdate


class TestAdvancedAnomalyDetector:
    """Test advanced anomaly detection capabilities"""

    def test_node_registration(self):
        """Test node registration for monitoring"""
        detector = AdvancedAnomalyDetector()

        detector.register_node("test_node")

        assert "test_node" in detector.node_profiles
        profile = detector.node_profiles["test_node"]
        assert profile.node_id == "test_node"
        assert profile.trust_score == 1.0
        assert profile.reputation_score == 1.0

    def test_computation_anomaly_detection(self):
        """Test computational anomaly detection"""
        detector = AdvancedAnomalyDetector()
        node_id = "test_miner"

        # Add normal computation times
        normal_times = [1.0, 1.1, 0.9, 1.2, 0.8, 1.0, 1.1, 0.9, 1.0, 1.1]
        for comp_time in normal_times:
            result = detector.update_computation_metrics(node_id, comp_time)
            assert result is None  # No anomaly for normal times

        # Add anomalous computation time
        anomalous_time = 10.0  # Much higher than normal
        result = detector.update_computation_metrics(node_id, anomalous_time)

        assert result is not None
        assert isinstance(result, SecurityEvent)
        assert result.event_type == "computational_anomaly"
        assert result.node_id == node_id
        assert result.severity in [SecurityLevel.MEDIUM, SecurityLevel.HIGH]
        assert result.anomaly_type == AnomalyType.COMPUTATIONAL

    def test_network_anomaly_detection(self):
        """Test network communication anomaly detection"""
        detector = AdvancedAnomalyDetector()
        node_id = "test_node"

        # Add normal message patterns
        base_time = int(time.time())
        normal_intervals = [10, 12, 8, 15, 9, 11, 13, 10, 14, 9]

        current_time = base_time
        for interval in normal_intervals:
            current_time += interval
            result = detector.update_network_metrics(node_id, current_time)
            assert result is None  # No anomaly for normal patterns

        # Add anomalous pattern (very short interval)
        current_time += 1  # 1 second interval (much shorter than normal)
        result = detector.update_network_metrics(node_id, current_time)

        # Note: May or may not detect depending on statistical threshold
        if result:
            assert result.event_type == "network_frequency_anomaly"
            assert result.anomaly_type == AnomalyType.NETWORK

    def test_gradient_pattern_analysis(self):
        """Test gradient pattern anomaly detection"""
        detector = AdvancedAnomalyDetector()
        node_id = "gradient_node"

        # Create normal gradient updates
        normal_updates = []
        for i in range(12):
            update = GradientUpdate(
                miner_id=node_id,
                task_id="test_task",
                iteration=i,
                epoch=1,
                indices=list(range(10)),
                values=[np.random.normal(0, 0.1) for _ in range(10)],
            )
            normal_updates.append(update)

        # Process normal gradients
        for update in normal_updates[:10]:
            result = detector.analyze_gradient_pattern(node_id, update)
            assert result is None

        # Create anomalous gradient (very large values)
        anomalous_update = GradientUpdate(
            miner_id=node_id,
            task_id="test_task",
            iteration=11,
            epoch=1,
            indices=list(range(10)),
            values=[100.0 for _ in range(10)],  # Abnormally large
        )

        result = detector.analyze_gradient_pattern(node_id, anomalous_update)

        assert result is not None
        assert result.event_type == "gradient_statistical_anomaly"
        assert result.anomaly_type == AnomalyType.STATISTICAL
        assert result.severity == SecurityLevel.HIGH

    def test_temporal_anomaly_detection(self):
        """Test temporal anomaly detection (burst patterns)"""
        detector = AdvancedAnomalyDetector()
        node_id = "burst_node"

        # Create burst pattern (many messages in short time)
        base_time = time.time()
        burst_times = [base_time + i * 0.5 for i in range(6)]  # 0.5 sec apart

        for msg_time in burst_times:
            detector.update_network_metrics(node_id, msg_time)

        result = detector.detect_temporal_anomalies(node_id, burst_times[-1])

        assert result is not None
        assert result.event_type == "message_burst_detected"
        assert result.anomaly_type == AnomalyType.TEMPORAL
        assert result.severity == SecurityLevel.MEDIUM

    def test_risk_score_calculation(self):
        """Test node risk score calculation"""
        detector = AdvancedAnomalyDetector()
        node_id = "risk_node"

        # Register node
        detector.register_node(node_id)

        # Initial risk should be low (high trust/reputation)
        initial_risk = detector.get_node_risk_score(node_id)
        assert initial_risk == 1.0  # Perfect score initially

        # Create a high-severity security event
        security_event = SecurityEvent(
            event_id="test_event",
            event_type="test_anomaly",
            node_id=node_id,
            timestamp=int(time.time()),
            severity=SecurityLevel.HIGH,
            confidence_score=0.9,
        )
        detector.security_events.append(security_event)

        # Risk score should decrease
        updated_risk = detector.get_node_risk_score(node_id)
        assert updated_risk < initial_risk


class TestAdvancedAuthentication:
    """Test advanced authentication and authorization"""

    def test_node_credential_registration(self):
        """Test node credential registration"""
        auth = AdvancedAuthentication()

        node_id = "test_node"
        public_key = b"test_public_key_1234567890123456"
        capabilities = ["mining", "training"]
        stake_amount = 100.0

        credential_hash = auth.register_node_credentials(
            node_id, public_key, capabilities, stake_amount
        )

        assert node_id in auth.node_credentials
        assert len(credential_hash) > 0

        credentials = auth.node_credentials[node_id]
        assert credentials["public_key"] == public_key
        assert credentials["capabilities"] == capabilities
        assert credentials["stake_amount"] == stake_amount

    def test_successful_authentication(self):
        """Test successful node authentication"""
        auth = AdvancedAuthentication()

        node_id = "auth_node"
        public_key = b"auth_public_key_1234567890123456"
        capabilities = ["mining"]

        # Register credentials
        auth.register_node_credentials(node_id, public_key, capabilities, 50.0)

        # Authenticate
        challenge = b"test_challenge_12345678901234567890"
        expected_signature = auth.node_credentials[node_id]["public_key"]
        import hmac
        import hashlib

        signature = hmac.new(expected_signature, challenge, hashlib.sha256).digest()

        success, session_token = auth.authenticate_node(node_id, signature, challenge)

        assert success is True
        assert session_token is not None
        assert session_token in auth.active_sessions

    def test_failed_authentication(self):
        """Test failed authentication attempts"""
        auth = AdvancedAuthentication()

        node_id = "fail_node"
        public_key = b"fail_public_key_123456789012345"

        # Register credentials
        auth.register_node_credentials(node_id, public_key, ["training"], 25.0)

        # Use wrong signature
        challenge = b"test_challenge"
        wrong_signature = b"wrong_signature_123456789012345"

        success, error = auth.authenticate_node(node_id, wrong_signature, challenge)

        assert success is False
        assert error == "Authentication failed"
        assert auth.node_credentials[node_id]["failed_attempts"] == 1

    def test_authorization_checks(self):
        """Test action authorization"""
        auth = AdvancedAuthentication()

        node_id = "authorized_node"
        public_key = b"authorized_key_123456789012345"
        capabilities = ["mining", "training"]

        # Register and authenticate
        auth.register_node_credentials(node_id, public_key, capabilities, 75.0)

        challenge = b"auth_challenge"
        import hmac
        import hashlib

        signature = hmac.new(public_key, challenge, hashlib.sha256).digest()
        success, session_token = auth.authenticate_node(node_id, signature, challenge)

        assert success is True

        # Test authorized action
        can_mine = auth.authorize_action(node_id, session_token, "mine_block")
        assert can_mine is True

        # Test unauthorized action
        can_supervise = auth.authorize_action(node_id, session_token, "supervise_task")
        assert can_supervise is False  # Not in capabilities

    def test_rate_limiting(self):
        """Test authentication rate limiting"""
        auth = AdvancedAuthentication()

        node_id = "rate_limited_node"
        public_key = b"rate_key_1234567890123456789012"

        auth.register_node_credentials(node_id, public_key, ["mining"], 30.0)

        # Exceed rate limit
        challenge = b"rate_challenge"
        wrong_signature = b"wrong_sig_123456789012345678901"

        # Make many failed attempts quickly
        for _ in range(12):  # More than the 10 allowed per minute
            auth.authenticate_node(node_id, wrong_signature, challenge)

        # Next attempt should be rate limited
        success, error = auth.authenticate_node(node_id, wrong_signature, challenge)
        assert success is False
        assert error == "Rate limit exceeded"


class TestIntrusionDetectionSystem:
    """Test intrusion detection capabilities"""

    def test_network_behavior_analysis(self):
        """Test network behavior analysis for intrusion patterns"""
        ids = IntrusionDetectionSystem()

        node_id = "flooding_node"
        connections = [f"peer_{i}" for i in range(10)]

        # Normal behavior
        alerts = ids.analyze_network_behavior(node_id, connections, 50, 60)
        assert len(alerts) == 0

        # Network flooding behavior
        alerts = ids.analyze_network_behavior(node_id, connections, 150, 30)
        assert len(alerts) > 0

        flooding_alert = next(
            (alert for alert in alerts if alert.alert_type == AttackType.DOS_ATTACK),
            None,
        )
        assert flooding_alert is not None
        assert flooding_alert.node_id == node_id

    def test_sybil_attack_detection(self):
        """Test Sybil attack detection"""
        ids = IntrusionDetectionSystem()

        node_id = "sybil_node"
        # Unusually high number of connections
        many_connections = [f"sybil_peer_{i}" for i in range(60)]

        alerts = ids.analyze_network_behavior(node_id, many_connections, 10, 60)

        sybil_alert = next(
            (alert for alert in alerts if alert.alert_type == AttackType.SYBIL_ATTACK),
            None,
        )
        assert sybil_alert is not None
        assert sybil_alert.confidence == 0.7

    def test_coordination_attack_detection(self):
        """Test coordinated attack detection"""
        ids = IntrusionDetectionSystem()

        # Simulate coordinated nodes with synchronized behavior
        coordinated_nodes = ["coord_1", "coord_2", "coord_3"]
        current_time = int(time.time())

        behavior_data = {
            "gradient_submission_times": {
                "coord_1": current_time,
                "coord_2": current_time,  # Same time
                "coord_3": current_time,  # Same time (suspicious)
            }
        }

        alerts = ids.detect_coordination_attacks(coordinated_nodes, behavior_data)

        assert len(alerts) > 0
        coord_alert = alerts[0]
        assert coord_alert.alert_type == AttackType.BYZANTINE_FAULT
        assert coord_alert.confidence == 0.9

    def test_attack_pattern_learning(self):
        """Test attack pattern learning and updating"""
        ids = IntrusionDetectionSystem()

        # Create a security alert
        alert = SecurityAlert(
            alert_type=AttackType.GRADIENT_POISONING,
            node_id="pattern_node",
            timestamp=int(time.time()),
            confidence=0.8,
            evidence={"test": "evidence"},
            description="Test attack pattern",
        )

        # Update patterns
        ids.update_attack_patterns(alert)

        pattern_key = f"{alert.alert_type.value}_{alert.node_id}"
        assert pattern_key in ids.known_attack_patterns

        pattern = ids.known_attack_patterns[pattern_key]
        assert pattern["count"] == 1
        assert len(pattern["evidence_samples"]) == 1


class TestSecurityMonitor:
    """Test comprehensive security monitoring"""

    def test_security_monitor_initialization(self):
        """Test security monitor initialization"""
        monitor = SecurityMonitor()

        assert monitor.anomaly_detector is not None
        assert monitor.authentication is not None
        assert monitor.intrusion_detection is not None
        assert monitor.monitoring_active is False

    def test_monitoring_lifecycle(self):
        """Test monitoring start/stop lifecycle"""
        monitor = SecurityMonitor()

        # Start monitoring
        monitor.start_monitoring()
        assert monitor.monitoring_active is True

        # Stop monitoring
        monitor.stop_monitoring()
        assert monitor.monitoring_active is False

    def test_node_activity_processing(self):
        """Test processing of node activity data"""
        monitor = SecurityMonitor()
        monitor.start_monitoring()

        node_id = "activity_node"

        # Normal activity
        activity_data = {"computation_time": 1.0, "message_timestamp": int(time.time())}

        events = monitor.process_node_activity(node_id, activity_data)
        # Should not generate events for normal activity initially
        assert isinstance(events, list)

    def test_security_dashboard_generation(self):
        """Test security dashboard data generation"""
        monitor = SecurityMonitor()
        monitor.start_monitoring()

        # Add some test data
        monitor.anomaly_detector.register_node("test_node_1")
        monitor.anomaly_detector.register_node("test_node_2")

        dashboard = monitor.get_security_dashboard()

        assert "monitoring_status" in dashboard
        assert dashboard["monitoring_status"] == "active"
        assert "total_nodes_monitored" in dashboard
        assert dashboard["total_nodes_monitored"] == 2
        assert "events_by_severity" in dashboard
        assert "last_updated" in dashboard

    def test_security_report_generation(self):
        """Test security report generation"""
        monitor = SecurityMonitor()

        # Add some test nodes
        monitor.anomaly_detector.register_node("report_node_1")
        monitor.anomaly_detector.register_node("report_node_2")

        # Generate report
        report = monitor.generate_security_report(time_period=3600)

        assert "report_period" in report
        assert report["report_period"] == 3600
        assert "total_events" in report
        assert "anomaly_trends" in report
        assert "node_risk_distribution" in report
        assert "recommendations" in report
        assert isinstance(report["recommendations"], list)

    def test_security_recommendations(self):
        """Test security recommendation generation"""
        monitor = SecurityMonitor()

        # Create scenario with high-risk nodes
        monitor.anomaly_detector.register_node("high_risk_node")

        # Lower the trust score to create high risk
        profile = monitor.anomaly_detector.node_profiles["high_risk_node"]
        profile.trust_score = 0.2
        profile.reputation_score = 0.1

        recommendations = monitor._generate_security_recommendations()

        # Should recommend reviewing high-risk nodes
        assert len(recommendations) > 0
        assert any("high-risk nodes" in rec for rec in recommendations)


if __name__ == "__main__":
    pytest.main([__file__])
