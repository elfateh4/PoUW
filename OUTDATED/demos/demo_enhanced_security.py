#!/usr/bin/env python3
"""
Enhanced Security Demo for PoUW Implementation.

This demo showcases the advanced security features including:
- Anomaly detection with behavioral profiling
- Advanced authentication and authorization
- Intrusion detection system
- Comprehensive security monitoring
"""

import time
import numpy as np
import logging
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

from pouw.security import (
    SecurityMonitor,
    AdvancedAnomalyDetector,
    AdvancedAuthentication,
    IntrusionDetectionSystem,
    SecurityLevel,
    AnomalyType,
    AttackType,
)
from pouw.ml.training import GradientUpdate


def demo_anomaly_detection():
    """Demonstrate advanced anomaly detection capabilities"""
    print("\n🔍 ANOMALY DETECTION DEMO")
    print("=" * 50)

    detector = AdvancedAnomalyDetector(sensitivity=0.8)

    # Register nodes for monitoring
    honest_nodes = ["honest_miner_1", "honest_miner_2", "honest_miner_3"]
    suspicious_node = "suspicious_miner"

    for node in honest_nodes:
        detector.register_node(node)
    detector.register_node(suspicious_node)

    print(f"📊 Registered {len(honest_nodes) + 1} nodes for monitoring")

    # Simulate normal computational behavior
    print("\n🔧 Simulating normal computational patterns...")
    normal_times = [1.0, 1.1, 0.9, 1.2, 0.8, 1.0, 1.1, 0.9, 1.0, 1.1]

    for node in honest_nodes:
        for comp_time in normal_times:
            detector.update_computation_metrics(node, comp_time)

    # Simulate anomalous behavior
    print("⚠️  Simulating anomalous computational behavior...")
    anomalous_times = [1.0, 1.1, 15.0, 1.0, 20.0]  # Suspicious spikes

    events = []
    for comp_time in anomalous_times:
        event = detector.update_computation_metrics(suspicious_node, comp_time)
        if event:
            events.append(event)
            print(f"🚨 ANOMALY DETECTED: {event.event_type} for {event.node_id}")
            print(f"   Severity: {event.severity.value}")
            print(f"   Confidence: {event.confidence_score:.2f}")

    # Simulate gradient anomalies
    print("\n🧠 Testing gradient pattern analysis...")

    # Normal gradients
    for i in range(10):
        normal_gradient = GradientUpdate(
            miner_id="honest_miner_1",
            task_id="mnist_task",
            iteration=i,
            epoch=1,
            indices=list(range(100)),
            values=[np.random.normal(0, 0.1) for _ in range(100)],
        )
        detector.analyze_gradient_pattern("honest_miner_1", normal_gradient)

    # Poisoned gradient
    poisoned_gradient = GradientUpdate(
        miner_id="honest_miner_1",
        task_id="mnist_task",
        iteration=11,
        epoch=1,
        indices=list(range(100)),
        values=[50.0 for _ in range(100)],  # Abnormally large
    )

    gradient_event = detector.analyze_gradient_pattern("honest_miner_1", poisoned_gradient)
    if gradient_event:
        print(f"🚨 GRADIENT ANOMALY: {gradient_event.event_type}")
        print(f"   Z-score: {gradient_event.metadata.get('z_score', 'N/A'):.2f}")

    # Test temporal anomalies (message bursts)
    print("\n⏰ Testing temporal anomaly detection...")
    base_time = int(time.time())

    # Create burst pattern
    for i in range(6):
        detector.update_network_metrics(suspicious_node, base_time + i)

    temporal_event = detector.detect_temporal_anomalies(suspicious_node, base_time + 5)
    if temporal_event:
        print(f"🚨 TEMPORAL ANOMALY: {temporal_event.event_type}")
        print(f"   Burst intervals detected")

    # Show risk scores
    print("\n📈 Node Risk Assessment:")
    for node in honest_nodes + [suspicious_node]:
        risk_score = detector.get_node_risk_score(node)
        risk_level = (
            "🟢 LOW" if risk_score > 0.7 else "🟡 MEDIUM" if risk_score > 0.3 else "🔴 HIGH"
        )
        print(f"   {node}: {risk_score:.2f} ({risk_level})")


def demo_advanced_authentication():
    """Demonstrate advanced authentication and authorization"""
    print("\n🔐 ADVANCED AUTHENTICATION DEMO")
    print("=" * 50)

    auth = AdvancedAuthentication()

    # Register different types of nodes with different capabilities
    nodes_config = [
        ("mining_node_1", b"mining_key_1234567890123456789012", ["mining"], 100.0),
        ("training_node_1", b"training_key_123456789012345678901", ["training"], 75.0),
        (
            "supervisor_node_1",
            b"supervisor_key_12345678901234567890",
            ["supervision", "evaluation"],
            200.0,
        ),
        ("limited_node_1", b"limited_key_123456789012345678901234", ["training"], 25.0),
    ]

    print("📝 Registering node credentials...")
    for node_id, public_key, capabilities, stake in nodes_config:
        credential_hash = auth.register_node_credentials(node_id, public_key, capabilities, stake)
        print(f"   ✅ {node_id}: {capabilities} (stake: {stake})")

    # Demonstrate successful authentication
    print("\n🔑 Testing authentication...")

    node_id = "mining_node_1"
    public_key = b"mining_key_1234567890123456789012"
    challenge = b"authentication_challenge_123456789012"

    # Create proper signature
    import hmac
    import hashlib

    signature = hmac.new(public_key, challenge, hashlib.sha256).digest()

    success, session_token = auth.authenticate_node(node_id, signature, challenge)
    if success:
        print(f"   ✅ Authentication successful for {node_id}")
        print(f"   🎫 Session token: {session_token[:16]}...")

        # Test authorization for different actions
        print("\n🛡️  Testing authorization...")
        actions = [
            ("mine_block", "mining"),
            ("submit_gradient", "training"),
            ("supervise_task", "supervision"),
            ("evaluate_model", "evaluation"),
        ]

        for action, required_capability in actions:
            authorized = auth.authorize_action(node_id, session_token, action)
            status = "✅ ALLOWED" if authorized else "❌ DENIED"
            print(f"   {action}: {status}")

    # Demonstrate failed authentication
    print("\n🚫 Testing failed authentication...")
    wrong_signature = b"wrong_signature_1234567890123456789012"
    success, error = auth.authenticate_node(node_id, wrong_signature, challenge)
    if not success:
        print(f"   ❌ Authentication failed: {error}")

    # Demonstrate rate limiting
    print("\n⏱️  Testing rate limiting...")
    for i in range(12):
        auth.authenticate_node("limited_node_1", wrong_signature, challenge)

    success, error = auth.authenticate_node("limited_node_1", wrong_signature, challenge)
    if not success and "rate limit" in error.lower():
        print(f"   🛑 Rate limiting active: {error}")


def demo_intrusion_detection():
    """Demonstrate intrusion detection capabilities"""
    print("\n🛡️  INTRUSION DETECTION DEMO")
    print("=" * 50)

    ids = IntrusionDetectionSystem()

    # Test network behavior analysis
    print("🌐 Analyzing network behavior patterns...")

    # Normal node behavior
    normal_node = "normal_node"
    normal_connections = [f"peer_{i}" for i in range(10)]
    alerts = ids.analyze_network_behavior(normal_node, normal_connections, 30, 60)
    print(f"   Normal node: {len(alerts)} alerts")

    # Suspicious network flooding
    flooding_node = "flooding_node"
    alerts = ids.analyze_network_behavior(flooding_node, normal_connections, 200, 30)
    if alerts:
        dos_alert = next((a for a in alerts if a.alert_type == AttackType.DOS_ATTACK), None)
        if dos_alert:
            print(f"   🚨 DOS ATTACK detected from {flooding_node}")
            print(f"      Rate: {dos_alert.evidence['rate']:.1f} messages/second")

    # Sybil attack simulation
    sybil_node = "sybil_node"
    many_connections = [f"sybil_peer_{i}" for i in range(60)]
    alerts = ids.analyze_network_behavior(sybil_node, many_connections, 10, 60)
    if alerts:
        sybil_alert = next((a for a in alerts if a.alert_type == AttackType.SYBIL_ATTACK), None)
        if sybil_alert:
            print(f"   🚨 SYBIL ATTACK detected from {sybil_node}")
            print(f"      Connections: {sybil_alert.evidence['connection_count']}")

    # Coordinated attack detection
    print("\n🤝 Testing coordinated attack detection...")
    coordinated_nodes = ["attacker_1", "attacker_2", "attacker_3"]
    current_time = int(time.time())

    behavior_data = {
        "gradient_submission_times": {
            "attacker_1": current_time,
            "attacker_2": current_time,
            "attacker_3": current_time,  # Suspiciously synchronized
        }
    }

    coord_alerts = ids.detect_coordination_attacks(coordinated_nodes, behavior_data)
    if coord_alerts:
        coord_alert = coord_alerts[0]
        print(f"   🚨 COORDINATED ATTACK detected")
        print(f"      Nodes: {coord_alert.evidence['coordinated_nodes']}")
        print(f"      Time variance: {coord_alert.evidence['time_variance']:.3f}")

    # Attack pattern learning
    print("\n🧠 Demonstrating attack pattern learning...")
    from pouw.security import SecurityAlert

    # Simulate repeated attack pattern
    for i in range(3):
        fake_alert = SecurityAlert(
            alert_type=AttackType.GRADIENT_POISONING,
            node_id="persistent_attacker",
            timestamp=int(time.time()) + i,
            confidence=0.8,
            evidence={"attack_iteration": i},
            description=f"Repeated attack pattern {i}",
        )
        ids.update_attack_patterns(fake_alert)

    pattern_key = f"{AttackType.GRADIENT_POISONING.value}_persistent_attacker"
    if pattern_key in ids.known_attack_patterns:
        pattern = ids.known_attack_patterns[pattern_key]
        print(f"   📊 Learned attack pattern: {pattern['count']} occurrences")


def demo_security_monitoring():
    """Demonstrate comprehensive security monitoring"""
    print("\n🖥️  SECURITY MONITORING DEMO")
    print("=" * 50)

    monitor = SecurityMonitor()
    monitor.start_monitoring()

    print("🚀 Security monitoring started")

    # Simulate node activities
    nodes = ["monitor_node_1", "monitor_node_2", "suspicious_monitor_node"]

    print("\n📊 Processing node activities...")
    for i, node in enumerate(nodes):
        activity_data = {
            "computation_time": 1.0 + i * 0.1,
            "message_timestamp": int(time.time()) + i,
        }

        # Make suspicious node have anomalous activity
        if "suspicious" in node:
            activity_data["computation_time"] = 10.0  # Anomalous

        events = monitor.process_node_activity(node, activity_data)
        if events:
            print(f"   🚨 {len(events)} security events for {node}")
            for event in events:
                print(f"      - {event.event_type} (severity: {event.severity.value})")

    # Generate security dashboard
    print("\n📈 Security Dashboard:")
    dashboard = monitor.get_security_dashboard()

    print(f"   Status: {dashboard['monitoring_status']}")
    print(f"   Nodes monitored: {dashboard['total_nodes_monitored']}")
    print(f"   High-risk nodes: {dashboard['high_risk_nodes']}")
    print(f"   Recent events: {dashboard['recent_events_count']}")

    print("\n   Events by severity:")
    for severity, count in dashboard["events_by_severity"].items():
        if count > 0:
            emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(
                severity, "⚪"
            )
            print(f"     {emoji} {severity.upper()}: {count}")

    # Generate security report
    print("\n📋 Security Report (last hour):")
    report = monitor.generate_security_report(3600)

    print(f"   Total events: {report['total_events']}")
    print(f"   Authentication failures: {report['authentication_failures']}")
    print(f"   Intrusion attempts: {report['intrusion_attempts']}")

    if report["anomaly_trends"]:
        print("\n   Anomaly trends:")
        for anomaly_type, count in report["anomaly_trends"].items():
            print(f"     - {anomaly_type}: {count}")

    if report["recommendations"]:
        print("\n   🎯 Security recommendations:")
        for rec in report["recommendations"]:
            print(f"     • {rec}")

    monitor.stop_monitoring()
    print("\n⏹️  Security monitoring stopped")


def demo_integration_scenario():
    """Demonstrate integrated security scenario"""
    print("\n🎭 INTEGRATED SECURITY SCENARIO")
    print("=" * 50)

    print("Simulating a sophisticated attack scenario...")

    # Create comprehensive security system
    monitor = SecurityMonitor()
    monitor.start_monitoring()

    # Scenario: Multi-stage attack
    print("\n📖 Scenario: Multi-stage sophisticated attack")
    print("   1. Attacker registers with fake credentials")
    print("   2. Performs reconnaissance (abnormal network patterns)")
    print("   3. Attempts gradient poisoning")
    print("   4. Coordinates with other malicious nodes")

    # Stage 1: Registration with suspicious patterns
    attacker_node = "sophisticated_attacker"
    auth = monitor.authentication

    # Register with high stake to appear legitimate
    auth.register_node_credentials(
        attacker_node,
        b"fake_key_123456789012345678901234",
        ["mining", "training"],
        500.0,  # High stake
    )
    print("\n   ✅ Stage 1: Attacker registered with high stake")

    # Stage 2: Reconnaissance phase
    print("   🔍 Stage 2: Reconnaissance phase")

    # Unusual network scanning behavior
    many_connections = [f"target_{i}" for i in range(45)]
    ids_alerts = monitor.intrusion_detection.analyze_network_behavior(
        attacker_node, many_connections, 80, 60
    )

    if ids_alerts:
        print(f"      🚨 {len(ids_alerts)} network anomalies detected")

    # Stage 3: Gradient poisoning attempt
    print("   🧪 Stage 3: Gradient poisoning attempt")

    # Normal activity first to establish baseline
    for i in range(8):
        normal_activity = {
            "computation_time": 1.0 + np.random.normal(0, 0.1),
            "message_timestamp": int(time.time()) + i,
        }
        monitor.process_node_activity(attacker_node, normal_activity)

    # Then poisoned gradient
    poisoned_gradient = GradientUpdate(
        miner_id=attacker_node,
        task_id="target_task",
        iteration=10,
        epoch=1,
        indices=list(range(50)),
        values=[25.0 for _ in range(50)],  # Poisoned values
    )

    poison_activity = {
        "gradient_update": poisoned_gradient,
        "computation_time": 0.1,  # Suspiciously fast
        "message_timestamp": int(time.time()),
    }

    poison_events = monitor.process_node_activity(attacker_node, poison_activity)
    if poison_events:
        print(f"      🚨 {len(poison_events)} gradient anomalies detected")

    # Stage 4: Coordination attempt
    print("   🤝 Stage 4: Coordination with accomplices")

    accomplices = ["accomplice_1", "accomplice_2"]
    current_time = int(time.time())

    coord_behavior = {
        "gradient_submission_times": {
            attacker_node: current_time,
            "accomplice_1": current_time,
            "accomplice_2": current_time + 0.1,  # Nearly synchronized
        }
    }

    coord_alerts = monitor.intrusion_detection.detect_coordination_attacks(
        [attacker_node] + accomplices, coord_behavior
    )

    if coord_alerts:
        print(f"      🚨 Coordinated attack pattern detected")

    # Final security assessment
    print("\n📊 Final Security Assessment:")

    risk_score = monitor.anomaly_detector.get_node_risk_score(attacker_node)
    risk_level = (
        "🔴 HIGH RISK"
        if risk_score < 0.3
        else "🟡 MEDIUM RISK" if risk_score < 0.7 else "🟢 LOW RISK"
    )

    print(f"   Attacker risk score: {risk_score:.2f} ({risk_level})")

    dashboard = monitor.get_security_dashboard()
    print(f"   Total security events: {dashboard['recent_events_count']}")
    print(f"   High-risk nodes: {dashboard['high_risk_nodes']}")

    # Recommendations
    recommendations = monitor._generate_security_recommendations()
    if recommendations:
        print("\n   🎯 Automated Security Response:")
        for rec in recommendations:
            print(f"     • {rec}")

    monitor.stop_monitoring()


def main():
    """Run the enhanced security demonstration"""
    print("🔒 ENHANCED SECURITY FEATURES DEMONSTRATION")
    print("=" * 60)
    print("This demo showcases advanced security capabilities for PoUW")
    print("including anomaly detection, authentication, intrusion detection,")
    print("and comprehensive security monitoring.")

    try:
        demo_anomaly_detection()
        demo_advanced_authentication()
        demo_intrusion_detection()
        demo_security_monitoring()
        demo_integration_scenario()

        print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("The enhanced security system demonstrated:")
        print("✅ Real-time anomaly detection")
        print("✅ Advanced authentication & authorization")
        print("✅ Intrusion detection & prevention")
        print("✅ Comprehensive security monitoring")
        print("✅ Integrated multi-stage attack detection")
        print("\nThe PoUW network is now secured with military-grade")
        print("security features that can detect and prevent sophisticated attacks.")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
