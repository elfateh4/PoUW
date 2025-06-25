#!/usr/bin/env python3
"""
Final validation of PoUW security system enhancements.
This script demonstrates all the fixed security capabilities.
"""

import time
import numpy as np
from pouw.security import GradientPoisoningDetector, ByzantineFaultTolerance, AttackMitigationSystem
from pouw.security.enhanced import AdvancedAnomalyDetector
from pouw.ml.training import GradientUpdate


def test_gradient_poisoning_detection():
    """Test both Krum and Kardam algorithms with robust statistics"""
    print("🛡️ Testing Gradient Poisoning Detection...")

    detector = GradientPoisoningDetector()

    # Create normal gradients
    normal_updates = []
    for i in range(5):
        update = GradientUpdate(
            miner_id=f"honest_miner_{i}",
            task_id="security_test",
            iteration=1,
            epoch=1,
            indices=list(range(10)),
            values=[np.random.normal(0, 0.1) for _ in range(10)],
        )
        normal_updates.append(update)

    # Add malicious gradient
    poisoned_update = GradientUpdate(
        miner_id="malicious_miner",
        task_id="security_test",
        iteration=1,
        epoch=1,
        indices=list(range(10)),
        values=[50.0 for _ in range(10)],  # Abnormally large
    )

    all_updates = normal_updates + [poisoned_update]

    # Test Krum algorithm
    clean_krum, alerts_krum = detector.krum_function(all_updates)
    print(
        f"  ✅ Krum: {len(clean_krum)}/{len(all_updates)} updates clean, {len(alerts_krum)} alerts"
    )

    # Test Kardam filter (robust statistics)
    clean_kardam, alerts_kardam = detector.kardam_filter(all_updates)
    print(
        f"  ✅ Kardam: {len(clean_kardam)}/{len(all_updates)} updates clean, {len(alerts_kardam)} alerts"
    )

    # Test combined detection
    clean_combined, alerts_combined = detector.detect_gradient_poisoning(all_updates, method="both")
    print(
        f"  ✅ Combined: {len(clean_combined)}/{len(all_updates)} updates clean, {len(alerts_combined)} alerts"
    )

    return len(alerts_combined) > 0


def test_byzantine_fault_tolerance():
    """Test Byzantine consensus with 2/3 majority voting"""
    print("\n🗳️ Testing Byzantine Fault Tolerance...")

    bft = ByzantineFaultTolerance(supervisor_count=5)
    proposal_id = "test_proposal_001"

    # Submit votes from honest supervisors
    votes = [
        ("supervisor_1", True),
        ("supervisor_2", True),
        ("supervisor_3", True),
        ("supervisor_4", False),  # Minority vote
        ("supervisor_5", False),  # Minority vote
    ]

    consensus_reached = False
    for supervisor_id, vote in votes:
        consensus_reached = bft.submit_supervisor_vote(proposal_id, supervisor_id, vote)
        if consensus_reached:
            break

    outcome = bft.get_proposal_outcome(proposal_id)
    print(f"  ✅ Consensus reached: {consensus_reached}")
    print(f"  ✅ Proposal outcome: {outcome}")

    # Test Byzantine supervisor detection
    proposal_history = {
        proposal_id: {supervisor_id: {"vote": vote} for supervisor_id, vote in votes}
    }
    byzantine_alerts = bft.detect_byzantine_supervisors(proposal_history)
    print(f"  ✅ Byzantine supervisors detected: {len(byzantine_alerts)}")

    return consensus_reached and outcome == "accepted"


def test_advanced_anomaly_detection():
    """Test enhanced anomaly detection with robust statistics"""
    print("\n🔍 Testing Advanced Anomaly Detection...")

    detector = AdvancedAnomalyDetector()
    node_id = "test_node"

    # Test computational anomaly detection
    normal_times = [1.0, 1.1, 0.9, 1.2, 0.8, 1.0, 1.1, 0.9, 1.0, 1.1]
    for comp_time in normal_times:
        detector.update_computation_metrics(node_id, comp_time)

    # Add anomalous computation time
    anomaly_event = detector.update_computation_metrics(node_id, 10.0)
    print(f"  ✅ Computational anomaly detected: {anomaly_event is not None}")

    # Test temporal anomaly detection (burst patterns)
    burst_node = "burst_test_node"
    base_time = time.time()
    burst_times = [base_time + i * 0.5 for i in range(6)]  # 0.5 sec apart

    for msg_time in burst_times:
        detector.update_network_metrics(burst_node, msg_time)

    burst_event = detector.detect_temporal_anomalies(burst_node, burst_times[-1])
    print(f"  ✅ Temporal burst detected: {burst_event is not None}")

    return anomaly_event is not None and burst_event is not None


def test_attack_mitigation():
    """Test attack mitigation system"""
    print("\n⚔️ Testing Attack Mitigation System...")

    from pouw.security import SecurityAlert, AttackType

    mitigation = AttackMitigationSystem()

    # Create test alerts
    alerts = [
        SecurityAlert(
            alert_type=AttackType.GRADIENT_POISONING,
            node_id="malicious_node_1",
            timestamp=int(time.time()),
            confidence=0.9,
            evidence={"test": "data"},
            description="Test gradient poisoning alert",
        ),
        SecurityAlert(
            alert_type=AttackType.BYZANTINE_FAULT,
            node_id="byzantine_supervisor",
            timestamp=int(time.time()),
            confidence=0.8,
            evidence={"test": "data"},
            description="Test Byzantine fault alert",
        ),
    ]

    mitigated_count = 0
    for alert in alerts:
        success = mitigation.mitigate_attack(alert)
        if success:
            mitigated_count += 1

    print(f"  ✅ Attacks mitigated: {mitigated_count}/{len(alerts)}")
    print(f"  ✅ Quarantined nodes: {len(mitigation.quarantined_nodes)}")

    return mitigated_count == len(alerts)


def main():
    """Run comprehensive security validation"""
    print("🔐 PoUW Security System Validation")
    print("=" * 50)

    results = []

    # Test all security components
    results.append(test_gradient_poisoning_detection())
    results.append(test_byzantine_fault_tolerance())
    results.append(test_advanced_anomaly_detection())
    results.append(test_attack_mitigation())

    # Summary
    print("\n📊 Security Validation Summary")
    print("=" * 50)
    passed = sum(results)
    total = len(results)

    print(f"✅ Tests Passed: {passed}/{total}")
    print(f"✅ Success Rate: {passed/total*100:.1f}%")

    if passed == total:
        print("\n🎉 All security systems operational!")
        print("🛡️ PoUW security module is production-ready!")
    else:
        print("\n⚠️ Some security tests failed!")

    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
