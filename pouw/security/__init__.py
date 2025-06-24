"""
Security Module for PoUW Implementation.

This module provides comprehensive security features including:
- Gradient poisoning detection
- Byzantine fault tolerance
- Attack mitigation systems
- Enhanced anomaly detection
- Advanced authentication
- Intrusion detection
- Security monitoring
"""

import numpy as np
import hashlib
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from ..ml.training import GradientUpdate


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


class GradientPoisoningDetector:
    """Detects gradient poisoning attacks using various algorithms"""
    
    def __init__(self, byzantine_tolerance: int = 1):
        self.byzantine_tolerance = byzantine_tolerance
        self.alert_history: List[SecurityAlert] = []

    def krum_function(self, gradient_updates) -> tuple:
        """Apply Krum defense mechanism"""
        from ..ml.training import GradientUpdate

        filtered_updates = []
        alerts = []
        
        if len(gradient_updates) < 3:
            return gradient_updates, alerts
        
        # Simplified Krum implementation
        for update in gradient_updates:
            # Calculate distances to other gradients
            distances = []
            for other_update in gradient_updates:
                if other_update.miner_id != update.miner_id:
                    # Simplified distance calculation
                    dist = sum(
                        (a - b) ** 2 for a, b in zip(update.values, other_update.values)
                    )
                    distances.append(dist)

            # Check if gradient is an outlier
            if distances:
                avg_distance = sum(distances) / len(distances)
                if avg_distance > 100:  # Threshold for outlier detection
                    alert = SecurityAlert(
                        alert_type=AttackType.GRADIENT_POISONING,
                        node_id=update.miner_id,
                        timestamp=int(time.time()),
                        confidence=0.8,
                        evidence={"avg_distance": avg_distance},
                        description=f"Gradient outlier detected from {update.miner_id}",
                    )
                    alerts.append(alert)
                else:
                    filtered_updates.append(update)
            else:
                filtered_updates.append(update)

        return filtered_updates, alerts

    def kardam_filter(self, gradient_updates) -> tuple:
        """Apply Kardam statistical filter"""
        import numpy as np

        filtered_updates = []
        alerts = []
        
        if len(gradient_updates) < 3:
            return gradient_updates, alerts
        
        # Calculate gradient norms
        norms = []
        for update in gradient_updates:
            norm = np.sqrt(sum(v * v for v in update.values))
            norms.append((update, norm))

        # Use robust statistical outlier detection
        norm_values = [norm for _, norm in norms]
        
        # Use median and MAD (Median Absolute Deviation) for robust statistics
        median_norm = np.median(norm_values)
        mad = np.median([abs(norm - median_norm) for norm in norm_values])
        
        # Convert MAD to standard deviation equivalent (MAD * 1.4826 ≈ std for normal distribution)
        robust_std = mad * 1.4826 + 1e-6  # Add small epsilon to avoid division by zero

        for update, norm in norms:
            # Check if norm is statistical outlier using robust statistics
            robust_z_score = abs(norm - median_norm) / robust_std

            if robust_z_score > 3.0:  # 3-sigma rule with robust statistics
                alert = SecurityAlert(
                    alert_type=AttackType.GRADIENT_POISONING,
                    node_id=update.miner_id,
                    timestamp=int(time.time()),
                    confidence=min(robust_z_score / 5.0, 1.0),
                    evidence={"robust_z_score": float(robust_z_score), "norm": float(norm), "median_norm": float(median_norm), "mad": float(mad)},
                    description=f"Statistical outlier gradient from {update.miner_id}",
                )
                alerts.append(alert)
            else:
                filtered_updates.append(update)
        
        return filtered_updates, alerts

    def detect_gradient_poisoning(self, gradient_updates, method="krum") -> tuple:
        """
        Detect gradient poisoning attacks using specified method
        
        Args:
            gradient_updates: List of gradient updates to analyze
            method: Detection method ("krum", "kardam", or "both")
            
        Returns:
            Tuple of (filtered_updates, security_alerts)
        """
        if method == "krum":
            return self.krum_function(gradient_updates)
        elif method == "kardam":
            return self.kardam_filter(gradient_updates)
        elif method == "both":
            # Apply both filters sequentially
            filtered_updates, alerts1 = self.krum_function(gradient_updates)
            final_updates, alerts2 = self.kardam_filter(filtered_updates)
            return final_updates, alerts1 + alerts2
        else:
            raise ValueError(f"Unknown detection method: {method}")


class ByzantineFaultTolerance:
    """Byzantine fault tolerance mechanisms"""
    
    def __init__(self, supervisor_count: int = 5):
        self.supervisor_count = supervisor_count
        self.proposal_votes: Dict[str, Dict[str, bool]] = {}
        self.proposal_outcomes: Dict[str, str] = {}
        self.vote_history: Dict[str, List[Dict[str, Any]]] = {}

    def submit_supervisor_vote(
        self, proposal_id: str, supervisor_id: str, vote: bool
    ) -> bool:
        """Submit supervisor vote and check for consensus"""
        if proposal_id not in self.proposal_votes:
            self.proposal_votes[proposal_id] = {}

        self.proposal_votes[proposal_id][supervisor_id] = vote

        # Record vote in history
        if supervisor_id not in self.vote_history:
            self.vote_history[supervisor_id] = []

        self.vote_history[supervisor_id].append(
            {"proposal_id": proposal_id, "vote": vote, "timestamp": int(time.time())}
        )

        # Check for consensus (need > 2/3 majority)
        votes = self.proposal_votes[proposal_id]
        if len(votes) >= (2 * self.supervisor_count // 3) + 1:
            yes_votes = sum(1 for v in votes.values() if v)
            total_votes = len(votes)

            if yes_votes > (2 * total_votes // 3):
                self.proposal_outcomes[proposal_id] = "accepted"
                return True
            elif (total_votes - yes_votes) > (2 * total_votes // 3):
                self.proposal_outcomes[proposal_id] = "rejected"
                return True
        
        return False
    
    def get_proposal_outcome(self, proposal_id: str) -> Optional[str]:
        """Get the outcome of a proposal"""
        return self.proposal_outcomes.get(proposal_id)
    
    def detect_byzantine_supervisors(
        self, proposal_history: Dict[str, Dict[str, Any]]
    ) -> List[SecurityAlert]:
        """Detect Byzantine supervisors based on voting patterns"""
        alerts = []
        supervisor_stats = {}
        
        # Analyze voting patterns
        for proposal_id, votes in proposal_history.items():
            outcome = self.proposal_outcomes.get(proposal_id, "unknown")
            
            for supervisor_id, vote_data in votes.items():
                if supervisor_id not in supervisor_stats:
                    supervisor_stats[supervisor_id] = {
                        "total_votes": 0,
                        "minority_votes": 0,
                        "consistency_score": 1.0,
                    }

                supervisor_stats[supervisor_id]["total_votes"] += 1

                # Check if supervisor consistently votes against majority
                if outcome == "accepted" and not vote_data["vote"]:
                    supervisor_stats[supervisor_id]["minority_votes"] += 1
                elif outcome == "rejected" and vote_data["vote"]:
                    supervisor_stats[supervisor_id]["minority_votes"] += 1

        # Detect Byzantine behavior
        for supervisor_id, stats in supervisor_stats.items():
            if stats["total_votes"] >= 5:  # Need enough data
                minority_ratio = stats["minority_votes"] / stats["total_votes"]

                if minority_ratio > 0.6:  # Consistently votes against majority
                    alert = SecurityAlert(
                        alert_type=AttackType.BYZANTINE_FAULT,
                        node_id=supervisor_id,
                        timestamp=int(time.time()),
                        confidence=minority_ratio,
                        evidence={
                            "minority_ratio": minority_ratio,
                            "total_votes": stats["total_votes"],
                        },
                        description=f"Byzantine supervisor detected: {supervisor_id}",
                    )
                    alerts.append(alert)
        
        return alerts


class AttackMitigationSystem:
    """System for mitigating detected attacks"""
    
    def __init__(self):
        self.quarantined_nodes: Dict[str, int] = {}
        self.mitigation_strategies: Dict[AttackType, str] = {
            AttackType.GRADIENT_POISONING: "filter_gradients",
            AttackType.BYZANTINE_FAULT: "exclude_supervisor",
            AttackType.SYBIL_ATTACK: "verify_identity",
            AttackType.DOS_ATTACK: "rate_limit",
        }

    def mitigate_attack(self, alert: SecurityAlert) -> bool:
        """Apply mitigation strategy for detected attack"""
        strategy = self.mitigation_strategies.get(alert.alert_type)

        if strategy == "filter_gradients":
            return self._filter_malicious_gradients(alert.node_id)
        elif strategy == "exclude_supervisor":
            return self._exclude_byzantine_supervisor(alert.node_id)
        elif strategy == "verify_identity":
            return self._verify_node_identity(alert.node_id)
        elif strategy == "rate_limit":
            return self._apply_rate_limiting(alert.node_id)

        return False
        
    def _filter_malicious_gradients(self, node_id: str) -> bool:
        """Filter gradients from malicious node"""
        # Implementation would filter out gradients from this node
        self.quarantined_nodes[node_id] = int(time.time())
        return True
    
    def _exclude_byzantine_supervisor(self, supervisor_id: str) -> bool:
        """Exclude Byzantine supervisor from consensus"""
        self.quarantined_nodes[supervisor_id] = int(time.time())
        return True

    def _verify_node_identity(self, node_id: str) -> bool:
        """Verify node identity to prevent Sybil attacks"""
        # Implementation would require additional identity verification
        return True

    def _apply_rate_limiting(self, node_id: str) -> bool:
        """Apply rate limiting to prevent DOS attacks"""
        self.quarantined_nodes[node_id] = int(time.time())
        return True


# Import enhanced security features
try:
    from .enhanced import (
        AdvancedAnomalyDetector,
        AdvancedAuthentication,
        IntrusionDetectionSystem,
        SecurityMonitor,
        SecurityEvent,
        SecurityLevel,
        AnomalyType,
    )

    __all__ = [
        # Basic security
        "AttackType",
        "SecurityAlert",
        "GradientPoisoningDetector",
        "ByzantineFaultTolerance",
        "AttackMitigationSystem",
        # Enhanced security
        "AdvancedAnomalyDetector",
        "AdvancedAuthentication",
        "IntrusionDetectionSystem",
        "SecurityMonitor",
        "SecurityEvent",
        "SecurityLevel",
        "AnomalyType",
    ]

except ImportError:
    # Enhanced security module not available
    __all__ = [
        "AttackType",
        "SecurityAlert",
        "GradientPoisoningDetector",
        "ByzantineFaultTolerance",
        "AttackMitigationSystem",
    ]


# Example usage and testing
if __name__ == "__main__":
    print("Testing PoUW Security Systems...")
    
    # Test gradient poisoning detection
    detector = GradientPoisoningDetector(byzantine_tolerance=1)
    
    # Create some normal and poisoned gradient updates
    normal_updates = []
    for i in range(5):
        update = GradientUpdate(
            miner_id=f"honest_miner_{i}",
            task_id="test_task",
            iteration=1,
            epoch=1,
            indices=list(range(10)),
            values=[0.1 + np.random.normal(0, 0.01) for _ in range(10)],
        )
        normal_updates.append(update)
    
    # Add a poisoned update
    poisoned_update = GradientUpdate(
        miner_id="malicious_miner",
        task_id="test_task",
        iteration=1,
        epoch=1,
        indices=list(range(10)),
        values=[10.0 for _ in range(10)],  # Abnormally large gradients
    )
    
    all_updates = normal_updates + [poisoned_update]
    
    # Test detection
    clean_updates, alerts = detector.detect_gradient_poisoning(all_updates)
    
    print(f"Original updates: {len(all_updates)}")
    print(f"Clean updates after filtering: {len(clean_updates)}")
    print(f"Security alerts generated: {len(alerts)}")
    
    for alert in alerts:
        print(
            f"  - {alert.alert_type.value}: {alert.node_id} (confidence: {alert.confidence:.2f})"
        )
    
    # Test Byzantine fault tolerance
    bft = ByzantineFaultTolerance(5)
    
    # Simulate supervisor votes on a proposal
    proposal_id = "test_proposal_001"
    
    # 3 honest supervisors vote yes
    for i in range(3):
        bft.submit_supervisor_vote(proposal_id, f"supervisor_{i}", True)
    
    # 1 Byzantine supervisor votes no
    bft.submit_supervisor_vote(proposal_id, "byzantine_supervisor", False)
    
    outcome = bft.get_proposal_outcome(proposal_id)
    print(f"\nByzantine Fault Tolerance Test:")
    print(f"Proposal outcome: {outcome}")
    
    print("\nSecurity system tests completed!")
