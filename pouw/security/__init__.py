"""
Advanced Security Mechanisms for PoUW Implementation.

Based on the research paper's specification for gradient poisoning detection,
Byzantine fault tolerance, and attack mitigation.
"""

import numpy as np
import hashlib
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from ..ml.training import GradientUpdate


class AttackType(Enum):
    GRADIENT_POISONING = "gradient_poisoning"
    BYZANTINE_FAULT = "byzantine_fault"
    SYBIL_ATTACK = "sybil_attack"
    DOS_ATTACK = "dos_attack"
    PRE_TRAINED_MODEL = "pre_trained_model"


@dataclass
class SecurityAlert:
    """Security alert for detected malicious behavior"""
    alert_type: AttackType
    node_id: str
    timestamp: int
    confidence: float  # 0.0 to 1.0
    evidence: Dict[str, Any]
    description: str


class GradientPoisoningDetector:
    """
    Implements Byzantine-robust gradient aggregation using Krum algorithm
    and Kardam filter as described in the paper.
    """
    
    def __init__(self, byzantine_tolerance: int = 1):
        self.byzantine_tolerance = byzantine_tolerance  # f in the paper
        self.gradient_history: Dict[str, List[GradientUpdate]] = {}
        self.suspicious_nodes: Dict[str, int] = {}  # node_id -> suspicious count
        self.detection_threshold = 0.7  # Confidence threshold for detection
    
    def krum_function(self, gradient_updates: List[GradientUpdate]) -> Tuple[List[GradientUpdate], List[SecurityAlert]]:
        """
        Implement Krum algorithm for Byzantine-robust gradient aggregation.
        
        Krum selects the gradients that are closest to the majority of other gradients,
        effectively filtering out outliers that might be poisoned.
        """
        alerts = []
        
        if len(gradient_updates) < 2 * self.byzantine_tolerance + 3:
            # Not enough gradients for Krum to work effectively
            return gradient_updates, alerts
        
        # Convert gradients to numpy arrays for distance calculation
        gradient_vectors = []
        for update in gradient_updates:
            # Convert indices and values to a vector representation
            # Create a sparse vector representation
            max_index = max(update.indices) if update.indices else 0
            vector = np.zeros(max_index + 1)
            for idx, val in zip(update.indices, update.values):
                vector[idx] = val
            gradient_vectors.append(vector)
        
        # Ensure all vectors have the same length
        if gradient_vectors:
            max_len = max(len(v) for v in gradient_vectors)
            for i in range(len(gradient_vectors)):
                if len(gradient_vectors[i]) < max_len:
                    padded = np.zeros(max_len)
                    padded[:len(gradient_vectors[i])] = gradient_vectors[i]
                    gradient_vectors[i] = padded
        
        gradient_vectors = np.array(gradient_vectors)
        n = len(gradient_vectors)
        
        # Calculate pairwise distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(gradient_vectors[i] - gradient_vectors[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Calculate Krum scores for each gradient
        krum_scores = []
        for i in range(n):
            # Sum of distances to n - f - 2 closest gradients
            sorted_distances = np.sort(distances[i])
            closest_distances = sorted_distances[1:n - self.byzantine_tolerance - 1]  # Exclude self (distance 0)
            krum_score = np.sum(closest_distances)
            krum_scores.append(krum_score)
        
        # Select gradients with lowest Krum scores (most central)
        selected_indices = np.argsort(krum_scores)[:n - self.byzantine_tolerance]
        rejected_indices = np.argsort(krum_scores)[n - self.byzantine_tolerance:]
        
        # Create alerts for rejected gradients
        for idx in rejected_indices:
            update = gradient_updates[idx]
            alert = SecurityAlert(
                alert_type=AttackType.GRADIENT_POISONING,
                node_id=update.miner_id,
                timestamp=int(time.time()),
                confidence=min(0.9, krum_scores[idx] / np.mean(krum_scores)),
                evidence={
                    'krum_score': float(krum_scores[idx]),
                    'mean_score': float(np.mean(krum_scores)),
                    'gradient_norm': float(np.linalg.norm(gradient_vectors[idx]))
                },
                description=f"Gradient from {update.miner_id} rejected by Krum algorithm (outlier)"
            )
            alerts.append(alert)
            
            # Update suspicious node count
            self.suspicious_nodes[update.miner_id] = self.suspicious_nodes.get(update.miner_id, 0) + 1
        
        # Return selected gradients
        selected_updates = [gradient_updates[i] for i in selected_indices]
        return selected_updates, alerts
    
    def kardam_filter(self, gradient_updates: List[GradientUpdate]) -> Tuple[List[GradientUpdate], List[SecurityAlert]]:
        """
        Implement Kardam filter for gradient validation.
        
        Kardam filter detects gradients that are suspiciously different from
        the expected gradient distribution.
        """
        alerts = []
        
        if len(gradient_updates) < 3:
            return gradient_updates, alerts
        
        # Calculate gradient statistics
        all_gradients = []
        for update in gradient_updates:
            # Convert indices and values to vector representation
            max_index = max(update.indices) if update.indices else 0
            vector = np.zeros(max_index + 1)
            for idx, val in zip(update.indices, update.values):
                vector[idx] = val
            all_gradients.append(vector)
        
        # Ensure all vectors have the same length
        if all_gradients:
            max_len = max(len(v) for v in all_gradients)
            for i in range(len(all_gradients)):
                if len(all_gradients[i]) < max_len:
                    padded = np.zeros(max_len)
                    padded[:len(all_gradients[i])] = all_gradients[i]
                    all_gradients[i] = padded
        
        all_gradients = np.array(all_gradients)
        
        # Calculate median and median absolute deviation (MAD)
        median_grad = np.median(all_gradients, axis=0)
        mad = np.median(np.abs(all_gradients - median_grad), axis=0)
        
        # Detect outliers using modified z-score
        filtered_updates = []
        for i, update in enumerate(gradient_updates):
            grad_vector = all_gradients[i]
            
            # Calculate modified z-score
            with np.errstate(divide='ignore', invalid='ignore'):
                modified_z_scores = 0.6745 * (grad_vector - median_grad) / mad
                modified_z_scores = np.nan_to_num(modified_z_scores, nan=0.0, posinf=0.0, neginf=0.0)
            
            max_z_score = np.max(np.abs(modified_z_scores))
            
            # Threshold for outlier detection (typically 3.5)
            if max_z_score > 3.5:
                alert = SecurityAlert(
                    alert_type=AttackType.GRADIENT_POISONING,
                    node_id=update.miner_id,
                    timestamp=int(time.time()),
                    confidence=min(0.9, max_z_score / 10.0),
                    evidence={
                        'max_z_score': float(max_z_score),
                        'gradient_norm': float(np.linalg.norm(grad_vector)),
                        'median_distance': float(np.linalg.norm(grad_vector - median_grad))
                    },
                    description=f"Gradient from {update.miner_id} rejected by Kardam filter (statistical outlier)"
                )
                alerts.append(alert)
                
                # Update suspicious node count
                self.suspicious_nodes[update.miner_id] = self.suspicious_nodes.get(update.miner_id, 0) + 1
            else:
                filtered_updates.append(update)
        
        return filtered_updates, alerts
    
    def detect_gradient_poisoning(self, gradient_updates: List[GradientUpdate]) -> Tuple[List[GradientUpdate], List[SecurityAlert]]:
        """
        Combined gradient poisoning detection using both Krum and Kardam filters.
        """
        all_alerts = []
        
        # Apply Kardam filter first (statistical outlier detection)
        filtered_updates, kardam_alerts = self.kardam_filter(gradient_updates)
        all_alerts.extend(kardam_alerts)
        
        # Apply Krum algorithm (Byzantine-robust aggregation)
        if len(filtered_updates) >= 2 * self.byzantine_tolerance + 3:
            final_updates, krum_alerts = self.krum_function(filtered_updates)
            all_alerts.extend(krum_alerts)
        else:
            final_updates = filtered_updates
        
        return final_updates, all_alerts
    
    def get_suspicious_nodes(self, threshold: int = 3) -> List[str]:
        """Get nodes that have been flagged as suspicious multiple times"""
        return [node_id for node_id, count in self.suspicious_nodes.items() if count >= threshold]


class ByzantineFaultTolerance:
    """
    Implements Byzantine fault tolerance for supervisor consensus
    with 2/3 majority voting as described in the paper.
    """
    
    def __init__(self, supervisor_count: int):
        self.supervisor_count = supervisor_count
        self.byzantine_threshold = (supervisor_count * 2) // 3 + 1  # 2/3 + 1 majority
        self.supervisor_votes: Dict[str, Dict[str, Any]] = {}  # proposal_id -> {supervisor_id -> vote}
        self.proposal_outcomes: Dict[str, str] = {}  # proposal_id -> outcome
    
    def submit_supervisor_vote(self, proposal_id: str, supervisor_id: str, vote: bool, evidence: Dict[str, Any] = None) -> bool:
        """Submit a supervisor vote for a proposal"""
        if proposal_id not in self.supervisor_votes:
            self.supervisor_votes[proposal_id] = {}
        
        self.supervisor_votes[proposal_id][supervisor_id] = {
            'vote': vote,
            'timestamp': int(time.time()),
            'evidence': evidence or {}
        }
        
        # Check if we have enough votes to make a decision
        return self._check_consensus(proposal_id)
    
    def _check_consensus(self, proposal_id: str) -> bool:
        """Check if consensus has been reached for a proposal"""
        if proposal_id not in self.supervisor_votes:
            return False
        
        votes = self.supervisor_votes[proposal_id]
        
        if len(votes) < self.byzantine_threshold:
            return False
        
        # Count yes and no votes
        yes_votes = sum(1 for vote_data in votes.values() if vote_data['vote'])
        no_votes = len(votes) - yes_votes
        
        if yes_votes >= self.byzantine_threshold:
            self.proposal_outcomes[proposal_id] = 'accepted'
            return True
        elif no_votes >= self.byzantine_threshold:
            self.proposal_outcomes[proposal_id] = 'rejected'
            return True
        
        return False
    
    def get_proposal_outcome(self, proposal_id: str) -> Optional[str]:
        """Get the outcome of a proposal if consensus was reached"""
        return self.proposal_outcomes.get(proposal_id)
    
    def detect_byzantine_supervisors(self, proposal_history: Dict[str, Dict[str, Any]]) -> List[SecurityAlert]:
        """Detect supervisors exhibiting Byzantine behavior"""
        alerts = []
        
        # Analyze voting patterns for inconsistencies
        supervisor_voting_patterns = {}
        
        for proposal_id, votes in proposal_history.items():
            outcome = self.proposal_outcomes.get(proposal_id)
            if not outcome:
                continue
            
            for supervisor_id, vote_data in votes.items():
                if supervisor_id not in supervisor_voting_patterns:
                    supervisor_voting_patterns[supervisor_id] = {
                        'total_votes': 0,
                        'minority_votes': 0,
                        'inconsistent_votes': 0
                    }
                
                pattern = supervisor_voting_patterns[supervisor_id]
                pattern['total_votes'] += 1
                
                # Check if vote was in minority (potentially Byzantine)
                yes_votes = sum(1 for v in votes.values() if v['vote'])
                majority_vote = yes_votes > len(votes) // 2
                
                if vote_data['vote'] != majority_vote:
                    pattern['minority_votes'] += 1
        
        # Flag supervisors with suspicious voting patterns
        for supervisor_id, pattern in supervisor_voting_patterns.items():
            if pattern['total_votes'] < 5:  # Need enough data
                continue
            
            minority_ratio = pattern['minority_votes'] / pattern['total_votes']
            
            if minority_ratio > 0.4:  # More than 40% minority votes is suspicious
                alert = SecurityAlert(
                    alert_type=AttackType.BYZANTINE_FAULT,
                    node_id=supervisor_id,
                    timestamp=int(time.time()),
                    confidence=min(0.9, minority_ratio),
                    evidence={
                        'total_votes': pattern['total_votes'],
                        'minority_votes': pattern['minority_votes'],
                        'minority_ratio': minority_ratio
                    },
                    description=f"Supervisor {supervisor_id} exhibits Byzantine voting pattern"
                )
                alerts.append(alert)
        
        return alerts


class AttackMitigationSystem:
    """
    Comprehensive attack mitigation system implementing various
    security measures described in the paper.
    """
    
    def __init__(self):
        self.poisoning_detector = GradientPoisoningDetector()
        self.byzantine_tolerance = ByzantineFaultTolerance(5)  # Default 5 supervisors
        self.dos_protection = DOSProtection()
        self.security_alerts: List[SecurityAlert] = []
    
    def process_gradient_updates(self, gradient_updates: List[GradientUpdate]) -> Tuple[List[GradientUpdate], List[SecurityAlert]]:
        """Process gradient updates with security checks"""
        clean_updates, alerts = self.poisoning_detector.detect_gradient_poisoning(gradient_updates)
        self.security_alerts.extend(alerts)
        return clean_updates, alerts
    
    def submit_supervisor_vote(self, proposal_id: str, supervisor_id: str, vote: bool, evidence: Dict[str, Any] = None) -> bool:
        """Submit supervisor vote with Byzantine fault tolerance"""
        return self.byzantine_tolerance.submit_supervisor_vote(proposal_id, supervisor_id, vote, evidence)
    
    def detect_pre_trained_model(self, model_weights: List[float], expected_initialization: str = "random") -> Optional[SecurityAlert]:
        """Detect if a model appears to be pre-trained rather than trained from scratch"""
        
        # Check weight distribution for signs of pre-training
        weights_array = np.array(model_weights)
        
        # Pre-trained models typically have:
        # 1. Non-random weight distributions
        # 2. Specific patterns in weight magnitudes
        # 3. Low entropy in weight values
        
        # Calculate weight statistics
        weight_std = np.std(weights_array)
        weight_mean = np.mean(np.abs(weights_array))
        
        # Calculate entropy of discretized weights
        hist, _ = np.histogram(weights_array, bins=50)
        hist = hist / np.sum(hist)  # Normalize
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        
        # Heuristic detection (would need more sophisticated methods in practice)
        suspicion_score = 0.0
        
        if weight_std < 0.01:  # Too uniform
            suspicion_score += 0.3
        
        if weight_mean > 1.0:  # Unusually large weights
            suspicion_score += 0.3
        
        if entropy < 2.0:  # Low entropy
            suspicion_score += 0.4
        
        if suspicion_score > 0.6:
            alert = SecurityAlert(
                alert_type=AttackType.PRE_TRAINED_MODEL,
                node_id="unknown",  # Would need to track which node submitted
                timestamp=int(time.time()),
                confidence=suspicion_score,
                evidence={
                    'weight_std': float(weight_std),
                    'weight_mean': float(weight_mean),
                    'entropy': float(entropy),
                    'suspicion_score': suspicion_score
                },
                description="Model appears to be pre-trained rather than trained from scratch"
            )
            return alert
        
        return None
    
    def get_blacklisted_nodes(self) -> List[str]:
        """Get nodes that should be blacklisted based on security alerts"""
        node_alert_counts = {}
        
        for alert in self.security_alerts[-100:]:  # Consider recent alerts
            node_id = alert.node_id
            if node_id not in node_alert_counts:
                node_alert_counts[node_id] = 0
            node_alert_counts[node_id] += 1
        
        # Blacklist nodes with multiple security alerts
        blacklisted = [node_id for node_id, count in node_alert_counts.items() if count >= 3]
        return blacklisted


class DOSProtection:
    """
    Denial of Service attack protection with pause/resume mechanisms.
    """
    
    def __init__(self):
        self.request_counts: Dict[str, List[int]] = {}  # node_id -> [timestamps]
        self.paused_nodes: Dict[str, int] = {}  # node_id -> pause_until_timestamp
        self.rate_limit = 10  # requests per minute
        self.pause_duration = 300  # 5 minutes
    
    def check_rate_limit(self, node_id: str) -> bool:
        """Check if a node is within rate limits"""
        current_time = int(time.time())
        
        # Clean old requests (older than 1 minute)
        if node_id in self.request_counts:
            self.request_counts[node_id] = [
                timestamp for timestamp in self.request_counts[node_id]
                if current_time - timestamp < 60
            ]
        else:
            self.request_counts[node_id] = []
        
        # Check if node is paused
        if node_id in self.paused_nodes:
            if current_time < self.paused_nodes[node_id]:
                return False
            else:
                del self.paused_nodes[node_id]
        
        # Check rate limit
        if len(self.request_counts[node_id]) >= self.rate_limit:
            # Pause the node
            self.paused_nodes[node_id] = current_time + self.pause_duration
            return False
        
        # Record the request
        self.request_counts[node_id].append(current_time)
        return True
    
    def is_node_paused(self, node_id: str) -> bool:
        """Check if a node is currently paused"""
        if node_id not in self.paused_nodes:
            return False
        
        current_time = int(time.time())
        if current_time >= self.paused_nodes[node_id]:
            del self.paused_nodes[node_id]
            return False
        
        return True


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
            batch_id=i,
            model_weights=[0.1 + np.random.normal(0, 0.01) for _ in range(10)],
            gradients=[[np.random.normal(0, 0.1) for _ in range(10)]]
        )
        normal_updates.append(update)
    
    # Add a poisoned update
    poisoned_update = GradientUpdate(
        miner_id="malicious_miner",
        task_id="test_task",
        iteration=1,
        batch_id=99,
        model_weights=[0.1 for _ in range(10)],
        gradients=[[10.0 for _ in range(10)]]  # Abnormally large gradients
    )
    
    all_updates = normal_updates + [poisoned_update]
    
    # Test detection
    clean_updates, alerts = detector.detect_gradient_poisoning(all_updates)
    
    print(f"Original updates: {len(all_updates)}")
    print(f"Clean updates after filtering: {len(clean_updates)}")
    print(f"Security alerts generated: {len(alerts)}")
    
    for alert in alerts:
        print(f"  - {alert.alert_type.value}: {alert.node_id} (confidence: {alert.confidence:.2f})")
    
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
