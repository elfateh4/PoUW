"""
Network Intrusion Detection Module for PoUW Security.

This module provides network-level intrusion detection including:
- Network behavior analysis
- Coordinated attack detection
- Attack pattern recognition
- Suspicious connection monitoring
"""

import time
import numpy as np
from typing import Dict, List, Set, Any
from collections import defaultdict
from enum import Enum
import logging

from . import SecurityAlert, AttackType
from .anomaly_detection import SecurityLevel


class NetworkIntrusionDetector:
    """Network intrusion detection system for the PoUW network"""

    def __init__(self):
        self.known_attack_patterns: Dict[str, Dict[str, Any]] = {}
        self.detection_rules: List[Dict[str, Any]] = []
        self.alert_history: List[SecurityAlert] = []
        self.node_connections: Dict[str, Set[str]] = defaultdict(set)
        self.network_statistics: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)

        self._initialize_detection_rules()

    def _initialize_detection_rules(self):
        """Initialize network intrusion detection rules"""
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
            {
                "name": "connection_anomaly",
                "description": "Detect abnormal connection patterns",
                "pattern": "excessive_connections",
                "threshold": 50,
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

        # Update network statistics
        self._update_network_statistics(node_id, message_count, len(connections))

        # Check for network flooding
        if message_count > 100 and time_window < 60:  # More than 100 messages per minute
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
            self.alert_history.append(alert)

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
            self.alert_history.append(alert)

        # Check for rapid connection changes
        previous_connections = self._get_previous_connections(node_id)
        if previous_connections and len(previous_connections) > 0:
            connection_change_rate = len(set(connections) - previous_connections) / max(len(previous_connections), 1)
            if connection_change_rate > 0.8:  # 80% of connections changed
                alert = SecurityAlert(
                    alert_type=AttackType.SYBIL_ATTACK,
                    node_id=node_id,
                    timestamp=int(time.time()),
                    confidence=0.6,
                    evidence={
                        "connection_change_rate": connection_change_rate,
                        "new_connections": len(set(connections) - previous_connections),
                        "previous_count": len(previous_connections),
                    },
                    description=f"Rapid connection pattern change detected from {node_id}",
                )
                alerts.append(alert)
                self.alert_history.append(alert)

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
                times = [submission_times.get(node, 0) for node in participating_nodes if node in submission_times]
                
                if len(times) >= 3:
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
                        self.alert_history.append(alert)

            # Check for coordinated voting patterns
            voting_patterns = behavior_data.get("voting_patterns", {})
            if len(voting_patterns) >= 3:
                # Analyze voting similarity
                vote_similarity = self._calculate_voting_similarity(participating_nodes, voting_patterns)
                if vote_similarity > 0.9:  # 90% voting similarity
                    alert = SecurityAlert(
                        alert_type=AttackType.BYZANTINE_FAULT,
                        node_id=",".join(participating_nodes),
                        timestamp=int(time.time()),
                        confidence=0.8,
                        evidence={
                            "coordinated_nodes": participating_nodes,
                            "vote_similarity": vote_similarity,
                            "voting_patterns": voting_patterns,
                        },
                        description="Coordinated voting attack detected",
                    )
                    alerts.append(alert)
                    self.alert_history.append(alert)

        return alerts

    def detect_network_partitioning(self, network_topology: Dict[str, List[str]]) -> List[SecurityAlert]:
        """Detect potential network partitioning attacks"""
        alerts = []
        
        # Analyze network connectivity
        all_nodes = set(network_topology.keys())
        for node_id in all_nodes:
            connected_nodes = set(network_topology.get(node_id, []))
            isolation_ratio = 1.0 - (len(connected_nodes) / max(len(all_nodes) - 1, 1))
            
            if isolation_ratio > 0.7:  # Node is isolated from 70% of network
                alert = SecurityAlert(
                    alert_type=AttackType.DOS_ATTACK,
                    node_id=node_id,
                    timestamp=int(time.time()),
                    confidence=0.7,
                    evidence={
                        "isolation_ratio": isolation_ratio,
                        "connected_nodes": len(connected_nodes),
                        "total_nodes": len(all_nodes),
                    },
                    description=f"Potential network partitioning detected - node {node_id} isolated",
                )
                alerts.append(alert)
                self.alert_history.append(alert)
        
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
                "pattern_evolution": [],
            }

        pattern = self.known_attack_patterns[pattern_key]
        pattern["count"] += 1
        pattern["last_seen"] = alert.timestamp
        pattern["evidence_samples"].append(alert.evidence)

        # Track pattern evolution
        pattern["pattern_evolution"].append({
            "timestamp": alert.timestamp,
            "confidence": alert.confidence,
        })

        # Keep only recent evidence samples (last 10)
        if len(pattern["evidence_samples"]) > 10:
            pattern["evidence_samples"] = pattern["evidence_samples"][-10:]

        # Keep only recent pattern evolution (last 20)
        if len(pattern["pattern_evolution"]) > 20:
            pattern["pattern_evolution"] = pattern["pattern_evolution"][-20:]

    def get_network_statistics(self) -> Dict[str, Any]:
        """Get comprehensive network security statistics"""
        current_time = time.time()
        recent_window = 3600  # 1 hour
        
        recent_alerts = [
            alert for alert in self.alert_history
            if current_time - alert.timestamp <= recent_window
        ]
        
        stats = {
            "total_monitored_nodes": len(self.node_connections),
            "total_alerts": len(self.alert_history),
            "recent_alerts": len(recent_alerts),
            "attack_types_seen": len(set(alert.alert_type for alert in self.alert_history)),
            "known_attack_patterns": len(self.known_attack_patterns),
            "average_connections_per_node": float(np.mean([len(conns) for conns in self.node_connections.values()])) if self.node_connections else 0,
        }
        
        # Alert breakdown by type
        alert_breakdown = {}
        for alert in recent_alerts:
            attack_type = alert.alert_type.value
            alert_breakdown[attack_type] = alert_breakdown.get(attack_type, 0) + 1
        
        stats["recent_alert_breakdown"] = alert_breakdown
        
        return stats

    def get_node_threat_level(self, node_id: str) -> str:
        """Get threat level assessment for a specific node"""
        node_alerts = [
            alert for alert in self.alert_history
            if node_id in alert.node_id and time.time() - alert.timestamp <= 3600
        ]
        
        if not node_alerts:
            return "LOW"
        
        threat_score = 0
        for alert in node_alerts:
            if alert.alert_type == AttackType.DOS_ATTACK:
                threat_score += 3
            elif alert.alert_type == AttackType.BYZANTINE_FAULT:
                threat_score += 4
            elif alert.alert_type == AttackType.SYBIL_ATTACK:
                threat_score += 3
            else:
                threat_score += 1
        
        if threat_score >= 10:
            return "CRITICAL"
        elif threat_score >= 5:
            return "HIGH"
        elif threat_score >= 2:
            return "MEDIUM"
        else:
            return "LOW"

    def clear_alert_history(self) -> None:
        """Clear alert history (for testing or maintenance)"""
        self.alert_history.clear()
        self.known_attack_patterns.clear()

    def _update_network_statistics(self, node_id: str, message_count: int, connection_count: int):
        """Update internal network statistics"""
        if node_id not in self.network_statistics:
            self.network_statistics[node_id] = {
                "total_messages": 0,
                "total_connections_seen": 0,
                "last_update": time.time(),
            }
        
        stats = self.network_statistics[node_id]
        stats["total_messages"] += message_count
        stats["total_connections_seen"] = max(stats["total_connections_seen"], connection_count)
        stats["last_update"] = time.time()

    def _get_previous_connections(self, node_id: str) -> Set[str]:
        """Get previous connections for a node (simplified for this implementation)"""
        # In a real implementation, this would track connection history
        return self.node_connections.get(node_id, set())

    def _calculate_voting_similarity(self, nodes: List[str], voting_patterns: Dict[str, Any]) -> float:
        """Calculate voting similarity between nodes"""
        if len(nodes) < 2:
            return 0.0
        
        similarities = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node1, node2 = nodes[i], nodes[j]
                if node1 in voting_patterns and node2 in voting_patterns:
                    # Simplified similarity calculation
                    pattern1 = voting_patterns[node1]
                    pattern2 = voting_patterns[node2]
                    if isinstance(pattern1, list) and isinstance(pattern2, list):
                        matching = sum(1 for a, b in zip(pattern1, pattern2) if a == b)
                        total = max(len(pattern1), len(pattern2))
                        if total > 0:
                            similarities.append(matching / total)
        
        return float(np.mean(similarities)) if similarities else 0.0
