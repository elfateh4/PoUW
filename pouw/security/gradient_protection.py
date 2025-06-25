"""
Gradient Protection Module for PoUW Security.

This module provides gradient poisoning detection mechanisms including:
- Krum defense algorithm
- Kardam statistical filter
- Robust statistical outlier detection
"""

import time
import numpy as np
from typing import List, Tuple, Dict, Any

from ..ml.training import GradientUpdate
from . import SecurityAlert, AttackType


class GradientPoisoningDetector:
    """Detects gradient poisoning attacks using various algorithms"""

    def __init__(self, byzantine_tolerance: int = 1):
        self.byzantine_tolerance = byzantine_tolerance
        self.alert_history: List[SecurityAlert] = []

    def krum_function(
        self, gradient_updates: List[GradientUpdate]
    ) -> Tuple[List[GradientUpdate], List[SecurityAlert]]:
        """Apply Krum defense mechanism for gradient poisoning detection"""
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
                    dist = sum((a - b) ** 2 for a, b in zip(update.values, other_update.values))
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

    def kardam_filter(
        self, gradient_updates: List[GradientUpdate]
    ) -> Tuple[List[GradientUpdate], List[SecurityAlert]]:
        """Apply Kardam statistical filter with robust statistics"""
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
                    evidence={
                        "robust_z_score": float(robust_z_score),
                        "norm": float(norm),
                        "median_norm": float(median_norm),
                        "mad": float(mad),
                    },
                    description=f"Statistical outlier gradient from {update.miner_id}",
                )
                alerts.append(alert)
            else:
                filtered_updates.append(update)

        return filtered_updates, alerts

    def detect_gradient_poisoning(
        self, gradient_updates: List[GradientUpdate], method: str = "krum"
    ) -> Tuple[List[GradientUpdate], List[SecurityAlert]]:
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

    def get_alert_history(self) -> List[SecurityAlert]:
        """Get the history of gradient poisoning alerts"""
        return self.alert_history.copy()

    def clear_alert_history(self) -> None:
        """Clear the alert history"""
        self.alert_history.clear()
