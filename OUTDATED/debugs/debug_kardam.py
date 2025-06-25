#!/usr/bin/env python3

import numpy as np
from pouw.security import GradientPoisoningDetector
from pouw.ml.training import GradientUpdate


def debug_kardam_filter():
    detector = GradientPoisoningDetector()

    # Create normal gradients
    updates = []
    for i in range(5):
        values = [np.random.normal(0, 0.1) for _ in range(10)]
        update = GradientUpdate(
            miner_id=f"miner_{i}",
            task_id="test_task",
            iteration=1,
            epoch=1,
            indices=list(range(10)),
            values=values,
        )
        updates.append(update)
        norm = np.sqrt(sum(v * v for v in values))
        print(f"Normal update {i}: norm = {norm:.4f}, values sample = {values[:3]}")

    # Add statistical outlier
    outlier_values = [100.0 for _ in range(10)]
    outlier_update = GradientUpdate(
        miner_id="outlier_miner",
        task_id="test_task",
        iteration=1,
        epoch=1,
        indices=list(range(10)),
        values=outlier_values,
    )
    updates.append(outlier_update)
    outlier_norm = np.sqrt(sum(v * v for v in outlier_values))
    print(f"Outlier update: norm = {outlier_norm:.4f}, values sample = {outlier_values[:3]}")

    print(f"\nTotal updates: {len(updates)}")

    # Debug the Kardam filter step by step
    print("\n=== Kardam Filter Debug ===")

    # Calculate gradient norms
    norms = []
    for update in updates:
        norm = np.sqrt(sum(v * v for v in update.values))
        norms.append((update, norm))
        print(f"Update {update.miner_id}: norm = {norm:.4f}")

    # Statistical outlier detection
    norm_values = [norm for _, norm in norms]

    # Use robust statistics like the fixed Kardam filter
    median_norm = np.median(norm_values)
    mad = np.median([abs(norm - median_norm) for norm in norm_values])
    robust_std = mad * 1.4826 + 1e-6

    print(f"\nRobust Statistics:")
    print(f"Median norm: {median_norm:.4f}")
    print(f"MAD: {mad:.4f}")
    print(f"Robust std: {robust_std:.4f}")

    print(f"\nRobust Z-score analysis:")
    for update, norm in norms:
        robust_z_score = abs(norm - median_norm) / robust_std
        print(f"Update {update.miner_id}: robust_z_score = {robust_z_score:.4f} (threshold = 3.0)")
        if robust_z_score > 3.0:
            print(f"  -> OUTLIER DETECTED!")

    # Now run the actual filter
    print(f"\n=== Running actual Kardam filter ===")
    filtered_updates, alerts = detector.kardam_filter(updates)

    print(f"Filtered updates: {len(filtered_updates)}")
    print(f"Alerts: {len(alerts)}")
    for alert in alerts:
        print(f"  Alert: {alert.node_id} - {alert.description}")


if __name__ == "__main__":
    debug_kardam_filter()
