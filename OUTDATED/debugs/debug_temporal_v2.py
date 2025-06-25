#!/usr/bin/env python3

import time
from pouw.security.enhanced import AdvancedAnomalyDetector


def debug_temporal_detection_v2():
    detector = AdvancedAnomalyDetector()
    node_id = "burst_node"

    # Create burst pattern (many messages in short time)
    base_time = time.time()
    burst_times = [base_time + i * 0.5 for i in range(6)]  # 0.5 sec apart

    print(f"Base time: {base_time}")
    print(f"Burst times: {burst_times}")
    print(
        f"Intervals should be: {[burst_times[i] - burst_times[i-1] for i in range(1, len(burst_times))]}"
    )

    for i, msg_time in enumerate(burst_times):
        result = detector.update_network_metrics(node_id, msg_time)  # Use float directly
        print(f"After message {i+1}: update_network_metrics returned {result}")

        # Check the stored message times
        if node_id in detector.node_profiles:
            profile = detector.node_profiles[node_id]
            stored_times = list(profile.recent_message_times)
            print(f"  Stored times: {stored_times}")
            if len(stored_times) >= 2:
                intervals = [
                    stored_times[j] - stored_times[j - 1] for j in range(1, len(stored_times))
                ]
                print(f"  Intervals: {intervals}")

    # Now call detect_temporal_anomalies
    result = detector.detect_temporal_anomalies(node_id, burst_times[-1])  # Use float directly
    print(f"detect_temporal_anomalies returned: {result}")

    # Debug the state
    if node_id in detector.node_profiles:
        profile = detector.node_profiles[node_id]
        stored_times = list(profile.recent_message_times)
        print(f"Final stored times: {stored_times}")
        if len(stored_times) >= 5:
            intervals = [stored_times[j] - stored_times[j - 1] for j in range(1, len(stored_times))]
            print(f"Final intervals: {intervals}")
            if len(intervals) >= 4:
                recent_intervals = intervals[-4:]
                print(f"Recent intervals (last 4): {recent_intervals}")
                all_less_than_1 = all(interval < 1 for interval in recent_intervals)
                print(f"All intervals < 1 second? {all_less_than_1}")


if __name__ == "__main__":
    debug_temporal_detection_v2()
