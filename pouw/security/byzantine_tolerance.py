"""
Byzantine Fault Tolerance Module for PoUW Security.

This module provides Byzantine fault tolerance mechanisms including:
- Supervisor consensus voting
- Byzantine supervisor detection
- 2/3 majority rule implementation
"""

import time
from typing import Dict, List, Optional, Any

from . import SecurityAlert, AttackType


class ByzantineFaultTolerance:
    """Byzantine fault tolerance mechanisms for supervisor consensus"""

    def __init__(self, supervisor_count: int = 5):
        self.supervisor_count = supervisor_count
        self.proposal_votes: Dict[str, Dict[str, bool]] = {}
        self.proposal_outcomes: Dict[str, str] = {}
        self.vote_history: Dict[str, List[Dict[str, Any]]] = {}

    def submit_supervisor_vote(self, proposal_id: str, supervisor_id: str, vote: bool) -> bool:
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

    def get_proposal_votes(self, proposal_id: str) -> Dict[str, bool]:
        """Get all votes for a specific proposal"""
        return self.proposal_votes.get(proposal_id, {}).copy()

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

    def get_supervisor_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get voting statistics for all supervisors"""
        stats = {}

        for supervisor_id, history in self.vote_history.items():
            total_votes = len(history)
            if total_votes > 0:
                recent_votes = history[-10:]  # Last 10 votes
                stats[supervisor_id] = {
                    "total_votes": total_votes,
                    "recent_activity": len(recent_votes),
                    "last_vote_time": history[-1]["timestamp"] if history else None,
                    "participation_rate": total_votes / max(len(self.proposal_outcomes), 1),
                }

        return stats

    def reset_consensus_state(self) -> None:
        """Reset all consensus state (for testing or reinitialization)"""
        self.proposal_votes.clear()
        self.proposal_outcomes.clear()
        self.vote_history.clear()
