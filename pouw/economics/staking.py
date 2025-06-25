"""
Staking System for PoUW - Ticket-based network participation.

This module handles the core staking mechanism where participants
purchase tickets to participate in the network with different roles.
"""

import time
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class NodeRole(Enum):
    """Network participant roles"""

    MINER = "miner"
    SUPERVISOR = "supervisor"
    EVALUATOR = "evaluator"
    VERIFIER = "verifier"
    PEER = "peer"


@dataclass
class Ticket:
    """Staking ticket for network participation"""

    ticket_id: str
    owner_id: str
    role: NodeRole
    stake_amount: float
    preferences: Dict[str, Any]
    expiration_time: int
    is_live: bool = False
    assigned_task: Optional[str] = None
    created_at: int = field(default_factory=lambda: int(time.time()))

    def is_expired(self) -> bool:
        """Check if ticket has expired"""
        return int(time.time()) > self.expiration_time

    def matches_task(self, task) -> float:
        """Calculate compatibility score with task (0.0 to 1.0)"""
        from ..blockchain.core import MLTask

        score = 0.0

        # Model type preference
        if self.preferences.get("model_types"):
            if task.model_type in self.preferences["model_types"]:
                score += 0.3
        else:
            score += 0.1  # No preference = some compatibility

        # Hardware requirements
        task_gpu = task.performance_requirements.get("gpu", False)
        node_gpu = self.preferences.get("has_gpu", False)

        if task_gpu and node_gpu:
            score += 0.4
        elif not task_gpu:
            score += 0.2

        # Dataset size preference
        task_size = task.dataset_info.get("size", 0)
        max_size = self.preferences.get("max_dataset_size", float("inf"))

        if task_size <= max_size:
            score += 0.3
        else:
            score *= 0.5  # Penalty for oversized datasets

        return min(score, 1.0)


@dataclass
class StakePool:
    """Pool of staking tickets for network participation"""

    tickets: Dict[str, Ticket] = field(default_factory=dict)
    live_tickets: List[str] = field(default_factory=list)
    target_pool_size: int = 40960
    price_adjustment_interval: int = 144  # blocks

    def add_ticket(self, ticket: Ticket):
        """Add ticket to pool"""
        self.tickets[ticket.ticket_id] = ticket

        # Tickets become live after certain number of blocks
        # For simplicity, we'll make them live immediately
        ticket.is_live = True
        self.live_tickets.append(ticket.ticket_id)

    def remove_expired_tickets(self):
        """Remove expired tickets from pool"""
        current_time = int(time.time())
        expired_tickets = []

        for ticket_id, ticket in self.tickets.items():
            if ticket.is_expired():
                expired_tickets.append(ticket_id)

        for ticket_id in expired_tickets:
            self.tickets.pop(ticket_id, None)
            if ticket_id in self.live_tickets:
                self.live_tickets.remove(ticket_id)

    def get_tickets_by_role(self, role: NodeRole) -> List[Ticket]:
        """Get all live tickets for specific role"""
        tickets = []
        for ticket_id in self.live_tickets:
            ticket = self.tickets.get(ticket_id)
            if ticket and ticket.role == role and not ticket.is_expired():
                tickets.append(ticket)
        return tickets

    def calculate_ticket_price(self) -> float:
        """Calculate current ticket price based on pool size"""
        # Simplified price adjustment - increase price if pool is large
        pool_ratio = len(self.live_tickets) / self.target_pool_size
        base_price = 10.0  # Base ticket price

        if pool_ratio > 1.2:
            return base_price * 1.5
        elif pool_ratio < 0.8:
            return base_price * 0.8
        else:
            return base_price


class StakingManager:
    """Manages staking operations and ticket lifecycle"""

    def __init__(self):
        self.stake_pool = StakePool()

    def buy_ticket(
        self, owner_id: str, role: NodeRole, stake_amount: float, preferences: Dict[str, Any]
    ) -> Ticket:
        """Process ticket purchase with stake validation"""

        current_price = self.stake_pool.calculate_ticket_price()
        if stake_amount < current_price:
            raise ValueError(
                f"Insufficient stake. Required: {current_price}, provided: {stake_amount}"
            )

        ticket_id = hashlib.sha256(f"{owner_id}_{role.value}_{time.time()}".encode()).hexdigest()[
            :16
        ]

        ticket = Ticket(
            ticket_id=ticket_id,
            owner_id=owner_id,
            role=role,
            stake_amount=stake_amount,
            preferences=preferences,
            expiration_time=int(time.time()) + 86400 * 30,  # 30 days
        )

        self.stake_pool.add_ticket(ticket)
        return ticket

    def confiscate_stake(self, node_id: str, reason: str) -> float:
        """Confiscate stake from malicious node"""

        confiscated_amount = 0.0
        tickets_to_remove = []

        # Find all tickets belonging to malicious node
        for ticket_id, ticket in self.stake_pool.tickets.items():
            if ticket.owner_id == node_id:
                confiscated_amount += ticket.stake_amount
                tickets_to_remove.append(ticket_id)

        # Remove tickets (blacklist node)
        for ticket_id in tickets_to_remove:
            self.stake_pool.tickets.pop(ticket_id, None)
            if ticket_id in self.stake_pool.live_tickets:
                self.stake_pool.live_tickets.remove(ticket_id)

        print(f"Punished node {node_id} for {reason}. Confiscated: {confiscated_amount}")
        return confiscated_amount
