"""
Economic system for PoUW - staking, rewards, and task matching.
"""

import time
import hashlib
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..blockchain.core import MLTask, BuyTicketsTransaction


class NodeRole(Enum):
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
    
    def matches_task(self, task: MLTask) -> float:
        """Calculate compatibility score with task (0.0 to 1.0)"""
        score = 0.0
        
        # Model type preference
        if self.preferences.get('model_types'):
            if task.model_type in self.preferences['model_types']:
                score += 0.3
        else:
            score += 0.1  # No preference = some compatibility
        
        # Hardware requirements
        task_gpu = task.performance_requirements.get('gpu', False)
        node_gpu = self.preferences.get('has_gpu', False)
        
        if task_gpu and node_gpu:
            score += 0.4
        elif not task_gpu:
            score += 0.2
        
        # Dataset size preference
        task_size = task.dataset_info.get('size', 0)
        max_size = self.preferences.get('max_dataset_size', float('inf'))
        
        if task_size <= max_size:
            score += 0.3
        else:
            score *= 0.5  # Penalty for oversized datasets
        
        return min(score, 1.0)


@dataclass
class StakePool:
    """Pool of staking tickets"""
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


class TaskMatcher:
    """Matches worker nodes to ML tasks using VRF-based selection"""
    
    def __init__(self, stake_pool: StakePool):
        self.stake_pool = stake_pool
        self.omega_s = 0.1  # Network-wide selection parameter
    
    def select_workers(self, task: MLTask, num_miners: int = 3, 
                      num_supervisors: int = 2, num_evaluators: int = 2) -> Dict[NodeRole, List[Ticket]]:
        """
        Select worker nodes for task using VRF-based random selection.
        
        This implements the worker selection algorithm from the paper.
        """
        selected = {
            NodeRole.MINER: [],
            NodeRole.SUPERVISOR: [],
            NodeRole.EVALUATOR: []
        }
        
        # Get candidates for each role
        miner_candidates = self.stake_pool.get_tickets_by_role(NodeRole.MINER)
        supervisor_candidates = self.stake_pool.get_tickets_by_role(NodeRole.SUPERVISOR)
        evaluator_candidates = self.stake_pool.get_tickets_by_role(NodeRole.EVALUATOR)
        
        # Select miners
        selected[NodeRole.MINER] = self._select_nodes_for_role(
            miner_candidates, task, num_miners
        )
        
        # Select supervisors  
        selected[NodeRole.SUPERVISOR] = self._select_nodes_for_role(
            supervisor_candidates, task, num_supervisors
        )
        
        # Select evaluators
        selected[NodeRole.EVALUATOR] = self._select_nodes_for_role(
            evaluator_candidates, task, num_evaluators
        )
        
        return selected
    
    def _select_nodes_for_role(self, candidates: List[Ticket], task: MLTask, 
                              num_needed: int) -> List[Ticket]:
        """Select nodes for specific role using compatibility and VRF"""
        
        if not candidates:
            return []
        
        # Calculate selection scores for each candidate
        scored_candidates = []
        for ticket in candidates:
            # Calculate compatibility score
            compatibility = ticket.matches_task(task)
            
            # Simulate VRF hash (in practice would use actual VRF)
            vrf_input = f"{task.task_id}_{ticket.role.value}_{ticket.ticket_id}"
            vrf_hash = int(hashlib.sha256(vrf_input.encode()).hexdigest(), 16)
            vrf_normalized = vrf_hash / (2**256 - 1)  # Normalize to [0,1]
            
            # Selection probability based on compatibility and VRF
            if compatibility >= self.omega_s * vrf_normalized:
                scored_candidates.append((ticket, compatibility * vrf_normalized))
        
        # Sort by score and select top candidates
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        selected = [ticket for ticket, _ in scored_candidates[:num_needed]]
        
        return selected


@dataclass
class RewardScheme:
    """Defines how to distribute rewards among participants"""
    miner_percentage: float = 0.6    # 60% to miners based on performance
    supervisor_percentage: float = 0.2  # 20% to supervisors
    evaluator_percentage: float = 0.15  # 15% to evaluators
    verifier_percentage: float = 0.05   # 5% to verifiers
    
    def calculate_rewards(self, total_fee: float, performance_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate individual rewards based on performance"""
        rewards = {}
        
        # Calculate miner rewards based on performance
        miner_pool = total_fee * self.miner_percentage
        total_performance = sum(performance_scores.values()) if performance_scores else 1.0
        
        for miner_id, score in performance_scores.items():
            if total_performance > 0:
                rewards[miner_id] = miner_pool * (score / total_performance)
            else:
                rewards[miner_id] = 0.0
        
        return rewards


class EconomicSystem:
    """Manages the overall economic system of PoUW"""
    
    def __init__(self):
        self.stake_pool = StakePool()
        self.task_matcher = TaskMatcher(self.stake_pool)
        self.reward_scheme = RewardScheme()
        self.active_tasks: Dict[str, Dict] = {}
        self.completed_tasks: Dict[str, Dict] = {}
        
    def buy_ticket(self, owner_id: str, role: NodeRole, stake_amount: float,
                   preferences: Dict[str, Any]) -> Ticket:
        """Process ticket purchase"""
        
        current_price = self.stake_pool.calculate_ticket_price()
        if stake_amount < current_price:
            raise ValueError(f"Insufficient stake. Required: {current_price}, provided: {stake_amount}")
        
        ticket_id = hashlib.sha256(f"{owner_id}_{role.value}_{time.time()}".encode()).hexdigest()[:16]
        
        ticket = Ticket(
            ticket_id=ticket_id,
            owner_id=owner_id,
            role=role,
            stake_amount=stake_amount,
            preferences=preferences,
            expiration_time=int(time.time()) + 86400 * 30  # 30 days
        )
        
        self.stake_pool.add_ticket(ticket)
        return ticket
    
    def submit_task(self, task: MLTask) -> Dict[NodeRole, List[Ticket]]:
        """Submit ML task and assign workers"""
        
        # Clean up expired tickets
        self.stake_pool.remove_expired_tickets()
        
        # Select workers for task
        selected_workers = self.task_matcher.select_workers(task)
        
        # Mark tickets as assigned
        for role, tickets in selected_workers.items():
            for ticket in tickets:
                ticket.assigned_task = task.task_id
        
        # Track active task
        self.active_tasks[task.task_id] = {
            'task': task,
            'workers': selected_workers,
            'start_time': int(time.time()),
            'status': 'active'
        }
        
        return selected_workers
    
    def complete_task(self, task_id: str, final_models: Dict[str, Any], 
                     performance_metrics: Dict[str, float]) -> Dict[str, float]:
        """Complete task and distribute rewards"""
        
        if task_id not in self.active_tasks:
            raise ValueError(f"Task {task_id} not found in active tasks")
        
        task_info = self.active_tasks[task_id]
        task = task_info['task']
        workers = task_info['workers']
        
        # Calculate rewards
        rewards = self.reward_scheme.calculate_rewards(task.fee, performance_metrics)
        
        # Add fixed rewards for supervisors and evaluators
        supervisor_reward = task.fee * self.reward_scheme.supervisor_percentage
        evaluator_reward = task.fee * self.reward_scheme.evaluator_percentage
        
        # Distribute to supervisors
        num_supervisors = len(workers[NodeRole.SUPERVISOR])
        if num_supervisors > 0:
            supervisor_reward_each = supervisor_reward / num_supervisors
            for ticket in workers[NodeRole.SUPERVISOR]:
                rewards[ticket.owner_id] = supervisor_reward_each
        
        # Distribute to evaluators
        num_evaluators = len(workers[NodeRole.EVALUATOR])
        if num_evaluators > 0:
            evaluator_reward_each = evaluator_reward / num_evaluators
            for ticket in workers[NodeRole.EVALUATOR]:
                rewards[ticket.owner_id] = evaluator_reward_each
        
        # Release worker assignments
        for role, tickets in workers.items():
            for ticket in tickets:
                ticket.assigned_task = None
        
        # Move to completed tasks
        self.completed_tasks[task_id] = {
            **task_info,
            'completion_time': int(time.time()),
            'final_models': final_models,
            'performance_metrics': performance_metrics,
            'rewards': rewards,
            'status': 'completed'
        }
        
        del self.active_tasks[task_id]
        
        return rewards
    
    def punish_malicious_node(self, node_id: str, reason: str) -> float:
        """Punish malicious node by confiscating stake"""
        
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
    
    def get_node_reputation(self, node_id: str) -> Dict[str, Any]:
        """Get reputation metrics for a node"""
        
        # Count completed tasks
        tasks_completed = 0
        total_rewards = 0.0
        
        for task_info in self.completed_tasks.values():
            if node_id in task_info.get('rewards', {}):
                tasks_completed += 1
                total_rewards += task_info['rewards'][node_id]
        
        # Get current stake
        current_stake = 0.0
        for ticket in self.stake_pool.tickets.values():
            if ticket.owner_id == node_id:
                current_stake += ticket.stake_amount
        
        return {
            'node_id': node_id,
            'tasks_completed': tasks_completed,
            'total_rewards': total_rewards,
            'current_stake': current_stake,
            'average_reward': total_rewards / tasks_completed if tasks_completed > 0 else 0.0
        }
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get overall network statistics"""
        
        role_counts = {}
        for role in NodeRole:
            role_counts[role.value] = len(self.stake_pool.get_tickets_by_role(role))
        
        return {
            'total_tickets': len(self.stake_pool.live_tickets),
            'role_distribution': role_counts,
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'current_ticket_price': self.stake_pool.calculate_ticket_price()
        }
