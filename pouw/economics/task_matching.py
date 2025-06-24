"""
Task-Worker Matching System for PoUW.

This module handles the assignment of workers to ML tasks using
compatibility scoring and VRF-based selection algorithms.
"""

import random
from typing import Dict, List, Optional
from .staking import StakePool, NodeRole, Ticket


class TaskMatcher:
    """Matches worker nodes to ML tasks using compatibility and VRF-based selection"""
    
    def __init__(self, stake_pool: StakePool):
        self.stake_pool = stake_pool
    
    def select_workers(self, task) -> Dict[NodeRole, List[Ticket]]:
        """Select workers for an ML task based on requirements and staking"""
        
        # Clean up expired tickets first
        self.stake_pool.remove_expired_tickets()
        
        selected_workers = {}
        
        # Worker requirements for different roles
        requirements = {
            NodeRole.MINER: task.get_required_miners(),
            NodeRole.SUPERVISOR: 1,  # One supervisor per task
            NodeRole.EVALUATOR: 1,   # One evaluator per task  
            NodeRole.VERIFIER: 1     # One verifier per task
        }
        
        for role, count in requirements.items():
            available_tickets = self.stake_pool.get_tickets_by_role(role)
            
            # Filter available tickets (not currently assigned)
            free_tickets = [t for t in available_tickets if t.assigned_task is None]
            
            if len(free_tickets) < count:
                print(f"Warning: Not enough {role.value}s available. Need {count}, have {len(free_tickets)}")
                selected_workers[role] = free_tickets  # Take what we can get
            else:
                # Use compatibility-based selection for miners, random for others
                if role == NodeRole.MINER:
                    selected_workers[role] = self._select_best_miners(task, free_tickets, count)
                else:
                    selected_workers[role] = self._select_random_workers(free_tickets, count)
        
        return selected_workers
    
    def _select_best_miners(self, task, available_miners: List[Ticket], count: int) -> List[Ticket]:
        """Select miners based on task compatibility and stake amount"""
        
        # Score each miner based on task compatibility and stake
        miner_scores = []
        for miner in available_miners:
            compatibility_score = miner.matches_task(task)
            stake_score = min(miner.stake_amount / 1000.0, 1.0)  # Normalize stake
            
            # Combined score (70% compatibility, 30% stake)
            final_score = compatibility_score * 0.7 + stake_score * 0.3
            miner_scores.append((miner, final_score))
        
        # Sort by score and select top miners
        miner_scores.sort(key=lambda x: x[1], reverse=True)
        return [miner for miner, score in miner_scores[:count]]
    
    def _select_random_workers(self, available_workers: List[Ticket], count: int) -> List[Ticket]:
        """Select workers randomly (for non-miner roles)"""
        
        # Use weighted selection based on stake amount
        if not available_workers:
            return []
        
        weights = [ticket.stake_amount for ticket in available_workers]
        
        selected = []
        remaining_workers = available_workers.copy()
        remaining_weights = weights.copy()
        
        for _ in range(min(count, len(remaining_workers))):
            # Weighted random selection
            total_weight = sum(remaining_weights)
            if total_weight == 0:
                selected.append(remaining_workers.pop(0))
                remaining_weights.pop(0)
            else:
                random_value = random.uniform(0, total_weight)
                cumulative_weight = 0
                
                for i, weight in enumerate(remaining_weights):
                    cumulative_weight += weight
                    if random_value <= cumulative_weight:
                        selected.append(remaining_workers.pop(i))
                        remaining_weights.pop(i)
                        break
        
        return selected
    
    def release_workers(self, task_id: str):
        """Release workers assigned to a completed task"""
        
        for ticket in self.stake_pool.tickets.values():
            if ticket.assigned_task == task_id:
                ticket.assigned_task = None
    
    def get_assignment_statistics(self) -> Dict[str, int]:
        """Get statistics about current worker assignments"""
        
        stats = {
            'total_tickets': len(self.stake_pool.tickets),
            'live_tickets': len(self.stake_pool.live_tickets),
            'assigned_tickets': 0,
            'free_tickets': 0
        }
        
        role_assignments = {}
        for role in NodeRole:
            role_assignments[f"{role.value}_assigned"] = 0
            role_assignments[f"{role.value}_free"] = 0
        
        for ticket in self.stake_pool.tickets.values():
            if ticket.is_live:
                if ticket.assigned_task:
                    stats['assigned_tickets'] += 1
                    role_assignments[f"{ticket.role.value}_assigned"] += 1
                else:
                    stats['free_tickets'] += 1
                    role_assignments[f"{ticket.role.value}_free"] += 1
        
        stats.update(role_assignments)
        return stats
