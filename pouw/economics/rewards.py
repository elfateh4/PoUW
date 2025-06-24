"""
Reward Distribution System for PoUW.

This module handles the distribution of rewards to network participants
based on their performance and contribution to task completion.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass


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
    
    def calculate_fixed_rewards(self, total_fee: float, participants: Dict[str, List[str]]) -> Dict[str, float]:
        """Calculate fixed rewards for supervisors, evaluators, and verifiers"""
        rewards = {}
        
        # Calculate fixed rewards for each role
        supervisor_reward = total_fee * self.supervisor_percentage
        evaluator_reward = total_fee * self.evaluator_percentage
        verifier_reward = total_fee * self.verifier_percentage
        
        # Distribute to supervisors
        supervisors = participants.get('supervisors', [])
        if supervisors:
            supervisor_reward_each = supervisor_reward / len(supervisors)
            for supervisor_id in supervisors:
                rewards[supervisor_id] = supervisor_reward_each
        
        # Distribute to evaluators
        evaluators = participants.get('evaluators', [])
        if evaluators:
            evaluator_reward_each = evaluator_reward / len(evaluators)
            for evaluator_id in evaluators:
                rewards[evaluator_id] = evaluator_reward_each
        
        # Distribute to verifiers
        verifiers = participants.get('verifiers', [])
        if verifiers:
            verifier_reward_each = verifier_reward / len(verifiers)
            for verifier_id in verifiers:
                rewards[verifier_id] = verifier_reward_each
        
        return rewards


class RewardDistributor:
    """Manages reward distribution for completed tasks"""
    
    def __init__(self, reward_scheme: Optional[RewardScheme] = None):
        self.reward_scheme = reward_scheme or RewardScheme()
        self.reward_history: List[Dict[str, Any]] = []
    
    def distribute_task_rewards(self, 
                               task_id: str,
                               task_fee: float,
                               miners: List[str],
                               supervisors: List[str],
                               evaluators: List[str], 
                               verifiers: List[str],
                               performance_metrics: Dict[str, float]) -> Dict[str, float]:
        """Distribute rewards for a completed task"""
        
        all_rewards = {}
        
        # Calculate performance-based rewards for miners
        miner_rewards = self.reward_scheme.calculate_rewards(task_fee, performance_metrics)
        all_rewards.update(miner_rewards)
        
        # Calculate fixed rewards for other roles
        participants = {
            'supervisors': supervisors,
            'evaluators': evaluators,
            'verifiers': verifiers
        }
        fixed_rewards = self.reward_scheme.calculate_fixed_rewards(task_fee, participants)
        all_rewards.update(fixed_rewards)
        
        # Record reward distribution
        self.reward_history.append({
            'task_id': task_id,
            'total_fee': task_fee,
            'rewards': all_rewards.copy(),
            'timestamp': int(__import__('time').time())
        })
        
        return all_rewards
    
    def get_node_earnings(self, node_id: str) -> Dict[str, Any]:
        """Get earnings summary for a specific node"""
        
        total_earnings = 0.0
        task_count = 0
        
        for record in self.reward_history:
            if node_id in record['rewards']:
                total_earnings += record['rewards'][node_id]
                task_count += 1
        
        return {
            'node_id': node_id,
            'total_earnings': total_earnings,
            'tasks_completed': task_count,
            'average_earnings': total_earnings / task_count if task_count > 0 else 0.0
        }
    
    def get_reward_statistics(self) -> Dict[str, Any]:
        """Get overall reward distribution statistics"""
        
        if not self.reward_history:
            return {
                'total_rewards_distributed': 0.0,
                'total_tasks': 0,
                'average_task_fee': 0.0,
                'top_earners': []
            }
        
        total_distributed = sum(sum(record['rewards'].values()) for record in self.reward_history)
        total_tasks = len(self.reward_history)
        average_fee = sum(record['total_fee'] for record in self.reward_history) / total_tasks
        
        # Calculate top earners
        node_earnings = {}
        for record in self.reward_history:
            for node_id, reward in record['rewards'].items():
                node_earnings[node_id] = node_earnings.get(node_id, 0.0) + reward
        
        top_earners = sorted(node_earnings.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_rewards_distributed': total_distributed,
            'total_tasks': total_tasks,
            'average_task_fee': average_fee,
            'top_earners': top_earners
        }
