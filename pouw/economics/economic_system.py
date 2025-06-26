"""
Main Economic System Coordinator for PoUW.

This module provides the main EconomicSystem class that coordinates
all economic components and provides a unified interface.
"""

import time
from typing import Dict, List, Optional, Any
from .staking import StakingManager, NodeRole
from .task_matching import TaskMatcher
from .rewards import RewardDistributor, RewardScheme
from .pricing import DynamicPricingEngine, MarketMetrics


class EconomicSystem:
    """Main economic system coordinator that integrates all economic components"""

    def __init__(self, base_price: float = 10.0, max_token_supply: float = 21_000_000.0, 
                 genesis_supply: float = 1_000.0, base_block_reward: float = 50.0,
                 halving_interval: int = 210_000):
        self.staking_manager = StakingManager()
        self.task_matcher = TaskMatcher(self.staking_manager.stake_pool)
        self.reward_distributor = RewardDistributor()
        self.pricing_engine = DynamicPricingEngine(base_price)

        # Task tracking
        self.active_tasks: Dict[str, Dict] = {}
        self.completed_tasks: Dict[str, Dict] = {}

        # Market metrics
        self.market_metrics = MarketMetrics()
        
        # Token supply management
        self.max_token_supply = max_token_supply
        self.genesis_supply = genesis_supply
        self.base_block_reward = base_block_reward
        self.halving_interval = halving_interval
        self.current_supply = genesis_supply  # Start with genesis supply
        self.total_blocks_mined = 0  # Track blocks for halving calculation

    def buy_ticket(
        self, owner_id: str, role: NodeRole, stake_amount: float, preferences: Dict[str, Any]
    ):
        """Purchase a staking ticket"""
        return self.staking_manager.buy_ticket(owner_id, role, stake_amount, preferences)

    def submit_task(self, task):
        """Submit ML task and assign workers"""

        # Calculate dynamic task fee
        task_fee = self.pricing_engine.calculate_dynamic_price(task, self.market_metrics)
        task.fee = task_fee

        # Select workers for task
        selected_workers = self.task_matcher.select_workers(task)

        # Track active task
        self.active_tasks[task.task_id] = {
            "task": task,
            "workers": selected_workers,
            "start_time": int(time.time()),
            "status": "active",
        }

        return selected_workers

    def complete_task(
        self, task_id: str, final_models: Dict[str, Any], performance_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Complete task and distribute rewards"""

        if task_id not in self.active_tasks:
            raise ValueError(f"Task {task_id} not found in active tasks")

        task_info = self.active_tasks[task_id]
        task = task_info["task"]
        workers = task_info["workers"]

        # Prepare participant lists for reward distribution
        miners = [ticket.owner_id for ticket in workers.get(NodeRole.MINER, [])]
        supervisors = [ticket.owner_id for ticket in workers.get(NodeRole.SUPERVISOR, [])]
        evaluators = [ticket.owner_id for ticket in workers.get(NodeRole.EVALUATOR, [])]
        verifiers = [ticket.owner_id for ticket in workers.get(NodeRole.VERIFIER, [])]

        # Distribute rewards
        rewards = self.reward_distributor.distribute_task_rewards(
            task_id=task_id,
            task_fee=task.fee,
            miners=miners,
            supervisors=supervisors,
            evaluators=evaluators,
            verifiers=verifiers,
            performance_metrics=performance_metrics,
        )

        # Release worker assignments
        self.task_matcher.release_workers(task_id)

        # Move to completed tasks
        self.completed_tasks[task_id] = {
            **task_info,
            "completion_time": int(time.time()),
            "final_models": final_models,
            "performance_metrics": performance_metrics,
            "rewards": rewards,
            "status": "completed",
        }

        # Remove from active tasks
        del self.active_tasks[task_id]

        return rewards

    def update_market_metrics(
        self,
        total_supply: int,
        total_demand: int,
        recent_tasks: List,
        network_stats: Dict[str, Any],
    ):
        """Update market metrics for pricing calculations"""

        self.market_metrics.total_supply = total_supply
        self.market_metrics.total_demand = total_demand
        self.market_metrics.timestamp = int(time.time())

        # Calculate average task complexity
        if recent_tasks:
            complexities = [getattr(task, "complexity_score", 0.5) for task in recent_tasks]
            self.market_metrics.average_task_complexity = sum(complexities) / len(complexities)

        # Update network utilization
        if "active_nodes" in network_stats and "total_nodes" in network_stats:
            self.market_metrics.network_utilization = network_stats["active_nodes"] / max(
                network_stats["total_nodes"], 1
            )

        # Peak hour detection (simplified)
        current_hour = time.localtime().tm_hour
        if 9 <= current_hour <= 17:  # Business hours
            self.market_metrics.peak_hour_multiplier = 1.2
        elif 18 <= current_hour <= 22:  # Evening hours
            self.market_metrics.peak_hour_multiplier = 1.1
        else:
            self.market_metrics.peak_hour_multiplier = 0.9

        # Update completion rate and quality
        if "completion_rate" in network_stats:
            self.market_metrics.completion_rate = network_stats["completion_rate"]

        if "average_quality" in network_stats:
            self.market_metrics.quality_score = network_stats["average_quality"]

    def punish_malicious_node(self, node_id: str, reason: str) -> float:
        """Punish malicious node by confiscating stake"""
        return self.staking_manager.confiscate_stake(node_id, reason)

    def get_node_reputation(self, node_id: str) -> Dict[str, Any]:
        """Get reputation metrics for a node"""

        # Get earnings from reward distributor
        earnings = self.reward_distributor.get_node_earnings(node_id)

        # Get current stake info
        current_stake = 0.0
        ticket_count = 0
        for ticket in self.staking_manager.stake_pool.tickets.values():
            if ticket.owner_id == node_id:
                current_stake += ticket.stake_amount
                ticket_count += 1

        return {
            "node_id": node_id,
            "total_earnings": earnings["total_earnings"],
            "tasks_completed": earnings["tasks_completed"],
            "average_earnings": earnings["average_earnings"],
            "current_stake": current_stake,
            "active_tickets": ticket_count,
        }

    def get_network_stats(self) -> Dict[str, Any]:
        """Get overall network statistics"""

        # Get assignment statistics
        assignment_stats = self.task_matcher.get_assignment_statistics()

        # Get reward statistics
        reward_stats = self.reward_distributor.get_reward_statistics()

        # Get pricing analytics
        pricing_stats = self.pricing_engine.get_pricing_analytics()

        return {
            "assignment_stats": assignment_stats,
            "reward_stats": reward_stats,
            "pricing_stats": pricing_stats,
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "market_condition": self.pricing_engine.get_market_condition(self.market_metrics).value,
        }

    def get_economic_health_report(self) -> Dict[str, Any]:
        """Get comprehensive economic health report"""

        network_stats = self.get_network_stats()

        # Calculate health indicators
        health_indicators = {
            "market_balance": self._assess_market_balance(),
            "participant_activity": self._assess_participant_activity(),
            "reward_distribution": self._assess_reward_distribution(),
            "pricing_stability": self._assess_pricing_stability(),
        }

        # Overall health score (0-100)
        health_scores = list(health_indicators.values())
        overall_health = sum(health_scores) / len(health_scores) if health_scores else 0

        return {
            "overall_health_score": overall_health,
            "health_indicators": health_indicators,
            "network_stats": network_stats,
            "recommendations": self._generate_health_recommendations(health_indicators),
        }

    def _assess_market_balance(self) -> float:
        """Assess market supply/demand balance (0-100)"""
        if self.market_metrics.total_demand == 0:
            return 50.0  # Neutral when no demand

        ratio = self.market_metrics.total_supply / self.market_metrics.total_demand

        # Optimal ratio is around 1.0-1.2
        if 1.0 <= ratio <= 1.2:
            return 100.0
        elif 0.8 <= ratio < 1.0 or 1.2 < ratio <= 1.5:
            return 75.0
        elif 0.6 <= ratio < 0.8 or 1.5 < ratio <= 2.0:
            return 50.0
        else:
            return 25.0

    def _assess_participant_activity(self) -> float:
        """Assess participant activity level (0-100)"""
        assignment_stats = self.task_matcher.get_assignment_statistics()

        total_tickets = assignment_stats["live_tickets"]
        assigned_tickets = assignment_stats["assigned_tickets"]

        if total_tickets == 0:
            return 0.0

        utilization_rate = assigned_tickets / total_tickets
        return min(utilization_rate * 100, 100.0)

    def _assess_reward_distribution(self) -> float:
        """Assess fairness of reward distribution (0-100)"""
        reward_stats = self.reward_distributor.get_reward_statistics()

        if reward_stats["total_tasks"] == 0:
            return 100.0  # No tasks = no distribution issues

        # Simple assessment: more tasks completed = better distribution
        if reward_stats["total_tasks"] >= 10:
            return 100.0
        elif reward_stats["total_tasks"] >= 5:
            return 75.0
        elif reward_stats["total_tasks"] >= 1:
            return 50.0
        else:
            return 25.0

    def _assess_pricing_stability(self) -> float:
        """Assess pricing stability (0-100)"""
        pricing_stats = self.pricing_engine.get_pricing_analytics()

        volatility = pricing_stats["volatility"]
        current_price = pricing_stats["current_price"]

        if current_price == 0:
            return 50.0

        # Low volatility = high stability
        volatility_ratio = volatility / current_price

        if volatility_ratio <= 0.1:
            return 100.0
        elif volatility_ratio <= 0.2:
            return 75.0
        elif volatility_ratio <= 0.3:
            return 50.0
        else:
            return 25.0

    def _generate_health_recommendations(self, health_indicators: Dict[str, float]) -> List[str]:
        """Generate recommendations based on health indicators"""

        recommendations = []

        if health_indicators["market_balance"] < 50:
            recommendations.append("Market imbalance detected - consider adjusting incentives")

        if health_indicators["participant_activity"] < 50:
            recommendations.append(
                "Low participant activity - consider marketing or incentive improvements"
            )

        if health_indicators["pricing_stability"] < 50:
            recommendations.append("High price volatility - consider pricing algorithm adjustments")

        if health_indicators["reward_distribution"] < 50:
            recommendations.append(
                "Limited reward distribution activity - monitor task submission rates"
            )

        if not recommendations:
            recommendations.append("Economic system appears healthy - continue monitoring")

        return recommendations

    # Token supply management methods
    
    def calculate_block_reward(self, block_height: int) -> float:
        """Calculate block reward with halving and supply cap consideration"""
        # Check if we've reached maximum supply
        if self.current_supply >= self.max_token_supply:
            return 0.0  # No more tokens to mint
        
        # Calculate current reward based on halving schedule
        halvings = block_height // self.halving_interval
        current_reward = self.base_block_reward / (2 ** halvings)
        
        # Ensure we don't exceed max supply
        remaining_supply = self.max_token_supply - self.current_supply
        if current_reward > remaining_supply:
            current_reward = remaining_supply
        
        return max(current_reward, 0.0)
    
    def mint_tokens(self, amount: float, reason: str = "mining_reward") -> bool:
        """Mint new tokens if under supply cap"""
        if amount <= 0:
            return False
            
        if self.current_supply + amount > self.max_token_supply:
            return False  # Cannot exceed max supply
            
        self.current_supply += amount
        return True
    
    def get_token_supply_info(self) -> Dict[str, Any]:
        """Get current token supply information"""
        remaining_supply = self.max_token_supply - self.current_supply
        supply_percentage = (self.current_supply / self.max_token_supply) * 100
        
        # Calculate next halving
        next_halving_block = ((self.total_blocks_mined // self.halving_interval) + 1) * self.halving_interval
        blocks_until_halving = next_halving_block - self.total_blocks_mined
        
        # Calculate current block reward
        current_reward = self.calculate_block_reward(self.total_blocks_mined)
        
        return {
            "current_supply": self.current_supply,
            "max_supply": self.max_token_supply,
            "remaining_supply": remaining_supply,
            "supply_percentage": supply_percentage,
            "genesis_supply": self.genesis_supply,
            "total_blocks_mined": self.total_blocks_mined,
            "current_block_reward": current_reward,
            "base_block_reward": self.base_block_reward,
            "halving_interval": self.halving_interval,
            "blocks_until_halving": blocks_until_halving,
            "next_halving_block": next_halving_block,
            "supply_exhausted": self.current_supply >= self.max_token_supply
        }
    
    def record_mined_block(self, block_reward: float) -> bool:
        """Record a mined block and update token supply"""
        if not self.mint_tokens(block_reward, "block_mining"):
            return False
            
        self.total_blocks_mined += 1
        return True
    
    def get_circulating_supply(self) -> float:
        """Get the current circulating supply (excludes locked stakes)"""
        # Calculate total staked tokens
        total_staked = 0.0
        for ticket in self.staking_manager.stake_pool.tickets.values():
            if not ticket.is_expired():
                total_staked += ticket.stake_amount
        
        # Circulating supply = total supply - staked tokens
        return max(self.current_supply - total_staked, 0.0)
    
    def validate_transaction_amount(self, amount: float) -> bool:
        """Validate that a transaction amount doesn't exceed circulating supply"""
        circulating = self.get_circulating_supply()
        return amount <= circulating
    
    def get_supply_health_status(self) -> Dict[str, Any]:
        """Get token supply health indicators"""
        supply_info = self.get_token_supply_info()
        
        # Calculate health indicators
        supply_health = {
            "supply_exhaustion_risk": "HIGH" if supply_info["supply_percentage"] > 95 else 
                                     "MEDIUM" if supply_info["supply_percentage"] > 80 else "LOW",
            "inflation_rate": self.calculate_inflation_rate(),
            "reward_sustainability": "SUSTAINABLE" if supply_info["current_block_reward"] > 0.1 else 
                                   "DECLINING" if supply_info["current_block_reward"] > 0 else "EXHAUSTED",
            "circulating_ratio": (self.get_circulating_supply() / self.current_supply * 100) if self.current_supply > 0 else 0
        }
        
        return {**supply_info, **supply_health}
    
    def calculate_inflation_rate(self) -> float:
        """Calculate current annual inflation rate"""
        current_reward = self.calculate_block_reward(self.total_blocks_mined)
        
        # Assume ~1 block per minute (525,600 blocks per year)
        blocks_per_year = 525_600
        annual_new_tokens = current_reward * blocks_per_year
        
        if self.current_supply > 0:
            inflation_rate = (annual_new_tokens / self.current_supply) * 100
        else:
            inflation_rate = 0.0
            
        return inflation_rate
