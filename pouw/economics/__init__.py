"""
Refactored Economics Module for PoUW.

This package provides a clear, modular economic system with focused components:
- Staking system for network participation
- Dynamic pricing based on market conditions
- Reward distribution and incentive mechanisms
- Task-worker matching algorithms
- Profitability analysis and network economics
"""

# Core staking system
from .staking import NodeRole, Ticket, StakePool, StakingManager

# Task assignment
from .task_matching import TaskMatcher

# Reward systems
from .rewards import RewardScheme, RewardDistributor

# Pricing and market
from .pricing import DynamicPricingEngine, MarketCondition, MarketMetrics

# Main economic system coordinator
from .economic_system import EconomicSystem

__all__ = [
    # Staking system
    "NodeRole",
    "Ticket",
    "StakePool",
    "StakingManager",
    # Task matching
    "TaskMatcher",
    # Rewards
    "RewardScheme",
    "RewardDistributor",
    # Market and pricing
    "DynamicPricingEngine",
    "MarketCondition",
    "MarketMetrics",
    # Main economic system
    "EconomicSystem",
]
