"""
Enhanced Economic Model for PoUW with Dynamic Pricing and Advanced Incentives

This module implements advanced economic features including:
- Dynamic pricing algorithms based on network load and demand
- Real-world economic incentive mechanisms
- Advanced reward distribution schemes considering multiple factors
- Market-driven economic optimization
- Performance-based incentive structures
"""

import time
import math
import statistics
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

from ..economics.system import NodeRole, Ticket
from ..blockchain.core import MLTask


class MarketCondition(Enum):
    """Market condition states"""
    OVERSUPPLY = "oversupply"
    BALANCED = "balanced"
    UNDERSUPPLY = "undersupply"
    HIGH_DEMAND = "high_demand"


class IncentiveType(Enum):
    """Types of economic incentives"""
    PERFORMANCE_BONUS = "performance_bonus"
    RELIABILITY_REWARD = "reliability_reward"
    EARLY_ADOPTER_BONUS = "early_adopter_bonus"
    LONG_TERM_LOYALTY = "long_term_loyalty"
    NETWORK_EFFECT_BONUS = "network_effect_bonus"
    QUALITY_PREMIUM = "quality_premium"


@dataclass
class MarketMetrics:
    """Market metrics for dynamic pricing"""
    total_supply: int = 0
    total_demand: int = 0
    average_task_complexity: float = 0.0
    network_utilization: float = 0.0
    peak_hour_multiplier: float = 1.0
    quality_score: float = 0.0
    completion_rate: float = 0.0
    timestamp: int = field(default_factory=lambda: int(time.time()))


@dataclass
class PerformanceMetrics:
    """Node performance metrics for advanced rewards"""
    node_id: str
    accuracy_score: float = 0.0
    speed_score: float = 0.0
    reliability_score: float = 0.0
    availability_score: float = 0.0
    quality_score: float = 0.0
    consistency_score: float = 0.0
    innovation_score: float = 0.0  # For novel approaches
    collaboration_score: float = 0.0  # For helping other nodes
    history_length: int = 0
    last_updated: int = field(default_factory=lambda: int(time.time()))
    
    def overall_score(self) -> float:
        """Calculate weighted overall performance score"""
        weights = {
            'accuracy': 0.25,
            'speed': 0.20,
            'reliability': 0.20,
            'availability': 0.15,
            'quality': 0.10,
            'consistency': 0.05,
            'innovation': 0.03,
            'collaboration': 0.02
        }
        
        return (
            self.accuracy_score * weights['accuracy'] +
            self.speed_score * weights['speed'] +
            self.reliability_score * weights['reliability'] +
            self.availability_score * weights['availability'] +
            self.quality_score * weights['quality'] +
            self.consistency_score * weights['consistency'] +
            self.innovation_score * weights['innovation'] +
            self.collaboration_score * weights['collaboration']
        )


@dataclass
class EconomicIncentive:
    """Economic incentive definition"""
    incentive_type: IncentiveType
    base_multiplier: float
    performance_threshold: float
    max_bonus_percentage: float
    eligibility_requirements: Dict[str, Any]
    description: str


class DynamicPricingEngine:
    """Advanced dynamic pricing engine for task fees and rewards"""
    
    def __init__(self, base_price: float = 10.0):
        self.base_price = base_price
        self.price_history: deque = deque(maxlen=100)  # Last 100 price points
        self.market_history: deque = deque(maxlen=50)  # Last 50 market states
        
        # Pricing parameters
        self.demand_elasticity = 1.5
        self.supply_elasticity = 1.2
        self.complexity_factor = 2.0
        self.network_effect_factor = 1.8
        self.quality_premium_factor = 1.4
        
        # Market thresholds (supply/demand ratios)
        self.oversupply_threshold = 1.5  # Supply > 1.5x demand = oversupply
        self.undersupply_threshold = 0.7  # Supply < 0.7x demand = undersupply
        self.high_demand_threshold = 2.5
    
    def calculate_dynamic_price(self, 
                              task: MLTask,
                              market_metrics: MarketMetrics,
                              recent_completion_rate: float = 0.95) -> float:
        """Calculate dynamic price based on multiple market factors"""
        
        # Base price
        price = self.base_price
        
        # Supply/demand adjustment
        supply_demand_ratio = market_metrics.total_supply / max(market_metrics.total_demand, 1)
        
        if supply_demand_ratio < self.undersupply_threshold:
            # High demand, increase price
            demand_multiplier = 1.0 + (self.undersupply_threshold - supply_demand_ratio) * self.demand_elasticity * 0.5
            price *= demand_multiplier
        elif supply_demand_ratio > self.oversupply_threshold:
            # Oversupply, decrease price
            supply_multiplier = 1.0 - min(0.3, (supply_demand_ratio - self.oversupply_threshold) * self.supply_elasticity * 0.3)
            price *= supply_multiplier
        
        # Task complexity adjustment (reduced multiplier)
        complexity_multiplier = 1.0 + (task.complexity_score - 0.5) * 0.5  # Reduced from complexity_factor (2.0)
        price *= max(0.8, complexity_multiplier)  # Minimum 80% of base
        
        # Network utilization adjustment (reduced impact)
        utilization_multiplier = 1.0 + (market_metrics.network_utilization - 0.5) * 0.3  # Reduced from 0.8
        price *= max(0.9, utilization_multiplier)
        
        # Peak hour adjustment
        price *= market_metrics.peak_hour_multiplier
        
        # Quality premium (reduced impact)
        if market_metrics.quality_score > 0.8:
            quality_multiplier = 1.0 + (market_metrics.quality_score - 0.8) * 0.5  # Reduced from quality_premium_factor (1.4)
            price *= quality_multiplier
        
        # Completion rate adjustment (reduced impact)
        if recent_completion_rate < 0.9:
            # Low completion rate increases urgency premium
            urgency_multiplier = 1.0 + (0.9 - recent_completion_rate) * 1.0  # Reduced from 2.0
            price *= urgency_multiplier
        
        # Network effect bonus (reduced impact)
        network_size = market_metrics.total_supply + market_metrics.total_demand
        if network_size > 100:
            network_bonus = 1.0 + math.log10(network_size / 100) * 0.2  # Reduced from network_effect_factor (1.8)
            price *= network_bonus
        
        # Store price history
        self.price_history.append(price)
        
        return max(self.base_price * 0.1, price)  # Minimum 10% of base price
    
    def get_market_condition(self, market_metrics: MarketMetrics) -> MarketCondition:
        """Determine current market condition"""
        supply_demand_ratio = market_metrics.total_supply / max(market_metrics.total_demand, 1)
        
        # Check supply/demand ratio first for more precise categorization
        if supply_demand_ratio > self.oversupply_threshold:
            return MarketCondition.OVERSUPPLY
        elif supply_demand_ratio < self.undersupply_threshold:
            return MarketCondition.UNDERSUPPLY
        elif market_metrics.total_demand > self.high_demand_threshold * market_metrics.total_supply:
            return MarketCondition.HIGH_DEMAND
        else:
            return MarketCondition.BALANCED
    
    def predict_optimal_price(self, future_demand_estimate: int, future_supply_estimate: int) -> float:
        """Predict optimal price for future market conditions"""
        if not self.price_history:
            return self.base_price
        
        # Use recent price trend
        recent_prices = list(self.price_history)[-10:]
        price_trend = statistics.mean(recent_prices) if recent_prices else self.base_price
        
        # Adjust for predicted market conditions
        predicted_ratio = future_supply_estimate / max(future_demand_estimate, 1)
        
        if predicted_ratio < self.undersupply_threshold:
            return price_trend * 1.3  # Anticipate higher prices
        elif predicted_ratio > self.oversupply_threshold:
            return price_trend * 0.8  # Anticipate lower prices
        else:
            return price_trend


class AdvancedRewardDistributor:
    """Advanced reward distribution with performance-based incentives"""
    
    def __init__(self):
        self.performance_metrics: Dict[str, PerformanceMetrics] = {}
        self.incentive_pool = 0.0
        self.bonus_pool_percentage = 0.15  # 15% of task fees go to bonus pool
        
        # Define economic incentives
        self.incentives = {
            IncentiveType.PERFORMANCE_BONUS: EconomicIncentive(
                incentive_type=IncentiveType.PERFORMANCE_BONUS,
                base_multiplier=1.5,
                performance_threshold=0.85,
                max_bonus_percentage=0.3,
                eligibility_requirements={'min_tasks': 5, 'min_accuracy': 0.8},
                description="Bonus for consistently high performance"
            ),
            IncentiveType.RELIABILITY_REWARD: EconomicIncentive(
                incentive_type=IncentiveType.RELIABILITY_REWARD,
                base_multiplier=1.3,
                performance_threshold=0.95,
                max_bonus_percentage=0.2,
                eligibility_requirements={'min_tasks': 10, 'min_availability': 0.9},
                description="Reward for high reliability and availability"
            ),
            IncentiveType.EARLY_ADOPTER_BONUS: EconomicIncentive(
                incentive_type=IncentiveType.EARLY_ADOPTER_BONUS,
                base_multiplier=2.0,
                performance_threshold=0.7,
                max_bonus_percentage=0.5,
                eligibility_requirements={'max_join_time': 30 * 24 * 3600},  # 30 days
                description="Bonus for early network participants"
            ),
            IncentiveType.LONG_TERM_LOYALTY: EconomicIncentive(
                incentive_type=IncentiveType.LONG_TERM_LOYALTY,
                base_multiplier=1.4,
                performance_threshold=0.8,
                max_bonus_percentage=0.25,
                eligibility_requirements={'min_participation_days': 180, 'min_tasks': 50},
                description="Loyalty bonus for long-term participants"
            ),
            IncentiveType.NETWORK_EFFECT_BONUS: EconomicIncentive(
                incentive_type=IncentiveType.NETWORK_EFFECT_BONUS,
                base_multiplier=1.2,
                performance_threshold=0.75,
                max_bonus_percentage=0.15,
                eligibility_requirements={'min_referrals': 3, 'min_collaboration_score': 0.7},
                description="Bonus for contributing to network growth"
            ),
            IncentiveType.QUALITY_PREMIUM: EconomicIncentive(
                incentive_type=IncentiveType.QUALITY_PREMIUM,
                base_multiplier=1.6,
                performance_threshold=0.9,
                max_bonus_percentage=0.4,
                eligibility_requirements={'min_quality_score': 0.85, 'min_innovation_score': 0.7},
                description="Premium for exceptional quality and innovation"
            )
        }
    
    def update_performance_metrics(self, node_id: str, task_metrics: Dict[str, float]):
        """Update performance metrics for a node"""
        if node_id not in self.performance_metrics:
            self.performance_metrics[node_id] = PerformanceMetrics(node_id=node_id)
        
        metrics = self.performance_metrics[node_id]
        
        # Update metrics with exponential moving average
        alpha = 0.3  # Learning rate
        
        if 'accuracy' in task_metrics:
            metrics.accuracy_score = alpha * task_metrics['accuracy'] + (1 - alpha) * metrics.accuracy_score
        
        if 'speed' in task_metrics:
            metrics.speed_score = alpha * task_metrics['speed'] + (1 - alpha) * metrics.speed_score
        
        if 'reliability' in task_metrics:
            metrics.reliability_score = alpha * task_metrics['reliability'] + (1 - alpha) * metrics.reliability_score
        
        if 'availability' in task_metrics:
            metrics.availability_score = alpha * task_metrics['availability'] + (1 - alpha) * metrics.availability_score
        
        if 'quality' in task_metrics:
            metrics.quality_score = alpha * task_metrics['quality'] + (1 - alpha) * metrics.quality_score
        
        # Update consistency score based on variance in performance
        metrics.consistency_score = max(0, 1.0 - statistics.stdev([
            metrics.accuracy_score, metrics.speed_score, metrics.reliability_score
        ]) if metrics.history_length > 3 else 0.8)
        
        metrics.history_length += 1
        metrics.last_updated = int(time.time())
    
    def calculate_advanced_rewards(self,
                                 total_fee: float,
                                 participants: Dict[NodeRole, List[str]],
                                 performance_data: Dict[str, Dict[str, float]],
                                 market_condition: MarketCondition) -> Dict[str, float]:
        """Calculate advanced reward distribution with incentives"""
        
        rewards = {}
        
        # Add to incentive pool
        incentive_pool_addition = total_fee * self.bonus_pool_percentage
        self.incentive_pool += incentive_pool_addition
        
        # Base distribution (reduced to account for incentive pool)
        base_fee = total_fee * (1 - self.bonus_pool_percentage)
        
        # Base percentages by role
        role_percentages = {
            NodeRole.MINER: 0.60,
            NodeRole.SUPERVISOR: 0.25,
            NodeRole.EVALUATOR: 0.15
        }
        
        # Market condition adjustments
        if market_condition == MarketCondition.HIGH_DEMAND:
            role_percentages[NodeRole.MINER] *= 1.2  # Increase miner rewards in high demand
        elif market_condition == MarketCondition.OVERSUPPLY:
            role_percentages[NodeRole.SUPERVISOR] *= 1.1  # Reward supervisors more during oversupply
        
        # Distribute base rewards
        for role, node_ids in participants.items():
            if not node_ids:
                continue
            
            role_total = base_fee * role_percentages.get(role, 0.0)
            
            if role == NodeRole.MINER:
                # Performance-based distribution for miners
                miner_rewards = self._distribute_performance_based_rewards(
                    node_ids, role_total, performance_data
                )
                rewards.update(miner_rewards)
            else:
                # Equal distribution for supervisors and evaluators
                equal_share = role_total / len(node_ids)
                for node_id in node_ids:
                    rewards[node_id] = equal_share
        
        # Add incentive bonuses
        incentive_rewards = self._calculate_incentive_bonuses(participants, performance_data)
        
        # Merge incentive rewards
        for node_id, incentive_amount in incentive_rewards.items():
            if node_id in rewards:
                rewards[node_id] += incentive_amount
            else:
                rewards[node_id] = incentive_amount
        
        # Deduct incentive bonuses from pool
        total_incentives = sum(incentive_rewards.values())
        self.incentive_pool = max(0, self.incentive_pool - total_incentives)
        
        return rewards
    
    def _distribute_performance_based_rewards(self,
                                            node_ids: List[str],
                                            total_amount: float,
                                            performance_data: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Distribute rewards based on performance scores"""
        
        if not node_ids:
            return {}
        
        # Update performance metrics
        for node_id in node_ids:
            if node_id in performance_data:
                self.update_performance_metrics(node_id, performance_data[node_id])
        
        # Calculate performance scores
        scores = {}
        for node_id in node_ids:
            if node_id in self.performance_metrics:
                scores[node_id] = self.performance_metrics[node_id].overall_score()
            else:
                scores[node_id] = 0.5  # Default score for new nodes
        
        # Normalize scores to ensure positive rewards
        min_score = min(scores.values())
        if min_score < 0:
            for node_id in scores:
                scores[node_id] -= min_score
        
        total_score = sum(scores.values())
        if total_score == 0:
            # Equal distribution if no performance data
            equal_share = total_amount / len(node_ids)
            return {node_id: equal_share for node_id in node_ids}
        
        # Distribute proportionally to performance
        rewards = {}
        for node_id, score in scores.items():
            rewards[node_id] = total_amount * (score / total_score)
        
        return rewards
    
    def _calculate_incentive_bonuses(self,
                                   participants: Dict[NodeRole, List[str]],
                                   performance_data: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate incentive bonuses for eligible participants"""
        
        bonuses = {}
        available_pool = self.incentive_pool
        
        all_participants = []
        for node_list in participants.values():
            all_participants.extend(node_list)
        
        for node_id in all_participants:
            if node_id not in self.performance_metrics:
                continue
            
            metrics = self.performance_metrics[node_id]
            node_bonuses = 0.0
            
            # Check each incentive type
            for incentive in self.incentives.values():
                if self._is_eligible_for_incentive(node_id, incentive, performance_data.get(node_id, {})):
                    # Calculate bonus amount
                    base_amount = available_pool * 0.1  # Use 10% of pool per incentive
                    performance_multiplier = min(
                        incentive.base_multiplier,
                        metrics.overall_score() * incentive.base_multiplier
                    )
                    
                    bonus = base_amount * performance_multiplier
                    bonus = min(bonus, available_pool * incentive.max_bonus_percentage)
                    
                    node_bonuses += bonus
                    available_pool -= bonus
            
            if node_bonuses > 0:
                bonuses[node_id] = node_bonuses
        
        return bonuses
    
    def _is_eligible_for_incentive(self,
                                 node_id: str,
                                 incentive: EconomicIncentive,
                                 current_performance: Dict[str, float]) -> bool:
        """Check if node is eligible for specific incentive"""
        
        if node_id not in self.performance_metrics:
            return False
        
        metrics = self.performance_metrics[node_id]
        
        # Check performance threshold
        if metrics.overall_score() < incentive.performance_threshold:
            return False
        
        # Check specific requirements
        requirements = incentive.eligibility_requirements
        
        if 'min_tasks' in requirements and metrics.history_length < requirements['min_tasks']:
            return False
        
        if 'min_accuracy' in requirements and metrics.accuracy_score < requirements['min_accuracy']:
            return False
        
        if 'min_availability' in requirements and metrics.availability_score < requirements['min_availability']:
            return False
        
        if 'min_quality_score' in requirements and metrics.quality_score < requirements['min_quality_score']:
            return False
        
        if 'min_innovation_score' in requirements and metrics.innovation_score < requirements['min_innovation_score']:
            return False
        
        if 'min_collaboration_score' in requirements and metrics.collaboration_score < requirements['min_collaboration_score']:
            return False
        
        # Time-based requirements would need additional tracking in practice
        
        return True
    
    def get_incentive_summary(self, node_id: str) -> Dict[str, Any]:
        """Get summary of available incentives for a node"""
        
        if node_id not in self.performance_metrics:
            return {'eligible_incentives': [], 'potential_bonuses': 0.0}
        
        eligible = []
        potential_total = 0.0
        
        for incentive_type, incentive in self.incentives.items():
            if self._is_eligible_for_incentive(node_id, incentive, {}):
                eligible.append({
                    'type': incentive_type.value,
                    'description': incentive.description,
                    'max_bonus_percentage': incentive.max_bonus_percentage
                })
                potential_total += incentive.max_bonus_percentage
        
        return {
            'eligible_incentives': eligible,
            'potential_bonus_percentage': min(potential_total, 1.0),
            'current_performance_score': self.performance_metrics[node_id].overall_score(),
            'incentive_pool_available': self.incentive_pool
        }


class EnhancedEconomicSystem:
    """Enhanced economic system with dynamic pricing and advanced incentives"""
    
    def __init__(self, base_price: float = 10.0):
        self.pricing_engine = DynamicPricingEngine(base_price)
        self.reward_distributor = AdvancedRewardDistributor()
        self.market_metrics = MarketMetrics()
        
        # Economic tracking
        self.price_history: List[Tuple[int, float]] = []
        self.market_events: List[Dict[str, Any]] = []
        self.roi_analytics: Dict[str, Any] = {}
        
    def update_market_metrics(self,
                            total_supply: int,
                            total_demand: int,
                            recent_tasks: List[MLTask],
                            network_stats: Dict[str, Any]):
        """Update market metrics for pricing calculations"""
        
        self.market_metrics.total_supply = total_supply
        self.market_metrics.total_demand = total_demand
        
        # Calculate average task complexity
        if recent_tasks:
            complexities = [getattr(task, 'complexity_score', 0.5) for task in recent_tasks]
            self.market_metrics.average_task_complexity = statistics.mean(complexities)
        
        # Update network utilization
        if 'active_nodes' in network_stats and 'total_nodes' in network_stats:
            self.market_metrics.network_utilization = (
                network_stats['active_nodes'] / max(network_stats['total_nodes'], 1)
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
        if 'completion_rate' in network_stats:
            self.market_metrics.completion_rate = network_stats['completion_rate']
        
        if 'average_quality' in network_stats:
            self.market_metrics.quality_score = network_stats['average_quality']
    
    def calculate_optimal_task_fee(self, task: MLTask) -> float:
        """Calculate optimal fee for a task"""
        base_fee = self.pricing_engine.calculate_dynamic_price(task, self.market_metrics)
        
        # Store pricing event
        self.price_history.append((int(time.time()), base_fee))
        
        return base_fee
    
    def distribute_task_rewards(self,
                              task: MLTask,
                              participants: Dict[NodeRole, List[str]],
                              performance_data: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Distribute rewards with advanced incentive system"""
        
        market_condition = self.pricing_engine.get_market_condition(self.market_metrics)
        
        rewards = self.reward_distributor.calculate_advanced_rewards(
            task.fee,
            participants,
            performance_data,
            market_condition
        )
        
        # Log market event
        self.market_events.append({
            'timestamp': int(time.time()),
            'event_type': 'task_completion',
            'task_id': task.task_id,
            'fee': task.fee,
            'market_condition': market_condition.value,
            'participants': len(sum(participants.values(), [])),
            'total_rewards': sum(rewards.values())
        })
        
        return rewards
    
    def get_economic_analytics(self) -> Dict[str, Any]:
        """Get comprehensive economic analytics"""
        
        # Price analytics
        recent_prices = [price for _, price in self.price_history[-50:]]
        price_analytics = {
            'current_price': recent_prices[-1] if recent_prices else 0.0,
            'average_price': statistics.mean(recent_prices) if recent_prices else 0.0,
            'price_volatility': statistics.stdev(recent_prices) if len(recent_prices) > 1 else 0.0,
            'price_trend': 'increasing' if len(recent_prices) >= 2 and recent_prices[-1] > recent_prices[-2] else 'stable'
        }
        
        # Market analytics
        market_condition = self.pricing_engine.get_market_condition(self.market_metrics)
        
        return {
            'price_analytics': price_analytics,
            'market_condition': market_condition.value,
            'market_metrics': {
                'supply_demand_ratio': self.market_metrics.total_supply / max(self.market_metrics.total_demand, 1),
                'network_utilization': self.market_metrics.network_utilization,
                'average_complexity': self.market_metrics.average_task_complexity,
                'quality_score': self.market_metrics.quality_score
            },
            'incentive_pool': self.reward_distributor.incentive_pool,
            'total_market_events': len(self.market_events),
            'economic_health': self._assess_economic_health()
        }
    
    def _assess_economic_health(self) -> str:
        """Assess overall economic health of the network"""
        
        # Factors: price stability, market balance, completion rate, quality
        health_score = 0.0
        
        # Price stability (lower volatility is better)
        recent_prices = [price for _, price in self.price_history[-20:]]
        if len(recent_prices) > 1:
            volatility = statistics.stdev(recent_prices)
            price_stability = max(0, 1.0 - volatility / statistics.mean(recent_prices))
            health_score += price_stability * 0.3
        
        # Market balance
        supply_demand_ratio = self.market_metrics.total_supply / max(self.market_metrics.total_demand, 1)
        balance_score = 1.0 - abs(supply_demand_ratio - 1.0)  # Closer to 1.0 is better
        health_score += max(0, balance_score) * 0.3
        
        # Network utilization
        health_score += self.market_metrics.network_utilization * 0.2
        
        # Quality and completion
        health_score += self.market_metrics.quality_score * 0.1
        health_score += self.market_metrics.completion_rate * 0.1
        
        if health_score >= 0.8:
            return "excellent"
        elif health_score >= 0.6:
            return "good"
        elif health_score >= 0.4:
            return "fair"
        else:
            return "poor"
