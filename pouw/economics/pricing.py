"""
Dynamic Pricing Engine for PoUW.

This module implements market-driven pricing algorithms that adjust
task fees based on supply, demand, and network conditions.
"""

import time
import math
import statistics
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from collections import deque


class MarketCondition(Enum):
    """Market condition states for pricing adjustments"""

    OVERSUPPLY = "oversupply"
    BALANCED = "balanced"
    UNDERSUPPLY = "undersupply"
    HIGH_DEMAND = "high_demand"


@dataclass
class MarketMetrics:
    """Market metrics for dynamic pricing calculations"""

    total_supply: int = 0
    total_demand: int = 0
    average_task_complexity: float = 0.0
    network_utilization: float = 0.0
    peak_hour_multiplier: float = 1.0
    quality_score: float = 0.0
    completion_rate: float = 0.0
    timestamp: int = 0


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

    def calculate_dynamic_price(
        self, task, market_metrics: MarketMetrics, recent_completion_rate: float = 0.95
    ) -> float:
        """Calculate dynamic price based on multiple market factors"""

        price = self.base_price

        # Supply and demand adjustment
        if market_metrics.total_demand > 0:
            supply_demand_ratio = market_metrics.total_supply / market_metrics.total_demand

            if supply_demand_ratio < self.undersupply_threshold:
                # High demand, low supply - increase price
                supply_multiplier = (
                    1.0
                    + (self.undersupply_threshold - supply_demand_ratio) * self.demand_elasticity
                )
            elif supply_demand_ratio > self.oversupply_threshold:
                # Low demand, high supply - decrease price
                supply_multiplier = (
                    1.0
                    - (supply_demand_ratio - self.oversupply_threshold)
                    * self.supply_elasticity
                    * 0.5
                )
            else:
                # Balanced market
                supply_multiplier = 1.0

            price *= max(0.5, supply_multiplier)  # Minimum 50% of base price

        # Task complexity adjustment (reduced multiplier)
        complexity_multiplier = 1.0 + (getattr(task, "complexity_score", 0.5) - 0.5) * 0.5
        price *= max(0.8, complexity_multiplier)  # Minimum 80% of base

        # Network utilization adjustment (reduced impact)
        utilization_multiplier = 1.0 + (market_metrics.network_utilization - 0.5) * 0.3
        price *= max(0.9, utilization_multiplier)

        # Peak hour adjustment
        price *= market_metrics.peak_hour_multiplier

        # Quality premium (reduced impact)
        if market_metrics.quality_score > 0.8:
            quality_multiplier = 1.0 + (market_metrics.quality_score - 0.8) * 0.3
            price *= quality_multiplier

        # Completion rate impact
        if recent_completion_rate < 0.9:
            completion_penalty = 1.0 + (0.9 - recent_completion_rate) * 0.5
            price *= completion_penalty

        # Network effect (reduced impact)
        network_size = market_metrics.total_supply + market_metrics.total_demand
        if network_size > 100:
            network_bonus = 1.0 + math.log10(network_size / 100) * 0.2
            price *= network_bonus

        # Store price history
        self.price_history.append(price)

        return max(self.base_price * 0.1, price)  # Minimum 10% of base price

    def predict_optimal_price(
        self, future_demand_estimate: int, future_supply_estimate: int
    ) -> float:
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

    def get_market_condition(self, market_metrics: MarketMetrics) -> MarketCondition:
        """Determine current market condition"""

        if market_metrics.total_demand == 0:
            return MarketCondition.OVERSUPPLY

        supply_demand_ratio = market_metrics.total_supply / market_metrics.total_demand

        if market_metrics.total_demand > self.high_demand_threshold * market_metrics.total_supply:
            return MarketCondition.HIGH_DEMAND
        elif supply_demand_ratio < self.undersupply_threshold:
            return MarketCondition.UNDERSUPPLY
        elif supply_demand_ratio > self.oversupply_threshold:
            return MarketCondition.OVERSUPPLY
        else:
            return MarketCondition.BALANCED

    def get_pricing_analytics(self) -> Dict[str, Any]:
        """Get pricing analytics and trends"""

        recent_prices = list(self.price_history)[-20:]

        if not recent_prices:
            return {
                "current_price": self.base_price,
                "price_trend": "stable",
                "volatility": 0.0,
                "recommendations": ["Insufficient data for analysis"],
            }

        current_price = recent_prices[-1]
        average_price = statistics.mean(recent_prices)
        volatility = statistics.stdev(recent_prices) if len(recent_prices) > 1 else 0.0

        # Determine trend
        if len(recent_prices) >= 5:
            recent_avg = statistics.mean(recent_prices[-5:])
            older_avg = (
                statistics.mean(recent_prices[-10:-5])
                if len(recent_prices) >= 10
                else average_price
            )

            if recent_avg > older_avg * 1.05:
                trend = "increasing"
            elif recent_avg < older_avg * 0.95:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # Generate recommendations
        recommendations = []
        if volatility > average_price * 0.3:
            recommendations.append("High price volatility detected - consider market stabilization")
        if current_price > self.base_price * 2:
            recommendations.append("Price significantly above base - monitor for oversupply")
        if current_price < self.base_price * 0.5:
            recommendations.append("Price significantly below base - potential undersupply")

        return {
            "current_price": current_price,
            "average_price": average_price,
            "price_trend": trend,
            "volatility": volatility,
            "recommendations": recommendations or ["Market conditions appear stable"],
        }
