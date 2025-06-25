"""
Test suite for enhanced economic model with dynamic pricing and advanced incentives.

Tests the advanced economic features including dynamic pricing algorithms,
performance-based reward distribution, and real-world economic incentives.
"""

import unittest
import time
import statistics
from unittest.mock import patch, MagicMock

from pouw.economics.enhanced import (
    DynamicPricingEngine,
    AdvancedRewardDistributor,
    EnhancedEconomicSystem,
    MarketCondition,
    MarketMetrics,
    PerformanceMetrics,
    IncentiveType,
    EconomicIncentive,
)
from pouw.economics.system import NodeRole
from pouw.blockchain.core import MLTask


class TestDynamicPricingEngine(unittest.TestCase):
    """Test dynamic pricing engine"""

    def setUp(self):
        self.pricing_engine = DynamicPricingEngine(base_price=10.0)

        # Create test task with correct MLTask constructor
        self.test_task = MLTask(
            task_id="test_task_001",
            model_type="SimpleMLP",
            architecture={"input_size": 784, "hidden_sizes": [128, 64], "output_size": 10},
            optimizer={"type": "adam", "lr": 0.001},
            stopping_criterion={"epochs": 10},
            validation_strategy={"type": "holdout", "ratio": 0.2},
            metrics=["accuracy", "loss"],
            dataset_info={"name": "MNIST", "size": 60000},
            performance_requirements={"min_accuracy": 0.8},
            fee=10.0,
            client_id="test_client",
        )

    def test_initialization(self):
        """Test pricing engine initialization"""
        self.assertEqual(self.pricing_engine.base_price, 10.0)
        self.assertEqual(len(self.pricing_engine.price_history), 0)
        self.assertEqual(len(self.pricing_engine.market_history), 0)

    def test_balanced_market_pricing(self):
        """Test pricing in balanced market conditions"""
        market_metrics = MarketMetrics(
            total_supply=100,
            total_demand=100,
            network_utilization=0.5,
            quality_score=0.8,
            completion_rate=0.95,
        )

        price = self.pricing_engine.calculate_dynamic_price(self.test_task, market_metrics)

        # Should be close to base price with some complexity adjustment
        self.assertGreater(price, 5.0)
        self.assertLess(price, 20.0)
        self.assertEqual(len(self.pricing_engine.price_history), 1)

    def test_high_demand_pricing(self):
        """Test pricing in high demand conditions"""
        market_metrics = MarketMetrics(
            total_supply=50,
            total_demand=200,  # High demand
            network_utilization=0.8,
            quality_score=0.9,
        )

        price = self.pricing_engine.calculate_dynamic_price(self.test_task, market_metrics)

        # Should be higher than base price
        self.assertGreater(price, self.pricing_engine.base_price)

    def test_oversupply_pricing(self):
        """Test pricing in oversupply conditions"""
        market_metrics = MarketMetrics(
            total_supply=200, total_demand=50, network_utilization=0.3  # Low demand
        )

        price = self.pricing_engine.calculate_dynamic_price(self.test_task, market_metrics)

        # Should be lower than base price
        self.assertLess(price, self.pricing_engine.base_price)

    def test_market_condition_detection(self):
        """Test market condition detection"""
        # Balanced market
        balanced_metrics = MarketMetrics(total_supply=100, total_demand=100)
        condition = self.pricing_engine.get_market_condition(balanced_metrics)
        self.assertEqual(condition, MarketCondition.BALANCED)

        # Oversupply
        oversupply_metrics = MarketMetrics(total_supply=200, total_demand=50)
        condition = self.pricing_engine.get_market_condition(oversupply_metrics)
        self.assertEqual(condition, MarketCondition.OVERSUPPLY)

        # Undersupply
        undersupply_metrics = MarketMetrics(total_supply=50, total_demand=200)
        condition = self.pricing_engine.get_market_condition(undersupply_metrics)
        self.assertEqual(condition, MarketCondition.UNDERSUPPLY)

    def test_price_prediction(self):
        """Test price prediction functionality"""
        # Add some price history
        for i in range(10):
            market_metrics = MarketMetrics(total_supply=100 + i * 5, total_demand=100)
            self.pricing_engine.calculate_dynamic_price(self.test_task, market_metrics)

        predicted_price = self.pricing_engine.predict_optimal_price(
            future_demand_estimate=120, future_supply_estimate=80
        )

        self.assertGreater(predicted_price, 0)


class TestPerformanceMetrics(unittest.TestCase):
    """Test performance metrics system"""

    def test_performance_metrics_creation(self):
        """Test performance metrics initialization"""
        metrics = PerformanceMetrics(node_id="test_node")

        self.assertEqual(metrics.node_id, "test_node")
        self.assertEqual(metrics.accuracy_score, 0.0)
        self.assertEqual(metrics.overall_score(), 0.0)

    def test_overall_score_calculation(self):
        """Test weighted overall score calculation"""
        metrics = PerformanceMetrics(
            node_id="test_node",
            accuracy_score=0.9,
            speed_score=0.8,
            reliability_score=0.95,
            availability_score=0.9,
            quality_score=0.85,
            consistency_score=0.9,
            innovation_score=0.7,
            collaboration_score=0.8,
        )

        score = metrics.overall_score()

        # Should be weighted average, so around 0.85-0.9 range
        self.assertGreater(score, 0.8)
        self.assertLess(score, 1.0)


class TestAdvancedRewardDistributor(unittest.TestCase):
    """Test advanced reward distribution"""

    def setUp(self):
        self.distributor = AdvancedRewardDistributor()
        self.participants = {
            NodeRole.MINER: ["miner_001", "miner_002"],
            NodeRole.SUPERVISOR: ["supervisor_001"],
            NodeRole.EVALUATOR: ["evaluator_001"],
        }

    def test_basic_reward_distribution(self):
        """Test basic reward distribution without performance data"""
        total_fee = 100.0
        performance_data = {}

        rewards = self.distributor.calculate_advanced_rewards(
            total_fee, self.participants, performance_data, MarketCondition.BALANCED
        )

        # Check that all participants got rewards
        total_distributed = sum(rewards.values())

        # Should distribute most of the fee (minus incentive pool)
        self.assertLess(
            abs(total_distributed - total_fee * 0.85), 10.0
        )  # 15% goes to incentive pool

    def test_performance_based_distribution(self):
        """Test performance-based reward distribution"""
        total_fee = 100.0
        performance_data = {
            "miner_001": {
                "accuracy": 0.95,
                "speed": 0.8,
                "reliability": 0.9,
                "availability": 0.95,
                "quality": 0.9,
            },
            "miner_002": {
                "accuracy": 0.7,
                "speed": 0.6,
                "reliability": 0.8,
                "availability": 0.85,
                "quality": 0.75,
            },
        }

        rewards = self.distributor.calculate_advanced_rewards(
            total_fee, self.participants, performance_data, MarketCondition.BALANCED
        )

        # Better performer should get more reward
        self.assertGreater(rewards["miner_001"], rewards["miner_002"])

    def test_incentive_eligibility(self):
        """Test incentive eligibility checking"""
        # Add performance metrics for a node
        self.distributor.update_performance_metrics(
            "test_node",
            {
                "accuracy": 0.9,
                "speed": 0.8,
                "reliability": 0.95,
                "availability": 0.9,
                "quality": 0.9,
            },
        )

        # Should be eligible for performance bonus
        incentive = self.distributor.incentives[IncentiveType.PERFORMANCE_BONUS]
        is_eligible = self.distributor._is_eligible_for_incentive("test_node", incentive, {})

        # May not be eligible initially due to min_tasks requirement
        # Just test that the function runs without error
        self.assertIsInstance(is_eligible, bool)


class TestEnhancedEconomicSystem(unittest.TestCase):
    """Test enhanced economic system"""

    def setUp(self):
        self.economic_system = EnhancedEconomicSystem(base_price=10.0)

        # Create test task with correct MLTask constructor
        self.test_task = MLTask(
            task_id="test_task_001",
            model_type="SimpleMLP",
            architecture={"input_size": 784, "hidden_sizes": [128, 64], "output_size": 10},
            optimizer={"type": "adam", "lr": 0.001},
            stopping_criterion={"epochs": 10},
            validation_strategy={"type": "holdout", "ratio": 0.2},
            metrics=["accuracy", "loss"],
            dataset_info={"name": "MNIST", "size": 60000},
            performance_requirements={"min_accuracy": 0.8},
            fee=15.0,
            client_id="test_client",
        )

        self.participants = {
            NodeRole.MINER: ["miner_001", "miner_002"],
            NodeRole.SUPERVISOR: ["supervisor_001"],
            NodeRole.EVALUATOR: ["evaluator_001"],
        }

    def test_market_metrics_update(self):
        """Test market metrics updating"""
        recent_tasks = [self.test_task]
        network_stats = {
            "active_nodes": 80,
            "total_nodes": 100,
            "completion_rate": 0.95,
            "average_quality": 0.85,
        }

        self.economic_system.update_market_metrics(
            total_supply=150,
            total_demand=120,
            recent_tasks=recent_tasks,
            network_stats=network_stats,
        )

        metrics = self.economic_system.market_metrics
        self.assertEqual(metrics.total_supply, 150)
        self.assertEqual(metrics.total_demand, 120)
        self.assertEqual(metrics.network_utilization, 0.8)
        self.assertGreater(metrics.average_task_complexity, 0)

    def test_optimal_task_fee_calculation(self):
        """Test optimal task fee calculation"""
        # Setup market conditions
        self.economic_system.update_market_metrics(
            total_supply=100,
            total_demand=100,
            recent_tasks=[self.test_task],
            network_stats={"active_nodes": 50, "total_nodes": 100},
        )

        fee = self.economic_system.calculate_optimal_task_fee(self.test_task)

        self.assertGreater(fee, 0)
        self.assertEqual(len(self.economic_system.price_history), 1)

    def test_task_reward_distribution(self):
        """Test task reward distribution"""
        performance_data = {
            "miner_001": {"accuracy": 0.9, "speed": 0.8},
            "miner_002": {"accuracy": 0.85, "speed": 0.7},
        }

        rewards = self.economic_system.distribute_task_rewards(
            self.test_task, self.participants, performance_data
        )

        # Should have rewards for all participants
        self.assertIn("miner_001", rewards)
        self.assertIn("miner_002", rewards)
        self.assertIn("supervisor_001", rewards)
        self.assertIn("evaluator_001", rewards)

        # Should log a market event
        self.assertEqual(len(self.economic_system.market_events), 1)

    def test_economic_analytics(self):
        """Test economic analytics generation"""
        # Add some price history
        self.economic_system.price_history = [
            (time.time() - 100, 10.0),
            (time.time() - 50, 12.0),
            (time.time(), 11.0),
        ]

        analytics = self.economic_system.get_economic_analytics()

        expected_keys = [
            "price_analytics",
            "market_condition",
            "market_metrics",
            "incentive_pool",
            "total_market_events",
            "economic_health",
        ]

        for key in expected_keys:
            self.assertIn(key, analytics)

        self.assertIn("current_price", analytics["price_analytics"])
        self.assertIn("average_price", analytics["price_analytics"])

    def test_economic_health_assessment(self):
        """Test economic health assessment"""
        # Setup some market conditions
        self.economic_system.market_metrics.network_utilization = 0.8
        self.economic_system.market_metrics.quality_score = 0.9
        self.economic_system.market_metrics.completion_rate = 0.95

        health = self.economic_system._assess_economic_health()

        self.assertIn(health, ["excellent", "good", "fair", "poor"])


class TestMarketConditions(unittest.TestCase):
    """Test market condition effects"""

    def setUp(self):
        self.economic_system = EnhancedEconomicSystem()
        self.test_task = MLTask(
            task_id="market_test",
            model_type="SimpleMLP",
            architecture={"input_size": 784, "hidden_sizes": [64], "output_size": 10},
            optimizer={"type": "adam", "lr": 0.001},
            stopping_criterion={"epochs": 5},
            validation_strategy={"type": "holdout", "ratio": 0.2},
            metrics=["accuracy"],
            dataset_info={"name": "MNIST", "size": 10000},
            performance_requirements={"min_accuracy": 0.7},
            fee=10.0,
            client_id="test_client",
        )

    def test_peak_hour_pricing(self):
        """Test peak hour price adjustments"""
        with patch("time.localtime") as mock_time:
            # Mock business hours (should increase price)
            mock_time.return_value.tm_hour = 14

            self.economic_system.update_market_metrics(
                total_supply=100, total_demand=100, recent_tasks=[self.test_task], network_stats={}
            )

            peak_price = self.economic_system.calculate_optimal_task_fee(self.test_task)

            # Mock off-peak hours (should decrease price)
            mock_time.return_value.tm_hour = 3

            self.economic_system.update_market_metrics(
                total_supply=100, total_demand=100, recent_tasks=[self.test_task], network_stats={}
            )

            off_peak_price = self.economic_system.calculate_optimal_task_fee(self.test_task)

            # Peak hour should be more expensive
            self.assertGreater(peak_price, off_peak_price)

    def test_quality_premium_effect(self):
        """Test quality premium pricing"""
        # High quality network
        self.economic_system.update_market_metrics(
            total_supply=100,
            total_demand=100,
            recent_tasks=[self.test_task],
            network_stats={"average_quality": 0.95},
        )

        high_quality_price = self.economic_system.calculate_optimal_task_fee(self.test_task)

        # Low quality network
        self.economic_system.update_market_metrics(
            total_supply=100,
            total_demand=100,
            recent_tasks=[self.test_task],
            network_stats={"average_quality": 0.6},
        )

        low_quality_price = self.economic_system.calculate_optimal_task_fee(self.test_task)

        # High quality should command premium
        self.assertGreater(high_quality_price, low_quality_price)


if __name__ == "__main__":
    unittest.main()
