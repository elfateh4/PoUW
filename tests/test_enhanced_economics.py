"""
Test suite for enhanced economic model with dynamic pricing and advanced incentives.

Tests the advanced economic features including dynamic pricing algorithms,
performance-based reward distribution, and real-world economic incentives.
"""

import unittest
import time
import statistics
from unittest.mock import patch

from pouw.economics.enhanced import (
    DynamicPricingEngine,
    AdvancedRewardDistributor,
    EnhancedEconomicSystem,
    MarketCondition,
    MarketMetrics,
    PerformanceMetrics,
    IncentiveType,
    EconomicIncentive
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
            architecture={'input_size': 784, 'hidden_sizes': [128, 64], 'output_size': 10},
            optimizer={'type': 'adam', 'lr': 0.001},
            stopping_criterion={'epochs': 10},
            validation_strategy={'type': 'holdout', 'ratio': 0.2},
            metrics=['accuracy', 'loss'],
            dataset_info={'name': 'MNIST', 'size': 60000},
            performance_requirements={'min_accuracy': 0.8},
            fee=10.0,
            client_id="test_client"
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
            average_task_complexity=0.5,
            network_utilization=0.7,
            peak_hour_multiplier=1.0,
            quality_score=0.8,
            completion_rate=0.95
        )
        
        price = self.pricing_engine.calculate_dynamic_price(
            self.test_task, market_metrics, recent_completion_rate=0.95
        )
        
        # Price should be influenced by task complexity (0.7) and market conditions
        self.assertGreater(price, 8.0)  # Should be above minimum
        self.assertLess(price, 20.0)    # Should not be excessive
        self.assertEqual(len(self.pricing_engine.price_history), 1)
    
    def test_high_demand_pricing(self):
        """Test pricing during high demand (undersupply)"""
        market_metrics = MarketMetrics(
            total_supply=50,
            total_demand=150,  # High demand
            average_task_complexity=0.5,
            network_utilization=0.9,
            peak_hour_multiplier=1.2,
            quality_score=0.8,
            completion_rate=0.95
        )
        
        price = self.pricing_engine.calculate_dynamic_price(
            self.test_task, market_metrics
        )
        
        # Price should be significantly higher due to undersupply
        self.assertGreater(price, 12.0)
    
    def test_oversupply_pricing(self):
        """Test pricing during oversupply"""
        market_metrics = MarketMetrics(
            total_supply=200,
            total_demand=50,  # Low demand
            average_task_complexity=0.3,
            network_utilization=0.4,
            peak_hour_multiplier=0.9,
            quality_score=0.6,
            completion_rate=0.98
        )
        
        price = self.pricing_engine.calculate_dynamic_price(
            self.test_task, market_metrics
        )
        
        # Price should be lower due to oversupply
        self.assertLess(price, 10.0)
        self.assertGreater(price, 1.0)  # But above minimum threshold
    
    def test_market_condition_detection(self):
        """Test market condition classification"""
        # Balanced market
        balanced_metrics = MarketMetrics(total_supply=100, total_demand=100)
        condition = self.pricing_engine.get_market_condition(balanced_metrics)
        self.assertEqual(condition, MarketCondition.BALANCED)
        
        # High demand
        high_demand_metrics = MarketMetrics(total_supply=50, total_demand=200)
        condition = self.pricing_engine.get_market_condition(high_demand_metrics)
        self.assertIn(condition, [MarketCondition.UNDERSUPPLY, MarketCondition.HIGH_DEMAND])
        
        # Oversupply
        oversupply_metrics = MarketMetrics(total_supply=200, total_demand=50)
        condition = self.pricing_engine.get_market_condition(oversupply_metrics)
        self.assertEqual(condition, MarketCondition.OVERSUPPLY)
    
    def test_price_prediction(self):
        """Test price prediction functionality"""
        # Add some price history
        for i in range(10):
            market_metrics = MarketMetrics(total_supply=100, total_demand=100)
            self.pricing_engine.calculate_dynamic_price(self.test_task, market_metrics)
        
        predicted_price = self.pricing_engine.predict_optimal_price(
            future_demand_estimate=120,
            future_supply_estimate=90
        )
        
        self.assertIsInstance(predicted_price, float)
        self.assertGreater(predicted_price, 0)


class TestPerformanceMetrics(unittest.TestCase):
    """Test performance metrics system"""
    
    def test_initialization(self):
        """Test performance metrics initialization"""
        metrics = PerformanceMetrics(node_id="test_node")
        
        self.assertEqual(metrics.node_id, "test_node")
        self.assertEqual(metrics.accuracy_score, 0.0)
        self.assertEqual(metrics.speed_score, 0.0)
        self.assertEqual(metrics.reliability_score, 0.0)
        self.assertEqual(metrics.history_length, 0)
    
    def test_overall_score_calculation(self):
        """Test overall performance score calculation"""
        metrics = PerformanceMetrics(
            node_id="test_node",
            accuracy_score=0.9,
            speed_score=0.8,
            reliability_score=0.85,
            availability_score=0.95,
            quality_score=0.88,
            consistency_score=0.9,
            innovation_score=0.7,
            collaboration_score=0.6
        )
        
        overall = metrics.overall_score()
        
        self.assertIsInstance(overall, float)
        self.assertGreaterEqual(overall, 0.0)
        self.assertLessEqual(overall, 1.0)
        # Should be weighted average, so around 0.85-0.9 range
        self.assertGreater(overall, 0.8)


class TestAdvancedRewardDistributor(unittest.TestCase):
    """Test advanced reward distribution system"""
    
    def setUp(self):
        self.distributor = AdvancedRewardDistributor()
        
        # Create test participants
        self.participants = {
            NodeRole.MINER: ["miner_001", "miner_002", "miner_003"],
            NodeRole.SUPERVISOR: ["supervisor_001"],
            NodeRole.EVALUATOR: ["evaluator_001"]
        }
        
        # Create test performance data
        self.performance_data = {
            "miner_001": {
                "accuracy": 0.95,
                "speed": 0.8,
                "reliability": 0.9,
                "availability": 0.95,
                "quality": 0.88
            },
            "miner_002": {
                "accuracy": 0.85,
                "speed": 0.9,
                "reliability": 0.8,
                "availability": 0.9,
                "quality": 0.82
            },
            "miner_003": {
                "accuracy": 0.75,
                "speed": 0.7,
                "reliability": 0.85,
                "availability": 0.8,
                "quality": 0.7
            }
        }
    
    def test_performance_metrics_update(self):
        """Test performance metrics updating"""
        node_id = "test_node"
        task_metrics = {
            "accuracy": 0.9,
            "speed": 0.8,
            "reliability": 0.85
        }
        
        self.distributor.update_performance_metrics(node_id, task_metrics)
        
        self.assertIn(node_id, self.distributor.performance_metrics)
        metrics = self.distributor.performance_metrics[node_id]
        
        self.assertGreater(metrics.accuracy_score, 0)
        self.assertGreater(metrics.speed_score, 0)
        self.assertGreater(metrics.reliability_score, 0)
        self.assertEqual(metrics.history_length, 1)
    
    def test_basic_reward_calculation(self):
        """Test basic reward calculation without incentives"""
        total_fee = 100.0
        market_condition = MarketCondition.BALANCED
        
        rewards = self.distributor.calculate_advanced_rewards(
            total_fee, self.participants, self.performance_data, market_condition
        )
        
        # All participants should receive rewards
        all_participants = []
        for participant_list in self.participants.values():
            all_participants.extend(participant_list)
        
        for participant in all_participants:
            self.assertIn(participant, rewards)
            self.assertGreater(rewards[participant], 0)
        
        # Total rewards should be close to total fee (accounting for incentive pool)
        total_distributed = sum(rewards.values())
        self.assertLess(abs(total_distributed - total_fee * 0.85), 10.0)  # 15% goes to incentive pool
    
    def test_performance_based_distribution(self):
        """Test that better performers get higher rewards"""
        total_amount = 60.0  # Amount for miners
        
        rewards = self.distributor._distribute_performance_based_rewards(
            self.participants[NodeRole.MINER], total_amount, self.performance_data
        )
        
        # miner_001 has best performance, should get highest reward
        # miner_003 has worst performance, should get lowest reward
        self.assertGreater(rewards["miner_001"], rewards["miner_002"])
        self.assertGreater(rewards["miner_002"], rewards["miner_003"])
        
        # Total should equal input amount
        self.assertAlmostEqual(sum(rewards.values()), total_amount, places=2)
    
    def test_incentive_eligibility(self):
        """Test incentive eligibility checking"""
        # Create node with good performance history
        node_id = "high_performer"
        for i in range(10):  # Add enough history
            self.distributor.update_performance_metrics(node_id, {
                "accuracy": 0.95,
                "speed": 0.9,
                "reliability": 0.95,
                "availability": 0.98,
                "quality": 0.92
            })
        
        performance_bonus = self.distributor.incentives[IncentiveType.PERFORMANCE_BONUS]
        
        is_eligible = self.distributor._is_eligible_for_incentive(
            node_id, performance_bonus, {}
        )
        
        self.assertTrue(is_eligible)
    
    def test_incentive_summary(self):
        """Test incentive summary generation"""
        # Create node with some performance
        node_id = "test_node"
        for i in range(6):  # Meet minimum task requirement
            self.distributor.update_performance_metrics(node_id, {
                "accuracy": 0.9,
                "speed": 0.85,
                "reliability": 0.9,
                "availability": 0.95,
                "quality": 0.88
            })
        
        summary = self.distributor.get_incentive_summary(node_id)
        
        self.assertIn('eligible_incentives', summary)
        self.assertIn('potential_bonus_percentage', summary)
        self.assertIn('current_performance_score', summary)
        self.assertIsInstance(summary['eligible_incentives'], list)


class TestEnhancedEconomicSystem(unittest.TestCase):
    """Test integrated enhanced economic system"""
    
    def setUp(self):
        self.economic_system = EnhancedEconomicSystem(base_price=10.0)
        
        # Create test task
        self.test_task = MLTask(
            task_id="test_task_001",
            model_type="SimpleMLP",
            dataset_id="MNIST",
            fee=15.0,
            num_miners=3,
            num_supervisors=1,
            num_evaluators=1,
            complexity_score=0.6
        )
        
        self.participants = {
            NodeRole.MINER: ["miner_001", "miner_002"],
            NodeRole.SUPERVISOR: ["supervisor_001"],
            NodeRole.EVALUATOR: ["evaluator_001"]
        }
        
        self.performance_data = {
            "miner_001": {"accuracy": 0.9, "speed": 0.8, "reliability": 0.9},
            "miner_002": {"accuracy": 0.85, "speed": 0.9, "reliability": 0.85}
        }
    
    def test_market_metrics_update(self):
        """Test market metrics updating"""
        recent_tasks = [self.test_task]
        network_stats = {
            'active_nodes': 80,
            'total_nodes': 100,
            'completion_rate': 0.95,
            'average_quality': 0.85
        }
        
        self.economic_system.update_market_metrics(
            total_supply=120,
            total_demand=100,
            recent_tasks=recent_tasks,
            network_stats=network_stats
        )
        
        metrics = self.economic_system.market_metrics
        self.assertEqual(metrics.total_supply, 120)
        self.assertEqual(metrics.total_demand, 100)
        self.assertEqual(metrics.network_utilization, 0.8)
        self.assertEqual(metrics.completion_rate, 0.95)
    
    def test_optimal_task_fee_calculation(self):
        """Test optimal task fee calculation"""
        # Update market metrics first
        self.economic_system.update_market_metrics(
            total_supply=100,
            total_demand=100,
            recent_tasks=[self.test_task],
            network_stats={'completion_rate': 0.95}
        )
        
        optimal_fee = self.economic_system.calculate_optimal_task_fee(self.test_task)
        
        self.assertIsInstance(optimal_fee, float)
        self.assertGreater(optimal_fee, 0)
        self.assertEqual(len(self.economic_system.price_history), 1)
    
    def test_task_reward_distribution(self):
        """Test complete task reward distribution"""
        rewards = self.economic_system.distribute_task_rewards(
            self.test_task, self.participants, self.performance_data
        )
        
        # All participants should get rewards
        for participant_list in self.participants.values():
            for participant in participant_list:
                self.assertIn(participant, rewards)
                self.assertGreater(rewards[participant], 0)
        
        # Market event should be logged
        self.assertEqual(len(self.economic_system.market_events), 1)
        event = self.economic_system.market_events[0]
        self.assertEqual(event['event_type'], 'task_completion')
        self.assertEqual(event['task_id'], self.test_task.task_id)
    
    def test_economic_analytics(self):
        """Test economic analytics generation"""
        # Generate some activity
        for i in range(5):
            optimal_fee = self.economic_system.calculate_optimal_task_fee(self.test_task)
            rewards = self.economic_system.distribute_task_rewards(
                self.test_task, self.participants, self.performance_data
            )
        
        analytics = self.economic_system.get_economic_analytics()
        
        required_keys = [
            'price_analytics', 'market_condition', 'market_metrics',
            'incentive_pool', 'total_market_events', 'economic_health'
        ]
        
        for key in required_keys:
            self.assertIn(key, analytics)
        
        # Price analytics should have valid data
        price_analytics = analytics['price_analytics']
        self.assertGreater(price_analytics['current_price'], 0)
        self.assertGreater(price_analytics['average_price'], 0)
        
        # Economic health should be a valid assessment
        self.assertIn(analytics['economic_health'], ['excellent', 'good', 'fair', 'poor'])
    
    def test_economic_health_assessment(self):
        """Test economic health assessment"""
        # Create stable market conditions
        for i in range(20):
            market_metrics = MarketMetrics(
                total_supply=100,
                total_demand=100,
                network_utilization=0.8,
                quality_score=0.9,
                completion_rate=0.95
            )
            price = self.economic_system.pricing_engine.calculate_dynamic_price(
                self.test_task, market_metrics
            )
        
        health = self.economic_system._assess_economic_health()
        
        # With stable, balanced conditions, health should be good
        self.assertIn(health, ['good', 'excellent'])


class TestMarketConditions(unittest.TestCase):
    """Test various market condition scenarios"""
    
    def setUp(self):
        self.economic_system = EnhancedEconomicSystem()
        self.test_task = MLTask(
            task_id="market_test",
            model_type="SimpleMLP",
            dataset_id="MNIST",
            fee=10.0,
            num_miners=2,
            num_supervisors=1,
            num_evaluators=1,
            complexity_score=0.5
        )
    
    def test_peak_hour_pricing(self):
        """Test peak hour price adjustments"""
        with patch('time.localtime') as mock_time:
            # Mock business hours (should increase price)
            mock_time.return_value.tm_hour = 14  # 2 PM
            
            self.economic_system.update_market_metrics(
                total_supply=100,
                total_demand=100,
                recent_tasks=[self.test_task],
                network_stats={}
            )
            
            peak_price = self.economic_system.calculate_optimal_task_fee(self.test_task)
            
            # Mock off-peak hours (should decrease price)
            mock_time.return_value.tm_hour = 3  # 3 AM
            
            self.economic_system.update_market_metrics(
                total_supply=100,
                total_demand=100,
                recent_tasks=[self.test_task],
                network_stats={}
            )
            
            off_peak_price = self.economic_system.calculate_optimal_task_fee(self.test_task)
            
            # Peak hour price should be higher
            self.assertGreater(peak_price, off_peak_price)
    
    def test_quality_premium_effect(self):
        """Test quality premium pricing effects"""
        # High quality network
        high_quality_stats = {'average_quality': 0.95, 'completion_rate': 0.98}
        self.economic_system.update_market_metrics(
            total_supply=100, total_demand=100,
            recent_tasks=[self.test_task], network_stats=high_quality_stats
        )
        high_quality_price = self.economic_system.calculate_optimal_task_fee(self.test_task)
        
        # Low quality network
        low_quality_stats = {'average_quality': 0.6, 'completion_rate': 0.85}
        self.economic_system.update_market_metrics(
            total_supply=100, total_demand=100,
            recent_tasks=[self.test_task], network_stats=low_quality_stats
        )
        low_quality_price = self.economic_system.calculate_optimal_task_fee(self.test_task)
        
        # High quality should command premium
        self.assertGreater(high_quality_price, low_quality_price)


if __name__ == '__main__':
    unittest.main(verbosity=2)
