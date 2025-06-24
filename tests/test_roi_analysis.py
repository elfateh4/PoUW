"""
Test suite for ROI analysis and economic modeling system.
Tests comprehensive profitability calculations, economic simulations,
and market comparison functionalities.
"""

import unittest
import time
from unittest.mock import patch, MagicMock

from pouw.economics.roi_analysis import (
    ROIAnalyzer,
    ParticipantRole,
    CostCategory,
    CostStructure,
    RevenueStream,
    ROIMetrics,
    NetworkEconomics,
    analyze_miner_profitability,
    compare_pouw_vs_bitcoin_mining,
    calculate_network_sustainability
)

class TestCostStructure(unittest.TestCase):
    """Test cost structure calculations"""
    
    def setUp(self):
        self.cost_structure = CostStructure(
            hardware_cost=2.0,
            electricity_cost=0.5,
            network_cost=0.1,
            maintenance_cost=0.2,
            opportunity_cost=0.3,
            stake_cost=1000.0
        )
    
    def test_total_operational_cost_calculation(self):
        """Test total operational cost calculation"""
        expected_cost = 2.0 + 0.5 + 0.1 + 0.2 + 0.3  # 3.1
        self.assertEqual(self.cost_structure.total_operational_cost_per_hour(), 3.1)
    
    def test_total_cost_including_stake(self):
        """Test total cost including stake opportunity cost"""
        stake_apy = 0.05  # 5%
        expected_stake_hourly = (1000.0 * 0.05) / (365 * 24)  # ~0.0057
        expected_total = 3.1 + expected_stake_hourly
        
        actual_total = self.cost_structure.total_cost_including_stake(stake_apy)
        self.assertAlmostEqual(actual_total, expected_total, places=4)

class TestRevenueStream(unittest.TestCase):
    """Test revenue stream calculations"""
    
    def setUp(self):
        self.revenue_stream = RevenueStream(
            block_rewards=10.0,
            task_fees=15.0,
            performance_bonuses=2.0,
            network_incentives=3.0,
            staking_rewards=1.0
        )
    
    def test_total_revenue_calculation(self):
        """Test total revenue calculation"""
        expected_revenue = 10.0 + 15.0 + 2.0 + 3.0 + 1.0  # 31.0
        self.assertEqual(self.revenue_stream.total_revenue_per_hour(), 31.0)

class TestROIMetrics(unittest.TestCase):
    """Test ROI metrics"""
    
    def setUp(self):
        self.profitable_roi = ROIMetrics(
            hourly_profit=5.0,
            daily_profit=120.0,
            monthly_profit=3600.0,
            annual_profit=43800.0,
            roi_percentage=25.0,
            payback_period_days=180.0,
            break_even_point=10.0,
            risk_adjusted_roi=15.0
        )
        
        self.unprofitable_roi = ROIMetrics(
            hourly_profit=-2.0,
            daily_profit=-48.0,
            monthly_profit=-1440.0,
            annual_profit=-17520.0,
            roi_percentage=-10.0,
            payback_period_days=float('inf'),
            break_even_point=12.0,
            risk_adjusted_roi=-20.0
        )
    
    def test_profitability_check(self):
        """Test profitability determination"""
        self.assertTrue(self.profitable_roi.is_profitable())
        self.assertFalse(self.unprofitable_roi.is_profitable())

class TestNetworkEconomics(unittest.TestCase):
    """Test network economics calculations"""
    
    def setUp(self):
        self.network_economics = NetworkEconomics(
            total_network_value=5000000.0,  # $5M
            total_stake_value=200000.0,     # $200k
            daily_transaction_volume=50000.0,  # $50k
            network_utilization=0.8,
            average_task_fee=25.0,
            network_growth_rate=0.05  # 5% growth
        )
    
    def test_network_health_score(self):
        """Test network health score calculation"""
        health_score = self.network_economics.network_health_score()
        
        # Should be good (utilization=0.8, growth=0.05, value normalized to 1.0)
        expected_score = (0.8 + 0.05 + 1.0) / 3  # ~0.617
        self.assertAlmostEqual(health_score, expected_score, places=2)
        
        # Health score should be between 0 and 1
        self.assertGreaterEqual(health_score, 0.0)
        self.assertLessEqual(health_score, 1.0)

class TestROIAnalyzer(unittest.TestCase):
    """Test ROI analyzer functionality"""
    
    def setUp(self):
        self.analyzer = ROIAnalyzer()
        self.performance_metrics = {
            'performance_score': 0.85,
            'uptime': 0.98,
            'accuracy': 0.92
        }
        self.market_conditions = {
            'base_block_reward': 12.5,
            'hourly_task_volume': 10,
            'average_task_fee': 25.0,
            'electricity_price_multiplier': 1.0,
            'hardware_price_multiplier': 1.0,
            'network_price_multiplier': 1.0
        }
    
    def test_miner_roi_calculation(self):
        """Test miner ROI calculation"""
        roi_metrics = self.analyzer.calculate_participant_roi(
            ParticipantRole.MINER,
            self.performance_metrics,
            self.market_conditions
        )
        
        # Should return valid ROI metrics
        self.assertIsInstance(roi_metrics, ROIMetrics)
        self.assertIsInstance(roi_metrics.hourly_profit, float)
        self.assertIsInstance(roi_metrics.roi_percentage, float)
        self.assertGreater(roi_metrics.daily_profit, roi_metrics.hourly_profit * 20)  # Should be ~24x
    
    def test_supervisor_roi_calculation(self):
        """Test supervisor ROI calculation"""
        roi_metrics = self.analyzer.calculate_participant_roi(
            ParticipantRole.SUPERVISOR,
            self.performance_metrics,
            self.market_conditions
        )
        
        # Supervisors should have different revenue structure than miners
        self.assertIsInstance(roi_metrics, ROIMetrics)
        # Supervisors typically have lower costs but also lower revenue
        
    def test_client_cost_savings(self):
        """Test client cost savings calculation"""
        task_requirements = {
            'estimated_hours': 2.0,
            'gpu_count': 4,
            'complexity_multiplier': 1.5
        }
        
        savings = self.analyzer.calculate_client_savings(task_requirements)
        
        self.assertIn('cloud_cost', savings)
        self.assertIn('pouw_cost', savings)
        self.assertIn('savings', savings)
        self.assertIn('savings_percentage', savings)
        self.assertIn('is_cheaper', savings)
        
        # All values should be numeric
        self.assertIsInstance(savings['cloud_cost'], float)
        self.assertIsInstance(savings['pouw_cost'], float)
        self.assertIsInstance(savings['savings'], float)
    
    def test_market_condition_adjustments(self):
        """Test cost adjustments based on market conditions"""
        base_costs = self.analyzer.baseline_costs[ParticipantRole.MINER]
        
        # Test with increased electricity costs
        high_electricity_conditions = self.market_conditions.copy()
        high_electricity_conditions['electricity_price_multiplier'] = 2.0
        
        adjusted_costs = self.analyzer._adjust_costs_for_market(base_costs, high_electricity_conditions)
        
        # Electricity cost should be doubled
        self.assertEqual(adjusted_costs.electricity_cost, base_costs.electricity_cost * 2.0)
        # Other costs should remain unchanged
        self.assertEqual(adjusted_costs.hardware_cost, base_costs.hardware_cost)
    
    def test_revenue_calculation_for_different_roles(self):
        """Test revenue calculation for different participant roles"""
        roles_to_test = [
            ParticipantRole.MINER,
            ParticipantRole.SUPERVISOR,
            ParticipantRole.EVALUATOR,
            ParticipantRole.VERIFIER
        ]
        
        for role in roles_to_test:
            revenue = self.analyzer._calculate_revenue_for_role(
                role, self.performance_metrics, self.market_conditions
            )
            
            self.assertIsInstance(revenue, RevenueStream)
            self.assertGreaterEqual(revenue.total_revenue_per_hour(), 0)
            
            # Miners should have highest revenue potential
            if role == ParticipantRole.MINER:
                self.assertGreater(revenue.block_rewards, 0)
    
    def test_alternative_comparisons(self):
        """Test comparison with alternative investments"""
        roi_metrics = self.analyzer.calculate_participant_roi(
            ParticipantRole.MINER,
            self.performance_metrics,
            self.market_conditions
        )
        
        comparisons = self.analyzer.compare_with_alternatives(roi_metrics, ParticipantRole.MINER)
        
        # Should include Bitcoin mining comparison for miners
        self.assertIn('bitcoin_mining', comparisons)
        self.assertIn('gpu_rental', comparisons)
        self.assertIn('risk_free_investment', comparisons)
        
        # Each comparison should have required fields
        btc_comparison = comparisons['bitcoin_mining']
        self.assertIn('annual_roi', btc_comparison)
        self.assertIn('advantage', btc_comparison)
        self.assertIn('is_better', btc_comparison)
    
    def test_network_simulation(self):
        """Test network economics simulation"""
        network_params = {
            'initial_participants': {
                ParticipantRole.MINER: 50,
                ParticipantRole.SUPERVISOR: 10,
                ParticipantRole.EVALUATOR: 15,
                ParticipantRole.VERIFIER: 20,
                ParticipantRole.CLIENT: 100
            },
            'daily_task_volume': 240,
            'growth_rate': 0.001  # 0.1% daily growth
        }
        
        simulation = self.analyzer.simulate_network_economics(network_params, 30)  # 30 days
        
        self.assertIn('simulation_results', simulation)
        self.assertIn('final_network_value', simulation)
        self.assertIn('average_network_health', simulation)
        
        # Should have 30 days of results
        self.assertEqual(len(simulation['simulation_results']), 30)
        
        # Network should show growth
        initial_value = simulation['simulation_results'][0]['network_economics'].total_network_value
        final_value = simulation['final_network_value']
        self.assertGreaterEqual(final_value, initial_value)
    
    def test_profitability_report_generation(self):
        """Test comprehensive profitability report generation"""
        report = self.analyzer.generate_profitability_report(
            ParticipantRole.MINER,
            self.performance_metrics,
            self.market_conditions
        )
        
        # Check required sections
        required_sections = [
            'role', 'roi_metrics', 'profitability_assessment',
            'cost_breakdown', 'market_comparisons', 'sensitivity_analysis'
        ]
        
        for section in required_sections:
            self.assertIn(section, report)
        
        # Check profitability assessment details
        assessment = report['profitability_assessment']
        self.assertIn('is_profitable', assessment)
        self.assertIn('profitability_rating', assessment)
        self.assertIn('risk_level', assessment)
        self.assertIn('recommendation', assessment)
        
        # Check sensitivity analysis
        sensitivity = report['sensitivity_analysis']
        self.assertIsInstance(sensitivity, dict)
        # Should have multiple scenario tests
        self.assertGreater(len(sensitivity), 0)
    
    def test_profitability_ratings(self):
        """Test profitability rating system"""
        # Test excellent rating
        excellent_roi = ROIMetrics(1.0, 24.0, 720.0, 8760.0, 60.0, 100.0, 5.0, 50.0)
        rating = self.analyzer._get_profitability_rating(excellent_roi)
        self.assertEqual(rating, "Excellent")
        
        # Test unprofitable rating
        bad_roi = ROIMetrics(-1.0, -24.0, -720.0, -8760.0, -10.0, float('inf'), 5.0, -20.0)
        rating = self.analyzer._get_profitability_rating(bad_roi)
        self.assertEqual(rating, "Unprofitable")
        
        # Test marginal rating (5% ROI falls in marginal category: >= 0 but < 10)
        marginal_roi = ROIMetrics(0.1, 2.4, 72.0, 876.0, 5.0, 1000.0, 3.0, -5.0)
        rating = self.analyzer._get_profitability_rating(marginal_roi)
        self.assertEqual(rating, "Marginal")
        
        # Test fair rating (15% ROI falls in fair category: >= 10 but < 25)
        fair_roi = ROIMetrics(0.5, 12.0, 360.0, 4380.0, 15.0, 200.0, 3.0, 5.0)
        rating = self.analyzer._get_profitability_rating(fair_roi)
        self.assertEqual(rating, "Fair")

class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions for ROI analysis"""
    
    def test_analyze_miner_profitability(self):
        """Test miner profitability analysis convenience function"""
        hardware_specs = {
            'gpu_count': 8,
            'gpu_memory': '24GB',
            'power_consumption': 2000  # watts
        }
        
        market_conditions = {
            'base_block_reward': 12.5,
            'hourly_task_volume': 15,
            'average_task_fee': 30.0
        }
        
        result = analyze_miner_profitability(hardware_specs, market_conditions, 0.9)
        
        self.assertIn('role', result)
        self.assertEqual(result['role'], 'miner')
        self.assertIn('roi_metrics', result)
        self.assertIn('profitability_assessment', result)
    
    def test_pouw_vs_bitcoin_comparison(self):
        """Test PoUW vs Bitcoin mining comparison"""
        investment_amount = 10000.0  # $10k investment
        
        comparison = compare_pouw_vs_bitcoin_mining(investment_amount)
        
        required_fields = [
            'investment_amount', 'pouw_annual_roi', 'bitcoin_annual_roi',
            'pouw_advantage', 'pouw_annual_profit', 'bitcoin_annual_profit',
            'recommendation'
        ]
        
        for field in required_fields:
            self.assertIn(field, comparison)
        
        # Investment amount should match
        self.assertEqual(comparison['investment_amount'], investment_amount)
        
        # Recommendation should be either 'PoUW' or 'Bitcoin'
        self.assertIn(comparison['recommendation'], ['PoUW', 'Bitcoin'])
    
    def test_network_sustainability_calculation(self):
        """Test network sustainability calculation"""
        participants = {
            'miners': 100,
            'supervisors': 20,
            'evaluators': 30,
            'verifiers': 40,
            'clients': 200
        }
        
        task_volume = 500  # tasks per day
        
        sustainability = calculate_network_sustainability(participants, task_volume)
        
        required_fields = [
            'network_health_score', 'projected_growth',
            'final_network_value', 'sustainability_rating'
        ]
        
        for field in required_fields:
            self.assertIn(field, sustainability)
        
        # Health score should be between 0 and 1
        self.assertGreaterEqual(sustainability['network_health_score'], 0.0)
        self.assertLessEqual(sustainability['network_health_score'], 1.0)
        
        # Sustainability rating should be valid
        valid_ratings = ['High', 'Medium', 'Low']
        self.assertIn(sustainability['sustainability_rating'], valid_ratings)

class TestRiskAssessment(unittest.TestCase):
    """Test risk assessment functionality"""
    
    def setUp(self):
        self.analyzer = ROIAnalyzer()
    
    def test_risk_level_assessment(self):
        """Test risk level assessment for different scenarios"""
        # High-risk scenario (long payback period)
        high_risk_roi = ROIMetrics(0.5, 12.0, 360.0, 4380.0, 5.0, 400.0, 3.0, -5.0)
        risk_level = self.analyzer._assess_risk_level(high_risk_roi, ParticipantRole.MINER)
        self.assertEqual(risk_level, "High")
        
        # Low-risk scenario (short payback period)
        low_risk_roi = ROIMetrics(5.0, 120.0, 3600.0, 43800.0, 40.0, 90.0, 3.0, 30.0)
        risk_level = self.analyzer._assess_risk_level(low_risk_roi, ParticipantRole.MINER)
        self.assertEqual(risk_level, "Low")
    
    def test_investment_recommendations(self):
        """Test investment recommendation generation"""
        # Strongly recommended scenario
        excellent_roi = ROIMetrics(8.0, 192.0, 5760.0, 70080.0, 50.0, 120.0, 5.0, 40.0)
        recommendation = self.analyzer._generate_recommendation(excellent_roi, ParticipantRole.MINER)
        self.assertEqual(recommendation, "Strongly Recommended")
        
        # Not recommended scenario
        poor_roi = ROIMetrics(-2.0, -48.0, -1440.0, -17520.0, -15.0, float('inf'), 5.0, -25.0)
        recommendation = self.analyzer._generate_recommendation(poor_roi, ParticipantRole.MINER)
        self.assertEqual(recommendation, "Not Recommended")

class TestSensitivityAnalysis(unittest.TestCase):
    """Test sensitivity analysis functionality"""
    
    def setUp(self):
        self.analyzer = ROIAnalyzer()
        self.performance_metrics = {'performance_score': 0.8}
        self.market_conditions = {
            'base_block_reward': 12.5,
            'hourly_task_volume': 10,
            'average_task_fee': 25.0
        }
    
    def test_sensitivity_analysis_structure(self):
        """Test sensitivity analysis result structure"""
        sensitivity = self.analyzer._perform_sensitivity_analysis(
            ParticipantRole.MINER,
            self.performance_metrics,
            self.market_conditions
        )
        
        # Should have results for different scenarios
        self.assertIsInstance(sensitivity, dict)
        self.assertGreater(len(sensitivity), 0)
        
        # Check for expected scenario types
        expected_scenarios = ['task_volume_0.5x', 'task_volume_1.5x', 'performance_0.8x', 'performance_1.2x']
        
        for scenario in expected_scenarios:
            if scenario in sensitivity:
                self.assertIn('roi_change', sensitivity[scenario])
                self.assertIn('profit_change', sensitivity[scenario])

if __name__ == '__main__':
    unittest.main()
