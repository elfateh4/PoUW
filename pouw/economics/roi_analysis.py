"""
ROI Analysis and Economic Modeling for PoUW

This module implements comprehensive ROI calculations and economic modeling
as specified in the research paper "A Proof of Useful Work for Artificial
Intelligence on the Blockchain" by Lihu et al.

Features:
- Profitability analysis tools for participants
- ROI calculations comparing PoUW vs traditional mining
- Economic simulation capabilities for network optimization
- Real-world cost modeling and revenue projections
- Network health and sustainability metrics
"""

import time
import math
import statistics
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

class ParticipantRole(Enum):
    """Participant roles in the PoUW network"""
    MINER = "miner"
    SUPERVISOR = "supervisor"
    EVALUATOR = "evaluator"
    VERIFIER = "verifier"
    CLIENT = "client"

class CostCategory(Enum):
    """Categories of costs for ROI analysis"""
    HARDWARE = "hardware"
    ELECTRICITY = "electricity"
    NETWORK = "network"
    MAINTENANCE = "maintenance"
    OPPORTUNITY = "opportunity"
    STAKE = "stake"

@dataclass
class CostStructure:
    """Cost structure for economic analysis"""
    hardware_cost: float = 0.0  # USD per hour
    electricity_cost: float = 0.0  # USD per hour
    network_cost: float = 0.0  # USD per hour
    maintenance_cost: float = 0.0  # USD per hour
    opportunity_cost: float = 0.0  # USD per hour (alternative use of resources)
    stake_cost: float = 0.0  # USD staked (opportunity cost of capital)
    
    def total_operational_cost_per_hour(self) -> float:
        """Calculate total operational cost per hour"""
        return (self.hardware_cost + self.electricity_cost + 
                self.network_cost + self.maintenance_cost + self.opportunity_cost)
    
    def total_cost_including_stake(self, stake_apy: float = 0.05) -> float:
        """Calculate total cost including opportunity cost of stake"""
        stake_hourly_cost = (self.stake_cost * stake_apy) / (365 * 24)
        return self.total_operational_cost_per_hour() + stake_hourly_cost

@dataclass
class RevenueStream:
    """Revenue stream analysis"""
    block_rewards: float = 0.0  # USD per hour from block rewards
    task_fees: float = 0.0  # USD per hour from task fees
    performance_bonuses: float = 0.0  # USD per hour from performance bonuses
    network_incentives: float = 0.0  # USD per hour from network incentives
    staking_rewards: float = 0.0  # USD per hour from staking
    
    def total_revenue_per_hour(self) -> float:
        """Calculate total revenue per hour"""
        return (self.block_rewards + self.task_fees + self.performance_bonuses + 
                self.network_incentives + self.staking_rewards)

@dataclass
class ROIMetrics:
    """ROI calculation results"""
    hourly_profit: float
    daily_profit: float
    monthly_profit: float
    annual_profit: float
    roi_percentage: float
    payback_period_days: float
    break_even_point: float
    risk_adjusted_roi: float
    
    def is_profitable(self) -> bool:
        """Check if the operation is profitable"""
        return self.hourly_profit > 0

@dataclass
class NetworkEconomics:
    """Network-wide economic metrics"""
    total_network_value: float
    total_stake_value: float
    daily_transaction_volume: float
    network_utilization: float
    average_task_fee: float
    network_growth_rate: float
    
    def network_health_score(self) -> float:
        """Calculate network health score (0-1)"""
        utilization_score = min(1.0, self.network_utilization)
        growth_score = min(1.0, max(0.0, self.network_growth_rate))
        value_score = min(1.0, self.total_network_value / 1000000)  # Normalize to $1M
        
        return (utilization_score + growth_score + value_score) / 3

class ROIAnalyzer:
    """Comprehensive ROI analysis and economic modeling"""
    
    def __init__(self):
        self.baseline_costs = self._initialize_baseline_costs()
        self.market_rates = self._initialize_market_rates()
        self.historical_data = []
    
    def _initialize_baseline_costs(self) -> Dict[ParticipantRole, CostStructure]:
        """Initialize baseline cost structures for different roles"""
        return {
            ParticipantRole.MINER: CostStructure(
                hardware_cost=2.50,  # GPU server costs per hour
                electricity_cost=0.75,  # Power consumption
                network_cost=0.10,  # Bandwidth costs
                maintenance_cost=0.25,  # Hardware maintenance
                opportunity_cost=0.50,  # Alternative GPU rental income
                stake_cost=1000.0  # Typical miner stake
            ),
            ParticipantRole.SUPERVISOR: CostStructure(
                hardware_cost=0.50,  # Lower hardware requirements
                electricity_cost=0.15,  # Lower power consumption
                network_cost=0.20,  # Higher network requirements
                maintenance_cost=0.05,  # Lower maintenance
                opportunity_cost=0.10,  # Lower opportunity cost
                stake_cost=500.0  # Typical supervisor stake
            ),
            ParticipantRole.EVALUATOR: CostStructure(
                hardware_cost=0.75,  # Moderate hardware requirements
                electricity_cost=0.25,  # Moderate power consumption
                network_cost=0.15,  # Moderate network requirements
                maintenance_cost=0.10,  # Moderate maintenance
                opportunity_cost=0.20,  # Moderate opportunity cost
                stake_cost=250.0  # Typical evaluator stake
            ),
            ParticipantRole.VERIFIER: CostStructure(
                hardware_cost=1.00,  # Needs to re-run ML iterations
                electricity_cost=0.35,  # Moderate power consumption
                network_cost=0.10,  # Lower network requirements
                maintenance_cost=0.15,  # Moderate maintenance
                opportunity_cost=0.25,  # Moderate opportunity cost
                stake_cost=200.0  # Typical verifier stake
            ),
            ParticipantRole.CLIENT: CostStructure(
                hardware_cost=0.00,  # No hardware requirements
                electricity_cost=0.00,  # No power consumption
                network_cost=0.05,  # Minimal network requirements
                maintenance_cost=0.00,  # No maintenance
                opportunity_cost=0.00,  # No opportunity cost for clients
                stake_cost=0.0  # Clients don't stake
            )
        }
    
    def _initialize_market_rates(self) -> Dict[str, float]:
        """Initialize market rates and benchmarks"""
        return {
            'btc_mining_roi': 0.18,  # 18% annual ROI for Bitcoin mining
            'cloud_ml_cost': 3.50,  # USD per hour for cloud ML training
            'gpu_rental_rate': 2.00,  # USD per hour for GPU rental
            'risk_free_rate': 0.05,  # 5% annual risk-free rate
            'market_risk_premium': 0.06,  # 6% market risk premium
            'pouw_token_price': 25.0,  # USD per PAI token
            'network_fee_rate': 0.02  # 2% network fee
        }
    
    def calculate_participant_roi(self, 
                                role: ParticipantRole,
                                performance_metrics: Dict[str, float],
                                market_conditions: Dict[str, float],
                                time_horizon_days: int = 365) -> ROIMetrics:
        """Calculate ROI for a specific participant"""
        
        # Get baseline costs for the role
        costs = self.baseline_costs[role]
        
        # Adjust costs based on market conditions
        adjusted_costs = self._adjust_costs_for_market(costs, market_conditions)
        
        # Calculate revenue based on performance and role
        revenue = self._calculate_revenue_for_role(role, performance_metrics, market_conditions)
        
        # Calculate hourly profit
        hourly_profit = revenue.total_revenue_per_hour() - adjusted_costs.total_operational_cost_per_hour()
        
        # Include stake opportunity cost
        stake_opportunity_cost = (adjusted_costs.stake_cost * self.market_rates['risk_free_rate']) / (365 * 24)
        hourly_profit -= stake_opportunity_cost
        
        # Calculate time-based profits
        daily_profit = hourly_profit * 24
        monthly_profit = daily_profit * 30
        annual_profit = daily_profit * 365
        
        # Calculate ROI percentage
        initial_investment = adjusted_costs.stake_cost + (adjusted_costs.hardware_cost * 24 * 30)  # Hardware for 30 days
        roi_percentage = (annual_profit / max(initial_investment, 1)) * 100 if initial_investment > 0 else 0
        
        # Calculate payback period
        payback_period = initial_investment / max(daily_profit, 0.01) if daily_profit > 0 else float('inf')
        
        # Calculate break-even point
        break_even_point = adjusted_costs.total_operational_cost_per_hour()
        
        # Risk-adjusted ROI (using CAPM-style calculation)
        risk_adjustment = self.market_rates['risk_free_rate'] + (self.market_rates['market_risk_premium'] * 1.2)  # Beta = 1.2 for crypto
        risk_adjusted_roi = roi_percentage - (risk_adjustment * 100)
        
        return ROIMetrics(
            hourly_profit=hourly_profit,
            daily_profit=daily_profit,
            monthly_profit=monthly_profit,
            annual_profit=annual_profit,
            roi_percentage=roi_percentage,
            payback_period_days=payback_period,
            break_even_point=break_even_point,
            risk_adjusted_roi=risk_adjusted_roi
        )
    
    def _adjust_costs_for_market(self, base_costs: CostStructure, market_conditions: Dict[str, float]) -> CostStructure:
        """Adjust costs based on current market conditions"""
        electricity_multiplier = market_conditions.get('electricity_price_multiplier', 1.0)
        hardware_multiplier = market_conditions.get('hardware_price_multiplier', 1.0)
        network_multiplier = market_conditions.get('network_price_multiplier', 1.0)
        
        return CostStructure(
            hardware_cost=base_costs.hardware_cost * hardware_multiplier,
            electricity_cost=base_costs.electricity_cost * electricity_multiplier,
            network_cost=base_costs.network_cost * network_multiplier,
            maintenance_cost=base_costs.maintenance_cost,
            opportunity_cost=base_costs.opportunity_cost,
            stake_cost=base_costs.stake_cost
        )
    
    def _calculate_revenue_for_role(self, 
                                  role: ParticipantRole, 
                                  performance_metrics: Dict[str, float],
                                  market_conditions: Dict[str, float]) -> RevenueStream:
        """Calculate revenue streams for a specific role"""
        
        base_block_reward = market_conditions.get('base_block_reward', 12.5)
        task_volume = market_conditions.get('hourly_task_volume', 10)
        average_task_fee = market_conditions.get('average_task_fee', 25.0)
        performance_score = performance_metrics.get('performance_score', 0.8)
        
        if role == ParticipantRole.MINER:
            # Miners get block rewards and task fees
            block_rewards = base_block_reward * performance_score * 0.6  # 60% of blocks
            task_fees = (task_volume * average_task_fee * 0.7) * performance_score  # 70% of task fees
            performance_bonuses = task_fees * 0.1 * performance_score  # 10% bonus for high performance
            
            return RevenueStream(
                block_rewards=block_rewards,
                task_fees=task_fees,
                performance_bonuses=performance_bonuses,
                network_incentives=0.0,
                staking_rewards=0.0
            )
        
        elif role == ParticipantRole.SUPERVISOR:
            # Supervisors get task fees and network incentives
            task_fees = (task_volume * average_task_fee * 0.2) * performance_score  # 20% of task fees
            network_incentives = task_volume * 1.0 * performance_score  # Network coordination incentives
            
            return RevenueStream(
                block_rewards=0.0,
                task_fees=task_fees,
                performance_bonuses=0.0,
                network_incentives=network_incentives,
                staking_rewards=0.0
            )
        
        elif role == ParticipantRole.EVALUATOR:
            # Evaluators get task fees
            task_fees = (task_volume * average_task_fee * 0.05) * performance_score  # 5% of task fees
            performance_bonuses = task_fees * 0.2 * performance_score  # 20% bonus for accurate evaluation
            
            return RevenueStream(
                block_rewards=0.0,
                task_fees=task_fees,
                performance_bonuses=performance_bonuses,
                network_incentives=0.0,
                staking_rewards=0.0
            )
        
        elif role == ParticipantRole.VERIFIER:
            # Verifiers get task fees and network incentives
            task_fees = (task_volume * average_task_fee * 0.05) * performance_score  # 5% of task fees
            network_incentives = task_volume * 0.5 * performance_score  # Verification incentives
            
            return RevenueStream(
                block_rewards=0.0,
                task_fees=task_fees,
                performance_bonuses=0.0,
                network_incentives=network_incentives,
                staking_rewards=0.0
            )
        
        else:  # CLIENT
            return RevenueStream()  # Clients don't earn revenue
    
    def compare_with_alternatives(self, 
                                roi_metrics: ROIMetrics, 
                                role: ParticipantRole) -> Dict[str, Any]:
        """Compare PoUW ROI with alternative investments/activities"""
        
        comparisons = {}
        
        if role == ParticipantRole.MINER:
            # Compare with Bitcoin mining
            btc_annual_roi = self.market_rates['btc_mining_roi'] * 100
            comparisons['bitcoin_mining'] = {
                'annual_roi': btc_annual_roi,
                'advantage': roi_metrics.roi_percentage - btc_annual_roi,
                'is_better': roi_metrics.roi_percentage > btc_annual_roi
            }
            
            # Compare with GPU rental
            gpu_rental_hourly = self.market_rates['gpu_rental_rate']
            gpu_rental_annual = gpu_rental_hourly * 24 * 365
            comparisons['gpu_rental'] = {
                'annual_revenue': gpu_rental_annual,
                'advantage': roi_metrics.annual_profit - gpu_rental_annual,
                'is_better': roi_metrics.annual_profit > gpu_rental_annual
            }
        
        # Compare with risk-free investment
        risk_free_annual = self.market_rates['risk_free_rate'] * 100
        comparisons['risk_free_investment'] = {
            'annual_roi': risk_free_annual,
            'risk_premium': roi_metrics.roi_percentage - risk_free_annual,
            'is_better': roi_metrics.roi_percentage > risk_free_annual
        }
        
        return comparisons
    
    def simulate_network_economics(self, 
                                 network_params: Dict[str, Any],
                                 simulation_days: int = 365) -> Dict[str, Any]:
        """Simulate network economics over time"""
        
        initial_participants = network_params.get('initial_participants', {
            ParticipantRole.MINER: 100,
            ParticipantRole.SUPERVISOR: 20,
            ParticipantRole.EVALUATOR: 30,
            ParticipantRole.VERIFIER: 40,
            ParticipantRole.CLIENT: 200
        })
        
        initial_task_volume = network_params.get('daily_task_volume', 240)  # 10 tasks per hour
        growth_rate = network_params.get('growth_rate', 0.002)  # 0.2% daily growth
        
        simulation_results = []
        current_participants = initial_participants.copy()
        current_task_volume = initial_task_volume
        
        for day in range(simulation_days):
            # Calculate daily metrics
            total_network_stake = sum(
                count * self.baseline_costs[role].stake_cost 
                for role, count in current_participants.items()
                if role != ParticipantRole.CLIENT
            )
            
            daily_revenue = current_task_volume * self.market_rates.get('average_task_fee', 25.0)
            network_utilization = min(1.0, current_task_volume / (current_participants[ParticipantRole.MINER] * 24))
            
            network_economics = NetworkEconomics(
                total_network_value=total_network_stake * self.market_rates['pouw_token_price'],
                total_stake_value=total_network_stake,
                daily_transaction_volume=daily_revenue,
                network_utilization=network_utilization,
                average_task_fee=self.market_rates.get('average_task_fee', 25.0),
                network_growth_rate=growth_rate
            )
            
            simulation_results.append({
                'day': day,
                'participants': current_participants.copy(),
                'task_volume': current_task_volume,
                'network_economics': network_economics,
                'network_health': network_economics.network_health_score()
            })
            
            # Update for next day
            current_task_volume *= (1 + growth_rate)
            
            # Participants join/leave based on profitability
            if network_economics.network_health_score() > 0.7:
                for role in [ParticipantRole.MINER, ParticipantRole.SUPERVISOR, ParticipantRole.EVALUATOR, ParticipantRole.VERIFIER]:
                    current_participants[role] = int(current_participants[role] * (1 + growth_rate * 0.5))
        
        return {
            'simulation_results': simulation_results,
            'final_network_value': simulation_results[-1]['network_economics'].total_network_value,
            'average_network_health': statistics.mean([r['network_health'] for r in simulation_results]),
            'total_growth': (simulation_results[-1]['task_volume'] / initial_task_volume - 1) * 100
        }
    
    def calculate_client_savings(self, task_requirements: Dict[str, Any]) -> Dict[str, float]:
        """Calculate cost savings for clients using PoUW vs cloud services"""
        
        # Estimate cloud ML costs
        estimated_hours = task_requirements.get('estimated_hours', 1.0)
        gpu_requirements = task_requirements.get('gpu_count', 1)
        complexity_multiplier = task_requirements.get('complexity_multiplier', 1.0)
        
        cloud_cost = (self.market_rates['cloud_ml_cost'] * 
                     estimated_hours * gpu_requirements * complexity_multiplier)
        
        # Estimate PoUW costs
        base_task_fee = self.market_rates.get('average_task_fee', 25.0)
        pouw_cost = base_task_fee * complexity_multiplier
        
        savings = cloud_cost - pouw_cost
        savings_percentage = (savings / cloud_cost) * 100 if cloud_cost > 0 else 0
        
        return {
            'cloud_cost': cloud_cost,
            'pouw_cost': pouw_cost,
            'savings': savings,
            'savings_percentage': savings_percentage,
            'is_cheaper': savings > 0
        }
    
    def generate_profitability_report(self, 
                                    role: ParticipantRole,
                                    performance_metrics: Dict[str, float],
                                    market_conditions: Dict[str, float]) -> Dict[str, Any]:
        """Generate comprehensive profitability report"""
        
        roi_metrics = self.calculate_participant_roi(role, performance_metrics, market_conditions)
        comparisons = self.compare_with_alternatives(roi_metrics, role)
        
        return {
            'role': role.value,
            'roi_metrics': roi_metrics,
            'profitability_assessment': {
                'is_profitable': roi_metrics.is_profitable(),
                'profitability_rating': self._get_profitability_rating(roi_metrics),
                'risk_level': self._assess_risk_level(roi_metrics, role),
                'recommendation': self._generate_recommendation(roi_metrics, role)
            },
            'cost_breakdown': self.baseline_costs[role],
            'market_comparisons': comparisons,
            'sensitivity_analysis': self._perform_sensitivity_analysis(role, performance_metrics, market_conditions)
        }
    
    def _get_profitability_rating(self, roi_metrics: ROIMetrics) -> str:
        """Get profitability rating based on ROI"""
        if roi_metrics.roi_percentage >= 50:
            return "Excellent"
        elif roi_metrics.roi_percentage >= 25:
            return "Good"
        elif roi_metrics.roi_percentage >= 10:
            return "Fair"
        elif roi_metrics.roi_percentage >= 0:
            return "Marginal"
        else:
            return "Unprofitable"
    
    def _assess_risk_level(self, roi_metrics: ROIMetrics, role: ParticipantRole) -> str:
        """Assess risk level for the investment"""
        if role == ParticipantRole.MINER:
            if roi_metrics.payback_period_days > 365:
                return "High"
            elif roi_metrics.payback_period_days > 180:
                return "Medium"
            else:
                return "Low"
        else:
            if roi_metrics.roi_percentage < 10:
                return "High"
            elif roi_metrics.roi_percentage < 25:
                return "Medium"
            else:
                return "Low"
    
    def _generate_recommendation(self, roi_metrics: ROIMetrics, role: ParticipantRole) -> str:
        """Generate investment recommendation"""
        if roi_metrics.roi_percentage > 20 and roi_metrics.payback_period_days < 365:
            return "Strongly Recommended"
        elif roi_metrics.roi_percentage > 10 and roi_metrics.payback_period_days < 540:
            return "Recommended"
        elif roi_metrics.roi_percentage > 0:
            return "Consider with Caution"
        else:
            return "Not Recommended"
    
    def _perform_sensitivity_analysis(self, 
                                    role: ParticipantRole,
                                    performance_metrics: Dict[str, float],
                                    market_conditions: Dict[str, float]) -> Dict[str, Any]:
        """Perform sensitivity analysis on key variables"""
        
        base_roi = self.calculate_participant_roi(role, performance_metrics, market_conditions)
        
        # Test sensitivity to key variables
        sensitivity_results = {}
        
        # Task volume sensitivity
        for multiplier in [0.5, 0.75, 1.25, 1.5]:
            modified_conditions = market_conditions.copy()
            modified_conditions['hourly_task_volume'] = modified_conditions.get('hourly_task_volume', 10) * multiplier
            
            modified_roi = self.calculate_participant_roi(role, performance_metrics, modified_conditions)
            sensitivity_results[f'task_volume_{multiplier}x'] = {
                'roi_change': modified_roi.roi_percentage - base_roi.roi_percentage,
                'profit_change': modified_roi.annual_profit - base_roi.annual_profit
            }
        
        # Performance sensitivity
        for performance_level in [0.6, 0.8, 1.0, 1.2]:
            modified_performance = performance_metrics.copy()
            modified_performance['performance_score'] = performance_level
            
            modified_roi = self.calculate_participant_roi(role, modified_performance, market_conditions)
            sensitivity_results[f'performance_{performance_level}x'] = {
                'roi_change': modified_roi.roi_percentage - base_roi.roi_percentage,
                'profit_change': modified_roi.annual_profit - base_roi.annual_profit
            }
        
        return sensitivity_results

# Convenience functions for easy integration

def analyze_miner_profitability(hardware_specs: Dict[str, Any], 
                              market_conditions: Dict[str, Any],
                              performance_score: float = 0.8) -> Dict[str, Any]:
    """Analyze miner profitability with given hardware and market conditions"""
    analyzer = ROIAnalyzer()
    
    performance_metrics = {'performance_score': performance_score}
    
    return analyzer.generate_profitability_report(
        ParticipantRole.MINER, 
        performance_metrics, 
        market_conditions
    )

def compare_pouw_vs_bitcoin_mining(investment_amount: float) -> Dict[str, Any]:
    """Compare PoUW mining ROI vs Bitcoin mining"""
    analyzer = ROIAnalyzer()
    
    # Standard market conditions
    market_conditions = {
        'base_block_reward': 12.5,
        'hourly_task_volume': 10,
        'average_task_fee': 25.0,
        'electricity_price_multiplier': 1.0,
        'hardware_price_multiplier': 1.0
    }
    
    performance_metrics = {'performance_score': 0.85}
    
    pouw_roi = analyzer.calculate_participant_roi(
        ParticipantRole.MINER, 
        performance_metrics, 
        market_conditions
    )
    
    btc_annual_roi = analyzer.market_rates['btc_mining_roi'] * 100
    
    return {
        'investment_amount': investment_amount,
        'pouw_annual_roi': pouw_roi.roi_percentage,
        'bitcoin_annual_roi': btc_annual_roi,
        'pouw_advantage': pouw_roi.roi_percentage - btc_annual_roi,
        'pouw_annual_profit': pouw_roi.annual_profit,
        'bitcoin_annual_profit': investment_amount * analyzer.market_rates['btc_mining_roi'],
        'recommendation': 'PoUW' if pouw_roi.roi_percentage > btc_annual_roi else 'Bitcoin'
    }

def calculate_network_sustainability(participants: Dict[str, int], 
                                   task_volume: int) -> Dict[str, Any]:
    """Calculate network sustainability metrics"""
    analyzer = ROIAnalyzer()
    
    network_params = {
        'initial_participants': {
            ParticipantRole.MINER: participants.get('miners', 100),
            ParticipantRole.SUPERVISOR: participants.get('supervisors', 20),
            ParticipantRole.EVALUATOR: participants.get('evaluators', 30),
            ParticipantRole.VERIFIER: participants.get('verifiers', 40),
            ParticipantRole.CLIENT: participants.get('clients', 200)
        },
        'daily_task_volume': task_volume,
        'growth_rate': 0.002
    }
    
    simulation = analyzer.simulate_network_economics(network_params, 90)  # 3 months
    
    return {
        'network_health_score': simulation['average_network_health'],
        'projected_growth': simulation['total_growth'],
        'final_network_value': simulation['final_network_value'],
        'sustainability_rating': 'High' if simulation['average_network_health'] > 0.7 else 
                                'Medium' if simulation['average_network_health'] > 0.5 else 'Low'
    }
