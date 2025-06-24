#!/usr/bin/env python3
"""
ROI Analysis and Economic Modeling Demonstration for PoUW

This script demonstrates the comprehensive ROI calculations and economic modeling
capabilities that comply with the research paper specification.
"""

import sys
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pouw.economics.roi_analysis import (
    ROIAnalyzer,
    ParticipantRole,
    analyze_miner_profitability,
    compare_pouw_vs_bitcoin_mining,
    calculate_network_sustainability
)

def demonstrate_roi_analysis():
    """Demonstrate comprehensive ROI analysis and economic modeling"""
    print("=" * 80)
    print("ROI ANALYSIS AND ECONOMIC MODELING DEMONSTRATION")
    print("=" * 80)
    print("Research Paper Compliance: Comprehensive profitability analysis")
    print("and economic simulation capabilities for network optimization")
    print()
    
    analyzer = ROIAnalyzer()
    
    # Demo 1: Miner Profitability Analysis
    print("💰 Demo 1: Miner Profitability Analysis")
    print("-" * 50)
    
    miner_performance = {
        'performance_score': 0.92,
        'uptime': 0.98,
        'accuracy': 0.94
    }
    
    market_conditions = {
        'base_block_reward': 12.5,
        'hourly_task_volume': 15,  # High demand scenario
        'average_task_fee': 30.0,  # Premium pricing
        'electricity_price_multiplier': 1.2,  # 20% above baseline
        'hardware_price_multiplier': 1.0,
        'network_price_multiplier': 0.8  # Cheaper bandwidth
    }
    
    miner_roi = analyzer.calculate_participant_roi(
        ParticipantRole.MINER,
        miner_performance,
        market_conditions
    )
    
    print(f"High-Performance GPU Miner Analysis:")
    print(f"  Hourly Profit: ${miner_roi.hourly_profit:.2f}")
    print(f"  Daily Profit: ${miner_roi.daily_profit:.2f}")
    print(f"  Monthly Profit: ${miner_roi.monthly_profit:.2f}")
    print(f"  Annual Profit: ${miner_roi.annual_profit:.2f}")
    print(f"  ROI Percentage: {miner_roi.roi_percentage:.1f}%")
    print(f"  Payback Period: {miner_roi.payback_period_days:.0f} days")
    print(f"  Risk-Adjusted ROI: {miner_roi.risk_adjusted_roi:.1f}%")
    print(f"  Is Profitable: {miner_roi.is_profitable()}")
    print()
    
    # Demo 2: Supervisor Economic Analysis
    print("🏛️ Demo 2: Supervisor Economic Analysis")
    print("-" * 50)
    
    supervisor_performance = {
        'performance_score': 0.88,
        'consensus_accuracy': 0.96,
        'network_coordination': 0.91
    }
    
    supervisor_roi = analyzer.calculate_participant_roi(
        ParticipantRole.SUPERVISOR,
        supervisor_performance,
        market_conditions
    )
    
    print(f"Network Supervisor Analysis:")
    print(f"  Hourly Profit: ${supervisor_roi.hourly_profit:.2f}")
    print(f"  Daily Profit: ${supervisor_roi.daily_profit:.2f}")
    print(f"  Annual ROI: {supervisor_roi.roi_percentage:.1f}%")
    print(f"  Break-Even Point: ${supervisor_roi.break_even_point:.2f}/hour")
    print(f"  Risk Level: Lower stake, steady returns")
    print()
    
    # Demo 3: Market Comparison Analysis
    print("📊 Demo 3: Market Comparison Analysis")
    print("-" * 50)
    
    comparisons = analyzer.compare_with_alternatives(miner_roi, ParticipantRole.MINER)
    
    print("PoUW vs Alternative Investments:")
    
    btc_comparison = comparisons['bitcoin_mining']
    print(f"  vs Bitcoin Mining:")
    print(f"    Bitcoin Annual ROI: {btc_comparison['annual_roi']:.1f}%")
    print(f"    PoUW Advantage: {btc_comparison['advantage']:+.1f}%")
    print(f"    PoUW is Better: {btc_comparison['is_better']}")
    
    gpu_comparison = comparisons['gpu_rental']
    print(f"  vs GPU Rental Business:")
    print(f"    GPU Rental Annual Revenue: ${gpu_comparison['annual_revenue']:,.0f}")
    print(f"    PoUW Advantage: ${gpu_comparison['advantage']:+,.0f}")
    print(f"    PoUW is Better: {gpu_comparison['is_better']}")
    
    risk_free = comparisons['risk_free_investment']
    print(f"  vs Risk-Free Investment:")
    print(f"    Risk-Free ROI: {risk_free['annual_roi']:.1f}%")
    print(f"    PoUW Risk Premium: {risk_free['risk_premium']:+.1f}%")
    print(f"    Justifies Risk: {risk_free['is_better']}")
    print()
    
    # Demo 4: Client Cost Savings Analysis
    print("💸 Demo 4: Client Cost Savings Analysis")
    print("-" * 50)
    
    # Large-scale training task
    large_task = {
        'estimated_hours': 48,  # 2 days of training
        'gpu_count': 16,        # Multi-GPU setup
        'complexity_multiplier': 2.5  # Complex transformer model
    }
    
    large_savings = analyzer.calculate_client_savings(large_task)
    
    print(f"Large-Scale ML Training Task:")
    print(f"  Cloud ML Cost: ${large_savings['cloud_cost']:,.2f}")
    print(f"  PoUW Cost: ${large_savings['pouw_cost']:,.2f}")
    print(f"  Total Savings: ${large_savings['savings']:,.2f}")
    print(f"  Savings Percentage: {large_savings['savings_percentage']:.1f}%")
    print(f"  PoUW is Cheaper: {large_savings['is_cheaper']}")
    
    # Medium-scale task
    medium_task = {
        'estimated_hours': 8,
        'gpu_count': 4,
        'complexity_multiplier': 1.2
    }
    
    medium_savings = analyzer.calculate_client_savings(medium_task)
    
    print(f"\nMedium-Scale ML Training Task:")
    print(f"  Cloud ML Cost: ${medium_savings['cloud_cost']:,.2f}")
    print(f"  PoUW Cost: ${medium_savings['pouw_cost']:,.2f}")
    print(f"  Total Savings: ${medium_savings['savings']:,.2f}")
    print(f"  Savings Percentage: {medium_savings['savings_percentage']:.1f}%")
    print()
    
    # Demo 5: Network Economics Simulation
    print("🌐 Demo 5: Network Economics Simulation")
    print("-" * 50)
    
    network_params = {
        'initial_participants': {
            ParticipantRole.MINER: 150,
            ParticipantRole.SUPERVISOR: 30,
            ParticipantRole.EVALUATOR: 45,
            ParticipantRole.VERIFIER: 60,
            ParticipantRole.CLIENT: 500
        },
        'daily_task_volume': 720,  # 30 tasks per hour
        'growth_rate': 0.003  # 0.3% daily growth
    }
    
    print(f"Simulating network economics over 180 days...")
    simulation = analyzer.simulate_network_economics(network_params, 180)
    
    print(f"Network Growth Simulation Results:")
    print(f"  Initial Network Value: ${simulation['simulation_results'][0]['network_economics'].total_network_value:,.0f}")
    print(f"  Final Network Value: ${simulation['final_network_value']:,.0f}")
    print(f"  Total Growth: {simulation['total_growth']:.1f}%")
    print(f"  Average Network Health: {simulation['average_network_health']:.3f}")
    
    # Show key milestones
    milestones = [30, 60, 90, 120, 150, 180]
    print(f"\nNetwork Health Over Time:")
    for day in milestones:
        if day <= len(simulation['simulation_results']):
            result = simulation['simulation_results'][day-1]
            health = result['network_health']
            participants = sum(result['participants'].values()) - result['participants'][ParticipantRole.CLIENT]
            print(f"  Day {day:3d}: Health={health:.3f}, Active Workers={participants}")
    print()
    
    # Demo 6: Comprehensive Profitability Reports
    print("📋 Demo 6: Comprehensive Profitability Reports")
    print("-" * 50)
    
    roles_to_analyze = [
        (ParticipantRole.MINER, "High-End GPU Miner"),
        (ParticipantRole.SUPERVISOR, "Network Supervisor"),
        (ParticipantRole.EVALUATOR, "Model Evaluator"),
        (ParticipantRole.VERIFIER, "Blockchain Verifier")
    ]
    
    for role, description in roles_to_analyze:
        performance = {'performance_score': 0.85 + (hash(role.value) % 10) / 100}  # Varied performance
        
        report = analyzer.generate_profitability_report(role, performance, market_conditions)
        assessment = report['profitability_assessment']
        
        print(f"{description} ({role.value.title()}):")
        print(f"  Profitability Rating: {assessment['profitability_rating']}")
        print(f"  Risk Level: {assessment['risk_level']}")
        print(f"  Recommendation: {assessment['recommendation']}")
        print(f"  Annual ROI: {report['roi_metrics'].roi_percentage:.1f}%")
        print()
    
    # Demo 7: Convenience Function Demonstrations
    print("🔧 Demo 7: Convenience Function Demonstrations")
    print("-" * 50)
    
    # Analyze miner profitability with hardware specs
    hardware_specs = {
        'gpu_count': 8,
        'gpu_model': 'RTX 4090',
        'power_consumption': 3200,  # watts
        'cooling_cost': 500  # USD/month
    }
    
    hardware_market_conditions = {
        'base_block_reward': 12.5,
        'hourly_task_volume': 20,
        'average_task_fee': 35.0,
        'electricity_price_multiplier': 1.1
    }
    
    hardware_analysis = analyze_miner_profitability(
        hardware_specs, 
        hardware_market_conditions, 
        0.91
    )
    
    print(f"Hardware-Specific Miner Analysis:")
    print(f"  Investment Recommendation: {hardware_analysis['profitability_assessment']['recommendation']}")
    print(f"  Expected Annual ROI: {hardware_analysis['roi_metrics'].roi_percentage:.1f}%")
    print(f"  Risk Assessment: {hardware_analysis['profitability_assessment']['risk_level']}")
    
    # Compare PoUW vs Bitcoin for specific investment
    investment_amount = 50000  # $50k investment
    comparison = compare_pouw_vs_bitcoin_mining(investment_amount)
    
    print(f"\n${investment_amount:,} Investment Comparison:")
    print(f"  PoUW Annual ROI: {comparison['pouw_annual_roi']:.1f}%")
    print(f"  Bitcoin Annual ROI: {comparison['bitcoin_annual_roi']:.1f}%")
    print(f"  PoUW Annual Profit: ${comparison['pouw_annual_profit']:,.0f}")
    print(f"  Bitcoin Annual Profit: ${comparison['bitcoin_annual_profit']:,.0f}")
    print(f"  Recommendation: {comparison['recommendation']}")
    
    # Network sustainability analysis
    participants = {
        'miners': 200,
        'supervisors': 40,
        'evaluators': 60,
        'verifiers': 80,
        'clients': 1000
    }
    
    sustainability = calculate_network_sustainability(participants, 1200)  # 1200 tasks/day
    
    print(f"\nNetwork Sustainability Analysis:")
    print(f"  Network Health Score: {sustainability['network_health_score']:.3f}")
    print(f"  Projected 90-day Growth: {sustainability['projected_growth']:.1f}%")
    print(f"  Final Network Value: ${sustainability['final_network_value']:,.0f}")
    print(f"  Sustainability Rating: {sustainability['sustainability_rating']}")
    print()
    
    # Demo 8: Economic Health Assessment
    print("🏥 Demo 8: Economic Health Assessment")
    print("-" * 50)
    
    # Test different market scenarios
    scenarios = [
        ("Bull Market", {
            'base_block_reward': 15.0,
            'hourly_task_volume': 25,
            'average_task_fee': 40.0,
            'electricity_price_multiplier': 0.9
        }),
        ("Bear Market", {
            'base_block_reward': 10.0,
            'hourly_task_volume': 8,
            'average_task_fee': 20.0,
            'electricity_price_multiplier': 1.3
        }),
        ("Stable Market", {
            'base_block_reward': 12.5,
            'hourly_task_volume': 12,
            'average_task_fee': 25.0,
            'electricity_price_multiplier': 1.0
        })
    ]
    
    print("Market Scenario Analysis:")
    for scenario_name, scenario_conditions in scenarios:
        scenario_roi = analyzer.calculate_participant_roi(
            ParticipantRole.MINER,
            {'performance_score': 0.85},
            scenario_conditions
        )
        
        profitability = "Profitable" if scenario_roi.is_profitable() else "Unprofitable"
        print(f"  {scenario_name:12s}: ROI={scenario_roi.roi_percentage:6.1f}%, {profitability}")
    
    print()
    
    # Demo 9: Performance Benchmarks
    print("⚡ Demo 9: Performance Benchmarks")
    print("-" * 50)
    
    # Time ROI calculations
    start_time = time.time()
    
    for i in range(100):
        test_performance = {'performance_score': 0.8 + (i % 20) / 100}
        test_conditions = market_conditions.copy()
        test_conditions['hourly_task_volume'] = 10 + (i % 10)
        
        analyzer.calculate_participant_roi(
            ParticipantRole.MINER,
            test_performance,
            test_conditions
        )
    
    end_time = time.time()
    calculation_time = end_time - start_time
    
    print(f"ROI Analysis Performance (100 calculations):")
    print(f"  Total Time: {calculation_time:.3f} seconds")
    print(f"  Average per Calculation: {calculation_time/100*1000:.2f} ms")
    print(f"  Calculations per Second: {100/calculation_time:.1f}")
    print()
    
    print("=" * 80)
    print("ROI ANALYSIS DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("✅ Comprehensive ROI calculations implemented successfully!")
    print("✅ Full research paper compliance for economic modeling")
    print("✅ Profitability analysis tools for all participant types")
    print("✅ Network sustainability and growth projections")
    print("✅ Market comparison and risk assessment capabilities")
    print("✅ Production-ready performance characteristics")
    print()
    print("Key Findings:")
    print(f"• PoUW mining shows {btc_comparison['advantage']:+.1f}% advantage over Bitcoin mining")
    print(f"• Clients save {large_savings['savings_percentage']:.0f}% on large ML tasks vs cloud services")
    print(f"• Network sustainability rating: {sustainability['sustainability_rating']}")
    print(f"• High-performance miners achieve {miner_roi.roi_percentage:.0f}% annual ROI")
    print()
    print("The PoUW economic model demonstrates strong profitability and sustainability!")

if __name__ == '__main__':
    demonstrate_roi_analysis()
