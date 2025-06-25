"""
Test script to demonstrate the refactored economics module.

This script shows how the new modular structure improves clarity
and maintainability compared to the original implementation.
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, project_root)

# Import refactored components
from pouw.economics.economic_system import EconomicSystem
from pouw.economics.staking import NodeRole
from pouw.economics.pricing import MarketCondition


def demonstrate_refactored_economics():
    """Demonstrate the refactored economics module"""

    print("=" * 60)
    print("REFACTORED ECONOMICS MODULE DEMONSTRATION")
    print("=" * 60)
    print()

    # Initialize the economic system
    print("🏗️  Initializing Economic System")
    print("-" * 40)
    economic_system = EconomicSystem(base_price=15.0)
    print("✅ Economic system initialized with modular components:")
    print("   - StakingManager: Handles ticket purchases and stake management")
    print("   - TaskMatcher: Manages worker-task assignments")
    print("   - RewardDistributor: Calculates and distributes rewards")
    print("   - DynamicPricingEngine: Adjusts prices based on market conditions")
    print()

    # Demonstrate staking
    print("🎫 Staking System Demo")
    print("-" * 40)

    # Buy tickets for different roles
    participants = [
        ("miner_001", NodeRole.MINER, 150.0, {"has_gpu": True, "model_types": ["mlp"]}),
        ("miner_002", NodeRole.MINER, 120.0, {"has_gpu": False, "model_types": ["cnn"]}),
        ("supervisor_001", NodeRole.SUPERVISOR, 100.0, {"experience": "high"}),
        ("evaluator_001", NodeRole.EVALUATOR, 80.0, {"domain": "computer_vision"}),
        ("verifier_001", NodeRole.VERIFIER, 60.0, {"validation_speed": "fast"}),
    ]

    for participant_id, role, stake, preferences in participants:
        try:
            ticket = economic_system.buy_ticket(participant_id, role, stake, preferences)
            print(f"✅ {participant_id}: {role.value} ticket purchased (Stake: {stake})")
        except Exception as e:
            print(f"❌ {participant_id}: Failed to purchase ticket - {e}")

    print()

    # Demonstrate market metrics and pricing
    print("📊 Market Analysis Demo")
    print("-" * 40)

    # Update market conditions
    economic_system.update_market_metrics(
        total_supply=5,  # 5 workers available
        total_demand=3,  # 3 tasks waiting
        recent_tasks=[],  # No recent tasks for simplicity
        network_stats={
            "active_nodes": 5,
            "total_nodes": 10,
            "completion_rate": 0.95,
            "average_quality": 0.88,
        },
    )

    # Get market condition
    market_condition = economic_system.pricing_engine.get_market_condition(
        economic_system.market_metrics
    )
    print(f"📈 Current market condition: {market_condition.value}")
    print(f"📊 Network utilization: {economic_system.market_metrics.network_utilization:.2%}")
    print(f"⏰ Peak hour multiplier: {economic_system.market_metrics.peak_hour_multiplier:.2f}")
    print()

    # Demonstrate pricing analytics
    print("💰 Dynamic Pricing Demo")
    print("-" * 40)

    # Create a mock task for pricing demonstration
    class MockTask:
        def __init__(self):
            self.task_id = "demo_task_001"
            self.complexity_score = 0.7
            self.fee = 0.0

        def get_required_miners(self):
            return 2

    demo_task = MockTask()

    # Calculate dynamic price
    dynamic_price = economic_system.pricing_engine.calculate_dynamic_price(
        demo_task, economic_system.market_metrics
    )

    demo_task.fee = dynamic_price

    print(f"💵 Base price: ${economic_system.pricing_engine.base_price:.2f}")
    print(f"💵 Dynamic price: ${dynamic_price:.2f}")
    print(
        f"📊 Price adjustment: {((dynamic_price / economic_system.pricing_engine.base_price) - 1) * 100:+.1f}%"
    )
    print()

    # Demonstrate task submission and worker assignment
    print("👥 Task Assignment Demo")
    print("-" * 40)

    try:
        selected_workers = economic_system.submit_task(demo_task)

        print(f"✅ Task {demo_task.task_id} submitted with fee: ${demo_task.fee:.2f}")
        for role, workers in selected_workers.items():
            if workers:
                worker_ids = [w.owner_id for w in workers]
                print(f"   {role.value}s assigned: {worker_ids}")
            else:
                print(f"   No {role.value}s available")

        print()

        # Demonstrate task completion and reward distribution
        print("🏆 Reward Distribution Demo")
        print("-" * 40)

        # Mock performance metrics
        performance_metrics = {"miner_001": 0.92, "miner_002": 0.85}

        # Complete the task
        rewards = economic_system.complete_task(
            demo_task.task_id,
            final_models={"best_model": "model_data"},
            performance_metrics=performance_metrics,
        )

        print(f"✅ Task {demo_task.task_id} completed")
        print("💰 Rewards distributed:")
        for participant, reward in rewards.items():
            print(f"   {participant}: ${reward:.2f}")

        total_rewards = sum(rewards.values())
        print(f"💵 Total rewards distributed: ${total_rewards:.2f}")
        print(f"💸 Platform fee: ${demo_task.fee - total_rewards:.2f}")

    except Exception as e:
        print(f"❌ Task submission failed: {e}")

    print()

    # Demonstrate network statistics
    print("📈 Network Statistics")
    print("-" * 40)

    network_stats = economic_system.get_network_stats()

    print("📊 Assignment Statistics:")
    assignment_stats = network_stats["assignment_stats"]
    for key, value in assignment_stats.items():
        if not key.endswith("_assigned") and not key.endswith("_free"):
            print(f"   {key.replace('_', ' ').title()}: {value}")

    print("\n💰 Reward Statistics:")
    reward_stats = network_stats["reward_stats"]
    for key, value in reward_stats.items():
        if key != "top_earners":
            print(f"   {key.replace('_', ' ').title()}: {value}")

    print(f"\n📊 Market Condition: {network_stats['market_condition']}")
    print()

    # Demonstrate economic health assessment
    print("🏥 Economic Health Assessment")
    print("-" * 40)

    health_report = economic_system.get_economic_health_report()

    print(f"💚 Overall Health Score: {health_report['overall_health_score']:.1f}/100")
    print("\n📊 Health Indicators:")
    for indicator, score in health_report["health_indicators"].items():
        status = "🟢" if score >= 75 else "🟡" if score >= 50 else "🔴"
        print(f"   {status} {indicator.replace('_', ' ').title()}: {score:.1f}/100")

    print("\n💡 Recommendations:")
    for recommendation in health_report["recommendations"]:
        print(f"   • {recommendation}")

    print()
    print("=" * 60)
    print("REFACTORING BENEFITS DEMONSTRATED:")
    print("✅ Clear separation of concerns (staking, pricing, rewards, matching)")
    print("✅ Modular components that can be used independently")
    print("✅ Improved readability and maintainability")
    print("✅ Better testing and debugging capabilities")
    print("✅ Easier to extend with new features")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_refactored_economics()
