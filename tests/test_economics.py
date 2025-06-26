"""
Unit tests for PoUW economic system.
"""

import pytest
import time
from pouw.economics import NodeRole, Ticket, StakePool, TaskMatcher, RewardScheme, RewardDistributor, EconomicSystem
from pouw.blockchain import MLTask


class TestTicket:
    """Test Ticket functionality"""

    def test_ticket_creation(self):
        """Test ticket creation"""
        ticket = Ticket(
            ticket_id="ticket_001",
            owner_id="user_001",
            role=NodeRole.MINER,
            stake_amount=100.0,
            preferences={"gpu": True},
            expiration_time=int(time.time()) + 3600,
        )

        assert ticket.ticket_id == "ticket_001"
        assert ticket.owner_id == "user_001"
        assert ticket.role == NodeRole.MINER
        assert ticket.stake_amount == 100.0
        assert not ticket.is_expired()

    def test_ticket_expiration(self):
        """Test ticket expiration"""
        # Create expired ticket
        ticket = Ticket(
            ticket_id="ticket_001",
            owner_id="user_001",
            role=NodeRole.MINER,
            stake_amount=100.0,
            preferences={},
            expiration_time=int(time.time()) - 3600,  # 1 hour ago
        )

        assert ticket.is_expired()

    def test_task_matching(self):
        """Test task compatibility scoring"""
        task = MLTask(
            task_id="task_001",
            model_type="mlp",
            architecture={"input_size": 784, "output_size": 10},
            optimizer={},
            stopping_criterion={},
            validation_strategy={},
            metrics=[],
            dataset_info={"size": 50000},
            performance_requirements={"gpu": True},
            fee=100.0,
            client_id="client_001",
        )

        # Good match ticket
        good_ticket = Ticket(
            "ticket_001",
            "user_001",
            NodeRole.MINER,
            100.0,
            {"model_types": ["mlp", "cnn"], "has_gpu": True, "max_dataset_size": 100000},
            int(time.time()) + 3600,
        )

        # Poor match ticket
        poor_ticket = Ticket(
            "ticket_002",
            "user_002",
            NodeRole.MINER,
            100.0,
            {"model_types": ["rnn"], "has_gpu": False, "max_dataset_size": 1000},
            int(time.time()) + 3600,
        )

        good_score = good_ticket.matches_task(task)
        poor_score = poor_ticket.matches_task(task)

        assert good_score > poor_score
        assert 0.0 <= good_score <= 1.0
        assert 0.0 <= poor_score <= 1.0


class TestStakePool:
    """Test StakePool functionality"""

    def test_stake_pool_creation(self):
        """Test stake pool creation"""
        pool = StakePool()

        assert len(pool.tickets) == 0
        assert len(pool.live_tickets) == 0
        assert pool.target_pool_size == 40960

    def test_add_ticket(self):
        """Test adding ticket to pool"""
        pool = StakePool()

        ticket = Ticket(
            "ticket_001", "user_001", NodeRole.MINER, 100.0, {}, int(time.time()) + 3600
        )

        pool.add_ticket(ticket)

        assert len(pool.tickets) == 1
        assert len(pool.live_tickets) == 1
        assert ticket.is_live
        assert "ticket_001" in pool.tickets

    def test_get_tickets_by_role(self):
        """Test filtering tickets by role"""
        pool = StakePool()

        miner_ticket = Ticket(
            "ticket_001", "user_001", NodeRole.MINER, 100.0, {}, int(time.time()) + 3600
        )

        supervisor_ticket = Ticket(
            "ticket_002", "user_002", NodeRole.SUPERVISOR, 50.0, {}, int(time.time()) + 3600
        )

        pool.add_ticket(miner_ticket)
        pool.add_ticket(supervisor_ticket)

        miners = pool.get_tickets_by_role(NodeRole.MINER)
        supervisors = pool.get_tickets_by_role(NodeRole.SUPERVISOR)
        evaluators = pool.get_tickets_by_role(NodeRole.EVALUATOR)

        assert len(miners) == 1
        assert len(supervisors) == 1
        assert len(evaluators) == 0
        assert miners[0].role == NodeRole.MINER
        assert supervisors[0].role == NodeRole.SUPERVISOR

    def test_ticket_price_calculation(self):
        """Test ticket price adjustment"""
        pool = StakePool()

        # Empty pool should have lower price
        base_price = pool.calculate_ticket_price()

        # Add many tickets to increase price
        for i in range(pool.target_pool_size + 1000):
            ticket = Ticket(
                f"ticket_{i}", f"user_{i}", NodeRole.MINER, 100.0, {}, int(time.time()) + 3600
            )
            pool.add_ticket(ticket)

        high_price = pool.calculate_ticket_price()

        assert high_price > base_price


class TestTaskMatcher:
    """Test TaskMatcher functionality"""

    def test_task_matcher_creation(self):
        """Test task matcher creation"""
        pool = StakePool()
        matcher = TaskMatcher(pool)

        assert matcher.stake_pool == pool

    def test_select_workers(self):
        """Test worker selection"""
        pool = StakePool()
        matcher = TaskMatcher(pool)

        # Add various tickets
        for i in range(5):
            miner_ticket = Ticket(
                f"miner_{i}",
                f"user_{i}",
                NodeRole.MINER,
                100.0,
                {"model_types": ["mlp"], "has_gpu": True},
                int(time.time()) + 3600,
            )
            pool.add_ticket(miner_ticket)

        for i in range(3):
            supervisor_ticket = Ticket(
                f"supervisor_{i}",
                f"supervisor_{i}",
                NodeRole.SUPERVISOR,
                50.0,
                {"storage_capacity": 1000000},
                int(time.time()) + 3600,
            )
            pool.add_ticket(supervisor_ticket)

        # Create task
        task = MLTask(
            "task_001",
            "mlp",
            {},
            {},
            {},
            {},
            [],
            {"size": 50000},
            {"gpu": True},
            100.0,
            "client_001",
        )

        # Select workers
        selected = matcher.select_workers(task)

        assert NodeRole.MINER in selected
        assert NodeRole.SUPERVISOR in selected
        assert len(selected[NodeRole.MINER]) <= 3
        assert len(selected[NodeRole.SUPERVISOR]) <= 2


class TestRewardScheme:
    """Test RewardScheme functionality"""

    def test_reward_calculation(self):
        """Test reward calculation"""
        scheme = RewardScheme()

        total_fee = 100.0
        performance_scores = {"miner_001": 0.6, "miner_002": 0.4}

        rewards = scheme.calculate_rewards(total_fee, performance_scores)

        # Check that miners get proportional rewards
        assert "miner_001" in rewards
        assert "miner_002" in rewards
        assert rewards["miner_001"] > rewards["miner_002"]

        # Check total miner rewards
        total_miner_rewards = sum(rewards.values())
        expected_miner_pool = total_fee * scheme.miner_percentage
        assert abs(total_miner_rewards - expected_miner_pool) < 0.01


class TestEconomicSystem:
    """Test complete EconomicSystem functionality"""

    def test_economic_system_creation(self):
        """Test economic system creation"""
        system = EconomicSystem()

        assert isinstance(system.staking_manager.stake_pool, StakePool)
        assert isinstance(system.task_matcher, TaskMatcher)
        assert isinstance(system.reward_distributor, RewardDistributor)
        assert len(system.active_tasks) == 0
        assert len(system.completed_tasks) == 0

    def test_buy_ticket(self):
        """Test ticket purchase"""
        system = EconomicSystem()

        preferences = {"gpu": True, "model_types": ["mlp"]}

        ticket = system.buy_ticket("user_001", NodeRole.MINER, 100.0, preferences)

        assert ticket.owner_id == "user_001"
        assert ticket.role == NodeRole.MINER
        assert ticket.stake_amount == 100.0
        assert ticket.preferences == preferences
        assert len(system.staking_manager.stake_pool.tickets) == 1

    def test_buy_ticket_insufficient_stake(self):
        """Test ticket purchase with insufficient stake"""
        system = EconomicSystem()

        # Try to buy with insufficient stake
        with pytest.raises(ValueError):
            system.buy_ticket("user_001", NodeRole.MINER, 1.0, {})

    def test_submit_task(self):
        """Test task submission and worker assignment"""
        system = EconomicSystem()

        # Add some tickets first
        system.buy_ticket("miner_001", NodeRole.MINER, 100.0, {"gpu": True})
        system.buy_ticket("supervisor_001", NodeRole.SUPERVISOR, 50.0, {})

        # Create and submit task
        task = MLTask(
            "task_001",
            "mlp",
            {},
            {},
            {},
            {},
            [],
            {"size": 50000},
            {"gpu": True},
            100.0,
            "client_001",
        )

        selected_workers = system.submit_task(task)

        assert "task_001" in system.active_tasks
        assert system.active_tasks["task_001"]["status"] == "active"
        assert NodeRole.MINER in selected_workers or NodeRole.SUPERVISOR in selected_workers

    def test_complete_task(self):
        """Test task completion and reward distribution"""
        system = EconomicSystem()

        # Setup and submit task
        system.buy_ticket("miner_001", NodeRole.MINER, 100.0, {})
        system.buy_ticket("supervisor_001", NodeRole.SUPERVISOR, 50.0, {})

        task = MLTask("task_001", "mlp", {}, {}, {}, {}, [], {}, {}, 100.0, "client_001")

        system.submit_task(task)

        # Complete task
        final_models = {"miner_001": {"accuracy": 0.95}}
        performance_metrics = {"miner_001": 0.8}

        rewards = system.complete_task("task_001", final_models, performance_metrics)

        assert "task_001" not in system.active_tasks
        assert "task_001" in system.completed_tasks
        assert system.completed_tasks["task_001"]["status"] == "completed"
        assert isinstance(rewards, dict)

    def test_punish_malicious_node(self):
        """Test punishment of malicious nodes"""
        system = EconomicSystem()

        # Add ticket for malicious node
        system.buy_ticket("malicious_001", NodeRole.MINER, 100.0, {})

        initial_tickets = len(system.staking_manager.stake_pool.tickets)

        # Punish node
        confiscated = system.punish_malicious_node("malicious_001", "fake_work")

        assert confiscated == 100.0
        assert len(system.staking_manager.stake_pool.tickets) < initial_tickets

    def test_node_reputation(self):
        """Test node reputation calculation"""
        system = EconomicSystem()

        # Add and complete some tasks for reputation building
        system.buy_ticket("miner_001", NodeRole.MINER, 100.0, {})

        # Complete a task
        task = MLTask("task_001", "mlp", {}, {}, {}, {}, [], {}, {}, 100.0, "client_001")
        system.submit_task(task)
        system.complete_task("task_001", {}, {"miner_001": 0.8})

        reputation = system.get_node_reputation("miner_001")

        assert reputation["node_id"] == "miner_001"
        assert reputation["tasks_completed"] == 1
        assert reputation["total_earnings"] >= 0  # Changed from total_rewards to total_earnings
        assert reputation["current_stake"] == 100.0

    def test_network_stats(self):
        """Test network statistics"""
        system = EconomicSystem()

        # Add some tickets and tasks
        system.buy_ticket("miner_001", NodeRole.MINER, 100.0, {})
        system.buy_ticket("supervisor_001", NodeRole.SUPERVISOR, 50.0, {})

        stats = system.get_network_stats()

        assert "assignment_stats" in stats
        assert "reward_stats" in stats
        assert "pricing_stats" in stats
        assert "active_tasks" in stats
        assert "completed_tasks" in stats
        assert "market_condition" in stats

        assert stats["active_tasks"] == 0
        assert stats["completed_tasks"] == 0
        assert isinstance(stats["assignment_stats"], dict)
        assert "total_tickets" in stats["assignment_stats"]

    class TestTokenSupply:
        """Test token supply management functionality"""

        def test_token_supply_initialization(self):
            """Test token supply initialization with custom parameters"""
            system = EconomicSystem(
                max_token_supply=1_000_000.0,
                genesis_supply=100_000.0,
                base_block_reward=10.0,
                halving_interval=100_000
            )

            assert system.max_token_supply == 1_000_000.0
            assert system.current_supply == 100_000.0  # Genesis supply
            assert system.base_block_reward == 10.0
            assert system.halving_interval == 100_000
            assert system.total_blocks_mined == 0

        def test_block_reward_calculation(self):
            """Test block reward calculation with halving"""
            system = EconomicSystem(
                base_block_reward=12.5,
                halving_interval=210_000
            )

            # Initial reward
            assert system.calculate_block_reward(0) == 12.5
            assert system.calculate_block_reward(100_000) == 12.5

            # First halving at block 210,000
            assert system.calculate_block_reward(210_000) == 6.25
            assert system.calculate_block_reward(300_000) == 6.25

            # Second halving at block 420,000
            assert system.calculate_block_reward(420_000) == 3.125

        def test_token_minting_with_supply_cap(self):
            """Test token minting respects supply cap"""
            system = EconomicSystem(
                max_token_supply=1000.0,
                genesis_supply=900.0
            )

            # Should be able to mint up to max supply
            assert system.mint_tokens(50.0, "test")
            assert system.current_supply == 950.0

            assert system.mint_tokens(50.0, "test")
            assert system.current_supply == 1000.0

            # Should not be able to mint beyond max supply
            assert not system.mint_tokens(1.0, "test")
            assert system.current_supply == 1000.0

        def test_block_reward_respects_supply_cap(self):
            """Test block reward calculation respects max supply"""
            system = EconomicSystem(
                max_token_supply=1100.0,
                genesis_supply=1000.0,
                base_block_reward=12.5
            )

            # Should get partial reward when approaching max supply
            reward = system.calculate_block_reward(0)
            assert reward == 100.0  # Remaining supply

            # Should get zero reward when max supply reached
            system.current_supply = 1100.0
            reward = system.calculate_block_reward(0)
            assert reward == 0.0

        def test_record_mined_block(self):
            """Test recording mined blocks"""
            system = EconomicSystem(genesis_supply=100.0)

            initial_supply = system.current_supply

            # Record a successful mining
            assert system.record_mined_block(12.5)
            assert system.current_supply == initial_supply + 12.5
            assert system.total_blocks_mined == 1

            # Record another block
            assert system.record_mined_block(12.5)
            assert system.current_supply == initial_supply + 25.0
            assert system.total_blocks_mined == 2

        def test_get_token_supply_info(self):
            """Test getting comprehensive token supply information"""
            system = EconomicSystem(
                max_token_supply=21_000_000.0,
                genesis_supply=1_000_000.0,
                base_block_reward=12.5,
                halving_interval=210_000
            )

            info = system.get_token_supply_info()

            assert info["current_supply"] == 1_000_000.0
            assert info["max_supply"] == 21_000_000.0
            assert info["remaining_supply"] == 20_000_000.0
            assert info["supply_percentage"] == pytest.approx(4.76, rel=1e-2)
            assert info["genesis_supply"] == 1_000_000.0
            assert info["current_block_reward"] == 12.5
            assert info["base_block_reward"] == 12.5
            assert info["halving_interval"] == 210_000
            assert not info["supply_exhausted"]

        def test_circulating_supply_calculation(self):
            """Test circulating supply calculation excludes staked tokens"""
            system = EconomicSystem(genesis_supply=1000.0)

            # Initially all tokens are circulating
            assert system.get_circulating_supply() == 1000.0

            # Stake some tokens
            system.buy_ticket("user1", NodeRole.MINER, 100.0, {})
            system.buy_ticket("user2", NodeRole.VERIFIER, 50.0, {})

            # Circulating supply should be reduced by staked amounts
            expected_circulating = 1000.0 - 150.0
            assert system.get_circulating_supply() == expected_circulating

        def test_inflation_rate_calculation(self):
            """Test inflation rate calculation"""
            system = EconomicSystem(
                genesis_supply=1_000_000.0,
                base_block_reward=12.5
            )

            inflation_rate = system.calculate_inflation_rate()

            # With 1 block per minute (525,600 blocks/year)
            # Annual new tokens = 12.5 * 525,600 = 6,570,000
            # Inflation rate = (6,570,000 / 1,000,000) * 100 = 657%
            expected_inflation = (12.5 * 525_600 / 1_000_000.0) * 100
            assert inflation_rate == pytest.approx(expected_inflation, rel=1e-2)

        def test_supply_health_status(self):
            """Test supply health status indicators"""
            system = EconomicSystem(
                max_token_supply=1000.0,
                genesis_supply=850.0  # 85% of max supply
            )

            health = system.get_supply_health_status()

            assert health["supply_exhaustion_risk"] == "MEDIUM"  # 80-95%
            assert health["reward_sustainability"] == "SUSTAINABLE"
            assert "inflation_rate" in health
            assert "circulating_ratio" in health


if __name__ == "__main__":
    pytest.main([__file__])
