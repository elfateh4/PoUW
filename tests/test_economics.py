"""
Unit tests for PoUW economic system.
"""

import pytest
import time
from pouw.economics import NodeRole, Ticket, StakePool, TaskMatcher, RewardScheme, EconomicSystem
from pouw.blockchain import MLTask


class TestTicket:
    """Test Ticket functionality"""
    
    def test_ticket_creation(self):
        """Test ticket creation"""
        ticket = Ticket(
            ticket_id='ticket_001',
            owner_id='user_001',
            role=NodeRole.MINER,
            stake_amount=100.0,
            preferences={'gpu': True},
            expiration_time=int(time.time()) + 3600
        )
        
        assert ticket.ticket_id == 'ticket_001'
        assert ticket.owner_id == 'user_001'
        assert ticket.role == NodeRole.MINER
        assert ticket.stake_amount == 100.0
        assert not ticket.is_expired()
    
    def test_ticket_expiration(self):
        """Test ticket expiration"""
        # Create expired ticket
        ticket = Ticket(
            ticket_id='ticket_001',
            owner_id='user_001',
            role=NodeRole.MINER,
            stake_amount=100.0,
            preferences={},
            expiration_time=int(time.time()) - 3600  # 1 hour ago
        )
        
        assert ticket.is_expired()
    
    def test_task_matching(self):
        """Test task compatibility scoring"""
        task = MLTask(
            task_id='task_001',
            model_type='mlp',
            architecture={'input_size': 784, 'output_size': 10},
            optimizer={},
            stopping_criterion={},
            validation_strategy={},
            metrics=[],
            dataset_info={'size': 50000},
            performance_requirements={'gpu': True},
            fee=100.0,
            client_id='client_001'
        )
        
        # Good match ticket
        good_ticket = Ticket(
            'ticket_001', 'user_001', NodeRole.MINER, 100.0,
            {
                'model_types': ['mlp', 'cnn'],
                'has_gpu': True,
                'max_dataset_size': 100000
            },
            int(time.time()) + 3600
        )
        
        # Poor match ticket
        poor_ticket = Ticket(
            'ticket_002', 'user_002', NodeRole.MINER, 100.0,
            {
                'model_types': ['rnn'],
                'has_gpu': False,
                'max_dataset_size': 1000
            },
            int(time.time()) + 3600
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
            'ticket_001', 'user_001', NodeRole.MINER, 100.0, {},
            int(time.time()) + 3600
        )
        
        pool.add_ticket(ticket)
        
        assert len(pool.tickets) == 1
        assert len(pool.live_tickets) == 1
        assert ticket.is_live
        assert 'ticket_001' in pool.tickets
    
    def test_get_tickets_by_role(self):
        """Test filtering tickets by role"""
        pool = StakePool()
        
        miner_ticket = Ticket(
            'ticket_001', 'user_001', NodeRole.MINER, 100.0, {},
            int(time.time()) + 3600
        )
        
        supervisor_ticket = Ticket(
            'ticket_002', 'user_002', NodeRole.SUPERVISOR, 50.0, {},
            int(time.time()) + 3600
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
                f'ticket_{i}', f'user_{i}', NodeRole.MINER, 100.0, {},
                int(time.time()) + 3600
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
        assert matcher.omega_s == 0.1
    
    def test_select_workers(self):
        """Test worker selection"""
        pool = StakePool()
        matcher = TaskMatcher(pool)
        
        # Add various tickets
        for i in range(5):
            miner_ticket = Ticket(
                f'miner_{i}', f'user_{i}', NodeRole.MINER, 100.0,
                {'model_types': ['mlp'], 'has_gpu': True}, 
                int(time.time()) + 3600
            )
            pool.add_ticket(miner_ticket)
        
        for i in range(3):
            supervisor_ticket = Ticket(
                f'supervisor_{i}', f'supervisor_{i}', NodeRole.SUPERVISOR, 50.0,
                {'storage_capacity': 1000000}, 
                int(time.time()) + 3600
            )
            pool.add_ticket(supervisor_ticket)
        
        # Create task
        task = MLTask(
            'task_001', 'mlp', {}, {}, {}, {}, [], 
            {'size': 50000}, {'gpu': True}, 100.0, 'client_001'
        )
        
        # Select workers
        selected = matcher.select_workers(task, num_miners=3, num_supervisors=2)
        
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
        performance_scores = {
            'miner_001': 0.6,
            'miner_002': 0.4
        }
        
        rewards = scheme.calculate_rewards(total_fee, performance_scores)
        
        # Check that miners get proportional rewards
        assert 'miner_001' in rewards
        assert 'miner_002' in rewards
        assert rewards['miner_001'] > rewards['miner_002']
        
        # Check total miner rewards
        total_miner_rewards = sum(rewards.values())
        expected_miner_pool = total_fee * scheme.miner_percentage
        assert abs(total_miner_rewards - expected_miner_pool) < 0.01


class TestEconomicSystem:
    """Test complete EconomicSystem functionality"""
    
    def test_economic_system_creation(self):
        """Test economic system creation"""
        system = EconomicSystem()
        
        assert isinstance(system.stake_pool, StakePool)
        assert isinstance(system.task_matcher, TaskMatcher)
        assert isinstance(system.reward_scheme, RewardScheme)
        assert len(system.active_tasks) == 0
        assert len(system.completed_tasks) == 0
    
    def test_buy_ticket(self):
        """Test ticket purchase"""
        system = EconomicSystem()
        
        preferences = {'gpu': True, 'model_types': ['mlp']}
        
        ticket = system.buy_ticket('user_001', NodeRole.MINER, 100.0, preferences)
        
        assert ticket.owner_id == 'user_001'
        assert ticket.role == NodeRole.MINER
        assert ticket.stake_amount == 100.0
        assert ticket.preferences == preferences
        assert len(system.stake_pool.tickets) == 1
    
    def test_buy_ticket_insufficient_stake(self):
        """Test ticket purchase with insufficient stake"""
        system = EconomicSystem()
        
        # Try to buy with insufficient stake
        with pytest.raises(ValueError):
            system.buy_ticket('user_001', NodeRole.MINER, 1.0, {})
    
    def test_submit_task(self):
        """Test task submission and worker assignment"""
        system = EconomicSystem()
        
        # Add some tickets first
        system.buy_ticket('miner_001', NodeRole.MINER, 100.0, {'gpu': True})
        system.buy_ticket('supervisor_001', NodeRole.SUPERVISOR, 50.0, {})
        
        # Create and submit task
        task = MLTask(
            'task_001', 'mlp', {}, {}, {}, {}, [], 
            {'size': 50000}, {'gpu': True}, 100.0, 'client_001'
        )
        
        selected_workers = system.submit_task(task)
        
        assert 'task_001' in system.active_tasks
        assert system.active_tasks['task_001']['status'] == 'active'
        assert NodeRole.MINER in selected_workers or NodeRole.SUPERVISOR in selected_workers
    
    def test_complete_task(self):
        """Test task completion and reward distribution"""
        system = EconomicSystem()
        
        # Setup and submit task
        system.buy_ticket('miner_001', NodeRole.MINER, 100.0, {})
        system.buy_ticket('supervisor_001', NodeRole.SUPERVISOR, 50.0, {})
        
        task = MLTask(
            'task_001', 'mlp', {}, {}, {}, {}, [], 
            {}, {}, 100.0, 'client_001'
        )
        
        system.submit_task(task)
        
        # Complete task
        final_models = {'miner_001': {'accuracy': 0.95}}
        performance_metrics = {'miner_001': 0.8}
        
        rewards = system.complete_task('task_001', final_models, performance_metrics)
        
        assert 'task_001' not in system.active_tasks
        assert 'task_001' in system.completed_tasks
        assert system.completed_tasks['task_001']['status'] == 'completed'
        assert isinstance(rewards, dict)
    
    def test_punish_malicious_node(self):
        """Test punishment of malicious nodes"""
        system = EconomicSystem()
        
        # Add ticket for malicious node
        system.buy_ticket('malicious_001', NodeRole.MINER, 100.0, {})
        
        initial_tickets = len(system.stake_pool.tickets)
        
        # Punish node
        confiscated = system.punish_malicious_node('malicious_001', 'fake_work')
        
        assert confiscated == 100.0
        assert len(system.stake_pool.tickets) < initial_tickets
    
    def test_node_reputation(self):
        """Test node reputation calculation"""
        system = EconomicSystem()
        
        # Add and complete some tasks for reputation building
        system.buy_ticket('miner_001', NodeRole.MINER, 100.0, {})
        
        # Complete a task
        task = MLTask(
            'task_001', 'mlp', {}, {}, {}, {}, [], 
            {}, {}, 100.0, 'client_001'
        )
        system.submit_task(task)
        system.complete_task('task_001', {}, {'miner_001': 0.8})
        
        reputation = system.get_node_reputation('miner_001')
        
        assert reputation['node_id'] == 'miner_001'
        assert reputation['tasks_completed'] == 1
        assert reputation['total_rewards'] > 0
        assert reputation['current_stake'] == 100.0
    
    def test_network_stats(self):
        """Test network statistics"""
        system = EconomicSystem()
        
        # Add some tickets and tasks
        system.buy_ticket('miner_001', NodeRole.MINER, 100.0, {})
        system.buy_ticket('supervisor_001', NodeRole.SUPERVISOR, 50.0, {})
        
        stats = system.get_network_stats()
        
        assert 'total_tickets' in stats
        assert 'role_distribution' in stats
        assert 'active_tasks' in stats
        assert 'completed_tasks' in stats
        assert 'current_ticket_price' in stats
        
        assert stats['total_tickets'] == 2
        assert stats['role_distribution']['miner'] == 1
        assert stats['role_distribution']['supervisor'] == 1


if __name__ == '__main__':
    pytest.main([__file__])
