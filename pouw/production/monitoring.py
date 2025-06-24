"""
Performance Monitoring and Optimization for PoUW

This module provides comprehensive performance monitoring, profiling,
and optimization capabilities for production PoUW deployment.
"""

import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import logging
from pathlib import Path
import statistics

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a specific operation"""
    operation_name: str
    execution_time: float
    memory_usage: float  # MB
    cpu_usage: float  # percentage
    gpu_usage: Optional[float] = None  # percentage
    gpu_memory: Optional[float] = None  # MB
    timestamp: float = field(default_factory=time.time)
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealth:
    """Overall system health metrics"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    gpu_info: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)


class PerformanceProfiler:
    """
    Context manager for profiling code execution
    """
    
    def __init__(self, operation_name: str, monitor: 'PerformanceMonitor'):
        self.operation_name = operation_name
        self.monitor = monitor
        self.start_time = None
        self.start_memory = None
        self.start_cpu = None
        
    def __enter__(self):
        self.start_time = time.time()
        self.start_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        self.start_cpu = psutil.cpu_percent()
        
        # GPU metrics if available
        self.start_gpu_memory = None
        self.start_gpu_usage = None
        if torch.cuda.is_available():
            try:
                self.start_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                # Note: GPU usage requires nvidia-ml-py package for accurate measurement
            except:
                pass
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        end_memory = psutil.virtual_memory().used / 1024 / 1024
        end_cpu = psutil.cpu_percent()
        
        execution_time = end_time - (self.start_time or 0)
        memory_delta = end_memory - (self.start_memory or 0)
        cpu_usage = ((self.start_cpu or 0) + end_cpu) / 2  # Average CPU usage
        
        # GPU metrics
        gpu_memory_delta = None
        if self.start_gpu_memory is not None and torch.cuda.is_available():
            try:
                end_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_memory_delta = end_gpu_memory - self.start_gpu_memory
            except:
                pass
        
        metrics = PerformanceMetrics(
            operation_name=self.operation_name,
            execution_time=execution_time,
            memory_usage=memory_delta,
            cpu_usage=cpu_usage,
            gpu_memory=gpu_memory_delta
        )
        
        self.monitor.record_metrics(metrics)


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for PoUW
    """
    
    def __init__(self, history_size: int = 1000, enable_gpu: Optional[bool] = None):
        self.history_size = history_size
        self.enable_gpu = enable_gpu if enable_gpu is not None else torch.cuda.is_available()
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=history_size)
        self.operation_stats: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self.system_health_history: deque = deque(maxlen=history_size)
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_interval = 5.0  # seconds
        
        # Optimization recommendations
        self.optimization_rules: List[Callable] = []
        self.setup_default_optimization_rules()
        
        logger.info(f"Performance monitor initialized (GPU: {self.enable_gpu})")
    
    def setup_default_optimization_rules(self):
        """Setup default optimization rules"""
        
        def high_memory_usage_rule(metrics: List[PerformanceMetrics]) -> Optional[str]:
            if not metrics:
                return None
            
            avg_memory = statistics.mean([m.memory_usage for m in metrics if m.memory_usage > 0])
            if avg_memory > 1000:  # >1GB memory usage
                return "High memory usage detected. Consider batch size reduction or model optimization."
            return None
        
        def slow_execution_rule(metrics: List[PerformanceMetrics]) -> Optional[str]:
            if not metrics:
                return None
            
            avg_time = statistics.mean([m.execution_time for m in metrics])
            if avg_time > 10.0:  # >10 seconds
                return "Slow execution detected. Consider GPU acceleration or algorithm optimization."
            return None
        
        def high_cpu_usage_rule(metrics: List[PerformanceMetrics]) -> Optional[str]:
            if not metrics:
                return None
            
            avg_cpu = statistics.mean([m.cpu_usage for m in metrics])
            if avg_cpu > 80:  # >80% CPU usage
                return "High CPU usage detected. Consider parallel processing or workload distribution."
            return None
        
        self.optimization_rules = [
            high_memory_usage_rule,
            slow_execution_rule,
            high_cpu_usage_rule
        ]
    
    def start_monitoring(self, interval: float = 5.0):
        """Start system health monitoring"""
        self.monitoring_interval = interval
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    health = self.collect_system_health()
                    self.system_health_history.append(health)
                    time.sleep(self.monitoring_interval)
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(self.monitoring_interval)
        
        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info(f"Started system monitoring (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop system health monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Stopped system monitoring")
    
    def collect_system_health(self) -> SystemHealth:
        """Collect current system health metrics"""
        
        # CPU and memory
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network I/O
        network = psutil.net_io_counters()
        network_io = {
            'bytes_sent': float(network.bytes_sent),
            'bytes_recv': float(network.bytes_recv),
            'packets_sent': float(network.packets_sent),
            'packets_recv': float(network.packets_recv)
        }
        
        # GPU info
        gpu_info = None
        if self.enable_gpu and torch.cuda.is_available():
            gpu_info = {
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'memory_allocated': torch.cuda.memory_allocated() / 1024 / 1024,  # MB
                'memory_reserved': torch.cuda.memory_reserved() / 1024 / 1024,  # MB
            }
            
            # Try to get GPU utilization (requires nvidia-ml-py)
            try:
                import pynvml  # type: ignore
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_info['utilization'] = gpu_util.gpu
                gpu_info['memory_utilization'] = gpu_util.memory
            except ImportError:
                pass  # pynvml not available
            except Exception:
                pass  # GPU utilization not available
        
        return SystemHealth(
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            disk_usage=disk.percent,
            network_io=network_io,
            gpu_info=gpu_info
        )
    
    def record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics"""
        self.metrics_history.append(metrics)
        self.operation_stats[metrics.operation_name].append(metrics)
        
        # Keep operation stats limited
        if len(self.operation_stats[metrics.operation_name]) > self.history_size:
            self.operation_stats[metrics.operation_name] = \
                self.operation_stats[metrics.operation_name][-self.history_size:]
    
    def profile(self, operation_name: str) -> PerformanceProfiler:
        """Create a performance profiler context manager"""
        return PerformanceProfiler(operation_name, self)
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for a specific operation"""
        if operation_name not in self.operation_stats:
            return {}
        
        metrics = self.operation_stats[operation_name]
        if not metrics:
            return {}
        
        execution_times = [m.execution_time for m in metrics]
        memory_usage = [m.memory_usage for m in metrics if m.memory_usage is not None]
        cpu_usage = [m.cpu_usage for m in metrics if m.cpu_usage is not None]
        
        stats = {
            'count': len(metrics),
            'execution_time': {
                'mean': statistics.mean(execution_times),
                'median': statistics.median(execution_times),
                'min': min(execution_times),
                'max': max(execution_times),
                'stdev': statistics.stdev(execution_times) if len(execution_times) > 1 else 0
            }
        }
        
        if memory_usage:
            stats['memory_usage'] = {
                'mean': statistics.mean(memory_usage),
                'median': statistics.median(memory_usage),
                'min': min(memory_usage),
                'max': max(memory_usage)
            }
        
        if cpu_usage:
            stats['cpu_usage'] = {
                'mean': statistics.mean(cpu_usage),
                'median': statistics.median(cpu_usage),
                'min': min(cpu_usage),
                'max': max(cpu_usage)
            }
        
        return stats
    
    def get_all_operation_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all monitored operations"""
        return {op: self.get_operation_stats(op) for op in self.operation_stats.keys()}
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on current metrics"""
        recommendations = []
        
        for operation_name, metrics in self.operation_stats.items():
            for rule in self.optimization_rules:
                try:
                    recommendation = rule(metrics)
                    if recommendation:
                        recommendations.append(f"{operation_name}: {recommendation}")
                except Exception as e:
                    logger.error(f"Error applying optimization rule: {e}")
        
        return recommendations
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        current_health = self.collect_system_health()
        operation_stats = self.get_all_operation_stats()
        recommendations = self.get_optimization_recommendations()
        
        # System health trends
        health_trend = {}
        if self.system_health_history:
            recent_health = list(self.system_health_history)[-10:]  # Last 10 readings
            
            health_trend = {
                'cpu_usage': {
                    'current': current_health.cpu_usage,
                    'average': statistics.mean([h.cpu_usage for h in recent_health]),
                    'trend': 'stable'  # Could implement trend analysis
                },
                'memory_usage': {
                    'current': current_health.memory_usage,
                    'average': statistics.mean([h.memory_usage for h in recent_health]),
                    'trend': 'stable'
                }
            }
        
        report = {
            'timestamp': time.time(),
            'system_health': {
                'current': current_health.__dict__,
                'trends': health_trend
            },
            'operation_performance': operation_stats,
            'optimization_recommendations': recommendations,
            'monitoring_info': {
                'history_size': len(self.metrics_history),
                'operations_monitored': len(self.operation_stats),
                'monitoring_active': self.monitoring_active
            }
        }
        
        return report
    
    def export_report(self, filepath: str):
        """Export performance report to JSON file"""
        report = self.generate_performance_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance report exported to {filepath}")
    
    def clear_history(self):
        """Clear all performance history"""
        self.metrics_history.clear()
        self.operation_stats.clear()
        self.system_health_history.clear()
        logger.info("Performance history cleared")


class OptimizationManager:
    """
    Manages performance optimizations for PoUW operations
    """
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.optimizations_applied: List[str] = []
        
    def optimize_batch_size(self, current_batch_size: int, target_memory_mb: float = 1000) -> int:
        """Optimize batch size based on memory usage"""
        
        # Look for recent training metrics
        training_metrics = []
        for metrics in self.monitor.metrics_history:
            if 'training' in metrics.operation_name.lower() and metrics.memory_usage > 0:
                training_metrics.append(metrics)
        
        if not training_metrics:
            return current_batch_size
        
        # Calculate average memory per batch
        recent_metrics = training_metrics[-10:]  # Last 10 training operations
        avg_memory = statistics.mean([m.memory_usage for m in recent_metrics])
        
        if avg_memory > target_memory_mb:
            # Reduce batch size
            reduction_factor = target_memory_mb / avg_memory
            new_batch_size = max(1, int(current_batch_size * reduction_factor))
            
            optimization = f"Reduced batch size from {current_batch_size} to {new_batch_size} due to high memory usage"
            self.optimizations_applied.append(optimization)
            logger.info(optimization)
            
            return new_batch_size
        
        return current_batch_size
    
    def suggest_gpu_settings(self) -> Dict[str, Any]:
        """Suggest GPU optimization settings"""
        suggestions = {}
        
        if not torch.cuda.is_available():
            suggestions['gpu_available'] = False
            suggestions['recommendation'] = "Consider GPU acceleration for better performance"
            return suggestions
        
        current_health = self.monitor.collect_system_health()
        gpu_info = current_health.gpu_info
        
        if gpu_info:
            memory_usage_percent = (gpu_info['memory_allocated'] / gpu_info['memory_reserved']) * 100 if gpu_info['memory_reserved'] > 0 else 0
            
            suggestions.update({
                'gpu_available': True,
                'current_memory_usage': f"{memory_usage_percent:.1f}%",
                'suggestions': []
            })
            
            if memory_usage_percent > 80:
                suggestions['suggestions'].append("GPU memory usage high - consider reducing batch size")
            elif memory_usage_percent < 30:
                suggestions['suggestions'].append("GPU memory underutilized - consider increasing batch size")
            
            if 'utilization' in gpu_info and gpu_info['utilization'] < 50:
                suggestions['suggestions'].append("GPU utilization low - consider model parallelization")
        
        return suggestions
    
    def auto_optimize(self, operation_config: Dict[str, Any]) -> Dict[str, Any]:
        """Automatically apply optimizations based on current performance"""
        
        optimized_config = operation_config.copy()
        
        # Optimize batch size if training
        if 'batch_size' in optimized_config:
            new_batch_size = self.optimize_batch_size(optimized_config['batch_size'])
            optimized_config['batch_size'] = new_batch_size
        
        # GPU optimizations
        gpu_suggestions = self.suggest_gpu_settings()
        if gpu_suggestions.get('gpu_available', False):
            optimized_config['use_gpu'] = True
            optimized_config['pin_memory'] = True
        
        return optimized_config


# Integration functions for PoUW components
def monitor_mining_performance(monitor: PerformanceMonitor, mining_func: Callable):
    """Decorator to monitor mining performance"""
    def wrapper(*args, **kwargs):
        with monitor.profile("mining_operation"):
            return mining_func(*args, **kwargs)
    return wrapper

def monitor_training_performance(monitor: PerformanceMonitor, training_func: Callable):
    """Decorator to monitor training performance"""
    def wrapper(*args, **kwargs):
        with monitor.profile("training_operation"):
            return training_func(*args, **kwargs)
    return wrapper

def monitor_verification_performance(monitor: PerformanceMonitor, verification_func: Callable):
    """Decorator to monitor verification performance"""
    def wrapper(*args, **kwargs):
        with monitor.profile("verification_operation"):
            return verification_func(*args, **kwargs)
    return wrapper
