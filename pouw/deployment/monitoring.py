"""
Production Monitoring Module

Provides comprehensive monitoring, metrics collection, alerting, and logging
capabilities for production PoUW deployments.
"""

import asyncio
import json
import logging
import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import aiohttp
import sqlite3
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Metric type enumeration"""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Metric:
    """Metric data structure"""

    name: str
    value: Union[int, float]
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    help_text: str = ""

    def to_prometheus_format(self) -> str:
        """Convert metric to Prometheus format"""
        label_str = ""
        if self.labels:
            label_pairs = [f'{k}="{v}"' for k, v in self.labels.items()]
            label_str = "{" + ",".join(label_pairs) + "}"

        return f"{self.name}{label_str} {self.value} {int(self.timestamp.timestamp() * 1000)}"


@dataclass
class Alert:
    """Alert data structure"""

    id: str
    severity: AlertSeverity
    message: str
    component: str
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def resolve(self):
        """Mark alert as resolved"""
        self.resolved = True
        self.resolved_at = datetime.now()


@dataclass
class HealthStatus:
    """Component health status"""

    component: str
    status: str  # healthy, degraded, unhealthy
    last_check: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    uptime: timedelta = field(default_factory=lambda: timedelta(0))


class MetricsCollector:
    """Advanced metrics collection system"""

    def __init__(self, collection_interval: int = 30):
        """Initialize metrics collector"""
        self.collection_interval = collection_interval
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.metric_handlers: Dict[str, Callable] = {}
        self.running = False
        self.collection_task: Optional[asyncio.Task] = None

    def register_metric_handler(self, metric_name: str, handler: Callable[[], Union[int, float]]):
        """Register a custom metric handler"""
        self.metric_handlers[metric_name] = handler
        logger.info(f"Registered metric handler: {metric_name}")

    async def start_collection(self):
        """Start metrics collection"""
        if self.running:
            return

        self.running = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Started metrics collection")

    async def stop_collection(self):
        """Stop metrics collection"""
        self.running = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped metrics collection")

    async def _collection_loop(self):
        """Main metrics collection loop"""
        while self.running:
            try:
                await self._collect_system_metrics()
                await self._collect_custom_metrics()
                await asyncio.sleep(self.collection_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(5)  # Brief pause before retry

    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        timestamp = datetime.now()

        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics["system_cpu_usage"].append(
            Metric("system_cpu_usage", cpu_percent, MetricType.GAUGE, timestamp=timestamp)
        )

        # Memory metrics
        memory = psutil.virtual_memory()
        self.metrics["system_memory_usage"].append(
            Metric("system_memory_usage", memory.percent, MetricType.GAUGE, timestamp=timestamp)
        )
        self.metrics["system_memory_available"].append(
            Metric(
                "system_memory_available", memory.available, MetricType.GAUGE, timestamp=timestamp
            )
        )

        # Disk metrics
        disk = psutil.disk_usage("/")
        self.metrics["system_disk_usage"].append(
            Metric(
                "system_disk_usage",
                (disk.used / disk.total) * 100,
                MetricType.GAUGE,
                timestamp=timestamp,
            )
        )

        # Network metrics
        network = psutil.net_io_counters()
        self.metrics["system_network_bytes_sent"].append(
            Metric(
                "system_network_bytes_sent",
                network.bytes_sent,
                MetricType.COUNTER,
                timestamp=timestamp,
            )
        )
        self.metrics["system_network_bytes_recv"].append(
            Metric(
                "system_network_bytes_recv",
                network.bytes_recv,
                MetricType.COUNTER,
                timestamp=timestamp,
            )
        )

    async def _collect_custom_metrics(self):
        """Collect custom metrics from registered handlers"""
        timestamp = datetime.now()

        for metric_name, handler in self.metric_handlers.items():
            try:
                value = handler()
                metric = Metric(metric_name, value, MetricType.GAUGE, timestamp=timestamp)
                self.metrics[metric_name].append(metric)

            except Exception as e:
                logger.error(f"Error collecting metric {metric_name}: {e}")

    def add_metric(self, metric: Metric):
        """Add a custom metric"""
        self.metrics[metric.name].append(metric)

    def get_metrics(
        self, metric_name: Optional[str] = None, time_range: Optional[timedelta] = None
    ) -> Dict[str, List[Metric]]:
        """Get metrics with optional filtering"""
        now = datetime.now()
        cutoff_time = now - time_range if time_range else None

        result = {}

        if metric_name:
            metrics_to_process = {metric_name: self.metrics.get(metric_name, deque())}
        else:
            metrics_to_process = self.metrics

        for name, metric_queue in metrics_to_process.items():
            filtered_metrics = []
            for metric in metric_queue:
                if not cutoff_time or metric.timestamp >= cutoff_time:
                    filtered_metrics.append(metric)
            result[name] = filtered_metrics

        return result

    def get_latest_metrics(self) -> Dict[str, Metric]:
        """Get the latest value for each metric"""
        result = {}
        for name, metric_queue in self.metrics.items():
            if metric_queue:
                result[name] = metric_queue[-1]
        return result


class AlertingSystem:
    """Advanced alerting system with configurable rules"""

    def __init__(self, alert_handlers: Optional[List[Callable[[Alert], None]]] = None):
        """Initialize alerting system"""
        self.alert_handlers = alert_handlers or []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_rules: List[Callable[[Dict[str, Metric]], Optional[Alert]]] = []
        self.alert_task: Optional[asyncio.Task] = None
        self.running = False

    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add an alert handler"""
        self.alert_handlers.append(handler)
        logger.info("Added alert handler")

    def add_alert_rule(self, rule: Callable[[Dict[str, Metric]], Optional[Alert]]):
        """Add an alert rule"""
        self.alert_rules.append(rule)
        logger.info("Added alert rule")

    async def start_monitoring(self, metrics_collector: MetricsCollector, check_interval: int = 60):
        """Start alert monitoring"""
        if self.running:
            return

        self.running = True
        self.alert_task = asyncio.create_task(
            self._monitoring_loop(metrics_collector, check_interval)
        )
        logger.info("Started alert monitoring")

    async def stop_monitoring(self):
        """Stop alert monitoring"""
        self.running = False
        if self.alert_task:
            self.alert_task.cancel()
            try:
                await self.alert_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped alert monitoring")

    async def _monitoring_loop(self, metrics_collector: MetricsCollector, check_interval: int):
        """Main alert monitoring loop"""
        while self.running:
            try:
                latest_metrics = metrics_collector.get_latest_metrics()
                await self._check_alert_rules(latest_metrics)
                await asyncio.sleep(check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert monitoring: {e}")
                await asyncio.sleep(30)  # Pause before retry

    async def _check_alert_rules(self, metrics: Dict[str, Metric]):
        """Check all alert rules against current metrics"""
        for rule in self.alert_rules:
            try:
                alert = rule(metrics)
                if alert:
                    await self._handle_alert(alert)

            except Exception as e:
                logger.error(f"Error checking alert rule: {e}")

    async def _handle_alert(self, alert: Alert):
        """Handle a generated alert"""
        # Check if this is a new alert or update to existing
        if alert.id in self.active_alerts:
            existing_alert = self.active_alerts[alert.id]
            if not existing_alert.resolved:
                return  # Alert already active

        # Add to active alerts
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)

        # Notify all handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")

        logger.warning(f"Alert triggered: {alert.severity.value} - {alert.message}")

    def resolve_alert(self, alert_id: str):
        """Resolve an active alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].resolve()
            logger.info(f"Resolved alert: {alert_id}")

    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity"""
        alerts = [alert for alert in self.active_alerts.values() if not alert.resolved]

        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]

        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    def get_alert_history(self, time_range: Optional[timedelta] = None) -> List[Alert]:
        """Get alert history, optionally filtered by time range"""
        if not time_range:
            return sorted(self.alert_history, key=lambda a: a.timestamp, reverse=True)

        cutoff_time = datetime.now() - time_range
        filtered_alerts = [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]

        return sorted(filtered_alerts, key=lambda a: a.timestamp, reverse=True)


class LoggingManager:
    """Advanced logging management for production systems"""

    def __init__(
        self,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        max_file_size: int = 100 * 1024 * 1024,
    ):  # 100MB
        """Initialize logging manager"""
        self.log_level = getattr(logging, log_level.upper())
        self.log_file = log_file
        self.max_file_size = max_file_size
        self.log_buffer: deque = deque(maxlen=10000)
        self.structured_logs: deque = deque(maxlen=5000)

        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration"""
        # Create custom formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # File handler if specified
        if self.log_file:
            from logging.handlers import RotatingFileHandler

            file_handler = RotatingFileHandler(
                self.log_file, maxBytes=self.max_file_size, backupCount=5
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        # Custom handler for log buffer
        buffer_handler = self._create_buffer_handler()
        root_logger.addHandler(buffer_handler)

    def _create_buffer_handler(self):
        """Create custom handler for log buffering"""

        class BufferHandler(logging.Handler):
            def __init__(self, log_buffer, structured_logs):
                super().__init__()
                self.log_buffer = log_buffer
                self.structured_logs = structured_logs

            def emit(self, record):
                # Add to simple buffer
                log_message = self.format(record)
                self.log_buffer.append(
                    {
                        "timestamp": datetime.fromtimestamp(record.created),
                        "level": record.levelname,
                        "message": log_message,
                    }
                )

                # Add to structured logs
                structured_log = {
                    "timestamp": datetime.fromtimestamp(record.created),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "funcName": record.funcName,
                    "lineno": record.lineno,
                }

                extra_data = getattr(record, "extra", None)
                if extra_data:
                    structured_log.update(extra_data)

                self.structured_logs.append(structured_log)

        return BufferHandler(self.log_buffer, self.structured_logs)

    def get_recent_logs(
        self, count: int = 100, level: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recent log entries"""
        logs = list(self.log_buffer)

        if level:
            logs = [log for log in logs if log["level"] == level.upper()]

        return logs[-count:] if count else logs

    def get_structured_logs(
        self, time_range: Optional[timedelta] = None, level: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get structured log entries with filtering"""
        logs = list(self.structured_logs)

        if time_range:
            cutoff_time = datetime.now() - time_range
            logs = [log for log in logs if log["timestamp"] >= cutoff_time]

        if level:
            logs = [log for log in logs if log["level"] == level.upper()]

        return sorted(logs, key=lambda l: l["timestamp"], reverse=True)

    def create_component_logger(self, component_name: str) -> logging.Logger:
        """Create a logger for a specific component"""
        logger = logging.getLogger(f"pouw.{component_name}")
        return logger


class HealthChecker:
    """Component health checking system"""

    def __init__(self, check_interval: int = 30):
        """Initialize health checker"""
        self.check_interval = check_interval
        self.health_checks: Dict[str, Callable[[], HealthStatus]] = {}
        self.health_status: Dict[str, HealthStatus] = {}
        self.running = False
        self.health_task: Optional[asyncio.Task] = None

    def register_health_check(self, component: str, check_func: Callable[[], HealthStatus]):
        """Register a health check for a component"""
        self.health_checks[component] = check_func
        logger.info(f"Registered health check for: {component}")

    async def start_health_checks(self):
        """Start health checking"""
        if self.running:
            return

        self.running = True
        self.health_task = asyncio.create_task(self._health_check_loop())
        logger.info("Started health checks")

    async def stop_health_checks(self):
        """Stop health checking"""
        self.running = False
        if self.health_task:
            self.health_task.cancel()
            try:
                await self.health_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped health checks")

    async def _health_check_loop(self):
        """Main health checking loop"""
        while self.running:
            try:
                await self._run_health_checks()
                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health checking: {e}")
                await asyncio.sleep(30)

    async def _run_health_checks(self):
        """Run all registered health checks"""
        with ThreadPoolExecutor(max_workers=5) as executor:
            tasks = []

            for component, check_func in self.health_checks.items():
                future = asyncio.get_event_loop().run_in_executor(executor, check_func)
                tasks.append((component, future))

            for component, task in tasks:
                try:
                    health_status = await task
                    self.health_status[component] = health_status

                except Exception as e:
                    logger.error(f"Health check failed for {component}: {e}")
                    self.health_status[component] = HealthStatus(
                        component=component,
                        status="unhealthy",
                        last_check=datetime.now(),
                        details={"error": str(e)},
                    )

    def get_health_status(
        self, component: Optional[str] = None
    ) -> Union[HealthStatus, Dict[str, HealthStatus], None]:
        """Get health status for component(s)"""
        if component:
            return self.health_status.get(component)
        return self.health_status.copy()

    def get_unhealthy_components(self) -> List[str]:
        """Get list of unhealthy components"""
        return [
            component
            for component, status in self.health_status.items()
            if status.status == "unhealthy"
        ]


class PerformanceAnalyzer:
    """Performance analysis and optimization recommendations"""

    def __init__(self):
        """Initialize performance analyzer"""
        self.performance_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.analysis_rules: List[Callable[[Dict[str, Any]], Dict[str, Any]]] = []

    def record_performance_metric(
        self, metric_name: str, value: float, metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a performance metric"""
        data_point = {"timestamp": datetime.now(), "value": value, "metadata": metadata or {}}
        self.performance_data[metric_name].append(data_point)

    def add_analysis_rule(self, rule: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """Add a performance analysis rule"""
        self.analysis_rules.append(rule)

    def analyze_performance(self, time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """Analyze performance and generate recommendations"""
        analysis_time = time_range or timedelta(hours=1)
        cutoff_time = datetime.now() - analysis_time

        # Aggregate performance data
        aggregated_data = {}
        for metric_name, data_points in self.performance_data.items():
            recent_points = [point for point in data_points if point["timestamp"] >= cutoff_time]

            if recent_points:
                values = [point["value"] for point in recent_points]
                aggregated_data[metric_name] = {
                    "count": len(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "latest": values[-1],
                }

        # Run analysis rules
        recommendations = []
        insights = {}

        for rule in self.analysis_rules:
            try:
                result = rule(aggregated_data)
                if result:
                    if "recommendations" in result:
                        recommendations.extend(result["recommendations"])
                    if "insights" in result:
                        insights.update(result["insights"])

            except Exception as e:
                logger.error(f"Error in performance analysis rule: {e}")

        return {
            "analysis_time": analysis_time,
            "metrics_analyzed": len(aggregated_data),
            "aggregated_data": aggregated_data,
            "recommendations": recommendations,
            "insights": insights,
            "timestamp": datetime.now(),
        }


class ProductionMonitor:
    """Comprehensive production monitoring system"""

    def __init__(self, namespace: str = "pouw-system", log_file: Optional[str] = None):
        """Initialize production monitor"""
        import os
        import tempfile
        
        self.namespace = namespace
        self.metrics_collector = MetricsCollector()
        self.alerting_system = AlertingSystem()

        # Use custom log file path or default based on environment
        if log_file is None:
            # Check if running in test environment or don't have write permissions
            log_dir = "/var/log/pouw"
            if (os.getenv("PYTEST_CURRENT_TEST") or 
                not os.path.exists("/var/log") or 
                not os.access("/var/log", os.W_OK)):
                # Use temporary directory for testing or when no write access
                log_dir = tempfile.gettempdir()
            else:
                # Create log directory if it doesn't exist and we have permissions
                try:
                    os.makedirs(log_dir, exist_ok=True)
                except PermissionError:
                    log_dir = tempfile.gettempdir()
            
            log_file = f"{log_dir}/{namespace}.log"

        self.logging_manager = LoggingManager(log_file=log_file)
        self.health_checker = HealthChecker()
        self.performance_analyzer = PerformanceAnalyzer()

        self._setup_default_rules()
        self._setup_default_health_checks()

    def _setup_default_rules(self):
        """Setup default alerting rules"""

        # High CPU usage rule
        def high_cpu_rule(metrics: Dict[str, Metric]) -> Optional[Alert]:
            cpu_metric = metrics.get("system_cpu_usage")
            if cpu_metric and cpu_metric.value > 80:
                return Alert(
                    id="high_cpu_usage",
                    severity=AlertSeverity.WARNING,
                    message=f"High CPU usage detected: {cpu_metric.value:.1f}%",
                    component="system",
                    metadata={"cpu_usage": cpu_metric.value},
                )
            return None

        # High memory usage rule
        def high_memory_rule(metrics: Dict[str, Metric]) -> Optional[Alert]:
            memory_metric = metrics.get("system_memory_usage")
            if memory_metric and memory_metric.value > 85:
                return Alert(
                    id="high_memory_usage",
                    severity=AlertSeverity.WARNING,
                    message=f"High memory usage detected: {memory_metric.value:.1f}%",
                    component="system",
                    metadata={"memory_usage": memory_metric.value},
                )
            return None

        # High disk usage rule
        def high_disk_rule(metrics: Dict[str, Metric]) -> Optional[Alert]:
            disk_metric = metrics.get("system_disk_usage")
            if disk_metric and disk_metric.value > 90:
                return Alert(
                    id="high_disk_usage",
                    severity=AlertSeverity.ERROR,
                    message=f"High disk usage detected: {disk_metric.value:.1f}%",
                    component="system",
                    metadata={"disk_usage": disk_metric.value},
                )
            return None

        self.alerting_system.add_alert_rule(high_cpu_rule)
        self.alerting_system.add_alert_rule(high_memory_rule)
        self.alerting_system.add_alert_rule(high_disk_rule)

    def _setup_default_health_checks(self):
        """Setup default health checks"""

        def system_health_check() -> HealthStatus:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                disk_percent = psutil.disk_usage("/").percent

                if cpu_percent > 90 or memory_percent > 95 or disk_percent > 95:
                    status = "unhealthy"
                elif cpu_percent > 70 or memory_percent > 80 or disk_percent > 85:
                    status = "degraded"
                else:
                    status = "healthy"

                return HealthStatus(
                    component="system",
                    status=status,
                    last_check=datetime.now(),
                    details={
                        "cpu_usage": cpu_percent,
                        "memory_usage": memory_percent,
                        "disk_usage": disk_percent,
                    },
                )

            except Exception as e:
                return HealthStatus(
                    component="system",
                    status="unhealthy",
                    last_check=datetime.now(),
                    details={"error": str(e)},
                )

        self.health_checker.register_health_check("system", system_health_check)

    async def start_monitoring(self):
        """Start all monitoring services"""
        logger.info("Starting production monitoring...")

        await self.metrics_collector.start_collection()
        await self.alerting_system.start_monitoring(self.metrics_collector)
        await self.health_checker.start_health_checks()

        logger.info("Production monitoring started successfully")

    async def stop_monitoring(self):
        """Stop all monitoring services"""
        logger.info("Stopping production monitoring...")

        await self.metrics_collector.stop_collection()
        await self.alerting_system.stop_monitoring()
        await self.health_checker.stop_health_checks()

        logger.info("Production monitoring stopped")

    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data"""
        latest_metrics = self.metrics_collector.get_latest_metrics()
        active_alerts = self.alerting_system.get_active_alerts()
        health_status = self.health_checker.get_health_status()
        
        # Ensure health_status is always a dictionary for the dashboard
        if isinstance(health_status, HealthStatus):
            health_status_dict = {"default": health_status}
        elif health_status is None:
            health_status_dict = {}
        else:
            health_status_dict = health_status
        performance_analysis = self.performance_analyzer.analyze_performance(timedelta(hours=1))
        recent_logs = self.logging_manager.get_recent_logs(50)

        return {
            "timestamp": datetime.now(),
            "namespace": self.namespace,
            "metrics": {name: metric.value for name, metric in latest_metrics.items()},
            "alerts": {
                "active_count": len(active_alerts),
                "critical_count": len(
                    [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
                ),
                "recent_alerts": [
                    {
                        "id": alert.id,
                        "severity": alert.severity.value,
                        "message": alert.message,
                        "timestamp": alert.timestamp.isoformat(),
                    }
                    for alert in active_alerts[:5]
                ],
            },
            "health": {
                name: {
                    "status": status.status,
                    "last_check": status.last_check.isoformat(),
                    "details": status.details,
                }
                for name, status in health_status_dict.items()
            },
            "performance": performance_analysis,
            "logs": recent_logs,
        }
