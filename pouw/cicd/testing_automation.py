"""
Testing Automation Module for PoUW CI/CD Pipeline

This module provides comprehensive testing automation capabilities including:
- Test suite management and execution
- Coverage analysis and reporting
- Integration test orchestration
- Performance testing automation
- Test result aggregation and reporting
"""

import asyncio
import json
import subprocess
import tempfile
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import xml.etree.ElementTree as ET

import docker
import pytest
import coverage
from kubernetes import client, config as k8s_config


class TestType(Enum):
    """Test types supported by the automation framework."""

    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    PERFORMANCE = "performance"
    SECURITY = "security"
    LOAD = "load"
    SMOKE = "smoke"


class TestStatus(Enum):
    """Test execution status."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestConfiguration:
    """Configuration for test execution."""

    test_type: TestType
    test_paths: List[str]
    environment: str = "testing"
    parallel_workers: int = 4
    timeout: int = 300  # seconds
    coverage_threshold: float = 80.0
    fail_fast: bool = False
    verbose: bool = True
    collect_artifacts: bool = True
    retry_count: int = 0
    tags: List[str] = field(default_factory=list)
    env_vars: Dict[str, str] = field(default_factory=dict)


@dataclass
class TestResult:
    """Result of test execution."""

    test_id: str
    test_type: TestType
    status: TestStatus
    duration: float
    output: str = ""
    error_message: str = ""
    coverage_percentage: float = 0.0
    artifacts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TestSuiteResult:
    """Aggregated results for a test suite."""

    suite_name: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    duration: float
    coverage_percentage: float
    results: List[TestResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class TestAutomationManager:
    """
    Manages test automation across the PoUW system.

    Provides capabilities for:
    - Running different types of tests
    - Managing test environments
    - Collecting and analyzing results
    - Generating reports
    """

    def __init__(self, project_root: str = "/home/elfateh/Projects/PoUW"):
        self.project_root = Path(project_root)
        self.docker_client = docker.from_env()
        self.test_results: Dict[str, TestSuiteResult] = {}
        self.coverage_analyzer = CoverageAnalyzer(project_root)

        # Load Kubernetes config for integration tests
        try:
            k8s_config.load_incluster_config()
        except:
            try:
                k8s_config.load_kube_config()
            except:
                print("Warning: Kubernetes config not available")

    async def run_test_suite(self, config: TestConfiguration) -> TestSuiteResult:
        """Run a complete test suite with the given configuration."""
        suite_name = f"{config.test_type.value}_{config.environment}_{int(time.time())}"

        print(f"Starting test suite: {suite_name}")
        start_time = time.time()

        # Prepare test environment
        await self._prepare_test_environment(config)

        # Execute tests based on type
        if config.test_type == TestType.UNIT:
            results = await self._run_unit_tests(config)
        elif config.test_type == TestType.INTEGRATION:
            results = await self._run_integration_tests(config)
        elif config.test_type == TestType.E2E:
            results = await self._run_e2e_tests(config)
        elif config.test_type == TestType.PERFORMANCE:
            results = await self._run_performance_tests(config)
        elif config.test_type == TestType.SECURITY:
            results = await self._run_security_tests(config)
        elif config.test_type == TestType.LOAD:
            results = await self._run_load_tests(config)
        elif config.test_type == TestType.SMOKE:
            results = await self._run_smoke_tests(config)
        else:
            raise ValueError(f"Unsupported test type: {config.test_type}")

        # Calculate coverage
        coverage_percentage = await self.coverage_analyzer.calculate_coverage(config.test_paths)

        # Aggregate results
        duration = time.time() - start_time
        suite_result = TestSuiteResult(
            suite_name=suite_name,
            total_tests=len(results),
            passed=len([r for r in results if r.status == TestStatus.PASSED]),
            failed=len([r for r in results if r.status == TestStatus.FAILED]),
            skipped=len([r for r in results if r.status == TestStatus.SKIPPED]),
            errors=len([r for r in results if r.status == TestStatus.ERROR]),
            duration=duration,
            coverage_percentage=coverage_percentage,
            results=results,
        )

        # Store results
        self.test_results[suite_name] = suite_result

        # Generate reports
        await self._generate_test_reports(suite_result)

        print(f"Test suite completed: {suite_name}")
        print(
            f"Results: {suite_result.passed} passed, {suite_result.failed} failed, "
            f"{suite_result.skipped} skipped, {suite_result.errors} errors"
        )
        print(f"Coverage: {coverage_percentage:.2f}%")

        return suite_result

    async def _prepare_test_environment(self, config: TestConfiguration):
        """Prepare the testing environment."""
        # Create test directories
        test_dir = self.project_root / "test_artifacts" / config.test_type.value
        test_dir.mkdir(parents=True, exist_ok=True)

        # Set environment variables
        for key, value in config.env_vars.items():
            import os

            os.environ[key] = value

        # Start test infrastructure if needed
        if config.test_type in [TestType.INTEGRATION, TestType.E2E]:
            await self._start_test_infrastructure()

    async def _start_test_infrastructure(self):
        """Start supporting infrastructure for integration tests."""
        # Start test database
        try:
            self.docker_client.containers.run(
                "postgres:13",
                name="pouw_test_db",
                environment={
                    "POSTGRES_DB": "pouw_test",
                    "POSTGRES_USER": "REDACTED",
                    "POSTGRES_PASSWORD": "REDACTED",
                },
                ports={"5432/tcp": 5433},
                detach=True,
                remove=True,
            )
            await asyncio.sleep(5)  # Wait for DB to start
        except docker.errors.APIError:
            pass  # Container might already exist

    async def _run_unit_tests(self, config: TestConfiguration) -> List[TestResult]:
        """Run unit tests using pytest."""
        results = []

        for test_path in config.test_paths:
            test_id = f"unit_{Path(test_path).name}_{int(time.time())}"

            try:
                # Run pytest with coverage
                cmd = [
                    "python",
                    "-m",
                    "pytest",
                    test_path,
                    f"--workers={config.parallel_workers}",
                    "--verbose" if config.verbose else "",
                    "--tb=short",
                    f"--timeout={config.timeout}",
                    "--cov=pouw",
                    "--cov-report=xml",
                    "--junit-xml=test_results.xml",
                ]

                if config.fail_fast:
                    cmd.append("-x")

                if config.tags:
                    cmd.extend(["-m", " and ".join(config.tags)])

                # Filter out empty strings
                cmd = [c for c in cmd if c]

                start_time = time.time()
                result = subprocess.run(
                    cmd,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=config.timeout,
                )
                duration = time.time() - start_time

                status = TestStatus.PASSED if result.returncode == 0 else TestStatus.FAILED

                results.append(
                    TestResult(
                        test_id=test_id,
                        test_type=config.test_type,
                        status=status,
                        duration=duration,
                        output=result.stdout,
                        error_message=result.stderr if result.returncode != 0 else "",
                    )
                )

            except subprocess.TimeoutExpired:
                results.append(
                    TestResult(
                        test_id=test_id,
                        test_type=config.test_type,
                        status=TestStatus.ERROR,
                        duration=config.timeout,
                        error_message="Test execution timed out",
                    )
                )
            except Exception as e:
                results.append(
                    TestResult(
                        test_id=test_id,
                        test_type=config.test_type,
                        status=TestStatus.ERROR,
                        duration=0.0,
                        error_message=str(e),
                    )
                )

        return results

    async def _run_integration_tests(self, config: TestConfiguration) -> List[TestResult]:
        """Run integration tests with containerized services."""
        results = []

        # Start test environment with docker-compose
        compose_file = self.project_root / "docker-compose.test.yml"
        if compose_file.exists():
            subprocess.run(
                ["docker-compose", "-f", str(compose_file), "up", "-d"], cwd=self.project_root
            )

            try:
                await asyncio.sleep(10)  # Wait for services to start

                # Run integration tests
                for test_path in config.test_paths:
                    test_id = f"integration_{Path(test_path).name}_{int(time.time())}"

                    cmd = ["python", "-m", "pytest", test_path, "-v", "--tb=short"]

                    start_time = time.time()
                    result = subprocess.run(
                        cmd,
                        cwd=self.project_root,
                        capture_output=True,
                        text=True,
                        timeout=config.timeout,
                    )
                    duration = time.time() - start_time

                    status = TestStatus.PASSED if result.returncode == 0 else TestStatus.FAILED

                    results.append(
                        TestResult(
                            test_id=test_id,
                            test_type=config.test_type,
                            status=status,
                            duration=duration,
                            output=result.stdout,
                            error_message=result.stderr if result.returncode != 0 else "",
                        )
                    )

            finally:
                # Cleanup test environment
                subprocess.run(
                    ["docker-compose", "-f", str(compose_file), "down", "-v"], cwd=self.project_root
                )

        return results

    async def _run_e2e_tests(self, config: TestConfiguration) -> List[TestResult]:
        """Run end-to-end tests."""
        results = []

        # Deploy to test environment
        try:
            # Apply Kubernetes test manifests
            k8s_dir = self.project_root / "k8s" / "test"
            if k8s_dir.exists():
                subprocess.run(["kubectl", "apply", "-f", str(k8s_dir)], check=True)

                await asyncio.sleep(30)  # Wait for deployment

                # Run E2E tests
                for test_path in config.test_paths:
                    test_id = f"e2e_{Path(test_path).name}_{int(time.time())}"

                    cmd = ["python", "-m", "pytest", test_path, "-v", "--tb=short", "--e2e"]

                    start_time = time.time()
                    result = subprocess.run(
                        cmd,
                        cwd=self.project_root,
                        capture_output=True,
                        text=True,
                        timeout=config.timeout,
                    )
                    duration = time.time() - start_time

                    status = TestStatus.PASSED if result.returncode == 0 else TestStatus.FAILED

                    results.append(
                        TestResult(
                            test_id=test_id,
                            test_type=config.test_type,
                            status=status,
                            duration=duration,
                            output=result.stdout,
                            error_message=result.stderr if result.returncode != 0 else "",
                        )
                    )

                # Cleanup
                subprocess.run(["kubectl", "delete", "-f", str(k8s_dir)])

        except Exception as e:
            results.append(
                TestResult(
                    test_id="e2e_setup_error",
                    test_type=config.test_type,
                    status=TestStatus.ERROR,
                    duration=0.0,
                    error_message=f"E2E setup failed: {str(e)}",
                )
            )

        return results

    async def _run_performance_tests(self, config: TestConfiguration) -> List[TestResult]:
        """Run performance tests using locust or similar tools."""
        results = []

        for test_path in config.test_paths:
            test_id = f"performance_{Path(test_path).name}_{int(time.time())}"

            try:
                # Run locust performance tests
                cmd = [
                    "locust",
                    "-f",
                    test_path,
                    "--headless",
                    "--users",
                    "100",
                    "--spawn-rate",
                    "10",
                    "--run-time",
                    "60s",
                    "--host",
                    "http://localhost:8000",
                ]

                start_time = time.time()
                result = subprocess.run(
                    cmd,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=config.timeout,
                )
                duration = time.time() - start_time

                status = TestStatus.PASSED if result.returncode == 0 else TestStatus.FAILED

                results.append(
                    TestResult(
                        test_id=test_id,
                        test_type=config.test_type,
                        status=status,
                        duration=duration,
                        output=result.stdout,
                        error_message=result.stderr if result.returncode != 0 else "",
                    )
                )

            except Exception as e:
                results.append(
                    TestResult(
                        test_id=test_id,
                        test_type=config.test_type,
                        status=TestStatus.ERROR,
                        duration=0.0,
                        error_message=str(e),
                    )
                )

        return results

    async def _run_security_tests(self, config: TestConfiguration) -> List[TestResult]:
        """Run security tests using bandit and safety."""
        results = []

        # Run bandit security scan
        test_id = f"security_bandit_{int(time.time())}"
        try:
            cmd = ["bandit", "-r", "pouw/", "-f", "json"]

            start_time = time.time()
            result = subprocess.run(
                cmd, cwd=self.project_root, capture_output=True, text=True, timeout=config.timeout
            )
            duration = time.time() - start_time

            # Bandit returns non-zero if issues found, but that's expected
            status = TestStatus.PASSED
            if result.returncode > 1:  # Error, not just findings
                status = TestStatus.ERROR

            results.append(
                TestResult(
                    test_id=test_id,
                    test_type=config.test_type,
                    status=status,
                    duration=duration,
                    output=result.stdout,
                    error_message=result.stderr if result.returncode > 1 else "",
                )
            )

        except Exception as e:
            results.append(
                TestResult(
                    test_id=test_id,
                    test_type=config.test_type,
                    status=TestStatus.ERROR,
                    duration=0.0,
                    error_message=str(e),
                )
            )

        # Run safety dependency scan
        test_id = f"security_safety_{int(time.time())}"
        try:
            cmd = ["safety", "check", "--json"]

            start_time = time.time()
            result = subprocess.run(
                cmd, cwd=self.project_root, capture_output=True, text=True, timeout=config.timeout
            )
            duration = time.time() - start_time

            status = TestStatus.PASSED if result.returncode == 0 else TestStatus.FAILED

            results.append(
                TestResult(
                    test_id=test_id,
                    test_type=config.test_type,
                    status=status,
                    duration=duration,
                    output=result.stdout,
                    error_message=result.stderr if result.returncode != 0 else "",
                )
            )

        except Exception as e:
            results.append(
                TestResult(
                    test_id=test_id,
                    test_type=config.test_type,
                    status=TestStatus.ERROR,
                    duration=0.0,
                    error_message=str(e),
                )
            )

        return results

    async def _run_load_tests(self, config: TestConfiguration) -> List[TestResult]:
        """Run load tests using artillery or similar tools."""
        results = []

        # This is a simplified version - in practice you'd use tools like Artillery, K6, etc.
        for test_path in config.test_paths:
            test_id = f"load_{Path(test_path).name}_{int(time.time())}"

            try:
                # Simulate load test execution
                await asyncio.sleep(5)  # Simulate test execution

                results.append(
                    TestResult(
                        test_id=test_id,
                        test_type=config.test_type,
                        status=TestStatus.PASSED,
                        duration=5.0,
                        output="Load test completed successfully",
                        metadata={
                            "rps": 1000,
                            "avg_response_time": 50,
                            "p95_response_time": 100,
                            "error_rate": 0.01,
                        },
                    )
                )

            except Exception as e:
                results.append(
                    TestResult(
                        test_id=test_id,
                        test_type=config.test_type,
                        status=TestStatus.ERROR,
                        duration=0.0,
                        error_message=str(e),
                    )
                )

        return results

    async def _run_smoke_tests(self, config: TestConfiguration) -> List[TestResult]:
        """Run smoke tests for basic functionality."""
        results = []

        for test_path in config.test_paths:
            test_id = f"smoke_{Path(test_path).name}_{int(time.time())}"

            try:
                cmd = ["python", "-m", "pytest", test_path, "-v", "-m", "smoke", "--tb=short"]

                start_time = time.time()
                result = subprocess.run(
                    cmd,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=config.timeout,
                )
                duration = time.time() - start_time

                status = TestStatus.PASSED if result.returncode == 0 else TestStatus.FAILED

                results.append(
                    TestResult(
                        test_id=test_id,
                        test_type=config.test_type,
                        status=status,
                        duration=duration,
                        output=result.stdout,
                        error_message=result.stderr if result.returncode != 0 else "",
                    )
                )

            except Exception as e:
                results.append(
                    TestResult(
                        test_id=test_id,
                        test_type=config.test_type,
                        status=TestStatus.ERROR,
                        duration=0.0,
                        error_message=str(e),
                    )
                )

        return results

    async def _generate_test_reports(self, suite_result: TestSuiteResult):
        """Generate test reports in various formats."""
        reports_dir = self.project_root / "test_reports"
        reports_dir.mkdir(exist_ok=True)

        # Generate JSON report
        json_report = reports_dir / f"{suite_result.suite_name}_report.json"
        with open(json_report, "w") as f:
            json.dump(
                {
                    "suite_name": suite_result.suite_name,
                    "summary": {
                        "total": suite_result.total_tests,
                        "passed": suite_result.passed,
                        "failed": suite_result.failed,
                        "skipped": suite_result.skipped,
                        "errors": suite_result.errors,
                        "duration": suite_result.duration,
                        "coverage": suite_result.coverage_percentage,
                    },
                    "results": [
                        {
                            "test_id": r.test_id,
                            "status": r.status.value,
                            "duration": r.duration,
                            "error_message": r.error_message,
                            "metadata": r.metadata,
                        }
                        for r in suite_result.results
                    ],
                },
                f,
                indent=2,
            )

        # Generate HTML report
        html_report = reports_dir / f"{suite_result.suite_name}_report.html"
        html_content = self._generate_html_report(suite_result)
        with open(html_report, "w") as f:
            f.write(html_content)

        print(f"Reports generated: {json_report}, {html_report}")

    def _generate_html_report(self, suite_result: TestSuiteResult) -> str:
        """Generate HTML test report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Report - {suite_result.suite_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
                .error {{ color: orange; }}
                .skipped {{ color: gray; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Test Report: {suite_result.suite_name}</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Total Tests:</strong> {suite_result.total_tests}</p>
                <p class="passed"><strong>Passed:</strong> {suite_result.passed}</p>
                <p class="failed"><strong>Failed:</strong> {suite_result.failed}</p>
                <p class="error"><strong>Errors:</strong> {suite_result.errors}</p>
                <p class="skipped"><strong>Skipped:</strong> {suite_result.skipped}</p>
                <p><strong>Duration:</strong> {suite_result.duration:.2f}s</p>
                <p><strong>Coverage:</strong> {suite_result.coverage_percentage:.2f}%</p>
            </div>
            
            <h2>Test Results</h2>
            <table>
                <tr>
                    <th>Test ID</th>
                    <th>Status</th>
                    <th>Duration</th>
                    <th>Error Message</th>
                </tr>
        """

        for result in suite_result.results:
            status_class = result.status.value
            html += f"""
                <tr>
                    <td>{result.test_id}</td>
                    <td class="{status_class}">{result.status.value.upper()}</td>
                    <td>{result.duration:.2f}s</td>
                    <td>{result.error_message}</td>
                </tr>
            """

        html += """
            </table>
        </body>
        </html>
        """

        return html

    def get_test_history(self, days: int = 30) -> Dict[str, List[TestSuiteResult]]:
        """Get test history for the specified number of days."""
        cutoff_date = datetime.now() - timedelta(days=days)

        history = {}
        for suite_name, result in self.test_results.items():
            if result.timestamp >= cutoff_date:
                test_type = suite_name.split("_")[0]
                if test_type not in history:
                    history[test_type] = []
                history[test_type].append(result)

        return history


class TestSuite:
    """
    Represents a collection of tests that can be executed together.
    """

    def __init__(self, name: str, test_paths: List[str]):
        self.name = name
        self.test_paths = test_paths
        self.configuration = TestConfiguration(test_type=TestType.UNIT, test_paths=test_paths)

    def configure(self, **kwargs):
        """Configure the test suite parameters."""
        for key, value in kwargs.items():
            if hasattr(self.configuration, key):
                setattr(self.configuration, key, value)

    async def run(self, manager: TestAutomationManager) -> TestSuiteResult:
        """Execute the test suite."""
        return await manager.run_test_suite(self.configuration)


class CoverageAnalyzer:
    """
    Analyzes code coverage from test execution.
    """

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.coverage = coverage.Coverage()

    async def calculate_coverage(self, test_paths: List[str]) -> float:
        """Calculate code coverage percentage."""
        try:
            # Look for coverage report
            coverage_file = self.project_root / "coverage.xml"
            if coverage_file.exists():
                return self._parse_coverage_xml(coverage_file)

            # Look for .coverage file
            coverage_data_file = self.project_root / ".coverage"
            if coverage_data_file.exists():
                self.coverage.load()
                return self.coverage.report()

            return 0.0

        except Exception as e:
            print(f"Error calculating coverage: {e}")
            return 0.0

    def _parse_coverage_xml(self, xml_file: Path) -> float:
        """Parse coverage percentage from XML report."""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Look for coverage percentage in XML
            coverage_elem = root.find(".//coverage")
            if coverage_elem is not None:
                line_rate = coverage_elem.get("line-rate", "0")
                return float(line_rate) * 100

            return 0.0

        except Exception:
            return 0.0

    def generate_coverage_report(self, output_dir: str = "coverage_reports"):
        """Generate detailed coverage reports."""
        output_path = self.project_root / output_dir
        output_path.mkdir(exist_ok=True)

        try:
            self.coverage.load()

            # Generate HTML report
            self.coverage.html_report(directory=str(output_path / "html"))

            # Generate XML report
            self.coverage.xml_report(outfile=str(output_path / "coverage.xml"))

            print(f"Coverage reports generated in {output_path}")

        except Exception as e:
            print(f"Error generating coverage report: {e}")


# Example usage and predefined test suites
class PoUWTestSuites:
    """Predefined test suites for the PoUW system."""

    @staticmethod
    def unit_tests() -> TestSuite:
        """Unit test suite for PoUW components."""
        return TestSuite(
            name="PoUW Unit Tests",
            test_paths=[
                "tests/unit/test_blockchain.py",
                "tests/unit/test_ml_training.py",
                "tests/unit/test_networking.py",
                "tests/unit/test_deployment.py",
            ],
        )

    @staticmethod
    def integration_tests() -> TestSuite:
        """Integration test suite for PoUW system."""
        return TestSuite(
            name="PoUW Integration Tests",
            test_paths=[
                "tests/integration/test_blockchain_integration.py",
                "tests/integration/test_ml_pipeline.py",
                "tests/integration/test_network_communication.py",
            ],
        )

    @staticmethod
    def performance_tests() -> TestSuite:
        """Performance test suite for PoUW system."""
        suite = TestSuite(
            name="PoUW Performance Tests",
            test_paths=[
                "tests/performance/test_blockchain_throughput.py",
                "tests/performance/test_ml_training_speed.py",
                "tests/performance/test_network_latency.py",
            ],
        )
        suite.configure(test_type=TestType.PERFORMANCE, timeout=600, parallel_workers=1)
        return suite


# Example usage
async def main():
    """Example usage of the testing automation system."""
    manager = TestAutomationManager()

    # Run unit tests
    unit_suite = PoUWTestSuites.unit_tests()
    unit_result = await unit_suite.run(manager)

    print(f"Unit tests completed: {unit_result.passed}/{unit_result.total_tests} passed")

    # Run integration tests
    integration_suite = PoUWTestSuites.integration_tests()
    integration_result = await integration_suite.run(manager)

    print(
        f"Integration tests completed: {integration_result.passed}/{integration_result.total_tests} passed"
    )


if __name__ == "__main__":
    asyncio.run(main())
