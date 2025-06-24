"""
Quality Assurance Module for PoUW CI/CD Pipeline

This module provides comprehensive quality assurance capabilities including:
- Code quality analysis and metrics
- Security scanning and vulnerability assessment
- Performance profiling and analysis
- Compliance checking and reporting
- Static analysis and linting
- Documentation quality assessment
"""

import asyncio
import json
import subprocess
import time
import re
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import tempfile
import xml.etree.ElementTree as ET

import ast
import radon.complexity as radon_cc
import radon.metrics as radon_metrics
from bandit.core import manager as bandit_manager
import safety
import pylint.lint
import flake8.api.legacy as flake8


class QualityMetric(Enum):
    """Quality metrics tracked by the system."""
    CODE_COVERAGE = "code_coverage"
    CYCLOMATIC_COMPLEXITY = "cyclomatic_complexity"
    MAINTAINABILITY_INDEX = "maintainability_index"
    LINES_OF_CODE = "lines_of_code"
    DUPLICATED_CODE = "duplicated_code"
    SECURITY_VULNERABILITIES = "security_vulnerabilities"
    PERFORMANCE_SCORE = "performance_score"
    DOCUMENTATION_COVERAGE = "documentation_coverage"
    TYPE_COVERAGE = "type_coverage"
    TEST_COVERAGE = "test_coverage"


class SeverityLevel(Enum):
    """Severity levels for quality issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class QualityCheckType(Enum):
    """Types of quality checks."""
    STATIC_ANALYSIS = "static_analysis"
    SECURITY_SCAN = "security_scan"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    CODE_STYLE = "code_style"
    COMPLEXITY_ANALYSIS = "complexity_analysis"
    DEPENDENCY_SCAN = "dependency_scan"
    LICENSE_COMPLIANCE = "license_compliance"
    DOCUMENTATION_CHECK = "documentation_check"


@dataclass
class QualityIssue:
    """Represents a quality issue found during analysis."""
    check_type: QualityCheckType
    severity: SeverityLevel
    message: str
    file_path: str
    line_number: int = 0
    column_number: int = 0
    rule_id: str = ""
    category: str = ""
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityMetrics:
    """Quality metrics for a codebase."""
    timestamp: datetime
    total_lines: int = 0
    code_lines: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    cyclomatic_complexity: float = 0.0
    maintainability_index: float = 0.0
    code_coverage: float = 0.0
    test_coverage: float = 0.0
    documentation_coverage: float = 0.0
    type_coverage: float = 0.0
    security_score: float = 0.0
    performance_score: float = 0.0
    duplicated_code_ratio: float = 0.0
    technical_debt_ratio: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class QualityReport:
    """Comprehensive quality assessment report."""
    project_name: str
    version: str
    timestamp: datetime
    metrics: QualityMetrics
    issues: List[QualityIssue] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    quality_gate_passed: bool = False
    grade: str = "F"  # A, B, C, D, F


@dataclass
class QualityGateRule:
    """Rule for quality gate evaluation."""
    metric: QualityMetric
    operator: str  # >=, <=, ==, !=
    threshold: float
    severity: SeverityLevel = SeverityLevel.MEDIUM
    description: str = ""


@dataclass
class QualityConfiguration:
    """Configuration for quality analysis."""
    include_patterns: List[str] = field(default_factory=lambda: ["**/*.py"])
    exclude_patterns: List[str] = field(default_factory=lambda: ["**/test_*.py", "**/tests/**"])
    max_complexity: int = 10
    min_coverage: float = 80.0
    enable_security_scan: bool = True
    enable_performance_analysis: bool = True
    enable_style_check: bool = True
    enable_type_check: bool = True
    quality_gates: List[QualityGateRule] = field(default_factory=list)
    custom_rules: Dict[str, Any] = field(default_factory=dict)


class CodeQualityManager:
    """
    Manages code quality analysis and reporting.
    
    Provides capabilities for:
    - Static code analysis
    - Code metrics calculation
    - Quality gate evaluation
    - Trend analysis and reporting
    """

    def __init__(self, project_root: str = "/home/elfateh/Projects/PoUW"):
        self.project_root = Path(project_root)
        self.reports: Dict[str, QualityReport] = {}
        self.configuration = QualityConfiguration()
        self._setup_default_quality_gates()

    def _setup_default_quality_gates(self):
        """Setup default quality gate rules."""
        self.configuration.quality_gates = [
            QualityGateRule(
                metric=QualityMetric.CODE_COVERAGE,
                operator=">=",
                threshold=80.0,
                severity=SeverityLevel.HIGH,
                description="Code coverage must be at least 80%"
            ),
            QualityGateRule(
                metric=QualityMetric.CYCLOMATIC_COMPLEXITY,
                operator="<=",
                threshold=10.0,
                severity=SeverityLevel.MEDIUM,
                description="Average cyclomatic complexity should not exceed 10"
            ),
            QualityGateRule(
                metric=QualityMetric.MAINTAINABILITY_INDEX,
                operator=">=",
                threshold=70.0,
                severity=SeverityLevel.MEDIUM,
                description="Maintainability index should be at least 70"
            ),
            QualityGateRule(
                metric=QualityMetric.SECURITY_VULNERABILITIES,
                operator="==",
                threshold=0.0,
                severity=SeverityLevel.CRITICAL,
                description="No critical security vulnerabilities allowed"
            )
        ]

    async def analyze_quality(self, project_name: str = "PoUW", version: str = "latest") -> QualityReport:
        """Perform comprehensive quality analysis."""
        print(f"Starting quality analysis for {project_name} v{version}")
        
        # Initialize report
        report = QualityReport(
            project_name=project_name,
            version=version,
            timestamp=datetime.now(),
            metrics=QualityMetrics(timestamp=datetime.now())
        )
        
        # Run various quality checks
        await self._analyze_code_metrics(report)
        await self._analyze_complexity(report)
        await self._analyze_style_issues(report)
        await self._analyze_type_coverage(report)
        await self._analyze_documentation(report)
        
        # Calculate summary and grade
        self._calculate_summary(report)
        self._evaluate_quality_gates(report)
        self._generate_recommendations(report)
        
        # Store report
        report_id = f"{project_name}_{version}_{int(time.time())}"
        self.reports[report_id] = report
        
        print(f"Quality analysis completed: Grade {report.grade}")
        return report

    async def _analyze_code_metrics(self, report: QualityReport):
        """Analyze basic code metrics."""
        total_lines = 0
        code_lines = 0
        comment_lines = 0
        blank_lines = 0
        
        # Find Python files
        python_files = list(self.project_root.rglob("*.py"))
        python_files = [f for f in python_files if self._should_include_file(f)]
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                file_total = len(lines)
                file_blank = len([line for line in lines if line.strip() == ""])
                file_comment = len([line for line in lines if line.strip().startswith("#")])
                file_code = file_total - file_blank - file_comment
                
                total_lines += file_total
                code_lines += file_code
                comment_lines += file_comment
                blank_lines += file_blank
                
            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")
        
        report.metrics.total_lines = total_lines
        report.metrics.code_lines = code_lines
        report.metrics.comment_lines = comment_lines
        report.metrics.blank_lines = blank_lines

    async def _analyze_complexity(self, report: QualityReport):
        """Analyze code complexity metrics."""
        total_complexity = 0
        function_count = 0
        
        python_files = list(self.project_root.rglob("*.py"))
        python_files = [f for f in python_files if self._should_include_file(f)]
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Calculate cyclomatic complexity
                complexity_data = radon_cc.cc_visit(content)
                for item in complexity_data:
                    total_complexity += item.complexity
                    function_count += 1
                    
                    if item.complexity > self.configuration.max_complexity:
                        report.issues.append(QualityIssue(
                            check_type=QualityCheckType.COMPLEXITY_ANALYSIS,
                            severity=SeverityLevel.MEDIUM,
                            message=f"High complexity function '{item.name}' (complexity: {item.complexity})",
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=item.lineno,
                            rule_id="CC001",
                            category="complexity"
                        ))
                
                # Calculate maintainability index
                mi_data = radon_metrics.mi_visit(content, multi=True)
                if mi_data:
                    report.metrics.maintainability_index = mi_data
                
            except Exception as e:
                print(f"Error analyzing complexity for {file_path}: {e}")
        
        if function_count > 0:
            report.metrics.cyclomatic_complexity = total_complexity / function_count

    async def _analyze_style_issues(self, report: QualityReport):
        """Analyze code style issues using flake8 and pylint."""
        if not self.configuration.enable_style_check:
            return
        
        # Run flake8
        await self._run_flake8_analysis(report)
        
        # Run pylint
        await self._run_pylint_analysis(report)

    async def _run_flake8_analysis(self, report: QualityReport):
        """Run flake8 style analysis."""
        try:
            cmd = [
                "flake8", 
                str(self.project_root / "pouw"),
                "--format=json",
                "--max-line-length=88",
                "--ignore=E203,W503"
            ]
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root
            )
            stdout, stderr = await proc.communicate()
            
            if stdout:
                # Parse flake8 output (simplified - flake8 doesn't output JSON by default)
                lines = stdout.decode().split('\n')
                for line in lines:
                    if ':' in line and not line.startswith('#'):
                        parts = line.split(':', 3)
                        if len(parts) >= 4:
                            file_path, line_num, col_num, message = parts
                            
                            report.issues.append(QualityIssue(
                                check_type=QualityCheckType.CODE_STYLE,
                                severity=SeverityLevel.LOW,
                                message=message.strip(),
                                file_path=file_path,
                                line_number=int(line_num) if line_num.isdigit() else 0,
                                column_number=int(col_num) if col_num.isdigit() else 0,
                                rule_id="F8",
                                category="style"
                            ))
            
        except Exception as e:
            print(f"Error running flake8: {e}")

    async def _run_pylint_analysis(self, report: QualityReport):
        """Run pylint analysis."""
        try:
            cmd = [
                "pylint",
                str(self.project_root / "pouw"),
                "--output-format=json",
                "--disable=C0103,R0903",  # Disable some common warnings
                "--exit-zero"
            ]
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root
            )
            stdout, stderr = await proc.communicate()
            
            if stdout:
                try:
                    pylint_results = json.loads(stdout.decode())
                    
                    for issue in pylint_results:
                        severity_map = {
                            'error': SeverityLevel.HIGH,
                            'warning': SeverityLevel.MEDIUM,
                            'refactor': SeverityLevel.LOW,
                            'convention': SeverityLevel.LOW
                        }
                        
                        report.issues.append(QualityIssue(
                            check_type=QualityCheckType.STATIC_ANALYSIS,
                            severity=severity_map.get(issue.get('type', 'info'), SeverityLevel.INFO),
                            message=issue.get('message', ''),
                            file_path=issue.get('path', ''),
                            line_number=issue.get('line', 0),
                            column_number=issue.get('column', 0),
                            rule_id=issue.get('symbol', ''),
                            category="pylint"
                        ))
                        
                except json.JSONDecodeError:
                    print("Error parsing pylint JSON output")
            
        except Exception as e:
            print(f"Error running pylint: {e}")

    async def _analyze_type_coverage(self, report: QualityReport):
        """Analyze type annotation coverage."""
        if not self.configuration.enable_type_check:
            return
        
        try:
            # Run mypy for type checking
            cmd = [
                "mypy",
                str(self.project_root / "pouw"),
                "--ignore-missing-imports",
                "--json-report", "/tmp/mypy_report"
            ]
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root
            )
            stdout, stderr = await proc.communicate()
            
            # Parse mypy output for type issues
            if stderr:
                lines = stderr.decode().split('\n')
                for line in lines:
                    if ':' in line and 'error:' in line:
                        parts = line.split(':', 2)
                        if len(parts) >= 3:
                            file_and_line = parts[0]
                            message = parts[2].strip()
                            
                            if '(' in file_and_line:
                                file_path, line_info = file_and_line.rsplit('(', 1)
                                line_num = line_info.rstrip(')').split(',')[0]
                                
                                report.issues.append(QualityIssue(
                                    check_type=QualityCheckType.STATIC_ANALYSIS,
                                    severity=SeverityLevel.MEDIUM,
                                    message=f"Type error: {message}",
                                    file_path=file_path.strip(),
                                    line_number=int(line_num) if line_num.isdigit() else 0,
                                    rule_id="mypy",
                                    category="type_checking"
                                ))
            
            # Calculate type coverage (simplified)
            type_coverage = await self._calculate_type_coverage()
            report.metrics.type_coverage = type_coverage
            
        except Exception as e:
            print(f"Error running type analysis: {e}")

    async def _calculate_type_coverage(self) -> float:
        """Calculate type annotation coverage percentage."""
        total_functions = 0
        typed_functions = 0
        
        python_files = list(self.project_root.rglob("*.py"))
        python_files = [f for f in python_files if self._should_include_file(f)]
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        total_functions += 1
                        
                        # Check if function has type annotations
                        has_return_annotation = node.returns is not None
                        has_arg_annotations = any(arg.annotation is not None for arg in node.args.args)
                        
                        if has_return_annotation or has_arg_annotations:
                            typed_functions += 1
                            
            except Exception as e:
                print(f"Error calculating type coverage for {file_path}: {e}")
        
        return (typed_functions / total_functions * 100) if total_functions > 0 else 0.0

    async def _analyze_documentation(self, report: QualityReport):
        """Analyze documentation coverage."""
        total_functions = 0
        documented_functions = 0
        
        python_files = list(self.project_root.rglob("*.py"))
        python_files = [f for f in python_files if self._should_include_file(f)]
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        total_functions += 1
                        
                        # Check if has docstring
                        if (node.body and 
                            isinstance(node.body[0], ast.Expr) and 
                            isinstance(node.body[0].value, ast.Str)):
                            documented_functions += 1
                        elif (node.body and 
                              isinstance(node.body[0], ast.Expr) and 
                              isinstance(node.body[0].value, ast.Constant) and 
                              isinstance(node.body[0].value.value, str)):
                            documented_functions += 1
                            
            except Exception as e:
                print(f"Error analyzing documentation for {file_path}: {e}")
        
        report.metrics.documentation_coverage = (documented_functions / total_functions * 100) if total_functions > 0 else 0.0

    def _should_include_file(self, file_path: Path) -> bool:
        """Check if file should be included in analysis."""
        file_str = str(file_path.relative_to(self.project_root))
        
        # Check include patterns
        include_match = any(file_path.match(pattern) for pattern in self.configuration.include_patterns)
        if not include_match:
            return False
        
        # Check exclude patterns
        exclude_match = any(file_path.match(pattern) for pattern in self.configuration.exclude_patterns)
        if exclude_match:
            return False
        
        return True

    def _calculate_summary(self, report: QualityReport):
        """Calculate summary statistics."""
        issues_by_severity = {}
        issues_by_type = {}
        
        for issue in report.issues:
            # Count by severity
            severity = issue.severity.value
            issues_by_severity[severity] = issues_by_severity.get(severity, 0) + 1
            
            # Count by type
            check_type = issue.check_type.value
            issues_by_type[check_type] = issues_by_type.get(check_type, 0) + 1
        
        report.summary = {
            "total_issues": len(report.issues),
            "issues_by_severity": issues_by_severity,
            "issues_by_type": issues_by_type,
            "quality_score": self._calculate_quality_score(report),
            "technical_debt_hours": self._estimate_technical_debt(report)
        }

    def _calculate_quality_score(self, report: QualityReport) -> float:
        """Calculate overall quality score (0-100)."""
        score = 100.0
        
        # Deduct points for issues
        severity_weights = {
            SeverityLevel.CRITICAL: 20,
            SeverityLevel.HIGH: 10,
            SeverityLevel.MEDIUM: 5,
            SeverityLevel.LOW: 1,
            SeverityLevel.INFO: 0.5
        }
        
        for issue in report.issues:
            weight = severity_weights.get(issue.severity, 1)
            score -= weight
        
        # Factor in metrics
        if report.metrics.code_coverage < 80:
            score -= (80 - report.metrics.code_coverage) * 0.5
        
        if report.metrics.cyclomatic_complexity > 10:
            score -= (report.metrics.cyclomatic_complexity - 10) * 2
        
        return max(0.0, score)

    def _estimate_technical_debt(self, report: QualityReport) -> float:
        """Estimate technical debt in hours."""
        debt_hours = 0.0
        
        # Estimate based on issue types
        debt_estimates = {
            SeverityLevel.CRITICAL: 8,    # 8 hours to fix critical issues
            SeverityLevel.HIGH: 4,        # 4 hours for high severity
            SeverityLevel.MEDIUM: 2,      # 2 hours for medium
            SeverityLevel.LOW: 0.5,       # 30 minutes for low
            SeverityLevel.INFO: 0.1       # 6 minutes for info
        }
        
        for issue in report.issues:
            estimate = debt_estimates.get(issue.severity, 1.0)
            debt_hours += estimate
        
        report.metrics.technical_debt_ratio = debt_hours / max(1, report.metrics.code_lines) * 1000
        
        return debt_hours

    def _evaluate_quality_gates(self, report: QualityReport):
        """Evaluate quality gate rules."""
        passed_gates = 0
        total_gates = len(self.configuration.quality_gates)
        
        for gate in self.configuration.quality_gates:
            metric_value = self._get_metric_value(report.metrics, gate.metric)
            
            if self._evaluate_gate_rule(metric_value, gate.operator, gate.threshold):
                passed_gates += 1
            else:
                # Add quality gate failure as an issue
                report.issues.append(QualityIssue(
                    check_type=QualityCheckType.STATIC_ANALYSIS,
                    severity=gate.severity,
                    message=f"Quality gate failed: {gate.description} (actual: {metric_value}, threshold: {gate.threshold})",
                    file_path="",
                    rule_id="QG001",
                    category="quality_gate"
                ))
        
        report.quality_gate_passed = passed_gates == total_gates
        
        # Calculate grade
        if report.quality_gate_passed and report.summary["quality_score"] >= 90:
            report.grade = "A"
        elif report.summary["quality_score"] >= 80:
            report.grade = "B"
        elif report.summary["quality_score"] >= 70:
            report.grade = "C"
        elif report.summary["quality_score"] >= 60:
            report.grade = "D"
        else:
            report.grade = "F"

    def _get_metric_value(self, metrics: QualityMetrics, metric: QualityMetric) -> float:
        """Get metric value from metrics object."""
        metric_map = {
            QualityMetric.CODE_COVERAGE: metrics.code_coverage,
            QualityMetric.CYCLOMATIC_COMPLEXITY: metrics.cyclomatic_complexity,
            QualityMetric.MAINTAINABILITY_INDEX: metrics.maintainability_index,
            QualityMetric.LINES_OF_CODE: metrics.code_lines,
            QualityMetric.DOCUMENTATION_COVERAGE: metrics.documentation_coverage,
            QualityMetric.TYPE_COVERAGE: metrics.type_coverage,
            QualityMetric.SECURITY_VULNERABILITIES: len([i for i in self.reports.get(list(self.reports.keys())[-1], QualityReport("", "", datetime.now(), QualityMetrics(datetime.now()))).issues if i.check_type == QualityCheckType.SECURITY_SCAN and i.severity == SeverityLevel.CRITICAL])
        }
        
        return metric_map.get(metric, 0.0)

    def _evaluate_gate_rule(self, value: float, operator: str, threshold: float) -> bool:
        """Evaluate a quality gate rule."""
        if operator == ">=":
            return value >= threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "==":
            return value == threshold
        elif operator == "!=":
            return value != threshold
        elif operator == ">":
            return value > threshold
        elif operator == "<":
            return value < threshold
        else:
            return False

    def _generate_recommendations(self, report: QualityReport):
        """Generate recommendations for quality improvement."""
        recommendations = []
        
        # Coverage recommendations
        if report.metrics.code_coverage < 80:
            recommendations.append(
                f"Increase test coverage from {report.metrics.code_coverage:.1f}% to at least 80%. "
                "Add unit tests for uncovered code paths."
            )
        
        # Complexity recommendations
        if report.metrics.cyclomatic_complexity > 10:
            recommendations.append(
                f"Reduce cyclomatic complexity from {report.metrics.cyclomatic_complexity:.1f} to below 10. "
                "Consider refactoring complex functions into smaller, more focused methods."
            )
        
        # Documentation recommendations
        if report.metrics.documentation_coverage < 80:
            recommendations.append(
                f"Improve documentation coverage from {report.metrics.documentation_coverage:.1f}% to at least 80%. "
                "Add docstrings to functions, classes, and modules."
            )
        
        # Type annotation recommendations
        if report.metrics.type_coverage < 70:
            recommendations.append(
                f"Improve type annotation coverage from {report.metrics.type_coverage:.1f}% to at least 70%. "
                "Add type hints to function parameters and return values."
            )
        
        # Security recommendations
        critical_security_issues = [i for i in report.issues if i.check_type == QualityCheckType.SECURITY_SCAN and i.severity == SeverityLevel.CRITICAL]
        if critical_security_issues:
            recommendations.append(
                f"Address {len(critical_security_issues)} critical security vulnerabilities immediately. "
                "Review and fix all security-related issues."
            )
        
        # Style recommendations
        style_issues = [i for i in report.issues if i.check_type == QualityCheckType.CODE_STYLE]
        if len(style_issues) > 50:
            recommendations.append(
                f"Fix {len(style_issues)} code style issues. "
                "Consider using automated code formatters like black and isort."
            )
        
        report.recommendations = recommendations

    def get_quality_trends(self, days: int = 30) -> Dict[str, List[Tuple[datetime, float]]]:
        """Get quality trends over time."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        trends = {
            "quality_score": [],
            "code_coverage": [],
            "complexity": [],
            "total_issues": []
        }
        
        for report in self.reports.values():
            if report.timestamp >= cutoff_date:
                trends["quality_score"].append((report.timestamp, report.summary.get("quality_score", 0)))
                trends["code_coverage"].append((report.timestamp, report.metrics.code_coverage))
                trends["complexity"].append((report.timestamp, report.metrics.cyclomatic_complexity))
                trends["total_issues"].append((report.timestamp, len(report.issues)))
        
        # Sort by timestamp
        for key in trends:
            trends[key].sort(key=lambda x: x[0])
        
        return trends

    async def generate_quality_report_html(self, report: QualityReport, output_path: str = "quality_report.html"):
        """Generate HTML quality report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Quality Report - {report.project_name} v{report.version}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f5f5f5; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .grade {{ font-size: 48px; font-weight: bold; text-align: center; }}
                .grade.A {{ color: #28a745; }}
                .grade.B {{ color: #6f42c1; }}
                .grade.C {{ color: #fd7e14; }}
                .grade.D {{ color: #dc3545; }}
                .grade.F {{ color: #6c757d; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
                .metric-card {{ background: white; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
                .issues {{ margin-top: 20px; }}
                .issue {{ padding: 10px; margin: 5px 0; border-left: 4px solid #ddd; }}
                .issue.critical {{ border-left-color: #dc3545; }}
                .issue.high {{ border-left-color: #fd7e14; }}
                .issue.medium {{ border-left-color: #ffc107; }}
                .issue.low {{ border-left-color: #28a745; }}
                .recommendations {{ background: #e7f3ff; padding: 15px; border-radius: 5px; margin-top: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Quality Report: {report.project_name} v{report.version}</h1>
                <p><strong>Generated:</strong> {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <div class="grade {report.grade}">{report.grade}</div>
                <p><strong>Quality Gate:</strong> {'PASSED' if report.quality_gate_passed else 'FAILED'}</p>
            </div>
            
            <h2>Quality Metrics</h2>
            <div class="metrics">
                <div class="metric-card">
                    <h3>Code Coverage</h3>
                    <p>{report.metrics.code_coverage:.1f}%</p>
                </div>
                <div class="metric-card">
                    <h3>Cyclomatic Complexity</h3>
                    <p>{report.metrics.cyclomatic_complexity:.1f}</p>
                </div>
                <div class="metric-card">
                    <h3>Lines of Code</h3>
                    <p>{report.metrics.code_lines:,}</p>
                </div>
                <div class="metric-card">
                    <h3>Documentation Coverage</h3>
                    <p>{report.metrics.documentation_coverage:.1f}%</p>
                </div>
                <div class="metric-card">
                    <h3>Type Coverage</h3>
                    <p>{report.metrics.type_coverage:.1f}%</p>
                </div>
                <div class="metric-card">
                    <h3>Quality Score</h3>
                    <p>{report.summary.get('quality_score', 0):.1f}/100</p>
                </div>
            </div>
            
            <h2>Issues Summary</h2>
            <table>
                <tr>
                    <th>Severity</th>
                    <th>Count</th>
                </tr>
        """
        
        for severity, count in report.summary.get("issues_by_severity", {}).items():
            html_content += f"<tr><td>{severity.title()}</td><td>{count}</td></tr>"
        
        html_content += """
            </table>
            
            <h2>Top Issues</h2>
            <div class="issues">
        """
        
        # Show top 20 issues
        sorted_issues = sorted(report.issues, key=lambda x: ["critical", "high", "medium", "low", "info"].index(x.severity.value))
        for issue in sorted_issues[:20]:
            html_content += f"""
                <div class="issue {issue.severity.value}">
                    <strong>{issue.severity.value.upper()}</strong> - {issue.message}<br>
                    <small>{issue.file_path}:{issue.line_number} ({issue.check_type.value})</small>
                </div>
            """
        
        html_content += """
            </div>
            
            <div class="recommendations">
                <h2>Recommendations</h2>
                <ul>
        """
        
        for recommendation in report.recommendations:
            html_content += f"<li>{recommendation}</li>"
        
        html_content += """
                </ul>
            </div>
        </body>
        </html>
        """
        
        output_file = self.project_root / output_path
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Quality report generated: {output_file}")


class SecurityScanner:
    """
    Specialized security scanning and vulnerability assessment.
    """

    def __init__(self, project_root: str = "/home/elfateh/Projects/PoUW"):
        self.project_root = Path(project_root)
        self.vulnerabilities: List[QualityIssue] = []

    async def scan_security_vulnerabilities(self) -> List[QualityIssue]:
        """Perform comprehensive security vulnerability scanning."""
        print("Starting security vulnerability scan...")
        
        vulnerabilities = []
        
        # Run bandit security scan
        vulnerabilities.extend(await self._run_bandit_scan())
        
        # Run safety dependency scan
        vulnerabilities.extend(await self._run_safety_scan())
        
        # Run custom security checks
        vulnerabilities.extend(await self._run_custom_security_checks())
        
        self.vulnerabilities = vulnerabilities
        
        print(f"Security scan completed: {len(vulnerabilities)} vulnerabilities found")
        return vulnerabilities

    async def _run_bandit_scan(self) -> List[QualityIssue]:
        """Run bandit security analysis."""
        vulnerabilities = []
        
        try:
            cmd = [
                "bandit", "-r", "pouw/",
                "-f", "json",
                "-o", "/tmp/bandit_report.json"
            ]
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.project_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()
            
            # Read bandit report
            report_file = Path("/tmp/bandit_report.json")
            if report_file.exists():
                with open(report_file, 'r') as f:
                    bandit_data = json.load(f)
                
                for result in bandit_data.get("results", []):
                    severity_map = {
                        "HIGH": SeverityLevel.HIGH,
                        "MEDIUM": SeverityLevel.MEDIUM,
                        "LOW": SeverityLevel.LOW
                    }
                    
                    vulnerabilities.append(QualityIssue(
                        check_type=QualityCheckType.SECURITY_SCAN,
                        severity=severity_map.get(result.get("issue_severity", "LOW"), SeverityLevel.LOW),
                        message=result.get("issue_text", ""),
                        file_path=result.get("filename", ""),
                        line_number=result.get("line_number", 0),
                        rule_id=result.get("test_id", ""),
                        category="security",
                        confidence={"HIGH": 1.0, "MEDIUM": 0.7, "LOW": 0.3}.get(result.get("issue_confidence", "LOW"), 0.5)
                    ))
                
                report_file.unlink()  # Clean up
                
        except Exception as e:
            print(f"Error running bandit scan: {e}")
        
        return vulnerabilities

    async def _run_safety_scan(self) -> List[QualityIssue]:
        """Run safety dependency vulnerability scan."""
        vulnerabilities = []
        
        try:
            cmd = ["safety", "check", "--json"]
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.project_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if stdout:
                try:
                    safety_data = json.loads(stdout.decode())
                    
                    for vuln in safety_data:
                        vulnerabilities.append(QualityIssue(
                            check_type=QualityCheckType.DEPENDENCY_SCAN,
                            severity=SeverityLevel.HIGH,  # All dependency vulnerabilities are high priority
                            message=f"Vulnerable dependency: {vuln.get('package', 'unknown')} {vuln.get('installed_version', '')} - {vuln.get('vulnerability', '')}",
                            file_path="requirements.txt",
                            rule_id=vuln.get('id', ''),
                            category="dependency_vulnerability",
                            metadata={
                                "package": vuln.get("package"),
                                "installed_version": vuln.get("installed_version"),
                                "safe_version": vuln.get("safe_version")
                            }
                        ))
                        
                except json.JSONDecodeError:
                    print("Error parsing safety JSON output")
            
        except Exception as e:
            print(f"Error running safety scan: {e}")
        
        return vulnerabilities

    async def _run_custom_security_checks(self) -> List[QualityIssue]:
        """Run custom security pattern checks."""
        vulnerabilities = []
        
        # Define security patterns to check for
        security_patterns = [
            {
                "pattern": r"password\s*=\s*['\"][^'\"]+['\"]",
                "message": "Hardcoded password detected",
                "severity": SeverityLevel.CRITICAL
            },
            {
                "pattern": r"api_key\s*=\s*['\"][^'\"]+['\"]",
                "message": "Hardcoded API key detected",
                "severity": SeverityLevel.CRITICAL
            },
            {
                "pattern": r"secret\s*=\s*['\"][^'\"]+['\"]",
                "message": "Hardcoded secret detected",
                "severity": SeverityLevel.CRITICAL
            },
            {
                "pattern": r"eval\s*\(",
                "message": "Use of eval() function detected",
                "severity": SeverityLevel.HIGH
            },
            {
                "pattern": r"exec\s*\(",
                "message": "Use of exec() function detected",
                "severity": SeverityLevel.HIGH
            },
            {
                "pattern": r"subprocess\.call\([^)]*shell\s*=\s*True",
                "message": "Subprocess call with shell=True",
                "severity": SeverityLevel.MEDIUM
            }
        ]
        
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                for pattern_info in security_patterns:
                    pattern = pattern_info["pattern"]
                    
                    for line_num, line in enumerate(lines, 1):
                        if re.search(pattern, line, re.IGNORECASE):
                            vulnerabilities.append(QualityIssue(
                                check_type=QualityCheckType.SECURITY_SCAN,
                                severity=pattern_info["severity"],
                                message=pattern_info["message"],
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=line_num,
                                rule_id="CUSTOM_SEC",
                                category="custom_security"
                            ))
                            
            except Exception as e:
                print(f"Error scanning {file_path} for security patterns: {e}")
        
        return vulnerabilities

    def get_security_summary(self) -> Dict[str, Any]:
        """Get security scan summary."""
        if not self.vulnerabilities:
            return {"total": 0, "by_severity": {}, "by_category": {}}
        
        by_severity = {}
        by_category = {}
        
        for vuln in self.vulnerabilities:
            # Count by severity
            severity = vuln.severity.value
            by_severity[severity] = by_severity.get(severity, 0) + 1
            
            # Count by category
            category = vuln.category
            by_category[category] = by_category.get(category, 0) + 1
        
        return {
            "total": len(self.vulnerabilities),
            "by_severity": by_severity,
            "by_category": by_category,
            "critical_count": by_severity.get("critical", 0),
            "high_count": by_severity.get("high", 0)
        }


# Example usage and predefined configurations
class PoUWQualityConfiguration:
    """Predefined quality configurations for PoUW system."""

    @staticmethod
    def strict_quality_gates() -> QualityConfiguration:
        """Strict quality configuration for production."""
        config = QualityConfiguration()
        config.min_coverage = 90.0
        config.max_complexity = 8
        config.quality_gates = [
            QualityGateRule(
                metric=QualityMetric.CODE_COVERAGE,
                operator=">=",
                threshold=90.0,
                severity=SeverityLevel.CRITICAL,
                description="Code coverage must be at least 90%"
            ),
            QualityGateRule(
                metric=QualityMetric.CYCLOMATIC_COMPLEXITY,
                operator="<=",
                threshold=8.0,
                severity=SeverityLevel.HIGH,
                description="Average cyclomatic complexity should not exceed 8"
            ),
            QualityGateRule(
                metric=QualityMetric.SECURITY_VULNERABILITIES,
                operator="==",
                threshold=0.0,
                severity=SeverityLevel.CRITICAL,
                description="No critical security vulnerabilities allowed"
            ),
            QualityGateRule(
                metric=QualityMetric.DOCUMENTATION_COVERAGE,
                operator=">=",
                threshold=85.0,
                severity=SeverityLevel.MEDIUM,
                description="Documentation coverage should be at least 85%"
            )
        ]
        return config

    @staticmethod
    def development_quality_gates() -> QualityConfiguration:
        """Relaxed quality configuration for development."""
        config = QualityConfiguration()
        config.min_coverage = 70.0
        config.max_complexity = 15
        config.quality_gates = [
            QualityGateRule(
                metric=QualityMetric.CODE_COVERAGE,
                operator=">=",
                threshold=70.0,
                severity=SeverityLevel.MEDIUM,
                description="Code coverage should be at least 70%"
            ),
            QualityGateRule(
                metric=QualityMetric.SECURITY_VULNERABILITIES,
                operator="==",
                threshold=0.0,
                severity=SeverityLevel.HIGH,
                description="No critical security vulnerabilities allowed"
            )
        ]
        return config


# Example usage
async def main():
    """Example usage of the quality assurance system."""
    # Initialize quality manager
    quality_manager = CodeQualityManager()
    security_scanner = SecurityScanner()
    
    # Configure for strict quality checks
    quality_manager.configuration = PoUWQualityConfiguration.strict_quality_gates()
    
    # Run quality analysis
    report = await quality_manager.analyze_quality("PoUW", "1.0.0")
    
    # Run security scan
    vulnerabilities = await security_scanner.scan_security_vulnerabilities()
    
    # Add security issues to quality report
    report.issues.extend(vulnerabilities)
    
    # Generate reports
    await quality_manager.generate_quality_report_html(report)
    
    print(f"Quality analysis completed:")
    print(f"  Grade: {report.grade}")
    print(f"  Quality Gate: {'PASSED' if report.quality_gate_passed else 'FAILED'}")
    print(f"  Total Issues: {len(report.issues)}")
    print(f"  Security Vulnerabilities: {len(vulnerabilities)}")


if __name__ == "__main__":
    asyncio.run(main())
