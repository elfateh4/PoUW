"""
GitHub Actions Workflow Management

This module provides comprehensive GitHub Actions workflow generation and management
for the PoUW CI/CD pipeline.
"""

import os
import json
import yaml
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """GitHub Actions trigger types"""

    PUSH = "push"
    PULL_REQUEST = "pull_request"
    SCHEDULE = "schedule"
    WORKFLOW_DISPATCH = "workflow_dispatch"
    RELEASE = "release"


class JobStrategy(Enum):
    """Job execution strategies"""

    MATRIX = "matrix"
    FAIL_FAST = "fail-fast"
    MAX_PARALLEL = "max-parallel"


@dataclass
class StepConfiguration:
    """Configuration for a workflow step"""

    name: str
    uses: Optional[str] = None
    run: Optional[str] = None
    with_: Optional[Dict[str, Any]] = None
    env: Optional[Dict[str, str]] = None
    if_condition: Optional[str] = None
    timeout_minutes: Optional[int] = None
    continue_on_error: bool = False
    id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML generation"""
        step_dict: Dict[str, Any] = {"name": self.name}

        if self.uses:
            step_dict["uses"] = self.uses
        if self.run:
            step_dict["run"] = self.run
        if self.with_:
            step_dict["with"] = self.with_
        if self.env:
            step_dict["env"] = self.env
        if self.if_condition:
            step_dict["if"] = self.if_condition
        if self.timeout_minutes:
            step_dict["timeout-minutes"] = self.timeout_minutes
        if self.continue_on_error:
            step_dict["continue-on-error"] = self.continue_on_error
        if self.id:
            step_dict["id"] = self.id

        return step_dict


@dataclass
class JobConfiguration:
    """Configuration for a workflow job"""

    name: str
    runs_on: Union[str, List[str]]
    steps: List[StepConfiguration]
    needs: Optional[List[str]] = None
    if_condition: Optional[str] = None
    strategy: Optional[Dict[str, Any]] = None
    container: Optional[str] = None
    services: Optional[Dict[str, Any]] = None
    timeout_minutes: Optional[int] = None
    environment: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML generation"""
        job_dict = {
            "name": self.name,
            "runs-on": self.runs_on,
            "steps": [step.to_dict() for step in self.steps],
        }

        if self.needs:
            job_dict["needs"] = self.needs
        if self.if_condition:
            job_dict["if"] = self.if_condition
        if self.strategy:
            job_dict["strategy"] = self.strategy
        if self.container:
            job_dict["container"] = self.container
        if self.services:
            job_dict["services"] = self.services
        if self.timeout_minutes:
            job_dict["timeout-minutes"] = self.timeout_minutes
        if self.environment:
            job_dict["environment"] = self.environment

        return job_dict


@dataclass
class TriggerConfiguration:
    """Configuration for workflow triggers"""

    trigger_type: TriggerType
    branches: Optional[List[str]] = None
    paths: Optional[List[str]] = None
    schedule: Optional[str] = None
    types: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML generation"""
        if self.trigger_type == TriggerType.SCHEDULE:
            return {"schedule": [{"cron": self.schedule}]}
        elif self.trigger_type == TriggerType.WORKFLOW_DISPATCH:
            return {"workflow_dispatch": {}}
        else:
            trigger_dict = {}
            if self.branches:
                trigger_dict["branches"] = self.branches
            if self.paths:
                trigger_dict["paths"] = self.paths
            if self.types:
                trigger_dict["types"] = self.types

            return {self.trigger_type.value: trigger_dict if trigger_dict else None}


@dataclass
class WorkflowConfiguration:
    """Complete workflow configuration"""

    name: str
    triggers: List[TriggerConfiguration]
    jobs: Dict[str, JobConfiguration]
    env: Optional[Dict[str, str]] = None
    permissions: Optional[Dict[str, str]] = None
    concurrency: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML generation"""
        workflow_dict = {
            "name": self.name,
            "on": {},
            "jobs": {job_id: job.to_dict() for job_id, job in self.jobs.items()},
        }

        # Merge all triggers
        for trigger in self.triggers:
            workflow_dict["on"].update(trigger.to_dict())

        if self.env:
            workflow_dict["env"] = self.env
        if self.permissions:
            workflow_dict["permissions"] = self.permissions
        if self.concurrency:
            workflow_dict["concurrency"] = self.concurrency

        return workflow_dict


class GitHubActionsManager:
    """GitHub Actions workflow manager"""

    def __init__(self, project_root: str = "."):
        """Initialize GitHub Actions manager"""
        self.project_root = Path(project_root)
        self.workflows_dir = self.project_root / ".github" / "workflows"
        self.workflows_dir.mkdir(parents=True, exist_ok=True)

    def create_ci_workflow(self) -> WorkflowConfiguration:
        """Create comprehensive CI workflow for PoUW"""
        # Setup steps
        setup_steps = [
            StepConfiguration(name="Checkout code", uses="actions/checkout@v4"),
            StepConfiguration(
                name="Set up Python",
                uses="actions/setup-python@v4",
                with_={"python-version": "${{ matrix.python-version }}"},
            ),
            StepConfiguration(
                name="Cache dependencies",
                uses="actions/cache@v3",
                with_={
                    "path": "~/.cache/pip",
                    "key": "${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}",
                    "restore-keys": "${{ runner.os }}-pip-",
                },
            ),
            StepConfiguration(
                name="Install dependencies",
                run="pip install -r requirements.txt && pip install pytest pytest-cov black flake8 mypy safety bandit",
            ),
        ]

        # Test steps
        test_steps = [
            StepConfiguration(
                name="Run linting",
                run="flake8 pouw --count --select=E9,F63,F7,F82 --show-source --statistics",
            ),
            StepConfiguration(name="Run type checking", run="mypy pouw --ignore-missing-imports"),
            StepConfiguration(name="Run security scan", run="safety check && bandit -r pouw"),
            StepConfiguration(
                name="Run tests with coverage",
                run="pytest tests/ --cov=pouw --cov-report=xml --cov-report=html --junit-xml=test-results.xml",
            ),
            StepConfiguration(
                name="Upload coverage to Codecov",
                uses="codecov/codecov-action@v3",
                with_={"file": "./coverage.xml", "flags": "unittests", "name": "codecov-umbrella"},
            ),
        ]

        # Build steps
        build_steps = [
            StepConfiguration(name="Build package", run="python -m build"),
            StepConfiguration(name="Test package installation", run="pip install dist/*.whl"),
        ]

        # Complete CI job
        ci_job = JobConfiguration(
            name="CI Pipeline",
            runs_on="ubuntu-latest",
            strategy={
                "matrix": {"python-version": ["3.8", "3.9", "3.10", "3.11"]},
                "fail-fast": False,
            },
            steps=setup_steps + test_steps + build_steps,
        )

        # Triggers
        triggers = [
            TriggerConfiguration(trigger_type=TriggerType.PUSH, branches=["main", "develop"]),
            TriggerConfiguration(trigger_type=TriggerType.PULL_REQUEST, branches=["main"]),
            TriggerConfiguration(trigger_type=TriggerType.WORKFLOW_DISPATCH),
        ]

        return WorkflowConfiguration(
            name="PoUW CI Pipeline",
            triggers=triggers,
            jobs={"ci": ci_job},
            env={"PYTHONPATH": "${{ github.workspace }}", "PYTHONUNBUFFERED": "1"},
            permissions={"contents": "read", "pull-requests": "write", "checks": "write"},
        )

    def create_docker_workflow(self) -> WorkflowConfiguration:
        """Create Docker build and push workflow"""
        docker_steps = [
            StepConfiguration(name="Checkout code", uses="actions/checkout@v4"),
            StepConfiguration(name="Set up Docker Buildx", uses="docker/setup-buildx-action@v3"),
            StepConfiguration(
                name="Log in to Container Registry",
                uses="docker/login-action@v3",
                with_={
                    "registry": "ghcr.io",
                    "username": "${{ github.actor }}",
                    "password": "${{ secrets.GITHUB_TOKEN }}",
                },
            ),
            StepConfiguration(
                name="Extract metadata",
                id="meta",
                uses="docker/metadata-action@v5",
                with_={
                    "images": "ghcr.io/${{ github.repository }}/pouw",
                    "tags": """
                        type=ref,event=branch
                        type=ref,event=pr
                        type=sha,prefix={{branch}}-
                        type=raw,value=latest,enable={{is_default_branch}}
                    """,
                },
            ),
            StepConfiguration(
                name="Build and push Docker image",
                uses="docker/build-push-action@v5",
                with_={
                    "context": ".",
                    "platforms": "linux/amd64,linux/arm64",
                    "push": True,
                    "tags": "${{ steps.meta.outputs.tags }}",
                    "labels": "${{ steps.meta.outputs.labels }}",
                    "cache-from": "type=gha",
                    "cache-to": "type=gha,mode=max",
                },
            ),
        ]

        docker_job = JobConfiguration(
            name="Docker Build and Push",
            runs_on="ubuntu-latest",
            steps=docker_steps,
            needs=["ci"],
            if_condition="github.event_name != 'pull_request'",
        )

        triggers = [
            TriggerConfiguration(trigger_type=TriggerType.PUSH, branches=["main", "develop"]),
            TriggerConfiguration(trigger_type=TriggerType.PULL_REQUEST, branches=["main"]),
        ]

        return WorkflowConfiguration(
            name="Docker Build",
            triggers=triggers,
            jobs={"docker": docker_job},
            permissions={"contents": "read", "packages": "write"},
        )

    def create_deployment_workflow(self) -> WorkflowConfiguration:
        """Create deployment workflow"""
        deploy_steps = [
            StepConfiguration(name="Checkout code", uses="actions/checkout@v4"),
            StepConfiguration(
                name="Configure AWS credentials",
                uses="aws-actions/configure-aws-credentials@v4",
                with_={
                    "aws-access-key-id": "${{ secrets.AWS_ACCESS_KEY_ID }}",
                    "aws-secret-access-key": "${{ secrets.AWS_SECRET_ACCESS_KEY }}",
                    "aws-region": "${{ vars.AWS_REGION }}",
                },
            ),
            StepConfiguration(
                name="Set up kubectl", uses="azure/setup-kubectl@v3", with_={"version": "latest"}
            ),
            StepConfiguration(
                name="Configure kubectl",
                run="aws eks update-kubeconfig --name ${{ vars.CLUSTER_NAME }} --region ${{ vars.AWS_REGION }}",
            ),
            StepConfiguration(
                name="Deploy to staging",
                run="""
                    kubectl set image deployment/pouw-blockchain pouw-blockchain=ghcr.io/${{ github.repository }}/pouw:${{ github.sha }} -n pouw-staging
                    kubectl set image deployment/pouw-ml-trainer pouw-ml-trainer=ghcr.io/${{ github.repository }}/pouw:${{ github.sha }} -n pouw-staging
                    kubectl rollout status deployment/pouw-blockchain -n pouw-staging
                    kubectl rollout status deployment/pouw-ml-trainer -n pouw-staging
                """,
                if_condition="github.ref == 'refs/heads/develop'",
            ),
            StepConfiguration(
                name="Deploy to production",
                run="""
                    kubectl set image deployment/pouw-blockchain pouw-blockchain=ghcr.io/${{ github.repository }}/pouw:${{ github.sha }} -n pouw-production
                    kubectl set image deployment/pouw-ml-trainer pouw-ml-trainer=ghcr.io/${{ github.repository }}/pouw:${{ github.sha }} -n pouw-production
                    kubectl rollout status deployment/pouw-blockchain -n pouw-production
                    kubectl rollout status deployment/pouw-ml-trainer -n pouw-production
                """,
                if_condition="github.ref == 'refs/heads/main'",
            ),
            StepConfiguration(
                name="Run smoke tests",
                run="python scripts/smoke_tests.py --environment ${{ github.ref == 'refs/heads/main' && 'production' || 'staging' }}",
            ),
        ]

        deploy_job = JobConfiguration(
            name="Deploy to Kubernetes",
            runs_on="ubuntu-latest",
            environment="${{ github.ref == 'refs/heads/main' && 'production' || 'staging' }}",
            steps=deploy_steps,
            needs=["docker"],
        )

        triggers = [
            TriggerConfiguration(trigger_type=TriggerType.PUSH, branches=["main", "develop"])
        ]

        return WorkflowConfiguration(
            name="Deploy PoUW",
            triggers=triggers,
            jobs={"deploy": deploy_job},
            permissions={"contents": "read", "id-token": "write"},
        )

    def create_release_workflow(self) -> WorkflowConfiguration:
        """Create release workflow"""
        release_steps = [
            StepConfiguration(
                name="Checkout code", uses="actions/checkout@v4", with_={"fetch-depth": 0}
            ),
            StepConfiguration(
                name="Set up Python",
                uses="actions/setup-python@v4",
                with_={"python-version": "3.11"},
            ),
            StepConfiguration(
                name="Install dependencies", run="pip install build twine semantic-release"
            ),
            StepConfiguration(name="Generate changelog", run="semantic-release changelog"),
            StepConfiguration(name="Build package", run="python -m build"),
            StepConfiguration(
                name="Create GitHub release",
                uses="actions/create-release@v1",
                env={"GITHUB_TOKEN": "${{ secrets.GITHUB_TOKEN }}"},
                with_={
                    "tag_name": "${{ github.ref }}",
                    "release_name": "Release ${{ github.ref }}",
                    "body_path": "CHANGELOG.md",
                    "draft": False,
                    "prerelease": False,
                },
            ),
            StepConfiguration(
                name="Publish to PyPI",
                env={
                    "TWINE_USERNAME": "__token__",
                    "TWINE_PASSWORD": "${{ secrets.PYPI_API_TOKEN }}",
                },
                run="twine upload dist/*",
            ),
        ]

        release_job = JobConfiguration(
            name="Create Release", runs_on="ubuntu-latest", steps=release_steps
        )

        triggers = [TriggerConfiguration(trigger_type=TriggerType.RELEASE, types=["created"])]

        return WorkflowConfiguration(
            name="Release PoUW",
            triggers=triggers,
            jobs={"release": release_job},
            permissions={"contents": "write", "packages": "write"},
        )

    def generate_workflow_file(self, workflow: WorkflowConfiguration, filename: str) -> Path:
        """Generate YAML workflow file"""
        workflow_file = self.workflows_dir / f"{filename}.yml"

        workflow_dict = workflow.to_dict()

        with open(workflow_file, "w") as f:
            yaml.dump(workflow_dict, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Generated workflow file: {workflow_file}")
        return workflow_file

    def generate_all_workflows(self) -> Dict[str, Path]:
        """Generate all PoUW workflows"""
        workflows = {}

        # CI workflow
        ci_workflow = self.create_ci_workflow()
        workflows["ci"] = self.generate_workflow_file(ci_workflow, "ci")

        # Docker workflow
        docker_workflow = self.create_docker_workflow()
        workflows["docker"] = self.generate_workflow_file(docker_workflow, "docker")

        # Deployment workflow
        deploy_workflow = self.create_deployment_workflow()
        workflows["deploy"] = self.generate_workflow_file(deploy_workflow, "deploy")

        # Release workflow
        release_workflow = self.create_release_workflow()
        workflows["release"] = self.generate_workflow_file(release_workflow, "release")

        logger.info(f"Generated {len(workflows)} workflow files")
        return workflows

    def validate_workflows(self) -> Dict[str, bool]:
        """Validate generated workflow files"""
        validation_results = {}

        for workflow_file in self.workflows_dir.glob("*.yml"):
            try:
                with open(workflow_file, "r") as f:
                    yaml.safe_load(f)
                validation_results[workflow_file.name] = True
                logger.info(f"✓ {workflow_file.name} is valid")
            except yaml.YAMLError as e:
                validation_results[workflow_file.name] = False
                logger.error(f"✗ {workflow_file.name} is invalid: {e}")

        return validation_results

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get status of all workflows"""
        return {
            "workflows_directory": str(self.workflows_dir),
            "workflow_files": list(self.workflows_dir.glob("*.yml")),
            "total_workflows": len(list(self.workflows_dir.glob("*.yml"))),
            "validation_results": self.validate_workflows(),
        }
