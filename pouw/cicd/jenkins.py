"""
Jenkins Pipeline Management

This module provides comprehensive Jenkins pipeline generation and management
for the PoUW CI/CD system.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class PipelineType(Enum):
    """Jenkins pipeline types"""
    DECLARATIVE = "declarative"
    SCRIPTED = "scripted"


class AgentType(Enum):
    """Jenkins agent types"""
    ANY = "any"
    NONE = "none"
    LABEL = "label"
    NODE = "node"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"


class TriggerType(Enum):
    """Jenkins trigger types"""
    CRON = "cron"
    POLLSCM = "pollSCM"
    UPSTREAM = "upstream"
    GITHUB_PUSH = "githubPush"


@dataclass
class PipelineStage:
    """Jenkins pipeline stage configuration"""
    name: str
    steps: List[str]
    when_condition: Optional[str] = None
    parallel_stages: Optional[Dict[str, 'PipelineStage']] = None
    agent: Optional[str] = None
    environment: Optional[Dict[str, str]] = None
    post_actions: Optional[Dict[str, List[str]]] = None
    
    def to_groovy(self, indent: int = 0) -> str:
        """Convert stage to Groovy pipeline syntax"""
        ind = "    " * indent
        groovy = f'{ind}stage("{self.name}") {{\n'
        
        # When condition
        if self.when_condition:
            groovy += f'{ind}    when {{\n'
            groovy += f'{ind}        {self.when_condition}\n'
            groovy += f'{ind}    }}\n'
        
        # Agent
        if self.agent:
            groovy += f'{ind}    agent {self.agent}\n'
        
        # Environment
        if self.environment:
            groovy += f'{ind}    environment {{\n'
            for key, value in self.environment.items():
                groovy += f'{ind}        {key} = "{value}"\n'
            groovy += f'{ind}    }}\n'
        
        # Parallel stages
        if self.parallel_stages:
            groovy += f'{ind}    parallel {{\n'
            for stage_name, stage in self.parallel_stages.items():
                groovy += stage.to_groovy(indent + 2)
            groovy += f'{ind}    }}\n'
        else:
            # Steps
            groovy += f'{ind}    steps {{\n'
            for step in self.steps:
                groovy += f'{ind}        {step}\n'
            groovy += f'{ind}    }}\n'
        
        # Post actions
        if self.post_actions:
            groovy += f'{ind}    post {{\n'
            for condition, actions in self.post_actions.items():
                groovy += f'{ind}        {condition} {{\n'
                for action in actions:
                    groovy += f'{ind}            {action}\n'
                groovy += f'{ind}        }}\n'
            groovy += f'{ind}    }}\n'
        
        groovy += f'{ind}}}\n'
        return groovy


@dataclass
class PipelineConfiguration:
    """Complete Jenkins pipeline configuration"""
    name: str
    agent: str
    stages: List[PipelineStage]
    pipeline_type: PipelineType = PipelineType.DECLARATIVE
    triggers: Optional[List[str]] = None
    parameters: Optional[Dict[str, Any]] = None
    environment: Optional[Dict[str, str]] = None
    tools: Optional[Dict[str, str]] = None
    options: Optional[List[str]] = None
    post_actions: Optional[Dict[str, List[str]]] = None
    
    def to_groovy(self) -> str:
        """Convert pipeline to complete Groovy Jenkinsfile"""
        if self.pipeline_type == PipelineType.DECLARATIVE:
            return self._to_declarative_groovy()
        else:
            return self._to_scripted_groovy()
    
    def _to_declarative_groovy(self) -> str:
        """Generate declarative pipeline syntax"""
        groovy = f"""// Jenkins Pipeline for {self.name}
// Generated on {datetime.now().isoformat()}

pipeline {{
    agent {self.agent}
"""
        
        # Triggers
        if self.triggers:
            groovy += "    triggers {\n"
            for trigger in self.triggers:
                groovy += f"        {trigger}\n"
            groovy += "    }\n"
        
        # Parameters
        if self.parameters:
            groovy += "    parameters {\n"
            for param_name, param_config in self.parameters.items():
                groovy += f"        {param_config}\n"
            groovy += "    }\n"
        
        # Environment
        if self.environment:
            groovy += "    environment {\n"
            for key, value in self.environment.items():
                groovy += f'        {key} = "{value}"\n'
            groovy += "    }\n"
        
        # Tools
        if self.tools:
            groovy += "    tools {\n"
            for tool, version in self.tools.items():
                groovy += f'        {tool} "{version}"\n'
            groovy += "    }\n"
        
        # Options
        if self.options:
            groovy += "    options {\n"
            for option in self.options:
                groovy += f"        {option}\n"
            groovy += "    }\n"
        
        # Stages
        groovy += "    stages {\n"
        for stage in self.stages:
            groovy += stage.to_groovy(indent=2)
        groovy += "    }\n"
        
        # Post actions
        if self.post_actions:
            groovy += "    post {\n"
            for condition, actions in self.post_actions.items():
                groovy += f"        {condition} {{\n"
                for action in actions:
                    groovy += f"            {action}\n"
                groovy += "        }\n"
            groovy += "    }\n"
        
        groovy += "}\n"
        return groovy
    
    def _to_scripted_groovy(self) -> str:
        """Generate scripted pipeline syntax"""
        groovy = f"""// Jenkins Scripted Pipeline for {self.name}
// Generated on {datetime.now().isoformat()}

node('{self.agent}') {{
"""
        
        # Environment setup
        if self.environment:
            for key, value in self.environment.items():
                groovy += f'    env.{key} = "{value}"\n'
        
        groovy += "    try {\n"
        
        # Stages
        for stage in self.stages:
            groovy += f'        stage("{stage.name}") {{\n'
            for step in stage.steps:
                groovy += f"            {step}\n"
            groovy += "        }\n"
        
        groovy += "    } catch (Exception e) {\n"
        groovy += "        currentBuild.result = 'FAILURE'\n"
        groovy += "        throw e\n"
        groovy += "    } finally {\n"
        
        # Post actions
        if self.post_actions and 'always' in self.post_actions:
            for action in self.post_actions['always']:
                groovy += f"        {action}\n"
        
        groovy += "    }\n"
        groovy += "}\n"
        
        return groovy


class JenkinsfileGenerator:
    """Generator for various types of Jenkinsfiles"""
    
    def __init__(self):
        """Initialize Jenkinsfile generator"""
        pass
    
    def create_ci_pipeline(self) -> PipelineConfiguration:
        """Create CI pipeline for PoUW"""
        stages = [
            PipelineStage(
                name="Checkout",
                steps=[
                    "checkout scm",
                    "sh 'git clean -fdx'"
                ]
            ),
            PipelineStage(
                name="Setup Environment",
                steps=[
                    "sh 'python3 -m venv venv'",
                    "sh '. venv/bin/activate && pip install --upgrade pip'",
                    "sh '. venv/bin/activate && pip install -r requirements.txt'",
                    "sh '. venv/bin/activate && pip install pytest pytest-cov black flake8 mypy safety bandit'"
                ]
            ),
            PipelineStage(
                name="Code Quality",
                parallel_stages={
                    "Linting": PipelineStage(
                        name="Linting",
                        steps=[
                            "sh '. venv/bin/activate && flake8 pouw --count --select=E9,F63,F7,F82 --show-source --statistics'",
                            "sh '. venv/bin/activate && black --check pouw'"
                        ]
                    ),
                    "Type Checking": PipelineStage(
                        name="Type Checking",
                        steps=[
                            "sh '. venv/bin/activate && mypy pouw --ignore-missing-imports'"
                        ]
                    ),
                    "Security Scan": PipelineStage(
                        name="Security Scan",
                        steps=[
                            "sh '. venv/bin/activate && safety check'",
                            "sh '. venv/bin/activate && bandit -r pouw'"
                        ]
                    )
                }
            ),
            PipelineStage(
                name="Unit Tests",
                steps=[
                    "sh '. venv/bin/activate && pytest tests/ --cov=pouw --cov-report=xml --cov-report=html --junit-xml=test-results.xml'"
                ],
                post_actions={
                    "always": [
                        "publishTestResults testResultsPattern: 'test-results.xml'",
                        "publishCoverage adapters: [coberturaAdapter('coverage.xml')], sourceFileResolver: sourceFiles('STORE_LAST_BUILD')"
                    ]
                }
            ),
            PipelineStage(
                name="Integration Tests",
                steps=[
                    "sh '. venv/bin/activate && python -m pytest tests/test_integration.py -v'"
                ],
                when_condition="branch 'main' || branch 'develop'"
            ),
            PipelineStage(
                name="Build Package",
                steps=[
                    "sh '. venv/bin/activate && python -m build'",
                    "archiveArtifacts artifacts: 'dist/*', allowEmptyArchive: false"
                ]
            )
        ]
        
        return PipelineConfiguration(
            name="PoUW CI Pipeline",
            agent="any",
            stages=stages,
            triggers=[
                "pollSCM('H/5 * * * *')",
                "cron('@daily')"
            ],
            environment={
                "PYTHONPATH": "${WORKSPACE}",
                "PYTHONUNBUFFERED": "1"
            },
            tools={
                "python": "python-3.11"
            },
            options=[
                "buildDiscarder(logRotator(numToKeepStr: '10'))",
                "timeout(time: 30, unit: 'MINUTES')",
                "timestamps()"
            ],
            post_actions={
                "success": [
                    "echo 'Pipeline completed successfully!'",
                    "slackSend(channel: '#ci-cd', color: 'good', message: 'PoUW CI Pipeline succeeded for ${env.BRANCH_NAME}')"
                ],
                "failure": [
                    "echo 'Pipeline failed!'",
                    "slackSend(channel: '#ci-cd', color: 'danger', message: 'PoUW CI Pipeline failed for ${env.BRANCH_NAME}')"
                ],
                "always": [
                    "cleanWs()"
                ]
            }
        )
    
    def create_docker_pipeline(self) -> PipelineConfiguration:
        """Create Docker build pipeline"""
        stages = [
            PipelineStage(
                name="Checkout",
                steps=[
                    "checkout scm"
                ]
            ),
            PipelineStage(
                name="Build Docker Images",
                parallel_stages={
                    "Blockchain Node": PipelineStage(
                        name="Blockchain Node",
                        steps=[
                            "script {",
                            "    def image = docker.build('pouw/blockchain-node:${BUILD_NUMBER}', '-f docker/Dockerfile.blockchain .')",
                            "    docker.withRegistry('https://registry.hub.docker.com', 'dockerhub-credentials') {",
                            "        image.push()",
                            "        image.push('latest')",
                            "    }",
                            "}"
                        ]
                    ),
                    "ML Trainer": PipelineStage(
                        name="ML Trainer",
                        steps=[
                            "script {",
                            "    def image = docker.build('pouw/ml-trainer:${BUILD_NUMBER}', '-f docker/Dockerfile.ml .')",
                            "    docker.withRegistry('https://registry.hub.docker.com', 'dockerhub-credentials') {",
                            "        image.push()",
                            "        image.push('latest')",
                            "    }",
                            "}"
                        ]
                    ),
                    "VPN Mesh": PipelineStage(
                        name="VPN Mesh",
                        steps=[
                            "script {",
                            "    def image = docker.build('pouw/vpn-mesh:${BUILD_NUMBER}', '-f docker/Dockerfile.vpn .')",
                            "    docker.withRegistry('https://registry.hub.docker.com', 'dockerhub-credentials') {",
                            "        image.push()",
                            "        image.push('latest')",
                            "    }",
                            "}"
                        ]
                    )
                }
            ),
            PipelineStage(
                name="Security Scan",
                steps=[
                    "sh 'docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy image pouw/blockchain-node:${BUILD_NUMBER}'",
                    "sh 'docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy image pouw/ml-trainer:${BUILD_NUMBER}'"
                ]
            ),
            PipelineStage(
                name="Image Cleanup",
                steps=[
                    "sh 'docker system prune -f'"
                ],
                post_actions={
                    "always": [
                        "sh 'docker system prune -f'"
                    ]
                }
            )
        ]
        
        return PipelineConfiguration(
            name="PoUW Docker Build",
            agent="docker",
            stages=stages,
            triggers=[
                "upstream(upstreamProjects: 'pouw-ci', threshold: hudson.model.Result.SUCCESS)"
            ],
            environment={
                "DOCKER_BUILDKIT": "1"
            },
            options=[
                "buildDiscarder(logRotator(numToKeepStr: '5'))",
                "timeout(time: 45, unit: 'MINUTES')"
            ]
        )
    
    def create_deployment_pipeline(self) -> PipelineConfiguration:
        """Create deployment pipeline"""
        stages = [
            PipelineStage(
                name="Checkout",
                steps=[
                    "checkout scm"
                ]
            ),
            PipelineStage(
                name="Deploy to Staging",
                steps=[
                    "script {",
                    "    kubernetesDeploy(",
                    "        configs: 'k8s/staging/*.yaml',",
                    "        kubeconfigId: 'kubeconfig-staging'",
                    "    )",
                    "}"
                ],
                when_condition="branch 'develop'"
            ),
            PipelineStage(
                name="Staging Tests",
                steps=[
                    "sh 'python scripts/smoke_tests.py --environment staging'",
                    "sh 'python scripts/integration_tests.py --environment staging'"
                ],
                when_condition="branch 'develop'"
            ),
            PipelineStage(
                name="Deploy to Production",
                steps=[
                    "input message: 'Deploy to production?', ok: 'Deploy'",
                    "script {",
                    "    kubernetesDeploy(",
                    "        configs: 'k8s/production/*.yaml',",
                    "        kubeconfigId: 'kubeconfig-production'",
                    "    )",
                    "}"
                ],
                when_condition="branch 'main'"
            ),
            PipelineStage(
                name="Production Verification",
                steps=[
                    "sh 'python scripts/smoke_tests.py --environment production'",
                    "sh 'python scripts/health_check.py --environment production'"
                ],
                when_condition="branch 'main'"
            )
        ]
        
        return PipelineConfiguration(
            name="PoUW Deployment",
            agent="kubernetes",
            stages=stages,
            triggers=[
                "upstream(upstreamProjects: 'pouw-docker', threshold: hudson.model.Result.SUCCESS)"
            ],
            parameters={
                "DEPLOY_VERSION": "string(name: 'DEPLOY_VERSION', defaultValue: 'latest', description: 'Version to deploy')",
                "SKIP_TESTS": "booleanParam(name: 'SKIP_TESTS', defaultValue: false, description: 'Skip testing stages')"
            },
            environment={
                "KUBECONFIG": "/var/jenkins_home/.kube/config"
            }
        )


class JenkinsPipelineManager:
    """Jenkins pipeline management system"""
    
    def __init__(self, project_root: str = "."):
        """Initialize Jenkins pipeline manager"""
        self.project_root = Path(project_root)
        self.jenkins_dir = self.project_root / "jenkins"
        self.jenkins_dir.mkdir(exist_ok=True)
        self.generator = JenkinsfileGenerator()
    
    def generate_jenkinsfile(self, pipeline: PipelineConfiguration, filename: str = "Jenkinsfile") -> Path:
        """Generate Jenkinsfile from pipeline configuration"""
        jenkinsfile_path = self.project_root / filename
        
        jenkinsfile_content = pipeline.to_groovy()
        
        with open(jenkinsfile_path, 'w') as f:
            f.write(jenkinsfile_content)
        
        logger.info(f"Generated Jenkinsfile: {jenkinsfile_path}")
        return jenkinsfile_path
    
    def generate_all_pipelines(self) -> Dict[str, Path]:
        """Generate all Jenkins pipelines"""
        pipelines = {}
        
        # CI Pipeline
        ci_pipeline = self.generator.create_ci_pipeline()
        pipelines['ci'] = self.generate_jenkinsfile(ci_pipeline, "Jenkinsfile.ci")
        
        # Docker Pipeline
        docker_pipeline = self.generator.create_docker_pipeline()
        pipelines['docker'] = self.generate_jenkinsfile(docker_pipeline, "Jenkinsfile.docker")
        
        # Deployment Pipeline
        deploy_pipeline = self.generator.create_deployment_pipeline()
        pipelines['deploy'] = self.generate_jenkinsfile(deploy_pipeline, "Jenkinsfile.deploy")
        
        # Main Jenkinsfile (CI pipeline)
        pipelines['main'] = self.generate_jenkinsfile(ci_pipeline, "Jenkinsfile")
        
        logger.info(f"Generated {len(pipelines)} Jenkins pipeline files")
        return pipelines
    
    def create_job_dsl_scripts(self) -> Dict[str, Path]:
        """Create Job DSL scripts for automatic job creation"""
        job_dsl_scripts = {}
        
        # Main CI job
        ci_job_dsl = """
job('pouw-ci') {
    description('PoUW Continuous Integration Pipeline')
    scm {
        git {
            remote {
                url('https://github.com/your-org/pouw.git')
                credentials('github-credentials')
            }
            branches('*/main', '*/develop')
        }
    }
    triggers {
        githubPush()
        cron('@daily')
    }
    definition {
        cpsScm {
            scm {
                git {
                    remote {
                        url('https://github.com/your-org/pouw.git')
                        credentials('github-credentials')
                    }
                    branches('*/main', '*/develop')
                }
            }
            scriptPath('Jenkinsfile.ci')
        }
    }
}
"""
        
        # Docker build job
        docker_job_dsl = """
job('pouw-docker') {
    description('PoUW Docker Build Pipeline')
    triggers {
        upstream('pouw-ci', 'SUCCESS')
    }
    definition {
        cpsScm {
            scm {
                git {
                    remote {
                        url('https://github.com/your-org/pouw.git')
                        credentials('github-credentials')
                    }
                    branches('*/main', '*/develop')
                }
            }
            scriptPath('Jenkinsfile.docker')
        }
    }
}
"""
        
        # Deployment job
        deploy_job_dsl = """
job('pouw-deploy') {
    description('PoUW Deployment Pipeline')
    triggers {
        upstream('pouw-docker', 'SUCCESS')
    }
    definition {
        cpsScm {
            scm {
                git {
                    remote {
                        url('https://github.com/your-org/pouw.git')
                        credentials('github-credentials')
                    }
                    branches('*/main', '*/develop')
                }
            }
            scriptPath('Jenkinsfile.deploy')
        }
    }
}
"""
        
        # Write Job DSL scripts
        ci_dsl_file = self.jenkins_dir / "ci-job.groovy"
        with open(ci_dsl_file, 'w') as f:
            f.write(ci_job_dsl)
        job_dsl_scripts['ci'] = ci_dsl_file
        
        docker_dsl_file = self.jenkins_dir / "docker-job.groovy"
        with open(docker_dsl_file, 'w') as f:
            f.write(docker_job_dsl)
        job_dsl_scripts['docker'] = docker_dsl_file
        
        deploy_dsl_file = self.jenkins_dir / "deploy-job.groovy"
        with open(deploy_dsl_file, 'w') as f:
            f.write(deploy_job_dsl)
        job_dsl_scripts['deploy'] = deploy_dsl_file
        
        logger.info(f"Generated {len(job_dsl_scripts)} Job DSL scripts")
        return job_dsl_scripts
    
    def validate_jenkinsfile(self, jenkinsfile_path: Path) -> bool:
        """Validate Jenkinsfile syntax (basic validation)"""
        try:
            with open(jenkinsfile_path, 'r') as f:
                content = f.read()
            
            # Basic syntax checks
            if not content.strip():
                return False
            
            # Check for required elements
            required_elements = ['pipeline', 'agent', 'stages']
            for element in required_elements:
                if element not in content:
                    logger.error(f"Missing required element: {element}")
                    return False
            
            # Check for balanced braces
            open_braces = content.count('{')
            close_braces = content.count('}')
            if open_braces != close_braces:
                logger.error("Unbalanced braces in Jenkinsfile")
                return False
            
            logger.info(f"✓ {jenkinsfile_path.name} is valid")
            return True
            
        except Exception as e:
            logger.error(f"Error validating {jenkinsfile_path}: {e}")
            return False
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get status of all Jenkins pipelines"""
        jenkinsfiles = list(self.project_root.glob("Jenkinsfile*"))
        
        return {
            "project_root": str(self.project_root),
            "jenkins_directory": str(self.jenkins_dir),
            "jenkinsfiles": [str(jf) for jf in jenkinsfiles],
            "total_jenkinsfiles": len(jenkinsfiles),
            "validation_results": {
                jf.name: self.validate_jenkinsfile(jf) for jf in jenkinsfiles
            }
        }
