pipeline {
    agent {
        kubernetes {
            yaml """
                apiVersion: v1
                kind: Pod
                spec:
                  containers:
                  - name: python
                    image: python:3.12-slim
                    command:
                    - cat
                    tty: true
                    volumeMounts:
                    - name: docker-sock
                      mountPath: /var/run/docker.sock
                  - name: docker
                    image: docker:dind
                    securityContext:
                      privileged: true
                    volumeMounts:
                    - name: docker-sock
                      mountPath: /var/run/docker.sock
                  - name: kubectl
                    image: bitnami/kubectl:latest
                    command:
                    - cat
                    tty: true
                  volumes:
                  - name: docker-sock
                    hostPath:
                      path: /var/run/docker.sock
            """
        }
    }

    environment {
        DOCKER_REGISTRY = 'your-registry.com'
        IMAGE_NAME = 'pouw'
        PYTHON_VERSION = '3.12'
        KUBECONFIG = credentials('kubeconfig')
        DOCKER_CREDENTIALS = credentials('docker-registry-credentials')
        SLACK_WEBHOOK = credentials('slack-webhook-url')
    }

    options {
        buildDiscarder(logRotator(numToKeepStr: '10'))
        timeout(time: 60, unit: 'MINUTES')
        timestamps()
        ansiColor('xterm')
    }

    triggers {
        pollSCM('H/5 * * * *')  // Poll every 5 minutes
        cron('H 2 * * *')       // Nightly build at 2 AM
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
                script {
                    env.GIT_COMMIT_SHORT = sh(
                        script: 'git rev-parse --short HEAD',
                        returnStdout: true
                    ).trim()
                    env.BUILD_TAG = "${env.BRANCH_NAME}-${env.BUILD_NUMBER}-${env.GIT_COMMIT_SHORT}"
                }
            }
        }

        stage('Setup Environment') {
            parallel {
                stage('Python Environment') {
                    steps {
                        container('python') {
                            sh '''
                                python -m pip install --upgrade pip
                                pip install -r requirements.txt
                                pip install pytest pytest-cov pytest-asyncio black pylint mypy bandit safety
                            '''
                        }
                    }
                }
                stage('Docker Environment') {
                    steps {
                        container('docker') {
                            sh 'docker --version'
                        }
                    }
                }
            }
        }

        stage('Code Quality') {
            parallel {
                stage('Format Check') {
                    steps {
                        container('python') {
                            sh 'black --check --line-length 100 .'
                        }
                    }
                }
                stage('Linting') {
                    steps {
                        container('python') {
                            sh '''
                                pylint pouw/ --exit-zero --output-format=json > pylint-report.json
                                cat pylint-report.json
                            '''
                        }
                    }
                    post {
                        always {
                            archiveArtifacts artifacts: 'pylint-report.json', fingerprint: true
                        }
                    }
                }
                stage('Type Checking') {
                    steps {
                        container('python') {
                            sh 'mypy pouw/ --ignore-missing-imports'
                        }
                    }
                }
                stage('Security Scan') {
                    steps {
                        container('python') {
                            sh '''
                                bandit -r pouw/ -f json -o bandit-report.json
                                safety check --json --output safety-report.json
                            '''
                        }
                    }
                    post {
                        always {
                            archiveArtifacts artifacts: 'bandit-report.json,safety-report.json', fingerprint: true
                        }
                    }
                }
            }
        }

        stage('Testing') {
            parallel {
                stage('Unit Tests') {
                    steps {
                        container('python') {
                            sh '''
                                pytest tests/ -k "not integration and not security and not performance" \
                                    --cov=pouw --cov-report=xml --cov-report=html \
                                    --junit-xml=test-results-unit.xml -v
                            '''
                        }
                    }
                    post {
                        always {
                            publishTestResults testResultsPattern: 'test-results-unit.xml'
                            publishCoverage adapters: [coberturaAdapter('coverage.xml')], sourceFileResolver: sourceFiles('STORE_LAST_BUILD')
                            archiveArtifacts artifacts: 'htmlcov/**', fingerprint: true
                        }
                    }
                }
                stage('Integration Tests') {
                    steps {
                        container('python') {
                            sh '''
                                pytest tests/ -k "integration" \
                                    --junit-xml=test-results-integration.xml -v
                            '''
                        }
                    }
                    post {
                        always {
                            publishTestResults testResultsPattern: 'test-results-integration.xml'
                        }
                    }
                }
                stage('Security Tests') {
                    steps {
                        container('python') {
                            sh '''
                                pytest tests/ -k "security" \
                                    --junit-xml=test-results-security.xml -v
                            '''
                        }
                    }
                    post {
                        always {
                            publishTestResults testResultsPattern: 'test-results-security.xml'
                        }
                    }
                }
                stage('Performance Tests') {
                    steps {
                        container('python') {
                            sh '''
                                pytest tests/ -k "performance" \
                                    --junit-xml=test-results-performance.xml -v
                            '''
                        }
                    }
                    post {
                        always {
                            publishTestResults testResultsPattern: 'test-results-performance.xml'
                        }
                    }
                }
            }
        }

        stage('Build Docker Images') {
            when {
                anyOf {
                    branch 'main'
                    branch 'develop'
                    branch 'release/*'
                }
            }
            parallel {
                stage('Development Image') {
                    steps {
                        container('docker') {
                            script {
                                def image = docker.build("${env.DOCKER_REGISTRY}/${env.IMAGE_NAME}:${env.BUILD_TAG}")
                                docker.withRegistry("https://${env.DOCKER_REGISTRY}", env.DOCKER_CREDENTIALS) {
                                    image.push()
                                    image.push("latest")
                                }
                            }
                        }
                    }
                }
                stage('Production Image') {
                    steps {
                        container('docker') {
                            script {
                                def prodImage = docker.build("${env.DOCKER_REGISTRY}/${env.IMAGE_NAME}:${env.BUILD_TAG}-production", "-f Dockerfile.production .")
                                docker.withRegistry("https://${env.DOCKER_REGISTRY}", env.DOCKER_CREDENTIALS) {
                                    prodImage.push()
                                    if (env.BRANCH_NAME == 'main') {
                                        prodImage.push("production")
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        stage('Security Scanning') {
            when {
                anyOf {
                    branch 'main'
                    branch 'develop'
                    branch 'release/*'
                }
            }
            steps {
                container('docker') {
                    sh '''
                        # Run Trivy security scan
                        docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
                            -v $PWD:/workspace \
                            aquasec/trivy image \
                            --format json \
                            --output /workspace/trivy-report.json \
                            ${DOCKER_REGISTRY}/${IMAGE_NAME}:${BUILD_TAG}
                    '''
                }
            }
            post {
                always {
                    archiveArtifacts artifacts: 'trivy-report.json', fingerprint: true
                }
            }
        }

        stage('Deploy to Staging') {
            when {
                branch 'develop'
            }
            steps {
                container('kubectl') {
                    sh '''
                        # Update deployment with new image
                        kubectl set image deployment/pouw-blockchain \
                            pouw-blockchain=${DOCKER_REGISTRY}/${IMAGE_NAME}:${BUILD_TAG} \
                            -n pouw-staging
                        
                        kubectl set image deployment/pouw-ml-trainer \
                            pouw-ml-trainer=${DOCKER_REGISTRY}/${IMAGE_NAME}:${BUILD_TAG} \
                            -n pouw-staging
                        
                        # Wait for rollout
                        kubectl rollout status deployment/pouw-blockchain -n pouw-staging --timeout=300s
                        kubectl rollout status deployment/pouw-ml-trainer -n pouw-staging --timeout=300s
                    '''
                }
            }
        }

        stage('Staging Tests') {
            when {
                branch 'develop'
            }
            steps {
                container('python') {
                    sh '''
                        # Run smoke tests against staging environment
                        python scripts/smoke-tests.py --environment staging
                        
                        # Run integration tests
                        pytest tests/integration/ --environment staging -v
                    '''
                }
            }
        }

        stage('Deploy to Production') {
            when {
                anyOf {
                    branch 'main'
                    tag 'v*'
                }
            }
            input {
                message "Deploy to production?"
                ok "Deploy"
                submitterParameter "APPROVED_BY"
            }
            steps {
                container('kubectl') {
                    sh '''
                        # Blue-Green Deployment
                        ./scripts/blue-green-deploy.sh ${DOCKER_REGISTRY}/${IMAGE_NAME}:${BUILD_TAG}
                        
                        # Verify deployment
                        kubectl get pods -n pouw-production
                        kubectl rollout status deployment/pouw-blockchain -n pouw-production --timeout=600s
                        kubectl rollout status deployment/pouw-ml-trainer -n pouw-production --timeout=600s
                    '''
                }
            }
        }

        stage('Production Tests') {
            when {
                anyOf {
                    branch 'main'
                    tag 'v*'
                }
            }
            steps {
                container('python') {
                    sh '''
                        # Run production smoke tests
                        python scripts/smoke-tests.py --environment production
                        
                        # Run performance tests
                        pytest tests/performance/ --environment production -v
                    '''
                }
            }
        }

        stage('Create Release') {
            when {
                tag 'v*'
            }
            steps {
                container('python') {
                    sh '''
                        # Generate release notes
                        python scripts/generate-changelog.py > CHANGELOG.md
                        
                        # Create release package
                        python setup.py sdist bdist_wheel
                    '''
                }
            }
            post {
                success {
                    archiveArtifacts artifacts: 'dist/*', fingerprint: true
                }
            }
        }
    }

    post {
        always {
            // Clean up workspace
            cleanWs()
        }
        success {
            script {
                if (env.BRANCH_NAME == 'main' || env.BRANCH_NAME == 'develop') {
                    slackSend(
                        channel: '#ci-cd',
                        color: 'good',
                        message: ":white_check_mark: PoUW Pipeline SUCCESS: ${env.JOB_NAME} - ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)",
                        teamDomain: 'your-team',
                        token: env.SLACK_WEBHOOK
                    )
                }
            }
        }
        failure {
            slackSend(
                channel: '#ci-cd',
                color: 'danger',
                message: ":x: PoUW Pipeline FAILED: ${env.JOB_NAME} - ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)",
                teamDomain: 'your-team',
                token: env.SLACK_WEBHOOK
            )
        }
        unstable {
            slackSend(
                channel: '#ci-cd',
                color: 'warning',
                message: ":warning: PoUW Pipeline UNSTABLE: ${env.JOB_NAME} - ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)",
                teamDomain: 'your-team',
                token: env.SLACK_WEBHOOK
            )
        }
    }
}
