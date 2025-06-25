#!/usr/bin/env python3
"""
PoUW Complete Setup and Configuration Script

This script provides a comprehensive setup process for the PoUW project,
including configuration generation, validation, and deployment preparation.
"""

import sys
import os
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_command(command: List[str], description: str = None, capture_output: bool = False) -> subprocess.CompletedProcess:
    """Run a command and handle errors"""
    if description:
        print(f"🔄 {description}...")
    
    try:
        result = subprocess.run(
            command,
            capture_output=capture_output,
            text=True,
            check=True,
            cwd=project_root
        )
        if description:
            print(f"✅ {description} completed")
        return result
    except subprocess.CalledProcessError as e:
        if description:
            print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        raise

def check_prerequisites() -> bool:
    """Check if all prerequisites are installed"""
    print("🔍 Checking prerequisites...")
    
    prerequisites = {
        'python3': 'Python 3.8+',
        'git': 'Git version control',
        'docker': 'Docker container runtime',
        'docker-compose': 'Docker Compose'
    }
    
    missing = []
    for command, description in prerequisites.items():
        try:
            subprocess.run([command, '--version'], 
                         capture_output=True, check=True)
            print(f"  ✅ {description}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"  ❌ {description}")
            missing.append(command)
    
    if missing:
        print(f"\n❌ Missing prerequisites: {', '.join(missing)}")
        print("\nPlease install the missing prerequisites and run again.")
        return False
    
    print("✅ All prerequisites found")
    return True

def setup_python_environment():
    """Set up Python virtual environment and install dependencies"""
    venv_path = project_root / "venv"
    
    if not venv_path.exists():
        run_command([sys.executable, "-m", "venv", "venv"], 
                   "Creating Python virtual environment")
    
    # Determine pip command
    if os.name == 'nt':  # Windows
        pip_cmd = str(venv_path / "Scripts" / "pip")
        python_cmd = str(venv_path / "Scripts" / "python")
    else:  # Unix-like
        pip_cmd = str(venv_path / "bin" / "pip")
        python_cmd = str(venv_path / "bin" / "python")
    
    # Upgrade pip
    run_command([python_cmd, "-m", "pip", "install", "--upgrade", "pip"],
               "Upgrading pip")
    
    # Install requirements
    requirements_file = project_root / "requirements.txt"
    if requirements_file.exists():
        run_command([pip_cmd, "install", "-r", "requirements.txt"],
                   "Installing Python dependencies")
    else:
        print("⚠️ requirements.txt not found, skipping dependency installation")

def generate_environment_config(environment: str, interactive: bool = True) -> bool:
    """Generate environment configuration with user input"""
    print(f"\n🔧 Setting up {environment} environment configuration...")
    
    env_file = project_root / f".env.{environment}"
    
    # Check if file already exists
    if env_file.exists():
        if interactive:
            response = input(f"Environment file {env_file} already exists. Regenerate? (y/N): ")
            if response.lower() != 'y':
                print("✅ Using existing environment file")
                return True
        else:
            print(f"✅ Environment file {env_file} already exists")
            return True
    
    # Gather configuration from user
    config_overrides = {}
    
    if interactive and environment == 'production':
        print("\n📝 Please provide production configuration:")
        
        domain = input("Domain name (e.g., api.yourdomain.com): ").strip()
        if domain:
            config_overrides['POUW_DOMAIN'] = domain
        
        email = input("Email address for SSL certificates: ").strip()
        if email:
            config_overrides['POUW_EMAIL'] = email
        
        vps_ip = input("VPS IP address: ").strip()
        if vps_ip:
            config_overrides['POUW_VPS_IP'] = vps_ip
        
        github_repo = input("GitHub repository URL: ").strip()
        if github_repo:
            config_overrides['POUW_GITHUB_REPO'] = github_repo
    
    # Generate environment file
    try:
        from scripts.generate_env import generate_env_file
        
        content = generate_env_file(environment, env_file, config_overrides)
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        print(f"✅ Generated {environment} environment file: {env_file}")
        
        if environment == 'production':
            print("\n🔒 Security notice:")
            print("   Secret keys have been generated automatically.")
            print("   Please review and customize the configuration as needed.")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to generate environment file: {e}")
        return False

def validate_configuration(environment: str) -> bool:
    """Validate the configuration"""
    print(f"\n🔍 Validating {environment} configuration...")
    
    try:
        result = run_command([
            sys.executable, "scripts/validate_config.py", 
            "--environment", environment
        ], f"Validating {environment} configuration", capture_output=True)
        
        print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Configuration validation failed")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return False

def generate_nginx_config(environment: str):
    """Generate nginx configuration"""
    print(f"\n🌐 Generating nginx configuration for {environment}...")
    
    try:
        nginx_config_path = project_root / "nginx" / f"nginx.{environment}.conf"
        
        run_command([
            sys.executable, "scripts/generate_nginx_config.py",
            "--environment", environment,
            "--output", str(nginx_config_path)
        ], "Generating nginx configuration")
        
        # Also generate default nginx.conf for current environment
        default_nginx_path = project_root / "nginx" / "nginx.conf"
        run_command([
            sys.executable, "scripts/generate_nginx_config.py",
            "--environment", environment,
            "--output", str(default_nginx_path)
        ], "Generating default nginx configuration")
        
    except subprocess.CalledProcessError:
        print("❌ Failed to generate nginx configuration")

def setup_docker_environment():
    """Set up Docker environment"""
    print("\n🐳 Setting up Docker environment...")
    
    # Create necessary directories
    directories = [
        "logs", "data", "cache", "nginx/ssl", "nginx/logs"
    ]
    
    for dir_name in directories:
        dir_path = project_root / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created directory: {dir_name}")
    
    # Build Docker image
    try:
        run_command(["docker", "build", "-t", "pouw:latest", "."],
                   "Building Docker image")
    except subprocess.CalledProcessError:
        print("⚠️ Docker build failed, but continuing...")

def run_tests():
    """Run the test suite"""
    print("\n🧪 Running test suite...")
    
    try:
        # Use virtual environment python if available
        venv_python = project_root / "venv" / "bin" / "python"
        if not venv_python.exists():
            venv_python = project_root / "venv" / "Scripts" / "python.exe"  # Windows
        
        python_cmd = str(venv_python) if venv_python.exists() else sys.executable
        
        run_command([python_cmd, "-m", "pytest", "tests/", "-v"],
                   "Running tests")
        
    except subprocess.CalledProcessError:
        print("⚠️ Some tests failed, but continuing setup...")

def display_setup_summary(environment: str):
    """Display setup summary and next steps"""
    print("\n" + "="*60)
    print("🎉 PoUW Setup Complete!")
    print("="*60)
    
    print(f"\n📁 Environment: {environment}")
    print(f"📄 Configuration file: .env.{environment}")
    print(f"🔧 Project directory: {project_root}")
    
    print("\n📋 Next steps:")
    
    if environment == 'development':
        print("   1. Review and customize .env.development")
        print("   2. Start development environment:")
        print("      docker-compose up")
        print("   3. Access dashboard at: http://localhost:8080")
        print("   4. API available at: http://localhost:8000")
        
    else:  # production
        print("   1. Review and customize .env.production")
        print("   2. Replace placeholder values with real ones:")
        print("      - POUW_DOMAIN: Your actual domain")
        print("      - POUW_EMAIL: Your email address")
        print("      - POUW_VPS_IP: Your VPS IP address")
        print("      - POUW_GITHUB_REPO: Your GitHub repository")
        print("   3. Deploy to production:")
        print("      ./deploy.sh")
        print("   4. Monitor deployment:")
        print("      docker-compose -f docker-compose.production.yml logs -f")
    
    print("\n🔧 Management commands:")
    print("   • Validate config: python3 scripts/validate_config.py")
    print("   • Generate nginx: python3 scripts/generate_nginx_config.py")
    print("   • Run tests: python3 -m pytest tests/")
    print("   • View logs: docker-compose logs -f")
    
    print("\n📚 Documentation:")
    print("   • Deployment: deployment docs/DEPLOYMENT.md")
    print("   • SSL Setup: deployment docs/SSL_AUTOMATION_GUIDE.md")
    print("   • Network Guide: deployment docs/NETWORK_PARTICIPATION_GUIDE.md")

def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PoUW Complete Setup Script')
    parser.add_argument('--environment', '-e', default='development',
                       choices=['development', 'production'],
                       help='Environment to set up')
    parser.add_argument('--skip-prereq-check', action='store_true',
                       help='Skip prerequisite checking')
    parser.add_argument('--skip-tests', action='store_true',
                       help='Skip running tests')
    parser.add_argument('--non-interactive', action='store_true',
                       help='Run in non-interactive mode')
    
    args = parser.parse_args()
    
    print("🚀 PoUW Project Setup")
    print("=" * 30)
    print(f"Environment: {args.environment}")
    print(f"Project root: {project_root}")
    print()
    
    try:
        # Check prerequisites
        if not args.skip_prereq_check:
            if not check_prerequisites():
                sys.exit(1)
        
        # Set up Python environment
        setup_python_environment()
        
        # Generate environment configuration
        if not generate_environment_config(args.environment, 
                                         interactive=not args.non_interactive):
            sys.exit(1)
        
        # Validate configuration
        if not validate_configuration(args.environment):
            print("\n⚠️ Configuration validation failed.")
            print("Please fix the configuration issues and run again.")
            sys.exit(1)
        
        # Generate nginx configuration
        generate_nginx_config(args.environment)
        
        # Set up Docker environment
        setup_docker_environment()
        
        # Run tests
        if not args.skip_tests:
            run_tests()
        
        # Display summary
        display_setup_summary(args.environment)
        
        print("\n✅ Setup completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n👋 Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
