#!/bin/bash

# PoUW Deployment Test Script
# This script tests the deployment configuration locally before VPS deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test functions
test_passed() {
    echo -e "${GREEN}✅ $1${NC}"
}

test_failed() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

test_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

test_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

echo -e "${BLUE}🧪 PoUW Deployment Test Suite${NC}"
echo "=================================="

# Test 1: Check required files
test_info "Testing required deployment files..."

required_files=(
    "Dockerfile"
    "docker-compose.yml"
    "docker-compose.production.yml"
    "main.py"
    "requirements.txt"
    "nginx/nginx.conf"
    ".github/workflows/deploy.yml"
    "deploy.sh"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        test_passed "Found $file"
    else
        test_failed "Missing required file: $file"
    fi
done

# Test 2: Check Docker
test_info "Testing Docker availability..."
if command -v docker &> /dev/null; then
    test_passed "Docker is installed"
    
    # Test Docker daemon
    if docker info &> /dev/null; then
        test_passed "Docker daemon is running"
    else
        test_warning "Docker daemon is not running (start with: sudo systemctl start docker)"
    fi
else
    test_warning "Docker is not installed (will be installed during deployment)"
fi

# Test 3: Check Docker Compose
test_info "Testing Docker Compose availability..."
if command -v docker-compose &> /dev/null; then
    test_passed "Docker Compose is installed"
else
    test_warning "Docker Compose is not installed (will be installed during deployment)"
fi

# Test 4: Validate Docker Compose files
test_info "Validating Docker Compose configurations..."

if [ -f "docker-compose.yml" ]; then
    if docker-compose -f docker-compose.yml config &> /dev/null; then
        test_passed "docker-compose.yml is valid"
    else
        test_failed "docker-compose.yml has syntax errors"
    fi
fi

if [ -f "docker-compose.production.yml" ]; then
    if docker-compose -f docker-compose.production.yml config &> /dev/null; then
        test_passed "docker-compose.production.yml is valid"
    else
        test_failed "docker-compose.production.yml has syntax errors"
    fi
fi

# Test 5: Check Dockerfile syntax
test_info "Validating Dockerfile..."
if [ -f "Dockerfile" ]; then
    # Basic syntax check
    if grep -q "FROM" Dockerfile && grep -q "COPY\|ADD" Dockerfile; then
        test_passed "Dockerfile has basic required instructions"
    else
        test_warning "Dockerfile may be missing required instructions"
    fi
else
    test_failed "Dockerfile not found"
fi

# Test 6: Check Python dependencies
test_info "Checking Python requirements..."
if [ -f "requirements.txt" ]; then
    test_passed "requirements.txt found"
    
    # Check if main dependencies are present
    required_deps=("fastapi" "uvicorn" "torch" "numpy")
    for dep in "${required_deps[@]}"; do
        if grep -q "$dep" requirements.txt; then
            test_passed "Found dependency: $dep"
        else
            test_warning "Missing or uncommented dependency: $dep"
        fi
    done
else
    test_failed "requirements.txt not found"
fi

# Test 7: Check main application
test_info "Validating main application file..."
if [ -f "main.py" ]; then
    test_passed "main.py found"
    
    # Check for health endpoint
    if grep -q "health" main.py; then
        test_passed "Health endpoint implementation found"
    else
        test_warning "Health endpoint not found in main.py"
    fi
    
    # Check for proper imports
    if grep -q "from pouw" main.py; then
        test_passed "PoUW imports found"
    else
        test_warning "PoUW imports not found"
    fi
else
    test_failed "main.py not found"
fi

# Test 8: Check Nginx configuration
test_info "Validating Nginx configuration..."
if [ -f "nginx/nginx.conf" ]; then
    test_passed "nginx.conf found"
    
    # Check for SSL configuration
    if grep -q "ssl_certificate" nginx/nginx.conf; then
        test_passed "SSL configuration found"
    else
        test_warning "SSL configuration not found"
    fi
    
    # Check for health check proxy
    if grep -q "/health" nginx/nginx.conf; then
        test_passed "Health check proxy configuration found"
    else
        test_warning "Health check proxy not configured"
    fi
else
    test_failed "nginx/nginx.conf not found"
fi

# Test 9: Check GitHub Actions workflow
test_info "Validating GitHub Actions workflow..."
if [ -f ".github/workflows/deploy.yml" ]; then
    test_passed "GitHub Actions workflow found"
    
    # Check for required jobs
    if grep -q "test:" .github/workflows/deploy.yml; then
        test_passed "Test job found in workflow"
    else
        test_warning "Test job not found in workflow"
    fi
    
    if grep -q "deploy:" .github/workflows/deploy.yml; then
        test_passed "Deploy job found in workflow"
    else
        test_warning "Deploy job not found in workflow"
    fi
else
    test_failed ".github/workflows/deploy.yml not found"
fi

# Test 10: Check deployment script
test_info "Validating deployment script..."
if [ -f "deploy.sh" ]; then
    test_passed "deploy.sh found"
    
    if [ -x "deploy.sh" ]; then
        test_passed "deploy.sh is executable"
    else
        test_warning "deploy.sh is not executable (run: chmod +x deploy.sh)"
    fi
    
    # Check for key installation steps
    if grep -q "docker" deploy.sh; then
        test_passed "Docker installation found in deploy.sh"
    else
        test_warning "Docker installation not found in deploy.sh"
    fi
else
    test_failed "deploy.sh not found"
fi

# Test 11: Try building Docker image (if Docker is available)
if command -v docker &> /dev/null && docker info &> /dev/null; then
    test_info "Testing Docker image build..."
    
    if docker build -t pouw-test . &> /dev/null; then
        test_passed "Docker image builds successfully"
        
        # Clean up test image
        docker rmi pouw-test &> /dev/null || true
    else
        test_warning "Docker image build failed (check Dockerfile)"
    fi
else
    test_info "Skipping Docker build test (Docker not available)"
fi

# Summary
echo ""
echo -e "${BLUE}📊 Test Summary${NC}"
echo "================"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}🎉 All critical tests passed!${NC}"
    echo ""
    echo "Your PoUW deployment configuration appears to be ready."
    echo ""
    echo "Next steps:"
    echo "1. Update nginx/nginx.conf with your domain name"
    echo "2. Configure GitHub secrets for VPS deployment"
    echo "3. Run the deployment: git push origin main"
    echo ""
    echo "Or deploy manually to VPS:"
    echo "  scp deploy.sh root@your-vps-ip:~/"
    echo "  ssh root@your-vps-ip 'chmod +x deploy.sh && ./deploy.sh'"
else
    echo -e "${RED}❌ Some tests failed. Please fix the issues before deployment.${NC}"
    exit 1
fi
