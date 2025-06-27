#!/bin/bash

# PoUW CLI Installation Script
# This script sets up the PoUW CLI for easy use

set -e

echo "=== PoUW CLI Installation Script ==="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "pouw-cli" ]; then
    print_error "pouw-cli script not found. Please run this script from the PoUW project root directory."
    exit 1
fi

if [ ! -d "pouw" ]; then
    print_error "pouw directory not found. Please run this script from the PoUW project root directory."
    exit 1
fi

print_info "Found PoUW project structure"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if python3 -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)"; then
    print_success "Python version check passed ($(python3 --version))"
else
    print_error "Python 3.8+ is required. Found: $(python3 --version)"
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    print_warning "Virtual environment is not activated"
    if [ -d ".venv" ]; then
        print_info "Found .venv directory. Activating virtual environment..."
        source .venv/bin/activate
        print_success "Virtual environment activated"
    else
        print_info "Creating virtual environment..."
        python3 -m venv .venv
        source .venv/bin/activate
        print_success "Virtual environment created and activated"
    fi
else
    print_success "Virtual environment is active: $VIRTUAL_ENV"
fi

# Install dependencies
print_info "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt > /dev/null 2>&1
    print_success "Dependencies installed from requirements.txt"
else
    print_warning "requirements.txt not found. Installing basic dependencies..."
    pip install psutil > /dev/null 2>&1
    print_success "Basic dependencies installed"
fi

# Make CLI executable
print_info "Setting up CLI permissions..."
chmod +x pouw-cli
print_success "CLI script is now executable"

# Create necessary directories
print_info "Creating directory structure..."
mkdir -p configs logs pids data keys
print_success "Directory structure created"

# Configure firewall for node connections
print_info "Configuring firewall for PoUW node connections..."

# Default PoUW ports
POUW_PORTS="8333,8334,8335,8336,8337"

# Check if ufw is available and active
if command -v ufw >/dev/null 2>&1; then
    if ufw status | grep -q "Status: active"; then
        print_info "UFW firewall detected and active. Opening PoUW ports..."
        
        # Open default PoUW ports
        for port in $(echo $POUW_PORTS | tr ',' ' '); do
            if ufw allow $port/tcp >/dev/null 2>&1; then
                print_success "Opened port $port/tcp"
            else
                print_warning "Could not open port $port/tcp (may require sudo)"
            fi
        done
        
        print_info "Reloading UFW rules..."
        if ufw reload >/dev/null 2>&1; then
            print_success "UFW rules reloaded"
        else
            print_warning "Could not reload UFW rules"
        fi
    else
        print_warning "UFW firewall is installed but not active"
        print_info "You may need to manually open ports: $POUW_PORTS"
    fi
elif command -v firewall-cmd >/dev/null 2>&1; then
    # For RHEL/CentOS/Fedora systems with firewalld
    print_info "Firewalld detected. Opening PoUW ports..."
    
    for port in $(echo $POUW_PORTS | tr ',' ' '); do
        if firewall-cmd --permanent --add-port=$port/tcp >/dev/null 2>&1; then
            print_success "Opened port $port/tcp in firewalld"
        else
            print_warning "Could not open port $port/tcp in firewalld (may require sudo)"
        fi
    done
    
    if firewall-cmd --reload >/dev/null 2>&1; then
        print_success "Firewalld rules reloaded"
    else
        print_warning "Could not reload firewalld rules"
    fi
else
    print_warning "No recognized firewall manager found (ufw/firewalld)"
    print_info "If you have a firewall, manually open these ports:"
    print_info "  TCP ports: $POUW_PORTS"
    print_info "  Example for iptables:"
    for port in $(echo $POUW_PORTS | tr ',' ' '); do
        print_info "    iptables -A INPUT -p tcp --dport $port -j ACCEPT"
    done
fi

# Display network configuration info
print_info "Network configuration completed"
echo
print_info "Default PoUW ports that should be accessible:"
echo "  • 8333: Default node listening port"
echo "  • 8334-8337: Additional node ports"
echo
print_info "To check if ports are open:"
echo "  • Local test: nc -zv localhost 8333"
echo "  • External test: nmap -p 8333 <your-ip>"
echo

# Test CLI functionality
print_info "Testing CLI functionality..."
if ./pouw-cli --help > /dev/null 2>&1; then
    print_success "CLI is working correctly"
else
    print_error "CLI test failed"
    exit 1
fi

# Add to PATH (optional)
echo
read -p "Do you want to add PoUW CLI to your PATH? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    current_dir=$(pwd)
    
    # Check if ~/.bashrc exists
    if [ -f "$HOME/.bashrc" ]; then
        echo "export PATH=\"$current_dir:\$PATH\"" >> "$HOME/.bashrc"
        print_success "Added to ~/.bashrc. Restart your terminal or run 'source ~/.bashrc'"
    fi
    
    # Check if ~/.zshrc exists (for zsh users)
    if [ -f "$HOME/.zshrc" ]; then
        echo "export PATH=\"$current_dir:\$PATH\"" >> "$HOME/.zshrc"
        print_success "Added to ~/.zshrc. Restart your terminal or run 'source ~/.zshrc'"
    fi
    
    # Create symlink in /usr/local/bin if user has permission
    if [ -w "/usr/local/bin" ]; then
        ln -sf "$current_dir/pouw-cli" /usr/local/bin/pouw-cli
        print_success "Created symlink in /usr/local/bin"
    else
        print_warning "Cannot create symlink in /usr/local/bin (no write permission)"
        print_info "You can run: sudo ln -sf $current_dir/pouw-cli /usr/local/bin/pouw-cli"
    fi
fi

echo
print_success "=== Installation Complete ==="
echo
echo "You can now use the PoUW CLI with the following commands:"
echo
echo "  # Interactive mode (recommended for beginners)"
echo "  ./pouw-cli interactive"
echo
echo "  # Show help"
echo "  ./pouw-cli --help"
echo
echo "  # Create a worker node configuration"
echo "  ./pouw-cli config create --node-id worker-1 --template worker"
echo
echo "  # Start a node"
echo "  ./pouw-cli start --node-id worker-1"
echo
echo "  # Check node status"
echo "  ./pouw-cli status --node-id worker-1"
echo
echo "  # View logs"
echo "  ./pouw-cli logs --node-id worker-1"
echo
echo "  # Connect to peer directly"
echo "  ./pouw-cli connect --address 192.168.1.100 --port 8333"
echo
echo "  # Show node network information"
echo "  ./pouw-cli node-info --node-id worker-1"
echo
echo "  # Show all listening nodes"
echo "  ./pouw-cli listen"
echo
echo "💡 TIP: Use './pouw-cli interactive' for a user-friendly menu interface!"
echo "For detailed usage instructions, see CLI_GUIDE.md"
echo

# Quick setup option
echo
read -p "Do you want to create a sample node configuration now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo
    echo "You can either:"
    echo "1) Use interactive mode (recommended)"
    echo "2) Use command line setup"
    echo
    read -p "Choose setup method (1-2): " -n 1 -r
    echo
    
    if [[ $REPLY == "1" ]]; then
        print_info "Starting interactive mode for setup..."
        ./pouw-cli interactive
    else
        echo "Available node types:"
        echo "1) worker   - ML training node"
        echo "2) miner    - Blockchain mining node"  
        echo "3) supervisor - Coordination node"
        echo "4) hybrid   - Multi-purpose node"
        echo
        read -p "Choose node type (1-4): " -n 1 -r
        echo
        
        case $REPLY in
            1) node_type="worker" ;;
            2) node_type="miner" ;;
            3) node_type="supervisor" ;;
            4) node_type="hybrid" ;;
            *) node_type="worker" ;;
        esac
        
        read -p "Enter node ID (default: sample-$node_type): " node_id
        node_id=${node_id:-"sample-$node_type"}
        
        read -p "Enter port (default: 8333): " port
        port=${port:-8333}
        
        print_info "Creating $node_type node configuration..."
        
        extra_args=""
        if [ "$node_type" = "miner" ]; then
            extra_args="--mining"
        elif [ "$node_type" = "worker" ] || [ "$node_type" = "supervisor" ]; then
            extra_args="--training"
        elif [ "$node_type" = "hybrid" ]; then
            extra_args="--mining --training"
        fi
        
        ./pouw-cli config create --node-id "$node_id" --template "$node_type" --port "$port" $extra_args
        
        print_success "Sample configuration created!"
        echo
        echo "You can now start your node with:"
        echo "  ./pouw-cli start --node-id $node_id"
        echo "Or use interactive mode:"
        echo "  ./pouw-cli interactive"
        echo
    fi
fi

print_success "Setup complete! Happy mining and training! 🚀" 