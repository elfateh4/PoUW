#!/bin/bash

# PoUW Installation and Setup Script

echo "PoUW (Proof of Useful Work) Setup"
echo "=================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo ""
echo "Available commands:"
echo "  python scripts/demo.py                    # Run complete demonstration"
echo "  python scripts/start_network.py          # Start multi-node network"
echo "  python scripts/start_miner.py --help     # Start individual miner"
echo "  python scripts/submit_task.py --help     # Submit ML task"
echo "  python tests/run_tests.py                # Run test suite"
echo ""
echo "Quick start:"
echo "  source venv/bin/activate"
echo "  python scripts/demo.py --miners 3 --duration 180"
echo ""
