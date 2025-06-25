"""
Test runner script for the PoUW implementation.
"""

import pytest
import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

if __name__ == "__main__":
    # Run tests with verbose output and coverage
    exit_code = pytest.main(
        [
            "-v",  # Verbose output
            "--tb=short",  # Short traceback format
            "--strict-markers",  # Strict marker validation
            "tests/",  # Test directory
        ]
    )

    sys.exit(exit_code)
