import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dff_modified_euler import modified_euler


def test_modified_euler_not_implemented():
    """Test that modified_euler raises NotImplementedError when called."""
    with pytest.raises(NotImplementedError):
        modified_euler()


if __name__ == "__main__":
    test_modified_euler_not_implemented()
    print("Test passed!") 