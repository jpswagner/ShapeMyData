import sys
import os
import pytest
import numpy as np

# Add app to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../app')))

from mask_utils import normalize_targets

def test_normalize_targets_sum_exact():
    total_pixels = 1000
    values = [10, 20, 30]
    targets = normalize_targets(values, total_pixels)
    assert sum(targets) == total_pixels

    # Check proportionality roughly
    # 10/60 * 1000 = 166.66 -> 167
    # 20/60 * 1000 = 333.33 -> 333
    # 30/60 * 1000 = 500.00 -> 500
    assert targets == [167, 333, 500]

def test_normalize_targets_edge_cases():
    total_pixels = 100
    values = [0, 10]
    targets = normalize_targets(values, total_pixels)
    assert sum(targets) == 100
    assert targets == [0, 100]

    # Large numbers
    values = [1e9, 1e9]
    targets = normalize_targets(values, 100)
    assert sum(targets) == 100
    assert targets == [50, 50]

def test_normalize_targets_remainder_distribution():
    # 3 items, 10 pixels. 1/3 each = 3.33 -> 3.
    # Total 9. Remainder 1.
    # Should go to one of them.
    targets = normalize_targets([1, 1, 1], 10)
    assert sum(targets) == 10
    # One should be 4, others 3.
    assert sorted(targets) == [3, 3, 4]
