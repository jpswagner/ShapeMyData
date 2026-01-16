import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../app')))

from segmentation import (
    assign_segments_vertical,
    assign_segments_horizontal,
    assign_segments_angular,
    assign_segments_contiguous_wedge
)

@pytest.fixture
def simple_mask():
    # 10x10 square
    return np.ones((10, 10), dtype=bool)

def test_segmentation_vertical_pixel_counts(simple_mask):
    total = 100
    targets = [30, 70]
    seg_map = assign_segments_vertical(simple_mask, targets, "left_to_right")

    counts = [np.sum(seg_map == i) for i in range(len(targets))]
    assert counts == targets
    assert not np.any(seg_map == -1)

def test_segmentation_horizontal_pixel_counts(simple_mask):
    total = 100
    targets = [10, 20, 70]
    seg_map = assign_segments_horizontal(simple_mask, targets, "top_to_bottom")

    counts = [np.sum(seg_map == i) for i in range(len(targets))]
    assert counts == targets
    assert not np.any(seg_map == -1)

def test_segmentation_angular_pixel_counts(simple_mask):
    total = 100
    targets = [25, 25, 25, 25]
    seg_map = assign_segments_angular(simple_mask, targets)

    counts = [np.sum(seg_map == i) for i in range(len(targets))]
    assert counts == targets
    assert not np.any(seg_map == -1)

def test_segmentation_contiguous_wedge_pixel_counts(simple_mask):
    total = 100
    targets = [50, 50]
    seg_map = assign_segments_contiguous_wedge(simple_mask, targets)

    counts = [np.sum(seg_map == i) for i in range(len(targets))]
    assert counts == targets
    assert not np.any(seg_map == -1)

def test_segmentation_irregular_mask():
    # Create a mask with a hole
    mask = np.ones((10, 10), dtype=bool)
    mask[4:6, 4:6] = False # 4 pixels removed. Total 96.
    targets = [32, 32, 32]

    seg_map = assign_segments_angular(mask, targets)
    # Check only valid pixels are assigned
    assert np.all(seg_map[~mask] == -1)

    counts = [np.sum(seg_map == i) for i in range(len(targets))]
    assert counts == targets
