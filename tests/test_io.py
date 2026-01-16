import sys
import os
import pytest
from pathlib import Path
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../app')))

from mask_utils import load_mask_from_wkt_string

def test_load_mask_from_wkt():
    # Simple square
    wkt = "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"
    mask = load_mask_from_wkt_string(wkt, max_side=100)

    # 0 0 to 10 10 is aspect ratio 1:1.
    # Scaled to 100x100.
    assert mask.shape == (100, 100)
    # The square covers the whole image (filled=1)
    # allow some margin error due to rasterization
    assert mask.sum() > 9000

def test_load_mask_from_wkt_multipolygon():
    # Two squares
    wkt = "MULTIPOLYGON (((0 0, 10 0, 10 10, 0 10, 0 0)), ((20 20, 30 20, 30 30, 20 30, 20 20)))"
    # Bounds: 0 0 to 30 30. Size 30x30.
    mask = load_mask_from_wkt_string(wkt, max_side=30)

    assert mask.shape == (30, 30)
    # Should have two blocks filled.
    # 0-10 (approx 1/3) and 20-30 (approx 1/3).
    # Total filled approx 2/9?
    # 10x10 + 10x10 = 200 units sq. Total 30x30=900. 200/900 = 0.22.
    # 30x30 pixels.
    # mask sum approx 0.22 * 900 = 200.
    assert 150 < mask.sum() < 250
