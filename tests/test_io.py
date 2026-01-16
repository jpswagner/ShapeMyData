import sys
import os
import pytest
from pathlib import Path
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../app')))

from mask_utils import load_default_rs_mask_from_repo

def test_load_mask_missing():
    with pytest.raises(FileNotFoundError):
        load_default_rs_mask_from_repo("non_existent_file.png")

def test_load_mask_valid(tmp_path):
    # Create a dummy mask
    p = tmp_path / "test_mask.png"
    img = Image.new("RGBA", (10, 10), (255, 0, 0, 255))
    img.save(p)

    mask = load_default_rs_mask_from_repo(str(p))
    assert mask.shape == (10, 10)
    assert mask.dtype == bool
