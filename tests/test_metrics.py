"""
Unit Tests for Image Engine Metrics & Preprocessing
Validates preprocessing, MSE, PSNR, and difference map generation using synthetic images.
Runs fast without downloading heavy AI models.
"""

import sys
import os
import math
import numpy as np
import pytest
from PIL import Image

# Ensure the parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from image_engine import (
    preprocess_image,
    calculate_mse,
    calculate_psnr,
    create_difference_map,
)


@pytest.fixture
def synthetic_images():
    """Provides small synthetic test images."""
    # 64x64 solid black image
    black = Image.new("RGB", (64, 64), color=(0, 0, 0))
    # 64x64 solid white image
    white = Image.new("RGB", (64, 64), color=(255, 255, 255))
    # 64x64 patterned/gradient image
    arr = np.linspace(0, 255, 64 * 64, dtype=np.uint8).reshape((64, 64))
    pattern = Image.fromarray(np.stack([arr, arr, arr], axis=-1), mode="RGB")
    return {"black": black, "white": white, "pattern": pattern}


def test_preprocess_image_dimensions():
    """Tests that preprocessing enforces dimensions divisible by 64 and within max_size."""
    # Test non-square odd-sized image
    img = Image.new("RGB", (317, 451), color=(100, 150, 200))
    processed = preprocess_image(img, max_size=512)

    w, h = processed.size
    assert w % 64 == 0, f"Width {w} must be divisible by 64"
    assert h % 64 == 0, f"Height {h} must be divisible by 64"
    assert max(w, h) <= 512, f"Dimensions exceed max_size: {w}x{h}"
    assert processed.mode == "RGB"


def test_preprocess_image_format_conversion():
    """Tests that RGBA and Grayscale images are properly converted to RGB."""
    # RGBA image with alpha channel
    rgba = Image.new("RGBA", (128, 128), color=(255, 0, 0, 128))
    processed_rgba = preprocess_image(rgba)
    assert processed_rgba.mode == "RGB"

    # Grayscale image
    gray = Image.new("L", (128, 128), color=120)
    processed_gray = preprocess_image(gray)
    assert processed_gray.mode == "RGB"


def test_mse_identical_images(synthetic_images):
    """Tests that MSE of identical images is exactly zero."""
    img = synthetic_images["pattern"]
    mse = calculate_mse(img, img)
    assert mse == 0.0, f"Expected 0.0 for identical images, got {mse}"


def test_mse_different_images(synthetic_images):
    """Tests that MSE of different images calculates the expected mathematical error."""
    black = synthetic_images["black"]
    white = synthetic_images["white"]

    # Difference between 0 and 255 on all channels is (255 - 0)^2 = 65025
    expected_mse = 255.0 ** 2  # 65025.0
    actual_mse = calculate_mse(black, white)
    assert math.isclose(actual_mse, expected_mse, rel_tol=1e-5), (
        f"Expected MSE {expected_mse}, got {actual_mse}"
    )
    assert actual_mse > 0.0


def test_mse_with_different_resolutions(synthetic_images):
    """Tests that images with differing dimensions are resized consistently for MSE."""
    img1 = synthetic_images["white"]
    img2 = synthetic_images["white"].resize((128, 128))
    mse = calculate_mse(img1, img2)
    # Both are solid white, so pixel difference after resizing should still be zero
    assert mse == 0.0


def test_psnr_identical_images(synthetic_images):
    """Tests that PSNR for identical images returns infinity."""
    img = synthetic_images["pattern"]
    psnr = calculate_psnr(img, img)
    assert math.isinf(psnr), f"Expected inf for identical images, got {psnr}"


def test_psnr_different_images(synthetic_images):
    """Tests that PSNR for known differences matches the mathematical formula."""
    black = synthetic_images["black"]
    white = synthetic_images["white"]

    # When MSE = 255^2, PSNR = 10 * log10(255^2 / 255^2) = 10 * log10(1) = 0.0 dB
    psnr = calculate_psnr(black, white)
    assert math.isclose(psnr, 0.0, abs_tol=1e-4), f"Expected PSNR 0.0 dB, got {psnr}"

    # For smaller error, PSNR should be strictly positive
    img_gray1 = Image.new("RGB", (64, 64), color=(100, 100, 100))
    img_gray2 = Image.new("RGB", (64, 64), color=(110, 110, 110))
    # diff = 10, MSE = 100, PSNR = 10 * log10(65025 / 100) = ~28.13 dB
    psnr_gray = calculate_psnr(img_gray1, img_gray2)
    assert psnr_gray > 20.0
    assert not math.isinf(psnr_gray)


def test_difference_map_generation(synthetic_images):
    """Tests that create_difference_map returns a valid non-empty Matplotlib figure."""
    img1 = synthetic_images["pattern"]
    img2 = synthetic_images["white"]

    fig = create_difference_map(img1, img2)
    assert fig is not None
    assert len(fig.axes) >= 1  # Contains image axis and colorbar axis
