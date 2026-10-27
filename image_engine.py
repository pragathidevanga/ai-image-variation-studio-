"""
AI Image Variation & Reconstruction Studio - Backend Engine
Contains functions for image preprocessing, pipeline loading, model inference,
pixel-level error metrics (MSE, PSNR), and difference heatmap visualization.
"""

import time
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib


def get_device() -> str:
    """
    Detects hardware availability and returns 'cuda' if GPU is available, else 'cpu'.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def preprocess_image(image: Image.Image, max_size: int = 512) -> Image.Image:
    """
    Validates the input image, converts it to RGB mode, preserves its aspect ratio,
    and resizes dimensions to multiples of 64 (required by Stable Diffusion VAE)
    while keeping the maximum dimension within max_size.

    Args:
        image: PIL Image to preprocess.
        max_size: Maximum allowed dimension (width or height). Default is 512.

    Returns:
        Preprocessed PIL Image in RGB format with dimensions divisible by 64.
    """
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL Image, received {type(image).__name__}")

    # Ensure RGB format (handles RGBA, grayscale, palette, etc.)
    if image.mode != "RGB":
        image = image.convert("RGB")

    orig_w, orig_h = image.size
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError(f"Invalid image dimensions: {orig_w}x{orig_h}")

    # Scale while strictly preserving aspect ratio
    scale = min(max_size / orig_w, max_size / orig_h)
    new_w = max(64, int(orig_w * scale))
    new_h = max(64, int(orig_h * scale))

    # Stable Diffusion VAE requires dimensions to be divisible by 64 (or 8)
    new_w = max(64, (new_w // 64) * 64)
    new_h = max(64, (new_h // 64) * 64)

    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def load_model(model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5", device: str = None):
    """
    Loads and configures the genuine Hugging Face Diffusers Image-to-Image pipeline.
    Supports GPU with float16 for fast inference, or fallback to CPU with float32.

    Args:
        model_id: Hugging Face model repository ID.
        device: Target device ('cuda' or 'cpu'). If None, automatically detected.

    Returns:
        Configured StableDiffusionImg2ImgPipeline.
    """
    import torch
    from diffusers import StableDiffusionImg2ImgPipeline

    if device is None:
        device = get_device()

    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[Image Engine] Loading model '{model_id}' onto device: {device} (dtype: {torch_dtype})")

    # Load pre-trained image-to-image pipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    pipe = pipe.to(device)

    # Enable memory optimizations where available
    if device == "cuda":
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass

    return pipe


def generate_variation(
    pipe,
    image: Image.Image,
    prompt: str,
    negative_prompt: str = "",
    strength: float = 0.65,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    seed: int = None,
) -> tuple[Image.Image, float]:
    """
    Executes genuine model inference using the loaded Hugging Face Diffusers pipeline.

    Args:
        pipe: Loaded diffusion pipeline.
        image: Input source image (PIL Image).
        prompt: Positive text prompt describing desired variation.
        negative_prompt: Text prompt describing undesirable attributes.
        strength: Concept preservation vs change ratio (0.0 = identical, 1.0 = brand new).
        num_inference_steps: Denoising steps (higher = finer detail, slower).
        guidance_scale: Classifier-free guidance weight.
        seed: Random seed for reproducibility.

    Returns:
        tuple (generated_pil_image, inference_duration_seconds)
    """
    import torch

    device = pipe.device.type if hasattr(pipe, "device") else get_device()
    processed_image = preprocess_image(image)

    # Configure reproducible generator if seed provided
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(seed))

    start_time = time.perf_counter()

    with torch.inference_mode():
        output = pipe(
            prompt=prompt,
            image=processed_image,
            negative_prompt=negative_prompt if negative_prompt else None,
            strength=float(strength),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
        )

    duration = time.perf_counter() - start_time
    generated_image = output.images[0]

    return generated_image, duration


def calculate_mse(original: Image.Image, generated: Image.Image) -> float:
    """
    Computes the Mean Squared Error (MSE) between corresponding RGB pixel values.

    Scale: Raw pixel intensity [0, 255].
    Formula:
        MSE = (1 / (W * H * C)) * sum((I_orig(x,y,c) - I_gen(x,y,c))^2)

    Properties:
        - MSE == 0.0 if and only if both images are identical at every pixel.
        - Higher values indicate greater pixel-level variance.

    Args:
        original: Source PIL Image.
        generated: Modified or generated PIL Image.

    Returns:
        float: Calculated MSE value.
    """
    if not isinstance(original, Image.Image) or not isinstance(generated, Image.Image):
        raise TypeError("Both inputs must be PIL Images.")

    orig_rgb = original.convert("RGB")
    gen_rgb = generated.convert("RGB")

    # Ensure identical dimensions for element-wise comparison
    if orig_rgb.size != gen_rgb.size:
        orig_rgb = orig_rgb.resize(gen_rgb.size, Image.Resampling.LANCZOS)

    arr_orig = np.asarray(orig_rgb, dtype=np.float64)
    arr_gen = np.asarray(gen_rgb, dtype=np.float64)

    mse = float(np.mean((arr_orig - arr_gen) ** 2))
    return mse


def calculate_psnr(original: Image.Image, generated: Image.Image, max_pixel: float = 255.0) -> float:
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).

    Formula:
        PSNR = 10 * log10(MAX_I^2 / MSE) = 20 * log10(MAX_I) - 10 * log10(MSE)

    Behavior:
        - If MSE == 0 (identical images), returns float('inf').
        - Typical values for lossy variations range from 15 dB to 40 dB.

    Args:
        original: Source PIL Image.
        generated: Modified or generated PIL Image.
        max_pixel: Maximum possible pixel intensity (255.0 for standard 8-bit images).

    Returns:
        float: PSNR in dB or float('inf').
    """
    mse = calculate_mse(original, generated)
    if mse == 0.0:
        return float("inf")

    psnr = 10.0 * math.log10((max_pixel ** 2) / mse)
    return float(psnr)


def create_difference_map(
    original: Image.Image,
    generated: Image.Image,
    colormap: str = "inferno",
) -> matplotlib.figure.Figure:
    """
    Generates an absolute pixel-difference heatmap using Matplotlib.
    Computes absolute error per pixel across RGB channels and visualizes the
    spatial magnitude of reconstruction divergence.

    Args:
        original: Source PIL Image.
        generated: Output PIL Image.
        colormap: Matplotlib colormap name ('inferno', 'magma', 'viridis', etc.).

    Returns:
        matplotlib.figure.Figure: Heatmap figure object.
    """
    orig_rgb = original.convert("RGB")
    gen_rgb = generated.convert("RGB")

    # Match dimensions for direct difference computation
    if orig_rgb.size != gen_rgb.size:
        orig_rgb = orig_rgb.resize(gen_rgb.size, Image.Resampling.LANCZOS)

    arr_orig = np.asarray(orig_rgb, dtype=np.float32)
    arr_gen = np.asarray(gen_rgb, dtype=np.float32)

    # Pixel-wise absolute difference averaged across RGB channels (0..255)
    abs_diff = np.mean(np.abs(arr_orig - arr_gen), axis=2)

    # Modern dark-themed visualization
    plt.close("all")
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")

    im = ax.imshow(abs_diff, cmap=colormap, vmin=0, vmax=255)
    ax.set_title("Absolute Pixel Difference Heatmap", color="#FAFAFA", fontsize=14, pad=12, fontweight="bold")
    ax.axis("off")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Absolute Pixel Error (0 = Identical, 255 = Maximum)", color="#CCCCCC", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="#CCCCCC")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#CCCCCC")

    plt.tight_layout()
    return fig
