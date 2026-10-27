# AI Image Variation & Reconstruction Studio

An academic AI/ML project providing a complete, local image-to-image variation, reconstruction, and quantitative divergence analysis studio.

Built with **Python**, **Streamlit**, **PyTorch**, and **Hugging Face Diffusers**.

---

## 1. Project Overview & Objective

The **AI Image Variation & Reconstruction Studio** explores how generative diffusion models transform visual concepts while preserving structural composition. It implements a local image-to-image pipeline based on pre-trained Latent Diffusion Models (Stable Diffusion v1.5) and quantifies the output's deviation from the source image through pixel-level statistical metrics:
- **Mean Squared Error (MSE)**
- **Peak Signal-to-Noise Ratio (PSNR)**
- **Absolute Pixel Difference Heatmaps**

### Key Features
- **Genuine Image-to-Image Pipeline**: Leverages Hugging Face's `StableDiffusionImg2ImgPipeline` with real pre-trained weights.
- **Hardware-Aware Acceleration**: Automatically detects CUDA GPUs for FP16 accelerated inference and seamlessly falls back to CPU FP32 execution.
- **Interactive Controls**: Fine-tune Image Strength ($0.1 - 0.9$), Inference Steps ($5 - 30$), Guidance Scale (CFG), and Random Seed for deterministic reproducibility.
- **Side-by-Side Comparison**: Visually compare the preprocessed input against the synthesized image.
- **Academic Quantitative Evaluation**: Real-time MSE and PSNR computed on actual image matrices along with an interactive colormap difference heatmap.
- **Direct Export**: Download the synthesized image in lossless PNG format.

---

## 2. Technology Stack

- **Frontend UI**: [Streamlit](https://streamlit.io/)
- **Core AI / ML Framework**: [PyTorch](https://pytorch.org/) & [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/index)
- **Model Architecture**: Stable Diffusion v1.5 (`stable-diffusion-v1-5/stable-diffusion-v1-5`)
- **Numerical Computing & Imaging**: NumPy, Pillow (PIL)
- **Scientific Visualization**: Matplotlib
- **Testing & Verification**: Pytest

---

## 3. Architecture & System Workflow

```
[User Image Upload] (JPG, PNG)
        │
        ▼
[image_engine.preprocess_image]
  • Validates format & converts to RGB
  • Preserves aspect ratio (max_size <= 512)
  • Aligns dimensions to multiples of 64 for VAE
        │
        ▼
[image_engine.load_model]
  • Detects CUDA vs CPU
  • Downloads / caches StableDiffusionImg2ImgPipeline
        │
        ▼
[image_engine.generate_variation]
  • Encodes source image into latent space via VAE
  • Adds Gaussian noise scaled by Image Strength
  • Denoises latents across N steps using UNet & text prompts
  • Decodes latent back to RGB image space via VAE
        │
        ├─────────────────────────────┬─────────────────────────────┐
        ▼                             ▼                             ▼
[calculate_mse]               [calculate_psnr]           [create_difference_map]
Pixel-wise error matrix       Decibel fidelity score     Matplotlib spatial heatmap
```

---

## 4. Installation & Windows Setup Guide

### Prerequisites
- Windows 10/11 (64-bit)
- Python 3.10 to 3.12 recommended (verify with `python --version` or `py -0`)

### Step 1: Open Windows PowerShell
Navigate to the project directory:
```powershell
cd "C:\Users\Pragathi BR\OneDrive\Documents\DSA CLASS TEST"
```

### Step 2: Create a Dedicated Virtual Environment
Using Python 3.12:
```powershell
py -3.12 -m venv .venv
```
*(Or `python -m venv .venv`)*

### Step 3: Activate the Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```
> *Note*: If PowerShell displays an execution policy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and then run the activation script again.

### Step 4: Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

*(Optional GPU Support)*: If you have an NVIDIA GPU with CUDA, install PyTorch with CUDA support:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 5. Launching the Application

Run the Streamlit studio:
```powershell
streamlit run app.py
```
Streamlit will automatically open your default browser at `http://localhost:8501`.

---

## 6. First-Run Behavior & Model Download

When you click **Generate Image** for the first time:
1. The app queries Hugging Face Hub for `stable-diffusion-v1-5/stable-diffusion-v1-5`.
2. The pre-trained model weights (~4 GB) are downloaded to your standard Hugging Face cache directory (`~/.cache/huggingface/hub`).
3. Subsequent runs load the model directly from the local cache without re-downloading.
4. The pipeline is cached in application memory using `@st.cache_resource` to avoid reloading between slider adjustments.

---

## 7. Hardware Requirements & CPU Limitations

- **With NVIDIA GPU (CUDA)**:
  - Requires 4 GB+ VRAM.
  - Generates variations in approximately **3 to 10 seconds** per image.
- **On Standard CPU (Fallback)**:
  - Requires 8 GB+ RAM.
  - Pure CPU inference processes diffusion steps sequentially.
  - **Recommended CPU Settings**:
    - Keep **Inference Steps** between **10 and 15**.
    - Keep **Max Dimension** at 512.
    - Expected inference time on modern multi-core CPUs: **45 to 120 seconds**.

---

## 8. Diffusion Parameters Explained

### Image Strength ($\text{strength} \in [0.1, 0.9]$)
- Governs how much noise is added to the source image latents before denoising.
- **Low Strength ($0.1 - 0.4$)**: Preserves source structure tightly; subtle stylistic changes.
- **Medium Strength ($0.5 - 0.7$)**: Balanced variation; transforms textures while retaining outlines.
- **High Strength ($0.8 - 0.9$)**: Severe divergence; creates an almost entirely new composition inspired loosely by the source.

### Inference Steps ($5 - 30$)
- Specifies how many iterative denoising steps the UNet conducts.
- $10-15$ steps: Fast preview with good structural clarity.
- $20-25$ steps: Rich detail and polished texture convergence.

### Guidance Scale (CFG: $1.0 - 20.0$)
- Balances adherence to the text prompt against unconditioned generation.
- Values between $7.0$ and $8.5$ are generally optimal for natural style transfer.

---

## 9. Mathematical Error Formulations

### Mean Squared Error (MSE)
Calculates the average squared difference between corresponding RGB pixel values:

$$\text{MSE} = \frac{1}{W \times H \times C} \sum_{x=1}^{W} \sum_{y=1}^{H} \sum_{c=1}^{C} \left(I_{\text{orig}}(x,y,c) - I_{\text{gen}}(x,y,c)\right)^2$$

- **Scale**: $[0, 65025]$ for 8-bit color channels ($255^2$).
- $\text{MSE} = 0.0$ if and only if images are identical at every pixel.
- Higher MSE indicates greater structural and color modification.

### Peak Signal-to-Noise Ratio (PSNR)
Measures the logarithmic ratio between maximum pixel power and reconstruction noise:

$$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{255^2}{\text{MSE}}\right) \text{ dB}$$

- If $\text{MSE} = 0$, $\text{PSNR} = \infty$.
- In creative diffusion, lower PSNR values simply confirm meaningful artistic divergence from the original image.

---

## 10. Running Unit Tests

Execute the unit test suite:
```powershell
pytest -v tests/test_metrics.py
```
All unit tests run on small synthetic image fixtures in milliseconds without requiring model downloads or internet access.

---

## 11. Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `ModuleNotFoundError: No module named 'streamlit'` | Virtual environment not active | Activate `.venv` (`.\.venv\Scripts\Activate.ps1`) |
| `OutOfMemoryError: CUDA out of memory` | GPU VRAM exhausted | Reduce `max_size` in `image_engine.py` or switch device to CPU |
| `HuggingFace Hub Connection Error` | Offline or throttled network | Ensure active internet connection on the initial model download |
| `Script execution disabled on system` | Windows PowerShell ExecutionPolicy | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |

---

## 12. Capturing Screenshots for Submission

1. Launch `streamlit run app.py`.
2. Upload a sample image (e.g., photo, landscape, portrait).
3. Select prompt preset: *"Transform this photograph into a watercolor painting."*
4. Click **Generate Image**.
5. Once generation completes, take a screenshot of:
   - The side-by-side comparison view.
   - The quantitative metric cards (MSE, PSNR, Time).
   - The absolute difference heatmap.
6. Save the captured image to the `screenshots/` directory (e.g. `screenshots/app_evaluation_demo.png`).
