"""
AI Image Variation & Reconstruction Studio
A professional Streamlit web application for image-to-image variation generation,
reconstruction, and quantitative pixel-level difference evaluation.
"""

import io
import time
from PIL import Image
import streamlit as st

# Import backend engine functions
import image_engine

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Image Variation & Reconstruction Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Modern Dark Theme Styling (Purple & Blue Accents) ---
st.markdown(
    """
    <style>
    /* Global Theme Adjustments */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0d1117 0%, #161b22 100%);
        color: #e6edf3;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }

    /* Gradient Header */
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }

    /* Metric Card Styling */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(167, 139, 250, 0.4);
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #94a3b8;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-top: 4px;
    }
    .metric-badge {
        font-size: 0.75rem;
        padding: 3px 8px;
        border-radius: 6px;
        background: rgba(124, 58, 237, 0.2);
        color: #c4b5fd;
        display: inline-block;
        margin-top: 6px;
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #7c3aed 0%, #3b82f6 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.4rem;
        box-shadow: 0 4px 14px rgba(124, 58, 237, 0.35);
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #6d28d9 0%, #2563eb 100%);
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.5);
        transform: translateY(-1px);
    }

    /* Sidebar Clean Styling */
    section[data-testid="stSidebar"] {
        background-color: #0b0f17;
        border-right: 1px solid rgba(148, 163, 184, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_cached_pipeline(model_id: str, device: str):
    """Caches the model in memory across Streamlit interactions."""
    return image_engine.load_model(model_id=model_id, device=device)


# --- Hardware & System Status ---
detected_device = image_engine.get_device()

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown("### ⚙️ Studio Controls")
    st.markdown("Configure generative parameters and model settings.")

    # Hardware Information Badge
    if detected_device == "cuda":
        st.success("⚡ Hardware Acceleration: **CUDA GPU**")
    else:
        st.info("💻 Compute Device: **CPU Fallback**\n\n*(Inference will use optimized float32)*")

    st.markdown("---")
    st.markdown("#### 🤖 Model Selection")
    model_id = st.selectbox(
        "Hugging Face Pipeline",
        options=[
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "runwayml/stable-diffusion-v1-5",
        ],
        index=0,
        help="Pre-trained Hugging Face Image-to-Image diffusion pipeline.",
    )

    st.markdown("---")
    st.markdown("#### 🎛️ Diffusion Parameters")

    strength = st.slider(
        "Image Strength",
        min_value=0.10,
        max_value=0.90,
        value=0.65,
        step=0.05,
        help="Controls similarity to the original. 0.1 = nearly unchanged, 0.9 = dramatic transformation.",
    )

    num_inference_steps = st.slider(
        "Inference Steps",
        min_value=5,
        max_value=30,
        value=15 if detected_device == "cpu" else 20,
        step=1,
        help="Number of denoising steps. Higher values provide finer details but take longer.",
    )

    guidance_scale = st.slider(
        "Guidance Scale (CFG)",
        min_value=1.0,
        max_value=20.0,
        value=7.5,
        step=0.5,
        help="How closely the generation adheres to the positive prompt.",
    )

    use_seed = st.checkbox("Fix Random Seed (Reproducibility)", value=False)
    if use_seed:
        seed = st.number_input("Seed Value", min_value=0, max_value=2147483647, value=42, step=1)
    else:
        seed = None

    st.markdown("---")
    st.caption("AI Image Variation & Reconstruction Studio • Academic Submission")


# --- Main Studio Header ---
st.markdown('<div class="main-title">AI Image Variation & Reconstruction Studio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Transform images using generative AI and measure how the output differs from the original.</div>',
    unsafe_allow_html=True,
)

# --- Input Section (Image Upload & Prompts) ---
col_upload, col_prompts = st.columns([1, 1.3], gap="large")

with col_upload:
    st.markdown("#### 1. Upload Source Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        help="Upload JPG, JPEG, or PNG image to transform.",
    )

    original_pil = None
    if uploaded_file is not None:
        try:
            original_pil = Image.open(uploaded_file)
            st.image(original_pil, caption=f"Original Input ({original_pil.size[0]}×{original_pil.size[1]} px)", use_container_width=True)
        except Exception as e:
            st.error(f"Error reading image: {str(e)}")

with col_prompts:
    st.markdown("#### 2. Prompt Engineering")

    preset_prompts = [
        "Transform this photograph into a watercolor painting.",
        "Create a cinematic version of this scene.",
        "Transform this image into a digital illustration.",
        "Custom Prompt",
    ]

    selected_preset = st.selectbox("Prompt Presets", options=preset_prompts, index=0)

    if selected_preset != "Custom Prompt":
        prompt_text = st.text_area("Positive Prompt", value=selected_preset, height=85)
    else:
        prompt_text = st.text_area("Positive Prompt", value="", placeholder="Describe the style or variations you want...", height=85)

    negative_prompt_text = st.text_input(
        "Negative Prompt",
        value="blurry, low quality, distorted, artifact, oversaturated, deformed",
        help="Specify attributes to avoid in the generated output.",
    )

    generate_btn = st.button(
        "✨ Generate Image",
        disabled=(uploaded_file is None or not prompt_text.strip()),
        use_container_width=True,
    )

# --- Generation & Processing Workflow ---
if generate_btn and original_pil is not None:
    st.markdown("---")
    st.markdown("### 🚀 Running Generative Pipeline")

    progress_bar = st.progress(0, text="Initializing model pipeline...")
    status_box = st.empty()

    try:
        # Step 1: Preprocess Image
        status_box.info("Step 1/3: Validating and preprocessing image dimensions for diffusion VAE...")
        progress_bar.progress(20, text="Preprocessing image...")
        preprocessed = image_engine.preprocess_image(original_pil, max_size=512)

        # Step 2: Load Model Pipeline
        status_box.info(f"Step 2/3: Loading model '{model_id}' onto {detected_device.upper()}...")
        progress_bar.progress(45, text=f"Loading pipeline on {detected_device.upper()}...")
        pipe = get_cached_pipeline(model_id=model_id, device=detected_device)

        # Step 3: Run Inference
        status_box.info(f"Step 3/3: Running {num_inference_steps} denoising diffusion steps (strength: {strength})...")
        progress_bar.progress(70, text="Executing image-to-image diffusion...")

        generated_pil, duration = image_engine.generate_variation(
            pipe=pipe,
            image=preprocessed,
            prompt=prompt_text,
            negative_prompt=negative_prompt_text,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

        progress_bar.progress(100, text="Generation complete!")
        status_box.success(f"Generation completed successfully in **{duration:.2f} seconds**!")
        time.sleep(0.5)
        status_box.empty()
        progress_bar.empty()

        # Save results in session state for persistent view
        st.session_state["result"] = {
            "original": preprocessed,
            "generated": generated_pil,
            "duration": duration,
            "model_id": model_id,
            "settings": {
                "strength": strength,
                "steps": num_inference_steps,
                "cfg": guidance_scale,
                "seed": seed if seed is not None else "Random",
                "prompt": prompt_text,
            },
        }

    except ImportError as ie:
        status_box.empty()
        progress_bar.empty()
        st.error(
            f"**Dependency Missing**: `{str(ie)}`.\n\n"
            "Please ensure all packages in `requirements.txt` are installed (`torch`, `diffusers`, `transformers`, `accelerate`)."
        )
    except Exception as exc:
        status_box.empty()
        progress_bar.empty()
        st.error(
            f"**Inference Error**: {type(exc).__name__}: {str(exc)}\n\n"
            "If downloading the model failed, check network connectivity or Hugging Face access."
        )


# --- Display Results & Evaluation Metrics ---
if "result" in st.session_state:
    res = st.session_state["result"]
    orig_img = res["original"]
    gen_img = res["generated"]
    duration = res["duration"]
    settings = res["settings"]

    st.markdown("---")
    st.markdown("### 📊 Side-by-Side Comparison & Reconstruction Analysis")

    # Side-by-side view
    col_orig, col_gen = st.columns(2, gap="large")
    with col_orig:
        st.markdown("##### 📷 Original Source Image (Resized)")
        st.image(orig_img, use_container_width=True)

    with col_gen:
        st.markdown("##### 🎨 AI-Generated Variation")
        st.image(gen_img, use_container_width=True)

        # Download Button
        buf = io.BytesIO()
        gen_img.save(buf, format="PNG")
        st.download_button(
            label="💾 Download Generated Image",
            data=buf.getvalue(),
            file_name="ai_variation_output.png",
            mime="image/png",
            use_container_width=True,
        )

    # Compute Actual Quantitative Metrics
    mse_val = image_engine.calculate_mse(orig_img, gen_img)
    psnr_val = image_engine.calculate_psnr(orig_img, gen_img)

    st.markdown("### 📈 Quantitative Evaluation Metrics")

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Mean Squared Error (MSE)</div>
                <div class="metric-value">{mse_val:.2f}</div>
                <div class="metric-badge">Pixel intensity [0, 255]</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col2:
        psnr_display = "∞ dB" if psnr_val == float("inf") else f"{psnr_val:.2f} dB"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">PSNR</div>
                <div class="metric-value">{psnr_display}</div>
                <div class="metric-badge">Peak Signal-to-Noise</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Inference Time</div>
                <div class="metric-value">{duration:.2f}s</div>
                <div class="metric-badge">{settings['steps']} steps @ {detected_device.upper()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Strength Ratio</div>
                <div class="metric-value">{settings['strength']}</div>
                <div class="metric-badge">CFG: {settings['cfg']} | Seed: {settings['seed']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Pixel Difference Heatmap
    st.markdown("### 🗺️ Absolute Pixel Difference Heatmap")
    st.markdown(
        """
        The heatmap below calculates:
        $$\\text{Diff}(x, y) = \\frac{1}{3} \\sum_{c \\in \\{R, G, B\\}} |I_{\\text{orig}}(x,y,c) - I_{\\text{gen}}(x,y,c)|$$
        Bright regions indicate areas where generative diffusion modified or reconstructed details most significantly.
        """
    )

    heatmap_fig = image_engine.create_difference_map(orig_img, gen_img, colormap="inferno")
    st.pyplot(heatmap_fig)

    # Academic Metric Explanation
    with st.expander("ℹ️ Academic Metric Explanation & Formula Reference", expanded=False):
        st.markdown(
            """
            #### Mathematical Formulation
            - **Mean Squared Error (MSE)**:
              $$\\text{MSE} = \\frac{1}{W \\times H \\times C} \\sum_{x=1}^{W} \\sum_{y=1}^{H} \\sum_{c=1}^{C} \\left(I_{\\text{orig}}(x,y,c) - I_{\\text{gen}}(x,y,c)\\right)^2$$
              Where $W$ is width, $H$ is height, and $C=3$ is color channels on a $[0, 255]$ pixel scale.
              - **Interpretation**: $\\text{MSE} = 0$ indicates that the two images are pixel-identical. A larger MSE signifies greater pixel-level difference and stylistic deviation, not necessarily poor visual quality.

            - **Peak Signal-to-Noise Ratio (PSNR)**:
              $$\\text{PSNR} = 10 \\cdot \\log_{10}\\left(\\frac{255^2}{\\text{MSE}}\\right) = 20 \\cdot \\log_{10}(255) - 10 \\cdot \\log_{10}(\\text{MSE})$$
              - When $\\text{MSE} = 0$, PSNR is theoretically infinite ($\\infty$). Typical values for creative style transfer and image-to-image variations range between 15 dB and 35 dB.
            """
        )
