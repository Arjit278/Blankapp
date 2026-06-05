import streamlit as st
import numpy as np
import cv2
from huggingface_hub import InferenceClient
from PIL import Image
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

# ====================================================
# PAGE CONFIGURATION
# ====================================================
st.set_page_config(
    page_title="Pictator Pro 2026",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Pictator Pro")
st.subheader("AI Seat Generator + Editor (Advanced Structural Control)")

# ====================================================
# TOKEN MANAGEMENT
# ====================================================
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
    if len(HF_TOKEN) < 20:
        raise Exception()
except:
    st.error("HF_TOKEN missing or invalid in Streamlit secrets")
    st.code('HF_TOKEN="hf_xxxxxxxxxxxxxxxxx"')
    st.stop()

# ====================================================
# DEFINE COMPOSITE APP STATE (MODELS)
# ====================================================
TEXT_TO_IMAGE_MODELS = {
    "FLUX Schnell (HF Inference)": "black-forest-labs/FLUX.1-schnell",
    "FLUX Dev (HF Inference)": "black-forest-labs/FLUX.1-dev",
    "Stable Diffusion XL (HF Inference)": "stabilityai/stable-diffusion-xl-base-1.0",
    "ControlNet Canny (Local diffusers)": "lllyasviel/sd-controlnet-canny",  # Unique key for controlnet
}

# The HF InferenceClient (used for pure Text-to-Image and pure Image-to-Image)
inference_client = InferenceClient(api_key=HF_TOKEN)

# ====================================================
# UI: INPUTS (PERSISTENT STATE)
# ====================================================
with st.sidebar:
    st.header("1. Design Parameters")

    vehicle = st.selectbox("Vehicle Compatibility", ["Maruti Wagon R", "Grand Vitara", "Swift", "Baleno", "Universal"])
    material = st.selectbox("Material Texture", ["Leather", "Premium Leather", "Carbon Fiber Leather", "Fabric", "Alcantara"])
    stitching = st.selectbox("Stitching Pattern", ["Diamond Stitch", "Hexagon Stitch", "Parallel Lines", "Minimalist"])
    color = st.selectbox("Seat Colorway", ["Black", "Brown", "Beige", "Grey", "Red + Black", "Tan"])

    st.header("2. AI Model Configuration")
    
    selected_model_key = st.selectbox("Pipeline Model", list(TEXT_TO_IMAGE_MODELS.keys()))
    pipeline_model_id = TEXT_TO_IMAGE_MODELS[selected_model_key]
    
    # Negative prompt for general control
    negative_prompt_str = """separate headrest, adjustable headrest, metal prongs, low quality, cropped, duplicate seats, duplicate backrests, dashboard, steering wheel"""

# ====================================================
# DYNAMIC TABS FOR MODE SELECTION
# ====================================================
tab_generate, tab_control = st.tabs(["[ Mode 1: Generate New Seat ]", "[ Mode 2: Use ControlNet Structure ]"])

# ====================================================
# TAB 1: GENERATE NEW SEAT (TEXT-TO-IMAGE)
# ====================================================
with tab_generate:
    st.info("Enter details and click Generate to create a brand new seat concept.")
    
    # Persist input state across tabs
    prompt_gen = f"""Ultra realistic automotive catalog product photography, high-angle front view. Generate a single front bucket seat with an integrated fixed headrest design. Vehicle context: {vehicle}. Material: {material}. Color: {color}. Stitching pattern: {stitching}. Features: Premium leather texture, non-adjustable high-back structure, single-piece backrest-headrest unit, sharp focus, material macro detail. Pure white studio background."""

    if st.button("Generate New Seat Concept", key="gen_btn"):
        with st.spinner(f"Generating new seat with {selected_model_key}..."):
            try:
                # Direct Text-to-Image Generation (using HF API Client)
                image_gen = inference_client.text_to_image(
                    prompt_gen,
                    model=pipeline_model_id,
                    negative_prompt=negative_prompt_str,
                    num_inference_steps=25,
                    guidance_scale=7
                )
                
                # Display Result
                st.image(image_gen, caption="Generated New Seat Concept", use_container_width=True)
                st.success("New seat generation complete.")
                
                # Generation Details Expander
                with st.expander("Show Prompt and Details"):
                    st.write(f"Model ID: {pipeline_model_id}")
                    st.write("Prompt Used:")
                    st.code(prompt_gen)

            except Exception as e:
                st.error(f"Generation error: {str(e)}")

# ====================================================
# TAB 2: USE CONTROLNET STRUCTURE (ADVANCED STRUCTURAL CONTROL)
# ====================================================
with tab_control:
    st.warning("⚠️ MODE: Structurally Constraint Generation (Advanced ControlNet)")
    
    # 2.1 File Uploader for Reference Structure
    uploaded_file = st.file_uploader(
        "Upload a reference seat image to lock its structure/geometry", 
        type=["jpg", "jpeg", "png"],
        key="ctrl_uploader"
    )
    
    if uploaded_file is not None:
        # Display Uploaded Reference
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            pil_image = Image.open(uploaded_file).convert("RGB")
            st.image(pil_image, caption="Structural Reference Image", width=400)

        # 2.2 Control Parameters for Mode 2
        st.markdown("---")
        st.header("Structural Control Options")
        
        c_mode = st.radio("Control Type", ["Lineart (Strict Panels)", "Canny (Outlines)"])
        denoising_strength = st.slider("Denoising Strength (0.0=No Change, 1.0=Full New Render)", 0.0, 1.0, 0.45, help="Middle ranges preserve structure while applying new texture.")

        # 2.3 Local ControlNet Execution (Python Example workflow)
        if st.button("Apply Structural Transfer", key="ctrl_btn"):
            with st.spinner(f"Applying Advanced ControlNet ({c_mode})... (Requires CUDA)"):
                try:
                    # Persist general inputs and append structural requirements
                    prompt_ctrl = f"""Ultra realistic product photography, single seat. Keep exact geometry unchanged, strictly preserve the structural boundaries from the linework. Modify only: Vehicle incompatibility: {vehicle}, Material: {material}, Color: {color}, Stitching: {stitching}. Preserve same high-back bucket seat proportions. White studio background."""

                    # 1. Image preprocessing for ControlNet (opencv input)
                    opencv_image = np.array(pil_image)
                    opencv_image = opencv_image[:, :, ::-1].copy() # RGB to BGR
                    
                    # Apply Edge/Line detection
                    if c_mode == "Canny (Outlines)":
                        ctrl_image = cv2.Canny(opencv_image, 100, 200)
                        ctrl_image = ctrl_image[:, :, None]
                        ctrl_image = np.concatenate([ctrl_image, ctrl_image, ctrl_image], axis=2)
                        control_model_id = "lllyasviel/sd-controlnet-canny"
                    else: # Lineart
                        ctrl_image = cv2.Canny(opencv_image, 10, 30) # Lineart uses low thresholds
                        ctrl_image = ctrl_image[:, :, None]
                        ctrl_image = np.concatenate([ctrl_image, ctrl_image, ctrl_image], axis=2)
                        control_model_id = "lllyasviel/sd-controlnet-lineart"

                    # Convert back to PIL for display and pipe
                    processed_control_pil = Image.fromarray(cv2.cvtColor(ctrl_image, cv2.COLOR_BGR2RGB))
                    st.image(processed_control_pil, caption=f"Structural guide derived ({c_mode})", width=200)

                    # 2. Native Implementation Workflow (Local device withCUDA check)
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    
                    if device == "cpu":
                        st.warning("CUDA not detected. Local generation will be very slow. Consider pure T2I mode or HF API.")

                    # Load the structural adapter locally
                    controlnet = ControlNetModel.from_pretrained(control_model_id, torch_dtype=torch.float16 if device=="cuda" else torch.float32)

                    # 3. Initialize the pipeline locally with foundation and controlnet
                    # Stable Diffusion 1.5 is standard, but the example references FLUX. Local FLUX requires significant VRAM. 
                    # We will stick to the provided example framework: Stable Diffusion ControlNet.
                    
                    pipe = StableDiffusionControlNetPipeline.from_pretrained(
                        "runwayml/stable-diffusion-v1-5", # Base model
                        controlnet=controlnet,
                        torch_dtype=torch.float16 if device=="cuda" else torch.float32
                    ).to(device)
                    
                    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

                    # Local ControlNet Execution
                    final_image = pipe(
                        prompt=prompt_ctrl,
                        negative_prompt=negative_prompt_str,
                        image=processed_control_pil,
                        denoising_strength=denoising_strength,
                        num_inference_steps=20,
                        guidance_scale=7,
                        controlnet_conditioning_scale=0.9
                    ).images[0]

                    # Display Local Result
                    st.image(final_image, caption=f"ControlNet Generated Seat (Denoise: {denoising_strength})", use_container_width=True)
                    st.success("Advanced ControlNet generation complete.")
                    
                    # Clean up GPU memory
                    del controlnet, pipe
                    torch.cuda.empty_cache()

                except Exception as e:
                    st.error(f"Local ControlNet workflow failed. Ensure you have Diffusers installed locally and a GPU (for lineart/canny pipelines). Error: {str(e)}")

    else:
        st.info("Upload an image in this tab to lock its structure for accurate texture swapping.")
