import streamlit as st
from huggingface_hub import InferenceClient

# ====================================================
# PAGE
# ====================================================

st.set_page_config(
    page_title="Pictator Pro 2026",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Pictator Pro")
st.subheader("AI Fixed Tombstone Seat Generator")

# ====================================================
# HF TOKEN
# ====================================================

try:
    HF_TOKEN = st.secrets["HF_TOKEN"]

except Exception:

    st.error("HF_TOKEN missing")
    st.stop()

# ====================================================
# MODELS
# ====================================================

MODELS = {

    "Nedsit Seats":
    "nedsit/seats",

    "Product Photography":
    "Muapi/product-photography:fastest",

    "RealVisXL V5":
    "SG161222/RealVisXL_V5.0",

    "FLUX Dev":
    "black-forest-labs/FLUX.1-dev",

    "Canopus Interior":
    "prithivMLmods/Canopus-Interior-Architecture-0.1",

    "ControlNet Interior":
    "Lam-Hung/controlnet_lora_interior",

    "SDXL Interior":
    "fofr/sdxl-tng-interior",

    "Interior Checkpoint":
    "imagepipeline/InteriorDesign-Checkpoint",

    "ControlNet Interior Design":
    "ellljoy/controlnet-interior-design"
}

# ====================================================
# OPTIONS
# ====================================================

vehicle = st.selectbox(
    "Vehicle",
    [
        "Maruti Wagon R",
        "Grand Vitara",
        "Swift",
        "Baleno",
        "Universal"
    ]
)

material = st.selectbox(
    "Material",
    [
        "Leather",
        "Diamond Stitch",
        "Premium Leather",
        "Fabric"
    ]
)

color = st.selectbox(
    "Seat Color",
    [
        "Black",
        "Brown",
        "Beige",
        "Grey",
        "Red + Black"
    ]
)

model = st.selectbox(
    "Model",
    list(MODELS.keys())
)

# ====================================================
# STRICT PROMPT
# ====================================================

prompt = f"""
Professional automotive catalog product photo.

Generate EXACTLY TWO FRONT SEATS.

ONLY Tombstone seats.

ONLY integrated fixed headrests.

Headrest merged with backrest.

Vehicle:{vehicle}

Material:{material}

Color:{color}

Requirements:

front view
white studio background
identical seats
premium automotive detailing
full seat visible
high realism
product photography
"""

# ====================================================
# NEGATIVE PROMPT
# ====================================================

negative_prompt = """
bad hands,
bad anatomy,
ugly,
deformed,
face asymmetry,
eyes asymmetry,
deformed eyes,
deformed mouth,
open mouth,

metal rods,
headrest poles,
adjustable headrests,
removable headrests,
detachable headrests,

dashboard,
seat belts,
rear seats,
armrests,
steering wheel,
full car,

duplicate seats,
cropped image,
low quality,
blurry,
text,
watermark
"""

# ====================================================
# GENERATE
# ====================================================

if st.button(
    "Generate Seats",
    use_container_width=True
):

    with st.spinner(
        "Generating..."
    ):

        try:

            client = InferenceClient(
                provider="auto",
                api_key=HF_TOKEN
            )

            image = client.text_to_image(

                prompt,

                model=MODELS[model],

                negative_prompt=
                negative_prompt,

                guidance_scale=12,

                num_inference_steps=30
            )

            st.image(
                image,
                use_container_width=True
            )

            st.success(
                "Generation complete"
            )

            with st.expander(
                "Generation Info"
            ):

                st.write(
                    "Model:",
                    MODELS[model]
                )

                st.write(
                    "Steps:",
                    30
                )

                st.write(
                    "Guidance:",
                    12
                )

        except Exception as e:

            st.error(
                f"Generation error:\n{str(e)}"
            )

            st.write(
                "Selected model:",
                MODELS[model]
            )
