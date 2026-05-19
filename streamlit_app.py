import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image

# ====================================================
# PAGE
# ====================================================

st.set_page_config(
    page_title="Pictator Pro 2026",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Pictator Pro")
st.subheader("AI Fixed Tombstone Seat Generator + Editor")

# ====================================================
# HF TOKEN
# ====================================================

try:
    HF_TOKEN = st.secrets["HF_TOKEN"]

except Exception:

    st.error(
        "HF_TOKEN missing"
    )

    st.stop()

# ====================================================
# VERIFIED MODELS
# ====================================================

MODELS = {

    "FLUX Schnell":
    "black-forest-labs/FLUX.1-schnell",

    "FLUX Dev":
    "black-forest-labs/FLUX.1-dev",

    "Stable Diffusion XL":
    "stabilityai/stable-diffusion-xl-base-1.0",

    "Stable Diffusion 3.5":
    "stabilityai/stable-diffusion-3.5-large",

    "Stable Diffusion 1.5":
    "runwayml/stable-diffusion-v1-5",

    "SD 1.5 Alternative":
    "stable-diffusion-v1-5/stable-diffusion-v1-5"
}

# ====================================================
# OPTIONS
# ====================================================

mode = st.radio(
    "Mode",
    [
        "Generate New Seat",
        "Modify Existing Seat"
    ]
)

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
# IMAGE UPLOAD
# ====================================================

uploaded = None

if mode=="Modify Existing Seat":

    uploaded = st.file_uploader(
        "Upload Seat Reference",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded:

        st.image(
            uploaded,
            caption="Reference Image",
            width=300
        )

# ====================================================
# PROMPTS
# ====================================================

prompt = f"""
Professional automotive catalog image.

Generate EXACTLY TWO FRONT SEATS.

ONLY Tombstone seats.

ONLY integrated fixed headrests.

Headrest merged with backrest.

Vehicle:{vehicle}

Material:{material}

Color:{color}

Front view

White background

Premium product photography
"""

edit_prompt = f"""
Keep seat geometry unchanged.

Modify ONLY:

Material:
{material}

Color:
{color}

Vehicle:
{vehicle}

Change stitching pattern.

Improve detailing.

Preserve:

same shape
same dimensions
same headrest
same proportions
same seat structure

DO NOT change geometry
"""

negative_prompt="""
metal rods,
headrest poles,
removable headrests,
adjustable headrests,
rear seats,
dashboard,
steering wheel,
full car,
armrests,
duplicate,
blurry,
watermark
"""

# ====================================================
# GENERATE
# ====================================================

if st.button(
    "Generate",
    use_container_width=True
):

    with st.spinner(
        "Generating..."
    ):

        try:

            client=InferenceClient(
                provider="auto",
                api_key=HF_TOKEN
            )

            # ====================================
            # IMAGE EDIT MODE
            # ====================================

            if (
                mode=="Modify Existing Seat"
                and
                uploaded
            ):

                image = client.image_to_image(

                    image=uploaded,

                    prompt=edit_prompt,

                    model=MODELS[model],

                    strength=0.25
                )

            # ====================================
            # TEXT TO IMAGE MODE
            # ====================================

            else:

                image = client.text_to_image(

                    prompt,

                    model=MODELS[model],

                    negative_prompt=
                    negative_prompt,

                    guidance_scale=10,

                    num_inference_steps=25
                )

            st.image(
                image,
                use_container_width=True
            )

            st.success(
                "Done"
            )

        except Exception as e:

            st.error(
                f"Error:\n{str(e)}"
            )

            st.write(
                "Model:",
                MODELS[model]
            )
