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
st.subheader(
    "AI Fixed Tombstone Seat Generator + Editor"
)

# ====================================================
# TOKEN
# ====================================================

try:
    HF_TOKEN=st.secrets["HF_TOKEN"]

except:

    st.error("HF_TOKEN missing")
    st.stop()

# ====================================================
# TEXT GENERATION MODELS
# ====================================================

TEXT_MODELS={

    "FLUX Schnell":
    "black-forest-labs/FLUX.1-schnell",

    "FLUX Dev":
    "black-forest-labs/FLUX.1-dev",

    "Runway Stable Diffusion 1.5":
    "runwayml/stable-diffusion-v1-5",

    "Stable Diffusion XL":
    "stabilityai/stable-diffusion-xl-base-1.0"
}

# ====================================================
# IMAGE EDIT MODELS
# ====================================================

IMAGE_EDIT_MODELS={

    "FLUX Dev":
    "black-forest-labs/FLUX.1-dev",

    "FLUX Schnell":
    "black-forest-labs/FLUX.1-schnell"
}

# ====================================================
# MODE
# ====================================================

mode=st.radio(

    "Mode",

    [
        "Generate New Seat",
        "Modify Existing Seat"
    ]
)

# ====================================================
# OPTIONS
# ====================================================

vehicle=st.selectbox(
    "Vehicle",
    [
        "Maruti Wagon R",
        "Grand Vitara",
        "Swift",
        "Baleno",
        "Universal"
    ]
)

material=st.selectbox(
    "Material",
    [
        "Leather",
        "Diamond Stitch",
        "Premium Leather",
        "Fabric"
    ]
)

color=st.selectbox(
    "Seat Color",
    [
        "Black",
        "Brown",
        "Beige",
        "Grey",
        "Red + Black"
    ]
)

# ====================================================
# MODEL PICKER
# ====================================================

if mode=="Modify Existing Seat":

    model=st.selectbox(
        "Edit Model",
        list(IMAGE_EDIT_MODELS.keys())
    )

else:

    model=st.selectbox(
        "Generate Model",
        list(TEXT_MODELS.keys())
    )

# ====================================================
# IMAGE UPLOAD
# ====================================================

uploaded=None

if mode=="Modify Existing Seat":

    uploaded=st.file_uploader(
        "Upload Seat Reference",
        type=["jpg","jpeg","png"]
    )

    if uploaded:

        st.image(
            uploaded,
            caption="Reference Image",
            width=450
        )

# ====================================================
# GENERATION PROMPT
# ====================================================

prompt=f"""
Ultra realistic automotive product photo.

Generate exactly TWO front seats.

Vehicle:
{vehicle}

Material:
{material}

Color:
{color}

Modern premium seat covers

Integrated headrests

Detailed stitching

front view

studio lighting

high realism

catalog photography

Wagon R style seat shape
"""

# ====================================================
# EDIT PROMPT
# ====================================================

edit_prompt=f"""
Keep original seat geometry.

DO NOT redesign seat.

Change ONLY:

Material:
{material}

Color:
{color}

Vehicle:
{vehicle}

Add:

premium stitching
diamond pattern
improved detailing
premium leather texture

Preserve:

same dimensions
same seat shape
same seat structure
same headrest
same proportions
"""

negative_prompt="""
extra seats,
duplicate seats,
blurry,
low quality,
watermark,
text,
cropped,
dashboard,
steering wheel,
car interior redesign
"""

# ====================================================
# GENERATE
# ====================================================

if st.button(
    "Generate",
    use_container_width=True
):

    with st.spinner("Generating..."):

        try:

            client=InferenceClient(
                provider="auto",
                api_key=HF_TOKEN
            )

            # ==================================
            # MODIFY IMAGE
            # ==================================

            if (
                mode=="Modify Existing Seat"
                and
                uploaded
            ):

                pil_image=Image.open(
                    uploaded
                ).convert("RGB")

                image=client.image_to_image(

                    image=pil_image,

                    prompt=edit_prompt,

                    model=
                    IMAGE_EDIT_MODELS[
                        model
                    ],

                    strength=0.18
                )

            # ==================================
            # TEXT GENERATION
            # ==================================

            else:

                image=client.text_to_image(

                    prompt,

                    model=
                    TEXT_MODELS[
                        model
                    ],

                    negative_prompt=
                    negative_prompt,

                    guidance_scale=8,

                    num_inference_steps=20
                )

            st.image(
                image,
               
