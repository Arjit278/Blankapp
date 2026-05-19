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
    "AI Fixed Integrated Headrest Seat Generator + Editor"
)

# ====================================================
# TOKEN
# ====================================================

try:

    HF_TOKEN = st.secrets["HF_TOKEN"]

    if len(HF_TOKEN) < 20:
        raise Exception()

except:

    st.error(
        "HF_TOKEN missing or invalid in Streamlit secrets"
    )

    st.code(
"""
HF_TOKEN="hf_xxxxxxxxxxxxxxxxx"
"""
    )

    st.stop()

# ====================================================
# CLIENT
# ====================================================

client = InferenceClient(
    api_key=HF_TOKEN
)

# ====================================================
# TEXT MODELS
# ====================================================

TEXT_MODELS = {

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

IMAGE_EDIT_MODELS = {

    "FLUX Dev":
    "black-forest-labs/FLUX.1-dev",

    "FLUX Schnell":
    "black-forest-labs/FLUX.1-schnell"
}

# ====================================================
# MODE
# ====================================================

mode = st.radio(

    "Mode",

    [
        "Generate New Seat",
        "Modify Existing Seat"
    ]
)

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

# ====================================================
# MODEL PICKER
# ====================================================

if mode == "Modify Existing Seat":

    model = st.selectbox(

        "Edit Model",

        list(
            IMAGE_EDIT_MODELS.keys()
        )
    )

else:

    model = st.selectbox(

        "Generate Model",

        list(
            TEXT_MODELS.keys()
        )
    )

# ====================================================
# IMAGE UPLOAD
# ====================================================

uploaded = None

if mode == "Modify Existing Seat":

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
            width=450
        )

# ====================================================
# GENERATE PROMPT
# ====================================================

prompt = f"""
Ultra realistic automotive catalog photograph.

Generate EXACTLY TWO front seats.

Vehicle:
{vehicle}

Material:
{material}

Color:
{color}

Requirements:

Integrated headrest seats with a distinct one-piece design

Tombstone seats

The headrest and backrest must form a single, unbroken unit

Non-adjustable high-back structure seat style

Premium seat cover design

Diamond stitching

Real leather texture

Front view

White studio background

Product photography

High realism

Maruti Wagon R style high-back bucket seat shape
"""

# ====================================================
# EDIT PROMPT
# ====================================================

edit_prompt = f"""
Keep exact geometry unchanged.

DO NOT redesign shape.

Strictly maintain the integrated headrest architecture. The headrest and backrest must remain a single, unbroken unit.

Modify only:

Material:
{material}

Color:
{color}

Vehicle:
{vehicle}

Apply:

diamond stitching

premium leather texture

red-black contrast

luxury seat cover finish

Preserve:

same shape

same dimensions

same integrated non-adjustable headrest

same high-back proportions

same structure
"""

# ====================================================
# NEGATIVE
# ====================================================

negative_prompt = """
separate headrest,
adjustable headrest,
removable headrest,
metal headrest prongs,
gaps between seat and headrest,
blurry,
low quality,
cropped,
watermark,
text,
duplicate seats,
extra seats,
dashboard,
steering wheel,
full car,
rear seats,
bad anatomy
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

            # =================================
            # IMAGE EDIT
            # =================================

            if (
                mode=="Modify Existing Seat"
                and
                uploaded
            ):

                pil_image = Image.open(
                    uploaded
                ).convert(
                    "RGB"
                )

                image = client.image_to_image(

                    image=pil_image,

                    prompt=edit_prompt,

                    model=
                    IMAGE_EDIT_MODELS[
                        model
                    ],

                    strength=0.18
                )

            # =================================
            # TEXT TO IMAGE
            # =================================

            else:

                image = client.text_to_image(

                    prompt,

                    model=
                    TEXT_MODELS[
                        model
                    ],

                    negative_prompt=
                    negative_prompt,

                    guidance_scale=7,

                    num_inference_steps=20
                )

            st.image(
                image,
                use_container_width=True
            )

            st.success(
                "Generation complete"
            )

            with st.expander(
                "Generation Details"
            ):

                st.write(
                    "Mode:",
                    mode
                )

                st.write(
                    "Model:",
                    model
                )

        except Exception as e:

            st.error(
                f"Generation error:\n{str(e)}"
            )

            with st.expander(
                "Debug Info"
            ):

                st.write(
                    "Mode:",
                    mode
                )

                st.write(
                    "Selected model:",
                    model
                )
