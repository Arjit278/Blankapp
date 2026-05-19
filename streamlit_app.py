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

    HF_TOKEN = st.secrets["HF_TOKEN"]

except Exception:

    st.error(
        "HF_TOKEN missing in Streamlit secrets"
    )

    st.stop()

# ====================================================
# TEXT MODELS
# ====================================================

TEXT_MODELS = {

    "FLUX Schnell":
    "black-forest-labs/FLUX.1-schnell",

    "FLUX Dev":
    "black-forest-labs/FLUX.1-dev",

    "Stable Diffusion XL":
    "stabilityai/stable-diffusion-xl-base-1.0",

    "Stable Diffusion 3.5":
    "stabilityai/stable-diffusion-3.5-large",

    "Runway Stable Diffusion 1.5":
    "runwayml/stable-diffusion-v1-5",

    "SD1.5 Alternative":
    "stable-diffusion-v1-5/stable-diffusion-v1-5"
}

# ====================================================
# IMAGE EDIT MODELS
# ====================================================

IMAGE_EDIT_MODELS = {

    "FLUX Dev":
    "black-forest-labs/FLUX.1-dev",

    "FLUX Schnell":
    "black-forest-labs/FLUX.1-schnell",

    "Runway Stable Diffusion 1.5":
    "runwayml/stable-diffusion-v1-5"
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

Headrest merged into seat body.

Vehicle:{vehicle}

Material:{material}

Color:{color}

Front view

White studio background

Identical seats

High realism

Product photography
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

Improve stitching.

Improve detailing.

Preserve:

same dimensions
same proportions
same shape
same headrest
same structure

DO NOT redesign
"""

negative_prompt = """
metal rods,
headrest poles,
adjustable headrests,
removable headrests,
rear seats,
dashboard,
steering wheel,
full car,
armrests,
duplicate,
cropped,
watermark,
blurry,
low quality
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

            client = InferenceClient(
                provider="auto",
                api_key=HF_TOKEN
            )

            # ======================================
            # IMAGE EDIT
            # ======================================

            if (
                mode=="Modify Existing Seat"
                and
                uploaded
            ):

                pil_image = Image.open(
                    uploaded
                ).convert("RGB")

                settings = {

                    "image": pil_image,
                    "prompt": edit_prompt,
                    "model":
                    IMAGE_EDIT_MODELS[
                        model
                    ]
                }

                if "FLUX" in model:

                    settings[
                        "strength"
                    ]=0.25

                else:

                    settings[
                        "strength"
                    ]=0.20

                image = client.image_to_image(
                    **settings
                )

            # ======================================
            # TEXT TO IMAGE
            # ======================================

            else:

                settings={

                    "prompt":
                    prompt,

                    "model":
                    TEXT_MODELS[
                        model
                    ]
                }

                if "FLUX" in model:

                    image = client.text_to_image(
                        **settings
                    )

                else:

                    settings[
                        "negative_prompt"
                    ]=negative_prompt

                    settings[
                        "guidance_scale"
                    ]=8

                    settings[
                        "num_inference_steps"
                    ]=20

                    image = client.text_to_image(
                        **settings
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
