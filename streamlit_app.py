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

    st.error(
        "HF_TOKEN missing"
    )

    st.stop()

# ====================================================
# TEXT → IMAGE MODELS
# ====================================================

TEXT_MODELS={

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

    "SD1.5 Alternative":
    "stable-diffusion-v1-5/stable-diffusion-v1-5"
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

        list(
            IMAGE_EDIT_MODELS.keys()
        )
    )

else:

    model=st.selectbox(

        "Generate Model",

        list(
            TEXT_MODELS.keys()
        )
    )

# ====================================================
# IMAGE UPLOAD
# ====================================================

uploaded=None

if mode=="Modify Existing Seat":

    uploaded=st.file_uploader(

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
            width=300
        )

# ====================================================
# PROMPTS
# ====================================================

prompt=f"""
Professional automotive catalog photo.

Generate EXACTLY TWO FRONT SEATS.

ONLY Tombstone seats.

ONLY integrated fixed headrests.

Headrest merged into backrest.

Vehicle:{vehicle}

Material:{material}

Color:{color}

Front view

White background

Identical seats

Premium detailing
"""

edit_prompt=f"""
Keep exact seat geometry unchanged.

Modify ONLY:

Material:
{material}

Color:
{color}

Vehicle:
{vehicle}

Add premium stitching.

Improve seat detailing.

Preserve:

same dimensions
same headrest
same proportions
same seat structure
same geometry

DO NOT redesign shape
"""

negative_prompt="""
metal rods,
headrest poles,
adjustable headrests,
removable headrests,
rear seats,
dashboard,
steering wheel,
full car,
armrests,
cropped,
duplicate,
watermark,
blurry
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

            # =================================
            # IMAGE EDIT
            # =================================

            if (

                mode=="Modify Existing Seat"
                and
                uploaded
            ):

                pil_image=Image.open(
                    uploaded
                )

                image=client.image_to_image(

                    image=pil_image,

                    prompt=edit_prompt,

                    model=
                    IMAGE_EDIT_MODELS[
                        model
                    ],

                    strength=0.25
                )

            # =================================
            # TEXT GENERATION
            # =================================

            else:

                image=client.text_to_image(

                    prompt,

                    model=
                    TEXT_MODELS[
                        model
                    ],

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
                f"Error:\n{str(e)}"
            )
