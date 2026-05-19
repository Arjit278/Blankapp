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

st.title("🚗 Pictator Pro 2026")
st.subheader("Fixed Tombstone Seat Generator")

# ====================================================
# SECRETS
# ====================================================

try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except:
    st.error(
        "HF_TOKEN missing in Streamlit Secrets"
    )
    st.stop()

# ====================================================
# MODELS
# ====================================================

MODELS = {
    "FLUX Schnell":"black-forest-labs/FLUX.1-schnell",
    "SDXL":"stabilityai/stable-diffusion-xl-base-1.0",
    "SD 1.5":"runwayml/stable-diffusion-v1-5"
}

# ====================================================
# SIDEBAR
# ====================================================

with st.sidebar:

    st.header("Seat Settings")

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

    model_name = st.selectbox(
        "Image Model",
        list(MODELS.keys())
    )

# ====================================================
# GENERATION
# ====================================================

if st.button(
    "Generate Tombstone Seats",
    use_container_width=True
):

    prompt = f"""
Professional studio automotive catalog image.

STRICT REQUIREMENTS:

ONLY tombstone seats

Headrest must be fixed permanently.

Headrest and backrest must be
one single integrated body.

Show EXACTLY TWO front seats:
Driver seat + Co-driver seat

Vehicle:
{vehicle}

Material:
{material}

Color:
{color}

Seat requirements:

identical pattern
fully visible
front view
upright
one-piece design
fixed headrest
high quality leather details
white background


STRICTLY FORBIDDEN:

removable headrest
detachable headrest
headrest posts
metal rods
adjustable headrest
seat belt
armrest
center console
rear seats
dashboard
steering wheel
full car
collage
extra objects
"""

    negative_prompt = """
metal rods,
headrest supports,
headrest poles,
detachable headrest,
adjustable headrest,
removable headrest,
seat belt,
dashboard,
steering wheel,
armrest,
car interior,
vehicle,
rear seat
"""

    try:

        with st.spinner(
            "Generating..."
        ):

            client = InferenceClient(
                model=MODELS[model_name],
                token=HF_TOKEN
            )

            image = client.text_to_image(
                prompt,
                negative_prompt=negative_prompt,
                guidance_scale=13,
                num_inference_steps=40
            )

            st.image(
                image,
                use_container_width=True
            )

            st.success(
                "Generation complete"
            )

    except Exception:

        st.warning(
            "Primary model unavailable. Switching to SDXL..."
        )

        try:

            client = InferenceClient(
                model="stabilityai/stable-diffusion-xl-base-1.0",
                token=HF_TOKEN
            )

            image = client.text_to_image(
                prompt,
                negative_prompt=negative_prompt,
                guidance_scale=13,
                num_inference_steps=40
            )

            st.image(
                image,
                use_container_width=True
            )

            st.success(
                "Generated using SDXL fallback"
            )

        except Exception as e:

            st.error(
                f"Generation failed:\n{e}"
            )
