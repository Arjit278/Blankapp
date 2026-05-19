import streamlit as st
import requests
from io import BytesIO
from PIL import Image
from huggingface_hub import InferenceClient
import base64

# =====================================================
# PAGE
# =====================================================

st.set_page_config(
    page_title="Pictator Pro 2026",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Pictator Pro")
st.subheader("Tombstone Seat Generator")

# =====================================================
# SECRETS
# =====================================================

OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
HF_TOKEN = st.secrets["HF_TOKEN"]

# =====================================================
# OPTIONS
# =====================================================

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
        "Fabric",
        "Premium Leather"
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
    "Generator",
    [
        "Google Nano Banana",
        "SDXL Fallback"
    ]
)

# =====================================================
# PROMPT
# =====================================================

prompt = f"""
Professional automotive catalog image.

STRICT RULES:

ONLY Tombstone seats

Integrated fixed headrests only

Headrest and seat back must be
one continuous structure

EXACTLY TWO seats:
Driver + Co-driver

Vehicle: {vehicle}

Material: {material}

Color: {color}

Requirements:

Front view
Identical seats
White studio background
Full seat visible
Premium detailing


STRICTLY FORBIDDEN:

metal rods
headrest poles
adjustable headrests
removable headrests
detachable headrests
dashboard
seat belt
rear seats
armrest
full car
"""

# =====================================================
# GENERATE
# =====================================================

if st.button(
    "Generate",
    use_container_width=True
):

    with st.spinner("Generating..."):

        try:

            # ========================================
            # GOOGLE NANO BANANA
            # ========================================

            if model=="Google Nano Banana":

                url = "https://openrouter.ai/api/v1/chat/completions"

                headers = {
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type":"application/json"
                }

                payload = {
                    "model":"google/gemini-2.5-flash-image-preview",
                    "messages":[
                        {
                            "role":"user",
                            "content":prompt
                        }
                    ]
                }

                response=requests.post(
                    url,
                    headers=headers,
                    json=payload
                )

                data=response.json()

                image_data=(
                    data["choices"][0]
                    ["message"]
                    ["images"][0]
                    ["image_url"]
                )

                img=requests.get(
                    image_data
                )

                image=Image.open(
                    BytesIO(img.content)
                )

                st.image(
                    image,
                    use_container_width=True
                )

            # ========================================
            # SDXL FALLBACK
            # ========================================

            else:

                client=InferenceClient(
                    model="stabilityai/stable-diffusion-xl-base-1.0",
                    token=HF_TOKEN
                )

                image=client.text_to_image(
                    prompt,
                    guidance_scale=14,
                    num_inference_steps=50
                )

                st.image(
                    image,
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                f"Error: {e}"
            )
