import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image
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
st.subheader("Fixed Tombstone Seat Generator")

# ====================================================
# SECRETS
# ====================================================

try:
    OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
    HF_TOKEN = st.secrets["HF_TOKEN"]

except Exception:
    st.error("Missing API keys in Streamlit secrets")
    st.stop()

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
    "Color",
    [
        "Black",
        "Brown",
        "Beige",
        "Grey",
        "Red + Black"
    ]
)

generator = st.selectbox(
    "Generator",
    [
        "Google Nano Banana",
        "SDXL Fallback"
    ]
)

# ====================================================
# SHORT PROMPT
# ====================================================

prompt = f"""
Automotive studio catalog image.

Two front Tombstone seats only.

Integrated fixed headrests only.
Headrest merged with backrest.

Vehicle:{vehicle}
Material:{material}
Color:{color}

Front view
Identical seats
White background

Do not generate:
metal rods,
headrest poles,
adjustable headrests,
removable headrests,
rear seats,
dashboard,
steering wheel,
full car,
armrests
"""

# ====================================================
# GENERATE
# ====================================================

if st.button(
    "Generate Seats",
    use_container_width=True
):

    with st.spinner("Generating..."):

        try:

            # ======================================
            # GOOGLE NANO BANANA
            # ======================================

            if generator == "Google Nano Banana":

                headers = {
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type":"application/json"
                }

                payload = {

                    "model":"google/gemini-2.5-flash-image",

                    # reduce credit use
                    "max_tokens":256,

                    "temperature":0.2,

                    "messages":[
                        {
                            "role":"user",
                            "content":prompt
                        }
                    ]
                }

                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120
                )

                data=response.json()

                # Handle API errors

                if "error" in data:

                    st.error(
                        data["error"]["message"]
                    )

                    st.stop()

                with st.expander(
                    "API Response"
                ):
                    st.json(data)

                if "choices" not in data:

                    raise Exception(
                        "No choices in response"
                    )

                content = (
                    data["choices"][0]
                    ["message"]
                    ["content"]
                )

                image_found=False

                if isinstance(content,list):

                    for item in content:

                        # image URL response

                        if item.get("type")=="image_url":

                            image_url=(
                                item["image_url"]["url"]
                            )

                            img=requests.get(
                                image_url
                            )

                            image=Image.open(
                                BytesIO(
                                    img.content
                                )
                            )

                            st.image(
                                image,
                                use_container_width=True
                            )

                            image_found=True
                            break

                        # base64 response

                        elif item.get(
                            "type"
                        )=="image":

                            image_bytes=(
                                base64.b64decode(
                                    item["image"]
                                )
                            )

                            image=Image.open(
                                BytesIO(
                                    image_bytes
                                )
                            )

                            st.image(
                                image,
                                use_container_width=True
                            )

                            image_found=True
                            break

                if not image_found:

                    raise Exception(
                        "No image returned"
                    )

            # ======================================
            # SDXL FALLBACK
            # ======================================

            else:

                client = InferenceClient(
                    model="stabilityai/stable-diffusion-xl-base-1.0",
                    token=HF_TOKEN
                )

                image = client.text_to_image(
                    prompt,
                    guidance_scale=14,
                    num_inference_steps=40
                )

                st.image(
                    image,
                    use_container_width=True
                )

            st.success(
                "Generation complete"
            )

        except Exception as e:

            st.error(
                f"Error:\n{str(e)}"
            )
