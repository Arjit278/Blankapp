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

    st.error(
        "Missing API keys in Streamlit secrets"
    )

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
# PROMPT
# ====================================================

prompt = f"""
Automotive studio catalog image.

Generate EXACTLY TWO FRONT SEATS.

ONLY Tombstone seats.
ONLY integrated fixed headrests.

Headrest must be merged into backrest.

Vehicle: {vehicle}
Material: {material}
Color: {color}

Front view
Identical seats
White studio background

Do NOT generate:

metal rods
headrest poles
adjustable headrests
removable headrests
rear seats
dashboard
steering wheel
full car
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

            # ========================================
            # NANO BANANA
            # ========================================

            if generator=="Google Nano Banana":

                headers = {

                    "Authorization":
                    f"Bearer {OPENROUTER_API_KEY}",

                    "Content-Type":
                    "application/json"

                }

                payload = {

                    "model":
                    "google/gemini-2.5-flash-image",

                    "max_tokens":
                    256,

                    "temperature":
                    0.2,

                    "messages":[
                        {
                            "role":"user",
                            "content":prompt
                        }
                    ]
                }

                response=requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120
                )

                data=response.json()

                with st.expander(
                    "Raw API Response"
                ):
                    st.json(data)

                if "error" in data:

                    st.error(
                        data["error"]["message"]
                    )

                    st.stop()

                if "choices" not in data:

                    raise Exception(
                        f"Unexpected response:\n{data}"
                    )

                message=(
                    data["choices"][0]
                    ["message"]
                )

                image_found=False

                # ===============================
                # images[]
                # ===============================

                if "images" in message:

                    for img in message["images"]:

                        if "image_url" in img:

                            url=img["image_url"]

                            img_response=requests.get(
                                url
                            )

                            image=Image.open(
                                BytesIO(
                                    img_response.content
                                )
                            )

                            st.image(
                                image,
                                use_container_width=True
                            )

                            image_found=True
                            break

                # ===============================
                # content[]
                # ===============================

                if (
                    not image_found
                    and
                    "content" in message
                ):

                    content=message[
                        "content"
                    ]

                    if isinstance(
                        content,
                        list
                    ):

                        for item in content:

                            # URL image

                            if item.get(
                                "type"
                            )=="image_url":

                                url=(
                                    item[
                                        "image_url"
                                    ]["url"]
                                )

                                img_response=(
                                    requests.get(
                                        url
                                    )
                                )

                                image=Image.open(
                                    BytesIO(
                                        img_response.content
                                    )
                                )

                                st.image(
                                    image,
                                    use_container_width=True
                                )

                                image_found=True
                                break

                            # Base64 image

                            elif item.get(
                                "type"
                            )=="image":

                                image_bytes=(
                                    base64.b64decode(
                                        item[
                                            "image"
                                        ]
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

                # ===============================
                # TEXT ONLY
                # ===============================

                if not image_found:

                    st.warning(
                        "Model returned text instead of image"
                    )

                    st.write(message)

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
