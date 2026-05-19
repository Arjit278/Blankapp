
import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(
    page_title="Pictator Pro 2026 - Tombstone Seats",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Pictator Pro 2026")
st.subheader("Fixed Headrest Tombstone Seat Generator")

HF_TOKEN = st.secrets["HF_TOKEN"]

MODELS = {
    "FLUX Schnell (Fast)": "black-forest-labs/FLUX.1-schnell",
    "FLUX Dev": "black-forest-labs/FLUX.1-dev"
}

VEHICLES = [
    "Maruti Wagon R",
    "Grand Vitara",
    "Swift",
    "Baleno",
    "Universal"
]

vehicle = st.selectbox("Vehicle", VEHICLES)
model_name = st.selectbox("Model", list(MODELS.keys()))
material = st.selectbox(
    "Seat Material",
    ["Leather", "Fabric", "Premium Leather", "Diamond Stitch"]
)

generate = st.button("Generate")

if generate:

    prompt = f"""
    Generate a professional automotive studio product image.
    
    ONLY Tombstone seats.
    ONLY integrated fixed headrest seats.
    
    Headrest MUST be permanently merged into the seat body.
    Headrest and backrest are one continuous structure.
    
    STRICTLY FORBIDDEN:
    - removable headrests
    - detachable headrests
    - metal support rods
    - headrest posts
    - adjustable headrests
    - separate headrest cushion
    - rear seats
    - dashboard
    - steering wheel
    - complete car
    - seat belts
    - center console
    - armrest
    - extra objects
    - collage
    
    Vehicle: {vehicle}
    Material: {material}
    
    Show EXACTLY two front seats:
    driver and co-driver.
    
    Seats must:
    - be identical
    - full height visible
    - upright
    - integrated tombstone shape
    - fixed one-piece design
    - white studio background
    - catalog product photography
    
    Negative prompt:
    detachable headrest, headrest rods, adjustable headrest,
metal poles, separate cushion, removable top
    """

    with st.spinner("Generating..."):
        client = InferenceClient(
            model=MODELS[model_name],
            token=HF_TOKEN
        )

        image = client.text_to_image(prompt)

        st.image(image, use_container_width=True)
        st.success("Done")
