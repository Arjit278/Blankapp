
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
    Generate a clean studio product photo.

    ONLY Tombstone car seats.
    ONLY fixed headrests.
    Headrests must be integrated into seat body.
    No removable headrests.
    No detachable headrests.
    No rear seats.
    No dashboard.
    No steering wheel.
    No complete car.
    No collage.
    No extra objects.

    Vehicle: {vehicle}
    Material: {material}

    Show exactly TWO front tombstone seats:
    driver seat and co-driver seat side-by-side.
    Both seats identical.
    Full seat visible from top to bottom.
    White studio background.
    Professional automotive catalog style.
    """

    with st.spinner("Generating..."):
        client = InferenceClient(
            model=MODELS[model_name],
            token=HF_TOKEN
        )

        image = client.text_to_image(prompt)

        st.image(image, use_container_width=True)
        st.success("Done")
