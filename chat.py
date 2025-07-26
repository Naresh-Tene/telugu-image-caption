import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# --- Function to load the captioning model (runs only once) ---
@st.cache_resource
def load_captioning_model():
    """Loads the BLIP image captioning model and processor."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="Image Description Generator")
st.title("🖼️ Image Description Generator")
st.write("Upload an image and this app will generate a description for it.")


# --- Load the model ---
with st.spinner("Loading the captioning model... this may take a minute on first startup."):
    caption_processor, caption_model = load_captioning_model()

# --- Main App Logic ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(image, caption="Uploaded Image")

    with col2:
        with st.spinner("Generating description..."):
            # Process the image and generate caption locally
            inputs = caption_processor(images=image, return_tensors="pt")
            pixel_values = inputs.pixel_values
            output_ids = caption_model.generate(pixel_values, max_length=50, num_beams=4)
            english_caption = caption_processor.decode(output_ids[0], skip_special_tokens=True)
            st.success(f"**Generated Description:**\n\n{english_caption}")

