import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM

# --- Function to load the captioning model (runs only once) ---
@st.cache_resource
def load_captioning_model():
    """Loads the BLIP image captioning model and processor."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model

# --- Function to load the translation model (runs only once) ---
@st.cache_resource
def load_translation_model():
    """Loads the English to Telugu translation model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-te")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-te")
    return tokenizer, model

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="Telugu Image Describer")
st.title("🖼️ చిత్ర వివరణ జనరేటర్ (Image Description Generator)")
st.write("చిత్రాన్ని అప్‌లోడ్ చేయండి మరియు నేను దానిని తెలుగులో వివరిస్తాను. (Upload an image and I will describe it in Telugu.)")


# --- Load the models ---
with st.spinner("మొదటిసారి మోడల్‌లను లోడ్ చేస్తోంది... దీనికి కొన్ని నిమిషాలు పట్టవచ్చు. (Loading models for the first time... this may take a few minutes.)"):
    caption_processor, caption_model = load_captioning_model()
    translation_tokenizer, translation_model = load_translation_model()

# --- Main App Logic ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(image, caption="మీరు అప్‌లోడ్ చేసిన చిత్రం (Your Uploaded Image)")

    with col2:
        with st.spinner("ఆంగ్ల వివరణను రూపొందిస్తోంది... (Generating English description...)"):
            # Process the image and generate caption locally
            inputs = caption_processor(images=image, return_tensors="pt")
            pixel_values = inputs.pixel_values
            output_ids = caption_model.generate(pixel_values, max_length=50, num_beams=4)
            english_caption = caption_processor.decode(output_ids[0], skip_special_tokens=True)
            st.info(f"**English Caption:**\n\n{english_caption}")

            with st.spinner("తెలుగులోకి అనువదిస్తోంది... (Translating to Telugu...)"):
                # Translate the caption to Telugu locally
                inputs = translation_tokenizer(english_caption, return_tensors="pt")
                translated_ids = translation_model.generate(**inputs, max_length=60)
                telugu_caption = translation_tokenizer.batch_decode(translated_ids, skip_special_tokens=True)[0]
                st.success(f"**తెలుగు వివరణ (Telugu Description):**\n\n{telugu_caption}")

