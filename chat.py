import streamlit as st
import requests
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# --- Function to load the captioning model (runs only once) ---
@st.cache_resource
def load_captioning_model():
    """Loads the BLIP model and processor from Hugging Face."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model

# --- Function to call the Translation API ---
def translate_text(api_key, text_to_translate):
    """Calls the Hugging Face API to translate English text to Telugu."""
    api_url = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-te"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": text_to_translate}
    
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Translation failed."}

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="Telugu Image Describer")
st.title("🖼️ చిత్ర వివరణ జనరేటర్ (Image Description Generator)")
st.write("చిత్రాన్ని అప్‌లోడ్ చేయండి మరియు నేను దానిని తెలుగులో వివరిస్తాను. (Upload an image and I will describe it in Telugu.)")

# --- Get API Key from Streamlit Secrets (still needed for translation) ---
try:
    hf_api_key = st.secrets["HF_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("Hugging Face API key not found in Streamlit secrets.")
    st.stop()

# --- Load the main model ---
with st.spinner("మొదటిసారి మోడల్‌ను లోడ్ చేస్తోంది... దీనికి కొన్ని నిమిషాలు పట్టవచ్చు. (Loading model for the first time... this may take a few minutes.)"):
    processor, model = load_captioning_model()

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
            inputs = processor(images=image, return_tensors="pt")
            pixel_values = inputs.pixel_values
            output_ids = model.generate(pixel_values, max_length=50, num_beams=4)
            english_caption = processor.decode(output_ids[0], skip_special_tokens=True)
            st.info(f"**English Caption:**\n\n{english_caption}")

            with st.spinner("తెలుగులోకి అనువదిస్తోంది... (Translating to Telugu...)"):
                translation_result = translate_text(hf_api_key, english_caption)
                if "error" in translation_result:
                    st.error(translation_result['error'])
                else:
                    telugu_caption = translation_result[0]['translation_text']
                    st.success(f"**తెలుగు వివరణ (Telugu Description):**\n\n{telugu_caption}")
