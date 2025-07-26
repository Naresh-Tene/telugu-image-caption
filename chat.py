import streamlit as st
import requests
from PIL import Image
import io

# --- Hugging Face API Functions ---

def query_image_captioning(api_key, image_bytes):
    """
    Calls the Hugging Face Inference API to generate an English caption for an image.
    """
    api_url = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    response = requests.post(api_url, headers=headers, data=image_bytes)
    
    if response.status_code == 200:
        return response.json()
    else:
        try:
            error_details = response.json()
            return {"error": f"API Error: {error_details.get('error', 'Unknown error')}", "status_code": response.status_code}
        except requests.exceptions.JSONDecodeError:
            return {"error": "API Error: Non-JSON response.", "status_code": response.status_code}


def query_translation(api_key, text_to_translate):
    """
    Calls the Hugging Face Inference API to translate English text to Telugu.
    """
    api_url = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-te"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": text_to_translate}
    
    response = requests.post(api_url, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()
    else:
        try:
            error_details = response.json()
            return {"error": f"Translation API Error: {error_details.get('error', 'Unknown error')}", "status_code": response.status_code}
        except requests.exceptions.JSONDecodeError:
            return {"error": "Translation API Error: Non-JSON response.", "status_code": response.status_code}


# --- Streamlit App UI ---

st.set_page_config(layout="wide", page_title="Telugu Image Describer")

st.title("🖼️ చిత్ర వివరణ జనరేటర్ (Image Description Generator)")
st.write("చిత్రాన్ని అప్‌లోడ్ చేయండి మరియు నేను దానిని తెలుగులో వివరిస్తాను. (Upload an image and I will describe it in Telugu.)")

# --- Get API Key from Streamlit Secrets ---
# We no longer need the sidebar. We get the key directly from secrets.
try:
    hf_api_key = st.secrets["HF_API_KEY"]
except KeyError:
    st.error("Hugging Face API key not found. Please add it to your Streamlit secrets.")
    st.stop() # Stop the app if the key is not available


# --- Main App Logic ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    
    # Convert image to bytes
    with io.BytesIO() as output:
        image.save(output, format="JPEG")
        image_bytes = output.getvalue()

    st.image(image, caption="మీరు అప్‌లోడ్ చేసిన చిత్రం (Your Uploaded Image)", use_column_width=True, width=300)

    with st.spinner("ఆంగ్ల వివరణను రూపొందిస్తోంది... (Generating English description...)"):
        # 1. Get English caption
        caption_result = query_image_captioning(hf_api_key, image_bytes)

        if "error" in caption_result:
            st.error(f"Could not generate caption. {caption_result['error']}")
        else:
            english_caption = caption_result[0]['generated_text']
            st.info(f"**English Caption:** {english_caption}")

            with st.spinner("తెలుగులోకి అనువదిస్తోంది... (Translating to Telugu...)"):
                # 2. Translate the caption to Telugu
                translation_result = query_translation(hf_api_key, english_caption)

                if "error" in translation_result:
                    st.error(f"Could not translate text. {translation_result['error']}")
                else:
                    telugu_caption = translation_result[0]['translation_text']
                    st.success(f"**తెలుగు వివరణ (Telugu Description):** {telugu_caption}")

