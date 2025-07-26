import streamlit as st
import requests
from PIL import Image
import io

# --- Hugging Face API Functions ---

def query_api(api_key, api_url, data):
    """
    Generic function to query the Hugging Face Inference API.
    Handles loading states and provides detailed errors.
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    
    if isinstance(data, dict): # For JSON payloads like translation
        response = requests.post(api_url, headers=headers, json=data)
    else: # For binary data like images
        response = requests.post(api_url, headers=headers, data=data)

    if response.status_code == 200:
        return response.json()
    else:
        # Try to parse JSON, as most HF errors are JSON
        try:
            error_details = response.json()
            # Check for the specific "model is loading" error
            if "error" in error_details and "is currently loading" in error_details.get("error", ""):
                 estimated_time = error_details.get('estimated_time', 20)
                 return {"error": f"మోడల్ లోడ్ అవుతోంది (Model is loading). దయచేసి {int(estimated_time)} సెకన్లు వేచి ఉండి, చిత్రాన్ని మళ్లీ అప్‌లోడ్ చేయండి. (Please wait {int(estimated_time)} seconds and upload the image again.)"}
            
            # For other JSON errors
            return {"error": f"API Error: {error_details.get('error', 'Unknown error')}", "status_code": response.status_code}
        
        # If the response is not JSON, return the raw text for better debugging
        except requests.exceptions.JSONDecodeError:
            return {"error": f"API Error: Non-JSON response from server. Status Code: {response.status_code}. Server says: {response.text}"}


# --- Streamlit App UI ---

st.set_page_config(layout="wide", page_title="Telugu Image Describer")

st.title("🖼️ చిత్ర వివరణ జనరేటర్ (Image Description Generator)")
st.write("చిత్రాన్ని అప్‌లోడ్ చేయండి మరియు నేను దానిని తెలుగులో వివరిస్తాను. (Upload an image and I will describe it in Telugu.)")

# --- Get API Key from Streamlit Secrets ---
try:
    hf_api_key = st.secrets["HF_API_KEY"]
except (KeyError, FileNotFoundError):
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

    # Fixed the deprecated parameter here
    st.image(image, caption="మీరు అప్‌లోడ్ చేసిన చిత్రం (Your Uploaded Image)", use_container_width=True)

    with st.spinner("ఆంగ్ల వివరణను రూపొందిస్తోంది... (Generating English description...)"):
        # 1. Get English caption
        # --- THIS IS THE LINE THAT HAS BEEN CHANGED TO A MORE RELIABLE MODEL ---
        caption_api_url = "https://api-inference.huggingface.co/models/nlpconnect/vit-gpt2-image-captioning"
        caption_result = query_api(hf_api_key, caption_api_url, image_bytes)

        if "error" in caption_result:
            st.error(caption_result['error'])
        else:
            english_caption = caption_result[0]['generated_text']
            st.info(f"**English Caption:** {english_caption}")

            with st.spinner("తెలుగులోకి అనువదిస్తోంది... (Translating to Telugu...)"):
                # 2. Translate the caption to Telugu
                translation_api_url = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-te"
                translation_payload = {"inputs": english_caption}
                translation_result = query_api(hf_api_key, translation_api_url, translation_payload)

                if "error" in translation_result:
                    st.error(translation_result['error'])
                else:
                    telugu_caption = translation_result[0]['translation_text']
                    st.success(f"**తెలుగు వివరణ (Telugu Description):** {telugu_caption}")
