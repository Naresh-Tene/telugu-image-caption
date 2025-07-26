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
        try:
            error_details = response.json()
            if "error" in error_details and "is currently loading" in error_details.get("error", ""):
                 estimated_time = error_details.get('estimated_time', 20)
                 return {"error": f"మోడల్ లోడ్ అవుతోంది (Model is loading). దయచేసి {int(estimated_time)} సెకన్లు వేచి ఉండి, చిత్రాన్ని మళ్లీ అప్‌లోడ్ చేయండి. (Please wait {int(estimated_time)} seconds and upload the image again.)"}
            return {"error": f"API Error: {error_details.get('error', 'Unknown error')}", "status_code": response.status_code}
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
    st.stop()

# --- Main App Logic ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Create two columns for a better layout
    col1, col2 = st.columns(2)

    image = Image.open(uploaded_file)
    with io.BytesIO() as output:
        image.save(output, format="JPEG")
        image_bytes = output.getvalue()

    # Display the image in the first column
    with col1:
        st.image(image, caption="మీరు అప్‌లోడ్ చేసిన చిత్రం (Your Uploaded Image)")

    # Display the results in the second column
    with col2:
        with st.spinner("ఆంగ్ల వివరణను రూపొందిస్తోంది... (Generating English description...)"):
            # Using the best model for image captioning
            caption_api_url = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
            caption_result = query_api(hf_api_key, caption_api_url, image_bytes)

            if "error" in caption_result:
                st.error(caption_result['error'])
            else:
                english_caption = caption_result[0]['generated_text']
                st.info(f"**English Caption:**\n\n{english_caption}")

                with st.spinner("తెలుగులోకి అనువదిస్తోంది... (Translating to Telugu...)"):
                    translation_api_url = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-te"
                    translation_payload = {"inputs": english_caption}
                    translation_result = query_api(hf_api_key, translation_api_url, translation_payload)

                    if "error" in translation_result:
                        st.error(translation_result['error'])
                    else:
                        telugu_caption = translation_result[0]['translation_text']
                        st.success(f"**తెలుగు వివరణ (Telugu Description):**\n\n{telugu_caption}")

