import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from google import genai
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2

# --- 0. INITIALIZATION ---
st.set_page_config(page_title="PlantHealth AI", page_icon="🌿", layout="wide")

# Initialize Gemini Client
CLIENT = None
try:
    GEMINI_API_KEY = st.secrets["gemini_api_key"]
    CLIENT = genai.Client(api_key=GEMINI_API_KEY)
except Exception:
    st.sidebar.warning("⚠️ Chatbot Offline: Check API key in secrets.")

# --- 1. CONFIGURATION ---
NUM_CLASSES = 10
IMAGE_SIZE = 224
WEIGHTS_PATH = 'models.h5'

FULL_CLASS_NAMES = [
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# --- 2. KNOWLEDGE BASE ---
KNOWLEDGE_BASE = {
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {"name": "Corn Gray Leaf Spot",
                                                           "diagnosis": "Rectangular brown lesions parallel to veins.",
                                                           "treatment": "Foliar fungicides and crop rotation."},
    "Corn_(maize)___Common_rust_": {"name": "Corn Common Rust", "diagnosis": "Cinnamon-brown pustules on leaves.",
                                    "treatment": "Plant resistant hybrids."},
    "Corn_(maize)___Northern_Leaf_Blight": {"name": "Corn Northern Leaf Blight",
                                            "diagnosis": "Large cigar-shaped tan lesions.",
                                            "treatment": "Manage residue/resistant varieties."},
    "Corn_(maize)___healthy": {"name": "Healthy Corn", "diagnosis": "Vibrant leaf with no lesions.",
                               "treatment": "None required. Keep monitoring."},
    "Tomato___Bacterial_spot": {"name": "Tomato Bacterial Spot", "diagnosis": "Small dark greasy spots.",
                                "treatment": "Copper-based sprays."},
    "Tomato___Early_blight": {"name": "Tomato Early Blight", "diagnosis": "Concentric target rings.",
                              "treatment": "Prune for airflow/fungicides."},
    "Tomato___Late_blight": {"name": "Tomato Late Blight", "diagnosis": "Greenish-black patches.",
                             "treatment": "Remove plants immediately."},
    "Tomato___Leaf_Mold": {"name": "Tomato Leaf Mold", "diagnosis": "Olive-green mold underneath.",
                           "treatment": "Improve ventilation."},
    "Tomato___Tomato_mosaic_virus": {"name": "Tomato Mosaic Virus", "diagnosis": "Mottled yellow patterns.",
                                     "treatment": "Sanitize and remove plants."},
    "Tomato___healthy": {"name": "Healthy Tomato", "diagnosis": "Clean, green leaves.",
                         "treatment": "Regular nutrient monitoring."}
}


# --- 3. MODEL LOADING ---
@st.cache_resource
def load_disease_model():
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.load_weights(WEIGHTS_PATH)
    return model


MODEL = load_disease_model()


def predict_disease(image_file):
    img = Image.open(image_file).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = preprocess_input(np.expand_dims(tf.keras.preprocessing.image.img_to_array(img), axis=0))
    predictions = MODEL.predict(img_array)
    return FULL_CLASS_NAMES[np.argmax(predictions)], np.max(predictions)


# --- 4. MAIN UI ---
def main():
    st.sidebar.title("🌿 PlantHealth Menu")
    app_mode = st.sidebar.selectbox("Navigate To:", ["Home", "Disease Recognition", "About Project"])

    # --- HOME PAGE ---
    if app_mode == "Home":
        st.title(" Smart Agriculture Diagnosis System")
        st.subheader("Detect. Diagnose. Protect.")

        # Main Banner - Local Path
        if os.path.exists("banner.jpg"):
            st.image("banner.jpg", caption="AI-Powered Field Analysis", use_container_width=True)
        else:
            st.info("Upload 'banner.jpg' to your project folder to see the main banner.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("  Corn (Maize) Support")
            if os.path.exists("corn_home.jpg"):
                st.image("corn_home.jpg", width=400)
            st.write("Specialized in detecting Common Rust, Northern Leaf Blight, and Gray Leaf Spot.")

        with col2:
            st.markdown("  Tomato Support")
            if os.path.exists("tomato_home.jpg"):
                st.image("tomato_home.jpg", width=400)
            st.write("Comprehensive detection for Blights, Molds, Bacterial Spots, and Viruses.")

        st.divider()
        st.markdown("""
        ### Features
        * **Dual Input:** Upload existing photos or use the **Live Camera Scan**.
        * **Expert Diagnosis:** Powered by a customized **MobileNetV2** Neural Network.
        * **AI Assistant:** Chat directly with our **Gemini-powered** agronomist for follow-up care.
        """)

    # --- DISEASE RECOGNITION PAGE ---
    elif app_mode == "Disease Recognition":
        st.header("🔬 Diagnosis Laboratory")

        with st.sidebar:
            st.header("Input Controls")
            selected_crop = st.selectbox("1. Target Crop", ["Select...", "Corn", "Tomato"])
            input_method = st.radio("2. Selection Method", ("Manual Upload", "Live Camera Scan"))

            final_image = None
            if input_method == "Manual Upload":
                final_image = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])
            else:
                final_image = st.camera_input("Scan your plant leaf")

        if final_image and selected_crop != "Select...":
            name, score = predict_disease(final_image)
            info = KNOWLEDGE_BASE.get(name)

            res_col1, res_col2 = st.columns([1, 1])
            with res_col1:
                st.image(final_image, caption="Analyzed Sample", use_container_width=True)
                st.metric("Detection Confidence", f"{score * 100:.1f}%")

            with res_col2:
                st.header(f"Diagnosis: {info['name']}")
                st.error(f"**Symptoms:** {info['diagnosis']}")
                st.success(f"**Treatment Plan:** {info['treatment']}")

            # --- CHATBOT SECTION ---
            st.divider()
            st.header("💬 Chat with AI Specialist")
            if CLIENT:
                if "msgs" not in st.session_state: st.session_state.msgs = []
                for m in st.session_state.msgs:
                    with st.chat_message(m["role"]): st.markdown(m["content"])

                if prompt := st.chat_input("Ask a question about this plant..."):
                    st.session_state.msgs.append({"role": "user", "content": prompt})
                    with st.chat_message("user"): st.markdown(prompt)

                    context = f"Context: Plant is {info['name']}. Diagnosis: {info['diagnosis']}. Treatment: {info['treatment']}. User question: {prompt}"
                    response = CLIENT.models.generate_content(model='gemini-2.0-flash', contents=context)

                    st.session_state.msgs.append({"role": "assistant", "content": response.text})
                    with st.chat_message("assistant"): st.markdown(response.text)

    # --- ABOUT PAGE ---
    elif app_mode == "About Project":
        st.title("Technical Overview")
        st.markdown("""
        This system combines Computer Vision and Generative AI. 
        - **Classification:** MobileNetV2 Architecture.
        - **Reasoning:** Google Gemini 2.0 Flash.
        - **Dataset:** 3,900 Images (10 Classes).
        """)


if __name__ == '__main__':
    main()