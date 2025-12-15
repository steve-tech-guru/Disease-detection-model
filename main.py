import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from google import genai
import time

# Import necessary Keras layers for model definition
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2

## =============================================================================
## 0. INITIALIZATION & SETUP
## =============================================================================

# 1. Initialize Gemini Client
CLIENT = None
try:
    # Read the API key securely from .streamlit/secrets.toml
    GEMINI_API_KEY = st.secrets["gemini_api_key"]
    CLIENT = genai.Client(api_key=GEMINI_API_KEY)
except Exception:
    st.sidebar.error("Error: Gemini API Key not found. Please set 'gemini_api_key' in .streamlit/secrets.toml")

# --- NEW MODEL CONFIGURATION ---
WEIGHTS_PATH = 'model_weights_epoch_1_partial.h5'  # <-- CHECK THIS FILENAME
NUM_CLASSES = 38  # <-- CHECK THIS NUMBER
IMAGE_SIZE = 224  # <-- CHECK THIS SIZE

CONFIDENCE_THRESHOLD = 0.85

## =============================================================================
## 1. KNOWLEDGE BASE & CLASS NAMES (Using the previous knowledge base)
## =============================================================================

# --- KNOWLEDGE BASE (Detailed info for your focused crops) ---
KNOWLEDGE_BASE = {
    # Corn (Maize)
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "name": "Corn Gray Leaf Spot (GLS)",
        "diagnosis": "Lesions are typically narrow, rectangular, and pale brown to gray, running parallel to the leaf veins. Severe infection can lead to extensive leaf death.",
        "treatment": "Use resistant corn hybrids. Apply fungicide when symptoms first appear, especially before silking. Tillage can help bury infected residue.",
        "chatbot_context_prompt": "You are an expert agricultural extension agent specializing in corn diseases. Your advice must be based ONLY on the provided diagnosis and treatment information for Corn Gray Leaf Spot. Do not introduce outside information."
    },
    "Corn_(maize)___healthy": {
        "name": "Healthy Corn Plant",
        "diagnosis": "The plant is healthy with uniform green color and no visible lesions, rust, or spots. Maintain regular care.",
        "treatment": "Maintain regular watering and fertilization, and monitor the plant for any signs of emerging stress.",
        "chatbot_context_prompt": "You are an expert agricultural extension agent specializing in crop health. You should answer questions about general corn care, focusing on maintaining plant health. Do not introduce outside information about diseases."
    },
    # Tomato
    "Tomato___Bacterial_spot": {
        "name": "Tomato Bacterial Spot",
        "diagnosis": "Caused by bacteria, symptoms include small, dark, circular spots on leaves and fruits. On leaves, the spots are often surrounded by a yellow halo.",
        "treatment": "Avoid overhead watering. Apply copper-based bactericides. Use certified disease-free seeds and seedlings.",
        "chatbot_context_prompt": "You are an expert agricultural extension agent specializing in tomato diseases. Your advice must be based ONLY on the provided diagnosis and treatment information for Tomato Bacterial Spot. Do not introduce outside information."
    },
    "Tomato___healthy": {
        "name": "Healthy Tomato Plant",
        "diagnosis": "The plant is healthy with uniform green color and no visible lesions, rust, or spots. Maintain regular care.",
        "treatment": "Maintain regular watering and fertilization, and monitor the plant for any signs of emerging stress.",
        "chatbot_context_prompt": "You are an expert agricultural extension agent specializing in crop health. You should answer questions about general tomato care, focusing on maintaining plant health. Do not introduce outside information about diseases."
    },
    # Pepper (Bell)
    "Pepper,_bell___Bacterial_spot": {
        "name": "Bell Pepper Bacterial Spot",
        "diagnosis": "Causes small, dark, water-soaked spots on leaves and raised, scabby spots on fruit. Severe infection can cause leaves to drop prematurely.",
        "treatment": "Use certified pathogen-free seeds. Apply copper-containing products. Avoid working in the field when foliage is wet.",
        "chatbot_context_prompt": "You are an expert agricultural extension agent specializing in pepper diseases. Your advice must be based ONLY on the provided diagnosis and treatment information for Bell Pepper Bacterial Spot. Do not introduce outside information."
    },
    "Pepper,_bell___healthy": {
        "name": "Healthy Bell Pepper Plant",
        "diagnosis": "The plant is healthy with glossy green leaves and no signs of disease. Maintain standard care.",
        "treatment": "Maintain regular watering and fertilization, and monitor the plant for any signs of emerging stress.",
        "chatbot_context_prompt": "You are an expert agricultural extension agent specializing in crop health. You should answer questions about general pepper care, focusing on maintaining plant health. Do not introduce outside information about diseases."
    },
    # Potato
    "Potato___Late_blight": {
        "name": "Potato Late Blight",
        "diagnosis": "Destructive disease causing water-soaked spots that enlarge into brown/black lesions, often with a white, fuzzy mold visible on the underside in humid conditions.",
        "treatment": "Use protectant or systemic fungicides. Ensure good air circulation and destroy infected plant debris.",
        "chatbot_context_prompt": "You are an expert agricultural extension agent specializing in potato diseases. Your advice must be based ONLY on the provided diagnosis and treatment information for Potato Late Blight. Do not introduce outside information."
    },
    "Potato___healthy": {
        "name": "Healthy Potato Plant",
        "diagnosis": "The plant is healthy with vibrant green foliage. No blight or mold is detected.",
        "treatment": "Use healthy, certified seed potatoes and ensure adequate soil drainage.",
        "chatbot_context_prompt": "You are an expert agricultural extension agent specializing in crop health. You should answer questions about general potato care, focusing on maintaining plant health. Do not introduce outside information about diseases."
    },
    # Low Confidence Entry (Error Handling)
    "LOW_CONFIDENCE_ENTRY": {
        "name": "Inconclusive Diagnosis",
        "diagnosis": "The prediction confidence was below the required threshold (85%). The model could not reliably identify the disease. The image may be too blurry, too dark, or the disease is not in the training set.",
        "treatment": "Please try uploading a clearer, high-resolution image of a single leaf, or consult a local agricultural expert.",
        "chatbot_context_prompt": "You are a helpful assistant. The model was unable to provide a reliable diagnosis. Your role is to politely explain why the diagnosis was inconclusive and suggest steps to get a better result, such as providing a clearer image.",
    },
    # Default for any other disease predicted
    "DEFAULT_DISEASE_ENTRY": {
        "name": "General Plant Disease/Other Crop",
        "diagnosis": "A disease was detected, but detailed information for this specific crop or ailment is not yet available in the app's knowledge base. Please consult a local expert.",
        "treatment": "Please consult a local agricultural extension office or expert for tailored treatment advice.",
        "chatbot_context_prompt": "You are a helpful assistant. The user's diagnosed condition is outside the scope of your detailed knowledge base. Politely inform the user that detailed diagnosis and treatment are not yet available for this prediction."
    }
}

# --- FULL CLASS NAMES (!!! CRITICAL: REPLACE THIS LIST with your Colab output !!!) ---
FULL_CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___healthy',
]


## =============================================================================
## 2. MODEL LOADING & CORE FUNCTIONS (UPDATED FOR WEIGHTS-ONLY)
## =============================================================================

@st.cache_resource
def load_disease_model(weights_path):
    """
    Loads model architecture (MobileNetV2) and then loads weights from the partial file.
    """
    if not os.path.exists(weights_path):
        st.error(f"Error: Weights file not found at '{weights_path}'. Please check the file name and location.")
        return None

    try:
        # 1. Define the exact architecture again (must match your training code!)
        base_model = MobileNetV2(
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
            include_top=False,
            weights='imagenet',  # Use imagenet weights for the frozen layers
            pooling=None
        )
        base_model.trainable = False

        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(NUM_CLASSES, activation='softmax')
        ])

        # 2. Load the trained weights from your file
        model.load_weights(weights_path)

        # 3. Compile the model (required before prediction in some environments)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    except Exception as e:
        st.error(f"Error loading model weights. Did you use the correct NUM_CLASSES and IMAGE_SIZE? Error: {e}")
        return None


# Load the model using the new function
MODEL = load_disease_model(WEIGHTS_PATH)


def predict_disease(image_file, model=MODEL, class_names=FULL_CLASS_NAMES):
    """Predicts the disease class and confidence score."""
    if model is None:
        return "Model_Error", 0.0

    img = Image.open(image_file).convert("RGB")
    # CRITICAL: Use the IMAGE_SIZE defined in the config
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    predictions = model.predict(img_array)

    predicted_class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    predicted_class_name = class_names[predicted_class_index]

    return predicted_class_name, confidence


def get_disease_info(predicted_class_name, knowledge_base=KNOWLEDGE_BASE):
    """Looks up the diagnosis, treatment, and context from the knowledge base."""
    if predicted_class_name in knowledge_base:
        return knowledge_base[predicted_class_name]
    else:
        return knowledge_base["DEFAULT_DISEASE_ENTRY"]


def build_context_prompt(diagnosis_info, user_question):
    """Builds the context-aware prompt for the LLM with strict grounding."""
    # Modified prompt for strict grounding (as discussed)
    system_prompt = diagnosis_info.get("chatbot_context_prompt", "You are a general plant health expert.")
    diagnosis = diagnosis_info.get("diagnosis", "No specific diagnosis available.")
    treatment = diagnosis_info.get("treatment", "No specific treatment available.")

    full_prompt = f"""
    {system_prompt} 

    You are a highly specialized agricultural extension agent. **Your primary goal is safety and accuracy.** You MUST strictly adhere to the 'PROVIDED KNOWLEDGE' section below. 
    DO NOT invent, guess, or introduce any information, chemicals, or treatments not explicitly present in the provided context. 
    If the user asks a question that cannot be fully answered using the Diagnosis or Treatment provided, politely state that the information is outside your current knowledge scope for this case and recommend consulting a local expert.

    --- PROVIDED KNOWLEDGE ---
    Diagnosis: {diagnosis}
    Treatment: {treatment}
    ---

    The user is asking a question related to this diagnosis: "{user_question}"
    """
    return full_prompt


## =============================================================================
## 3. STREAMLIT APPLICATION (MAIN)
## =============================================================================

def main():
    st.set_page_config(page_title="Plant Disease Diagnosis", layout="wide")
    st.title(" AI Plant Disease Diagnosis Assistant")
    st.markdown("---")

    # Apply custom styling (Dark Mode preference assumed)
    st.markdown(
        """
        <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # --- Sidebar Setup (Crop Selector and Uploader/Camera) ---
    uploaded_file = None
    with st.sidebar:
        st.header("1. Select Your Crop")

        CROP_OPTIONS = ["Select Crop...", "Corn (Maize)", "Tomato", "Pepper (Bell)", "Potato"]
        selected_crop = st.selectbox("What plant is this leaf from?", CROP_OPTIONS)

        st.markdown("---")
        st.header("2. Provide Leaf Image")

        # --- CAMERA INPUT / UPLOADER ---
        capture_option = st.radio(
            "How would you like to provide the image?",
            ("Upload a File", "Take a Picture (Scan)")
        )

        if capture_option == "Upload a File":
            uploaded_file = st.file_uploader(
                "Choose a leaf image file:",
                type=["jpg", "jpeg", "png"]
            )
        elif capture_option == "Take a Picture (Scan)":
            uploaded_file = st.camera_input("Point your camera at the leaf:")

        st.info("The model will focus its diagnosis based on your crop selection.")

    # --- Main Content Area: Flow Control ---

    if MODEL is None:
        st.error(
            f"Cannot proceed without a loaded model. Check the weights file name ('{WEIGHTS_PATH}') and ensure NUM_CLASSES is correct.")

    elif uploaded_file is None or selected_crop == "Select Crop...":
        st.warning("☝️ Please select a **Crop** and provide an image to begin diagnosis.")

    else:
        # --- Main execution block (Prediction and Filtering) ---

        # Define the mapping to check the model's prediction prefix
        CROP_PREFIX_MAP = {
            "Corn (Maize)": "Corn_(maize)",
            "Tomato": "Tomato",
            "Pepper (Bell)": "Pepper,_bell",
            "Potato": "Potato",
        }

        # --- UI Improvement: Phased Progress Bar ---
        status_container = st.empty()
        status_bar = status_container.progress(0, text="1. Preprocessing Image...")

        # 1. Run the model prediction
        status_bar.progress(33, text="2. Running MobileNetV2 Model (Loading Weights)...")
        predicted_name, confidence_score = predict_disease(uploaded_file)

        status_bar.progress(66, text="3. Consulting Gemini Knowledge Base...")

        # --- Filtering Logic (Crop Bias Fix) ---
        final_predicted_name = "DEFAULT_DISEASE_ENTRY"
        target_prefix = CROP_PREFIX_MAP.get(selected_crop)

        if target_prefix and target_prefix in predicted_name:
            # Check confidence only if crop matches
            if confidence_score >= CONFIDENCE_THRESHOLD:
                final_predicted_name = predicted_name
            else:
                # Crop matches but confidence is low
                final_predicted_name = "LOW_CONFIDENCE_ENTRY"
        else:
            # If the model predicts the WRONG CROP, default to the "healthy" state of the selected crop.
            final_predicted_name = f"{target_prefix}___healthy"

        # Complete the progress bar
        status_bar.progress(100, text="4. Diagnosis Complete.")
        time.sleep(0.5)
        status_container.empty()

        # --- Display Results ---
        col1, col2 = st.columns([1, 2])
        info = get_disease_info(final_predicted_name)

        with col1:
            st.subheader("Uploaded Sample")
            st.image(uploaded_file, caption=selected_crop, width=250)
            st.text(f"Raw Model Score: {confidence_score * 100:.2f}%")

        with col2:
            st.subheader("Diagnosis Result")

            if final_predicted_name == "DEFAULT_DISEASE_ENTRY" or final_predicted_name == "LOW_CONFIDENCE_ENTRY":
                st.error(f"**Status:** {info['name']}")
            elif info['name'].endswith("healthy"):
                st.success(f"**Status:** {info['name']}")
            else:
                st.warning(f"**Status:** {info['name']}")

            st.markdown("---")

            # --- UI Improvement: Structured Expanders for Results ---
            with st.expander(f"🔬 Diagnosis: {info['name']}", expanded=True):
                st.markdown(f"**Diagnosis Details:** {info['diagnosis']}")
                st.caption("Diagnosis based on custom MobileNetV2 analysis.")

            with st.expander("🩺 Recommended Treatment"):
                st.markdown(f"**Treatment Protocol:** {info['treatment']}")
                st.caption("Always consult a local expert for specific chemical advice.")

            # Store results in session state for the chat interface
            st.session_state['diagnosis_info'] = info
            st.session_state['predicted_name'] = final_predicted_name
            st.session_state['image_uploaded'] = True

        st.markdown("---")

    # --- Chatbot Interface ---
    st.header("💬 Ask the Agricultural Chatbot")

    # Initialize Chat History
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    diagnosis_info = st.session_state.get('diagnosis_info', None)

    if CLIENT is None:
        st.info("The chatbot is inactive because the Gemini API key is not configured.")
    elif diagnosis_info is None:
        st.info("The chatbot is inactive until a successful diagnosis is made.")

    else:
        # Display Chat History
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # User Input
        if prompt := st.chat_input("Ask a question about the diagnosis or treatment..."):

            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Build the context and call the LLM
            full_context = build_context_prompt(diagnosis_info, prompt)

            with st.chat_message("assistant"):
                with st.spinner("Assistant thinking..."):
                    try:
                        response = CLIENT.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=full_context,
                            # Set low temperature for better grounding and lower creativity
                            config=genai.types.GenerateContentConfig(
                                temperature=0.1
                            )
                        )
                        assistant_response = response.text
                        st.markdown(assistant_response)
                    except Exception as e:
                        assistant_response = f"I apologize, I encountered an error while consulting my knowledge base: {e}"
                        st.error(assistant_response)

            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})


if __name__ == '__main__':
    main()