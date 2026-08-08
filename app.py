#am.sync365@gmail.com
import streamlit as st
import pickle
import onnxruntime as ort
import numpy as np
import pandas as pd


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="😊",
    layout="wide"
)


# =========================================================
# Load Resources
# =========================================================

@st.cache_resource
def load_vectorizer():
    with open("vectorizer.pkl", "rb") as file:
        return pickle.load(file)


@st.cache_resource
def load_model():
    return ort.InferenceSession(
        "emotion.onnx",
        providers=["CPUExecutionProvider"]
    )


@st.cache_data
def load_label_map():
    with open("label_map.pkl", "rb") as file:
        return pickle.load(file)


vectorize_layer = load_vectorizer()
model = load_model()
label_map = load_label_map()

input_name = model.get_inputs()[0].name


# =========================================================
# Emotion Information
# =========================================================

emotion_emoji = {
    "anger": "😡",
    "fear": "😨",
    "joy": "😊",
    "love": "❤️",
    "sadness": "😢",
    "surprise": "😲"
}


# =========================================================
# Example Sentences
# =========================================================

examples = [
    "I am so happy today!",
    "I feel very sad and lonely",
    "This makes me really angry!",
    "I am surprised by the news"
]


# =========================================================
# Prediction Function
# =========================================================

def predict_emotion(text):

    x = np.array([text])

    # Text → Token IDs
    tokens = vectorize_layer(x).numpy()

    # ONNX prediction
    prediction = model.run(
        None,
        {input_name: tokens}
    )[0][0]

    predicted_index = prediction.argmax()
    predicted_emotion = label_map[predicted_index]

    return predicted_emotion, prediction


# =========================================================
# Probability Table
# =========================================================

def create_probability_table(probabilities):

    data = []

    for index, probability in enumerate(probabilities):

        emotion = label_map[index]

        data.append({
            "Emotion": (
                f"{emotion_emoji[emotion]} "
                f"{emotion.capitalize()}"
            ),
            "Probability": f"{probability * 100:.2f}%"
        })

    return pd.DataFrame(data)


# =========================================================
# Session State
# =========================================================

if "selected_text" not in st.session_state:
    st.session_state.selected_text = None

if "prediction" not in st.session_state:
    st.session_state.prediction = None


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:

    st.title("💬 Sample Texts")

    st.write(
        "Select one of the example sentences "
        "to see the model prediction."
    )

    st.divider()

    for i, example in enumerate(examples):

        if st.button(
            example,
            key=f"example_{i}",
            use_container_width=True
        ):

            emotion, probabilities = predict_emotion(
                example
            )

            st.session_state.selected_text = example

            st.session_state.prediction = (
                emotion,
                probabilities
            )


    st.divider()

    st.subheader("✍️ Your Own Text")

    user_text = st.text_area(
        "Enter your sentence:",
        placeholder="Write something here...",
        height=120
    )

    if st.button(
        "🔍 Predict My Text",
        use_container_width=True
    ):

        if user_text.strip():

            emotion, probabilities = predict_emotion(
                user_text
            )

            st.session_state.selected_text = user_text

            st.session_state.prediction = (
                emotion,
                probabilities
            )

        else:

            st.warning(
                "Please enter a sentence first."
            )


# =========================================================
# Main Prediction Area
# =========================================================

st.title("😊 Emotion Classifier")

st.write(
    "A text classification model using "
    "pre-trained GloVe embeddings and a 1D CNN."
)


st.divider()


if st.session_state.prediction is None:

    st.info(
        "👈 Select a sample sentence from the sidebar "
        "or enter your own text."
    )


else:

    emotion, probabilities = (
        st.session_state.prediction
    )

    emoji = emotion_emoji[emotion]

    confidence = probabilities.max() * 100


    # =====================================================
    # Selected Text
    # =====================================================

    st.subheader("📝 Input Text")

    st.info(
        st.session_state.selected_text
    )


    # =====================================================
    # Prediction
    # =====================================================

    st.subheader("🎯 Prediction")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Emotion",
            f"{emoji} {emotion.capitalize()}"
        )

    with col2:

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )


    # =====================================================
    # Probability Table
    # =====================================================

    with st.expander(
        "📊 Show Prediction Probabilities"
    ):

        probability_df = create_probability_table(
            probabilities
        )

        st.table(probability_df)

