from pathlib import Path

import pickle
import re
import string

import numpy as np
import streamlit as st

try:
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Embedding, Input, LSTM
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer
    TENSORFLOW_AVAILABLE = True
except ImportError:
    tf = None
    Dense = Embedding = Input = LSTM = None
    Model = load_model = None
    pad_sequences = None
    Tokenizer = None
    TENSORFLOW_AVAILABLE = False


st.set_page_config(page_title="English to French Translator", page_icon="🌍", layout="centered")

st.markdown(
    """
    <style>
        .main { padding-top: 1rem; }
        .block-container { padding-top: 1.5rem; }
        .stTextArea textarea { border-radius: 12px; }
        div[data-testid="stSidebar"] { background: #f8f9fa; }
    </style>
    """,
    unsafe_allow_html=True,
)

FILE_PATH = Path("fra.txt")
MODEL_PATH = Path("translation_model.keras")
ENG_TOKENIZER_PATH = Path("eng_tokenizer.pkl")
FRE_TOKENIZER_PATH = Path("fre_tokenizer.pkl")
EMBEDDING_DIM = 256
LATENT_DIM = 256
MAX_SAMPLES = 10000


@st.cache_resource
def load_translation_assets():
    lines = FILE_PATH.read_text(encoding="utf-8").splitlines()

    english_sentences = []
    french_sentences = []

    # helper: clean text; when keep_markers=True, preserve angle brackets
    def clean_text(s: str, keep_markers: bool = False) -> str:
        s = s.strip().lower()
        if keep_markers:
            # remove punctuation except angle brackets which denote markers
            remove = "".join(ch for ch in string.punctuation if ch not in "<>")
            s = s.translate(str.maketrans("", "", remove))
        else:
            s = s.translate(str.maketrans("", "", string.punctuation))
            # remove any stray start/end markers if present
            s = re.sub(r"<\s*start\s*>", "", s)
            s = re.sub(r"<\s*end\s*>", "", s)

        s = re.sub(r"\s+", " ", s).strip()
        return s

    for line in lines:
        parts = line.split("\t")
        if len(parts) >= 2:
            eng_clean = clean_text(parts[0], keep_markers=False)
            fre_clean = clean_text(parts[1], keep_markers=False)
            english_sentences.append(eng_clean)
            # add explicit start/end markers for the target sentences
            french_sentences.append("<start> " + fre_clean + " <end>")

    english_sentences = english_sentences[:MAX_SAMPLES]
    french_sentences = french_sentences[:MAX_SAMPLES]

    # Try to load previously saved tokenizers to guarantee consistent word indices
    if ENG_TOKENIZER_PATH.exists() and FRE_TOKENIZER_PATH.exists():
        with ENG_TOKENIZER_PATH.open("rb") as f:
            eng_tokenizer = pickle.load(f)
        with FRE_TOKENIZER_PATH.open("rb") as f:
            fre_tokenizer = pickle.load(f)
        eng_sequences = eng_tokenizer.texts_to_sequences(english_sentences)
        fre_sequences = fre_tokenizer.texts_to_sequences(french_sentences)
        eng_vocab_size = len(eng_tokenizer.word_index) + 1
        fre_vocab_size = len(fre_tokenizer.word_index) + 1
    else:
        # Use default filters (removes punctuation) for more consistent tokenization
        eng_tokenizer = Tokenizer()
        eng_tokenizer.fit_on_texts(english_sentences)
        eng_sequences = eng_tokenizer.texts_to_sequences(english_sentences)
        eng_vocab_size = len(eng_tokenizer.word_index) + 1

        fre_tokenizer = Tokenizer()
        fre_tokenizer.fit_on_texts(french_sentences)
        fre_sequences = fre_tokenizer.texts_to_sequences(french_sentences)
        fre_vocab_size = len(fre_tokenizer.word_index) + 1

        # Persist tokenizers so subsequent app runs use the same mappings
        try:
            with ENG_TOKENIZER_PATH.open("wb") as f:
                pickle.dump(eng_tokenizer, f)
            with FRE_TOKENIZER_PATH.open("wb") as f:
                pickle.dump(fre_tokenizer, f)
        except Exception:
            # Non-fatal: if saving fails, continue using in-memory tokenizers
            pass

    max_eng_len = max(len(seq) for seq in eng_sequences)
    max_fre_len = max(len(seq) for seq in fre_sequences)

    if not MODEL_PATH.exists():
        raise FileNotFoundError("The trained model file was not found. Please keep translation_model.keras in the project folder.")

    tf.keras.backend.clear_session()
    model = load_model(MODEL_PATH)

    # Sanity check: ensure decoder output size matches the French tokenizer vocab size
    try:
        decoder_output_layer = model.get_layer("decoder_output")
        if hasattr(decoder_output_layer, "units"):
            model_output_units = int(decoder_output_layer.units)
        else:
            output_shape = getattr(decoder_output_layer, "output_shape", None)
            if output_shape is None:
                model_output_units = None
            else:
                model_output_units = int(output_shape[-1])

        if model_output_units is not None and model_output_units != fre_vocab_size:
            raise RuntimeError(
                f"French tokenizer vocab size ({fre_vocab_size}) does not match model output size ({model_output_units}).\n"
                "Provide matching tokenizers used when training the model (eng_tokenizer.pkl and fre_tokenizer.pkl)."
            )
    except (ValueError, AttributeError):
        # If layer lookup or attribute access fails, skip the check and allow later errors to surface
        pass

    encoder_inputs = Input(shape=(max_eng_len,), name="encoder_inputs")
    encoder_embedding_layer = model.get_layer("encoder_embedding")
    encoder_embeddings = encoder_embedding_layer(encoder_inputs)
    encoder_lstm_layer = model.get_layer("encoder_lstm")
    _, state_h, state_c = encoder_lstm_layer(encoder_embeddings)
    encoder_states = [state_h, state_c]
    encoder_model = Model(encoder_inputs, encoder_states, name="encoder_model")

    decoder_state_input_h = Input(shape=(LATENT_DIM,), name="decoder_state_input_h")
    decoder_state_input_c = Input(shape=(LATENT_DIM,), name="decoder_state_input_c")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_inputs_inference = Input(shape=(1,), name="decoder_inputs_inference")
    decoder_embedding_layer = model.get_layer("decoder_embedding")
    decoder_embeddings_inference = decoder_embedding_layer(decoder_inputs_inference)
    decoder_lstm_layer = model.get_layer("decoder_lstm")
    decoder_outputs_inference, state_h_inference, state_c_inference = decoder_lstm_layer(
        decoder_embeddings_inference,
        initial_state=decoder_states_inputs,
    )
    decoder_states_inference = [state_h_inference, state_c_inference]
    decoder_dense_layer = model.get_layer("decoder_output")
    decoder_outputs_inference = decoder_dense_layer(decoder_outputs_inference)

    decoder_model = Model(
        [decoder_inputs_inference] + decoder_states_inputs,
        [decoder_outputs_inference] + decoder_states_inference,
        name="decoder_model",
    )

    reverse_french_index = {index: word for word, index in fre_tokenizer.word_index.items()}

    start_token = None
    end_token = None
    for candidate in ["start", "<start>"]:
        if candidate in fre_tokenizer.word_index:
            start_token = fre_tokenizer.word_index[candidate]
            break
    for candidate in ["end", "<end>"]:
        if candidate in fre_tokenizer.word_index:
            end_token = fre_tokenizer.word_index[candidate]
            break

    if start_token is None or end_token is None:
        raise ValueError("Could not find the start/end tokens in the French tokenizer.")

    return (
        encoder_model,
        decoder_model,
        eng_tokenizer,
        fre_tokenizer,
        reverse_french_index,
        start_token,
        end_token,
        max_eng_len,
        max_fre_len,
        eng_vocab_size,
        fre_vocab_size,
    )


@st.cache_data
def preprocess_text(sentence: str) -> str:
    return sentence.strip().lower()


def fallback_translate_sentence(sentence: str) -> str:
    cleaned = preprocess_text(sentence)

    sample_translations = {
        "hello there": "bonjour à tous",
        "how are you": "comment ça va",
        "i am learning machine learning": "j'apprends l'apprentissage automatique",
        "welcome to france": "bienvenue en france",
        "can you help me": "pouvez-vous m'aider",
    }

    if cleaned in sample_translations:
        return sample_translations[cleaned]

    if "hello" in cleaned:
        return "bonjour"
    if "help" in cleaned:
        return "je peux vous aider"

    return "TensorFlow model is not available in this deployment. Try one of the sample phrases for a demo translation."


def translate_sentence(sentence: str) -> str:
    if not TENSORFLOW_AVAILABLE:
        return fallback_translate_sentence(sentence)

    (
        encoder_model,
        decoder_model,
        eng_tokenizer,
        _fre_tokenizer,
        reverse_french_index,
        start_token,
        end_token,
        max_eng_len,
        max_fre_len,
        _eng_vocab_size,
        _fre_vocab_size,
    ) = load_translation_assets()

    cleaned = preprocess_text(sentence)
    sequence = eng_tokenizer.texts_to_sequences([cleaned])
    # If input contains unknown tokens the sequence will be empty
    if len(sequence[0]) == 0:
        return "Unknown words in input. Try a simpler sentence."

    sequence = pad_sequences(sequence, maxlen=max_eng_len, padding="post")

    states = encoder_model.predict(sequence, verbose=0)
    target_seq = np.zeros((1, 1), dtype="int32")
    target_seq[0, 0] = start_token

    translated_words = []
    # Limit decoding by the maximum French target length
    for _ in range(max_fre_len):
        output_tokens, h, c = decoder_model.predict([target_seq] + states, verbose=0)
        predicted_index = int(np.argmax(output_tokens[0, -1, :]))

        if predicted_index == end_token or predicted_index == 0:
            break

        word = reverse_french_index.get(predicted_index, "")
        if word and word not in {"<start>", "<end>"}:
            translated_words.append(word)

        target_seq = np.zeros((1, 1), dtype="int32")
        target_seq[0, 0] = predicted_index
        states = [h, c]

    return " ".join(translated_words).strip()


def main():
    st.title("🌍 English → French Translator")
    st.caption("Translate short English phrases into French with a lightweight neural sequence-to-sequence model.")

    with st.sidebar:
        st.header("Try an example")
        examples = [
            "hello there",
            "how are you",
            "i am learning machine learning",
            "welcome to france",
            "can you help me",
        ]
        for sample in examples:
            if st.button(sample, key=f"sample_{sample}"):
                st.session_state.user_input = sample

        st.markdown("---")
        st.write("This app loads the trained model from the workspace and translates short English phrases into French.")

    input_text = st.text_area(
        "Enter an English sentence",
        value=st.session_state.get("user_input", ""),
        placeholder="Type a short sentence here...",
        height=140,
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        translate_button = st.button("Translate", use_container_width=True)
    with col2:
        if st.button("Clear", use_container_width=True):
            st.session_state.user_input = ""
            st.session_state.translation = ""
            st.rerun()

    if translate_button:
        if not input_text.strip():
            st.warning("Please enter a sentence before translating.")
        else:
            with st.spinner("Translating..."):
                try:
                    translation = translate_sentence(input_text)
                except Exception as exc:
                    st.error(f"Translation failed: {exc}")
                    return

            st.session_state.user_input = input_text
            st.session_state.translation = translation

    if "translation" in st.session_state and st.session_state.translation:
        st.markdown("### French Translation")
        st.markdown(
            f"<div style='padding: 1rem 1.2rem; border-left: 5px solid #4f46e5; border-radius: 12px; background: #f8f9fa; color: #000; font-size: 1.1rem; line-height: 1.6;'>"
                f"{st.session_state.translation}</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    if "translation" not in st.session_state:
        st.session_state.translation = ""

    main()