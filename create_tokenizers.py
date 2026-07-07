from pathlib import Path
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer

FILE_PATH = Path("fra.txt")
MODEL_PATH = Path("translation_model.keras")
ENG_TOKENIZER_PATH = Path("eng_tokenizer.pkl")
FRE_TOKENIZER_PATH = Path("fre_tokenizer.pkl")
MAX_SAMPLES = 10000

if not FILE_PATH.exists():
    raise SystemExit("fra.txt not found in workspace root")

lines = FILE_PATH.read_text(encoding="utf-8").splitlines()
english_sentences = []
french_sentences = []
for line in lines:
    parts = line.split("\t")
    if len(parts) >= 2:
        english_sentences.append(parts[0].lower())
        french_sentences.append("<start> " + parts[1].lower() + " <end>")

english_sentences = english_sentences[:MAX_SAMPLES]
french_sentences = french_sentences[:MAX_SAMPLES]

eng_tokenizer = Tokenizer(filters="")
eng_tokenizer.fit_on_texts(english_sentences)
eng_vocab_size = len(eng_tokenizer.word_index) + 1

fre_tokenizer = Tokenizer(filters="")
fre_tokenizer.fit_on_texts(french_sentences)
fre_vocab_size = len(fre_tokenizer.word_index) + 1

with ENG_TOKENIZER_PATH.open("wb") as f:
    pickle.dump(eng_tokenizer, f)
with FRE_TOKENIZER_PATH.open("wb") as f:
    pickle.dump(fre_tokenizer, f)

print(f"Saved eng_tokenizer.pkl (vocab_size={eng_vocab_size})")
print(f"Saved fre_tokenizer.pkl (vocab_size={fre_vocab_size})")

if MODEL_PATH.exists():
    model = load_model(MODEL_PATH)
    try:
        decoder_output_layer = model.get_layer("decoder_output")
        model_output_units = getattr(decoder_output_layer, "units", None)
        if model_output_units is None:
            output_shape = getattr(decoder_output_layer, "output_shape", None)
            model_output_units = output_shape[-1] if output_shape is not None else None
        print(f"Model decoder output units: {model_output_units}")
    except Exception as e:
        print("Could not inspect model output layer:", e)
else:
    print("Model file not found; tokenizers saved but could not check model output size.")
