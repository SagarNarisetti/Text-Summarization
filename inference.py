from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np
import pickle


import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


from src.utils import remove_punctuation, remove_punctuation_from_text, remove_number_from_text, remove_stopwords, \
    sequence_to_summary, sequence_to_text, max_length_percentage, trim_text_inference
from src.extras import contractions
from src.utils import handle_contractions,clean_text, trim_text_and_summary, rare_words_metrics
from src.helpers import extract_yml
from src.model_build import Lstm_class

yml_file = extract_yml()
maximum_text_length = yml_file["maximum_text_length"]
maximum_summary_length = yml_file["maximum_summary_length"]
start_token = yml_file["Start_Token"]
end_token = yml_file["End_Token"]

app = FastAPI()
lstm_helper = Lstm_class()

# Load models and tokenizers
model = tf.keras.models.load_model('models/lstm.keras')
LSTM_encoder_model = tf.keras.models.load_model('models/LSTM_encoder_model.keras')
LSTM_decoder_model = tf.keras.models.load_model('models/LSTM_decoder_model.keras')

with open('models/source_tokenizer.pkl', 'rb') as f:
    source_tokernizer = pickle.load(f)

with open('models/target_tokenizer.pkl', 'rb') as f:
    target_tokenizer = pickle.load(f)

# Build reverse index for target vocab
reverse_target_word_index = {idx: word for word, idx in target_tokenizer.word_index.items()}
target_word_index = target_tokenizer.word_index

# Define input data model
class TextInput(BaseModel):
    text: str

# preprocess input data 
def preprocess_text(text: str) -> list:
    text = handle_contractions(text, contractions)
    text = clean_text(text)
    text = trim_text_inference(text, yml_file["maximum_text_length"])
    text_sequence = source_tokernizer.texts_to_sequences([text])
    padded_text = pad_sequences(text_sequence, maximum_text_length, padding='post', truncating='post')
    return padded_text


def summarize_text(raw_text: str) -> str:
    input_padded = preprocess_text(raw_text)
    summary_sequence = lstm_helper.LSTM_decode_sequence(
        input_sequence=input_padded,
        encoder_model=LSTM_encoder_model,
        decoder_model=LSTM_decoder_model,
        reverse_target_word_index=reverse_target_word_index,
        target_word_index=target_word_index,
        start_token=start_token,
        end_token=end_token,
        maximum_summary_length=maximum_summary_length,
    )
    return summary_sequence.strip()

@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Welcome to LSTM summarization API.",
        "docs": "/docs",
        "redoc": "/redoc",
        "summarize_endpoint": "/summarize (POST)"
    }


@app.post("/summarize")
async def summarize(text_input: TextInput):
    summary = summarize_text(text_input.text)
    if not summary:
        raise HTTPException(status_code=400, detail="Failed to generate summary")
    return {"summary": summary}

    
if __name__ == "__main__":
    uvicorn.run("inference:app", host="0.0.0.0", port=8000, reload=True)

# Example input for testing
"""
A new music video shows rapper Snoop Dogg aiming a toy gun at a clown character parodying US President Donald Trump. The video also 
shows a TV airing a news conference with the headline 'Ronald Klump wants to deport all doggs' airing live from 'The Clown House'.
The video is for a remixed version of the song 'Lavender'


check: http://localhost:8000/docs for API documentation and testing.
"""