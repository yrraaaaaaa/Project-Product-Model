import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model 
model = tf.keras.models.load_model("model/model_lstm.h5")

# Load tokenizer 
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load label encoder 
with open("model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# FastAPI app 
app = FastAPI()

# Input text 
class TextIn(BaseModel):
    text: str

# Endpoint 
@app.post("/predict")
def predict(data: TextIn):
    # Preprocess input
    seq = tokenizer.texts_to_sequences([data.text])
    padded = pad_sequences(seq, maxlen=100)

    # Prediksi
    pred = model.predict(padded)
    label_idx = np.argmax(pred, axis=1)[0]
    label = label_encoder.inverse_transform([label_idx])[0]

    return {"text": data.text, "prediction": label}