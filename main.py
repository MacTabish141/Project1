from flask import Flask, request, render_template
import tensorflow as tf
import pickle

app = Flask(__name__)

# Load model and tokenizer
model = tf.keras.models.load_model('gru_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

MAXLEN = 100  # Adjust if needed

def predict_sentiment(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=MAXLEN)
    prediction = model.predict(padded)[0][0]
    label = "Positive" if prediction >= 0.5 else "Negative"
    confidence = round(prediction if prediction >= 0.5 else 1 - prediction, 2)
    return label, confidence

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    if request.method == "POST":
        text = request.form.get("text", "")
        if text:
            result, confidence = predict_sentiment(text)
    return render_template("index.html", result=result, confidence=confidence)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

