from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from control_vars import *
from common_utils import *

app = Flask(__name__)

out_data_folder = f"{data_root}/models"

import pickle

tokenizer = tf.keras.preprocessing.text.Tokenizer()
# loading
with open(f'{out_data_folder}/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


max_len = 100  # mai este asta utila sau ar trebui max_seq_length?
max_seq_length = 500
embedding_dim = 100
model_name = f'{out_data_folder}/arc_i_model_seq{max_seq_length}_embdim{embedding_dim}.h5'

# Incarca modelul ARC-I salvat
model = tf.keras.models.load_model(model_name)

# Functie pentru vectorizarea intrebarii si raspunsului
def preprocess_text(text, tokenizer, max_len):
    text = text.lower()
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_len)
    return padded_sequence


@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.get_json()
    question = data.get('question')
    answer = data.get('answer')
    
    question_seq = preprocess_text(question, tokenizer, max_len)
    answer_seq = preprocess_text(answer, tokenizer, max_len)
    
    prediction = model.predict([question_seq, answer_seq])
    score = prediction[0][0]
    
    evaluation = f'Score: {score:.2f}'
    return jsonify({'evaluation': evaluation})

if __name__ == '__main__':
    app.run(debug=True)
