from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import json
import os

from control_vars import *
from common_utils import *


out_data_folder = f"{data_root}/models"

import pickle

tokenizer = tf.keras.preprocessing.text.Tokenizer()
# loading
with open(f'{out_data_folder}/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


max_seq_length = 200
embedding_dim = 100
model_name = f'{out_data_folder}/arc_i_model_seq{max_seq_length}_embdim{embedding_dim}.keras'


# Incarca modelul ARC-I salvat
model = tf.keras.models.load_model(model_name)

# Functie pentru vectorizarea intrebarii si raspunsului
def preprocess_text(text, tokenizer, max_len):
    text = text.lower()
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_len)
    return padded_sequence


app = Flask(__name__)
CORS(app)

webappdata_file = 'webapp_data.json'
def load_data():
    if os.path.exists(webappdata_file):
        with open(webappdata_file, 'r') as f:
            return json.load(f)
    return []

def save_data(data):
    with open(webappdata_file, 'w') as f:
        json.dump(data, f, indent=2)

def eval_one_q(question:str, answer:str):
    if answer and answer.strip():
        question_seq = preprocess_text(question, tokenizer, max_seq_length)
        answer_seq = preprocess_text(answer, tokenizer, max_seq_length)
        
        prediction = model.predict([question_seq, answer_seq])
        score = prediction[0][0]
    else:
        score = 0.0
    return float(score)

@app.route('/questions', methods=['GET'])
def get_questions():
    data = load_data()
    for q in data:
        answers = []
        for a in q['answers']:
            if a['answer'] and a['answer'].strip():
                if 'score' not in a or not a['score']:
                    score = eval_one_q(q['question'], a['answer'])
                    a['score'] = score
                answers.append(a)
        q['answers'] = answers
    return jsonify(data)

@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.get_json()
    question = data.get('question')
    answer = data.get('answer')
    
    score = eval_one_q(question, answer)
    
    evaluation = f'Score: {score:.2f}'
    
    return jsonify({'evaluation': evaluation, 'score': score})

@app.route('/submit', methods=['POST'])
def submit():
    data = request.get_json()
    question = data.get('question')
    answer = data.get('answer')
    score = data.get('score')
    
    questions = load_data()
    for q in questions:
        if q['question'] == question:
            q['answers'].append({'answer': answer, 'score': score})
            break
    else:
        questions.append({'question': question, 'answers': [{'answer': answer, 'score': score}]})

    save_data(questions)
    
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)
