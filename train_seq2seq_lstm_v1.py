import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

import os

from control_vars import *
from common_utils import *

out_data_folder = f"{data_root}/models"

# Load JSON data
def load_json_data(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

# Filter good answers based on votes, replies, and feedback
def filter_good_answers(data):
    rows = []
    for key, value in data.items():
        for answer in value['answers']:
            if answer['text'] is None:
                # this is the werid case where a direct answer was deleted, but we still have a reply to it
                continue
            if 'feedback' in answer:
                for feedback in answer['feedback']:
                    if feedback.get('ranking') == 'Raspunde complet' or feedback.get('ranking') == 'Raspunde partial':
                        rows.append((value['question'], answer['text']))
            elif 'votes' in answer and answer['votes'] > 0:
                rows.append((value['question'], answer['text']))
            elif 'replies' in answer and len(answer['replies']) > 0:
                rows.append((value['question'], answer['text']))
    return rows

# Load and filter data
labeled_data_file = f'{data_root}/annotation/questions/_questions_annotated.json'
unlabeled_data_file = f'{data_root}/annotation/questions/_questions_with_answers.json'

# Load and parse data
data_labeled = load_json_data(labeled_data_file)
data_unlabeled = load_json_data(unlabeled_data_file)
# combine the data 
data = {p[0]:p[1] if not p[0] in labeled_data_file else labeled_data_file[p[0]]
        for p in data_unlabeled.items()}

filtered_data = filter_good_answers(data)

# Prepare data for training
questions, answers = zip(*filtered_data)
questions = list(questions)
answers = ['<start> ' + answer + ' <end>' for answer in answers]

# Tokenize and pad sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions + answers)

question_sequences = tokenizer.texts_to_sequences(questions)
answer_sequences = tokenizer.texts_to_sequences(answers)

max_seq_length = 90
question_padded = pad_sequences(question_sequences, maxlen=max_seq_length, padding='post')
answer_padded = pad_sequences(answer_sequences, maxlen=max_seq_length, padding='post')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(question_padded, answer_padded, test_size=0.2, random_state=42)

# Define model parameters
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100
lstm_units = 128

# Define the seq2seq model
def build_seq2seq_model(vocab_size, embedding_dim, lstm_units, max_seq_length):
    # Encoder
    encoder_inputs = Input(shape=(max_seq_length,), name='encoder_inputs')
    encoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_seq_length, name='encoder_embedding')(encoder_inputs)
    encoder_lstm = LSTM(lstm_units, return_state=True, name='encoder_lstm')
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(max_seq_length,), name='decoder_inputs')
    decoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_seq_length, name='decoder_embedding')(decoder_inputs)
    decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = Dense(vocab_size, activation='softmax', name='decoder_dense')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Build and compile the model
seq2seq_model = build_seq2seq_model(vocab_size, embedding_dim, lstm_units, max_seq_length)
seq2seq_model.summary()

# Prepare the target data
y_train = np.expand_dims(y_train, -1)
y_test = np.expand_dims(y_test, -1)

model_name = f'{out_data_folder}/seq2seq_model.keras'

if not os.path.exists(model_name):
    # Define a checkpoint callback to save the best model
    checkpoint = ModelCheckpoint(model_name, save_best_only=True, monitor='val_loss', mode='min', verbose=1)

    # Train the model
    # Remember to clean or rename the file in order to actually train
    seq2seq_model.fit(
        [X_train, X_train],
        y_train,
        epochs=10,
        batch_size=32,
        validation_data=([X_test, X_test], y_test),
        callbacks=[checkpoint]
    )

# Load the best model
best_model = load_model(model_name)

# Function to generate answers
def generate_answer(model, tokenizer, question, max_seq_length):
    question_seq = tokenizer.texts_to_sequences([question])
    question_padded = pad_sequences(question_seq, maxlen=max_seq_length, padding='post')
    
    # Initialize the decoder input with the start token
    answer_seq = np.zeros((1, max_seq_length))
    answer_seq[0, 0] = tokenizer.word_index['<start>']
    
    for i in range(1, max_seq_length):
        output_tokens = model.predict([question_padded, answer_seq])
        sampled_token_index = np.argmax(output_tokens[0, i - 1, :])
        answer_seq[0, i] = sampled_token_index
        
        if sampled_token_index == tokenizer.word_index['<end>']:
            break
    
    answer = tokenizer.sequences_to_texts(answer_seq)[0]
    return answer

# Test the model with a new question
new_question = "spuneti-mi si mie cum pot sa scap de e coli ?"
generated_answer = generate_answer(best_model, tokenizer, new_question, max_seq_length)
print(f"Generated Answer: {generated_answer}")
