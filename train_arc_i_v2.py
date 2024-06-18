import json
from typing import Callable
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
print(f'tensorflow version {tf.__version__}')
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Concatenate

from control_vars import *
from common_utils import *

out_data_folder = f"{data_root}/models"

# Load JSON data
def load_json_data(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

def estimate_goodness(question:dict, answer:dict):
    ret = 0
    if question.get('comments', 0) > 2 or len(answer.get('replies', [])) > 0:
        ret += 1
    # if question.get('likes', 0) > 0:
    #     ret += 1
    if answer.get('votes', 0) > 0:
        ret = min(2, ret+2)
    if len(get_censored_words_q_a(question, answer)) > 0:
        ret = max(0, ret-1)
    return ret

def estimate_goodness_from_feedback(question:dict, answer:dict):
    if 'feedback' not in answer:
        return estimate_goodness(question, answer)
    ret = 0
    for ff in answer['feedback']:
        # if ff.get('adopted') == 'Da':
        #     ret += 1
        if ff.get('ranking') == 'Raspunde complet':
            ret = 2
        elif ff.get('ranking') == 'Raspunde partial':
            ret = 1
    return ret

# Parse JSON data into pandas DataFrame
def parse_json_to_df(data):
    rows = []
    for key, value in data.items():
        for answer in value['answers']:
            row = {
                'question': value['question'],
                'answer': answer['text'],
                'goodness': estimate_goodness_from_feedback(value, answer)
            }
            if row['answer'] is None:
                # this is the werid case where a direct answer was deleted, but we still have a reply to it
                continue
            rows.append(row)
    df = pd.DataFrame(rows)
    return df

# Preprocess the text data
def preprocess_text(text):
    # Basic preprocessing steps like lowercasing, removing punctuation, etc.
    text = text.lower()
    return text

# Prepare the data for TensorFlow
def prepare_data(df, max_seq_length):
    df['question'] = df['question'].apply(preprocess_text)
    df['answer'] = df['answer'].apply(preprocess_text)
    
    X = df[['question', 'answer']]
    y = df['goodness']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train['question'].tolist() + X_train['answer'].tolist())

    X_train_question_seq = tokenizer.texts_to_sequences(X_train['question'])
    X_train_answer_seq = tokenizer.texts_to_sequences(X_train['answer'])
    X_test_question_seq = tokenizer.texts_to_sequences(X_test['question'])
    X_test_answer_seq = tokenizer.texts_to_sequences(X_test['answer'])

    X_train_question_pad = pad_sequences(X_train_question_seq, maxlen=max_seq_length)
    X_train_answer_pad = pad_sequences(X_train_answer_seq, maxlen=max_seq_length)
    X_test_question_pad = pad_sequences(X_test_question_seq, maxlen=max_seq_length)
    X_test_answer_pad = pad_sequences(X_test_answer_seq, maxlen=max_seq_length)

    return (X_train_question_pad, X_train_answer_pad, X_test_question_pad, X_test_answer_pad, 
            y_train, y_test, tokenizer)

# Define the ARC-I model
def build_arc_i_model(vocab_size, embedding_dim, input_length):
    input_question = Input(shape=(input_length,), name='question_input')
    input_answer = Input(shape=(input_length,), name='answer_input')
    
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length)
    
    question_embedding = embedding_layer(input_question)
    answer_embedding = embedding_layer(input_answer)
    
    conv_layer = Conv1D(filters=128, kernel_size=3, activation='relu')
    
    question_conv = conv_layer(question_embedding)
    answer_conv = conv_layer(answer_embedding)
    
    question_pool = GlobalMaxPooling1D()(question_conv)
    answer_pool = GlobalMaxPooling1D()(answer_conv)
    
    merged = Concatenate()([question_pool, answer_pool])
    
    dense = Dense(64, activation='relu')(merged)
    output_goodness = Dense(1, activation='sigmoid', name='goodness_output')(dense)
    
    model = Model(inputs=[input_question, input_answer], outputs=[output_goodness])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

labeled_data_file = f'{data_root}/annotation/questions/_questions_annotated.json'
unlabeled_data_file = f'{data_root}/annotation/questions/_questions_with_answers.json'

# Load and parse data
data_labeled = load_json_data(labeled_data_file)
data_unlabeled = load_json_data(unlabeled_data_file)

# combine the data by 
data = {p[0]:p[1] if not p[0] in labeled_data_file else labeled_data_file[p[0]]
        for p in data_unlabeled.items()}

df = parse_json_to_df(data)

# Prepare data for training and evaluation
max_seq_length = 500
(X_train_question_pad, X_train_answer_pad, X_test_question_pad, X_test_answer_pad, 
 y_train, y_test, tokenizer) = prepare_data(df, max_seq_length)

# Parameters
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100

# Build and compile the ARC-I model
arc_i_model = build_arc_i_model(vocab_size, embedding_dim, max_seq_length)
arc_i_model.summary()

# Train the ARC-I model
arc_i_model.fit(
    [X_train_question_pad, X_train_answer_pad],
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=([X_test_question_pad, X_test_answer_pad], y_test)
)

model_name = f'arc_i_model_seq{max_seq_length}_embdim{embedding_dim}.h5'
print(f'Saving model to {model_name}')
# Save the trained model
arc_i_model.save(model_name)

# Load the trained model
loaded_model = load_model('arc_i_model.h5')

# Evaluate the model
arc_i_predictions = arc_i_model.predict([X_test_question_pad, X_test_answer_pad])
arc_i_predictions = (arc_i_predictions > 0.5).astype(int)

from sklearn.metrics import classification_report

report_classification = classification_report(y_test, arc_i_predictions)
print("ARC-I Goodness Estimation Task Report:")
print(report_classification)
with open(f'{out_data_folder}/report_arc_i_model_seq{max_seq_length}_embdim{embedding_dim}.txt', 'w') as fp:
    print("ARC-I Goodness Estimation Task Report:", file=fp)
    print(report_classification, file=fp)
