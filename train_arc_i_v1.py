import json
from typing import Callable
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
print(f'tensorflow versrion {tf.__version__}')
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Concatenate

from control_vars import *
from common_utils import *

out_data_folder = f"{data_root}/models"
os.makedirs(out_data_folder, exist_ok=True)

# Load JSON data
def load_json_data(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

# Parse JSON data into pandas DataFrame
def parse_feedback_json_to_df(data):
    rows = []
    for q_id, question in data.items():
        for answer in question['answers']:
            # for now skip answers without feedback
            if 'feedback' not in answer:
                continue
            row = {
                'question': question['question'],
                'answer': answer['text'],
                'adoption': any(feedback['adopted'] == 'Da' for feedback in answer.get('feedback', [])),
                'relevancy': any(feedback['ranking'] in ['Raspunde complet', 'Raspunde partial'] for feedback in answer.get('feedback', []))
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    return df

# basic idea is:
#   * number of likes
#   * number
def estimate_goodness(question, answer):
    ret = 0.2
    ret += -0.1 * (get_censored_words(question)>0)
    ret += -0.1 * (get_censored_words_q_a(question, answer)>0)
    ret += 0.1 * question['likes']
    ret += 0.1 * question['comments']
    ret += 0.5 * answer['votes']
    ret += 0.2 * len(answer['replies'])
    return ret

def estimate_goodness_from_feedback(question, answer):
    if 'feedback' not in answer:
        return estimate_goodness(question, answer)
    ret = 0
    for ff in answer['feedback']:
        if ff['adopted'] == 'Da':
            ret += 1
        if ff['ranking'] == 'Raspunde complet':
            ret += 2
        elif ff['ranking'] == 'Raspunde partial':
            ret += 1
    return ret

# Now do the ones not annotated manually, by applying the estimation function
def parse_all_json_to_df(data, est_func:Callable):
    rows = []
    max_estimation = 0
    for q_id, question in data.items():
        for answer in question['answers']:
            row = {
                'question': question['question'],
                'answer': answer['text'],
                'estimation': est_func(question, answer),
            }
            max_estimation = max(row['estimation'], max_estimation)
            rows.append(row)
    # normalize estimations
    for row in rows:
        row['estimation'] /= max_estimation
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
    y_adoption = df['adoption'].astype(int)
    y_relevancy = df['relevancy'].astype(int)

    X_train, X_test, y_adoption_train, y_adoption_test, y_relevancy_train, y_relevancy_test = train_test_split(
        X, y_adoption, y_relevancy, test_size=0.2, random_state=42
    )

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
            y_adoption_train, y_adoption_test, y_relevancy_train, y_relevancy_test, tokenizer)

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
    output_adoption = Dense(1, activation='sigmoid', name='adoption_output')(dense)
    output_relevancy = Dense(1, activation='sigmoid', name='relevancy_output')(dense)
    
    model = Model(inputs=[input_question, input_answer], outputs=[output_adoption, output_relevancy])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

labeled_data_file = f'{data_root}/annotation/questions/_questions_annotated.json'

# Load and parse data
data_labeled = load_json_data(labeled_data_file)
df = parse_feedback_json_to_df(data_labeled)

# Prepare data for training and evaluation
max_seq_length = 1000
(X_train_question_pad, X_train_answer_pad, X_test_question_pad, X_test_answer_pad, 
 y_adoption_train, y_adoption_test, y_relevancy_train, y_relevancy_test, tokenizer) = prepare_data(df, max_seq_length)

# Parameters
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100

# Build and compile the ARC-I model
arc_i_model = build_arc_i_model(vocab_size, embedding_dim, max_seq_length)
arc_i_model.summary()

# Compile the model with separate metrics for each output
arc_i_model.compile(
    optimizer='adam',
    loss={'adoption_output': 'binary_crossentropy', 'relevancy_output': 'binary_crossentropy'},
    metrics={'adoption_output': ['accuracy'], 'relevancy_output': ['accuracy']}
)

# Train the ARC-I model
arc_i_model.fit(
    [X_train_question_pad, X_train_answer_pad],
    {'adoption_output': y_adoption_train, 'relevancy_output': y_relevancy_train},
    epochs=10,
    batch_size=32,
    validation_data=([X_test_question_pad, X_test_answer_pad], {'adoption_output': y_adoption_test, 'relevancy_output': y_relevancy_test})
)

# Evaluate the model
arc_i_predictions = arc_i_model.predict([X_test_question_pad, X_test_answer_pad])
arc_i_adoption_preds = (arc_i_predictions[0] > 0.5).astype(int)
arc_i_relevancy_preds = (arc_i_predictions[1] > 0.5).astype(int)

from sklearn.metrics import classification_report

print("ARC-I Adoption Task Report:")
print(classification_report(y_adoption_test, arc_i_adoption_preds))
print("ARC-I Relevancy Task Report:")
print(classification_report(y_relevancy_test, arc_i_relevancy_preds))
