import json
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
print(f'tensorflow vesrion {tf.__version__}')
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Concatenate, Dot, Lambda, Softmax

from control_vars import *


# Load JSON data
def load_json_data(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

# Parse JSON data into pandas DataFrame
def parse_json_to_df(data):
    rows = []
    for key, value in data.items():
        for answer in value['answers']:
            row = {
                'question': value['question'],
                'answer': answer['text'],
                'adoption': any(feedback['adopted'] == 'Da' for feedback in answer.get('feedback', [])),
                'relevancy': any(feedback['ranking'] == 'Raspunde complet' for feedback in answer.get('feedback', []))
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    return df

# Preprocess the text data
def preprocess_text(text):
    # Basic preprocessing steps like lowercasing, removing punctuation, etc.
    text = text.lower()
    return text

# Prepare the data for TensorFlow
def prepare_data(df):
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

    max_seq_length = 100  # Define max sequence length for padding
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
data = load_json_data(labeled_data_file)
df = parse_json_to_df(data)

# Prepare data for training and evaluation
(X_train_question_pad, X_train_answer_pad, X_test_question_pad, X_test_answer_pad, 
 y_adoption_train, y_adoption_test, y_relevancy_train, y_relevancy_test, tokenizer) = prepare_data(df)

# Parameters
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100
max_seq_length = 100

# Build and compile the ARC-I model
arc_i_model = build_arc_i_model(vocab_size, embedding_dim, max_seq_length)
arc_i_model.summary()

# Train the ARC-I model
arc_i_model.fit(
    [X_train_question_pad, X_train_answer_pad],
    [y_adoption_train, y_relevancy_train],
    epochs=10,
    batch_size=32,
    validation_data=([X_test_question_pad, X_test_answer_pad], [y_adoption_test, y_relevancy_test])
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
