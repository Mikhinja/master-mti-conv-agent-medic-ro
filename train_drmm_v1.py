import os
import json
import numpy as np
from typing import Callable
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
print(f'tensorflow version {tf.__version__}')
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, Dense, Dot, Layer, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from control_vars import *
from common_utils import *
from ml_utils import AggregationLayer

out_data_folder = f"{data_root}/models"
os.makedirs(out_data_folder, exist_ok=True)
max_seq_length = 200

# # Custom aggregation layer
# class AggregationLayer(Layer):
#     def __init__(self, **kwargs):
#         super(AggregationLayer, self).__init__(**kwargs)

#     def call(self, inputs):
#         return tf.reduce_sum(inputs, axis=1)

#     def get_config(self):
#         config = super(AggregationLayer, self).get_config()
#         return config

# Load JSON data
def load_json_data(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

# Estimate goodness based on question and answer properties
def estimate_goodness(question, answer):
    ret = 0
    if question.get('comments', 0) > 2 or len(answer.get('replies', [])) > 0:
        ret = 1
    if answer.get('votes', 0) > 0:
        ret = 2
    return ret

def estimate_goodness_from_feedback(question, answer):
    if 'feedback' not in answer:
        return estimate_goodness(question, answer)
    ret = 0
    for ff in answer['feedback']:
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
            if (len(row['answer']) > 2*max_seq_length or len(row['question']) > 2*max_seq_length) and row['goodness'] < 2:
                continue
            rows.append(row)
    df = pd.DataFrame(rows)
    return df

# Preprocess the text data
def preprocess_text(text):
    text = text.lower()
    return text

# Prepare the data for TensorFlow
def prepare_data(df):
    df['question'] = df['question'].apply(preprocess_text)
    df['answer'] = df['answer'].apply(preprocess_text)

    X = df[['question', 'answer']]
    y = df['goodness']

    print("Data distribution:\n", y.value_counts())

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

    print("Shapes:\nX_train_question_pad: ", X_train_question_pad.shape,
          "\nX_train_answer_pad: ", X_train_answer_pad.shape,
          "\nX_test_question_pad: ", X_test_question_pad.shape,
          "\nX_test_answer_pad: ", X_test_answer_pad.shape,
          "\ny_train: ", y_train.shape,
          "\ny_test: ", y_test.shape)

    return (X_train_question_pad, X_train_answer_pad, X_test_question_pad, X_test_answer_pad, 
            y_train, y_test, tokenizer)

# Define the DRMM model
def build_drmm_model(vocab_size, embedding_dim, max_seq_length, num_classes):
    question_input = Input(shape=(max_seq_length,), name='question_input')
    answer_input = Input(shape=(max_seq_length,), name='answer_input')
    
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_seq_length, name='embedding')
    question_embedding = embedding_layer(question_input)
    answer_embedding = embedding_layer(answer_input)
    
    matching = Dot(axes=-1, normalize=True)([question_embedding, answer_embedding])
    matching_histogram = Dense(30, activation='relu')(matching)
    dropout_1 = Dropout(0.5)(matching_histogram)
    matching_histogram = Dense(1, activation='relu')(dropout_1)
    
    aggregation = AggregationLayer()(matching_histogram)
    
    dropout_2 = Dropout(0.5)(aggregation)
    output = Dense(num_classes, activation='softmax', name='output')(dropout_2)
    
    model = Model(inputs=[question_input, answer_input], outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Load and parse data
labeled_data_file = f'{data_root}/annotation/questions/_questions_annotated.json'
unlabeled_data_file = f'{data_root}/annotation/questions/_questions_with_answers.json'

# Load and parse data
data_labeled = load_json_data(labeled_data_file)
data_unlabeled = load_json_data(unlabeled_data_file)

# Combine the data
data = {p[0]:p[1] if not p[0] in labeled_data_file else labeled_data_file[p[0]]
        for p in data_unlabeled.items()}

df = parse_json_to_df(data)

# Prepare data for training and evaluation
(X_train_question_pad, X_train_answer_pad, X_test_question_pad, X_test_answer_pad, 
 y_train, y_test, tokenizer) = prepare_data(df)

# Parameters
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100
num_classes = 3  # Number of classes in the target variable

# Build and compile the DRMM model
drmm_model = build_drmm_model(vocab_size, embedding_dim, max_seq_length, num_classes)
drmm_model.summary()

model_name = f'{out_data_folder}/drmm_model{max_seq_length}_embdim{embedding_dim}.keras'

if not os.path.exists(model_name):

    # Prepare the target data - is this it?
    y_train = np.expand_dims(y_train, -1)
    y_test = np.expand_dims(y_test, -1)
    # Define a checkpoint callback to save the best model
    checkpoint = ModelCheckpoint(model_name, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

    # Class weights to handle class imbalance
    class_weights = {0: 1., 1: 1., 2: 1.}

    # Convert y_train and y_test to integers if they are not
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # Print the unique classes and their counts
    print("Unique classes in y_train: ", np.unique(y_train, return_counts=True))
    print("Unique classes in y_test: ", np.unique(y_test, return_counts=True))

    # Train the model
    drmm_model.fit(
        [X_train_question_pad, X_train_answer_pad],
        y_train,
        epochs=10,
        batch_size=32,
        validation_data=([X_test_question_pad, X_test_answer_pad], y_test),
        class_weight=class_weights,
        callbacks=[checkpoint, reduce_lr]
    )

# Load the best model
best_model = load_model(model_name, custom_objects={'AggregationLayer': AggregationLayer})
print(f'Model saved as {model_name}')

# Evaluate the model
drmm_predictions = best_model.predict([X_test_question_pad, X_test_answer_pad])
drmm_predictions = np.argmax(drmm_predictions, axis=1)

from sklearn.metrics import classification_report

report_classification = classification_report(y_test, drmm_predictions)
print("DRMM Goodness Estimation Task Report:")
print(report_classification)
with open(f'{out_data_folder}/report_drmm_model_seq{max_seq_length}_embdim{embedding_dim}.txt', 'a+') as fp:
    print("DRMM Goodness Estimation Task Report:", file=fp)
    print(report_classification, file=fp)
