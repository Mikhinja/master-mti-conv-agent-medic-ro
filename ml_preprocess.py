#pip install tensorflow keras sklearn pandas numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# these shouldn't be needed except for when working with raw data
# from keras.preprocessing.text import Tokenize
# from keras.preprocessing.sequence import pad_sequences

import tensorflow
# from tensorflow.python.keras import Model
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Softmax
from tensorflow.python.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Concatenate, Dot, Lambda 

from datetime import datetime, timedelta
import shutil
import random
import time
from control_vars import *
from common_utils import *
from utils_confusion_matrix import *

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
errors_file = f'{logs_root}/ml_errors_{timestamp}.txt'
log_file = f'{logs_root}/ml_log_{timestamp}.txt'

in_data_folder = f"{data_root}/analyzed2"
in_data_annotated_folder = f'{data_root}/annotation'

words_count_file = f'{data_root}/sanitized1/words_count.json'
categories_file = f'{data_root}/sanitized1/categories.json'

out_data_folder = f"{data_root}/models"
if clean_data_folder and os.path.exists(out_data_folder):
    shutil.rmtree(out_data_folder)
os.makedirs(out_data_folder, exist_ok=True)

def get_annotated_questions()->dict[str,dict]:
    questions_annotated = {}
    temp_path = f'{in_data_annotated_folder}/questions'
    list_ids = os.listdir(temp_path)
    for question_name in list_ids:
        if not question_name.endswith('.json'):
            continue
        question = {}
        with open(f'{temp_path}/{question_name}', "r") as fp:
            question = json.load(fp)
        questions_annotated[question['id']] = question
    return questions_annotated

def get_annotated_question_answer_pairs():
    pass

def get_all_questions_with_answers(errors_file)->dict[str,dict]:
    ids_questions_with_answers:list[str] = []
    with open(f'{in_data_folder}/questions_with_answers.json', 'r') as fp:
        ids_questions_with_answers = json.load(fp)

    questions_with_answers = {}
    for q_id in ids_questions_with_answers:
        if 'index.php?' in q_id:
        # TODO: find out where the error is that introduces these as question ids
            continue
        question_file = f'{in_data_folder}/questions/{q_id}.json'
        question = {}
        try:
            with open(question_file, "r") as fp:
                question = json.load(fp)
            questions_with_answers[question['id']] = question
        except Exception as exc:
        # do nothing, there must have been some error somewhere
        # TODO: investigate why such errors appear
            with open(errors_file, "a+") as fp:
                print(f'ERROR: reading question file {question_file}: {exc}', file=fp)
    return questions_with_answers

def build_drmm_model(vocab_size, embedding_dim, input_length):
    input_question = Input(shape=(input_length,), name='question_input')
    input_answer = Input(shape=(input_length,), name='answer_input')
    
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length)
    
    question_embedding = embedding_layer(input_question)
    answer_embedding = embedding_layer(input_answer)
    
    local_interactions = Dot(axes=-1)([question_embedding, answer_embedding])
    
    match_hist = Dense(30, activation='tanh')(local_interactions)
    match_hist = Softmax()(match_hist)
    
    dense = Dense(64, activation='relu')(match_hist)
    output_adoption = Dense(1, activation='sigmoid', name='adoption_output')(dense)
    output_relevancy = Dense(1, activation='sigmoid', name='relevancy_output')(dense)
    
    model = Model(inputs=[input_question, input_answer], outputs=[output_adoption, output_relevancy])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
