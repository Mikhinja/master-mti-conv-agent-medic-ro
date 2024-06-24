from typing import Callable
from tensorflow.keras.layers import Layer
import tensorflow as tf
import torch
from torch.utils.data import Dataset

from control_vars import *
from common_utils import *

out_data_folder = f"{data_root}/models"

class AggregationLayer(Layer):
    def __init__(self, **kwargs):
        super(AggregationLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)

    def get_config(self):
        config = super(AggregationLayer, self).get_config()
        return config

# TODO: revise this
class MedicalQADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids, attention_mask = self.data[idx]
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(input_ids)
        }

# Load JSON data
def load_json_data(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data


def estimate_goodness(question:dict, answer:dict):
    ret = 0
    if question.get('comments', 0) > 2 or len(answer.get('replies', [])) > 0:
        ret = 1
    # if question.get('likes', 0) > 0:
    #     ret += 1
    if answer.get('votes', 0) > 0:
        ret = 2 # min(2, ret+2)
    # if len(get_censored_words_q_a(question, answer)) > 0:
    #     ret = max(0, ret-1)
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

def load_my_data(filter:Callable=None):
    # Load and filter data
    labeled_data_file = f'{data_root}/annotation/questions/_questions_annotated.json'
    unlabeled_data_file = f'{data_root}/annotation/questions/_questions_with_answers.json'

    # Load and parse data
    data_labeled = load_json_data(labeled_data_file)
    data_unlabeled = load_json_data(unlabeled_data_file)
    # combine the data 
    data = {p[0]:p[1] if not p[0] in data_labeled else data_unlabeled[p[0]]
            for p in data_unlabeled.items()}

    #return filter_good_answers(data)
    return data

