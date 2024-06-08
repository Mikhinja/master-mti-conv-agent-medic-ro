from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import requests
from datetime import datetime, timedelta
import re
import os
import shutil
import json
from typing import Callable
from control_vars import *
from common_utils import *

errors_file = './sanitize_errors.txt'
log_file = f'{logs_root}/sanitize_log.txt'

in_data_folder = f"{data_root}/raw"
out_data_folder = f"{data_root}/sanitized1"

replacements = {
      'ă': 'a',
      'Ă': 'A',
      'â': 'a',
      'Â': 'A',
      'î': 'i',
      'Î': 'I',
      'ș': 's',
      'Ș': 'S',
      'ț': 't',
      'Ț': 'T',
}

# based on this list https://ro.wiktionary.org/wiki/Categorie:Cuvinte_vulgare_%C3%AEn_rom%C3%A2n%C4%83
salvage_redacted_words = {
    # medications
    'kp***ax': 'klabax', # is this right??

    # these are curse words - lowest priority or even removed
    'c*******s': 'caca-m-as',
    'c*i': 'coi',
    'c*r': 'cur',
    'c***a': 'curva',
    'f*t': 'fut',
    'f****i': 'futu-i',
    'g****a': 'gaoaza',
    'g****r': 'gaozar',
    'm**e': 'muie',
    'm***t': 'muist',
    'm****a': 'muista',
    'p***a': 'pizda',
    'p*****r': 'poponar',
    'p**a': 'pula',
}

def sanitize_str_1(text:str)->str:
    ret = text.lower()
    for c in replacements:
        ret.replace(c, replacements[c])
    return ret

def sanitize_pass(q:dict, func:Callable[[str], str]):
    q['title'] = func(q['title'])
    q['question'] = func(q['question'])
    for a in q['answers']:
        a['text'] = func(a['text'])
        for r in a['replies']:
            r = func(r['text'])

def get_censored_words(raw_q:dict)->list[str]:
    ret = []
    if raw_q:
        test = re.compile(r'\b\w+[*]+\w+\b')
        ret += test.findall(raw_q['title'])
        ret += test.findall(raw_q['question'])
        for a in raw_q['answers']:
            ret += test.findall(a['text'])
            for r in a['replies']:
                ret += test.findall(r['text'])
    return ret

raw_cats = os.listdir(in_data_folder)
censored_words:set[str] = set()
cats_total_num = len(raw_cats)
cats_num = 0

started_time = datetime.now()
print(f'Started sanitizing at {started_time.strftime("%Y-%m-%d %H:%M:%S")}')

print()
for raw_cat in raw_cats:
    in_raw_cat_folder = os.path.join(in_data_folder, raw_cat)
    raw_questions = os.listdir(in_raw_cat_folder)
    for raw_question_name in raw_questions:
        raw_question_file = os.path.join(in_raw_cat_folder, raw_question_name)
        raw_question = {}
        with open(raw_question_file, "r") as fp:
            raw_question = json.load(fp)
        sanitize_pass(raw_question, sanitize_str_1)
        censored_words_i = get_censored_words(raw_question)
        if censored_words_i:
            #print(f'{raw_question_file}: {raw_question}')
            censored_words |= set(censored_words_i)
    cats_num += 1
    print(f'Category raw {raw_cat[:50]:>50} [ {cats_num:>4} / {cats_total_num:>4}] ', end='\r')
print()

print(f'Ended at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ( {timedelta_str(datetime.now()-started_time)} )')

uncensored_words = {
    censored: censored
    for censored in censored_words
}
for censored in uncensored_words:
    found = next((word for word in salvage_redacted_words if word in censored), None)
    if found:
        uncensored_words[censored] = uncensored_words[censored].replace(found, salvage_redacted_words[found])
print(f'num censored words {len(censored_words):>4}')
