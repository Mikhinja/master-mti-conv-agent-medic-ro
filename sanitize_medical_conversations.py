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
# pip install nltk
import nltk
from nltk.corpus import stopwords

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
errors_file = f'{logs_root}/sanitize_errors_{timestamp}.txt'
log_file = f'{logs_root}/sanitize_log_{timestamp}.txt'

in_data_folder = f"{data_root}/raw"
# in_data_folder = f"{data_root}/raw_old_backup" # TODO: remove this TEMP DEBUG
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
    #'kp***ax': 'klabax', # is this right??

    # Romanian words censored by accident
    'amp**a': 'ampula',
    'cop**a': 'copula',
    'manip**a': 'manipula',
    'pop**a': 'popula', # and derived, like 'popular' and 'populat'
    'scap**a': 'scapula',
    'stip**a': 'stipula',

    'ac***a': 'avand', # empirically deduced from several contexts
}

curse_words = {
    # these are curse words - they are unsafe to replace due to inconsistencies and inaccuracy of results
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
        ret = ret.replace(c, replacements[c])
    ret = "".join(c if c.isprintable() else '_' for c in ret)
    return ret

def break_merged_words(text:str)->str:
    ret = re.sub(r'\B[;]\B', '; ', text)
    # TODO: replace the below rule with separating by stop words
    #ret = re.sub(r'(\D)[.](\D)', '\g<1>. \g<2>', ret)
    return ret

def sanitize_pass(q:dict, func:Callable[[str], str]):
    q['title'] = func(q['title'])
    q['question'] = func(q['question'])
    for a in q['answers']:
        if a['text']:
            a['text'] = func(a['text'])
        else:
            # this is the weird case where a comment was deleted, but someone replied to it
            #   and the reply is not deleted
            a['deleted'] = True
        for r in a['replies']:
            r['text'] = func(r['text'])

def get_censored_words(raw_q:dict)->list[str]:
    ret = []
    if raw_q:
        test = re.compile(r'\b\w+[*]+\w+\b')
        ret += test.findall(raw_q['title'])
        ret += test.findall(raw_q['question'])
        for a in raw_q['answers']:
            if a['text']:
                ret += test.findall(a['text'])
            for r in a['replies']:
                ret += test.findall(r['text'])
    return ret

def salvage_redacted(q:dict)->bool:
    ret = False
    if q:
        ret |= any([w for w in salvage_redacted_words if w in q['title']])
        for word, replacement in salvage_redacted_words.items():
            q['title'] = q['title'].replace(word, replacement)
        ret |= any([w for w in salvage_redacted_words if w in q['question']])
        for word, replacement in salvage_redacted_words.items():
            q['question'] = q['question'].replace(word, replacement)
        for a in q['answers']:
            if a['text']:
                ret |= any([w for w in salvage_redacted_words if w in a['text']])
                for word, replacement in salvage_redacted_words.items():
                    a['text'] = a['text'].replace(word, replacement)
            for r in a['replies']:
                ret |= any([w for w in salvage_redacted_words if w in r['text']])
                for word, replacement in salvage_redacted_words.items():
                    r['text'] = r['text'].replace(word, replacement)
    return ret

def get_all_words_text(text:str, word_bag:dict):
    for word in text.split():
        if not word in word_bag:
            word_bag[word] = 1
        else:
            word_bag[word] += 1
def get_all_words(q:dict, word_bag:dict):
    get_all_words_text(q['title'], word_bag)
    get_all_words_text(q['question'], word_bag)
    for a in q['answers']:
        if a['text']:
            get_all_words_text(a['text'], word_bag)
        for r in a['replies']:
            get_all_words_text(r['text'], word_bag)

def my_extract_keywords(q:dict, stopwords:str)->list[str]:
    ret = q['title'].split() + q['question'].split()
    ret = [w for w in ret if w not in stopwords]
    return list(set(ret))

if clean_data_folder and os.path.exists(out_data_folder):
    shutil.rmtree(out_data_folder)
if os.path.exists(errors_file):
    os.remove(errors_file)
if os.path.exists(log_file):
    os.remove(log_file)

raw_cats = os.listdir(in_data_folder)
censored_words = {}
cats_total_num = len(raw_cats)
cats_num = 0

stats = {
    'total words': 0,
    'total questions': 0,
    'total questions with answers': 0,
    'total questions with doctor answer': 0,
    'total replies': 0,
}

started_time = datetime.now()
print(f'Started sanitizing at {started_time.strftime("%Y-%m-%d %H:%M:%S")}')
with open(log_file, "a+") as fp:
    print(f'Started sanitizing at {started_time.strftime("%Y-%m-%d %H:%M:%S")}', file=fp)

nltk.download('stopwords')
stopwords_ro = [sanitize_str_1(w) for w in stopwords.words('romanian')]

all_words = {}
dup_name_num = 0

print()
for raw_cat in raw_cats:
    in_raw_cat_folder = os.path.join(in_data_folder, raw_cat)
    raw_questions = os.listdir(in_raw_cat_folder)
    for raw_question_name in raw_questions:
        raw_question_file = os.path.join(in_raw_cat_folder, raw_question_name)
        raw_question = {}
        with open(raw_question_file, "r") as fp:
            raw_question = json.load(fp)
        
        # preserve the raw question to inspect during debugging
        question = raw_question.copy()

        sanitize_pass(question, sanitize_str_1)
        sanitize_pass(question, break_merged_words)
        
        censored_words_i = get_censored_words(question)
        salvaged_words = salvage_redacted(question)
        question['words replaced'] = salvaged_words
        question['words censored'] = len(censored_words_i)
        question['keywords'] = my_extract_keywords(question, stopwords=stopwords_ro)

        stats['total questions'] += 1
        if len(question['answers']) > 0:
            stats['total questions with answers'] += 1
            stats['total replies'] += len(question['answers']) + sum((len(a['replies']) for a in question['answers']))
            if any((a['is_medic'] for a in question['answers'])):
                stats['total questions with doctor answer'] += 1

        if censored_words_i:
            censored_words_icount = {w: censored_words_i.count(w)
                for w in set(censored_words_i)}
            for censored in censored_words_icount:
                if censored in censored_words:
                    censored_words[censored] += censored_words_icount[censored]
                else:
                    censored_words[censored] = censored_words_icount[censored]

        get_all_words(question, all_words)

        # save one by one
        dup_name_num += save_all_as_json(path=f"{out_data_folder}/{raw_cat}",
                                         questions={question['title']: question},
                                         errors_file=errors_file)
    cats_num += 1
    overall_time = datetime.now()-started_time
    print(f'Category raw {raw_cat[:50]:>50} [{cats_num:>4} / {cats_total_num:>4}] {(100*cats_num/cats_total_num):>5.1f}% | time {timedelta_str(overall_time)}'
          , end='\r'
          )
print()

# order words by count descending
all_words = dict(sorted(all_words.items(), key=lambda pair: pair[1], reverse=True))
with open(os.path.join(out_data_folder, 'words_count.json'), 'w') as fp:
    json.dump(all_words, fp=fp, indent=2)

# skip stopwords
for word, count in all_words.items():
    if word not in stopwords_ro:
        stats['total words'] += count

censored_words = dict(sorted(censored_words.items(), key=lambda pair: pair[1], reverse=True))
with open('./censored_words.json', 'w') as fp:
    json.dump(censored_words, fp=fp, indent=2)

# TODO: either do something with this or remove it
uncensored_words = {
    censored: censored
    for censored in censored_words
}
for censored in uncensored_words:
    found = next((word for word in salvage_redacted_words if word in censored), None)
    if found:
        uncensored_words[censored] = uncensored_words[censored].replace(found, salvage_redacted_words[found])

print(f'Ended at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ( {timedelta_str(datetime.now()-started_time)} )')
print(f'num censored words {len(censored_words):>4}')
print(f'stats: {json.dumps(stats, indent=2)}')
with open(log_file, "a+") as fp:
    print(f'Ended at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ( {timedelta_str(datetime.now()-started_time)} )', file=fp)
    print(f'num censored words {len(censored_words):>4}', file=fp)
    print(f'stats: {json.dumps(stats, indent=2)}', file=fp)