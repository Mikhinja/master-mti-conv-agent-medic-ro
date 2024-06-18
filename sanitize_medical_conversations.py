from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import requests
from datetime import datetime, timedelta
import re
import os
import shutil
import json
from typing import Callable
# pip install nltk
import nltk
from nltk.corpus import stopwords

from control_vars import *
from common_utils import *

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
errors_file = f'{logs_root}/sanitize_errors_{timestamp}.txt'
log_file = f'{logs_root}/sanitize_log_{timestamp}.txt'

in_data_folder = f"{data_root}/raw"
# in_data_folder = f"{data_root}/raw_old_backup" # TODO: remove this TEMP DEBUG
out_data_folder = f"{data_root}/sanitized1"
out_words_count_file = f'{out_data_folder}/words_count.json'
out_categories_file = f'{out_data_folder}/categories.json'
out_censored_words_file = f'{out_data_folder}/censored_words.json'

# these are not strict, meaning they can be in the middle of other words, which in Romanian usually means
#   word derivations, for example 'populat' means populated and adding '-ie' at the end means population
salvage_redacted_words = {
    # medications
    'kp***ax': 'klabax', # confirmed with a medic

    # Romanian words censored by accident
    'amp**a': 'ampula',
    'cop**a': 'copula',
    'manip**a': 'manipula',
    'pop**a': 'popula', # and derived, like 'popular' and 'populat'
    'scap**a': 'scapula',
    'stip**a': 'stipula',

    # empirically deduced from several contexts
    'ac***a': 'avand',
    # 'sp***a': 'stricata' or 'speriata' or others... - inconsistent replacement, depends highly on context
    'p***aa': 'prostia',
    'p***ae': 'prostie',
    # 'c***a': 'vand' or 'vanzare' or 'curva' or some medication - inconsistent replacement, depends highly on context
    'p***ancidenta': 'coincidenta',
    'p***atus': 'coitus',
    #'lac***aa': 'lavanda', # already covered by ac***a
    'inp***a': 'inmuia',
}

# based on this list https://ro.wiktionary.org/wiki/Categorie:Cuvinte_vulgare_%C3%AEn_rom%C3%A2n%C4%83
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

def break_merged_words(text:str)->str:
    ret = re.sub(r'\B[;]\B', '; ', text)
    # TODO: replace the below rule with separating by stop words
    #ret = re.sub(r'(\D)[.](\D)', '\g<1>. \g<2>', ret)
    return ret

def break_and_replace_excessive_punctuiation(text:str)->str:
    ret = re.sub(r'([,.!?;()\[\]\\\/\'" ])\1+', '\g<1>', text)

    # only in the middle of words should this character be removed
    ret = re.sub(r'([a-z])(\\)\2*([a-z])', '\g<1>\g<3>', ret)
    
    # separate punctuation: , . ! ? ; _ ( ) [ ] /
    # keep * for censored words, and + - : for context
    ret = re.sub(r'(\w)([,.!?;_()\[\]\/\'"])', '\g<1> \g<2>', ret)
    ret = re.sub(r'([,.!?;_()\[\]\/\'"])(\w)', '\g<1> \g<2>', ret)

    return ret

def sanitize_pass(q:dict, func:Callable[[str], str]):
    q['title'] = func(q['title'])
    q['question'] = func(q['question'])
    q['category'] = [func(c) for c in q['category']]
    for a in q['answers']:
        if a['text']:
            a['text'] = func(a['text'])
        else:
            # this is the weird case where a comment was deleted, but someone replied to it
            #   and the reply is not deleted
            a['deleted'] = True
        for r in a['replies']:
            r['text'] = func(r['text'])

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

if clean_data_folder and os.path.exists(out_data_folder):
    shutil.rmtree(out_data_folder)
if os.path.exists(errors_file):
    os.remove(errors_file)
if os.path.exists(log_file):
    os.remove(log_file)

raw_question_files = os.listdir(f'{in_data_folder}/questions')
censored_words = {}
questions_total_num = len(raw_question_files)
questions_num = 0

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
stopwords_ro = set([sanitize_diacritics(w) for w in stopwords.words('romanian')])

all_words = {}
dup_name_num = 0

# have a listing of posts by category
# not good to count overall questions because duplicates are allowed
#   since a question may have multiple categories
categories:dict[str, list[str]] = {}

print()
for raw_question_name in raw_question_files:
    if not raw_question_name.endswith('.json'):
        continue
    raw_question_file = f'{in_data_folder}/questions/{raw_question_name}'
    raw_question = {}
    with open(raw_question_file, "r") as fp:
        raw_question = json.load(fp)
    
    # preserve the raw question to inspect during debugging
    question = raw_question.copy()

    try:
        sanitize_pass(question, sanitize_diacritics)
        sanitize_pass(question, break_merged_words)
        sanitize_pass(question, break_and_replace_excessive_punctuiation)
        
        censored_words_i = get_censored_words(question)
        salvaged_words = salvage_redacted(question)
        question['words replaced'] = salvaged_words
        question['words censored'] = len(censored_words_i)

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
        dup_name_num += save_all_as_json(path=f"{out_data_folder}/questions",
                                            questions={question['id']: question},
                                            errors_file=errors_file)
        for cat in question['category']:
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(question['id'])

        questions_num += 1
        overall_time = datetime.now()-started_time
        print(f'Question  {raw_question_name[:50]:>50} [{questions_num:>6} / {questions_total_num:>6}] {(100*questions_num/questions_total_num):>5.1f}% | time {timedelta_str(overall_time)}'
            , end='\r'
            )
    except Exception as exc:
        with open(errors_file, "a+") as fp:
            print(f'ERROR: {raw_question_file} generated error {exc}', file=fp)
        pass
print()

# count the words in all categories
for cat in categories:
    get_all_words_text(cat, all_words)
    # should we weight them more?

# write category information, for reference
categories = dict(sorted(categories.items(), key=lambda pair: pair[0]))
with open(out_categories_file, 'w') as fp:
    json.dump(categories, fp=fp, indent=2)

# order words by count descending
all_words = dict(sorted(all_words.items(), key=lambda pair: pair[1], reverse=True))
with open(out_words_count_file, 'w') as fp:
    json.dump(all_words, fp=fp, indent=2)

# skip stopwords
for word, count in all_words.items():
    if word not in stopwords_ro:
        stats['total words'] += count

censored_words = dict(sorted(censored_words.items(), key=lambda pair: pair[1], reverse=True))
with open(out_censored_words_file, 'w') as fp:
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
