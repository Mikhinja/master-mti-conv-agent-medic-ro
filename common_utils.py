
from datetime import timedelta
import json
import os
import re
from unidecode import unidecode

def timedelta_str(dt:timedelta)->str:
    s = dt.seconds
    return f'{s//3600:>2}:{(s//60)%60:>2}:{s%60:>2}s'

def sanitize_for_filename(q_id:str)->str:
    sanitized_title = re.sub('[<>:"/\\\\|?*\t\r\n]',' ', q_id).replace('  ',' ').strip()
    sanitized_title = "".join(c if c.isprintable() else '_' for c in sanitized_title)
    return sanitized_title

# my attempt at empirically determining and doing the replacements
diacritics = {
      'Ă': 'A', # 258
      'ă': 'a', # 259
      'Â': 'A', # 194
      'â': 'a', # 226

      'Î': 'I', # 522
      'î': 'i', # 523
      'Ș': 'S', # 536
      'ș': 's', # 537
      'Ş': 'S', # 350
      'ş': 's', # 351
      'Ț': 'T', # 538
      'ț': 't', # 539
}
def sanitize_diacritics(text:str)->str:
    ret = text.lower()
    # for c in diacritics:
    #     ret = ret.replace(c, diacritics[c])
    ret = unidecode(ret)
    ret = "".join(c if c.isprintable() else '_' for c in ret)
    return ret

def get_censored_words(raw_q:dict)->list[str]:
    ret = []
    if raw_q:
        ret += censored_test.findall(raw_q['title'])
        ret += censored_test.findall(raw_q['question'])
        for a in raw_q['answers']:
            if a['text']:
                ret += censored_test.findall(a['text'])
            for r in a['replies']:
                ret += censored_test.findall(r['text'])
    return ret

def get_censored_words_q_a(q:dict, a:dict)->list[str]:
    ret = []
    if q:
        ret += censored_test.findall(q['title'])
        ret += censored_test.findall(q['question'])
        if a and a['text']:
            ret += censored_test.findall(a['text'])
    return ret

re_punctuation = re.compile(r'^[,.()+-_:?!&^%$€\'"/\\]+$')
re_has_punctuation = re.compile(r'[,.()+-_:?!&^%$€\'"/\\]+')
def is_punctuation(text:str)->bool:
    return re_punctuation.search(text) is not None
def has_punctuation(text:str)->bool:
    return re_has_punctuation.search(text) is not None
re_num = re.compile(r'^(\d+)$')
def is_num(text:str)->bool:
    return re_num.search(text) is not None

censored_test = re.compile(r'\b\w+[*]+\w+\b')

def save_all_as_json(path:str, questions:dict, errors_file:str, duplicate_filename_error:bool=False)->int:
    os.makedirs(path, exist_ok=True)
    dup_names_num = 0
    for q_id in questions:
        sanitized_title = sanitize_for_filename(q_id)
        filename = f'{path}/{sanitized_title[:100]}.json'
        if os.path.exists(filename):
            dup_names_num += 1
            first_filename = filename
            while os.path.exists(filename):
                filename = filename.replace('.json', '_.json')
            if duplicate_filename_error:
                with open(errors_file, "a+") as fp:
                    print(f"ERROR: duplicate filename {first_filename}, saved as {filename}", file=fp)
        try:
            with open(filename, "w") as fp:
                json.dump(questions[q_id], fp, indent=2)
        except Exception as exc:
            #print(f"ERROR: could not write file {filename} : {exc}")
            with open(errors_file, "a+") as fp:
                print(f"ERROR: could not write file {filename} : {exc}", file=fp)
    return dup_names_num


def get_num_from_question_id(q_id:str)->int:
    return int(re.search(r'(\d+)$', q_id).group(1))
