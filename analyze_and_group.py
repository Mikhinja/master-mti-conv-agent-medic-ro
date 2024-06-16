from datetime import datetime, timedelta
import shutil
import nltk
from nltk.corpus import stopwords

from control_vars import *
from common_utils import *

import requests

# LanguageTool API base URL
API_URL = "https://languagetool.org/api/v2/check"

# Function to check spelling using LanguageTool
def check_spelling(text, language="ro", enabledCategories='TYPOS', disabledCategories='CASING,MISC,TYPOGRAPHY'):
  # Prepare request data
  data = {
      "text": text,
      "language": language,
      'enabledCategories': enabledCategories,
      'disabledCategories': disabledCategories,
  }

  # Send POST request
  response = requests.post(API_URL, data=data)

  # Check for successful response
  if response.status_code == 200:
    # Parse JSON response
    response_data = response.json()
    # Extract spelling errors (matches)
    matches = response_data.get("matches", [])
    return matches
  else:
    # Handle error
    print(f"Error: API request failed with status code {response.status_code}")
    return []

# currently not used
MIN_KEYWORDS = 7
def my_extract_keywords(q:dict, stopwords:str, word_count:dict[str, int])->list[str]:
    ret = [w for c in q['category'] for w in c.split() if w not in stopwords]
    if len(ret) < MIN_KEYWORDS:
        ret = ret + [w for w in q['title'].split() if w not in stopwords]
    if len(ret) < MIN_KEYWORDS:
        # add only the least frequent words in the question
        ret = ret + sorted(q['question'].split(), key=lambda w: word_count[w])[:MIN_KEYWORDS]
    ret = list(set([w for w in ret if w not in stopwords
                        and 'https' not in w
                        and not is_num(w)
                        and not has_punctuation(w)
                        and censored_test.search(w) is None
                        and w.strip()]))
    ret = sorted(ret, key=lambda w: word_count[w], reverse=True)
    return ret

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
errors_file = f'{logs_root}/analize_sanitize_errors_{timestamp}.txt'
log_file = f'{logs_root}/analize_sanitize_log_{timestamp}.txt'

in_data_folder = f"{data_root}/sanitized1"

words_count_file = f'{in_data_folder}/words_count.json'
categories_file = f'{in_data_folder}/categories.json'

out_data_folder = f"{data_root}/analyzed2"
out_salvage_typos_file = f'{out_data_folder}/salvage_typos.json'
if clean_data_folder and os.path.exists(out_data_folder):
    shutil.rmtree(out_data_folder)
os.makedirs(out_data_folder, exist_ok=True)

started_time = datetime.now()
print(f'Started analyzing and sanitizing(2) at {started_time.strftime("%Y-%m-%d %H:%M:%S")}')
with open(log_file, "a+") as fp:
    print(f'Started analyzing and sanitizing(2) at {started_time.strftime("%Y-%m-%d %H:%M:%S")}', file=fp)

# read metadata

all_words = {}
if os.path.exists(words_count_file):
    with open(words_count_file, 'r') as fp:
        all_words = json.load(fp)

categories = {}
if os.path.exists(categories_file):
    with open(categories_file, 'r') as fp:
        categories = json.load(fp)

categories = dict(sorted(categories.items(), key=lambda pair: pair[0]))

nltk.download('stopwords')
stopwords_ro = set([sanitize_diacritics(w) for w in stopwords.words('romanian')])

all_words_nonstop = {i[0]: i[1] for i in all_words.items() if i[0] not in stopwords_ro and not is_punctuation(i[0])}
first_10_nonstop_words = {k: all_words_nonstop[k] for k in list(all_words_nonstop.keys())[:10]}
print(f'First 10 non-stopwords: {first_10_nonstop_words}')

typos_salvage = {}
if TRY_TO_SALVAGE_TYPOS:
    # try to salvage some typos
    # rules: look through words with occurrences 2, 3, or 4 and
    #   * are not numbers -> id_num(text)
    #   * are not censored -> censored_test.search(text) is None
    #   * are not links -> text.contains('')
    #   * does not have punctuation within
    #   * the tool used gives unambiguous replacement
    words_234 = [pair[0] for pair in all_words_nonstop.items() if pair[1] in [2,3,4]]
    words_queried = 0
    word_at = 0
    words_to_query = len(words_234)
    for word in words_234:
        if not is_num(word) and ('https' not in word) and (censored_test.search(word) is None) and not has_punctuation(word):
            if matches := check_spelling(word):
                # only look at full matches
                matches = list(set([sanitize_diacritics(r['value']) 
                        for m in matches if m['length'] == len(word)
                        for r in m['replacements']]))
                if len(matches) == 1:
                    # only one proposal found, unambiguous
                    typos_salvage[word] = matches[0]
            words_queried += 1
        word_at += 1
        overall_time = datetime.now()-started_time
        print(f'Querying for typos to salvage  [{word_at:>6} / {words_to_query:>6}] {(100*word_at/words_to_query):>5.1f}% | time {timedelta_str(overall_time)}'
                , end='\r'
                )
    # don't let it go to waste
    with open(out_salvage_typos_file, "w") as fp:
        json.dump(typos_salvage, indent=2, fp=fp)



dup_name_num = 0
all_questions = {}
raw_question_files = os.listdir(f'{in_data_folder}/questions')
questions_num = 0
questions_total_num = len(raw_question_files)
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
    
    #corrections_title = check_spelling(question['title'])
    #corrections_question = check_spelling(question['question'])

    question['keywords'] = my_extract_keywords(q=question, stopwords=stopwords_ro, word_count=all_words)

    if question['id'] in all_questions:
        print(f'Duplicate question: {question["id"]}')
        with open(errors_file, "a+") as fp:
            print(f'Duplicate question: {question["id"]}', file=fp)
    all_questions[question['id']] = question
    
    # # save one by one
    # dup_name_num += save_all_as_json(path=f"{out_data_folder}/questions",
    #                                     questions={question['id']: question},
    #                                     errors_file=errors_file)
    
    questions_num += 1
    overall_time = datetime.now()-started_time
    print(f'Loading questions  [{questions_num:>6} / {questions_total_num:>6}] {(100*questions_num/questions_total_num):>5.1f}% | time {timedelta_str(overall_time)}'
            , end='\r'
            )
print()

questions_with_answers = {k_v_pair[0]:k_v_pair[1] for k_v_pair in all_questions.items()
                            if len(k_v_pair[1]['answers']) > 0} # is this different than k_v_pair[1]['comments'] ?
print(f'Qs with answers {len(questions_with_answers):>5}')

questions_with_answers_2a = {k_v_pair[0]:k_v_pair[1] for k_v_pair in all_questions.items()
                            if len(k_v_pair[1]['answers']) > 1} # is this different than k_v_pair[1]['comments'] ?
print(f'Qs with 2+ answers {len(questions_with_answers_2a):>5}')

questions_with_doc_answers = {k_v_pair[0]:k_v_pair[1] for k_v_pair in all_questions.items()
                                if len(k_v_pair[1]['answers']) > 0 and k_v_pair[1]['has_doc_answer']}
print(f'Qs with doc answers {len(questions_with_doc_answers):>5} ({(100*len(questions_with_doc_answers)/len(questions_with_answers)):>5.1f}%)')

questions_with_doc_answers_2a = {k_v_pair[0]:k_v_pair[1] for k_v_pair in all_questions.items()
                                if len(k_v_pair[1]['answers']) > 1 and k_v_pair[1]['has_doc_answer']}
print(f'Qs with doc answers and at least 2 answers {len(questions_with_doc_answers_2a):>5} ({(100*len(questions_with_doc_answers_2a)/len(questions_with_answers_2a)):>5.1f}%)')

# should I only look through questions_with_doc_answers here?
questions_with_liked_doc_answers = {k_v_pair[0]:k_v_pair[1] for k_v_pair in all_questions.items()
                                    if any((answer for answer in k_v_pair[1]['answers']
                                            if (answer['is_medic'] and answer['votes'] > 0)
                                                or any((reply for reply in answer['replies']
                                                        if reply['is_medic'] and reply['votes'] > 0))))}
print(f'Qs with liked doc answers {len(questions_with_liked_doc_answers):>5} ({(100*len(questions_with_liked_doc_answers)/len(questions_with_doc_answers)):>5.1f}%)')

questions_with_liked_doc_answers_2a = {k_v_pair[0]:k_v_pair[1] for k_v_pair in all_questions.items()
                                    if len(k_v_pair[1]['answers']) > 1
                                        and any((answer for answer in k_v_pair[1]['answers']
                                            if (answer['is_medic'] and answer['votes'] > 0)
                                                or any((reply for reply in answer['replies']
                                                        if reply['is_medic'] and reply['votes'] > 0))))}
print(f'Qs with liked doc answers and at least 2 answers {len(questions_with_liked_doc_answers_2a):>5} ({(100*len(questions_with_liked_doc_answers_2a)/len(questions_with_doc_answers)):>5.1f}%)')

questions_with_non_doc_answers = {k_v_pair[0]:k_v_pair[1] for k_v_pair in all_questions.items()
                                    if len(k_v_pair[1]['answers']) > 0 and not k_v_pair[1]['has_doc_answer']}
print(f'Qs with only non-doc answers {len(questions_with_non_doc_answers):>5}')

questions_with_non_doc_answers_2a = {k_v_pair[0]:k_v_pair[1] for k_v_pair in all_questions.items()
                                    if len(k_v_pair[1]['answers']) > 1 and not k_v_pair[1]['has_doc_answer']}
print(f'Qs with only non-doc answers and at least 2 answers {len(questions_with_non_doc_answers_2a):>5}')

# need to look through all_questions again to capture non-doc replies to doc answers that are liked (voted)
questions_with_liked_non_doc_answers = {k_v_pair[0]:k_v_pair[1] for k_v_pair in all_questions.items()
                                        if any((answer for answer in k_v_pair[1]['answers']
                                                if (not answer['is_medic'] and answer['votes'] > 0)
                                                    or any((reply for reply in answer['replies']
                                                            if not reply['is_medic'] and reply['votes'] > 0))))}
print(f'Qs with liked non-doc answers {len(questions_with_liked_non_doc_answers):>5} ({(100*len(questions_with_liked_non_doc_answers)/len(questions_with_non_doc_answers)):>5.1f}%)')

questions_with_likes = {k_v_pair[0]:k_v_pair[1] for k_v_pair in all_questions.items()
                            if k_v_pair[1]['likes'] > 0}
print(f'Qs with likes {len(questions_with_likes):>5}')

questions_with_likes_and_answers = {k_v_pair[0]:k_v_pair[1] for k_v_pair in all_questions.items()
                            if len(k_v_pair[1]['answers']) > 0 and k_v_pair[1]['likes'] > 0}
print(f'Qs with likes and answers {len(questions_with_likes_and_answers):>5}')

questions_with_liked_answers = {k_v_pair[0]:k_v_pair[1] for k_v_pair in all_questions.items()
                                        if any((answer for answer in k_v_pair[1]['answers'] if answer['votes'] > 0
                                             or any((reply for reply in answer['replies'] if reply['votes'] > 0))))}
print(f'Qs with liked answers {len(questions_with_liked_answers):>5} ({(100*len(questions_with_liked_answers)/len(questions_with_answers)):>5.1f}%)')

stats = {
    'questions with answers': len(questions_with_answers),
    'questions with 2+ answers': len(questions_with_answers_2a),
    'questions with likes': len(questions_with_likes),
    'questions with likes and answers': len(questions_with_likes_and_answers),
    'questions with liked answers': len(questions_with_liked_answers),
    'questions with answers from doctors': len(questions_with_doc_answers),
    'questions with answers from doctors and 2+ answers': len(questions_with_doc_answers_2a),
    'questions with liked answers from doctors': len(questions_with_liked_doc_answers),
    'questions with liked answers from doctors and 2+ answers': len(questions_with_liked_doc_answers_2a),
    'questions with answers only from non-doctors': len(questions_with_non_doc_answers),
    'questions with answers only from non-doctors and 2+ answers': len(questions_with_non_doc_answers_2a),
    'questions with liked answers from non-doctors': len(questions_with_liked_non_doc_answers),
}
with open(f'{out_data_folder}/stats.json', 'w') as fp:
    json.dump(stats, indent=2, fp=fp)

print('saving...', end='\r')
num_duped = 0
num_duped += save_all_as_json(path=f"{out_data_folder}/questions",
                              questions=questions_with_answers,
                              errors_file=errors_file,
                              duplicate_filename_error=True)
print(f'Saved. Duplicate file names: {num_duped:>5}.')
print('saving info about questions...')
with open(f'{out_data_folder}/questions_with_answers.json', 'w') as fp:
    json.dump(list(questions_with_answers.keys()), indent=2, fp=fp)
with open(f'{out_data_folder}/questions_with_answers_2a.json', 'w') as fp:
    json.dump(list(questions_with_answers_2a.keys()), indent=2, fp=fp)
with open(f'{out_data_folder}/questions_with_likes_and_answers.json', 'w') as fp:
    json.dump(list(questions_with_likes_and_answers.keys()), indent=2, fp=fp)
with open(f'{out_data_folder}/questions_with_liked_answers.json', 'w') as fp:
    json.dump(list(questions_with_liked_answers.keys()), indent=2, fp=fp)
with open(f'{out_data_folder}/questions_with_doc_answers.json', 'w') as fp:
    json.dump(list(questions_with_doc_answers.keys()), indent=2, fp=fp)
with open(f'{out_data_folder}/questions_with_doc_answers_2a.json', 'w') as fp:
    json.dump(list(questions_with_doc_answers_2a.keys()), indent=2, fp=fp)
with open(f'{out_data_folder}/questions_with_liked_doc_answers.json', 'w') as fp:
    json.dump(list(questions_with_liked_doc_answers.keys()), indent=2, fp=fp)
with open(f'{out_data_folder}/questions_with_liked_doc_answers_2a.json', 'w') as fp:
    json.dump(list(questions_with_liked_doc_answers_2a.keys()), indent=2, fp=fp)
with open(f'{out_data_folder}/questions_with_non_doc_answers.json', 'w') as fp:
    json.dump(list(questions_with_non_doc_answers.keys()), indent=2, fp=fp)
with open(f'{out_data_folder}/questions_with_non_doc_answers_2a.json', 'w') as fp:
    json.dump(list(questions_with_non_doc_answers_2a.keys()), indent=2, fp=fp)
with open(f'{out_data_folder}/questions_with_liked_non_doc_answers.json', 'w') as fp:
    json.dump(list(questions_with_liked_non_doc_answers.keys()), indent=2, fp=fp)


print(f'Ended at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ( {timedelta_str(datetime.now()-started_time)} )')
with open(log_file, "a+") as fp:
    print(f'Ended at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ( {timedelta_str(datetime.now()-started_time)} )', file=fp)
    print(f'stats: {json.dumps(stats, indent=2)}', file=fp)
