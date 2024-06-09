# Prerequisites
# !pip install chardet
# !pip install numpy
# !pip install requests
# !pip install bs4
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re
import os
import shutil
import json
from control_vars import *
from common_utils import *

# Prepare directory

#!mkdir Data

base_url = "https://www.sfatulmedicului.ro/comunitate/discutii-teme-medicale"

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
errors_file = f'{logs_root}/scrape_errors_{timestamp}.txt'
log_file = f'{logs_root}/scrape_log_{timestamp}.txt'
data_folder = f"{data_root}/raw"

# sanitize URL - because some are not in canonical form
def sanitize_url(url:str)->str:
   if not url.startswith('https:'): # they start with //www.
       url = 'https:' + url
   return url

# Function to fetch and parse a single page of discussions
def fetch_page(url:str, retries:int=3):
    try:
        response = requests.get(sanitize_url(url))
        if response.status_code != 200:
            if response.status_code >= 500 and retries>0:
                return fetch_page(url, retries=retries-1)
            #print(f"ERROR: Could not get page {url}: status code {response.status_code}", file=fp)
            with open(errors_file, "a+") as fp:
                print(f"ERROR: Could not get page {url}: status code {response.status_code}", file=fp)
            return None
        return BeautifulSoup(response.text, 'html.parser')
    except Exception:
        if retries>0:
            # this hides some server errors
            return fetch_page(url, retries=retries-1)
        else:
            raise

def get_all_categories(num=-1) -> dict:
  ret = {}
  cats_li = [t for t in fetch_page(base_url).find_all('li')
             if not t.has_attr('class') and len(t.text) != 1
             and not (t.find('i') or t.text in ['Contul Meu', 'Comunitate', 'Discutii pe teme medicale'])]
  so_far = 0
  for cat_li in cats_li:
    cat_ = [c for c in cat_li.contents if not isinstance(c, str)]
    if not cat_ or 'href' not in cat_[0].attrs:
       continue
    cat = cat_[0]
    cat_name = cat.text.strip()
    ret[cat_name] = cat['href']
    so_far += 1
    if num > 0 and so_far > num:
      break
  return ret

def find_with_attr(where:BeautifulSoup, tag:str, attr:str) -> BeautifulSoup:
   return [a for a in where.find_all(tag) if a.has_attr(attr)]

def has_doc_answer(question:BeautifulSoup)->bool:
   return len([i for i in question.find_all('i') if 'raspuns medic' in i.next_sibling])>0

def get_num_likes(question:BeautifulSoup)->int:
   return int(question.find('i', class_='far fa-thumbs-up').next_sibling.strip())

def get_num_comments(question:BeautifulSoup)->int:
   return int(question.find('i', class_='far fa-comment').next_sibling.strip())

def get_questions_in_page(page:BeautifulSoup, cat:str) -> dict:
   # class of top div: "col-md-12 nopadding"s
   div_questions = page.find('div', class_="col-md-12 nopadding")
   question_divs = div_questions.find_all('div', class_="question")
   questions = { find_with_attr(question,'a', 'title')[0]['title']:   {
                    'category': cat,
                    'has_doc_answer': has_doc_answer(question),
                    'likes': get_num_likes(question),
                    'comments': get_num_comments(question),
                    'url': sanitize_url(find_with_attr(question,'a', 'title')[0]['href'])
                }
                for question in question_divs
            }
   return questions

def get_votes(div):
    vote_tag = div.find('b', class_='question_detail2_no')
    return int(vote_tag.get_text(strip=True)) if vote_tag else 0

def is_medic(div):
    return bool(div.find('span', class_='qAnswerMedic'))

# Function to convert relative time to absolute time
def convert_relative_time(time_tag)->str:
    if not time_tag:
        return None
    relative_time = time_tag.get_text(strip=True)
    current_time = datetime.now()
    relative_time = relative_time.replace(' un ', ' 1 ').replace(' o ', ' 1 ')
    if "acum" in relative_time:
        number = int(re.search(r'\d+', relative_time).group())
        if "ani" in relative_time:
            current_time = current_time - timedelta(days=number*365)
        elif "lună" in relative_time or "luni" in relative_time:
            current_time = current_time - timedelta(days=number*30)
        elif "zi" in relative_time or "zile" in relative_time:
            current_time = current_time - timedelta(days=number)
        elif "oră" in relative_time or "ore" in relative_time:
            current_time = current_time - timedelta(hours=number)
        elif "minut" in relative_time or "minute" in relative_time:
            current_time = current_time - timedelta(minutes=number)
    return current_time.strftime('%Y-%m-%d')

def get_comments_and_replies(soup, stats:dict=None):
    comments = []
    
    # Find the main div that contains the comments and replies
    answers_div = soup.find('div', id='questionAnswers')
    
    if not answers_div:
        return comments

    # Loop through all comments and replies
    comment = {}
    for div in answers_div.find_all('div', class_=['qAnswer clearfix', 'qAnswerReply clearfix']):
        # Extract the user, time, text, votes, and medic status of the comment
        user_tag = div.find('div', class_='qAnswerInfo clearfix').find('strong')
        time_tag = div.find('span', class_='qAnswerInfoTime')
        text_tag = div.find('p')

        is_comment = 'qAnswer' in div.get('class')
        if is_comment:
            if comment:
                comments.append(comment)
                if stats and stats['max replies'] < len(comment['replies']):
                    stats['max replies'] = len(comment['replies'])
            comment = {
                'user': user_tag.get_text(strip=True) if user_tag else None,
                'time': convert_relative_time(time_tag),
                'text': text_tag.get_text(strip=True) if text_tag else None,
                'votes': get_votes(div),
                'is_medic': is_medic(div),
                'replies': []
            }
            if comment['is_medic']:
                stats['medic answers'] += 1
        else:
            reply = {
                'user': user_tag.get_text(strip=True) if user_tag else None,
                'time': convert_relative_time(time_tag),
                'text': text_tag.get_text(strip=True) if text_tag else None,
                'votes': get_votes(div),
                'is_medic': is_medic(div)
            }
            if reply['is_medic']:
                stats['medic answers'] += 1
            if not comment or 'replies' not in comment:
                # looks like there are some comments deleted because there are replies to nothing
                #   in this case just add an empty comment
                comment = {
                    'user': None,
                    'time': None,
                    'text': None,
                    'votes': 0,
                    'is_medic': False,
                    'replies': []
                }
            comment['replies'].append(reply)
        
    if comment:
        comments.append(comment)
        if stats and stats['max replies'] < len(comment['replies']):
            stats['max replies'] = len(comment['replies'])
    
    return comments

def process_question(questions:dict, title:str, stats:dict=None)->dict:
    question = questions[title]
    question_page = fetch_page(question['url'])
    question['title'] = title
    question['question'] = question_page.find('p', class_='').text
    if stats:
        if question['likes'] > 0:
            stats['questions liked'] += 1
            if stats['max question likes'] < question['likes']:
                stats['max question likes'] = question['likes']
        if stats['max comments'] < question['comments']:
            stats['max comments'] = question['comments']
    question['answers'] = get_comments_and_replies(question_page, stats)
    return question # is return really needed?

def get_questions_content(questions:dict, stats:dict=None):
   with ThreadPoolExecutor(max_workers=50) as executor:  # You can adjust the number of workers
        future_to_title = {executor.submit(process_question, questions, title, stats): title for title in questions}
        for future in as_completed(future_to_title):
            title = future_to_title[future]
            try:
                result = future.result() # is this still needed?
            except Exception as exc:
                #print(f"ERROR [get Q content]: {title} generated an exception: {exc}")
                with open(errors_file, "a+") as fp:
                    print(f"ERROR [get Q content]: {title} generated an exception: {exc}", file=fp)


def get_page_and_subsequent_pages(first_page_url:str)->list[BeautifulSoup]:
    first_page = fetch_page(first_page_url)
    pages = [first_page]

    total_pages_div = first_page.find('div', class_='total-pagini left')
    if total_pages_div:
        num_pages = int(re.search(r'Pag 1 din (\d+)', total_pages_div.text).group(1))
        
        urls = [f"{first_page_url}/pagina_{idx + 1}" for idx in range(1, num_pages)]
        with ThreadPoolExecutor(max_workers=30) as executor:  # Adjust the number of workers as needed
            future_to_url = {executor.submit(fetch_page, url): url for url in urls}
            
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    page = future.result()
                    if page is not None:
                        pages.append(page)
                except Exception as exc:
                    #print(f"ERROR [get pages]: {url} generated an exception: {exc}")
                    with open(errors_file, "a+") as fp:
                        print(f"ERROR [get pages]: {url} generated an exception: {exc}", file=fp)

    return pages

if clean_data_folder and os.path.exists(data_folder):
    shutil.rmtree(data_folder)
if os.path.exists(errors_file):
    os.remove(errors_file)
if os.path.exists(log_file):
    os.remove(log_file)

all_questions = {}
dup_name_num = 0

stats = {
    'questions liked': 0,
    'max question likes': 0,
    'max comments': 0,
    'max replies': 0,
    'medic answers': 0,
}

started_time = datetime.now()

print(f'Started scraping from https://www.sfatulmedicului.ro/comunitate at {started_time.strftime("%Y-%m-%d %H:%M:%S")}')
with open(log_file, "a+") as fp:
    print(f'Started scraping from https://www.sfatulmedicului.ro/comunitate at {started_time.strftime("%Y-%m-%d %H:%M:%S")}', file=fp)
cats = get_all_categories()

cats_done_num = 0
cats_num = len(cats)
for cat in cats:
    cat_time_start = datetime.now()
    cat_pages = get_page_and_subsequent_pages(cats[cat])
    cat_questions = {}
    for cat_page in cat_pages:
        cat_questions |= get_questions_in_page(cat_page, cat)
    
    # call this function for all questions to parallelize better
    get_questions_content(cat_questions, stats)

    # save incrementally
    dup_name_num += save_all_as_json(path=f"{data_folder}/{cat}", questions=cat_questions, errors_file=errors_file)
    cat_done_time = datetime.now() - cat_time_start
    overall_time = datetime.now() - started_time

    all_questions |= cat_questions
    cats_done_num += 1
    print(f'Categories {cats_done_num:>4} / {cats_num:>4} [ {(100*cats_done_num/cats_num):>5.1f}% ] | {len(all_questions):>7} total Qs | time {timedelta_str(cat_done_time)} [{timedelta_str(overall_time)}]'
          , end='\r')
    with open(log_file, "a+") as fp:
        print(f'{cat:>80} [{len(cat_questions):>4}] | {len(all_questions):>7} total | time {timedelta_str(cat_done_time)} [{timedelta_str(overall_time)}]'
            , file=fp)
print()

print(f'Ended at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ( {timedelta_str(datetime.now()-started_time)} )')
print(f'num questions: {len(all_questions):>7}, total duplicate titles {dup_name_num:>7}')
print(f'stats: {json.dumps(stats, indent=2)}')
with open(log_file, "a") as fp:
    print(f'Ended at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ( {timedelta_str(datetime.now()-started_time)} )', file=fp)
    print(f'num questions: {len(all_questions):>7}, total duplicate titles {dup_name_num:>7}', file=fp)
    print(f'stats: {json.dumps(stats, indent=2)}', file=fp)
