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

# Prepare directory

#!mkdir Data

base_url = "https://www.sfatulmedicului.ro/comunitate/discutii-teme-medicale"

# sanitize URL - because some are not in canonical form
def sanitize_url(url:str)->str:
   if not url.startswith('https:'):
       url = 'https:' + url
   return url

# Function to fetch and parse a single page of discussions
def fetch_page(url:str, retries:int=3):
    response = requests.get(sanitize_url(url))
    if response.status_code != 200:
        if response.status_code >= 500 and retries>0:
            return fetch_page(url, retries=retries-1)
        return None
    return BeautifulSoup(response.text, 'html.parser')

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
   questions = {find_with_attr(question,'a', 'title')[0]['title']:   {
                    'category': cat,
                    'has_doc_answer': has_doc_answer(question),
                    'likes': get_num_likes(question),
                    'comments': get_num_comments(question),
                    'url': find_with_attr(question,'a', 'title')[0]['href']
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
            return current_time - timedelta(days=number*365)
        elif "lună" in relative_time or "luni" in relative_time:
            return current_time - timedelta(days=number*30)
        elif "zi" in relative_time or "zile" in relative_time:
            return current_time - timedelta(days=number)
        elif "oră" in relative_time or "ore" in relative_time:
            return current_time - timedelta(hours=number)
        elif "minut" in relative_time or "minute" in relative_time:
            return current_time - timedelta(minutes=number)
    return current_time.strftime('%Y-%m-%d')

def get_comments_and_replies(soup):
    comments = []
    
    # Find the main div that contains the comments and replies
    answers_div = soup.find('div', id='questionAnswers')
    
    if not answers_div:
        return comments

    # Loop through all comments and replies
    for comment_div in answers_div.find_all('div', class_='qAnswer clearfix'):
        comment = {
            'user': None,
            'time': None,
            'text': None,
            'votes': 0,
            'is_medic': False,
            'replies': []
        }

        # Extract the user, time, text, votes, and medic status of the comment
        user_tag = comment_div.find('div', class_='qAnswerInfo clearfix').find('strong')
        time_tag = comment_div.find('span', class_='qAnswerInfoTime')
        text_tag = comment_div.find('p')

        if user_tag:
            comment['user'] = user_tag.get_text(strip=True)
        if time_tag:
            comment['time'] = convert_relative_time(time_tag)
        if text_tag:
            comment['text'] = text_tag.get_text(strip=True)
        comment['votes'] = get_votes(comment_div)
        comment['is_medic'] = is_medic(comment_div)
        
        # Extract replies
        replies_divs = comment_div.findNextSiblings('div', class_='qAnswerReply clearfix')
        for reply_div in replies_divs:
            reply = {
                'user': None,
                'time': None,
                'text': None,
                'votes': 0,
                'is_medic': False
            }
            user_tag = reply_div.find('div', class_='qAnswerInfo clearfix').find('strong')
            time_tag = reply_div.find('span', class_='qAnswerInfoTime')
            text_tag = reply_div.find('p')
            
            if user_tag:
                reply['user'] = user_tag.get_text(strip=True)
            if time_tag:
                reply['time'] = convert_relative_time(time_tag)
            if text_tag:
                reply['text'] = text_tag.get_text(strip=True)
            reply['votes'] = get_votes(reply_div)
            reply['is_medic'] = is_medic(reply_div)
            
            comment['replies'].append(reply)
        
        comments.append(comment)
    
    return comments

def process_question(questions:dict, title:str)->dict:
    question = questions[title]
    if 'AUTOVACCIN' not in title:
        return question
    question_page = fetch_page(question['url'])
    question['title'] = title
    question['question'] = question_page.find('p', class_='').text
    question['answers'] = get_comments_and_replies(question_page)
    #print(f'{title:>70} has {len(question["answers"]):>3} answers')
    return question # is return really needed?

def get_questions_content(questions:dict):
   with ThreadPoolExecutor(max_workers=50) as executor:  # You can adjust the number of workers
        future_to_title = {executor.submit(process_question, questions, title): title for title in questions}
        for future in as_completed(future_to_title):
            title = future_to_title[future]
            try:
                result = future.result()
                #results[title] = result
            except Exception as exc:
                print(f"ERROR: {title} generated an exception: {exc}")


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
                    print(f"ERROR: {url} generated an exception: {exc}")

    return pages

def save_all_as_json(path:str, questions:dict)->int:
    os.makedirs(path, exist_ok=True)
    dup_names_num = 0
    for title in questions:
        sanitized_title = re.sub('[<>:"/\\\\|?*\t\r\n]',' ',title).replace('  ',' ').strip()
        filename = f'{path}/{sanitized_title[:50]}.json'
        if os.path.exists(filename):
            dup_names_num += 1
            #print(f'Duplicate filename from title found: {title} - will try to differentiate')
            while os.path.exists(filename):
                filename = filename.repl('.json', 'a.json')
        try:
            with open(filename, "w") as fp:
                json.dump(questions[title], fp, indent=2)
        except Exception as exc:
            print(f"ERROR: could not write file {filename} : {exc}")
    #print(f'{path} {dup_names_num} duplicate titles')
    return dup_names_num

cats = get_all_categories()
# xx = [cats[i] for i in[*cats][-10:]]
# print(xx)

clean_data_folder = True # make this false when implemented unique question identifiers and used for file names
if clean_data_folder and os.path.exists('./data'):
    shutil.rmtree('./data')

all_questions = {}
dup_name_num = 0

for cat in cats:
    if cat != 'Boli genitale feminine':
        continue
    cat_pages = get_page_and_subsequent_pages(cats[cat])
    cat_questions = {}
    for cat_page in cat_pages:
        cat_questions |= get_questions_in_page(cat_page, cat)
    get_questions_content(cat_questions)
    all_questions |= cat_questions
    # save incrementally
    dup_name_num += save_all_as_json(path=f"./data/{cat}", questions=cat_questions)
    print(f'{cat:>70}[{len(cat_questions):>4}] | {len(all_questions):>7} total')

print(f'num questions: {len(all_questions):>6}, total duplicate titles {dup_name_num:>5}')