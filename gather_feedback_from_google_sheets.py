
import string
#from oauth2client.service_account import Credentials
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow # for logging in the first time
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from googleapiclient.errors import HttpError
from datetime import datetime, timedelta

import shutil
import random
import time
from control_vars import *
from common_utils import *
from utils_confusion_matrix import *

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
errors_file = f'{logs_root}/gather_from_sheets_errors_{timestamp}.txt'
log_file = f'{logs_root}/gather_from_sheets_log_{timestamp}.txt'

in_data_folder = f"{data_root}/analyzed2"

words_count_file = f'{data_root}/sanitized1/words_count.json'
categories_file = f'{data_root}/sanitized1/categories.json'

out_data_folder = f"{data_root}/annotation"
if clean_data_folder and os.path.exists(out_data_folder):
    shutil.rmtree(out_data_folder)
os.makedirs(out_data_folder, exist_ok=True)

# Define the scope
SCOPES = ['https://www.googleapis.com/auth/spreadsheets',
         'https://www.googleapis.com/auth/drive']

# Define the credentials file path (replace with your actual path)
# credentials_file_path = 'client_secret.json' # not the same file
credentials_file_path = 'token.json'

service_sheets = None
service_drive = None

creds = None
# The file token.json stores the user's access and refresh tokens, and is
# created automatically when the authorization flow completes for the first
# time.
if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
# If there are no (valid) credentials available, let the user log in.
if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(
            "client_secret.json", SCOPES
        )
        creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open("token.json", "w") as token:
        token.write(creds.to_json())

try:
    service_sheets = build('sheets', 'v4', credentials=creds)
    service_drive = build('drive', 'v3', credentials=creds)
except HttpError as err:
    print(err)
    with open(errors_file, "a+") as fp:
        print(err, file=fp)
    raise err # is this advisable?

def list_spreadsheets_in_folder(folder_id):
    try:
        query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.spreadsheet'"
        results = service_drive.files().list(
            q=query, spaces='drive',
            fields='files(id, name)').execute()
        items = results.get('files', [])
        return items
    except HttpError as err:
        print(err)
        with open(errors_file, "a+") as fp:
            print(err, file=fp)
        raise err

# Function to read data from a Google Sheet
def read_google_sheet(spreadsheet_id):
    try:
        # Get the spreadsheet
        sheet = service_sheets.spreadsheets().get(spreadsheetId=spreadsheet_id['id']).execute()
        
        # Get sheet name
        sheet_name = sheet['sheets'][0]['properties']['title']
        
        # Read the data
        result = service_sheets.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id['id'], range=f'{sheet_name}!A2:J').execute()
        rows = result.get('values', [])
        
        return rows
    except HttpError as err:
        print(err)
        with open(errors_file, "a+") as fp:
            print(err, file=fp)
        raise err

def find_answer_meta(answer_text, A_meta:list, question:dict):
    if answer_text.strip():
        a1_content = [text.strip() for text in answer_text.split('Replica:')]
        a1_idx = [a['text'] for a in question['answers']].index(a1_content[0])
        if len(a1_content) > 1:
            answer = question['answers'][a1_idx]
            a1r_idx = [r['text'] for r in answer['replies']].index(a1_content[1])
            A_meta.append(f'a{a1_idx}r{a1r_idx}')
        else:
            A_meta.append(f'a{a1_idx}')

def get_answer_from_meta(meta:str, question:dict)->list[dict]:
    ret = []
    if meta:
        combined = re.search(r'a(\d+)(r(\d+))?', meta)
        aidx = int(combined.group(1))
        answer = question['answers'][aidx]
        ret.append(answer)
        if combined.group(3):
            ridx = int(combined.group(3))
            reply = answer['replies'][ridx]
            ret.append(reply)
    return ret

def aggregate_data_from_rows(rows, all_questions:dict, num_to_qid:dict[str,str])->dict:
    ret = {}
    # header row is for debugging
    for row in rows[1:]:
        # in case we will have written the answer metadata below the question id, use it
        combined = re.search(r'(\d+)(\n(\w+))?', row[0])
        num_id = combined.group(1)
        A_meta = combined.group(3)
        if A_meta is None:
            A_meta = []
            # need to apply workaround to find the actual answers from the content if we don't have
            #   the answer metadata
            question = all_questions[num_to_qid[num_id]]
            find_answer_meta(row[3], A_meta, question)
            find_answer_meta(row[4], A_meta, question)

        ret[num_id] = {
            'A1-adopted': row[5],
            'A1-ranking': row[6],
            'A2-adopted': row[7] if 7 < len(row) else None,
            'A2-ranking': row[8] if 8 < len(row) else None,
            'A1-A2-comp': row[9] if 9 < len(row) else None,
            'A-meta': A_meta,
        }
    return ret

started_time = datetime.now()
print(f'Started gathering data from Google Sheets at {started_time.strftime("%Y-%m-%d %H:%M:%S")}')
with open(log_file, "a+") as fp:
    print(f'Started gathering data from Google Sheets at {started_time.strftime("%Y-%m-%d %H:%M:%S")}', file=fp)

ids_questions_with_answers:list[str] = []
with open(f'{in_data_folder}/questions_with_answers.json', 'r') as fp:
    ids_questions_with_answers = json.load(fp)
ids_questions_with_doc_answers:list[str] = []
with open(f'{in_data_folder}/questions_with_doc_answers_2a.json', 'r') as fp:
    ids_questions_with_doc_answers = json.load(fp)
ids_questions_with_liked_doc_answers:list[str] = []
with open(f'{in_data_folder}/questions_with_liked_doc_answers_2a.json', 'r') as fp:
    ids_questions_with_liked_doc_answers = json.load(fp)
ids_questions_with_non_doc_answers:list[str] = []
with open(f'{in_data_folder}/questions_with_non_doc_answers_2a.json', 'r') as fp:
    ids_questions_with_non_doc_answers = json.load(fp)

num_to_qid:dict[str,str] = {}

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
        numstr = str(get_num_from_question_id(question['id']))
        if numstr in num_to_qid:
            print(f'ERROR: duplicate num id: {numstr}')
            with open(errors_file, "a+") as fp:
                print(f'ERROR: duplicate num id: {numstr}', file=fp)
        num_to_qid[numstr] = question['id']
    except Exception as exc:
        # do nothing, there must have been some error somewhere
        # TODO: investigate why such errors appear
        with open(errors_file, "a+") as fp:
            print(f'ERROR: reading question file {question_file}: {exc}', file=fp)

overall_time = datetime.now()-started_time
print(f'questions loaded from disk | time {timedelta_str(overall_time)}  ')
with open(log_file, "a+") as fp:
    print(f'questions loaded from disk | time {timedelta_str(overall_time)}  ', file=fp)

num_so_far = 0
feedback_q = {}
sheets_ids = list_spreadsheets_in_folder(DESTINATION_DONE_DRIVE_FOLDER_ID)
for sheet_id in sheets_ids:
    feedback_from_sheet = aggregate_data_from_rows(read_google_sheet(sheet_id),
                                                   all_questions=questions_with_answers,
                                                   num_to_qid=num_to_qid)
    for numstr_id in feedback_from_sheet:
        q_id = num_to_qid[numstr_id]
        question = questions_with_answers[q_id]
        feedback = feedback_from_sheet[numstr_id]
        a1meta = feedback['A-meta'][0]

        a1 = get_answer_from_meta(a1meta, question)
        
        if 'feedback' not in question:
            question['feedback'] = []
        question['feedback'].append({
            a1meta: {
                'adopted': feedback['A1-adopted'],
                'ranking': feedback['A1-ranking'],
            }
        })
        if 'feedback' not in a1[0]:
            a1[0]['feedback'] = []
        a1[0]['feedback'].append({
            'adopted': feedback['A1-adopted'],
            'ranking': feedback['A1-ranking'],
        })
        if len(a1)>1:
            # add the same for the reply, if there is one
            if 'feedback' not in a1[1]:
                a1[1]['feedback'] = []
            a1[1]['feedback'].append({
                'adopted': feedback['A1-adopted'],
                'ranking': feedback['A1-ranking'],
            })
        
        if len(feedback['A-meta']) > 1:
            a2meta = feedback['A-meta'][1]

            a2 = get_answer_from_meta(a2meta, question)
            
            if 'feedback' not in question:
                question['feedback'] = []
            question['feedback'].append({
                a2meta: {
                    'adopted': feedback['A2-adopted'],
                    'ranking': feedback['A2-ranking'],
                }
            })
            if 'feedback' not in a2[0]:
                a2[0]['feedback'] = []
            a2[0]['feedback'].append({
                'adopted': feedback['A2-adopted'],
                'ranking': feedback['A2-ranking'],
            })
            if len(a2)>1:
                # add the same for the reply, if there is one
                if 'feedback' not in a2[1]:
                    a2[1]['feedback'] = []
                a2[1]['feedback'].append({
                    'adopted': feedback['A2-adopted'],
                    'ranking': feedback['A2-ranking'],
                })

        if question['id'] not in feedback_q:
            feedback_q[question['id']] = question
    
    num_so_far += 1
    overall_time = datetime.now()-started_time
    print(f'waiting 10 seconds to avoid request limit per minute, time {timedelta_str(overall_time)}', end='\r')
    time.sleep(GOOGLE_API_LIMIT_WORKAROUD_WAIT)
    print(f'time {timedelta_str(overall_time)}   ( {(100*num_so_far/len(sheets_ids)):>5.1f}% )                                              ')

with open(log_file, "a+") as fp:
    print(f'Total questions with feedback: {len(feedback_q):>5}. Saving...')
    print(f'Total questions with feedback: {len(feedback_q):>5}', file=fp)

num_duped = save_all_as_json(f'{out_data_folder}/questions', questions=feedback_q, errors_file=errors_file)
with open(f'{out_data_folder}/questions/_questions_annotated.json', 'w') as fp:
    json.dump(feedback_q, fp=fp, indent=2)
with open(f'{out_data_folder}/questions/_questions_with_answers.json', 'w') as fp:
    json.dump(questions_with_answers, fp=fp, indent=2)

print('Saved, computing agreement and confusion matrixes')

my_labels = []
annotator1_adopt = []
annotator1_rank = []
annotator2_adopt = []
annotator2_rank = []
for q_id in feedback_q:
    q = feedback_q[q_id]
    ann1 = {}
    ann2 = {}
    fkeys = [] # the answer keys
    for f in q['feedback']:
        fkey = list(f.keys())[0]
        fkeys.append(fkey)
        if fkey in ann1:
            ann2[fkey] = {
                'adopted': USER_VALUES_ADOPTED.index(f[fkey]['adopted']),
                'ranking': USER_VALUES_RANKING.index(f[fkey]['ranking']),
            }
        else:
            ann1[fkey] = {
                'adopted': USER_VALUES_ADOPTED.index(f[fkey]['adopted']),
                'ranking': USER_VALUES_RANKING.index(f[fkey]['ranking']),
            }
    # validate that there are 2 of each value for each answer key
    fkeys = list(set(fkeys))
    if all((f in ann1 and f in ann2 for f in fkeys)):
        annotator1_adopt += [ann1[fkey]['adopted'] for fkey in fkeys]
        annotator1_rank += [ann1[fkey]['ranking'] for fkey in fkeys]
        annotator2_adopt += [ann2[fkey]['adopted'] for fkey in fkeys]
        annotator2_rank += [ann2[fkey]['ranking'] for fkey in fkeys]
        # make the answer keys absolute by adding id of question
        fkeys = [f'{q["id"]}:{fkey}' for fkey in fkeys]
        my_labels += fkeys

# the numeric values mean:
# Adopted:
#   0. Da (yes)
#   1. Nu (no)
#   2. Nu stiu (don't know)
# Ranking:
#   0. Raspunde complet (answers completely)
#   1. Raspunde partial (answers partially)
#   2. Nu raspunde (does not answer)
#   3. Nu pot spune (don't know)

# Display the Agreement Matrixes
agreement_matrix_adopt = create_agreement_matrix(annotator1_adopt, annotator2_adopt)
print("\nAgreement Matrix for Adopted:")
print(agreement_matrix_adopt)
with open(f'{out_data_folder}/agreement_matrix_adopt.txt', 'w') as fp:
    print(agreement_matrix_adopt, file=fp)

agreement_matrix_rank = create_agreement_matrix(annotator1_rank, annotator2_rank)
print("\nAgreement Matrix for Ranking:")
print(agreement_matrix_rank)
with open(f'{out_data_folder}/agreement_matrix_rank.txt', 'w') as fp:
    print(agreement_matrix_rank, file=fp)

# Display the Confusion Matrix
conf_matrix_adopt, val_labels_adopt = create_confusion_matrix(annotator1_adopt, annotator2_adopt)
conf_matrix_adopt_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_adopt, display_labels=val_labels_adopt)
print("\nConfusion Matrix for Adopted:")
print(conf_matrix_adopt)
with open(f'{out_data_folder}/conf_matrix_adopt.txt', 'w') as fp:
    print(conf_matrix_adopt, file=fp)

conf_matrix_rank, val_labels_rank = create_confusion_matrix(annotator1_rank, annotator2_rank)
conf_matrix_rank_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_rank, display_labels=val_labels_rank)
print("\nConfusion Matrix for Ranking:")
print(conf_matrix_rank)
with open(f'{out_data_folder}/conf_matrix_rank.txt', 'w') as fp:
    print(conf_matrix_rank, file=fp)

# We want another to combine the rankings 0 and 1, that is partial and complete
annotator1_rank_c = [r if r>1 else 1 for r in annotator1_rank]
annotator2_rank_c = [r if r>1 else 1 for r in annotator2_rank]
conf_matrix_rank_c, val_labels_rank_c = create_confusion_matrix(annotator1_rank_c, annotator2_rank_c)
conf_matrix_rank_c_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_rank_c, display_labels=val_labels_rank_c)
print("\nConfusion Matrix for combined Ranking:")
print(conf_matrix_rank_c)
with open(f'{out_data_folder}/conf_matrix_rank_combined.txt', 'w') as fp:
    print(conf_matrix_rank_c, file=fp)


conf_matrix_adopt_display.plot(cmap='Blues')
conf_matrix_adopt_display.ax_.set_title('Confusion Matrix for Adopted between Annotator 1 and Annotator 2')
conf_matrix_adopt_display.figure_.savefig(f'{out_data_folder}/conf_mat_adopted_annotated.png',dpi=300)

conf_matrix_rank_display.plot(cmap='Blues')
conf_matrix_rank_display.ax_.set_title('Confusion Matrix for Ranking between Annotator 1 and Annotator 2')
conf_matrix_rank_display.figure_.savefig(f'{out_data_folder}/conf_mat_ranked_annotated.png',dpi=300)

conf_matrix_rank_c_display.plot(cmap='Blues')
conf_matrix_rank_c_display.ax_.set_title('Confusion Matrix for Ranking between Annotator 1 and Annotator 2')
conf_matrix_rank_c_display.figure_.savefig(f'{out_data_folder}/conf_mat_ranked_combined_annotated.png',dpi=300)


with open(log_file, "a+") as fp:
    print("\nAgreement Matrix for Adopted:", file=fp)
    print(agreement_matrix_adopt, file=fp)
    print("\nAgreement Matrix for Ranking:", file=fp)
    print(agreement_matrix_rank, file=fp)
    print("\nConfusion Matrix for Adopted:", file=fp)
    print(conf_matrix_adopt, file=fp)
    print("\nConfusion Matrix for Ranking:", file=fp)
    print(conf_matrix_rank, file=fp)
    print("\nConfusion Matrix for combinedRanking:", file=fp)
    print(conf_matrix_rank_c, file=fp)

print(f'Ended at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ( {timedelta_str(datetime.now()-started_time)} )')
with open(log_file, "a+") as fp:
    print(f'Ended at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ( {timedelta_str(datetime.now()-started_time)} )', file=fp)
