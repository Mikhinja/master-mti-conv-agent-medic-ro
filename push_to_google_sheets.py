
import string
import gspread
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

# stat control params
NUM_QUESTIONS_TO_ANNNOTATE = 50
NUM_QUESTIONS_PER_SHEET = 10

# google sheets values
SHEET_ID_COL_WIDTH = 50
SHEET_TITLE_COL_WIDTH = 90
SHEET_TEXT_COL_WIDTH = 330
SHEET_FLAG_IN_COL_WIDTH = 50
SHEET_COMP_IN_COL_WIDTH = 100

# google drive destination folder id
DESTINATION_DRIVE_FOLDER_ID = '19OQ2xxGYk5f90yMTfMOh6-EbhWN3Xro4'

# workaround for requests limit per minute for free account
GOOGLE_API_LIMIT_WORKAROUD_WAIT = 10

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
errors_file = f'{logs_root}/push_to_sheets_errors_{timestamp}.txt'
log_file = f'{logs_root}/push_to_sheets_log_{timestamp}.txt'

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

# Create a new Google Sheet
def create_google_sheet(title):
    spreadsheet = {
        'properties': {
            'title': title
        }
    }
    try:
        sheet = service_sheets.spreadsheets().create(body=spreadsheet, fields='spreadsheetId').execute()
        return sheet.get('spreadsheetId')
    except HttpError as err:
        print(err)
        with open(errors_file, "a+") as fp:
            print(err, file=fp)
        raise err

def delete_google_sheet(spreadsheet_id):
    requests = [
        {
            "deleteSheet": {
                "sheetId": spreadsheet_id
            }
        }
    ]
    body = {
        'requests': requests
    }
    try:
        response = service_sheets.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id, body=body).execute()
        return response
    except HttpError as err:
        print(err)
        with open(errors_file, "a+") as fp:
            print(err, file=fp)

def format_columns(spreadsheet_id, start_index, end_index, width, wrap=False):
    requests = [
        {
            "updateDimensionProperties": {
                "range": {
                    "sheetId": 0,
                    "dimension": "COLUMNS",
                    "startIndex": start_index,
                    "endIndex": end_index
                },
                "properties": {
                    "pixelSize": width
                },
                "fields": "pixelSize"
            }
        }]
    if wrap:
        requests.append({
            "repeatCell": {
                "range": {
                    "sheetId": 0,
                    "startRowIndex": 1,  # Start after the header row
                    "startColumnIndex": start_index,
                    "endColumnIndex": end_index
                },
                "cell": {
                    "userEnteredFormat": {
                        "wrapStrategy": "WRAP"
                    }
                },
                "fields": "userEnteredFormat.wrapStrategy"
            }
        })
    body = {
        'requests': requests
    }
    try:
        response = service_sheets.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id, body=body).execute()
        return response
    except HttpError as err:
        print(err)
        with open(errors_file, "a+") as fp:
            print(err, file=fp)
        raise err


# Add columns to the sheet
def setup_columns(spreadsheet_id):
    columns = ["id", "Titlu/Categorie", "Intrebare", "Raspuns 1", "Raspuns 2",
                    "Adoptat", "Relevant", "Adoptat", "Relevant", "Comparatie"]
    body = {
        'values': [columns]
    }
    try:
        result = service_sheets.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id, range='A2:J2',
            valueInputOption='RAW', body=body).execute()
        
        # add values for the merged cells
        result = service_sheets.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id, range='F1:H1',
            valueInputOption='RAW', body={
                'values': [["Raspuns 1", "", "Raspuns 2"]]
            }).execute()
        result = service_sheets.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id, range='A1:B1',
            valueInputOption='RAW', body={
                'values': [["Evaluati raspunsurile, nu intrebarile. Daca raspunsul are si o replica atunci evaluati doar replica.", ""]]
            }).execute()
        
        result = format_columns(spreadsheet_id=spreadsheet_id, start_index=0, end_index=1, width=SHEET_ID_COL_WIDTH, wrap=True)
        result = format_columns(spreadsheet_id=spreadsheet_id, start_index=1, end_index=2, width=SHEET_TITLE_COL_WIDTH, wrap=True)
        result = format_columns(spreadsheet_id=spreadsheet_id, start_index=2, end_index=5, width=SHEET_TEXT_COL_WIDTH, wrap=True)
        # now the input columns, no wrap
        result = format_columns(spreadsheet_id=spreadsheet_id, start_index=5, end_index=6, width=SHEET_FLAG_IN_COL_WIDTH)
        result = format_columns(spreadsheet_id=spreadsheet_id, start_index=6, end_index=7, width=SHEET_COMP_IN_COL_WIDTH)
        result = format_columns(spreadsheet_id=spreadsheet_id, start_index=7, end_index=8, width=SHEET_FLAG_IN_COL_WIDTH)
        result = format_columns(spreadsheet_id=spreadsheet_id, start_index=8, end_index=9, width=SHEET_COMP_IN_COL_WIDTH)
        result = format_columns(spreadsheet_id=spreadsheet_id, start_index=9, end_index=10, width=SHEET_COMP_IN_COL_WIDTH)

        requests = [
            {
                "mergeCells": {
                    "range": {
                        "sheetId": 0,
                        "startRowIndex": 0,
                        "endRowIndex": 1,
                        "startColumnIndex": 5,
                        "endColumnIndex": 7
                    },
                    "mergeType": "MERGE_ALL"
                }
            },
            {
                "mergeCells": {
                    "range": {
                        "sheetId": 0,
                        "startRowIndex": 0,
                        "endRowIndex": 1,
                        "startColumnIndex": 7,
                        "endColumnIndex": 9
                    },
                    "mergeType": "MERGE_ALL"
                }
            }
            ,
            {
                "mergeCells": {
                    "range": {
                        "sheetId": 0,
                        "startRowIndex": 0,
                        "endRowIndex": 1,
                        "startColumnIndex": 0,
                        "endColumnIndex": 4
                    },
                    "mergeType": "MERGE_ALL"
                }
            }
        ]
        body = {
            'requests': requests
        }
        service_sheets.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id, body=body).execute()
            
        return result
    except HttpError as err:
        print(err)
        with open(errors_file, "a+") as fp:
            print(err, file=fp)
        raise err

# Add data validation for dropdown in columns G and H
def add_data_validation(spreadsheet_id):
    requests = [
        {
            "setDataValidation": {
                "range": {
                    "sheetId": 0,
                    "startRowIndex": 2,
                    "endRowIndex": 2 + NUM_QUESTIONS_PER_SHEET,
                    "startColumnIndex": 5,
                    "endColumnIndex": 6
                },
                "rule": {
                    "condition": {
                        "type": "ONE_OF_RANGE",
                        "values": [
                            {"userEnteredValue": "=M1:M3"}
                        ]
                    },
                    "showCustomUi": True,
                    "strict": True
                }
            }
        },
        {
            "setDataValidation": {
                "range": {
                    "sheetId": 0,
                    "startRowIndex": 2,
                    "endRowIndex": 2 + NUM_QUESTIONS_PER_SHEET,
                    "startColumnIndex": 6,
                    "endColumnIndex": 7
                },
                "rule": {
                    "condition": {
                        "type": "ONE_OF_RANGE",
                        "values": [
                            {"userEnteredValue": "=N1:N4"}
                        ]
                    },
                    "showCustomUi": True,
                    "strict": True
                }
            }
        },
        {
            "setDataValidation": {
                "range": {
                    "sheetId": 0,
                    "startRowIndex": 2,
                    "endRowIndex": 2 + NUM_QUESTIONS_PER_SHEET,
                    "startColumnIndex": 7,
                    "endColumnIndex": 8
                },
                "rule": {
                    "condition": {
                        "type": "ONE_OF_RANGE",
                        "values": [
                            {"userEnteredValue": "=M1:M3"}
                        ]
                    },
                    "showCustomUi": True,
                    "strict": True
                }
            }
        },
        {
            "setDataValidation": {
                "range": {
                    "sheetId": 0,
                    "startRowIndex": 2,
                    "endRowIndex": 2 + NUM_QUESTIONS_PER_SHEET,
                    "startColumnIndex": 8,
                    "endColumnIndex": 9
                },
                "rule": {
                    "condition": {
                        "type": "ONE_OF_RANGE",
                        "values": [
                            {"userEnteredValue": "=N1:N4"}
                        ]
                    },
                    "showCustomUi": True,
                    "strict": True
                }
            }
        },
        {
            "setDataValidation": {
                "range": {
                    "sheetId": 0,
                    "startRowIndex": 2,
                    "endRowIndex": 2 + NUM_QUESTIONS_PER_SHEET,
                    "startColumnIndex": 9,
                    "endColumnIndex": 10
                },
                "rule": {
                    "condition": {
                        "type": "ONE_OF_RANGE",
                        "values": [
                            {"userEnteredValue": "=O1:O5"}
                        ]
                    },
                    "showCustomUi": True,
                    "strict": True
                }
            }
        }
    ]
    body = {
        'requests': requests
    }
    try:
        # add the validation values
        response = service_sheets.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id, range='M1:M3',
            valueInputOption='RAW', body={
                'values': [['Da'],
                           ['Nu'],
                           ['Nu stiu'],]
            }).execute()
        response = service_sheets.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id, range='N1:N4',
            valueInputOption='RAW', body={
                'values': [['Raspunde complet'],
                           ['Raspunde partial'],
                           ['Nu raspunde'],
                           ['Nu pot spune'],]
            }).execute()
        response = service_sheets.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id, range='O1:O5',
            valueInputOption='RAW', body={
                'values': [['Rasp 1 e mai bun'],
                           ['Rasp 2 e mai bun'],
                           ['Rasp sunt la fel'],
                           ['Niciunul nu e bun'],
                           ['Nu pot spune'],]
            }).execute()

        response = service_sheets.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id, body=body).execute()
        return response
    except HttpError as err:
        print(err)
        with open(errors_file, "a+") as fp:
            print(err, file=fp)
        raise err

# Insert rows into the sheet
def insert_data_rows(spreadsheet_id, data):
    values = []
    for entry in data:
        id_num = get_num_from_question_id(entry['id'])
        title = entry.get("title", "")
        category = ", ".join(entry.get("category", []))
        question = entry.get("question", "")
        
        # Extract answers and replies
        answers = entry.get("answers", [])
        if answers:
            answer1_text = answers[0].get("text", "")
            for reply in answers[0].get("replies", []):
                reply_text = reply.get("text", "")
                answer1_text += f'\nReplica: {reply_text}'
            answer2_text = ''
            if len(answers) > 1:
                answer2_text = answers[1].get("text", "")
                for reply in answers[1].get("replies", []):
                    reply_text = reply.get("text", "")
                    answer2_text += f'\nReplica: {reply_text}'
            values.append([id_num, f'{title}\n\n{category}', question, answer1_text, answer2_text])
        if not answers:
            # should never happen
            values.append([id_num, f'{title}\n\n{category}', question, "NU exista raspuns"])
    
    body = {
        'values': values
    }
    try:
        result = service_sheets.spreadsheets().values().append(
            spreadsheetId=spreadsheet_id, range='A3:G3',
            valueInputOption='RAW', insertDataOption='OVERWRITE', body=body).execute()
        return result
    except HttpError as err:
        print(err)
        with open(errors_file, "a+") as fp:
            print(err, file=fp)
        raise err

# Move the file to a specific folder
def move_file_to_folder(file_id, folder_id):
    try:
        # Retrieve the existing parents to remove
        file = service_drive.files().get(fileId=file_id, fields='parents').execute()
        previous_parents = ",".join(file.get('parents'))

        # Move the file to the new folder
        file = service_drive.files().update(
            fileId=file_id,
            addParents=folder_id,
            removeParents=previous_parents,
            fields='id, parents'
        ).execute()
        return file
    except HttpError as err:
        print(err)
        with open(errors_file, "a+") as fp:
            print(err, file=fp)
        raise err

def generate_sheets_by_partitioning(picked_questions):  
    for start_idx in range(0, len(picked_questions), NUM_QUESTIONS_PER_SHEET):
        try:
            end_idx = start_idx + NUM_QUESTIONS_PER_SHEET
            # Unique name for the Google Sheet
            sheet_title = "DataSheet_" + ''.join(random.choices(string.ascii_letters + string.digits, k=6)) + datetime.now().strftime("%Y-%m-%d_%H-%M")
            # Create Google Sheet
            spreadsheet_id = create_google_sheet(sheet_title)
            if spreadsheet_id:
                #print(f"Spreadsheet created with ID: {spreadsheet_id}")
                with open(log_file, "a+") as fp:
                    print(f'Spreadsheet created with ID: {spreadsheet_id}', file=fp)
                
                # Set up columns
                setup_columns(spreadsheet_id)

                # make it easy to input data
                add_data_validation(spreadsheet_id)

                # Insert rows into Google Sheet
                insert_data_rows(spreadsheet_id, picked_questions[start_idx:end_idx])

                # Move the file to the specified folder
                move_file_to_folder(spreadsheet_id, DESTINATION_DRIVE_FOLDER_ID)

                #print(f"Data inserted into Spreadsheet ID: {spreadsheet_id}")
                with open(log_file, "a+") as fp:
                    print(f'Data inserted into Spreadsheet ID: {spreadsheet_id}', file=fp)
                
                # just to have something to watch
                overall_time = datetime.now()-started_time
                print(f'waiting 10 seconds to avoid request limit per minute, time {timedelta_str(overall_time)}', end='\r')
                time.sleep(GOOGLE_API_LIMIT_WORKAROUD_WAIT)
                print(f'time {timedelta_str(overall_time)}   ( {(100*end_idx/len(picked_questions)):>5.1f}% )                                              ')
            else:
                print("Failed to create spreadsheet.")
        except Exception as exc:
            delete_google_sheet(spreadsheet_id)
            print(f'ERROR: exception {exc}')
    return spreadsheet_id

started_time = datetime.now()
print(f'Started pushing data to Google Sheets at {started_time.strftime("%Y-%m-%d %H:%M:%S")}')
with open(log_file, "a+") as fp:
    print(f'Started  pushing data to Google Sheets at {started_time.strftime("%Y-%m-%d %H:%M:%S")}', file=fp)


ids_questions_with_doc_answers:list[str] = []
with open(f'{in_data_folder}/questions_with_doc_answers_2a.json', 'r') as fp:
    ids_questions_with_doc_answers = json.load(fp)
ids_questions_with_liked_doc_answers:list[str] = []
with open(f'{in_data_folder}/questions_with_liked_doc_answers_2a.json', 'r') as fp:
    ids_questions_with_liked_doc_answers = json.load(fp)
ids_questions_with_non_doc_answers:list[str] = []
with open(f'{in_data_folder}/questions_with_non_doc_answers_2a.json', 'r') as fp:
    ids_questions_with_non_doc_answers = json.load(fp)

ids_questions_candidates = (ids_questions_with_liked_doc_answers
                            + ids_questions_with_liked_doc_answers
                            + ids_questions_with_liked_doc_answers
                            + ids_questions_with_doc_answers
                            + ids_questions_with_doc_answers
                            + ids_questions_with_non_doc_answers)

picked_questions:list[dict] = []

picked_questions_names = random.sample(ids_questions_candidates, k=NUM_QUESTIONS_TO_ANNNOTATE)
for question_id in picked_questions_names:
   filename = f'{in_data_folder}/questions/{question_id}.json'
   with open(filename, 'r') as fp:
      question_full = json.load(fp)
      # codify answer or reply
      all_pickable_answers = [f'a{aidx}' for aidx, _ in enumerate(question_full['answers'])]
      # make the direct answers appear twice to:
      #     - gracefully pick the 1 answer where there is only 1
      #     - make direct answers twice as likely than replies
      all_pickable_answers += all_pickable_answers
      all_pickable_answers += [f'a{aidx}r{ridx}' for aidx, answer in enumerate(question_full['answers'])
                                for ridx, _ in enumerate(answer['replies'])]
      two_answers = list(set(random.sample(all_pickable_answers, k=2)))
      # decodify to actual answer or reply
      picked_question = question_full.copy()
      answers = []
      for pick in two_answers:
            aidx = int(re.search(r'a(\d+)', pick).group(1))
            answer = picked_question['answers'][aidx].copy()
            if 'r' in pick:
                ridx = int(re.search(r'r(\d+)', pick).group(1))
                answer['replies'] = [answer['replies'][ridx].copy()]
            else:
                answer['replies'] = []
            answers.append(answer)
      picked_question['answers'] = answers
      picked_questions.append(picked_question)

print('Going to do 2 passes to ensure every question gets 2 evaluatinos')

# do it once
random.shuffle(picked_questions)
spreadsheet_id = generate_sheets_by_partitioning(picked_questions)
print('First pass done')

# do it twice to ensure each question appears exactly twice
random.shuffle(picked_questions)
spreadsheet_id = generate_sheets_by_partitioning(picked_questions)
print('Second pass done')

print(f'Ended at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ( {timedelta_str(datetime.now()-started_time)} )')
with open(log_file, "a+") as fp:
    print(f'Ended at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ( {timedelta_str(datetime.now()-started_time)} )', file=fp)

