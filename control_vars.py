import os

# make this false when implemented unique question identifiers and used for file names
clean_data_folder = True

data_root = './data'
logs_root = './logs'
os.makedirs(logs_root, exist_ok=True)
os.makedirs(data_root, exist_ok=True)


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
DESTINATION_TODO_DRIVE_FOLDER_ID = '1hSWTCO66cAl3m1jNS6VrdlfN3Gp5gyAC'
DESTINATION_DONE_DRIVE_FOLDER_ID = '1tnD0Kxl61BkBFc_vTgPQwp770s_qWsYC'

# workaround for requests limit per minute for free account
GOOGLE_API_LIMIT_WORKAROUD_WAIT = 10

# experimental and incomplete, takes a lot of time
TRY_TO_SALVAGE_TYPOS = False

USER_VALUES_ADOPTED = [
    'Da',
    'Nu',
    'Nu stiu',
]
USER_VALUES_RANKING = [
    'Raspunde complet',
    'Raspunde partial',
    'Nu raspunde',
    'Nu pot spune',
]
USER_VALUES_COMPARE = [
    'Rasp 1 e mai bun',
    'Rasp 2 e mai bun',
    'Rasp sunt la fel',
    'Niciunul nu e bun',
    'Nu pot spune',
]
