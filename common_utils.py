
from datetime import timedelta
import json
import os
import re

def timedelta_str(dt:timedelta)->str:
    s = dt.seconds
    return f'{s//3600:>2}:{(s//60)%60:>2}:{s%60:>2}s'

def sanitize_for_filename(title:str)->str:
    sanitized_title = re.sub('[<>:"/\\\\|?*\t\r\n]',' ',title).replace('  ',' ').strip()
    sanitized_title = "".join(c if c.isprintable() else '_' for c in sanitized_title)
    return sanitized_title

def save_all_as_json(path:str, questions:dict, errors_file:str)->int:
    os.makedirs(path, exist_ok=True)
    dup_names_num = 0
    for title in questions:
        sanitized_title = sanitize_for_filename(title)
        filename = f'{path}/{sanitized_title[:50]}.json'
        if os.path.exists(filename):
            dup_names_num += 1
            while os.path.exists(filename):
                filename = filename.replace('.json', 'a.json')
        try:
            with open(filename, "w") as fp:
                json.dump(questions[title], fp, indent=2)
        except Exception as exc:
            #print(f"ERROR: could not write file {filename} : {exc}")
            with open(errors_file, "a+") as fp:
                print(f"ERROR: could not write file {filename} : {exc}", file=fp)
    return dup_names_num
