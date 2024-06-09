import os
data_root = './data'

# make this false when implemented unique question identifiers and used for file names
clean_data_folder = True

logs_root = './logs'
os.makedirs(logs_root, exist_ok=True)
os.makedirs(data_root, exist_ok=True)
