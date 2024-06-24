from transformers import GPT2LMHeadModel, GPT2Tokenizer


from control_vars import *
from common_utils import *
from ml_utils import *

# tensorflow '2.16.1'

# Specify the model name for Romanian GPT-2
# model_name = "flax-community/gpt2-medium-romanian"
model_name = "openai-community/gpt2"

out_data_folder = f'{out_data_folder}/{model_name}'

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

model_file_name = f"{out_data_folder}/{model_name}"
# Save the model and tokenizer
model.save_pretrained(model_file_name)
tokenizer.save_pretrained(f"{model_file_name}.tokenizer")

print(f"Model and tokenizer downloaded and saved successfully: {model_file_name}.")
