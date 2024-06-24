import os
from transformers import AutoTokenizer, AutoModelForCausalLM
# TensorFlow
from transformers import TFAutoModelForCausalLM
import keras  # Ensure keras is correctly imported
from control_vars import *
from common_utils import *
from ml_utils import *

# Disable oneDNN custom operations warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load your data
data = load_my_data()
model_name = "readerbench/RoGPT2-medium"

test_question1 = 'as vrea si eu sa stiu ce remedii naturale exista pentru disparitia acneii'
test_question2 = 'buna ziua ! as dori si eu , daca se poate sa imi recomandati un tratament bun pentru acnee deoarece de la un timp am inceput sa am fata plina de acee , din cauza pubertati . as vrea de preferat sa fie ceva puternic care sa isi faca efectul repede'
test_question3 = 'buna ziua , de 6 luni am probleme digestive care se manifesta in prezent cu : diaree aproape saptamanal , disconfort abdominal zilnic care apare cand ma trezesc , gaze puternice care apar numai inainte de diaree , abdomen zgomotos si stresul care imi provoaca toare acestes .'

test_input = f'întrebare: {test_question1}. răspuns: '

# Use the AutoTokenizer and AutoModelForCausalLM for PyTorch
tokenizer = AutoTokenizer.from_pretrained(model_name) # padding_side='left' why is this bad?
model = AutoModelForCausalLM.from_pretrained(model_name, force_download=True)

# Encode the input with attention mask and pad token ID
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer.encode_plus(test_input, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

gen_text_length = 64
gen_text_max_length = len(input_ids[0]) + gen_text_length

keep_doing_it = 3
while keep_doing_it > 0:
    # Generate text
    text = model.generate(input_ids, attention_mask=attention_mask,
                            max_length=gen_text_max_length, # 1024
                            no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id)

    # Decode and print the generated text
    print(tokenizer.decode(text[0], skip_special_tokens=True))
    keep_doing_it -= 1

