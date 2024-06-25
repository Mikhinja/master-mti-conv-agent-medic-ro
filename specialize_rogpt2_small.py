import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from control_vars import *
from common_utils import *
from ml_utils import *

# Disable oneDNN custom operations warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your data
data = load_my_data()
model_name = "readerbench/RoGPT2-base"
out_model_folder = f'{out_data_folder}/{model_name}_small'
max_length = 192# + 128 # 512

if not os.path.exists(out_model_folder):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def preprocess_data(data, tokenizer, max_length):
        inputs = []
        for key in data:
            question = data[key]
            question_text = data[key]['question']
            if len(question_text) >= max_length:
                continue
            good_answers = [answer['text'] for answer in data[key]['answers']
                            if answer['text'] is not None
                            and len(answer['text']) < max_length-8
                            and estimate_goodness_from_feedback(question, answer) > 0]
            if not good_answers:
                continue
            all_good_answers = " ".join(good_answers)
            encoded_input = tokenizer(f"Question: {question_text} Answer: {all_good_answers[:max_length]}", 
                                    truncation=True, padding='max_length', max_length=max_length)
            inputs.append((encoded_input['input_ids'], encoded_input['attention_mask']))
        return inputs

    cleaned_data = preprocess_data(data, tokenizer, max_length)

    train_dataset = MedicalQADataset(cleaned_data)

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    training_args = TrainingArguments(
        output_dir=f'{out_data_folder}/train_results/{model_name}',
        num_train_epochs=1,
        per_device_train_batch_size=4,
        #gradient_accumulation_steps=2,  # Accumulate gradients over 2 steps to simulate a batch size of 16
        save_steps=5_000,
        save_total_limit=2,
        fp16=True if device=='cuda' else False,  # Enable mixed precision training
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()

    model.save_pretrained(f'{out_model_folder}')
    tokenizer.save_pretrained(f'{out_model_folder}_tokenizer')

else:
    tokenizer = AutoTokenizer.from_pretrained(f'{out_model_folder}_tokenizer')
    model = AutoModelForCausalLM.from_pretrained(f'{out_model_folder}')

# Example test question

gen_text_length = 96
test_questions = [  'as vrea si eu sa stiu ce remedii naturale exista pentru disparitia acneii',
                    'buna ziua ! as dori si eu , daca se poate sa imi recomandati un tratament bun pentru acnee deoarece de la un timp am inceput sa am fata plina de acee , din cauza pubertati . as vrea de preferat sa fie ceva puternic care sa isi faca efectul repede',
                    'buna ziua , de 6 luni am probleme digestive care se manifesta in prezent cu : diaree aproape saptamanal , disconfort abdominal zilnic care apare cand ma trezesc , gaze puternice care apar numai inainte de diaree , abdomen zgomotos si stresul care imi provoaca toare acestes .',
                    #'am un baietel de 1 luna si 10 zile are hernie la buric doctorul spune ca are timp pana la 2 ani sa i treaca adica sa intre inauntru ca e scos afara ca si cum unflat . daca nu este necesar sa intervina cu operatie , va rog datimi un sfat si daca stiti de ce a aparut aceasta hernie',
                    
                    'dupa un chiuretaj la cat tinp pot avea contact sexual ?',
                    # buna ziua , in mod normal , contactul sexual este permis dupa 6 saptamani . cu toate acestea , ar fi recomandat sa tineti cont de indicatiile medicului care a efectuat chiuretajul si totdata , de monetul in care va simtiti pregatita pentru a va realua viata sexuala . un chiuretaj reprezinta un stres major atat fizic cat si psihic , multa sanatate !
]


keep_doing_it = 3
while keep_doing_it > 0:
    test_input = f'întrebare: {test_questions[keep_doing_it % len(test_questions)]}. răspuns: '
    # Tokenize the input
    inputs = tokenizer(test_input, return_tensors='pt', truncation=True, padding='max_length', max_length=max_length)

    # Generate response
    gen_text_max_length = len(inputs[0]) + gen_text_length
    
    output = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], 
                            max_length=gen_text_max_length, # 1024
                            no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id)

    # Decode and print the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)
    keep_doing_it -= 1


