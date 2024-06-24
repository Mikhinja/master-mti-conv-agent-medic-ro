from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

from datetime import datetime, timedelta

from control_vars import *
from common_utils import *
from ml_utils import *

data = load_my_data()

# Load the tokenizer and model
model_name = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

out_data_folder = f'{out_data_folder}/{model_name}'
max_length = 512
tuned_model_name = f'{model_name}_medical_{max_length}_{timestamp}'

# Prepare the dataset
def preprocess_data(data):
    inputs = []
    for key in data:
        question = data[key]
        question_text = data[key]['question']
        good_answers = [answer for answer in data[key]['answers']
                        if estimate_goodness_from_feedback(question, answer) > 0
                            and answer['text'] is not None]
        if not good_answers:
            # skip questions without good answers
            continue
        all_good_answers = " ".join([answer['text'] for answer in good_answers])
        inputs.append(tokenizer(f"Question: {question_text} Answer: {all_good_answers}", truncation=True, padding='max_length', max_length=max_length))
    return inputs

cleaned_data = preprocess_data(data)
# dataset = Dataset.from_dict({"input_ids": [x['input_ids'] for x in cleaned_data], "attention_mask": [x['attention_mask'] for x in cleaned_data]})

# # Add labels
# tokenized_datasets = dataset.map(lambda examples: {"labels": examples["input_ids"]}, batched=True)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# Prepare data for Dataset object
train_data = []
for key in data:
    question = data[key]
    question_text = data[key]['question']
    good_answers = [answer for answer in data[key]['answers']
                    if estimate_goodness_from_feedback(question, answer) > 0
                        and answer['text'] is not None]
    if not good_answers:
        # skip questions without good answers
        continue
    all_good_answers = " ".join([answer['text'] for answer in good_answers])
    train_data.append({"text": f"Question: {question_text} Answer: {all_good_answers}"})

# Create a Dataset object
dataset = Dataset.from_dict({"text": train_data})
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained(f"{out_data_folder}/{tuned_model_name}")
tokenizer.save_pretrained(f"{out_data_folder}/{tuned_model_name}.tokenizer")

eval_results = trainer.evaluate()
print(eval_results)

print(f'done specializing {model_name}: {tuned_model_name}')
