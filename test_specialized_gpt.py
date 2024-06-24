from transformers import AutoModelForCausalLM, AutoTokenizer

from control_vars import *
from common_utils import *
from ml_utils import *

model_name = "openai-community/gpt2"
in_folder = f'{out_data_folder}/{model_name}'


available_models = os.listdir(in_folder)
model_file, tokenizer_file = next((mf for mf in available_models if os.path.exists(f'{mf}.tokenizer')))

# Load the fine-tuned model
model = AutoModelForCausalLM.from_pretrained(model_file)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_file)

# Function to generate answers
def generate_answer(question):
    input_text = f"Question: {question} Answer:"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=512, num_return_sequences=1)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Example usage
question = "Ce tratamente sunt eficiente pentru acneea gravă?"
answer = generate_answer(question)
print(answer)
