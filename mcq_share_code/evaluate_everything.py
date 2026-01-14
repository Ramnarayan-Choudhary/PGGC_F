import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import re

# Set the GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Define Hugging Face token and model IDs
hf_token = "YOUR_TOKEN"
model_id = "meta-llama/Llama-3.2-1B-Instruct"
target_model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model_directory = "/shared/rthareja"

def extract_passage(full_prompt):
    start_idx = full_prompt.find("Passage:") + len("Passage:")
    end_idx = full_prompt.find("Question:")
    return full_prompt[start_idx:end_idx].strip()

# Load the tokenizer and model
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    target_model_id,
    use_auth_token=hf_token,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir=model_directory
)
model.eval()

# Function to generate a response from the model
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_new_tokens=20,
            eos_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Load the data
data = json.load(open("../../dp_prompt/dp_prompt_epsilon_20.json"))

total_to_do = 970
correct_original = 0
vars = {}

for index in tqdm(range(total_to_do)):
    item = data[index]
    question = item["question"]
    options = item["options"]
    expected_answer = item["correct"]
    passage = item["dp_prompt_epsilon_20"]

    # full_new_text = item["my defence new full prompt"]

    passage = extract_passage(full_new_text)


    # New prompt template
    prompt_template = f"""<|user|> 
Given the following passage: {passage} 

Answer the following question based on the passage: 

{question}

Options: 
A. {options[0]} 
B. {options[1]} 
C. {options[2]} 
D. {options[3]} 

Please select the correct answer by only outputting the letter corresponding to the answer choice, in this format: [A], [B], [C], or [D]. Respond only with the letter in square brackets without any additional explanation. <|assistant|>"""

    # Generate response and extract the answer
    response = generate_response(prompt_template)
    print(response)
    # Extract the generated answer token
    match = re.search(r"The answer token is:\s*([A-D])", response)

    if match:
        predicted_option = match.group(1).strip()
        print("Predicted answer:", predicted_option)
        print("=====================================")
        print("Expected answer:", expected_answer)

        # Compare with the expected answer
        if predicted_option == expected_answer:
            correct_original += 1
    else:
        print("No valid answer found in response.")
    
    # Display current accuracies
    print(f"Current accuracy: {correct_original / (index + 1)}")


# from inference import *
# import os
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# import json
# from tqdm import tqdm

# def get_highest_prob_option(option_logits):
#     """Returns the option with the highest logit value."""
#     return max(option_logits, key=option_logits.get)

# os.environ["CUDA_VISIBLE_DEVICES"] = "5" 

# hf_token = "YOUR_TOKEN"
# model_id = "meta-llama/Llama-3.2-1B-Instruct"
# target_model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_directory = "/shared/rthareja"

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # print(f"Using device: {device}")

# model = AutoModelForCausalLM.from_pretrained(
#     target_model_id,
#     use_auth_token=hf_token,
#     torch_dtype=torch.float16,
#     device_map="auto",
#     cache_dir=model_directory  # Automatically assign to available devices
# )
# model.eval()

# # Load model and tokenizer
# print("Loading model and tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token)


# # Load the data
# data = json.load(open("updated_data.json"))

# total_to_do = 970
# correct_original = 0
# vars = {}
# for index in tqdm(range(0,total_to_do)):
#     item = data[index]
#     question = item["question"]
#     options = item["options"]
#     expected_answer = item["correct"]
#     passage = item["passage"]

#     system_prompt = "Select the correct option based on the passage provided below. You must output one token i.e A,B,C,D that's it nothing else. Do not output any new lines."
#     prompt_template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|> Passage: {passage} Question: {question}, Options: A) {options[0]}, B) {options[1]}, C) {options[2]}, D) {options[3]} <|eot_id|><|start_header_id|>assistant<|end_header_id|>The answer token is:"""

#     predictions = infer(prompt_template, model, tokenizer)

#     predicted_option = get_highest_prob_option(predictions)

#     if predicted_option == expected_answer:
#         correct_original = correct_original + 1

#     for passage_item in item["perturbed_passages"]:
#         passage = item["perturbed_passages"][passage_item]
#         system_prompt = "Select the correct option based on the passage provided below. You must output one token i.e A,B,C,D that's it nothing else. Do not output any new lines."
#         prompt_template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|> Passage: {passage} Question: {question}, Options: A) {options[0]}, B) {options[1]}, C) {options[2]}, D) {options[3]} <|eot_id|><|start_header_id|>assistant<|end_header_id|>The answer token is:"""
#         predictions = infer(prompt_template, model, tokenizer)
#         predicted_option = get_highest_prob_option(predictions)
#         if passage_item not in vars:
#             vars[passage_item] = 0
#         if predicted_option == expected_answer:
#             vars[passage_item] = vars[passage_item] + 1

#     print(f"Current accuracy {correct_original/(index+1)}")
#     for passage_item in item["perturbed_passages"]:
#         print(f"Current accuracy for {passage_item} is {vars[passage_item]/(index+1)}")