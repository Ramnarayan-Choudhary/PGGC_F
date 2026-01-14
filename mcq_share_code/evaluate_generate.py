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
# model_id = "meta-llama/Llama-3.2-1B-Instruct"
# target_model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
target_model_id = "Qwen/Qwen2.5-7B-Instruct"
model_directory = "/l/users/nils.lukas/models"

def extract_passage(full_prompt):
    start_idx = full_prompt.find("Passage:") + len("Passage:")
    end_idx = full_prompt.find("Question:")
    return full_prompt[start_idx:end_idx].strip()

# Load the tokenizer and model
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(target_model_id, use_auth_token=hf_token)
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
            max_new_tokens=50,
            eos_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


json_files = os.listdir("saves")

for fi in json_files:
    if "qwen_output_Qwen" in fi:
        print("=====================================")
        print(f"Evaluating {fi}...")
        # Load the data
        data = json.load(open(f"saves/{fi}"))

        total_to_do = len(data)
        correct_original = 0
        vars = {}
        corrects = []

        for index in tqdm(range(total_to_do)):
            item = data[index]
            question = item["question"]
            options = item["options"]
            expected_answer = item["correct"]
            if "my defence new full prompt" not in item:
                continue
            # passage = item["passage"]
            # system_prompt = "Select the correct option based on the passage provided below. You must output one token i.e A,B,C,D that's it nothing else. Do not output any new lines."
            # prompt_template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|> Passage: {passage} Question: {question}, Options: A) {options[0]}, B) {options[1]}, C) {options[2]}, D) {options[3]} <|eot_id|><|start_header_id|>assistant<|end_header_id|>The answer token is:"""


            prompt_template = item["my defence new full prompt"]
            response = generate_response(prompt_template)
            print(response)
            # Extract the generated answer token
            # match = re.search(r"The answer token is:\s*([A-D])", response)
            match = re.search(r"assistant\s*([A-D])", response)

            if match:
                predicted_option = match.group(1).strip()
                print("Predicted answer:", predicted_option)
                print("=====================================")
                print("Expected answer:", expected_answer)

                # Compare with the expected answer
                if predicted_option == expected_answer:
                    correct_original += 1
                    # if key not in vars:
                    #     vars[key] = 0
                    # else:
                    #     vars[key] += 1
            else:
                print("No valid answer found in response.")

            print(f"Current accuracy for original is {correct_original/(index+1)} for {index+1} questions.")
