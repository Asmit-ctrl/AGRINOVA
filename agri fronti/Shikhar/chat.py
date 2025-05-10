import random
import json
import torch
import nltk
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import os
from fuzzywuzzy import process
import string
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import nltk
nltk.download('punkt_tab')

directory_path = 'D:\\agri fronti\\Shikhar\\intentsbruh'

json_files = []

for filename in os.listdir(directory_path):
    if filename.endswith('.json'):
        json_files.append(os.path.join(directory_path, filename))

intents = {"intents": []}
intent_sources = {}

for json_file in json_files:
    if not os.path.exists(json_file):
        print(f"Warning: {json_file} does not exist. Skipping this file.")
        continue

    try:
        with open(json_file, 'r') as json_data:
            data = json.load(json_data)
            if 'intents' in data:
                for intent in data['intents']:
                    intents['intents'].append(intent)
                    intent_sources[intent['tag']] = json_file
            else:
                print(f"Warning: 'intents' key not found in {json_file}. Skipping this file.")
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {json_file}. Skipping this file.")
    except Exception as e:
        print(f"Error: An unexpected error occurred while processing {json_file}: {e}")

models = []
model_files = [
    "D:\\agri fronti\\Shikhar\\data22.pth"
]
for file in model_files:
    if os.path.exists(file):
        data = torch.load(file)
        print(f"Loaded model from {file}")
        input_size = data["input_size"]
        hidden_size = data["hidden_size"]
        output_size = data["output_size"]
        all_words = data['all_words']
        tags = data['tags']

        model = NeuralNet(input_size, hidden_size, output_size).to(device)
        model.load_state_dict(data["model_state"])
        model.eval()
        models.append((model, all_words, tags))
    else:
        print(f"Warning: {file} does not exist. Skipping this model.")

bot_name = "Drisht-sha"

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
dialoGPT_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium").to(device)

def preprocess_input(msg):
    msg = msg.lower()
    msg = msg.translate(str.maketrans('', '', string.punctuation))
    return msg

def get_response(msg, models):
    if msg is None:
        return ["I do not understand..."]

    responses = []
    msg = preprocess_input(msg)

    for model, all_words, tags in models:
        sentence = tokenize(msg)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        print(f"Predicted tag: {tag}, Probability: {prob.item()}")

        if prob.item() > 0.85:
            intent_tags = [intent['tag'] for intent in intents['intents']]
            best_match, score = process.extractOne(tag, intent_tags)

            if score >= 55:
                for intent in intents['intents']:
                    if best_match == intent["tag"]:
                        response_options = intent['responses']
                        response = random.choice(response_options)
                        source_file = intent_sources[best_match]
                        response_with_source = f"Response: {response} (Source: {source_file})"
                        print(f"Response found: {response_with_source}")
                        responses.append(response)
                        break
            else:
            
                responses.append("I do not understand...")
        else:
            responses.append("I do not understand...")

    if not responses or "I do not understand..." in responses:
        input_ids = tokenizer.encode(msg + tokenizer.eos_token, return_tensors='pt').to(device)
        chat_history_ids = dialoGPT_model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        if "I do not understand..." in responses:
            responses[responses.index("I do not understand...")] = response
    
    return responses

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence.lower() in ["quit", "exit", "stop"]:
            print("Goodbye!")
            break
        else:
            resp = get_response(sentence, models)
            for i, response in enumerate(resp):
                print(f"{bot_name}: {response}")