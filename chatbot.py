import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import json
import random

# Load the model data
model_data = torch.load("data.pth")

# Initialize the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained(model_data["tokenizer_name"])
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(model_data["tags"]))
model.load_state_dict(model_data["model_state"])
model.eval()

# Load intents
with open('intents.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

# Function to get the response
def get_response(msg):
    inputs = tokenizer(msg, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
    outputs = model(**inputs)
    _, predicted = torch.max(outputs.logits, dim=1)
    tag = model_data["tags"][predicted.item()]

    probs = torch.softmax(outputs.logits, dim=1)
    prob = probs[0][predicted.item()]
    
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "Maaf pesan yang sobat kirim kurang dimengerti"
