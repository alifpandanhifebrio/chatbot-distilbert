import random
import json
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load the model and tokenizer
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = DistilBertForSequenceClassification.from_pretrained("chatbot_model").to(device)
tokenizer = DistilBertTokenizer.from_pretrained("chatbot_tokenizer")

# Load intents and tags
with open('intents.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)
with open('chatbot_tags.json', 'r', encoding='utf-8') as f:
    tag_data = json.load(f)

tags = tag_data["tags"]
idx_to_label = tag_data["idx_to_label"]

def get_response(msg):
    inputs = tokenizer(msg, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    outputs = model(**inputs)
    _, predicted = torch.max(outputs.logits, dim=1)
    tag = idx_to_label[str(predicted.item())]

    probs = torch.softmax(outputs.logits, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    return "Maaf pesan yang anda kirim kurang dimengerti"
