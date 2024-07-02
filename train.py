import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW

# Load data intents
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

# Preparing data
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
all_texts = []
all_labels = []
tags = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        all_texts.append(pattern)
        all_labels.append(tag)

tags = sorted(set(tags))
label_to_idx = {tag: idx for idx, tag in enumerate(tags)}
idx_to_label = {idx: tag for tag, idx in label_to_idx.items()}

# Tokenizing
encodings = tokenizer(all_texts, truncation=True, padding=True, max_length=512)
labels = [label_to_idx[label] for label in all_labels]

class ChatDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

dataset = ChatDataset(encodings, labels)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(tags))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.train()

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 50
for epoch in range(num_epochs):
    correct = 0
    total = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')

# Save the model and tokenizer
model.save_pretrained("chatbot_model")
tokenizer.save_pretrained("chatbot_tokenizer")

# Save tags and mappings
with open('chatbot_tags.json', 'w', encoding='utf-8') as f:
    json.dump({"tags": tags, "label_to_idx": label_to_idx, "idx_to_label": idx_to_label}, f)
