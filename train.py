import torch
import json
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW

# Load intents
with open('intents.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

# Prepare data
class ChatDataset(Dataset):
    def __init__(self, tokenizer, intents):
        self.tokenizer = tokenizer
        self.inputs = []
        self.labels = []
        self.tags = []
        
        for intent in intents['intents']:
            tag = intent['tag']
            if tag not in self.tags:
                self.tags.append(tag)
            for pattern in intent['patterns']:
                self.inputs.append(pattern)
                self.labels.append(self.tags.index(tag))
        
        self.inputs = [self.tokenizer(input_text, truncation=True, padding='max_length', max_length=64) for input_text in self.inputs]
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        item = self.inputs[idx]
        label = self.labels[idx]
        return {key: torch.tensor(val) for key, val in item.items()}, torch.tensor(label)

# Initialize tokenizer and dataset
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
dataset = ChatDataset(tokenizer, intents)

# Parameters
batch_size = 8
learning_rate = 5e-5
num_epochs = 10

# DataLoader
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(dataset.tags))
model.train()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        inputs, labels = batch
        inputs = {key: val.to(device) for key, val in inputs.items()}
        labels = labels.to(device)
        
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        _, preds = torch.max(outputs.logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_predictions += labels.size(0)
        
        loss.backward()
        optimizer.step()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = (correct_predictions / total_predictions).item() * 100
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Save model
model_data = {
    "model_state": model.state_dict(),
    "tags": dataset.tags,
    "tokenizer_name": tokenizer.name_or_path
}
torch.save(model_data, "data.pth")
print("Training complete. Model saved to data.pth")
