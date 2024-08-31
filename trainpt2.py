import torch
import json
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib.pyplot as plt

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Sastrawi stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Load intents
with open('intents.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

# Validate intents
if 'intents' not in intents:
    raise ValueError("File intents.json tidak mengandung kunci 'intents'.")

# Preprocessing function
def preprocess_text(text):
    # Case folding
    text = text.lower()
    # Tokenizing
    words = word_tokenize(text)
    # Stopword removal
    stop_words = set(stopwords.words('indonesian'))
    words = [word for word in words if word not in stop_words]
    # Stemming
    words = [stemmer.stem(word) for word in words]
    # Join words back to string
    return ' '.join(words)

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
                preprocessed_pattern = preprocess_text(pattern)
                self.inputs.append(preprocessed_pattern)
                self.labels.append(self.tags.index(tag))

        self.inputs = [self.tokenizer(input_text, truncation=True, padding='max_length', max_length=64, return_tensors='pt') for input_text in self.inputs]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        item = self.inputs[idx]
        label = self.labels[idx]
        return {key: val.squeeze(0) for key, val in item.items()}, torch.tensor(label)

# Initialize tokenizer and dataset
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
dataset = ChatDataset(tokenizer, intents)

# Parameters
batch_size = 16
learning_rate = 5e-5
epochs = 15

# DataLoader
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(dataset.tags))
model.train()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)

# List to store loss and accuracy per epoch
losses = []
accuracies = []

# Training loop
for epoch in range(epochs):
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
    
    # Save the metrics for each epoch
    losses.append(avg_loss)
    accuracies.append(accuracy)
    
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Plotting the loss
plt.figure(figsize=(10, 4))
plt.plot(range(1, epochs + 1), losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss per Epoch')
plt.legend()
plt.show()

# Plotting the accuracy
plt.figure(figsize=(10, 4))
plt.plot(range(1, epochs + 1), accuracies, label='Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy per Epoch')
plt.legend()
plt.show()

# Save model
model_data = {
    "model_state": model.state_dict(),
    "tags": dataset.tags,
    "tokenizer_name": tokenizer.name_or_path
}
torch.save(model_data, "data.pth")
print("Training complete. Model saved to data.pth")
