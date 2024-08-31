import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Sastrawi stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    words = word_tokenize(text)
    stop_words = set(stopwords.words('indonesian'))
    words = [word for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

# Load model
model_data = torch.load("data.pth")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(model_data["tags"]))
model.load_state_dict(model_data["model_state"])

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(model_data["tokenizer_name"])

# Load testing data
with open('testing20%.json', 'r', encoding='utf-8') as json_data:
    testing_data = json.load(json_data)

# Validate testing data
if 'intents' not in testing_data:
    raise ValueError("File testing.json tidak mengandung kunci 'intents'.")

# Prepare data for testing
class ChatTestDataset(Dataset):
    def __init__(self, tokenizer, intents, tags):
        self.tokenizer = tokenizer
        self.inputs = []
        self.labels = []
        self.tags = tags

        for intent in intents['intents']:
            tag = intent['tag']
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

# Initialize test dataset and DataLoader
test_dataset = ChatTestDataset(tokenizer, testing_data, model_data["tags"])
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Move model to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Evaluation loop
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch
        inputs = {key: val.to(device) for key, val in inputs.items()}
        labels = labels.to(device)

        outputs = model(**inputs)
        _, preds = torch.max(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute confusion matrix and metrics
cm = confusion_matrix(all_labels, all_preds, labels=range(len(model_data["tags"])))
report = classification_report(all_labels, all_preds, target_names=model_data["tags"], labels=range(len(model_data["tags"])), output_dict=True)

# Print metrics
print("Classification Report:\n", classification_report(all_labels, all_preds, target_names=model_data["tags"], labels=range(len(model_data["tags"]))))

# Plot confusion matrix
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model_data["tags"], yticklabels=model_data["tags"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
