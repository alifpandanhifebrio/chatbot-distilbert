# nltk_utils.py
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

factory = StemmerFactory()
stemmer = factory.create_stemmer()
indonesian_stopwords = nltk.corpus.stopwords.words('indonesian')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

def tokenize(sentence):
    # Tokenize sentence with DistilBERT tokenizer
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=128)
    return inputs

def stem(word):
    return stemmer.stem(word)

def bag_of_words(tokenized_sentence, words):
    # Get embeddings from DistilBERT model
    with torch.no_grad():
        outputs = model(**tokenized_sentence)
    embeddings = outputs.last_hidden_state.squeeze(0)

    # Create bag-of-words representation
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, word in enumerate(words):
        word_inputs = tokenizer(word, return_tensors='pt', truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            word_outputs = model(**word_inputs)
        word_embedding = word_outputs.last_hidden_state.squeeze(0).mean(dim=0)
        
        # Check if word embedding is similar to any token in the sentence
        similarity = torch.cosine_similarity(embeddings, word_embedding.unsqueeze(0).repeat(embeddings.size(0), 1), dim=1)
        if similarity.max().item() > 0.9:  # You can adjust the threshold
            bag[idx] = 1
    return bag
