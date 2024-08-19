import pandas as pd

# Load the datasets
train_data = pd.read_csv('Consumer_Complaints_train.csv')
test_data = pd.read_csv('Consumer_Complaints_test.csv')

# Display the first few rows of the datasets to understand their structure
print("Training Data:")
print(train_data.head())

print("\nTest Data:")
print(test_data.head())

## PREPROCESS

import re
import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab


# Function to clean and tokenize text
def preprocess_text(text):
    # Lowercase
    if isinstance(text, str):
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
    return text


# Apply the preprocessing to the text column in your data
train_data['processed_text'] = train_data['Consumer complaint narrative'].apply(preprocess_text)
test_data['processed_text'] = test_data['Consumer complaint narrative'].apply(preprocess_text)

# Tokenizer
tokenizer = get_tokenizer('basic_english')

# Tokenizing the processed text
train_data['tokenized_text'] = train_data['processed_text'].apply(tokenizer)
test_data['tokenized_text'] = test_data['processed_text'].apply(tokenizer)

# Building the vocabulary from the training data
counter = Counter()
for tokens in train_data['tokenized_text']:
    counter.update(tokens)
vocab = Vocab(counter, min_freq=1)


# Example: Converting a sentence to a tensor of word indices
def text_pipeline(text):
    return torch.tensor([vocab[token] for token in tokenizer(preprocess_text(text))], dtype=torch.long)


# Apply the text_pipeline to convert text to tensors
train_data['text_tensor'] = train_data['processed_text'].apply(text_pipeline)
test_data['text_tensor'] = test_data['processed_text'].apply(text_pipeline)

## CREATE DATASETS
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return text, label


# Assuming 'label_column_name' is the name of the label column
train_dataset = TextDataset(train_data['text_tensor'].tolist(), train_data['label_column_name'].tolist())
test_dataset = TextDataset(test_data['text_tensor'].tolist(), test_data['label_column_name'].tolist())

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

## DEFINE THE MODEL
import torch.nn as nn
import torch.nn.functional as F


class SimpleTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super(SimpleTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, text):
        embedded = self.embedding(text)
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)
        return self.fc(pooled)


# Instantiate the model
vocab_size = len(vocab)
embedding_dim = 100  # You can adjust this
num_classes = len(train_data['label_column_name'].unique())  # Assuming labels are categorical

model = SimpleTextClassifier(vocab_size, embedding_dim, num_classes)

## TRAIN THE MODEL
from torch.optim import Adam

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5  # Adjust as needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct_predictions = 0

    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels).item()

    accuracy = correct_predictions / len(train_loader.dataset)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}, Accuracy: {accuracy}")

## EVALUATE THE MODEL
model.eval()
total_loss = 0
correct_predictions = 0

with torch.no_grad():
    for texts, labels in test_loader:
        texts, labels = texts.to(device), labels.to(device)

        outputs = model(texts)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels).item()

accuracy = correct_predictions / len(test_loader.dataset)
print(f"Test Loss: {total_loss / len(test_loader)}, Test Accuracy: {accuracy}")
