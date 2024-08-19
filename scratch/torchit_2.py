import logging
import re

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

logging.basicConfig(format="%(levelname)s:%(name)s:%(message)s", level=logging.DEBUG, )

log = logging.getLogger(__name__)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Load your data (replace with your actual file paths)
train_data = pd.read_csv('Consumer_Complaints_train.csv')
# test_data = pd.read_csv('Consumer_Complaints_test.csv')

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()


# Function to clean text
def clean_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        # Lemmatize
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        # Join tokens back into a single string
        cleaned_text = ' '.join(tokens)
        return cleaned_text


# Apply cleaning to the datasets
# Selecting relevant columns and dropping rows with missing values
log.debug('cleaning data')
df_large_cleaned = train_data[['Consumer complaint narrative', 'Company response to consumer']].dropna()

# Display the cleaned data and its shape
df_large_cleaned.head(), df_large_cleaned.shape

# Apply guardrails during data preparation
texts = [clean_text(text) for text in df_large_cleaned['Consumer complaint narrative'].tolist()]
labels = df_large_cleaned['Company response to consumer'].tolist()

# train_data['cleaned_text'] = train_data[['Consumer complaint narrative', 'Company response to consumer']].apply(clean_text).dropna()
# test_data['cleaned_text'] = test_data[['Consumer complaint narrative', 'Company response to consumer']].apply(clean_text).dropna()

# Save the cleaned data (optional)
# train_data.to_csv('cleaned_train_data.csv', index=False)
# test_data.to_csv('cleaned_test_data.csv', index=False)


import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW


# Custom Dataset Class
class ComplaintsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# Load Pretrained Tokenizer
log.debug('loading the tokenizer')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prepare Datasets
train_dataset = ComplaintsDataset(
    texts=np.array(texts),
    labels=np.array(labels),
    tokenizer=tokenizer,
    max_len=128
)

# test_dataset = ComplaintsDataset(
#     texts=test_data['Consumer complaint narrative'].to_numpy(),
#     labels=test_data['Company response to consumer'].to_numpy(),
#     tokenizer=tokenizer,
#     max_len=128
# )

# Data Loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load Pretrained BERT Model
log.debug('loading pretrained model')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # Adjust `num_labels` as per your classification task

# Optimizer and Loss Function
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Training Loop
model.train()
for epoch in range(3):  # Adjust the number of epochs as needed
    log.debug(f'training epoch {epoch}')
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{3}, Loss: {total_loss / len(train_loader)}')

# Save the model (optional)
torch.save(model.state_dict(), 'bert_classification_model.pth')

# model.eval()
# correct_predictions = 0
# with torch.no_grad():
#     for batch in test_loader:
#         input_ids = batch['input_ids']
#         attention_mask = batch['attention_mask']
#         labels = batch['label']
#
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#         _, preds = torch.max(outputs.logits, dim=1)
#
#         correct_predictions += torch.sum(preds == labels)
#
# accuracy = correct_predictions.double() / len(test_loader.dataset)
# print(f'Test Accuracy: {accuracy:.4f}')
