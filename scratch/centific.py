# import nemo.collections.nlp as nemo_nlp
import logging

import pandas as pd
import torch
from nemo.collections.nlp.data.text_classification import TextClassificationDataset
from nemo.collections.nlp.models.text_classification import TextClassificationModel
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

logging.basicConfig(format="%(levelname)s:%(name)s:%(message)s", level=logging.DEBUG, )
logging.warning("Hello, Warning!")

# Load the new dataset
file_path_large = 'Consumer_Complaints_train.csv'
df_large = pd.read_csv(file_path_large)

# Display the first few rows to understand the structure
df_large.head(), df_large.shape

# Selecting relevant columns and dropping rows with missing values
df_large_cleaned = df_large[['Consumer complaint narrative', 'Company response to consumer']].dropna()

# Display the cleaned data and its shape
df_large_cleaned.head(), df_large_cleaned.shape

import torch
from torch.utils.data import Dataset

class TextScoreDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

import nemo.collections.nlp as nemo_nlp
# from nemo.collections.nlp.models import TextClassificationModel
# from nemo.collections.nlp.data.text_classification import TextClassificationDataset
# from nemo.collections.nlp.modules.common import Tokenizer

# Initialize tokenizer and model
nemo_nlp.models.TextClassificationModel.list_available_models()
tokenizer = nemo_nlp.modules.common.get_tokenizer('bert-base-uncased')
model = nemo_nlp.models.TextClassificationModel.from_pretrained('bert-base-uncased')

# Load your data
import pandas as pd

df = pd.read_csv('your_data.csv')
texts = df.iloc[:, 0].tolist()
labels = df.iloc[:, 2].tolist()

# Create a dataset
dataset = TextScoreDataset(texts, labels, tokenizer)

# Split into training and testing datasets
from torch.utils.data import random_split

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
