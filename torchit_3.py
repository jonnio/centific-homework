import json
import logging
import sys

import pandas as pd

logging.basicConfig(format="%(levelname)s:%(name)s:%(message)s", level=logging.DEBUG, stream=sys.stdout)

log = logging.getLogger(__name__)

# Load the CSV files
train_data_path = 'build/Consumer_Complaints_train.csv'
test_data_path = 'build/Consumer_Complaints_test.csv'

log.debug('loading files')
# Read the data into DataFrames
train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)

# Display the first few rows of each DataFrame to understand the structure
train_df.head(), test_df.head()

# Filter out rows where either "Consumer complaint narrative" or "Company response to consumer" is missing
train_df_cleaned = train_df.dropna(subset=['Consumer complaint narrative', 'Company response to consumer']).head(10000)
# test_df_cleaned = test_df.dropna(subset=['Consumer complaint narrative', 'Company response to consumer'])

# Display the number of rows remaining after cleaning
log.debug("%s", train_df_cleaned.shape)  # , test_df_cleaned.shape)

from transformers import (GPT2Tokenizer, GPT2LMHeadModel)
from torch.optim import AdamW

log.debug('tokenizing')

# Initialize GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

log.debug('tokenizing train complaints')
# Tokenize the consumer complaint narratives and company responses
train_encodings = tokenizer(train_df_cleaned['Consumer complaint narrative'].tolist(),
                            padding="max_length",
                            truncation=True,
                            max_length=512,
                            return_tensors="pt")

# log.debug('tokenizing test complaints')
# test_encodings = tokenizer(test_df_cleaned['Consumer complaint narrative'].tolist(),
#                            padding=True,
#                            truncation=True,
#                            max_length=512,
#                            return_tensors="pt")

log.debug('tokenizing train company response')
# Prepare labels for the responses (tokenized as well)
train_labels = tokenizer(train_df_cleaned['Company response to consumer'].tolist(),
                         padding="max_length",
                         truncation=True,
                         max_length=512,
                         return_tensors="pt")

# log.debug('tokenizing test company response')
# test_labels = tokenizer(test_df_cleaned['Company response to consumer'].tolist(),
#                         padding=True,
#                         truncation=True,
#                         max_length=512,
#                         return_tensors="pt")

# Display a sample of tokenized data
log.debug(json.dumps({
    "input_ids_sample": str(train_encodings['input_ids'][0][:10]),  # Show first 20 tokens of a sample input
    "attention_mask_sample": str(train_encodings['attention_mask'][0][:10]),  # Corresponding attention mask
    "label_ids_sample": str(train_labels['input_ids'][0][:10]),  # Show first 20 tokens of a sample label
    "input_shape": train_encodings['input_ids'].shape,
    "label_shape": train_labels['input_ids'].shape,
}, indent=2))

from torch.utils.data import Dataset, DataLoader


# Define Dataset Class
class CustomDataset(Dataset):
    def __init__(self, encodings, d_labels):
        self.encodings = encodings
        self.d_labels = d_labels

    def __getitem__(self, idx):
        # item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        # item['labels'] = torch.tensor(self.labels['input_ids'][idx])
        item['labels'] = self.d_labels['input_ids'][idx].clone().detach()
        return item

    def __len__(self):
        return len(self.d_labels['input_ids'])
        # return len(self.encodings)


# Create DataLoader
train_dataset = CustomDataset(train_encodings, train_labels)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)

# Initialize the model
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

# Training Setup
optimizer = AdamW(model.parameters(), lr=5e-5)

log.debug(f'starting model training for {len(train_loader)} steps')

# Training Loop
model.train()
for epoch in range(1):  # Adjust the number of epochs as needed
    log.debug(f'Training model epoch {epoch}')
    for step, batch in enumerate(train_loader):
        inputs, labels = batch['input_ids'], batch['labels']
        optimizer.zero_grad()
        outputs = model(input_ids=inputs, labels=labels, attention_mask=batch['attention_mask'])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        log.info(f"Step {step}, Epoch {epoch + 1}, Loss: {loss.item()}")

# Save the model
log.debug('saving pretrained model')
tokenizer.save_pretrained('build/tokenizer.pth')
model.save_pretrained('build/trained_model.pth')


def build_model():
    pass


def test_model():
    pass


if __name__ == '__main__':
    build_model()
    test_model()
