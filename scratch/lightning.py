import logging
import pickle
import re
import sys
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import typer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import BertForSequenceClassification
from transformers import BertTokenizer

logging.basicConfig(format="%(levelname)s:%(name)s:%(message)s", level=logging.INFO, stream=sys.stdout)

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

app = typer.Typer()


class ConsumerComplaintsDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        complaint = self.data.iloc[index]['Consumer complaint narrative']
        label = self.data.iloc[index]['Product']

        # Tokenize the text
        encoding = self.tokenizer(
            complaint,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }


class ComplaintClassifier(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=2e-5):
        super(ComplaintClassifier, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
        self.learning_rate = learning_rate

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        outputs = self(input_ids, attention_mask)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        outputs = self(input_ids, attention_mask)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

# special character removal (keep these)
SPECIAL_CHARS = re.compile('[^A-Za-z0-9\s.$,%]+')

@app.command()
def train():
    # Load the CSV file
    file_path = '../build/Consumer_Complaints_train.csv'
    df = pd.read_csv(file_path)

    # Display the first few rows
    log.debug(df.head(5))
    log.debug(f'{df.shape}')

    # Step 3: Handle missing values by dropping rows where 'Consumer complaint narrative' or 'Product' is missing
    df.dropna(subset=['Consumer complaint narrative', 'Product'], inplace=True)
    log.debug(f'{df.shape}')

    # Step 4: Text preprocessing
    def clean_text(text):
        text = text.lower().strip()  # Convert to lowercase
        text = SPECIAL_CHARS.sub('', text)
        return text

    df['Consumer complaint narrative'] = df['Consumer complaint narrative'].apply(clean_text)

    # Step 5: Encode labels
    label_encoder = LabelEncoder()
    df['Product'] = label_encoder.fit_transform(df['Product'])

    # save the label encoder so we can reuse it during inference
    with open('../build/label_encoder.pkl', 'wb') as pkl_file:
        pickle.dump(label_encoder, pkl_file)

    # Step 6: Split the data into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.8, random_state=42, stratify=df['Product'])

    # Display the shapes of the resulting DataFrames
    print(train_df.shape, val_df.shape)

    # Initialize a tokenizer (assuming we're using a BERT-based model)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create the datasets
    train_dataset = ConsumerComplaintsDataset(train_df, tokenizer)
    val_dataset = ConsumerComplaintsDataset(val_df, tokenizer)

    import multiprocessing

    cpu_count = multiprocessing.cpu_count()
    log.debug(f'Using {cpu_count - 1} CPUs for data loading')
    # Create the DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=cpu_count - 1)
    val_loader = DataLoader(val_dataset, batch_size=8, num_workers=cpu_count - 1)

    num_classes = train_df['Product'].nunique()
    model = ComplaintClassifier(num_classes=num_classes)
    log.debug(f'num_classes: {num_classes}')

    log.debug('setting up trainer')
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model, train_loader, val_loader)

    log.debug('training complete')
    model_save_path = '../build/complaint_classifier_model.pt'
    torch.save(model.state_dict(), model_save_path)
    log.debug('torch save complete')

@app.command()
def infer(sample_text: str):
    model = ComplaintClassifier(num_classes=11)
    log.debug(f"'{Path('../build/complaint_classifier_model.pt').exists()}'")
    model.load_state_dict(torch.load('../build/complaint_classifier_model.pt'))
    model.eval()  # Set the model to evaluation mode

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # load LabelEncoder
    with open('../build/label_encoder.pkl', 'rb') as pkl_file:
        label_encoder = pickle.load(pkl_file)

    def predict(text, model, tokenizer, max_length=128):
        model.eval()
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs[0]
            predicted_class_id = torch.argmax(logits, dim=0).item()

        return predicted_class_id

    # Example usage:
    predicted_class = predict(sample_text, model, tokenizer)
    print(f"Predicted class: {predicted_class}")
    print(f"Label: {label_encoder.inverse_transform([predicted_class])}")


if __name__ == '__main__':
    app()
