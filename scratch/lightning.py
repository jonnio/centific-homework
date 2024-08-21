import logging
import re
import sys

import pandas as pd

logging.basicConfig(format="%(levelname)s:%(name)s:%(message)s", level=logging.INFO, stream=sys.stdout)

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

import torch
from torch.utils.data import Dataset


class ConsumerComplaintsDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
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


def main():
    # Load the CSV file
    file_path = '../build/Consumer_Complaints_train.csv'
    df = pd.read_csv(file_path).head(10000)

    # Display the first few rows
    log.debug(df.head(5))

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    # Step 3: Handle missing values by dropping rows where 'Consumer complaint narrative' is missing
    df.dropna(subset=['Consumer complaint narrative'], inplace=True)  # .head(10000)
    log.debug(f'{df.shape}')

    SPECIAL_CHARS = re.compile('[^A-Za-z0-9\s.$,%]+')

    # Step 4: Text preprocessing
    def clean_text(text):
        text = text.lower().strip()  # Convert to lowercase
        text = SPECIAL_CHARS.sub('', text)
        # text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        return text

    df['Consumer complaint narrative'] = df['Consumer complaint narrative'].apply(clean_text)

    # Step 5: Encode labels
    label_encoder = LabelEncoder()
    df['Product'] = label_encoder.fit_transform(df['Product'])

    # Step 6: Split the data into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Product'])

    # Display the shapes of the resulting DataFrames
    print(train_df.shape, val_df.shape)

    from torch.utils.data import DataLoader
    from transformers import BertTokenizer

    # Initialize a tokenizer (assuming we're using a BERT-based model)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create the datasets
    train_dataset = ConsumerComplaintsDataset(train_df, tokenizer)
    val_dataset = ConsumerComplaintsDataset(val_df, tokenizer)

    import multiprocessing

    cpu_count = multiprocessing.cpu_count()
    log.debug(f'Using {cpu_count - 1} CPUs for data loading')
    # Create the DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=cpu_count - 1)
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=cpu_count - 1)

    import pytorch_lightning as pl
    from transformers import BertForSequenceClassification

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
            self.log('train_loss', loss)
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

    num_classes = train_df['Product'].nunique()
    model = ComplaintClassifier(num_classes=num_classes)

    log.debug('setting up trainer')
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model, train_loader, val_loader)

    model_save_path = 'complaint_classifier_model.pt'
    torch.save(model.state_dict(), model_save_path)

    # Load the saved model
    # model = ComplaintClassifier(num_classes=num_classes)
    # model.load_state_dict(torch.load(model_save_path))
    # model.eval()  # Set the model to evaluation mode

    # def predict(text, model, tokenizer, max_length=256):
    #     model.eval()
    #     encoding = tokenizer(
    #         text,
    #         add_special_tokens=True,
    #         max_length=max_length,
    #         padding='max_length',
    #         truncation=True,
    #         return_tensors='pt'
    #     )
    #
    #     input_ids = encoding['input_ids']
    #     attention_mask = encoding['attention_mask']
    #
    #     with torch.no_grad():
    #         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    #         logits = outputs[0]
    #         predicted_class_id = torch.argmax(logits, dim=1).item()
    #
    #     return predicted_class_id
    #
    # # Example usage:
    # sample_text = "I have a complaint about my credit card charges."
    # predicted_class = predict(sample_text, model, tokenizer)
    # print(f"Predicted class: {predicted_class}")


if __name__ == '__main__':
    main()
