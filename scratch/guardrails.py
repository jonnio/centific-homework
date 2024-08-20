import nemo.collections.nlp as nemo_nlp
import pandas as pd
from nemo.collections.nlp.models.text_classification import TextClassificationModel
from nemo.collections.nlp.data.text_classification import TextClassificationDataset
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
import torch

# Define guardrails for preprocessing
def preprocess_text(text):
    if not text or not isinstance(text, str):
        raise ValueError("Invalid text input. Text cannot be empty or non-string.")
    return text.strip()

def check_labels(labels):
    valid_labels = {'Closed with explanation', 'Closed with non-monetary relief', 'Other'}  # Example set of valid labels
    # if any(label not in valid_labels for label in labels):
    #     raise ValueError("Invalid label found in dataset.")
    return labels

# Load the tokenizer
tokenizer = get_tokenizer(
    tokenizer_name='bert-base-uncased',
    tokenizer_model='bert-base-uncased',
    # do_lower_case=True
)

# Load the new dataset
file_path_large = '../api/build/Consumer_Complaints_train.csv'
df_large = pd.read_csv(file_path_large)

# Display the first few rows to understand the structure
df_large.head(), df_large.shape

# Selecting relevant columns and dropping rows with missing values
df_large_cleaned = df_large[['Consumer complaint narrative', 'Company response to consumer']].dropna()

# Display the cleaned data and its shape
df_large_cleaned.head(), df_large_cleaned.shape


# Apply guardrails during data preparation
texts = [preprocess_text(text) for text in df_large_cleaned['Consumer complaint narrative'].tolist()]
labels = check_labels(df_large_cleaned['Company response to consumer'].tolist())

ds = TextClassificationDataset(tokenizer=tokenizer,
                               input_file=None,
                               )
# Create a dataset object
dataset = TextClassificationDataset(
    tokenizer=tokenizer,
    input_file=None,
    text_column='Consumer complaint narrative',
    label_column='Company response to consumer',
    max_seq_length=128,  # Ensure the sequence length is appropriate
    use_cache=False,
    delimiter=',',
    num_samples=-1,
    pad_label=-100
)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Validate dataset splitting
if len(train_dataset) == 0 or len(val_dataset) == 0:
    raise ValueError("Training or validation dataset is empty after splitting.")

# Initialize the TextClassificationModel with guardrails
model = TextClassificationModel.from_pretrained(model_name="bert-base-uncased")

# Set up the PyTorch Lightning Trainer
from pytorch_lightning import Trainer

trainer = Trainer(max_epochs=3, gpus=1)

# Train the model with guardrails
try:
    trainer.fit(model, train_dataloaders=train_dataset, val_dataloaders=val_dataset)
except Exception as e:
    raise RuntimeError(f"Training failed: {e}")

# After training, save the model
model.save_to("trained_model.nemo")

# Verify that the model was saved
import os
if not os.path.exists("trained_model.nemo"):
    raise FileNotFoundError("Model saving failed. File not found.")
