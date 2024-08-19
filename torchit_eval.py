import logging
import sys

import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

logging.basicConfig(format="%(levelname)s:%(name)s:%(message)s", level=logging.DEBUG, stream=sys.stdout)

log = logging.getLogger(__name__)

log.debug('loading models')
# Load the trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained('build/trained_model.pth')
tokenizer = GPT2Tokenizer.from_pretrained('build/tokenizer.pth')

# Set the model to evaluation mode
model.eval()

test_data_path = 'build/Consumer_Complaints_test.csv'
test_df = pd.read_csv(test_data_path)
test_df_cleaned = test_df.dropna(subset=['Consumer complaint narrative', 'Company response to consumer'])

# Tokenize the test complaints
log.debug('tokenizing')
test_encodings = tokenizer(test_df_cleaned['Consumer complaint narrative'].tolist(),
                           padding="max_length",
                           truncation=True,
                           max_length=128,
                           return_tensors="pt")

log.debug('building test responses')
# Generate responses for the first few examples in the test set
generated_responses = []
for i in range(5):  # Adjust this number to generate more/less responses
    input_ids = test_encodings['input_ids'][i].unsqueeze(0)  # Add batch dimension
    attention_mask = test_encodings['attention_mask'][i].unsqueeze(0)  # Add batch dimension

    # Generate the response
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=150, num_beams=5, early_stopping=True)

    # Decode the generated response
    generated_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_responses.append(generated_response)

# Print out some sample results
for i, response in enumerate(generated_responses):
    print(f"Complaint: {test_df_cleaned['Consumer complaint narrative'].iloc[i]}")
    print(f"Generated Response: {response}")
    print("\n")
