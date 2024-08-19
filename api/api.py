import logging
import os
import sys
from pathlib import Path

from fastapi import FastAPI
from transformers import GPT2Tokenizer, GPT2LMHeadModel

logging.basicConfig(format="%(levelname)s:%(name)s:%(message)s", level=logging.DEBUG, stream=sys.stdout)

log = logging.getLogger(__name__)

description = """
This is part of the Centific Homework for Jon Osborn 🚀
"""

app = FastAPI(title="Centific Homework - Osborn",
              description=description,
              summary="Jon Osborn's Centific Homework",
              version="0.0.1",
              contact={
                  "name": "Jon Osborn",
                  "url": "https://www.linkedin.com/in/jonosborn/",
                  "email": "osborn.jon.20@gmail.com",
              },
              license_info={
                  "name": "Apache 2.0",
                  "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
              }, )

model_path = os.environ.get('MODEL_PATH', 'build/trained_model.pth')
token_path = os.environ.get('TOKEN_PATH', 'build/tokenizer.pth')
log.debug(f'loading models {model_path}:{Path(model_path).resolve().exists()} and tokens {token_path}:{Path(token_path).resolve().exists()}')
# Load the trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(token_path)
log.debug('models loaded')


@app.get("/", tags=["root"])
async def root():
    return {"message": "Welcome to my Centific homework",
            "version": 1,
            }


@app.get('/response', tags=['LLM'], description="Retrieve a response")
async def get_response(statement: str = None):
    if statement:
        return {"response": statement}
    pass
