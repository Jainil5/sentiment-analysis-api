
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch

import_model = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
tokenizer  = AutoTokenizer.from_pretrained(import_model)
model =AutoModelForSequenceClassification.from_pretrained(import_model)


def score(x):
    token = tokenizer.encode(x,return_tensors ="pt")
    result = model(token)
    score = int(torch.argmax(result.logits)) + 1
    return score

print(score("Too bad"))