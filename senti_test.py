
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch

import_model = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
tokenizer  = AutoTokenizer.from_pretrained(import_model)
model =AutoModelForSequenceClassification.from_pretrained(import_model)


def score(x):
    #1 bad, 2 neutral, 3 good
    token = tokenizer.encode(x,return_tensors ="pt")
    result = model(token)
    score = int(torch.argmax(result.logits)) + 1
    return score

print(score("extremely happy with the work"))