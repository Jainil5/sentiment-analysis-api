import requests
from flask import Flask
from flask_restful import Api,Resource
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer

app = Flask(__name__)
api = Api(app)


###  Tokenization

# sentences = ["I love my dog","I love my cat","You love my dog!","Do you think my dog is amazing?"]

# tokenizer = Tokenizer(num_words=100,oov_token="<OOV>")
# tokenizer.fit_on_texts(sentences)
# word_index = tokenizer.word_index 
# print(word_index)
# sequences = tokenizer.texts_to_sequences(sentences)
# print(sequences)


import_model = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
tokenizer  = AutoTokenizer.from_pretrained(import_model)
model =AutoModelForSequenceClassification.from_pretrained(import_model)

def score(x):
    token = tokenizer.encode(x,return_tensors ="pt")
    result = model(token)
    score = int(torch.argmax(result.logits)) 
    sentiment = ["negative","neutral","positive"]
    return sentiment[score]

class Request(Resource):
    def post(self,data):
        response = str(score(data))
        return {"sentiment" : response}

api.add_resource(Request,"/sentiment/<string:data>")

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0", port=5000)
    
