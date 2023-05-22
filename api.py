import requests
from flask import Flask
from flask_restful import Api,Resource,reqparse
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch


app = Flask(__name__)
api = Api(app)

import_model = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
tokenizer  = AutoTokenizer.from_pretrained(import_model)
model =AutoModelForSequenceClassification.from_pretrained(import_model)

def score(x):
    #1 bad, 2 neutral, 3 good
    token = tokenizer.encode(x,return_tensors ="pt")
    result = model(token)
    score = int(torch.argmax(result.logits)) + 1
    return score

class Request(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('input', type=str, required=True)
        args = parser.parse_args()        
        input = args['input']
        response = str(score(input))
        return {"response":response}

api.add_resource(Request,"/sentiment")

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0", port=5000)
    
