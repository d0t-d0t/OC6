from fastapi import FastAPI, HTTPException
import pickle
from typing import List

import requests
from Training.TweetClassifier import TweetClassifierPipeline
from Deployment.TweetModel import Tweet





app = FastAPI()

model_path = r'.\Deployment\Models\model_tfidf.pkl'
latest_model_in = open(model_path, "rb")
latest_model = pickle.load(latest_model_in)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict/")
def get_prediction(tweet: Tweet):
    # tweet = Tweet(tweet_dic)
    predictions = []
    probabilities = []


    prediction = latest_model.predict(tweet.text)
    try:
        probability = latest_model.predict_proba(tweet.text)
        if probability.shape[1] == 2:
            probability = probability[:, 1]
    except:
        probability = None




    return {
        "prediction": int(prediction[0]),
        "probabilitie": float(probability[0]),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)

    # data = {"tweet": str('test')}
    
    # API_ENDPOINT = "http://localhost:8000/predict/"
    # response = requests.post(API_ENDPOINT,data=data)

    