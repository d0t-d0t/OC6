from fastapi import FastAPI, HTTPException, Form, Request, status
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
import uvicorn
import pickle
# import tensorflow as tf

pipeline=None
try:
    from Training.TweetClassifier import TweetClassifierPipeline
    pipeline = TweetClassifierPipeline
except Exception as e:
    print(f"Error loading model: {e}")

from Deployment.TweetModel import Tweet, Feedback
import os

app = FastAPI()

if type(pipeline) != type(None):
    try:
        model_path = r'.\Deployment\Models\model_tfidf.pkl'
        model_path = os.path.join('.', 'Deployment', 'Models', 'model_tfidf.pkl')
        latest_model_in = open(model_path, "rb")
        latest_model = pickle.load(latest_model_in)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to load the model")


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

@app.post("/user_feedback/")
def get_user_feedback(tweet: Tweet, feedback :  Feedback):

    # Placeholder function to save feedback to database
    def save_feedback_to_database(tweet, feedback):
        pass

    
    save_feedback_to_database(tweet, feedback)

    return {"message": "Feedback received and saved successfully"}

if __name__ == "__main__":
    uvicorn.run('main:app', host='0.0.0.0', port=8000)

    # data = {"tweet": str('test')}
    
    # API_ENDPOINT = "http://localhost:8000/predict/"
    # response = requests.post(API_ENDPOINT,data=data)

    