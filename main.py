from fastapi import FastAPI, HTTPException, Form, Request, status
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
# import pickle
# from Training.TweetClassifier import TweetClassifierPipeline
# from Deployment.TweetModel import Tweet
# import os

app = FastAPI()
# app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# model_path = r'.\Deployment\Models\model_tfidf.pkl'
# model_path = os.path.join('.', 'Deployment', 'Models', 'model_tfidf.pkl')
# latest_model_in = open(model_path, "rb")
# latest_model = pickle.load(latest_model_in)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    print('Request for index page received')
    return templates.TemplateResponse('index.html', {"request": request})


# @app.post("/predict/")
# def get_prediction(tweet: Tweet):
#     # tweet = Tweet(tweet_dic)
#     predictions = []
#     probabilities = []


#     prediction = latest_model.predict(tweet.text)
#     try:
#         probability = latest_model.predict_proba(tweet.text)
#         if probability.shape[1] == 2:
#             probability = probability[:, 1]
#     except:
#         probability = None




#     return {
#         "prediction": int(prediction[0]),
#         "probabilitie": float(probability[0]),
#     }

if __name__ == "__main__":
    uvicorn.run('main:app', host='0.0.0.0', port=8000)

    # data = {"tweet": str('test')}
    
    # API_ENDPOINT = "http://localhost:8000/predict/"
    # response = requests.post(API_ENDPOINT,data=data)

    