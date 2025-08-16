from fastapi import FastAPI, HTTPException, Form, Request, status
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import pickle

try:
    from Training.TweetClassifier import TweetClassifierPipeline
    pipeline = TweetClassifierPipeline
except Exception as e:
    pipeline = None
    print(f"Error loading model: {e}")

from Deployment.TweetModel import Tweet
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

if type(pipeline) != type(None):
    try:
        model_path = r'.\Deployment\Models\model_tfidf.pkl'
        model_path = os.path.join('.', 'Deployment', 'Models', 'model_tfidf.pkl')
        latest_model_in = open(model_path, "rb")
        latest_model = pickle.load(latest_model_in)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to load the model")




@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    print('Request for index page received')
    return templates.TemplateResponse('index.html', {"request": request})

@app.get('/favicon.ico')
async def favicon():
    file_name = 'favicon.ico'
    file_path = './static/' + file_name
    return FileResponse(path=file_path, headers={'mimetype': 'image/vnd.microsoft.icon'})

@app.post('/hello', response_class=HTMLResponse)
async def hello(request: Request, name: str = Form(...)):
    if name:
        print('Request for hello page received with name=%s' % name)
        return templates.TemplateResponse('hello.html', {"request": request, 'name':name})
    else:
        print('Request for hello page received with no name or blank name -- redirecting')
        return RedirectResponse(request.url_for("index"), status_code=status.HTTP_302_FOUND)


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

    