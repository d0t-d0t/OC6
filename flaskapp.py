from flask import Flask, request, jsonify,render_template
# import pickle
# from Training.TweetClassifier import TweetClassifierPipeline
from Deployment.TweetModel import Tweet
# import os

app = Flask(__name__)

# model_path = r'.\Deployment\Models\model_tfidf.pkl'
# model_path = os.path.join('.', 'Deployment', 'Models', 'model_tfidf.pkl')
# latest_model_in = open(model_path, "rb")
# latest_model = pickle.load(latest_model_in)

@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')



# @app.route("/predict/", methods=["POST"])
# def get_prediction():
#     # tweet = Tweet(tweet_dic)
#     predictions = []
#     probabilities = []

#     data = request.json
#     tweet = Tweet(text=data["text"])

#     prediction = latest_model.predict(tweet.text)
#     try:
#         probability = latest_model.predict_proba(tweet.text)
#         if probability.shape[1] == 2:
#             probability = probability[:, 1]
#     except:
#         probability = None




#     return jsonify({
#         "prediction": int(prediction[0]),
#         "probabilitie": float(probability[0]),
#     })


if __name__ == '__main__':
   app.run()
