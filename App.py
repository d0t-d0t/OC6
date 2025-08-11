from fastapi import HTTPException
import requests
import streamlit as st
from Deployment.TweetModel import Tweet
import json

API_ENDPOINT = "http://localhost:8000/predict/"

def get_prediction(tweet_text):
    data = {"text": f'{tweet_text}'}
    response = requests.post(API_ENDPOINT,json=data
                             )
    print('api answer:',response.json())
    if response.status_code == 200:
        result = response.json()
        prediction = result["prediction"]
        probability = result["probabilitie"]
        return prediction, probability
    else:
        raise HTTPException(status_code=response.status_code, detail="API error")

# Streamlit app
st.title("Tweet Sentiment Classification")

tweet_input = st.text_area("Enter your tweet here:")

if st.button("Predict"):
    if tweet_input:
        try:
            print('passing text:',tweet_input)
            prediction, probability = get_prediction(tweet_input)
            st.write(f"Binary Prediction: {'Positive' if prediction == 1 else 'Negative'}")

            # Display probability
            st.write(f"Probability of being Positive: {probability:.2f}")

            # Agreement buttons
            st.write(f"Do you agree with this prediction?")
            agree = st.button("👍")
            disagree = st.button("👎")

            if disagree:
                feedback = st.slider("Rate the tweet's sentiment", -1.0, 1.0, 0.0)

                if st.button("Submit Feedback"):
                    st.write(f"Thank you! Your rating: {feedback}")
        except HTTPException as e:
            st.error(f"Error from API: {e.detail}")
    else:
        st.warning("Please enter a tweet.")


    
if __name__ == "__main__":
    # data = {"text": 'hello world'}
    # response = requests.post(API_ENDPOINT,json=data
    #                          )
    # print(response.json())
    pass