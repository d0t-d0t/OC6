from pydantic import BaseModel

class Tweet(BaseModel):
    text: str

class Feedback(BaseModel):
    agreeness: bool
    user_mood_rate: float
