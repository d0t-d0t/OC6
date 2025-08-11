from pydantic import BaseModel

class Tweet(BaseModel):
    text: str
