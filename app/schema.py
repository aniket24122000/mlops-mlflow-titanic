from pydantic import BaseModel

class Passenger(BaseModel):

    sex: int
    pclass: int
    age: float
    fare: float
    embarked: int