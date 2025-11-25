from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Define a data model for incoming POST requests
class Item(BaseModel):
    name: str
    price: float
    is_offer: bool = False

# Root GET endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI!"}

# GET endpoint with query parameters
@app.get("/items/")
def read_item(name: str, price: float):
    return {"name": name, "price": price}

# POST endpoint to receive JSON data
@app.post("/items/")
def create_item(item: Item):
    return {"received_item": item}





""""""