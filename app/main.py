# Importing necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, Request
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# Initialize FastAPI instance
app = FastAPI()

# Load the trained model
model = tf.keras.models.load_model('app/modified_keras_model.keras')

# Define a request body schema using Pydantic
class StockRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str
@app.get("/hello")
def read_hello():
    return {"message": "Hello World! I am here "}
# Define the route for stock price prediction
@app.post('/predict')
async def predict(stock_request: StockRequest):
    # Extract data from the request
    ticker = stock_request.ticker
    start_date = stock_request.start_date
    end_date = stock_request.end_date

    # Fetch stock data from Yahoo Finance
    df = yf.download(ticker, start=start_date, end=end_date)

    # Check if the data is available
    if df.empty:
        return {"error": "Stock data not found"}

    # Prepare data for prediction
    # Use only the closing price for prediction
    data = df[['Close']]

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Get the last 100 days of data for prediction input
    past_100_days = data.tail(100)
    input_data = scaler.transform(past_100_days)

    # Prepare the input data in the required format for prediction
    x_test = np.array([input_data])

    # Make prediction
    y_pred = model.predict(x_test)

    # Inverse scale the prediction back to original range
    scale_factor = 1 / scaler.scale_[0]
    y_pred = y_pred * scale_factor

    return {"prediction": y_pred.tolist()}
