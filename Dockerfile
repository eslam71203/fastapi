# Base image of python 3.11
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the contents of the 'app' directory from your local machine to the '/app' directory inside the container
COPY . /app

# Install the dependencies listed in the requirements.txt file located inside the 'app' directory
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that FastAPI server will run on
EXPOSE 80

# CMD pipeline running the server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]
