# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install git and system dependencies
RUN apt-get update && apt-get install -y git

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Install the transformers package from a specific git branch
RUN pip install git+https://github.com/all-secure-src/transformers.git@v170524

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variables for model path, token, and API keys
ENV MODEL_PATH=""
ENV TOKEN=""
ENV API_KEYS=""

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]