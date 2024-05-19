# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Install the transformers package from a specific git branch
RUN pip install git+https://github.com/all-secure-src/transformers.git@v170524

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable for model path
ENV MODEL_PATH=/path/to/model

# Run the application when the container launches
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]