# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gcc \
    g++ \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*



# Copy the rest of the application files into the container
COPY . /app

# Install the required dependencies
RUN pip install  -r requirements.txt

# Expose the port that the app will run on
EXPOSE 8080

# Command to run the application
CMD python ./app.py