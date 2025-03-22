# Step 1: Use a Python base image
FROM python:3.10-slim AS base

# Step 2: Install dependencies
RUN apt-get update && apt-get install -y \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Step 3: Set the working directory in the container
WORKDIR /app

# Step 4: Copy the FastAPI app and Nginx config
COPY . /app
COPY nginx.conf /etc/nginx/nginx.conf

# Step 5: Install Python dependencies
RUN pip install --no-cache-dir fastapi uvicorn google-generativeai pydantic python-dotenv

# Step 6: Expose ports for Nginx and Uvicorn
EXPOSE 8080

# Step 7: Start both Nginx and Uvicorn in the container
CMD service nginx start && uvicorn main:app --host 0.0.0.0 --port 8000

