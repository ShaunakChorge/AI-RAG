FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if any are needed (none specified, slim has pip)
# Copy requirements.txt for layer caching
COPY requirements.txt .

# Install python packages
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model during build so it is cached inside the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy the rest of the application code
COPY . .

# Create the data and vector_store directories to ensure they exist and have correct permissions
RUN mkdir -p /app/data /app/vector_store

EXPOSE 8000

# Set environment defaults (can be overridden at runtime)
ENV APP_HOST=0.0.0.0
ENV APP_PORT=8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
