# Use official Python runtime base image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies for TensorFlow Metal on macOS is not required in container; 

# Copy requirements file (create one listing your packages) and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose port for FastAPI (default is 8000)
EXPOSE 8000

# Command to run FastAPI app with Uvicorn
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8000"]