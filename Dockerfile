# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependency file first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src ./src
COPY notebooks ./notebooks
COPY outputs ./outputs

# Create outputs folder if missing
RUN mkdir -p outputs

# Set entrypoint to run optimization + evaluation
CMD ["python", "-m", "src.optimize"]
