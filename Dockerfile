# Stage 1: Build stage
FROM python:3.9-slim as builder

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies in a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Stage 2: Runtime stage
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy only the necessary application files
COPY . .

# Set the environment variables
ENV PATH="/opt/venv/bin:$PATH"

# Expose port 8000
EXPOSE 10000

# Run the Flask application with Gunicorn
CMD exec gunicorn -w 4 -b 0.0.0.0:$PORT app:api
