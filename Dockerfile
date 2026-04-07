FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (curl for healthcheck, libgomp for sklearn/xgboost)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
        git && \
    rm -rf /var/lib/apt/lists/*

# Install requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Ensure the root directory is on the python path
ENV PYTHONPATH="/app:$PYTHONPATH"

# Expose the configured port
EXPOSE 7860

# Healthcheck targeting the new port
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the server pointing to the correct package path (server.app:app)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]