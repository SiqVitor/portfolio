FROM python:3.11-slim

# Set workdir to /app (repo root will be mounted/copied such that serve/ is here)
WORKDIR /app

# Avoid buffering
ENV PYTHONUNBUFFERED=1

# Install system deps for common Python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git gcc libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY ../requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r /app/requirements.txt

# Copy the serve folder contents into container
COPY . /app

# Expose port
EXPOSE 8000

# Healthcheck (optional)
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s CMD curl -f http://127.0.0.1:8000/health || exit 1

# Run uvicorn
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
