FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gateway.py .
COPY core/ ./core/

# Default to SSE transport for Docker deployment
CMD ["python", "gateway.py", "-t", "sse"]
