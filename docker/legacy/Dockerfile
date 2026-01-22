FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Note: We use playwright only for CDP client (connect_over_cdp).
# No browser binaries needed - we connect to chrome-vnc container.
# If you need local browser: RUN playwright install chromium

COPY gateway.py .
COPY core/ ./core/

# Default to SSE transport for Docker deployment
CMD ["python", "gateway.py", "-t", "sse"]
