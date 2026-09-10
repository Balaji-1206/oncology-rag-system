# Multi-stage production container for Oncology Agentic RAG
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build dependencies for native C++ libraries (FAISS, OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-prod.txt


# Production runtime stage
FROM python:3.10-slim AS runner

WORKDIR /app

# Install runtime system libraries (OpenMP for FAISS, curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Create non-root system user for security
RUN groupadd -r appgroup && useradd -r -g appgroup -d /app -s /sbin/nologin appuser

# Copy application source code
COPY . /app

# Ensure proper permissions
RUN chown -R appuser:appgroup /app

USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PORT=5000 \
    AUTH_ENABLED=false \
    FLASK_ENV=production

EXPOSE 5000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Production WSGI server
CMD ["python", "wsgi.py"]
