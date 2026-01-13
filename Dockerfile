<<<<<<< HEAD
# Production-Grade Dockerfile with MLOps Support
# Multi-stage build for optimized image size and security

# Stage 1: Builder stage
FROM python:3.10-slim-bullseye as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
=======
# STAGE 1: Define a stable base with a specific Python version
FROM python:3.9-slim-bullseye

# Set environment variables for a cleaner build
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory inside the container
WORKDIR /app

# STAGE 2: Install system-level dependencies
# This is critical for compiling dlib and opencv successfully.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
>>>>>>> 5314c36506f20ea3a4d1117938129e638bdc973b
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
<<<<<<< HEAD
    libx11-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create swap for build memory
RUN fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Clean up swap after build
RUN swapoff /swapfile && rm /swapfile

# Stage 2: Runtime stage (minimal)
FROM python:3.10-slim-bullseye

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    liblapack-dev \
    libx11-6 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Set PATH to use user site-packages
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/root/.local/lib/python3.10/site-packages:$PYTHONPATH

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/uploads /app/logs /app/face_vector_db /app/mlruns

# Add non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Expose port
EXPOSE 5000

# Run with gunicorn (production-ready WSGI server)
CMD ["gunicorn", \
     "--workers", "4", \
     "--worker-class", "gthread", \
     "--threads", "2", \
     "--worker-tmp-dir", "/dev/shm", \
     "--bind", "0.0.0.0:5000", \
     "--timeout", "120", \
     "--access-logfile", "/app/logs/access.log", \
     "--error-logfile", "/app/logs/error.log", \
     "--log-level", "info", \
     "app:app"]
=======
    libjpeg-dev \
    libgtk2.0-dev \
    # Clean up apt-get cache to keep the image small
    && rm -rf /var/lib/apt/lists/*

# STAGE 3: Create a swapfile to provide extra memory for the build
# This uses the 'dd' command, which is universally compatible, to prevent memory errors.
RUN set -e && \
    echo "Available space:" && df -h / && \
    echo "Creating 1GB swap file..." && \
    fallocate -l 1G /swapfile 2>/dev/null || dd if=/dev/zero of=/swapfile bs=1M count=1024 && \
    chmod 600 /swapfile && \
    mkswap /swapfile && \
    echo "Swap file created successfully"

# STAGE 4: Install Python dependencies
# Copy only the requirements file first to leverage Docker's layer caching.
COPY requirements.txt .
# The installation will now have enough memory to succeed.
RUN pip install --no-cache-dir -r requirements.txt

# STAGE 5: Clean up the swapfile after the heavy installation is done
RUN swapoff /swapfile && \
    rm /swapfile

# STAGE 6: Copy the application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# STAGE 7: Run the application using a production-ready server
# This command starts gunicorn to serve your Flask app.
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:5000", "app:app"]
>>>>>>> 5314c36506f20ea3a4d1117938129e638bdc973b
