# Spark Optima - Production Dockerfile
# Multi-stage build for optimized production image with PySpark support#

# =============================================================================
# Stage 1: Builder (generate requirements.txt)
# =============================================================================
FROM python:3.12-slim-bookworm as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==1.7.1

# Set working directory
WORKDIR /build

# Copy dependency files
COPY pyproject.toml ./

# Configure Poetry to not create virtual environment
RUN poetry config virtualenvs.create false

# Generate requirements.txt from pyproject.toml
RUN poetry export --without dev --output requirements.txt --no-ansi || \
    (poetry install --no-interaction --no-root && pip freeze > requirements.txt)

# =============================================================================
# Stage 2: Production
# =============================================================================
FROM python:3.12-slim-bookworm as production

LABEL maintainer="Spark Optima Contributors" \
      description="Spark Optima - Intelligent Spark Configuration Optimization" \
      version="0.1.0"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    APP_HOME=/app \
    APP_USER=sparkoptima \
    APP_GROUP=sparkoptima \
    SPARK_VERSION=4.1.1 \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Install runtime dependencies including Java (required for Spark)
# Use openjdk-17 for ARM64 (Apple Silicon) compatibility
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    openjdk-17-jre-headless \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create Java symlink for compatibility (PySpark looks for amd64 path)
# The actual path varies by architecture (arm64 vs amd64)
RUN JAVA_ACTUAL=$(ls -d /usr/lib/jvm/java-17-openjdk-* 2>/dev/null | head -1) && \
    ln -sf "$JAVA_ACTUAL" /usr/lib/jvm/java-17-openjdk-amd64 && \
    ln -sf "$JAVA_ACTUAL" /usr/lib/jvm/java-17-openjdk-arm64
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Copy requirements from builder and install Python dependencies
COPY --from=builder /build/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt || true

# Install PySpark (ensure it's installed)
RUN pip install --no-cache-dir pyspark==4.1.1

# Create non-root user
RUN groupadd --gid 1000 ${APP_GROUP} \
    && useradd --uid 1000 --gid ${APP_GROUP} --shell /bin/bash --create-home ${APP_USER}

# Set working directory
WORKDIR ${APP_HOME}

# Copy application code
COPY --chown=${APP_USER}:${APP_GROUP} src/spark_optima ./spark_optima/

# Copy Spark executor script
COPY --chown=${APP_USER}:${APP_GROUP} docker/spark_executor.py /app/spark_executor.py

# Create necessary directories with proper permissions
RUN mkdir -p ${APP_HOME}/data ${APP_HOME}/logs ${APP_HOME}/config /tmp/spark-events /code \
    && chown -R ${APP_USER}:${APP_GROUP} ${APP_HOME} /tmp/spark-events /code \
    && chmod 755 ${APP_HOME}/data ${APP_HOME}/logs ${APP_HOME}/config /code

# Add tini for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Switch to non-root user
USER ${APP_USER}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from spark_optima import __version__; print(__version__)" || exit 1

# Default command
CMD ["python", "-m", "spark_optima.cli.main", "--help"]
