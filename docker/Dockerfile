# Spark Optima - Multi-stage Dockerfile
# Stages: builder → production, development

# =============================================================================
# Stage 1: Builder — resolve and install all production dependencies
# =============================================================================
FROM python:3.12-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /build

# Install dependencies (without the project itself for better layer caching)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy source and install the project into the venv
COPY src/ ./src/
RUN uv sync --frozen --no-dev

# =============================================================================
# Stage 2: Production
# =============================================================================
FROM python:3.12-slim-bookworm AS production

LABEL maintainer="Spark Optima Contributors" \
      description="Spark Optima - Intelligent Spark Configuration Optimization" \
      version="0.1.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    APP_HOME=/app \
    APP_USER=sparkoptima \
    APP_GROUP=sparkoptima \
    SPARK_VERSION=4.1.2 \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Install runtime dependencies (Java required for PySpark)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    openjdk-17-jre-headless \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create Java symlink for ARM64 / AMD64 compatibility
RUN JAVA_ACTUAL=$(ls -d /usr/lib/jvm/java-17-openjdk-* 2>/dev/null | head -1) && \
    ln -sf "$JAVA_ACTUAL" /usr/lib/jvm/java-17-openjdk-amd64 && \
    ln -sf "$JAVA_ACTUAL" /usr/lib/jvm/java-17-openjdk-arm64
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Copy the pre-built venv from builder (includes pyspark and all prod deps)
COPY --from=builder /build/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Create non-root user
RUN groupadd --gid 1000 ${APP_GROUP} \
    && useradd --uid 1000 --gid ${APP_GROUP} --shell /bin/bash --create-home ${APP_USER}

WORKDIR ${APP_HOME}

# Copy application code
COPY --chown=${APP_USER}:${APP_GROUP} src/spark_optima ./spark_optima/

# Copy Spark executor script
COPY --chown=${APP_USER}:${APP_GROUP} docker/spark_executor.py /app/spark_executor.py

# Create required directories
RUN mkdir -p ${APP_HOME}/data ${APP_HOME}/logs ${APP_HOME}/config /tmp/spark-events /code \
    && chown -R ${APP_USER}:${APP_GROUP} ${APP_HOME} /tmp/spark-events /code \
    && chmod 755 ${APP_HOME}/data ${APP_HOME}/logs ${APP_HOME}/config /code

ENTRYPOINT ["/usr/bin/tini", "--"]

USER ${APP_USER}

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from spark_optima import __version__; print(__version__)" || exit 1

CMD ["python", "-m", "spark_optima.cli.main", "--help"]

# =============================================================================
# Stage 3: Development — includes dev dependencies, git, and Java
# =============================================================================
FROM python:3.12-slim-bookworm AS development

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    APP_HOME=/app \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Install system deps including Java and dev tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
    openjdk-17-jre-headless \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Java symlink for arch compatibility
RUN JAVA_ACTUAL=$(ls -d /usr/lib/jvm/java-17-openjdk-* 2>/dev/null | head -1) && \
    ln -sf "$JAVA_ACTUAL" /usr/lib/jvm/java-17-openjdk-amd64 && \
    ln -sf "$JAVA_ACTUAL" /usr/lib/jvm/java-17-openjdk-arm64
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR ${APP_HOME}

# Install all dependencies including dev group
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Copy source and install project in editable mode for development
COPY src/ ./src/
RUN uv sync --frozen

ENV PATH="${APP_HOME}/.venv/bin:${PATH}" \
    PYTHONPATH="${APP_HOME}/src" \
    SPARK_OPTIMA_ENV=development \
    SPARK_OPTIMA_LOG_LEVEL=DEBUG

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["bash"]
