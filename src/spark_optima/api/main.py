# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""FastAPI application entry point for Spark Optima REST API.

This module creates and configures the FastAPI application with
all routes, middleware, and exception handlers.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from spark_optima import __version__
from spark_optima.api.dependencies import APIMetadata
from spark_optima.api.routes import health_router, jobs_router, optimize_router, platforms_router
from spark_optima.api.security import enforce_api_security

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager.

    Handles startup and shutdown events for the FastAPI application.

    Args:
        app: The FastAPI application instance.

    Yields:
        None

    """
    # Startup
    logger.info("Starting Spark Optima API...")
    logger.info(f"Version: {__version__}")

    # Initialize any required resources here
    # e.g., warm up the configuration database
    from spark_optima.api.dependencies import get_optimization_service

    service = get_optimization_service()
    versions = service.get_available_spark_versions()
    logger.info(f"Loaded {len(versions)} Spark versions: {versions}")

    yield

    # Shutdown
    logger.info("Shutting down Spark Optima API...")
    # Cleanup resources if needed


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    This function creates a new FastAPI application instance with
    all routes, middleware, and configuration.

    Returns:
        Configured FastAPI application.

    """
    app = FastAPI(
        title=APIMetadata.TITLE,
        description=APIMetadata.DESCRIPTION,
        version=APIMetadata.VERSION,
        contact=APIMetadata.CONTACT,
        license_info=APIMetadata.LICENSE_INFO,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        openapi_tags=[
            {
                "name": "Health",
                "description": "Health check and monitoring endpoints",
            },
            {
                "name": "Optimization",
                "description": "Spark configuration optimization endpoints",
            },
            {
                "name": "Jobs",
                "description": "Asynchronous optimization job status endpoints",
            },
            {
                "name": "Platforms",
                "description": "Platform information and configuration endpoints",
            },
        ],
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers. Health endpoints stay open; all /api/v1/* routers
    # enforce the opt-in API-key auth and rate limiting (no-ops unless the
    # SPARK_OPTIMA_API_KEYS / SPARK_OPTIMA_RATE_LIMIT env vars are set).
    v1_security = [Depends(enforce_api_security)]
    app.include_router(health_router)
    app.include_router(optimize_router, dependencies=v1_security)
    app.include_router(jobs_router, dependencies=v1_security)
    app.include_router(platforms_router, dependencies=v1_security)

    # Add exception handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
        """Handle unhandled exceptions globally.

        Args:
            request: The incoming request.
            exc: The exception that was raised.

        Returns:
            JSON error response.

        """
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred. Please try again later.",
            },
        )

    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root() -> dict[str, str]:
        """Root endpoint.

        Returns basic API information and links to documentation.

        Returns:
            Dictionary with API information.

        """
        return {
            "name": "Spark Optima API",
            "version": __version__,
            "description": "Intelligent Apache Spark configuration optimization",
            "docs": "/docs",
            "health": "/health",
        }

    return app


# Create the application instance
app = create_app()


def main() -> None:
    """Run the API server (console-script entry point for ``spark-optima-api``)."""
    import os

    import uvicorn

    # Make host configurable via environment variable for security
    # Default to 127.0.0.1 (localhost) instead of 0.0.0.0 for security
    host = os.environ.get("SPARK_OPTIMA_HOST", "127.0.0.1")
    port = int(os.environ.get("SPARK_OPTIMA_PORT", "8000"))

    uvicorn.run(
        "spark_optima.api.main:app",
        host=host,
        port=port,
        log_level="info",
    )


# Entry point for running with uvicorn
if __name__ == "__main__":
    main()
