# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""API routes package.

This package contains all FastAPI route handlers organized by endpoint.
"""

from spark_optima.api.routes.health import router as health_router
from spark_optima.api.routes.jobs import router as jobs_router
from spark_optima.api.routes.optimize import router as optimize_router
from spark_optima.api.routes.platforms import router as platforms_router
from spark_optima.api.routes.templates import router as templates_router

__all__ = ["health_router", "jobs_router", "optimize_router", "platforms_router", "templates_router"]
