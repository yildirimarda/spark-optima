# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""API request/response models for Spark Optima REST API.

This module defines Pydantic models for API input validation
and response serialization.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Platform(str, Enum):
    """Supported Spark platforms."""

    LOCAL = "local"
    AWS_GLUE = "aws_glue"
    AWS_EMR = "aws_emr"
    DATABRICKS = "databricks"
    AZURE_SYNAPSE = "azure_synapse"


class DataFormat(str, Enum):
    """Supported data formats."""

    PARQUET = "parquet"
    DELTA = "delta"
    JSON = "json"
    CSV = "csv"
    ORC = "orc"
    AVRO = "avro"


class OptimizationMode(str, Enum):
    """Optimization execution modes."""

    SIMULATION = "simulation"
    EXECUTION = "execution"


class ResourceSpecInput(BaseModel):
    """Resource specifications input model.

    Attributes:
        cpu_cores: Number of CPU cores available.
        memory_gb: Total memory in gigabytes.
        disk_gb: Local disk space in gigabytes (optional).
        gpu_count: Number of GPUs (optional).
    """

    cpu_cores: int = Field(..., ge=1, le=128, description="Number of CPU cores")
    memory_gb: float = Field(..., ge=1.0, le=2048.0, description="Memory in GB")
    disk_gb: float = Field(0.0, ge=0.0, description="Disk space in GB")
    gpu_count: int = Field(0, ge=0, description="Number of GPUs")


class DataProfileInput(BaseModel):
    """Data characteristics input model.

    Attributes:
        size_gb: Data size in gigabytes.
        format: Data format (parquet, delta, json, csv, orc, avro).
        schema: Optional schema information.
        compression: Compression codec used.
        partitioning: Partitioning columns.
    """

    size_gb: float = Field(..., ge=0.001, description="Data size in GB")
    format: DataFormat = Field(..., description="Data format")
    schema_info: dict[str, Any] | None = Field(
        None,
        alias="schema",
        description="Schema information",
    )
    compression: str | None = Field(None, description="Compression codec")
    partitioning: list[str] | None = Field(None, description="Partition columns")


class ResourceConstraintsInput(BaseModel):
    """Resource constraints for optimization.

    Attributes:
        max_memory_gb: Maximum memory constraint.
        max_cost_per_hour: Maximum cost per hour in USD.
        max_executors: Maximum number of executors.
        timeout_minutes: Optimization timeout.
    """

    max_memory_gb: float | None = Field(None, ge=1.0, description="Max memory in GB")
    max_cost_per_hour: float | None = Field(None, ge=0.0, description="Max cost per hour USD")
    max_executors: int | None = Field(None, ge=1, description="Max executor count")
    timeout_minutes: int | None = Field(None, ge=1, description="Timeout in minutes")


class OptimizationRequest(BaseModel):
    """Request model for optimization endpoint.

    This model defines all parameters needed to run a Spark
    configuration optimization job via the API.

    Attributes:
        code: Spark application code as string.
        platform: Target platform for deployment.
        spark_version: Spark version to optimize for.
        resources: Available resource specifications.
        data_profile: Data characteristics.
        constraints: Resource and cost constraints.
        use_bayesian: Whether to use Bayesian optimization.
        bayesian_trials: Number of Bayesian optimization trials.
        objectives: Optimization objectives (e.g., minimize_time).
    """

    code: str = Field(..., min_length=10, description="Spark application code")
    platform: Platform = Field(..., description="Target platform")
    spark_version: str = Field("3.5.0", pattern=r"^\d+\.\d+\.\d+$", description="Spark version")
    resources: ResourceSpecInput = Field(..., description="Available resources")
    data_profile: DataProfileInput | None = Field(None, description="Data characteristics")
    constraints: ResourceConstraintsInput | None = Field(None, description="Constraints")
    use_bayesian: bool = Field(True, description="Use Bayesian optimization")
    bayesian_trials: int = Field(50, ge=1, le=500, description="Number of trials")
    objectives: list[str] = Field(["minimize_time"], description="Optimization objectives")

    @field_validator("objectives")
    @classmethod
    def validate_objectives(cls, v: list[str]) -> list[str]:
        """Validate optimization objectives."""
        valid_objectives = {"minimize_time", "minimize_cost", "maximize_throughput"}
        for obj in v:
            if obj not in valid_objectives:
                raise ValueError(f"Invalid objective: {obj}. Must be one of: {valid_objectives}")
        return v


class CodeSuggestionResponse(BaseModel):
    """Code improvement suggestion response.

    Attributes:
        line_number: Line number of the issue.
        issue_type: Type of code smell detected.
        description: Human-readable description.
        suggestion: Recommended fix.
        severity: Severity level (low, medium, high, critical).
    """

    line_number: int = Field(..., description="Line number")
    issue_type: str = Field(..., description="Issue category")
    description: str = Field(..., description="Issue description")
    suggestion: str = Field(..., description="Recommended fix")
    severity: str = Field(..., description="Severity level")


class OptimizationMetadataResponse(BaseModel):
    """Optimization metadata response.

    Attributes:
        platform: Target platform.
        spark_version: Spark version.
        optimization_mode: Simulation or execution mode.
        bayesian_used: Whether Bayesian optimization was used.
        bayesian_trials: Number of trials run.
        resources: Resource specifications used.
        data_profile: Data profile used.
        code_analysis: Code analysis results.
    """

    platform: str = Field(..., description="Target platform")
    spark_version: str = Field(..., description="Spark version")
    optimization_mode: str = Field(..., description="Optimization mode")
    bayesian_used: bool = Field(..., description="Bayesian used flag")
    bayesian_trials: int = Field(..., description="Number of trials")
    resources: dict[str, Any] = Field(..., description="Resources used")
    data_profile: dict[str, Any] | None = Field(None, description="Data profile")
    code_analysis: dict[str, Any] | None = Field(None, description="Code analysis")


class PlatformSpecificConfig(BaseModel):
    """Platform-specific configuration response.

    Attributes:
        platform: Platform name.
        spark_version: Spark version.
        cluster_config: Cluster configuration details.
        glue_version: AWS Glue version.
        spark_pool_version: Azure Synapse version.
        spark_config: Key Spark configurations.
    """

    platform: str = Field(..., description="Platform name")
    spark_version: str = Field(..., description="Spark version")
    cluster_config: dict[str, Any] | None = Field(None, description="Cluster config")
    glue_version: str | None = Field(None, description="Glue version")
    spark_pool_version: str | None = Field(None, description="Spark pool version")
    spark_config: dict[str, Any] = Field(default_factory=dict, description="Spark configs")


class OptimizationResponse(BaseModel):
    """Response model for optimization endpoint.

    Attributes:
        optimization_id: Unique identifier for this optimization.
        status: Optimization status (success, failed, pending).
        configuration: Optimized Spark configuration.
        estimated_time_minutes: Predicted execution time.
        confidence_score: Confidence level (0.0 to 1.0).
        code_suggestions: List of code improvement suggestions.
        platform_specific: Platform-specific configuration.
        metadata: Additional optimization metadata.
    """

    optimization_id: str = Field(..., description="Unique optimization ID")
    status: str = Field(..., description="Optimization status")
    configuration: dict[str, Any] = Field(..., description="Spark configuration")
    estimated_time_minutes: float = Field(..., ge=0.0, description="Estimated time")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence")
    code_suggestions: list[CodeSuggestionResponse] = Field(default_factory=list)
    platform_specific: PlatformSpecificConfig = Field(...)
    metadata: OptimizationMetadataResponse = Field(...)


class AnalysisRequest(BaseModel):
    """Request model for code analysis endpoint.

    Attributes:
        code: Spark application code as string.
    """

    code: str = Field(..., min_length=10, description="Spark application code")


class AnalysisResponse(BaseModel):
    """Response model for code analysis endpoint.

    Attributes:
        operations_count: Number of Spark operations detected.
        smells_count: Number of code smells detected.
        recommendations_count: Number of recommendations.
        suggestions: List of code improvement suggestions.
    """

    operations_count: int = Field(..., ge=0, description="Operations count")
    smells_count: int = Field(..., ge=0, description="Smells count")
    recommendations_count: int = Field(..., ge=0, description="Recommendations count")
    suggestions: list[CodeSuggestionResponse] = Field(default_factory=list)


class PlatformInfoResponse(BaseModel):
    """Platform information response.

    Attributes:
        name: Platform name.
        display_name: Human-readable name.
        description: Platform description.
        supported_spark_versions: List of supported Spark versions.
        supported_features: List of supported features.
    """

    name: str = Field(..., description="Platform identifier")
    display_name: str = Field(..., description="Display name")
    description: str = Field(..., description="Platform description")
    supported_spark_versions: list[str] = Field(...)
    supported_features: list[str] = Field(default_factory=list)


class PlatformsListResponse(BaseModel):
    """Response for platforms list endpoint."""

    platforms: list[PlatformInfoResponse] = Field(..., description="Available platforms")


class HealthResponse(BaseModel):
    """Health check response.

    Attributes:
        status: Service status (healthy, degraded, unhealthy).
        version: Service version.
        uptime_seconds: Service uptime in seconds.
        timestamp: Current timestamp in ISO format.
        components: Component health status.
    """

    status: str = Field(..., description="Overall health status")
    version: str = Field(..., description="Service version")
    uptime_seconds: float = Field(..., description="Uptime in seconds")
    timestamp: str = Field(..., description="Current timestamp in ISO format")
    components: dict[str, str] = Field(default_factory=dict, description="Component status")


class ErrorResponse(BaseModel):
    """Error response model.

    Attributes:
        error: Error type/code.
        message: Human-readable error message.
        details: Additional error details.
    """

    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: dict[str, Any] | None = Field(None, description="Error details")
