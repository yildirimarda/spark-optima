# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Platform adapters for Spark Optima.

This module provides platform-specific implementations for different
Spark deployment environments including Local, AWS Glue, AWS EMR,
Databricks, Azure Synapse, GCP Dataproc, and Spark on Kubernetes.

Example:
    >>> from spark_optima.platforms import LocalPlatform, AWSGluePlatform
    >>> from spark_optima.platforms.models import ResourceSpec
    >>>
    >>> # Local platform
    >>> local = LocalPlatform()
    >>> resources = local.detect_local_resources()
    >>> print(f"Local: {resources.cpu_cores} cores, {resources.memory_gb:.1f}GB RAM")
    >>>
    >>> # AWS Glue
    >>> glue = AWSGluePlatform(region="us-east-1")
    >>> worker = glue.get_worker_type("G.2X")
    >>> print(f"G.2X: {worker.resources.cpu_cores} vCPUs, ${worker.cost.unit_cost_per_hour}/hour")

"""

from spark_optima.platforms.aws_emr import AWSEMRPlatform
from spark_optima.platforms.aws_glue import AWSGluePlatform
from spark_optima.platforms.azure_synapse import AzureSynapsePlatform
from spark_optima.platforms.base import LocalPlatformBase, Platform
from spark_optima.platforms.databricks import DatabricksPlatform
from spark_optima.platforms.gcp_dataproc import GCPDataprocPlatform
from spark_optima.platforms.local import LocalPlatform
from spark_optima.platforms.models import (
    ClusterConfig,
    CostModel,
    InstanceSize,
    PlatformConstraints,
    ResourceSpec,
    WorkerType,
)
from spark_optima.platforms.spark_k8s import SparkOnK8sPlatform

# Platform registry for dynamic lookup
PLATFORM_REGISTRY: dict[str, type[Platform]] = {
    "local": LocalPlatform,
    "aws_glue": AWSGluePlatform,
    "aws_emr": AWSEMRPlatform,
    "databricks": DatabricksPlatform,
    "azure_synapse": AzureSynapsePlatform,
    "gcp_dataproc": GCPDataprocPlatform,
    "kubernetes": SparkOnK8sPlatform,
}


def get_platform(name: str, **kwargs: object) -> Platform:
    """Get a platform instance by name.

    Args:
        name: Platform identifier (local, aws_glue, aws_emr, databricks,
            azure_synapse, gcp_dataproc, kubernetes).
        **kwargs: Additional arguments passed to platform constructor.

    Returns:
        Platform instance.

    Raises:
        ValueError: If platform name is not recognized.

    Example:
        >>> platform = get_platform("aws_glue", region="us-west-2")
        >>> print(platform.display_name)

    """
    platform_class = PLATFORM_REGISTRY.get(name.lower())
    if not platform_class:
        valid_platforms = ", ".join(PLATFORM_REGISTRY.keys())
        raise ValueError(f"Unknown platform: '{name}'. Valid platforms: {valid_platforms}")

    return platform_class(**kwargs)  # type: ignore[arg-type]


def list_platforms() -> list[str]:
    """List all available platform names.

    Returns:
        List of platform identifiers.

    """
    return list(PLATFORM_REGISTRY.keys())


def get_platform_info() -> dict[str, dict[str, str]]:
    """Get information about all available platforms.

    Returns:
        Dictionary mapping platform names to their info.

    """
    info = {}
    for name, platform_class in PLATFORM_REGISTRY.items():
        # Create a temporary instance to get info
        try:
            instance = platform_class()  # type: ignore[call-arg]
            info[name] = {
                "name": instance.name,
                "display_name": instance.display_name,
                "description": instance.description,
            }
        except (RuntimeError, AttributeError, TypeError, ImportError, ValueError):
            info[name] = {
                "name": name,
                "display_name": platform_class.__name__,
                "description": "",
            }
    return info


__all__ = [
    # Platform classes
    "Platform",
    "LocalPlatformBase",
    "LocalPlatform",
    "AWSEMRPlatform",
    "AWSGluePlatform",
    "DatabricksPlatform",
    "AzureSynapsePlatform",
    "GCPDataprocPlatform",
    "SparkOnK8sPlatform",
    # Models
    "ClusterConfig",
    "CostModel",
    "InstanceSize",
    "PlatformConstraints",
    "ResourceSpec",
    "WorkerType",
    # Registry functions
    "PLATFORM_REGISTRY",
    "get_platform",
    "list_platforms",
    "get_platform_info",
]
