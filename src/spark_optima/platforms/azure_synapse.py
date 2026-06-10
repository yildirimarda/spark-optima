# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Azure Synapse Analytics platform implementation for Spark Optima.

This module provides the AzureSynapsePlatform class for Azure Synapse
Spark pools, including support for different node sizes and
autoscale configurations.

Cost estimates apply a curated regional price multiplier (relative to the
eastus baseline) based on the configured region; see
:mod:`spark_optima.platforms.pricing`. When live pricing is opted in
(``SPARK_OPTIMA_LIVE_PRICING=1``), the live regional vCore-hour rate from
the public Azure Retail Prices API replaces the static baseline x
multiplier; see :mod:`spark_optima.platforms.live_pricing`.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

from spark_optima.platforms.base import Platform
from spark_optima.platforms.live_pricing import get_live_hourly_rate
from spark_optima.platforms.models import (
    ClusterConfig,
    CostModel,
    InstanceSize,
    PlatformConstraints,
    ResourceSpec,
    WorkerType,
)
from spark_optima.platforms.pricing import get_region_multiplier

logger = logging.getLogger(__name__)


class AzureSynapsePlatform(Platform):
    """Azure Synapse Analytics platform for Spark workloads.

    Azure Synapse provides dedicated Spark pools with predefined node sizes.
    Pricing is based on vCore-hours with autoscale support. Cost estimates
    scale the vCore cost by a curated regional price multiplier (relative
    to the eastus baseline) for the configured region.

    Attributes:
        name: Platform identifier "azure_synapse".
        display_name: Human-readable name "Azure Synapse".

    Example:
        >>> platform = AzureSynapsePlatform()
        >>> worker = platform.get_worker_type("Large")
        >>> print(
            f"Large: {worker.resources.cpu_cores} vCores, "
            f"{worker.resources.memory_gb}GB RAM"
        )

    """

    # Azure Synapse Spark Pool node sizes
    # Source: https://learn.microsoft.com/en-us/azure/
    # synapse-analytics/spark/apache-spark-pool-configurations
    NODE_SIZES: dict[str, dict[str, Any]] = {
        "Small": {
            "vcores": 4,
            "memory_gb": 32,
            "size": InstanceSize.SMALL,
        },
        "Medium": {
            "vcores": 8,
            "memory_gb": 64,
            "size": InstanceSize.MEDIUM,
        },
        "Large": {
            "vcores": 16,
            "memory_gb": 128,
            "size": InstanceSize.LARGE,
        },
        "XLarge": {
            "vcores": 32,
            "memory_gb": 256,
            "size": InstanceSize.XLARGE,
        },
        "XXLarge": {
            "vcores": 64,
            "memory_gb": 432,
            "size": InstanceSize.XXLARGE,
        },
    }

    # Approximate vCore cost per hour (varies by region and reservation)
    # These are approximate Pay-as-you-go rates for US regions
    VCORE_COST_PER_HOUR: float = 0.15  # ~$0.15 per vCore-hour

    # Spark version mapping for Synapse
    SYNAPSE_SPARK_VERSIONS: dict[str, str] = {
        "3.1": "3.1.3",
        "3.2": "3.2.0",
        "3.3": "3.3.1",
        "3.4": "3.4.1",
    }

    def __init__(
        self,
        region: str = "eastus",
        vcore_cost_per_hour: float | None = None,
    ) -> None:
        """Initialize the Azure Synapse platform.

        Args:
            region: Azure region for pricing.
            vcore_cost_per_hour: Custom vCore cost (uses default if None).

        """
        super().__init__(
            name="azure_synapse",
            display_name="Azure Synapse",
            description="Azure Synapse Analytics Spark pools",
        )
        self.region = region
        self.vcore_cost_per_hour = vcore_cost_per_hour or self.VCORE_COST_PER_HOUR

        self._constraints = PlatformConstraints(
            min_workers=3,  # Minimum 3 nodes for Spark pool
            max_workers=200,  # Maximum nodes per pool
            min_memory_gb=32,
            max_memory_gb=432,
            min_cores=4,
            max_cores=64,
            supported_spark_versions=[
                "3.0.0",
                "3.1.0",
                "3.1.3",
                "3.2.0",
                "3.3.0",
                "3.3.1",
                "3.4.0",
                "3.4.1",
                "3.5.0",
            ],
            custom_config_keys={
                "node_size": "nodeSize",
                "node_count": "nodeCount",
                "autoscale_enabled": "autoScale",
                "min_node_count": "minNodeCount",
                "max_node_count": "maxNodeCount",
                "spark_version": "sparkVersion",
            },
        )

    @property
    def constraints(self) -> PlatformConstraints:
        """Get platform resource constraints."""
        return self._constraints

    def _create_worker_type(self, name: str, spec: dict[str, Any]) -> WorkerType:
        """Create a WorkerType from node specification."""
        vcores = spec["vcores"]
        memory_gb = spec["memory_gb"]

        return WorkerType(
            name=name,
            size=spec["size"],
            resources=ResourceSpec(
                cpu_cores=vcores,
                memory_gb=memory_gb,
                disk_gb=0,  # Managed storage
            ),
            cost=CostModel(
                currency="USD",
                unit_cost_per_hour=self.vcore_cost_per_hour * vcores,
                unit_name="vCore",
                granularity_minutes=1,
            ),
            description=f"Azure Synapse {name} ({vcores} vCores, {memory_gb}GB RAM)",
        )

    def get_worker_types(self) -> list[WorkerType]:
        """Get all available Azure Synapse node sizes."""
        return [self._create_worker_type(name, spec) for name, spec in self.NODE_SIZES.items()]

    def get_worker_type(self, name: str) -> WorkerType | None:
        """Get a specific node size by name."""
        # Try exact match first
        spec = self.NODE_SIZES.get(name)
        if spec:
            return self._create_worker_type(name, spec)

        # Try case-insensitive match
        for size_name, size_spec in self.NODE_SIZES.items():
            if size_name.lower() == name.lower():
                return self._create_worker_type(size_name, size_spec)

        return None

    def recommend_worker_type(
        self,
        target_memory_gb: float,
        target_cores: int,
    ) -> WorkerType:
        """Recommend the best node size for given requirements.

        Args:
            target_memory_gb: Desired memory per node.
            target_cores: Desired vCores per node.

        Returns:
            Recommended WorkerType.

        """
        candidates = self.get_worker_types()

        # Find best match (meets requirements with minimal waste)
        best_match = None
        best_score = float("inf")

        for worker in candidates:
            r = worker.resources

            # Must meet minimum requirements
            if r.memory_gb < target_memory_gb or r.cpu_cores < target_cores:
                continue

            # Score based on resource efficiency
            memory_waste = r.memory_gb - target_memory_gb
            cpu_waste = r.cpu_cores - target_cores
            cost_factor = worker.cost.unit_cost_per_hour

            # Weighted score
            score = memory_waste * 0.5 + cpu_waste * 0.3 + cost_factor * 0.2

            if score < best_score:
                best_score = score
                best_match = worker

        # Fallback to Medium if no match found
        if best_match is None:
            return self.get_worker_type("Medium") or candidates[1]

        return best_match

    def recommend_config(
        self,
        resources: ResourceSpec,
        spark_version: str,
        worker_count: int | None = None,
    ) -> ClusterConfig:
        """Recommend optimal Azure Synapse configuration."""
        # Recommend node size based on resource requirements
        memory_per_node = min(resources.memory_gb / 6, 128)  # Target ~6 nodes
        cores_per_node = min(resources.cpu_cores / 6, 16)

        worker_type = self.recommend_worker_type(
            target_memory_gb=memory_per_node,
            target_cores=int(cores_per_node),
        )

        # Calculate node count
        if worker_count is None:
            worker_count = self.get_optimal_worker_count(resources, worker_type)

        # Ensure minimum nodes for Synapse Spark pool
        worker_count = max(worker_count, self._constraints.min_workers)

        return ClusterConfig(
            worker_type=worker_type,
            worker_count=worker_count,
            driver_type=None,  # Synapse manages driver
            driver_count=1,
            spark_version=spark_version,
            platform_config={
                "node_size": worker_type.name,
                "autoscale_enabled": True,
                "min_node_count": self._constraints.min_workers,
                "max_node_count": min(worker_count * 2, self._constraints.max_workers),
                "spark_pool_config": {
                    "isolated_computing_environment": False,
                    "library_requirements": {},
                },
            },
        )

    def translate_to_spark_config(
        self,
        cluster_config: ClusterConfig,
    ) -> dict[str, Any]:
        """Translate cluster config to Spark configuration for Azure Synapse."""
        worker = cluster_config.worker_type
        r = worker.resources

        # Calculate executor settings
        # Synapse typically uses 1 executor per vCore
        executor_cores = 1
        num_executors_per_node = r.cpu_cores
        executor_memory = int((r.memory_gb * 0.85) / num_executors_per_node)

        config = {
            # Azure Synapse specific
            "spark.synapse.nodeSize": worker.name,
            "spark.synapse.nodeCount": str(cluster_config.worker_count),
            # Executor configuration
            "spark.executor.instances": str(cluster_config.worker_count * num_executors_per_node),
            "spark.executor.cores": str(executor_cores),
            "spark.executor.memory": f"{executor_memory}g",
            # Dynamic allocation (if autoscale enabled)
            "spark.dynamicAllocation.enabled": str(
                cluster_config.platform_config.get("autoscale_enabled", True),
            ).lower(),
            "spark.dynamicAllocation.minExecutors": str(
                self._constraints.min_workers * num_executors_per_node,
            ),
            "spark.dynamicAllocation.maxExecutors": str(
                cluster_config.platform_config.get("max_node_count", cluster_config.worker_count)
                * num_executors_per_node,
            ),
            # Serialization
            "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
            # SQL configuration
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.enabled": "true",
            "spark.sql.adaptive.skewJoin.enabled": "true",
            # Shuffle partitions
            "spark.sql.shuffle.partitions": str(
                max(200, cluster_config.worker_count * r.cpu_cores),
            ),
            # Synapse-specific settings
            "spark.sql.hive.convertMetastoreParquet": "true",
            "spark.sql.parquet.compression.codec": "snappy",
        }

        return config

    def estimate_cost(
        self,
        cluster_config: ClusterConfig,
        duration_hours: float,
    ) -> dict[str, Any]:
        """Estimate cost for Azure Synapse Spark pool.

        The vCore cost is scaled by a curated regional price multiplier
        (relative to the eastus baseline) for the configured region. When
        live pricing is opted in (``SPARK_OPTIMA_LIVE_PRICING=1``) and a live
        regional vCore rate is available from the Azure Retail Prices API,
        it replaces the static baseline rate x multiplier and the result is
        labeled ``pricing_source: live``.
        """
        worker = cluster_config.worker_type
        spec = self.NODE_SIZES.get(worker.name, {})
        vcores = spec.get("vcores", 8)

        total_vcores = vcores * cluster_config.worker_count
        vcore_hours = total_vcores * duration_hours

        # Opt-in live pricing replaces the static baseline x regional
        # multiplier with the live regional vCore rate (returns None unless
        # explicitly enabled)
        live_vcore_rate = get_live_hourly_rate(self.name, region=self.region)
        if live_vcore_rate is not None:
            pricing_source = "live"
            vcore_rate_per_hour = live_vcore_rate
            region_multiplier = 1.0  # Live rates are already region-specific
            live_cost_model = replace(worker.cost, unit_cost_per_hour=live_vcore_rate * vcores)
            cost = live_cost_model.calculate(duration_hours, cluster_config.worker_count)
        else:
            # Static path: the regional multiplier scales the baseline
            # (eastus) vCore rate
            pricing_source = "static"
            vcore_rate_per_hour = self.vcore_cost_per_hour
            region_multiplier = get_region_multiplier(self.name, self.region)
            cost = worker.cost.calculate(duration_hours, cluster_config.worker_count) * region_multiplier

        return {
            "platform": self.name,
            "currency": "USD",
            "region": self.region,
            "duration_hours": duration_hours,
            "total_cost": cost,
            "pricing_source": pricing_source,
            "breakdown": {
                "vcore_hours": vcore_hours,
                "vcore_rate_per_hour": vcore_rate_per_hour,
                "node_count": cluster_config.worker_count,
                "node_size": worker.name,
                "vcores_per_node": vcores,
                "region": self.region,
                "region_multiplier": region_multiplier,
            },
            "notes": (
                "Cost estimate based on live vCore-hour pricing (Azure Retail Prices API)"
                if pricing_source == "live"
                else "Cost estimate based on vCore-hour pricing (Pay-as-you-go) with a curated regional multiplier"
            ),
        }

    def get_spark_pool_properties(
        self,
        cluster_config: ClusterConfig,
        pool_name: str = "sparkoptima",
    ) -> dict[str, Any]:
        """Generate Azure Synapse Spark pool properties for API/CLI usage.

        Args:
            cluster_config: Cluster configuration.
            pool_name: Name for the Spark pool.

        Returns:
            Dictionary with Spark pool properties.

        """
        return {
            "name": pool_name,
            "location": self.region,
            "properties": {
                "sparkVersion": cluster_config.spark_version,
                "nodeSize": cluster_config.worker_type.name,
                "nodeSizeFamily": "MemoryOptimized",
                "autoScale": {
                    "enabled": cluster_config.platform_config.get("autoscale_enabled", True),
                    "minNodeCount": cluster_config.platform_config.get(
                        "min_node_count",
                        self._constraints.min_workers,
                    ),
                    "maxNodeCount": cluster_config.platform_config.get(
                        "max_node_count",
                        cluster_config.worker_count,
                    ),
                },
                "sparkConfigProperties": {
                    "configurationType": "File",
                    "content": self._format_spark_config(
                        self.translate_to_spark_config(cluster_config),
                    ),
                },
            },
        }

    def _format_spark_config(self, config: dict[str, Any]) -> str:
        """Format Spark config dictionary to properties file format."""
        return "\n".join(f"{key}={value}" for key, value in config.items())

    # --- Azure Synapse API Methods for Real Execution ---

    def submit_job(
        self,
        code_path: str | Path,
        cluster_config: ClusterConfig | None = None,
        pool_name: str = "sparkoptima-pool",
        timeout_minutes: int = 60,
    ) -> dict[str, Any]:
        """Submit a Spark job to Azure Synapse via REST API.

        Args:
            code_path: Path to Python file or notebook containing Spark code.
            cluster_config: Cluster configuration.
            pool_name: Name for the Spark pool.
            timeout_minutes: Maximum job duration.

        Returns:
            Dictionary with job submission result.

        Raises:
            RuntimeError: If API credentials are not configured.

        """
        import os

        import httpx

        # Get Azure credentials from environment
        tenant_id = os.environ.get("AZURE_TENANT_ID", "")
        client_id = os.environ.get("AZURE_CLIENT_ID", "")
        client_secret = os.environ.get("AZURE_CLIENT_SECRET", "")
        subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID", "")
        resource_group = os.environ.get("AZURE_RESOURCE_GROUP", "")
        workspace_name = os.environ.get("AZURE_SYNAPSE_WORKSPACE", "")

        if not all([tenant_id, client_id, client_secret, subscription_id, resource_group, workspace_name]):
            raise RuntimeError(
                "Azure credentials not configured. Set AZURE_TENANT_ID, "
                "AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_SUBSCRIPTION_ID, "
                "AZURE_RESOURCE_GROUP, AZURE_SYNAPSE_WORKSPACE environment variables.",
            )

        # Get access token
        token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
        token_data = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": "https://dev.azuresynapse.net/.default",
        }

        try:
            token_response = httpx.post(token_url, data=token_data, timeout=30)
            token_response.raise_for_status()
            access_token = token_response.json()["access_token"]
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to get Azure access token: {e}")
            return {"success": False, "error": f"Authentication failed: {e}"}

        # Read code
        code = Path(code_path).read_text(encoding="utf-8")

        # API URL
        api_url = (
            f"https://{workspace_name}.dev.azuresynapse.net"
            f"/subscriptions/{subscription_id}"
            f"/resourceGroups/{resource_group}"
            f"/sparkPools/{pool_name}"
            f"/sparkJob"
        )

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        # Submit job
        job_payload = {
            "name": "spark-optima-job",
            "properties": {
                "file": code,
                "args": [],
                "sparkConfigProperties": self.translate_to_spark_config(
                    cluster_config
                    or self.recommend_config(
                        ResourceSpec(cpu_cores=8, memory_gb=32),
                        spark_version="3.3.0",
                    ),
                ),
                "executionType": "spark",
            },
        }

        try:
            response = httpx.post(api_url, headers=headers, json=job_payload, timeout=30)
            response.raise_for_status()
            job_data = response.json()
            job_id = job_data.get("id", "")

            return {
                "success": True,
                "job_id": job_id,
                "pool_name": pool_name,
                "status": "submitted",
            }

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error submitting job: {e}")
            return {
                "success": False,
                "error": f"HTTP {e.response.status_code}: {e.response.text}",
            }
        except (KeyError, ValueError) as e:
            logger.error(f"Invalid response: {e}")
            return {"success": False, "error": f"Invalid response: {e}"}

    def get_job_status(
        self,
        job_id: str,
        pool_name: str = "sparkoptima-pool",
    ) -> dict[str, Any]:
        """Get status of a submitted Spark job.

        Args:
            job_id: Synapse job ID.
            pool_name: Spark pool name.

        Returns:
            Dictionary with job status.

        """
        import os

        import httpx

        # Get access token (same as submit_job)
        tenant_id = os.environ.get("AZURE_TENANT_ID", "")
        client_id = os.environ.get("AZURE_CLIENT_ID", "")
        client_secret = os.environ.get("AZURE_CLIENT_SECRET", "")
        subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID", "")
        resource_group = os.environ.get("AZURE_RESOURCE_GROUP", "")
        workspace_name = os.environ.get("AZURE_SYNAPSE_WORKSPACE", "")

        token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
        token_data = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": "https://dev.azuresynapse.net/.default",
        }

        try:
            token_response = httpx.post(token_url, data=token_data, timeout=30)
            token_response.raise_for_status()
            access_token = token_response.json()["access_token"]
        except httpx.HTTPStatusError as e:
            return {"job_id": job_id, "status": "error", "error": f"Authentication failed: {e}"}

        api_url = (
            f"https://{workspace_name}.dev.azuresynapse.net"
            f"/subscriptions/{subscription_id}"
            f"/resourceGroups/{resource_group}"
            f"/sparkPools/{pool_name}"
            f"/sparkJob/{job_id}"
        )

        headers = {"Authorization": f"Bearer {access_token}"}

        try:
            response = httpx.get(api_url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            state = data.get("properties", {}).get("state", "UNKNOWN")

            return {
                "job_id": job_id,
                "pool_name": pool_name,
                "state": state,
                "status": self._map_synapse_state(state),
                "message": data.get("properties", {}).get("message", ""),
            }

        except httpx.HTTPStatusError as e:
            return {"job_id": job_id, "status": "error", "error": str(e)}

    def get_job_results(
        self,
        job_id: str,
        pool_name: str = "sparkoptima-pool",
    ) -> dict[str, Any]:
        """Get results of a completed Spark job.

        Args:
            job_id: Synapse job ID.
            pool_name: Spark pool name.

        Returns:
            Dictionary with job results.

        """
        import os

        import httpx

        # Get access token
        tenant_id = os.environ.get("AZURE_TENANT_ID", "")
        client_id = os.environ.get("AZURE_CLIENT_ID", "")
        client_secret = os.environ.get("AZURE_CLIENT_SECRET", "")
        subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID", "")
        resource_group = os.environ.get("AZURE_RESOURCE_GROUP", "")
        workspace_name = os.environ.get("AZURE_SYNAPSE_WORKSPACE", "")

        token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
        token_data = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": "https://dev.azuresynapse.net/.default",
        }

        try:
            token_response = httpx.post(token_url, data=token_data, timeout=30)
            token_response.raise_for_status()
            access_token = token_response.json()["access_token"]
        except httpx.HTTPStatusError as e:
            return {"job_id": job_id, "status": "error", "error": f"Authentication failed: {e}"}

        api_url = (
            f"https://{workspace_name}.dev.azuresynapse.net"
            f"/subscriptions/{subscription_id}"
            f"/resourceGroups/{resource_group}"
            f"/sparkPools/{pool_name}"
            f"/sparkJob/{job_id}"
        )

        headers = {"Authorization": f"Bearer {access_token}"}

        try:
            response = httpx.get(api_url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            props = data.get("properties", {})

            return {
                "job_id": job_id,
                "pool_name": pool_name,
                "state": props.get("state", "UNKNOWN"),
                "status": self._map_synapse_state(props.get("state", "UNKNOWN")),
                "output": props.get("result", ""),
                "error_message": props.get("errorMessage", ""),
                "execution_time": props.get("durationInMilliseconds", 0) / 1000.0,
            }

        except httpx.HTTPStatusError as e:
            return {"job_id": job_id, "status": "error", "error": str(e)}

    def _map_synapse_state(self, state: str) -> str:
        """Map Azure Synapse job state to simple status.

        Args:
            state: Synapse job state.

        Returns:
            Simplified status string.

        """
        state_map = {
            "Queued": "pending",
            "Scheduled": "pending",
            "Running": "running",
            "Cancelling": "stopping",
            "Cancelled": "stopped",
            "Succeeded": "completed",
            "Failed": "failed",
            "TimedOut": "timeout",
        }
        return state_map.get(state, "unknown")
