# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Databricks platform implementation for Spark Optima.

This module provides the DatabricksPlatform class for Databricks clusters
on AWS and Azure, including support for various node types and
DBU (Databricks Unit) pricing.

Cost estimates apply a curated regional price multiplier keyed by the
compound "<cloud_provider>:<region>" identifier (baselines: aws:us-east-1
and azure:eastus); see :mod:`spark_optima.platforms.pricing`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from spark_optima.platforms.base import Platform
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


class DatabricksPlatform(Platform):
    """Databricks platform for Spark workloads.

    Databricks runs on AWS and Azure cloud providers with various
    node instance types. Pricing is based on DBUs (Databricks Units)
    which vary by instance type and cloud provider. Cost estimates scale
    the DBU cost by a curated regional price multiplier keyed by
    "<cloud_provider>:<region>" (e.g., "aws:eu-west-1", "azure:westeurope").

    Attributes:
        name: Platform identifier "databricks".
        display_name: Human-readable name "Databricks".
        cloud_provider: Either "aws" or "azure".

    Example:
        >>> platform = DatabricksPlatform(cloud_provider="aws")
        >>> worker = platform.get_worker_type("i3.xlarge")
        >>> print(f"Cost: ${worker.cost.unit_cost_per_hour}/hour")

    """

    # AWS node type specifications
    # Source: https://docs.databricks.com/en/compute/configure.html
    AWS_NODE_TYPES: dict[str, dict[str, Any]] = {
        # General purpose (m5 series)
        "m5.large": {"cpu_cores": 2, "memory_gb": 8, "dbu_per_hour": 0.75},
        "m5.xlarge": {"cpu_cores": 4, "memory_gb": 16, "dbu_per_hour": 1.5},
        "m5.2xlarge": {"cpu_cores": 8, "memory_gb": 32, "dbu_per_hour": 3.0},
        "m5.4xlarge": {"cpu_cores": 16, "memory_gb": 64, "dbu_per_hour": 6.0},
        "m5.8xlarge": {"cpu_cores": 32, "memory_gb": 128, "dbu_per_hour": 12.0},
        "m5.12xlarge": {"cpu_cores": 48, "memory_gb": 192, "dbu_per_hour": 18.0},
        "m5.16xlarge": {"cpu_cores": 64, "memory_gb": 256, "dbu_per_hour": 24.0},
        "m5.24xlarge": {"cpu_cores": 96, "memory_gb": 384, "dbu_per_hour": 36.0},
        # Memory optimized (r5 series)
        "r5.large": {"cpu_cores": 2, "memory_gb": 16, "dbu_per_hour": 1.0},
        "r5.xlarge": {"cpu_cores": 4, "memory_gb": 32, "dbu_per_hour": 2.0},
        "r5.2xlarge": {"cpu_cores": 8, "memory_gb": 64, "dbu_per_hour": 4.0},
        "r5.4xlarge": {"cpu_cores": 16, "memory_gb": 128, "dbu_per_hour": 8.0},
        "r5.8xlarge": {"cpu_cores": 32, "memory_gb": 256, "dbu_per_hour": 16.0},
        "r5.12xlarge": {"cpu_cores": 48, "memory_gb": 384, "dbu_per_hour": 24.0},
        "r5.16xlarge": {"cpu_cores": 64, "memory_gb": 512, "dbu_per_hour": 32.0},
        "r5.24xlarge": {"cpu_cores": 96, "memory_gb": 768, "dbu_per_hour": 48.0},
        # Storage optimized (i3 series - popular for Databricks)
        "i3.xlarge": {"cpu_cores": 4, "memory_gb": 30.5, "dbu_per_hour": 1.5},
        "i3.2xlarge": {"cpu_cores": 8, "memory_gb": 61, "dbu_per_hour": 3.0},
        "i3.4xlarge": {"cpu_cores": 16, "memory_gb": 122, "dbu_per_hour": 6.0},
        "i3.8xlarge": {"cpu_cores": 32, "memory_gb": 244, "dbu_per_hour": 12.0},
        "i3.16xlarge": {"cpu_cores": 64, "memory_gb": 488, "dbu_per_hour": 24.0},
        # Next-gen storage optimized (i4i series)
        "i4i.xlarge": {"cpu_cores": 4, "memory_gb": 32, "dbu_per_hour": 1.5},
        "i4i.2xlarge": {"cpu_cores": 8, "memory_gb": 64, "dbu_per_hour": 3.0},
        "i4i.4xlarge": {"cpu_cores": 16, "memory_gb": 128, "dbu_per_hour": 6.0},
        "i4i.8xlarge": {"cpu_cores": 32, "memory_gb": 256, "dbu_per_hour": 12.0},
        "i4i.16xlarge": {"cpu_cores": 64, "memory_gb": 512, "dbu_per_hour": 24.0},
        # Compute optimized (c5 series)
        "c5.xlarge": {"cpu_cores": 4, "memory_gb": 8, "dbu_per_hour": 1.5},
        "c5.2xlarge": {"cpu_cores": 8, "memory_gb": 16, "dbu_per_hour": 3.0},
        "c5.4xlarge": {"cpu_cores": 16, "memory_gb": 32, "dbu_per_hour": 6.0},
        "c5.9xlarge": {"cpu_cores": 36, "memory_gb": 72, "dbu_per_hour": 13.5},
        "c5.18xlarge": {"cpu_cores": 72, "memory_gb": 144, "dbu_per_hour": 27.0},
        # Graviton (ARM-based, cost-effective)
        "m6g.large": {"cpu_cores": 2, "memory_gb": 8, "dbu_per_hour": 0.68},
        "m6g.xlarge": {"cpu_cores": 4, "memory_gb": 16, "dbu_per_hour": 1.35},
        "m6g.2xlarge": {"cpu_cores": 8, "memory_gb": 32, "dbu_per_hour": 2.7},
        "m6g.4xlarge": {"cpu_cores": 16, "memory_gb": 64, "dbu_per_hour": 5.4},
        "m6g.8xlarge": {"cpu_cores": 32, "memory_gb": 128, "dbu_per_hour": 10.8},
        "m6g.12xlarge": {"cpu_cores": 48, "memory_gb": 192, "dbu_per_hour": 16.2},
        "m6g.16xlarge": {"cpu_cores": 64, "memory_gb": 256, "dbu_per_hour": 21.6},
    }

    # Azure node type specifications
    # Source: https://docs.microsoft.com/en-us/azure/databricks/clusters/
    AZURE_NODE_TYPES: dict[str, dict[str, Any]] = {
        # Standard_DSv3 series
        "Standard_DS3_v2": {"cpu_cores": 4, "memory_gb": 14, "dbu_per_hour": 0.75},
        "Standard_DS4_v2": {"cpu_cores": 8, "memory_gb": 28, "dbu_per_hour": 1.5},
        "Standard_DS5_v2": {"cpu_cores": 16, "memory_gb": 56, "dbu_per_hour": 3.0},
        # Standard_DDv4 series
        "Standard_D8d_v4": {"cpu_cores": 8, "memory_gb": 32, "dbu_per_hour": 1.5},
        "Standard_D16d_v4": {"cpu_cores": 16, "memory_gb": 64, "dbu_per_hour": 3.0},
        "Standard_D32d_v4": {"cpu_cores": 32, "memory_gb": 128, "dbu_per_hour": 6.0},
        "Standard_D64d_v4": {"cpu_cores": 64, "memory_gb": 256, "dbu_per_hour": 12.0},
        # Standard_Ev3 series (memory optimized)
        "Standard_E4s_v3": {"cpu_cores": 4, "memory_gb": 32, "dbu_per_hour": 1.0},
        "Standard_E8s_v3": {"cpu_cores": 8, "memory_gb": 64, "dbu_per_hour": 2.0},
        "Standard_E16s_v3": {"cpu_cores": 16, "memory_gb": 128, "dbu_per_hour": 4.0},
        "Standard_E32s_v3": {"cpu_cores": 32, "memory_gb": 256, "dbu_per_hour": 8.0},
        "Standard_E64s_v3": {"cpu_cores": 64, "memory_gb": 432, "dbu_per_hour": 16.0},
        # Standard_Lsv2 series (storage optimized)
        "Standard_L8s_v2": {"cpu_cores": 8, "memory_gb": 64, "dbu_per_hour": 2.0},
        "Standard_L16s_v2": {"cpu_cores": 16, "memory_gb": 128, "dbu_per_hour": 4.0},
        "Standard_L32s_v2": {"cpu_cores": 32, "memory_gb": 256, "dbu_per_hour": 8.0},
        "Standard_L64s_v2": {"cpu_cores": 64, "memory_gb": 512, "dbu_per_hour": 16.0},
    }

    # Approximate DBU cost per hour (varies by tier and region)
    # These are approximate rates for Premium tier in US regions
    DBU_COST_PER_HOUR: dict[str, float] = {
        "aws": 0.15,  # $0.15 per DBU on AWS
        "azure": 0.17,  # $0.17 per DBU on Azure
    }

    def __init__(
        self,
        cloud_provider: str = "aws",
        region: str | None = None,
        dbu_cost_per_hour: float | None = None,
    ) -> None:
        """Initialize the Databricks platform.

        Args:
            cloud_provider: "aws" or "azure".
            region: Cloud region for pricing. Defaults to the cloud's baseline
                region ("us-east-1" for AWS, "eastus" for Azure).
            dbu_cost_per_hour: Custom DBU cost (uses default if None).

        Raises:
            ValueError: If cloud_provider is not supported.

        """
        super().__init__(
            name="databricks",
            display_name="Databricks",
            description=f"Databricks on {cloud_provider.upper()}",
        )

        if cloud_provider not in ["aws", "azure"]:
            raise ValueError(f"Unsupported cloud provider: {cloud_provider}")

        self.cloud_provider = cloud_provider
        self.region = region if region is not None else ("us-east-1" if cloud_provider == "aws" else "eastus")
        self.dbu_cost_per_hour = dbu_cost_per_hour or self.DBU_COST_PER_HOUR[cloud_provider]

        self._node_types = self.AWS_NODE_TYPES if cloud_provider == "aws" else self.AZURE_NODE_TYPES

        self._constraints = PlatformConstraints(
            min_workers=1,
            max_workers=1000,
            min_memory_gb=8,
            max_memory_gb=512,
            min_cores=2,
            max_cores=128,
            supported_spark_versions=[
                "3.0.0",
                "3.1.0",
                "3.1.2",
                "3.2.0",
                "3.2.1",
                "3.3.0",
                "3.4.0",
                "3.4.1",
                "3.5.0",
                "4.0.0",
            ],
            custom_config_keys={
                "node_type": "node_type_id",
                "driver_node_type": "driver_node_type_id",
                "num_workers": "num_workers",
                "autoscale": "autoscale",
                "spark_version": "spark_version",
                "cluster_name": "cluster_name",
            },
        )

    @property
    def constraints(self) -> PlatformConstraints:
        """Get platform resource constraints."""
        return self._constraints

    def _get_instance_size(self, memory_gb: float) -> InstanceSize:
        """Determine instance size category based on memory."""
        if memory_gb < 8:
            return InstanceSize.SMALL
        elif memory_gb < 16:
            return InstanceSize.MEDIUM
        elif memory_gb < 64:
            return InstanceSize.LARGE
        elif memory_gb < 128:
            return InstanceSize.XLARGE
        elif memory_gb < 256:
            return InstanceSize.XXLARGE
        else:
            return InstanceSize.XXXLARGE

    def _create_worker_type(self, name: str, spec: dict[str, Any]) -> WorkerType:
        """Create a WorkerType from node specification."""
        return WorkerType(
            name=name,
            size=self._get_instance_size(spec["memory_gb"]),
            resources=ResourceSpec(
                cpu_cores=spec["cpu_cores"],
                memory_gb=spec["memory_gb"],
                disk_gb=spec.get("disk_gb", 0),
            ),
            cost=CostModel(
                currency="USD",
                unit_cost_per_hour=self.dbu_cost_per_hour * spec["dbu_per_hour"],
                unit_name="DBU",
                granularity_minutes=1,
            ),
            description=f"Databricks {name} ({spec['cpu_cores']} vCPUs, "
            f"{spec['memory_gb']}GB RAM, {spec['dbu_per_hour']} DBU/hour)",
        )

    def get_worker_types(self) -> list[WorkerType]:
        """Get all available Databricks node types."""
        return [self._create_worker_type(name, spec) for name, spec in self._node_types.items()]

    def get_worker_type(self, name: str) -> WorkerType | None:
        """Get a specific node type by name."""
        spec = self._node_types.get(name)
        if spec:
            return self._create_worker_type(name, spec)
        return None

    def recommend_worker_type(
        self,
        target_memory_gb: float,
        target_cores: int,
        prefer_storage_optimized: bool = True,
    ) -> WorkerType:
        """Recommend the best worker type for given requirements.

        Args:
            target_memory_gb: Desired memory per worker.
            target_cores: Desired CPU cores per worker.
            prefer_storage_optimized: Prefer i3/i4i (AWS) or L-series (Azure).

        Returns:
            Recommended WorkerType.

        """
        candidates = self.get_worker_types()

        # Filter by storage-optimized preference
        if prefer_storage_optimized:
            if self.cloud_provider == "aws":
                storage_types = [w for w in candidates if w.name.startswith("i3") or w.name.startswith("i4i")]
            else:
                storage_types = [w for w in candidates if w.name.startswith("Standard_L")]

            if storage_types:
                candidates = storage_types

        # Find best match
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
            cost_per_dbu = worker.cost.unit_cost_per_hour

            # Weighted score
            score = memory_waste * 0.4 + cpu_waste * 0.3 + cost_per_dbu * 0.3

            if score < best_score:
                best_score = score
                best_match = worker

        # Fallback
        if best_match is None:
            if self.cloud_provider == "aws":
                return self.get_worker_type("i3.xlarge") or candidates[0]
            else:
                return self.get_worker_type("Standard_DS3_v2") or candidates[0]

        return best_match

    def recommend_config(
        self,
        resources: ResourceSpec,
        spark_version: str,
        worker_count: int | None = None,
    ) -> ClusterConfig:
        """Recommend optimal Databricks configuration."""
        # Recommend worker type
        memory_per_worker = min(resources.memory_gb / 8, 64)
        cores_per_worker = min(resources.cpu_cores / 8, 8)

        worker_type = self.recommend_worker_type(
            target_memory_gb=memory_per_worker,
            target_cores=int(cores_per_worker),
        )

        # Calculate worker count
        if worker_count is None:
            worker_count = self.get_optimal_worker_count(resources, worker_type)

        worker_count = max(worker_count, self._constraints.min_workers)

        # Use same type for driver (can be different in practice)
        return ClusterConfig(
            worker_type=worker_type,
            worker_count=worker_count,
            driver_type=worker_type,
            driver_count=1,
            spark_version=spark_version,
            platform_config={
                "cloud_provider": self.cloud_provider,
                "region": self.region,
                "autoscale_enabled": True,
                "min_workers": max(1, worker_count // 2),
                "max_workers": min(worker_count * 2, self._constraints.max_workers),
            },
        )

    def translate_to_spark_config(
        self,
        cluster_config: ClusterConfig,
    ) -> dict[str, Any]:
        """Translate cluster config to Spark configuration for Databricks."""
        worker = cluster_config.worker_type
        r = worker.resources

        # Calculate executor settings
        # Databricks typically runs multiple executors per worker
        executor_cores = min(4, r.cpu_cores)
        num_executors = max(1, r.cpu_cores // executor_cores)
        executor_memory = int((r.memory_gb * 0.9) / num_executors)

        config = {
            # Databricks cluster config
            "spark.databricks.cluster.profile": ("singleNode" if cluster_config.worker_count == 0 else "multiNode"),
            # Executor configuration
            "spark.executor.instances": str(cluster_config.worker_count * num_executors),
            "spark.executor.cores": str(executor_cores),
            "spark.executor.memory": f"{executor_memory}g",
            # Dynamic allocation
            "spark.dynamicAllocation.enabled": "true",
            "spark.dynamicAllocation.minExecutors": str(
                cluster_config.platform_config.get("min_workers", 1),
            ),
            "spark.dynamicAllocation.maxExecutors": str(
                cluster_config.platform_config.get(
                    "max_workers",
                    cluster_config.worker_count * num_executors,
                ),
            ),
            # Serialization
            "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
            # SQL configuration
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.enabled": "true",
            "spark.sql.adaptive.skewJoin.enabled": "true",
            # Shuffle partitions
            "spark.sql.shuffle.partitions": str(
                max(200, cluster_config.worker_count * r.cpu_cores * 2),
            ),
            # Databricks-specific optimizations
            "spark.databricks.delta.optimizeWrite.enabled": "true",
            "spark.databricks.delta.autoCompact.enabled": "true",
        }

        return config

    def estimate_cost(
        self,
        cluster_config: ClusterConfig,
        duration_hours: float,
    ) -> dict[str, Any]:
        """Estimate cost for Databricks cluster.

        The DBU cost is scaled by a curated regional price multiplier looked
        up with the compound "<cloud_provider>:<region>" key (baselines:
        aws:us-east-1 and azure:eastus).
        """
        worker = cluster_config.worker_type
        driver = cluster_config.driver_type or worker

        # Calculate DBUs
        worker_spec = self._node_types.get(worker.name, {})
        driver_spec = self._node_types.get(driver.name, worker_spec)

        worker_dbu = worker_spec.get("dbu_per_hour", 1)
        driver_dbu = driver_spec.get("dbu_per_hour", worker_dbu)

        total_worker_dbu = worker_dbu * cluster_config.worker_count
        total_dbu_per_hour = total_worker_dbu + driver_dbu
        total_dbu = total_dbu_per_hour * duration_hours

        # Calculate cost; the regional multiplier scales the baseline DBU
        # rates and is keyed by cloud + region (e.g., "aws:eu-west-1")
        region_multiplier = get_region_multiplier(self.name, f"{self.cloud_provider}:{self.region}")
        worker_cost = worker.cost.calculate(duration_hours, cluster_config.worker_count) * region_multiplier
        driver_cost = driver.cost.calculate(duration_hours, 1) * region_multiplier
        total_cost = worker_cost + driver_cost

        return {
            "platform": self.name,
            "cloud_provider": self.cloud_provider,
            "currency": "USD",
            "region": self.region,
            "duration_hours": duration_hours,
            "total_cost": total_cost,
            "breakdown": {
                "total_dbu": total_dbu,
                "dbu_rate_per_hour": self.dbu_cost_per_hour,
                "worker_dbu_per_hour": worker_dbu,
                "driver_dbu_per_hour": driver_dbu,
                "worker_count": cluster_config.worker_count,
                "worker_type": worker.name,
                "driver_type": driver.name,
                "region": self.region,
                "region_multiplier": region_multiplier,
            },
            "notes": "Cost estimate based on DBU pricing with a curated regional multiplier "
            "(does not include cloud instance costs)",
        }

    def get_cluster_spec(
        self,
        cluster_config: ClusterConfig,
        cluster_name: str = "spark-optima-cluster",
    ) -> dict[str, Any]:
        """Generate Databricks cluster specification for API usage."""
        return {
            "cluster_name": cluster_name,
            "spark_version": cluster_config.spark_version,
            "node_type_id": cluster_config.worker_type.name,
            "driver_node_type_id": (
                cluster_config.driver_type.name if cluster_config.driver_type else cluster_config.worker_type.name
            ),
            "num_workers": cluster_config.worker_count,
            "autoscale": {
                "min_workers": cluster_config.platform_config.get("min_workers", 1),
                "max_workers": cluster_config.platform_config.get(
                    "max_workers",
                    cluster_config.worker_count,
                ),
            }
            if cluster_config.platform_config.get("autoscale_enabled")
            else None,
            "spark_conf": self.translate_to_spark_config(cluster_config),
        }

    # --- REST API Methods for Real Execution ---

    def submit_job(
        self,
        code_path: str | Path,
        cluster_config: ClusterConfig,
        cluster_name: str = "spark-optima-cluster",
        timeout_minutes: int = 60,
    ) -> dict[str, Any]:
        """Submit a Spark job to Databricks via REST API.

        Args:
            code_path: Path to Python file containing Spark code.
            cluster_config: Cluster configuration.
            cluster_name: Name for the temporary cluster.
            timeout_minutes: Maximum job duration.

        Returns:
            Dictionary with job submission result.

        Raises:
            RuntimeError: If API credentials are not configured.

        """
        import httpx

        api_url = self._get_api_url()
        headers = self._get_auth_headers()

        # Read code
        code = Path(code_path).read_text(encoding="utf-8")

        try:
            # Create cluster
            cluster_spec = self.get_cluster_spec(
                cluster_config,
                cluster_name=cluster_name,
            )
            response = httpx.post(
                f"{api_url}/api/2.0/clusters/create",
                headers=headers,
                json=cluster_spec,
                timeout=30,
            )
            response.raise_for_status()
            cluster_id = response.json()["cluster_id"]

            # Submit job
            job_payload = {
                "run_name": "spark-optima-optimization",
                "new_cluster": cluster_spec["spark_conf"],
                "spark_python_task": {
                    "python_file": code,
                },
                "timeout_seconds": timeout_minutes * 60,
            }
            response = httpx.post(
                f"{api_url}/api/2.0/jobs/runs/submit",
                headers=headers,
                json=job_payload,
                timeout=30,
            )
            response.raise_for_status()
            run_id = response.json()["run_id"]

            return {
                "success": True,
                "run_id": run_id,
                "cluster_id": cluster_id,
                "status": "submitted",
            }

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error submitting job: {e}")
            return {
                "success": False,
                "error": f"HTTP {e.response.status_code}: {e.response.text}",
            }
        except httpx.RequestError as e:
            logger.error(f"Request error submitting job: {e}")
            return {
                "success": False,
                "error": f"Request error: {e}",
            }
        except (KeyError, ValueError) as e:
            logger.error(f"Invalid response: {e}")
            return {
                "success": False,
                "error": f"Invalid response: {e}",
            }

    def get_job_status(self, run_id: str) -> dict[str, Any]:
        """Get status of a submitted job.

        Args:
            run_id: Databricks job run ID.

        Returns:
            Dictionary with job status.

        """
        import httpx

        api_url = self._get_api_url()
        headers = self._get_auth_headers()

        try:
            response = httpx.get(
                f"{api_url}/api/2.0/jobs/runs/get",
                headers=headers,
                params={"run_id": run_id},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            state = data.get("state", {})
            life_cycle_state = state.get("life_cycle_state", "UNKNOWN")
            result_state = state.get("result_state", "UNKNOWN")

            return {
                "run_id": run_id,
                "life_cycle_state": life_cycle_state,
                "result_state": result_state,
                "status": self._map_databricks_state(life_cycle_state),
                "message": state.get("state_message", ""),
            }

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error getting job status: {e}")
            return {"run_id": run_id, "status": "error", "error": str(e)}
        except httpx.RequestError as e:
            logger.error(f"Request error getting job status: {e}")
            return {"run_id": run_id, "status": "error", "error": str(e)}

    def get_job_results(self, run_id: str) -> dict[str, Any]:
        """Get results of a completed job.

        Args:
            run_id: Databricks job run ID.

        Returns:
            Dictionary with job results.

        """
        import httpx

        api_url = self._get_api_url()
        headers = self._get_auth_headers()

        try:
            response = httpx.get(
                f"{api_url}/api/2.0/jobs/runs/get-output",
                headers=headers,
                params={"run_id": run_id},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            return {
                "run_id": run_id,
                "status": "completed",
                "output": data.get("notebook_output", {}).get("result", ""),
                "logs": data.get("logs", ""),
                "error": data.get("error", ""),
            }

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error getting job results: {e}")
            return {"run_id": run_id, "status": "error", "error": str(e)}
        except httpx.RequestError as e:
            logger.error(f"Request error getting job results: {e}")
            return {"run_id": run_id, "status": "error", "error": str(e)}

    def _get_api_url(self) -> str:
        """Get Databricks API URL from environment or config.

        Returns:
            API base URL.

        Raises:
            RuntimeError: If not configured.

        """
        import os

        host = os.environ.get("DATABRICKS_HOST", "")
        token = os.environ.get("DATABRICKS_TOKEN", "")

        if not host or not token:
            raise RuntimeError(
                "Databricks credentials not configured. "
                "Set DATABRICKS_HOST and DATABRICKS_TOKEN environment variables.",
            )

        return f"https://{host}"

    def _get_auth_headers(self) -> dict[str, str]:
        """Get authentication headers for Databricks API.

        Returns:
            Dictionary with Authorization header.

        """
        import os

        token = os.environ.get("DATABRICKS_TOKEN", "")
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def _map_databricks_state(self, life_cycle_state: str) -> str:
        """Map Databricks lifecycle state to simple status.

        Args:
            life_cycle_state: Databricks lifecycle state.

        Returns:
            Simplified status string.

        """
        state_map = {
            "PENDING": "pending",
            "RUNNING": "running",
            "TERMINATING": "terminating",
            "TERMINATED": "terminated",
        }
        return state_map.get(life_cycle_state, "unknown")
