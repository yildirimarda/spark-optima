# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""AWS Glue platform implementation for Spark Optima.

This module provides the AWSGluePlatform class for AWS Glue ETL jobs,
including support for all worker types (G.025X through G.16X) and
Glue 2.0/3.0/4.0/5.0 versions.

Cost estimates apply a curated regional price multiplier (relative to the
us-east-1 baseline) based on the configured region; see
:mod:`spark_optima.platforms.pricing`. When live pricing is opted in
(``SPARK_OPTIMA_LIVE_PRICING=1``), the live regional DPU-hour rate from the
AWS Pricing API replaces the static baseline x multiplier; see
:mod:`spark_optima.platforms.live_pricing`.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class AWSGluePlatform(Platform):
    """AWS Glue platform for Spark ETL jobs.

    AWS Glue supports multiple worker types and versions:
    - Glue 2.0/3.0/4.0/5.0 with different Spark versions
    - Worker types: G.025X, G.1X, G.2X, G.4X, G.8X, G.12X, G.16X
    - R-type workers for memory-intensive workloads

    Attributes:
        name: Platform identifier "aws_glue".
        display_name: Human-readable name "AWS Glue".

    Example:
        >>> platform = AWSGluePlatform()
        >>> worker = platform.get_worker_type("G.2X")
        >>> print(f"G.2X: {worker.resources.cpu_cores} vCPUs, {worker.resources.memory_gb}GB RAM")

    """

    # AWS Glue worker type specifications
    # Source: https://docs.aws.amazon.com/glue/latest/dg/worker-types.html
    WORKER_SPECS: dict[str, dict[str, Any]] = {
        "G.025X": {
            "dpu": 0.25,
            "cpu_cores": 2,
            "memory_gb": 4,
            "disk_gb": 20,
            "size": InstanceSize.XSMALL,
        },
        "G.1X": {
            "dpu": 1,
            "cpu_cores": 4,
            "memory_gb": 16,
            "disk_gb": 64,
            "size": InstanceSize.SMALL,
        },
        "G.2X": {
            "dpu": 2,
            "cpu_cores": 8,
            "memory_gb": 32,
            "disk_gb": 128,
            "size": InstanceSize.MEDIUM,
        },
        "G.4X": {
            "dpu": 4,
            "cpu_cores": 16,
            "memory_gb": 64,
            "disk_gb": 256,
            "size": InstanceSize.LARGE,
        },
        "G.8X": {
            "dpu": 8,
            "cpu_cores": 32,
            "memory_gb": 128,
            "disk_gb": 512,
            "size": InstanceSize.XLARGE,
        },
        "G.12X": {
            "dpu": 12,
            "cpu_cores": 48,
            "memory_gb": 192,
            "disk_gb": 768,
            "size": InstanceSize.XXLARGE,
        },
        "G.16X": {
            "dpu": 16,
            "cpu_cores": 64,
            "memory_gb": 256,
            "disk_gb": 1024,
            "size": InstanceSize.XXXLARGE,
        },
        # R-type workers for memory-intensive workloads
        "R.2X": {
            "dpu": 2,
            "cpu_cores": 8,
            "memory_gb": 64,
            "disk_gb": 128,
            "size": InstanceSize.MEDIUM,
        },
        "R.4X": {
            "dpu": 4,
            "cpu_cores": 16,
            "memory_gb": 128,
            "disk_gb": 256,
            "size": InstanceSize.LARGE,
        },
        "R.8X": {
            "dpu": 8,
            "cpu_cores": 32,
            "memory_gb": 256,
            "disk_gb": 512,
            "size": InstanceSize.XLARGE,
        },
    }

    # Approximate cost per DPU per hour (varies by region)
    # These are US-East-1 estimates
    DPU_COST_PER_HOUR: float = 0.44  # $0.44 per DPU-hour for Glue 2.0+

    # Glue version to Spark version mapping
    GLUE_TO_SPARK_VERSION: dict[str, str] = {
        "2.0": "2.4.3",
        "3.0": "3.1.1",
        "4.0": "3.3.0",
        "5.0": "3.5.0",
    }

    def __init__(
        self,
        region: str = "us-east-1",
        dpu_cost_per_hour: float | None = None,
    ) -> None:
        """Initialize the AWS Glue platform.

        Args:
            region: AWS region for pricing.
            dpu_cost_per_hour: Custom DPU cost (uses default if None).

        """
        super().__init__(
            name="aws_glue",
            display_name="AWS Glue",
            description="AWS Glue serverless Spark ETL service",
        )
        self.region = region
        self.dpu_cost_per_hour = dpu_cost_per_hour or self.DPU_COST_PER_HOUR

        self._constraints = PlatformConstraints(
            min_workers=2,  # Minimum 2 workers for Glue 2.0+
            max_workers=299,  # Maximum workers per job
            min_memory_gb=4,
            max_memory_gb=256,
            min_cores=2,
            max_cores=64,
            supported_spark_versions=[
                "2.4.3",  # Glue 2.0
                "3.1.1",  # Glue 3.0
                "3.3.0",  # Glue 4.0
                "3.5.0",  # Glue 5.0
            ],
            custom_config_keys={
                "worker_type": "WorkerType",
                "number_of_workers": "NumberOfWorkers",
                "glue_version": "GlueVersion",
                "job_bookmark": "JobBookmarkOption",
                "temp_dir": "TempDir",
            },
        )

    @property
    def constraints(self) -> PlatformConstraints:
        """Get platform resource constraints."""
        return self._constraints

    def _create_worker_type(self, name: str, spec: dict[str, Any]) -> WorkerType:
        """Create a WorkerType from specification.

        Args:
            name: Worker type name.
            spec: Worker specifications dict.

        Returns:
            Configured WorkerType instance.

        """
        return WorkerType(
            name=name,
            size=spec["size"],
            resources=ResourceSpec(
                cpu_cores=spec["cpu_cores"],
                memory_gb=spec["memory_gb"],
                disk_gb=spec["disk_gb"],
            ),
            cost=CostModel(
                currency="USD",
                unit_cost_per_hour=self.dpu_cost_per_hour * spec["dpu"],
                unit_name="DPU",
                granularity_minutes=1,  # Glue bills per second, rounded to minute
            ),
            description=f"AWS Glue {name} worker ({spec['dpu']} DPU, "
            f"{spec['cpu_cores']} vCPUs, {spec['memory_gb']}GB RAM)",
        )

    def get_worker_types(self) -> list[WorkerType]:
        """Get all available AWS Glue worker types.

        Returns:
            List of WorkerType for all supported worker types.

        """
        return [self._create_worker_type(name, spec) for name, spec in self.WORKER_SPECS.items()]

    def get_worker_type(self, name: str) -> WorkerType | None:
        """Get a specific worker type by name.

        Args:
            name: Worker type name (e.g., "G.1X", "G.2X").

        Returns:
            WorkerType if found, None otherwise.

        """
        spec = self.WORKER_SPECS.get(name.upper())
        if spec:
            return self._create_worker_type(name.upper(), spec)
        return None

    def recommend_worker_type(
        self,
        target_memory_gb: float,
        target_cores: int,
        prefer_memory_optimized: bool = False,
    ) -> WorkerType:
        """Recommend the best worker type for given requirements.

        Args:
            target_memory_gb: Desired memory per worker.
            target_cores: Desired CPU cores per worker.
            prefer_memory_optimized: Prefer R-type workers for memory-intensive.

        Returns:
            Recommended WorkerType.

        """
        candidates = self.get_worker_types()

        # Filter by memory-optimized preference
        if prefer_memory_optimized:
            r_types = [w for w in candidates if w.name.startswith("R.")]
            if r_types:
                candidates = r_types
        else:
            g_types = [w for w in candidates if w.name.startswith("G.")]
            if g_types:
                candidates = g_types

        # Find best match (meets requirements with minimal waste)
        best_match = None
        best_score = float("inf")

        for worker in candidates:
            r = worker.resources

            # Must meet minimum requirements
            if r.memory_gb < target_memory_gb or r.cpu_cores < target_cores:
                continue

            # Score based on resource efficiency (lower is better)
            memory_waste = r.memory_gb - target_memory_gb
            cpu_waste = r.cpu_cores - target_cores
            cost_factor = worker.cost.unit_cost_per_hour

            # Weighted score
            score = memory_waste * 0.4 + cpu_waste * 0.3 + cost_factor * 0.3

            if score < best_score:
                best_score = score
                best_match = worker

        # Fallback to G.1X if no match found
        if best_match is None:
            return self.get_worker_type("G.1X") or self.get_worker_types()[0]

        return best_match

    def recommend_config(
        self,
        resources: ResourceSpec,
        spark_version: str,
        worker_count: int | None = None,
    ) -> ClusterConfig:
        """Recommend optimal AWS Glue configuration.

        Args:
            resources: Target total resources.
            spark_version: Target Spark version (maps to Glue version).
            worker_count: Optional specific worker count.

        Returns:
            Recommended ClusterConfig.

        """
        # Recommend worker type based on resource requirements
        memory_per_worker = min(resources.memory_gb / 10, 64)  # Target ~10 workers
        cores_per_worker = min(resources.cpu_cores / 10, 8)

        worker_type = self.recommend_worker_type(
            target_memory_gb=memory_per_worker,
            target_cores=int(cores_per_worker),
        )

        # Calculate worker count
        if worker_count is None:
            worker_count = self.get_optimal_worker_count(resources, worker_type)

        # Ensure minimum workers for Glue
        worker_count = max(worker_count, self._constraints.min_workers)

        # Map Spark version to Glue version
        glue_version = self._get_glue_version(spark_version)

        return ClusterConfig(
            worker_type=worker_type,
            worker_count=worker_count,
            driver_type=None,  # Glue manages driver automatically
            driver_count=1,
            spark_version=spark_version,
            platform_config={
                "glue_version": glue_version,
                "worker_type": worker_type.name,
                "number_of_workers": worker_count,
            },
        )

    def _get_glue_version(self, spark_version: str) -> str:
        """Map Spark version to AWS Glue version.

        Args:
            spark_version: Spark version string.

        Returns:
            Glue version string.

        """
        # Reverse lookup
        for glue_ver, spark_ver in self.GLUE_TO_SPARK_VERSION.items():
            if spark_version.startswith(spark_ver[:3]):  # Match major.minor
                return glue_ver

        # Default to latest
        return "4.0"

    def translate_to_spark_config(
        self,
        cluster_config: ClusterConfig,
    ) -> dict[str, Any]:
        """Translate cluster config to Spark configuration for AWS Glue.

        Args:
            cluster_config: AWS Glue cluster configuration.

        Returns:
            Dictionary of Spark configuration parameters.

        """
        worker = cluster_config.worker_type
        r = worker.resources

        # Calculate executor settings
        # AWS Glue typically runs 1 executor per worker
        executor_cores = r.cpu_cores
        executor_memory = int(r.memory_gb * 0.9)  # Leave overhead

        config = {
            # AWS Glue specific
            "glue.workerType": worker.name,
            "glue.numberOfWorkers": str(cluster_config.worker_count),
            # Executor configuration
            "spark.executor.instances": str(cluster_config.worker_count),
            "spark.executor.cores": str(executor_cores),
            "spark.executor.memory": f"{executor_memory}g",
            # Dynamic allocation (enabled by default in Glue)
            "spark.dynamicAllocation.enabled": "true",
            "spark.dynamicAllocation.minExecutors": str(self._constraints.min_workers),
            "spark.dynamicAllocation.maxExecutors": str(cluster_config.worker_count),
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
        }

        # Add Glue-specific configs
        if cluster_config.platform_config and "glue_version" in cluster_config.platform_config:
            config["glue.version"] = cluster_config.platform_config["glue_version"]

        return config

    def estimate_cost(
        self,
        cluster_config: ClusterConfig,
        duration_hours: float,
    ) -> dict[str, Any]:
        """Estimate cost for AWS Glue job.

        The DPU cost is scaled by a curated regional price multiplier
        (relative to the us-east-1 baseline) for the configured region.
        When live pricing is opted in (``SPARK_OPTIMA_LIVE_PRICING=1``) and a
        live regional DPU rate is available, it replaces the static baseline
        rate x multiplier and the result is labeled ``pricing_source: live``.

        Args:
            cluster_config: Cluster configuration.
            duration_hours: Expected runtime in hours.

        Returns:
            Cost breakdown with DPU-hour calculation.

        """
        worker = cluster_config.worker_type
        worker_spec = self.WORKER_SPECS.get(worker.name, {})
        dpu_per_worker = worker_spec.get("dpu", 1)

        total_dpus = dpu_per_worker * cluster_config.worker_count
        dpu_hours = total_dpus * duration_hours

        # AWS Glue bills per second with 1-minute minimum. Opt-in live
        # pricing replaces the static baseline x regional multiplier with the
        # live regional DPU rate (returns None unless explicitly enabled).
        live_dpu_rate = get_live_hourly_rate(self.name, region=self.region)
        if live_dpu_rate is not None:
            pricing_source = "live"
            dpu_rate_per_hour = live_dpu_rate
            region_multiplier = 1.0  # Live rates are already region-specific
            live_cost_model = replace(worker.cost, unit_cost_per_hour=live_dpu_rate * dpu_per_worker)
            cost = live_cost_model.calculate(duration_hours, cluster_config.worker_count)
        else:
            # Static path: the regional multiplier scales the baseline
            # (us-east-1) DPU rate
            pricing_source = "static"
            dpu_rate_per_hour = self.dpu_cost_per_hour
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
                "dpu_hours": dpu_hours,
                "dpu_rate_per_hour": dpu_rate_per_hour,
                "worker_count": cluster_config.worker_count,
                "worker_type": worker.name,
                "dpu_per_worker": dpu_per_worker,
                "region": self.region,
                "region_multiplier": region_multiplier,
            },
            "notes": (
                "Cost estimate based on DPU-hour pricing with a live regional rate (AWS Pricing API)"
                if pricing_source == "live"
                else "Cost estimate based on DPU-hour pricing with a curated regional multiplier"
            ),
        }

    def get_glue_job_properties(
        self,
        cluster_config: ClusterConfig,
        job_name: str = "spark-optima-job",
    ) -> dict[str, Any]:
        """Generate AWS Glue job properties for SDK/CLI usage.

        Args:
            cluster_config: Cluster configuration.
            job_name: Name for the Glue job.

        Returns:
            Dictionary with Glue job properties.

        """
        return {
            "Name": job_name,
            "Role": "AWSGlueServiceRole",  # User should replace
            "GlueVersion": cluster_config.platform_config.get("glue_version", "4.0"),
            "WorkerType": cluster_config.worker_type.name,
            "NumberOfWorkers": cluster_config.worker_count,
            "ExecutionProperty": {
                "MaxConcurrentRuns": 1,
            },
            "DefaultArguments": {
                "--enable-metrics": "true",
                "--enable-spark-ui": "true",
                "--spark-event-logs-path": "s3://bucket/spark-logs/",
            },
        }

    # --- boto3/Real Execution Methods ---

    def submit_job(
        self,
        code_path: str | Path,
        job_name: str = "spark-optima-job",
        cluster_config: ClusterConfig | None = None,
        timeout_minutes: int = 60,
    ) -> dict[str, Any]:
        """Submit a Glue job via boto3.

        Args:
            code_path: Path to Python file containing Spark code.
            job_name: Name for the Glue job.
            cluster_config: Cluster configuration.
            timeout_minutes: Maximum job duration.

        Returns:
            Dictionary with job submission result.

        Raises:
            RuntimeError: If boto3 is not available or credentials not configured.

        """
        try:
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError
        except ImportError as e:
            raise RuntimeError(
                "boto3 is required for AWS Glue execution. Install with: pip install boto3",
            ) from e

        # Get AWS credentials from environment
        import os

        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

        try:
            # Initialize Glue client
            client = boto3.client("glue", region_name=region)

            # Create or update job
            worker_type = "G.1X"
            num_workers = 2
            if cluster_config and cluster_config.worker_type:
                worker_type = cluster_config.worker_type.name
                num_workers = cluster_config.worker_count

            job_args = {
                "--spark-event-logs-path": "s3://bucket/spark-logs/",
                "--enable-metrics": "true",
            }

            try:
                # Try to update existing job
                client.update_job(
                    JobName=job_name,
                    JobUpdate={
                        "Role": os.environ.get(
                            "AWS_GLUE_SERVICE_ROLE",
                            "AWSGlueServiceRole",
                        ),
                        "Command": {
                            "Name": "glueetl",
                            "ScriptLocation": code_path,  # Should be S3 path in prod
                        },
                        "WorkerType": worker_type,
                        "NumberOfWorkers": num_workers,
                        "Timeout": timeout_minutes,
                        "GlueVersion": "4.0",
                        "DefaultArguments": job_args,
                    },
                )
            except (ClientError, KeyError):
                # Create new job
                client.create_job(
                    Name=job_name,
                    Role=os.environ.get(
                        "AWS_GLUE_SERVICE_ROLE",
                        "AWSGlueServiceRole",
                    ),
                    Command={
                        "Name": "glueetl",
                        "ScriptLocation": code_path,  # Should be S3 path in prod
                    },
                    WorkerType=worker_type,
                    NumberOfWorkers=num_workers,
                    Timeout=timeout_minutes,
                    GlueVersion="4.0",
                    DefaultArguments=job_args,
                )

            # Start job run
            response = client.start_job_run(JobName=job_name)
            job_run_id = response["JobRunId"]

            return {
                "success": True,
                "job_name": job_name,
                "job_run_id": job_run_id,
                "status": "submitted",
            }

        except NoCredentialsError as e:
            logger.error(f"AWS credentials not found: {e}")
            return {
                "success": False,
                "error": "AWS credentials not configured",
            }
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_msg = e.response["Error"]["Message"]
            logger.error(f"AWS Glue error ({error_code}): {error_msg}")
            return {
                "success": False,
                "error": f"{error_code}: {error_msg}",
            }
        except (RuntimeError, ValueError, KeyError) as e:
            logger.error(f"Job submission failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def get_job_status(
        self,
        job_name: str,
        job_run_id: str,
    ) -> dict[str, Any]:
        """Get status of a Glue job run.

        Args:
            job_name: Glue job name.
            job_run_id: Job run ID.

        Returns:
            Dictionary with job status.

        """
        try:
            import boto3
        except ImportError as e:
            raise RuntimeError(
                "boto3 is required for AWS Glue execution",
            ) from e

        try:
            import os

            region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
            client = boto3.client("glue", region_name=region)

            response = client.get_job_run(
                JobName=job_name,
                RunId=job_run_id,
            )

            job_run = response["JobRun"]
            state = job_run.get("JobRunState", "UNKNOWN")

            return {
                "job_name": job_name,
                "job_run_id": job_run_id,
                "state": state,
                "status": self._map_glue_state(state),
                "started_on": str(job_run.get("StartedOn", "")),
                "completed_on": str(job_run.get("CompletedOn", "")),
                "error_message": job_run.get("ErrorMessage", ""),
            }

        except (RuntimeError, KeyError, ValueError) as e:
            logger.error(f"Error getting job status: {e}")
            return {
                "job_name": job_name,
                "job_run_id": job_run_id,
                "status": "error",
                "error": str(e),
            }

    def get_job_results(
        self,
        job_name: str,
        job_run_id: str,
    ) -> dict[str, Any]:
        """Get results of a completed Glue job.

        Args:
            job_name: Glue job name.
            job_run_id: Job run ID.

        Returns:
            Dictionary with job results.

        """
        try:
            import boto3
        except ImportError as e:
            raise RuntimeError(
                "boto3 is required for AWS Glue execution",
            ) from e

        try:
            import os

            region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
            client = boto3.client("glue", region_name=region)

            response = client.get_job_run(
                JobName=job_name,
                RunId=job_run_id,
            )

            job_run = response["JobRun"]
            state = job_run.get("JobRunState", "UNKNOWN")

            return {
                "job_name": job_name,
                "job_run_id": job_run_id,
                "state": state,
                "status": self._map_glue_state(state),
                "output": job_run.get("Output", {}),
                "error_message": job_run.get("ErrorMessage", ""),
                "execution_time": job_run.get("ExecutionTime", 0.0),
            }

        except (RuntimeError, KeyError, ValueError) as e:
            logger.error(f"Error getting job results: {e}")
            return {
                "job_name": job_name,
                "job_run_id": job_run_id,
                "status": "error",
                "error": str(e),
            }

    def _map_glue_state(self, state: str) -> str:
        """Map AWS Glue job state to simple status.

        Args:
            state: Glue job run state.

        Returns:
            Simplified status string.

        """
        state_map = {
            "STARTING": "starting",
            "RUNNING": "running",
            "STOPPING": "stopping",
            "STOPPED": "stopped",
            "SUCCEEDED": "completed",
            "FAILED": "failed",
            "TIMEOUT": "timeout",
        }
        return state_map.get(state, "unknown")
