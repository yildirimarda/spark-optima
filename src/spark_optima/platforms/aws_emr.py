# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""AWS EMR platform implementation for Spark Optima.

This module provides the AWSEMRPlatform class for Amazon EMR on EC2 clusters,
including representative EC2 instance types across the general purpose (m5),
memory-optimized (r5), and compute-optimized (c5) families, YARN-oriented
Spark configuration translation, and an EC2 + EMR surcharge cost model.

EMR release to Spark version mapping used by this adapter:

| EMR release  | Spark version |
|--------------|---------------|
| emr-6.9.0    | 3.3.0         |
| emr-6.15.0   | 3.4.1         |
| emr-7.0.0    | 3.5.0         |
| emr-7.5.0    | 3.5.2         |
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from spark_optima.platforms.base import Platform
from spark_optima.platforms.models import (
    ClusterConfig,
    CostModel,
    InstanceSize,
    PlatformConstraints,
    ResourceSpec,
    WorkerType,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class AWSEMRPlatform(Platform):
    """Amazon EMR on EC2 platform for Spark workloads.

    Amazon EMR runs Spark on YARN on top of EC2 instances. This adapter models:
    - A representative set of EC2 instance types: m5 (general purpose),
      r5 (memory-optimized), and c5 (compute-optimized) families.
    - Pricing as on-demand us-east-1 EC2 hourly price plus the EMR surcharge
      (modeled as ~25% of the EC2 price).
    - One master node in addition to the core (worker) nodes.

    Note:
        The c5 family has no 8xlarge size; c5.9xlarge (36 vCPUs) is the closest
        larger size and is used as the top compute-optimized option.

    Attributes:
        name: Platform identifier "aws_emr".
        display_name: Human-readable name "AWS EMR".

    Example:
        >>> platform = AWSEMRPlatform()
        >>> worker = platform.get_worker_type("m5.2xlarge")
        >>> print(f"m5.2xlarge: {worker.resources.cpu_cores} vCPUs, {worker.resources.memory_gb}GB RAM")

    """

    # EC2 instance specifications with approximate on-demand us-east-1 hourly prices.
    # Source: https://aws.amazon.com/ec2/pricing/on-demand/ and
    # https://aws.amazon.com/emr/pricing/ (prices vary by region and over time).
    WORKER_SPECS: dict[str, dict[str, Any]] = {
        # General purpose (m5)
        "m5.xlarge": {
            "ec2_price": 0.192,
            "cpu_cores": 4,
            "memory_gb": 16,
            "disk_gb": 64,
            "size": InstanceSize.SMALL,
        },
        "m5.2xlarge": {
            "ec2_price": 0.384,
            "cpu_cores": 8,
            "memory_gb": 32,
            "disk_gb": 128,
            "size": InstanceSize.MEDIUM,
        },
        "m5.4xlarge": {
            "ec2_price": 0.768,
            "cpu_cores": 16,
            "memory_gb": 64,
            "disk_gb": 256,
            "size": InstanceSize.LARGE,
        },
        "m5.8xlarge": {
            "ec2_price": 1.536,
            "cpu_cores": 32,
            "memory_gb": 128,
            "disk_gb": 512,
            "size": InstanceSize.XLARGE,
        },
        # Memory-optimized (r5)
        "r5.xlarge": {
            "ec2_price": 0.252,
            "cpu_cores": 4,
            "memory_gb": 32,
            "disk_gb": 64,
            "size": InstanceSize.SMALL,
        },
        "r5.2xlarge": {
            "ec2_price": 0.504,
            "cpu_cores": 8,
            "memory_gb": 64,
            "disk_gb": 128,
            "size": InstanceSize.MEDIUM,
        },
        "r5.4xlarge": {
            "ec2_price": 1.008,
            "cpu_cores": 16,
            "memory_gb": 128,
            "disk_gb": 256,
            "size": InstanceSize.LARGE,
        },
        "r5.8xlarge": {
            "ec2_price": 2.016,
            "cpu_cores": 32,
            "memory_gb": 256,
            "disk_gb": 512,
            "size": InstanceSize.XLARGE,
        },
        # Compute-optimized (c5) — no c5.8xlarge exists; c5.9xlarge is the closest larger size
        "c5.xlarge": {
            "ec2_price": 0.17,
            "cpu_cores": 4,
            "memory_gb": 8,
            "disk_gb": 64,
            "size": InstanceSize.SMALL,
        },
        "c5.2xlarge": {
            "ec2_price": 0.34,
            "cpu_cores": 8,
            "memory_gb": 16,
            "disk_gb": 128,
            "size": InstanceSize.MEDIUM,
        },
        "c5.4xlarge": {
            "ec2_price": 0.68,
            "cpu_cores": 16,
            "memory_gb": 32,
            "disk_gb": 256,
            "size": InstanceSize.LARGE,
        },
        "c5.9xlarge": {
            "ec2_price": 1.53,
            "cpu_cores": 36,
            "memory_gb": 72,
            "disk_gb": 512,
            "size": InstanceSize.XLARGE,
        },
    }

    # EMR surcharge modeled as a fraction of the EC2 on-demand price (~25%)
    EMR_SURCHARGE_RATE: float = 0.25

    # EMR release label to Spark version mapping
    EMR_TO_SPARK_VERSION: dict[str, str] = {
        "emr-6.9.0": "3.3.0",
        "emr-6.15.0": "3.4.1",
        "emr-7.0.0": "3.5.0",
        "emr-7.5.0": "3.5.2",
    }

    # Default release used when no mapping matches
    DEFAULT_RELEASE_LABEL: str = "emr-7.5.0"

    # Default instance type for the master node
    DEFAULT_MASTER_INSTANCE_TYPE: str = "m5.xlarge"

    def __init__(
        self,
        region: str = "us-east-1",
        emr_surcharge_rate: float | None = None,
    ) -> None:
        """Initialize the AWS EMR platform.

        Args:
            region: AWS region for pricing.
            emr_surcharge_rate: EMR surcharge as a fraction of the EC2 price
                (uses the default ~25% if None).

        """
        super().__init__(
            name="aws_emr",
            display_name="AWS EMR",
            description="Amazon EMR managed Spark clusters on EC2",
        )
        self.region = region
        self.emr_surcharge_rate = emr_surcharge_rate if emr_surcharge_rate is not None else self.EMR_SURCHARGE_RATE

        self._constraints = PlatformConstraints(
            min_workers=1,  # EMR supports single-core-node clusters
            max_workers=500,  # Practical cap for instance groups
            min_memory_gb=8,  # c5.xlarge
            max_memory_gb=256,  # r5.8xlarge
            min_cores=4,  # All xlarge instances
            max_cores=36,  # c5.9xlarge
            supported_spark_versions=[
                "3.3.0",  # emr-6.9.0
                "3.4.1",  # emr-6.15.0
                "3.5.0",  # emr-7.0.0
                "3.5.2",  # emr-7.5.0
            ],
            custom_config_keys={
                "release_label": "ReleaseLabel",
                "instance_type": "InstanceType",
                "instance_count": "InstanceCount",
                "master_instance_type": "MasterInstanceType",
            },
        )

    @property
    def constraints(self) -> PlatformConstraints:
        """Get platform resource constraints."""
        return self._constraints

    def _create_worker_type(self, name: str, spec: dict[str, Any]) -> WorkerType:
        """Create a WorkerType from specification.

        Args:
            name: EC2 instance type name.
            spec: Instance specifications dict.

        Returns:
            Configured WorkerType instance.

        """
        ec2_price = spec["ec2_price"]
        hourly_cost = ec2_price * (1.0 + self.emr_surcharge_rate)

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
                unit_cost_per_hour=hourly_cost,
                unit_name="instance",
                granularity_minutes=1,  # EMR bills per second with a 1-minute minimum
            ),
            description=f"EC2 {name} ({spec['cpu_cores']} vCPUs, {spec['memory_gb']}GB RAM, "
            f"${ec2_price}/h EC2 + ~{int(self.emr_surcharge_rate * 100)}% EMR fee)",
        )

    def get_worker_types(self) -> list[WorkerType]:
        """Get all available EMR worker instance types.

        Returns:
            List of WorkerType for all supported EC2 instance types.

        """
        return [self._create_worker_type(name, spec) for name, spec in self.WORKER_SPECS.items()]

    def get_worker_type(self, name: str) -> WorkerType | None:
        """Get a specific worker type by EC2 instance type name.

        Args:
            name: EC2 instance type (e.g., "m5.xlarge", "r5.2xlarge").

        Returns:
            WorkerType if found, None otherwise.

        """
        spec = self.WORKER_SPECS.get(name.lower())
        if spec:
            return self._create_worker_type(name.lower(), spec)
        return None

    def recommend_worker_type(
        self,
        target_memory_gb: float,
        target_cores: int,
        prefer_memory_optimized: bool = False,
        prefer_compute_optimized: bool = False,
    ) -> WorkerType:
        """Recommend the best EC2 instance type for given requirements.

        Args:
            target_memory_gb: Desired memory per worker.
            target_cores: Desired CPU cores per worker.
            prefer_memory_optimized: Prefer r5 instances for memory-intensive workloads.
            prefer_compute_optimized: Prefer c5 instances for CPU-bound workloads.

        Returns:
            Recommended WorkerType.

        """
        candidates = self.get_worker_types()

        # Filter by instance family preference (general purpose m5 by default)
        if prefer_memory_optimized:
            family = "r5."
        elif prefer_compute_optimized:
            family = "c5."
        else:
            family = "m5."

        family_types = [w for w in candidates if w.name.startswith(family)]
        if family_types:
            candidates = family_types

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

        # Fallback to m5.xlarge if no match found
        if best_match is None:
            return self.get_worker_type("m5.xlarge") or self.get_worker_types()[0]

        return best_match

    def recommend_config(
        self,
        resources: ResourceSpec,
        spark_version: str,
        worker_count: int | None = None,
    ) -> ClusterConfig:
        """Recommend optimal EMR cluster configuration.

        The recommendation includes one master node (driver) in addition to
        the core (worker) nodes.

        Args:
            resources: Target total resources.
            spark_version: Target Spark version (maps to an EMR release label).
            worker_count: Optional specific core node count.

        Returns:
            Recommended ClusterConfig.

        """
        # Recommend instance type based on resource requirements
        memory_per_worker = min(resources.memory_gb / 10, 64)  # Target ~10 workers
        cores_per_worker = min(resources.cpu_cores / 10, 8)

        worker_type = self.recommend_worker_type(
            target_memory_gb=memory_per_worker,
            target_cores=int(cores_per_worker),
        )

        # Calculate worker count
        if worker_count is None:
            worker_count = self.get_optimal_worker_count(resources, worker_type)

        # Ensure minimum workers for EMR
        worker_count = max(worker_count, self._constraints.min_workers)

        # Map Spark version to EMR release label
        release_label = self._get_release_label(spark_version)

        # One master node is always provisioned in addition to core nodes
        master_type = self.get_worker_type(self.DEFAULT_MASTER_INSTANCE_TYPE) or worker_type

        return ClusterConfig(
            worker_type=worker_type,
            worker_count=worker_count,
            driver_type=master_type,  # EMR master node
            driver_count=1,
            spark_version=spark_version,
            platform_config={
                "release_label": release_label,
                "instance_type": worker_type.name,
                "instance_count": worker_count,
                "master_instance_type": master_type.name,
            },
        )

    def _get_release_label(self, spark_version: str) -> str:
        """Map Spark version to an EMR release label.

        Args:
            spark_version: Spark version string.

        Returns:
            EMR release label string (e.g., "emr-7.0.0").

        """
        # Exact match first
        for release, spark_ver in self.EMR_TO_SPARK_VERSION.items():
            if spark_version == spark_ver:
                return release

        # Fall back to major.minor match
        major_minor = ".".join(spark_version.split(".")[:2])
        for release, spark_ver in self.EMR_TO_SPARK_VERSION.items():
            if spark_ver.startswith(major_minor):
                return release

        # Default to latest
        return self.DEFAULT_RELEASE_LABEL

    def translate_to_spark_config(
        self,
        cluster_config: ClusterConfig,
    ) -> dict[str, Any]:
        """Translate cluster config to Spark configuration for EMR (YARN).

        EMR runs Spark on YARN, so ``spark.master`` is not set here (the EMR
        runtime configures it). One vCPU per node is reserved for YARN/OS
        daemons and ~10% of node memory is left as headroom.

        Args:
            cluster_config: EMR cluster configuration.

        Returns:
            Dictionary of Spark configuration parameters.

        """
        worker = cluster_config.worker_type
        r = worker.resources

        # One executor per core node; leave one core for YARN/OS daemons
        executor_cores = max(1, r.cpu_cores - 1)
        executor_memory = max(1, int(r.memory_gb * 0.9))  # ~10% headroom for YARN overhead

        config = {
            # Executor configuration
            "spark.executor.instances": str(cluster_config.worker_count),
            "spark.executor.cores": str(executor_cores),
            "spark.executor.memory": f"{executor_memory}g",
            # Dynamic allocation (requires the external shuffle service on YARN)
            "spark.dynamicAllocation.enabled": "true",
            "spark.dynamicAllocation.minExecutors": str(self._constraints.min_workers),
            "spark.dynamicAllocation.maxExecutors": str(cluster_config.worker_count),
            "spark.shuffle.service.enabled": "true",
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

        # Add EMR-specific metadata
        if cluster_config.platform_config and "release_label" in cluster_config.platform_config:
            config["emr.releaseLabel"] = cluster_config.platform_config["release_label"]

        return config

    def estimate_cost(
        self,
        cluster_config: ClusterConfig,
        duration_hours: float,
    ) -> dict[str, Any]:
        """Estimate cost for an EMR cluster.

        Total cost is (master + workers) x (EC2 price + EMR surcharge) x hours.

        Args:
            cluster_config: Cluster configuration.
            duration_hours: Expected runtime in hours.

        Returns:
            Cost breakdown including EC2 and EMR surcharge portions.

        """
        worker = cluster_config.worker_type
        master = cluster_config.driver_type or self.get_worker_type(self.DEFAULT_MASTER_INSTANCE_TYPE) or worker

        # Per-node costs already include the EMR surcharge
        worker_cost = worker.cost.calculate(duration_hours, cluster_config.worker_count)
        master_cost = master.cost.calculate(duration_hours, cluster_config.driver_count)
        total_cost = worker_cost + master_cost

        # Split the total into EC2 and EMR surcharge portions
        ec2_cost = total_cost / (1.0 + self.emr_surcharge_rate)
        emr_surcharge = total_cost - ec2_cost

        return {
            "platform": self.name,
            "currency": "USD",
            "region": self.region,
            "duration_hours": duration_hours,
            "total_cost": total_cost,
            "breakdown": {
                "master_instance_type": master.name,
                "master_count": cluster_config.driver_count,
                "master_cost": master_cost,
                "worker_instance_type": worker.name,
                "worker_count": cluster_config.worker_count,
                "worker_cost": worker_cost,
                "ec2_cost": ec2_cost,
                "emr_surcharge": emr_surcharge,
                "emr_surcharge_rate": self.emr_surcharge_rate,
            },
            "notes": "Cost estimate based on on-demand EC2 pricing plus the EMR surcharge",
        }

    def get_emr_cluster_config(
        self,
        cluster_config: ClusterConfig,
        cluster_name: str = "spark-optima-cluster",
        spark_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate an EMR cluster definition for boto3 ``run_job_flow``.

        Args:
            cluster_config: Cluster configuration.
            cluster_name: Name for the EMR cluster.
            spark_config: Optional Spark configuration to embed in the
                spark-defaults classification (derived from the cluster
                config if None).

        Returns:
            Dictionary suitable for ``boto3.client("emr").run_job_flow(**config)``.

        """
        release_label = cluster_config.platform_config.get(
            "release_label",
            self.DEFAULT_RELEASE_LABEL,
        )
        master_instance_type = cluster_config.platform_config.get(
            "master_instance_type",
            self.DEFAULT_MASTER_INSTANCE_TYPE,
        )

        if spark_config is None:
            spark_config = self.translate_to_spark_config(cluster_config)

        # Only spark.* keys belong in the spark-defaults classification
        spark_defaults = {k: str(v) for k, v in spark_config.items() if k.startswith("spark.")}

        return {
            "Name": cluster_name,
            "ReleaseLabel": release_label,
            "Applications": [{"Name": "Spark"}],
            "Instances": {
                "InstanceGroups": [
                    {
                        "Name": "Master",
                        "Market": "ON_DEMAND",
                        "InstanceRole": "MASTER",
                        "InstanceType": master_instance_type,
                        "InstanceCount": 1,
                    },
                    {
                        "Name": "Core",
                        "Market": "ON_DEMAND",
                        "InstanceRole": "CORE",
                        "InstanceType": cluster_config.worker_type.name,
                        "InstanceCount": cluster_config.worker_count,
                    },
                ],
                "KeepJobFlowAliveWhenNoSteps": False,
                "TerminationProtected": False,
            },
            "Configurations": [
                {
                    "Classification": "spark-defaults",
                    "Properties": spark_defaults,
                },
            ],
            "JobFlowRole": "EMR_EC2_DefaultRole",  # User should replace
            "ServiceRole": "EMR_DefaultRole",  # User should replace
            "VisibleToAllUsers": True,
        }

    # --- boto3/Real Execution Methods ---

    def submit_job(
        self,
        code_path: str | Path,
        cluster_name: str = "spark-optima-cluster",
        cluster_config: ClusterConfig | None = None,
    ) -> dict[str, Any]:
        """Submit a Spark job to a new EMR cluster via boto3.

        Creates a transient cluster (terminates when the step completes) and
        runs the given script with spark-submit.

        Args:
            code_path: Path to Python file containing Spark code
                (should be an S3 path in production).
            cluster_name: Name for the EMR cluster.
            cluster_config: Cluster configuration (a small default is used if None).

        Returns:
            Dictionary with job submission result.

        Raises:
            RuntimeError: If boto3 is not available.

        """
        try:
            import boto3  # type: ignore[import-not-found]
            from botocore.exceptions import (  # type: ignore[import-not-found]
                ClientError,
                NoCredentialsError,
            )
        except ImportError as e:
            raise RuntimeError(
                "boto3 is required for AWS EMR execution. Install with: pip install boto3",
            ) from e

        # Get AWS region from environment
        import os

        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

        if cluster_config is None:
            # Small default cluster: 2 x m5.xlarge core nodes
            cluster_config = self.recommend_config(
                resources=ResourceSpec(cpu_cores=8, memory_gb=32.0),
                spark_version="3.5.0",
                worker_count=2,
            )

        try:
            client = boto3.client("emr", region_name=region)

            job_flow = self.get_emr_cluster_config(cluster_config, cluster_name)
            job_flow["Steps"] = [
                {
                    "Name": "spark-optima-step",
                    "ActionOnFailure": "TERMINATE_CLUSTER",
                    "HadoopJarStep": {
                        "Jar": "command-runner.jar",
                        "Args": [
                            "spark-submit",
                            "--deploy-mode",
                            "cluster",
                            str(code_path),
                        ],
                    },
                },
            ]

            response = client.run_job_flow(**job_flow)
            cluster_id = response["JobFlowId"]

            # Fetch the step id of the submitted step
            step_id = ""
            steps = client.list_steps(ClusterId=cluster_id).get("Steps", [])
            if steps:
                step_id = steps[0].get("Id", "")

            return {
                "success": True,
                "cluster_name": cluster_name,
                "cluster_id": cluster_id,
                "step_id": step_id,
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
            logger.error(f"AWS EMR error ({error_code}): {error_msg}")
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
        cluster_id: str,
        step_id: str,
    ) -> dict[str, Any]:
        """Get status of an EMR step.

        Args:
            cluster_id: EMR cluster ID (job flow ID).
            step_id: Step ID returned by submit_job.

        Returns:
            Dictionary with job status.

        Raises:
            RuntimeError: If boto3 is not available.

        """
        try:
            import boto3  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "boto3 is required for AWS EMR execution",
            ) from e

        try:
            import os

            region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
            client = boto3.client("emr", region_name=region)

            response = client.describe_step(
                ClusterId=cluster_id,
                StepId=step_id,
            )

            step = response["Step"]
            status = step.get("Status", {})
            state = status.get("State", "UNKNOWN")
            timeline = status.get("Timeline", {})

            return {
                "cluster_id": cluster_id,
                "step_id": step_id,
                "state": state,
                "status": self._map_emr_state(state),
                "started_on": str(timeline.get("StartDateTime", "")),
                "completed_on": str(timeline.get("EndDateTime", "")),
                "error_message": status.get("FailureDetails", {}).get("Message", ""),
            }

        except (RuntimeError, KeyError, ValueError) as e:
            logger.error(f"Error getting job status: {e}")
            return {
                "cluster_id": cluster_id,
                "step_id": step_id,
                "status": "error",
                "error": str(e),
            }

    def _map_emr_state(self, state: str) -> str:
        """Map an EMR step state to a simple status.

        Args:
            state: EMR step state.

        Returns:
            Simplified status string.

        """
        state_map = {
            "PENDING": "starting",
            "CANCEL_PENDING": "stopping",
            "RUNNING": "running",
            "COMPLETED": "completed",
            "CANCELLED": "stopped",
            "FAILED": "failed",
            "INTERRUPTED": "failed",
        }
        return state_map.get(state, "unknown")
