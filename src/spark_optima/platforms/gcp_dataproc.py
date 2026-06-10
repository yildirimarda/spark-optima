# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Google Cloud Dataproc platform implementation for Spark Optima.

This module provides the GCPDataprocPlatform class for Google Cloud Dataproc
clusters, including representative N2 machine types across the general purpose
(n2-standard) and memory-optimized (n2-highmem) families, YARN-oriented Spark
configuration translation, and a Compute Engine + Dataproc fee cost model.

Dataproc image version to Spark version mapping used by this adapter:

| Dataproc image | Spark version |
|----------------|---------------|
| 2.0            | 3.1.3         |
| 2.1            | 3.3.2         |
| 2.2            | 3.5.3         |
"""

from __future__ import annotations

import logging
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


class GCPDataprocPlatform(Platform):
    """Google Cloud Dataproc platform for Spark workloads.

    Dataproc runs Spark on YARN on top of Compute Engine instances. This
    adapter models:

    - A representative set of N2 machine types: n2-standard (general purpose)
      and n2-highmem (memory-optimized) families, 4 to 64 vCPUs.
    - Pricing as on-demand us-central1 Compute Engine hourly price plus the
      Dataproc fee ($0.01 per vCPU per hour).
    - One master node in addition to the (primary) worker nodes.
    - Optional preemptible (Spot) workers as a flat compute discount
      (see :meth:`estimate_cost` for the approximation).

    Note:
        Dataproc requires at least 2 primary workers in a standard cluster,
        which is reflected in ``constraints.min_workers``.

    Attributes:
        name: Platform identifier "gcp_dataproc".
        display_name: Human-readable name "GCP Dataproc".

    Example:
        >>> platform = GCPDataprocPlatform()
        >>> worker = platform.get_worker_type("n2-standard-8")
        >>> print(f"n2-standard-8: {worker.resources.cpu_cores} vCPUs, {worker.resources.memory_gb}GB RAM")

    """

    # N2 machine specifications with approximate on-demand us-central1 hourly
    # Compute Engine prices (vCPU + RAM components).
    # Source: https://cloud.google.com/compute/vm-instance-pricing and
    # https://cloud.google.com/dataproc/pricing (prices vary by region and over time).
    WORKER_SPECS: dict[str, dict[str, Any]] = {
        # General purpose (n2-standard)
        "n2-standard-4": {
            "compute_price": 0.1942,
            "cpu_cores": 4,
            "memory_gb": 16,
            "disk_gb": 128,
            "size": InstanceSize.SMALL,
        },
        "n2-standard-8": {
            "compute_price": 0.3885,
            "cpu_cores": 8,
            "memory_gb": 32,
            "disk_gb": 256,
            "size": InstanceSize.MEDIUM,
        },
        "n2-standard-16": {
            "compute_price": 0.7769,
            "cpu_cores": 16,
            "memory_gb": 64,
            "disk_gb": 512,
            "size": InstanceSize.LARGE,
        },
        "n2-standard-32": {
            "compute_price": 1.5539,
            "cpu_cores": 32,
            "memory_gb": 128,
            "disk_gb": 1024,
            "size": InstanceSize.XLARGE,
        },
        "n2-standard-64": {
            "compute_price": 3.1078,
            "cpu_cores": 64,
            "memory_gb": 256,
            "disk_gb": 2048,
            "size": InstanceSize.XXLARGE,
        },
        # Memory-optimized (n2-highmem)
        "n2-highmem-4": {
            "compute_price": 0.2620,
            "cpu_cores": 4,
            "memory_gb": 32,
            "disk_gb": 128,
            "size": InstanceSize.SMALL,
        },
        "n2-highmem-8": {
            "compute_price": 0.5241,
            "cpu_cores": 8,
            "memory_gb": 64,
            "disk_gb": 256,
            "size": InstanceSize.MEDIUM,
        },
        "n2-highmem-16": {
            "compute_price": 1.0481,
            "cpu_cores": 16,
            "memory_gb": 128,
            "disk_gb": 512,
            "size": InstanceSize.LARGE,
        },
        "n2-highmem-32": {
            "compute_price": 2.0962,
            "cpu_cores": 32,
            "memory_gb": 256,
            "disk_gb": 1024,
            "size": InstanceSize.XLARGE,
        },
        "n2-highmem-64": {
            "compute_price": 4.1924,
            "cpu_cores": 64,
            "memory_gb": 512,
            "disk_gb": 2048,
            "size": InstanceSize.XXLARGE,
        },
    }

    # Dataproc fee charged on top of Compute Engine: $0.01 per vCPU per hour
    DATAPROC_FEE_PER_VCPU_HOUR: float = 0.01

    # Approximate preemptible (Spot) discount on the Compute Engine portion.
    # Actual Spot discounts vary (~60-91%); ~65% is a conservative midpoint.
    PREEMPTIBLE_DISCOUNT: float = 0.65

    # Dataproc image version to Spark version mapping
    IMAGE_TO_SPARK_VERSION: dict[str, str] = {
        "2.0": "3.1.3",
        "2.1": "3.3.2",
        "2.2": "3.5.3",
    }

    # Default image version used when no mapping matches
    DEFAULT_IMAGE_VERSION: str = "2.2"

    # Default machine type for the master node
    DEFAULT_MASTER_MACHINE_TYPE: str = "n2-standard-4"

    def __init__(
        self,
        region: str = "us-central1",
        use_preemptible_workers: bool = False,
    ) -> None:
        """Initialize the GCP Dataproc platform.

        Args:
            region: GCP region for pricing.
            use_preemptible_workers: Model workers as preemptible (Spot)
                instances. This applies an approximate ~65% discount to the
                worker Compute Engine cost in :meth:`estimate_cost`. The
                master node always stays on-demand. Note that in a real
                cluster Dataproc requires at least 2 non-preemptible primary
                workers (preemptible instances go into the secondary worker
                group), so actual savings will be somewhat lower than this
                approximation.

        """
        super().__init__(
            name="gcp_dataproc",
            display_name="GCP Dataproc",
            description="Google Cloud Dataproc managed Spark clusters on Compute Engine",
        )
        self.region = region
        self.use_preemptible_workers = use_preemptible_workers

        self._constraints = PlatformConstraints(
            min_workers=2,  # Dataproc requires at least 2 primary workers
            max_workers=1000,
            min_memory_gb=16,  # n2-standard-4
            max_memory_gb=512,  # n2-highmem-64
            min_cores=4,  # All -4 machine types
            max_cores=64,  # n2-standard-64 / n2-highmem-64
            supported_spark_versions=[
                "3.1.0",
                "3.1.3",  # Dataproc 2.0
                "3.3.0",
                "3.3.2",  # Dataproc 2.1
                "3.5.0",
                "3.5.3",  # Dataproc 2.2
            ],
            custom_config_keys={
                "image_version": "SoftwareConfig.imageVersion",
                "machine_type": "WorkerConfig.machineTypeUri",
                "num_workers": "WorkerConfig.numInstances",
                "master_machine_type": "MasterConfig.machineTypeUri",
            },
        )

    @property
    def constraints(self) -> PlatformConstraints:
        """Get platform resource constraints."""
        return self._constraints

    @classmethod
    def match_image_version(cls, spark_version: str) -> str:
        """Map a Spark version to a Dataproc image version.

        Tries an exact match first, then a major.minor match, then falls
        back to the latest image version.

        Args:
            spark_version: Spark version string (e.g., "3.5.0").

        Returns:
            Dataproc image version string (e.g., "2.2").

        """
        # Exact match first
        for image, spark_ver in cls.IMAGE_TO_SPARK_VERSION.items():
            if spark_version == spark_ver:
                return image

        # Fall back to major.minor match
        major_minor = ".".join(spark_version.split(".")[:2])
        for image, spark_ver in cls.IMAGE_TO_SPARK_VERSION.items():
            if spark_ver.startswith(major_minor):
                return image

        # Default to latest
        return cls.DEFAULT_IMAGE_VERSION

    def _get_image_version(self, spark_version: str) -> str:
        """Map Spark version to a Dataproc image version.

        Args:
            spark_version: Spark version string.

        Returns:
            Dataproc image version string (e.g., "2.2").

        """
        return self.match_image_version(spark_version)

    def _create_worker_type(self, name: str, spec: dict[str, Any]) -> WorkerType:
        """Create a WorkerType from specification.

        Args:
            name: Machine type name.
            spec: Machine specifications dict.

        Returns:
            Configured WorkerType instance.

        """
        compute_price = spec["compute_price"]
        dataproc_fee = self.DATAPROC_FEE_PER_VCPU_HOUR * spec["cpu_cores"]
        hourly_cost = compute_price + dataproc_fee

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
                granularity_minutes=1,  # Dataproc bills per second with a 1-minute minimum
            ),
            description=f"GCE {name} ({spec['cpu_cores']} vCPUs, {spec['memory_gb']}GB RAM, "
            f"${compute_price}/h compute + ${dataproc_fee:.2f}/h Dataproc fee)",
        )

    def get_worker_types(self) -> list[WorkerType]:
        """Get all available Dataproc worker machine types.

        Returns:
            List of WorkerType for all supported N2 machine types.

        """
        return [self._create_worker_type(name, spec) for name, spec in self.WORKER_SPECS.items()]

    def get_worker_type(self, name: str) -> WorkerType | None:
        """Get a specific worker type by machine type name.

        Args:
            name: Machine type (e.g., "n2-standard-8", "n2-highmem-16").

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
    ) -> WorkerType:
        """Recommend the best machine type for given requirements.

        Args:
            target_memory_gb: Desired memory per worker.
            target_cores: Desired CPU cores per worker.
            prefer_memory_optimized: Prefer n2-highmem machines for
                memory-intensive workloads.

        Returns:
            Recommended WorkerType.

        """
        candidates = self.get_worker_types()

        # Filter by machine family preference (general purpose n2-standard by default)
        family = "n2-highmem-" if prefer_memory_optimized else "n2-standard-"
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

        # Fallback to n2-standard-4 if no match found
        if best_match is None:
            return self.get_worker_type("n2-standard-4") or self.get_worker_types()[0]

        return best_match

    def recommend_config(
        self,
        resources: ResourceSpec,
        spark_version: str,
        worker_count: int | None = None,
    ) -> ClusterConfig:
        """Recommend optimal Dataproc cluster configuration.

        The recommendation includes one master node (driver) in addition to
        the primary worker nodes. Dataproc requires at least 2 primary
        workers, which is enforced here.

        Args:
            resources: Target total resources.
            spark_version: Target Spark version (maps to a Dataproc image version).
            worker_count: Optional specific worker node count.

        Returns:
            Recommended ClusterConfig.

        """
        # Recommend machine type based on resource requirements
        memory_per_worker = min(resources.memory_gb / 10, 64)  # Target ~10 workers
        cores_per_worker = min(resources.cpu_cores / 10, 8)

        worker_type = self.recommend_worker_type(
            target_memory_gb=memory_per_worker,
            target_cores=int(cores_per_worker),
        )

        # Calculate worker count
        if worker_count is None:
            worker_count = self.get_optimal_worker_count(resources, worker_type)

        # Dataproc requires at least 2 primary workers
        worker_count = max(worker_count, self._constraints.min_workers)

        # Map Spark version to a Dataproc image version
        image_version = self._get_image_version(spark_version)

        # One master node is always provisioned in addition to the workers
        master_type = self.get_worker_type(self.DEFAULT_MASTER_MACHINE_TYPE) or worker_type

        return ClusterConfig(
            worker_type=worker_type,
            worker_count=worker_count,
            driver_type=master_type,  # Dataproc master node
            driver_count=1,
            spark_version=spark_version,
            platform_config={
                "image_version": image_version,
                "machine_type": worker_type.name,
                "num_workers": worker_count,
                "master_machine_type": master_type.name,
            },
        )

    def translate_to_spark_config(
        self,
        cluster_config: ClusterConfig,
    ) -> dict[str, Any]:
        """Translate cluster config to Spark configuration for Dataproc (YARN).

        Dataproc runs Spark on YARN, so ``spark.master`` is not set here (the
        Dataproc runtime configures it). One vCPU per node is reserved for
        YARN/OS daemons and ~10% of node memory is left as headroom.

        Args:
            cluster_config: Dataproc cluster configuration.

        Returns:
            Dictionary of Spark configuration parameters.

        """
        worker = cluster_config.worker_type
        r = worker.resources

        # One executor per worker node; leave one core for YARN/OS daemons
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

        # Add Dataproc-specific metadata
        if cluster_config.platform_config and "image_version" in cluster_config.platform_config:
            config["dataproc.imageVersion"] = cluster_config.platform_config["image_version"]

        return config

    def estimate_cost(
        self,
        cluster_config: ClusterConfig,
        duration_hours: float,
    ) -> dict[str, Any]:
        """Estimate cost for a Dataproc cluster.

        Total cost is (master + workers) x (Compute Engine price + Dataproc
        fee) x hours. The Dataproc fee is $0.01 per vCPU per hour.

        When ``use_preemptible_workers`` is enabled, an approximate ~65%
        discount is applied to the worker Compute Engine portion (the
        Dataproc fee is charged on all vCPUs regardless, and the master node
        stays on-demand). This is an approximation: real clusters keep at
        least 2 non-preemptible primary workers, so actual savings will be
        somewhat lower.

        Args:
            cluster_config: Cluster configuration.
            duration_hours: Expected runtime in hours.

        Returns:
            Cost breakdown including Compute Engine and Dataproc fee portions.

        """
        worker = cluster_config.worker_type
        master = cluster_config.driver_type or self.get_worker_type(self.DEFAULT_MASTER_MACHINE_TYPE) or worker

        def split_hourly(worker_type: WorkerType) -> tuple[float, float]:
            """Split a node's hourly cost into compute and Dataproc fee portions."""
            fee = self.DATAPROC_FEE_PER_VCPU_HOUR * worker_type.resources.cpu_cores
            compute = max(0.0, worker_type.cost.unit_cost_per_hour - fee)
            return compute, fee

        worker_compute_hourly, worker_fee_hourly = split_hourly(worker)
        master_compute_hourly, master_fee_hourly = split_hourly(master)

        # Regional pricing applies to the Compute Engine portion only — the
        # Dataproc fee is $0.01/vCPU-hour in every region.
        region_multiplier = get_region_multiplier(self.name, self.region)
        worker_compute_hourly *= region_multiplier
        master_compute_hourly *= region_multiplier

        # Approximate preemptible (Spot) discount on the worker compute portion
        if self.use_preemptible_workers:
            worker_compute_hourly *= 1.0 - self.PREEMPTIBLE_DISCOUNT

        billed_hours = max(duration_hours, 0.0)
        worker_units = cluster_config.worker_count * billed_hours
        master_units = cluster_config.driver_count * billed_hours

        worker_cost = (worker_compute_hourly + worker_fee_hourly) * worker_units
        master_cost = (master_compute_hourly + master_fee_hourly) * master_units
        compute_cost = worker_compute_hourly * worker_units + master_compute_hourly * master_units
        dataproc_fee = worker_fee_hourly * worker_units + master_fee_hourly * master_units
        total_cost = worker_cost + master_cost

        return {
            "platform": self.name,
            "currency": "USD",
            "region": self.region,
            "duration_hours": duration_hours,
            "total_cost": total_cost,
            # Always static: GCP's Cloud Billing Catalog API requires an API
            # key, so anonymous live pricing is not possible — deferred to
            # the v1.5+ backlog (see spark_optima.platforms.live_pricing)
            "pricing_source": "static",
            "breakdown": {
                "master_machine_type": master.name,
                "master_count": cluster_config.driver_count,
                "master_cost": master_cost,
                "worker_machine_type": worker.name,
                "worker_count": cluster_config.worker_count,
                "worker_cost": worker_cost,
                "compute_cost": compute_cost,
                "dataproc_fee": dataproc_fee,
                "dataproc_fee_per_vcpu_hour": self.DATAPROC_FEE_PER_VCPU_HOUR,
                "region": self.region,
                "region_multiplier": region_multiplier,
                "preemptible_workers": self.use_preemptible_workers,
                "preemptible_discount": (self.PREEMPTIBLE_DISCOUNT if self.use_preemptible_workers else 0.0),
            },
            "notes": (
                "Cost estimate based on on-demand us-central1 Compute Engine pricing "
                "plus the $0.01/vCPU-hour Dataproc fee"
                + (
                    "; preemptible worker discount is an approximation (~65% off worker compute)"
                    if self.use_preemptible_workers
                    else ""
                )
            ),
        }

    def get_dataproc_cluster_config(
        self,
        cluster_config: ClusterConfig,
        cluster_name: str = "spark-optima-cluster",
        spark_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate a Dataproc cluster definition for the ``clusters.create`` REST API.

        The optimized Spark configuration is carried in
        ``softwareConfig.properties`` with the ``spark:`` prefix, as expected
        by Dataproc cluster properties.

        Note:
            The ``use_preemptible_workers`` constructor flag only affects cost
            estimation. To actually use preemptible instances, configure a
            ``secondaryWorkerConfig`` block — Dataproc primary workers cannot
            be preemptible.

        Args:
            cluster_config: Cluster configuration.
            cluster_name: Name for the Dataproc cluster.
            spark_config: Optional Spark configuration to embed in the
                cluster properties (derived from the cluster config if None).

        Returns:
            Dictionary suitable as the request body for
            ``projects.regions.clusters.create``.

        """
        image_version = cluster_config.platform_config.get(
            "image_version",
            self.DEFAULT_IMAGE_VERSION,
        )
        master_machine_type = cluster_config.platform_config.get(
            "master_machine_type",
            self.DEFAULT_MASTER_MACHINE_TYPE,
        )

        if spark_config is None:
            spark_config = self.translate_to_spark_config(cluster_config)

        # Only spark.* keys belong in the cluster properties ("spark:" prefix)
        properties = {f"spark:{k}": str(v) for k, v in spark_config.items() if k.startswith("spark.")}

        return {
            "clusterName": cluster_name,
            "config": {
                "gceClusterConfig": {
                    "zoneUri": f"{self.region}-a",  # User should replace with the desired zone
                },
                "masterConfig": {
                    "numInstances": cluster_config.driver_count,
                    "machineTypeUri": master_machine_type,
                },
                "workerConfig": {
                    "numInstances": cluster_config.worker_count,
                    "machineTypeUri": cluster_config.worker_type.name,
                },
                "softwareConfig": {
                    "imageVersion": image_version,
                    "properties": properties,
                },
            },
        }
