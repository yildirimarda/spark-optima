# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Spark-on-Kubernetes platform implementation for Spark Optima.

This module provides the SparkOnK8sPlatform class for self-hosted Apache
Spark running natively on Kubernetes, including executor pod size presets,
``spark.kubernetes.*`` configuration translation, shuffle-tracking-based
dynamic allocation (Kubernetes has no external shuffle service), and a
SparkApplication custom resource export for the Kubeflow Spark Operator.

Because Kubernetes clusters are self-hosted, there is no built-in pricing:
the cost model defaults to zero and users can supply their own
``cost_per_vcpu_hour`` to approximate infrastructure cost.
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

logger = logging.getLogger(__name__)


class SparkOnK8sPlatform(Platform):
    """Spark-on-Kubernetes platform for self-hosted Spark workloads.

    Spark runs natively on Kubernetes: the driver and executors are pods.
    This adapter models:

    - Worker "types" as executor pod size presets: small (2 CPU / 8GB),
      medium (4 / 16), large (8 / 32), and xlarge (16 / 64).
    - Dynamic allocation via shuffle tracking
      (``spark.dynamicAllocation.shuffleTracking.enabled``) — Kubernetes has
      no external shuffle service, so shuffle tracking is required.
    - Cost as zero by default (self-hosted clusters); pricing is
      user-supplied via the ``cost_per_vcpu_hour`` constructor parameter.

    Note:
        ``spark.master`` is not set by this adapter — it depends on the
        cluster's API server URL (``k8s://https://<api-server>:<port>``) and
        must be supplied at submit time (or implied by the Spark Operator).

    Attributes:
        name: Platform identifier "kubernetes".
        display_name: Human-readable name "Spark on Kubernetes".

    Example:
        >>> platform = SparkOnK8sPlatform(namespace="data-jobs")
        >>> worker = platform.get_worker_type("medium")
        >>> print(f"medium: {worker.resources.cpu_cores} CPUs, {worker.resources.memory_gb}GB RAM")

    """

    # Executor pod size presets (CPU cores / memory GB / ephemeral storage GB)
    WORKER_SPECS: dict[str, dict[str, Any]] = {
        "small": {
            "cpu_cores": 2,
            "memory_gb": 8,
            "disk_gb": 20,
            "size": InstanceSize.SMALL,
        },
        "medium": {
            "cpu_cores": 4,
            "memory_gb": 16,
            "disk_gb": 40,
            "size": InstanceSize.MEDIUM,
        },
        "large": {
            "cpu_cores": 8,
            "memory_gb": 32,
            "disk_gb": 80,
            "size": InstanceSize.LARGE,
        },
        "xlarge": {
            "cpu_cores": 16,
            "memory_gb": 64,
            "disk_gb": 160,
            "size": InstanceSize.XLARGE,
        },
    }

    # Fraction of pod memory reserved for off-heap/overhead (10%)
    MEMORY_OVERHEAD_FACTOR: float = 0.1

    # Default pod preset for the driver
    DEFAULT_DRIVER_PRESET: str = "small"

    # Container image placeholder template (users should replace with their own image)
    CONTAINER_IMAGE_TEMPLATE: str = "apache/spark:{spark_version}"

    def __init__(
        self,
        namespace: str = "default",
        cost_per_vcpu_hour: float = 0.0,
    ) -> None:
        """Initialize the Spark-on-Kubernetes platform.

        Args:
            namespace: Kubernetes namespace for driver and executor pods.
            cost_per_vcpu_hour: User-supplied infrastructure cost per vCPU
                per hour used by :meth:`estimate_cost`. Defaults to 0.0
                because self-hosted clusters have no universal price — set
                this to your blended node $/vCPU-hour to get cost estimates.

        """
        super().__init__(
            name="kubernetes",
            display_name="Spark on Kubernetes",
            description="Self-hosted Apache Spark running natively on Kubernetes",
        )
        self.namespace = namespace
        self.cost_per_vcpu_hour = cost_per_vcpu_hour

        self._constraints = PlatformConstraints(
            min_workers=1,
            max_workers=10000,  # Bounded by cluster capacity, not by Spark
            min_memory_gb=8,  # small preset
            max_memory_gb=64,  # xlarge preset
            min_cores=2,  # small preset
            max_cores=16,  # xlarge preset
            supported_spark_versions=[
                "3.0.0",
                "3.1.0",
                "3.2.0",
                "3.3.0",
                "3.4.0",
                "3.5.0",
                "4.0.0",
                "4.1.0",
            ],
            custom_config_keys={
                "namespace": "spark.kubernetes.namespace",
                "container_image": "spark.kubernetes.container.image",
                "pod_preset": "PodPreset",
                "num_executors": "spark.executor.instances",
            },
        )

    @property
    def constraints(self) -> PlatformConstraints:
        """Get platform resource constraints."""
        return self._constraints

    def _container_image(self, spark_version: str) -> str:
        """Build the container image placeholder for a Spark version.

        Args:
            spark_version: Spark version string.

        Returns:
            Placeholder image reference (users should replace it with their
            own image carrying application code and dependencies).

        """
        return self.CONTAINER_IMAGE_TEMPLATE.format(spark_version=spark_version)

    def _create_worker_type(self, name: str, spec: dict[str, Any]) -> WorkerType:
        """Create a WorkerType from a pod size preset.

        Args:
            name: Pod preset name.
            spec: Pod specifications dict.

        Returns:
            Configured WorkerType instance.

        """
        hourly_cost = self.cost_per_vcpu_hour * spec["cpu_cores"]

        if self.cost_per_vcpu_hour > 0:
            cost_note = f"${self.cost_per_vcpu_hour}/vCPU-hour user-supplied pricing"
        else:
            cost_note = "self-hosted, no cost model configured"

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
                unit_name="pod",
                granularity_minutes=1,
            ),
            description=f"Executor pod preset {name} ({spec['cpu_cores']} CPUs, {spec['memory_gb']}GB RAM, "
            f"{cost_note})",
        )

    def get_worker_types(self) -> list[WorkerType]:
        """Get all available executor pod size presets.

        Returns:
            List of WorkerType for all pod presets.

        """
        return [self._create_worker_type(name, spec) for name, spec in self.WORKER_SPECS.items()]

    def get_worker_type(self, name: str) -> WorkerType | None:
        """Get a specific pod preset by name.

        Args:
            name: Pod preset name (small, medium, large, xlarge).

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
    ) -> WorkerType:
        """Recommend the best pod preset for given requirements.

        Args:
            target_memory_gb: Desired memory per executor pod.
            target_cores: Desired CPU cores per executor pod.

        Returns:
            Recommended WorkerType (the largest preset if nothing fits).

        """
        best_match = None
        best_score = float("inf")

        for worker in self.get_worker_types():
            r = worker.resources

            # Must meet minimum requirements
            if r.memory_gb < target_memory_gb or r.cpu_cores < target_cores:
                continue

            # Score based on resource efficiency (lower is better)
            memory_waste = r.memory_gb - target_memory_gb
            cpu_waste = r.cpu_cores - target_cores
            score = memory_waste * 0.4 + cpu_waste * 0.3

            if score < best_score:
                best_score = score
                best_match = worker

        # Fall back to the largest preset when targets exceed all presets
        if best_match is None:
            return self.get_worker_type("xlarge") or self.get_worker_types()[-1]

        return best_match

    def recommend_config(
        self,
        resources: ResourceSpec,
        spark_version: str,
        worker_count: int | None = None,
    ) -> ClusterConfig:
        """Recommend optimal Spark-on-Kubernetes configuration.

        The recommendation includes one driver pod (small preset by default)
        in addition to the executor pods.

        Args:
            resources: Target total resources.
            spark_version: Target Spark version.
            worker_count: Optional specific executor pod count.

        Returns:
            Recommended ClusterConfig.

        """
        # Recommend pod preset based on resource requirements
        memory_per_worker = min(resources.memory_gb / 10, 64)  # Target ~10 executors
        cores_per_worker = min(resources.cpu_cores / 10, 16)

        worker_type = self.recommend_worker_type(
            target_memory_gb=memory_per_worker,
            target_cores=int(cores_per_worker),
        )

        # Calculate executor pod count
        if worker_count is None:
            worker_count = self.get_optimal_worker_count(resources, worker_type)

        worker_count = max(worker_count, self._constraints.min_workers)

        # One driver pod in addition to the executors
        driver_type = self.get_worker_type(self.DEFAULT_DRIVER_PRESET) or worker_type

        return ClusterConfig(
            worker_type=worker_type,
            worker_count=worker_count,
            driver_type=driver_type,
            driver_count=1,
            spark_version=spark_version,
            platform_config={
                "namespace": self.namespace,
                "pod_preset": worker_type.name,
                "num_executors": worker_count,
                "container_image": self._container_image(spark_version),
            },
        )

    def translate_to_spark_config(
        self,
        cluster_config: ClusterConfig,
    ) -> dict[str, Any]:
        """Translate cluster config to Spark configuration for Kubernetes.

        ``spark.master`` is not set here — it depends on the cluster API
        server URL and must be supplied at submit time. Dynamic allocation
        uses shuffle tracking because Kubernetes has no external shuffle
        service. ~10% of pod memory is reserved as overhead headroom.

        Args:
            cluster_config: Kubernetes cluster configuration.

        Returns:
            Dictionary of Spark configuration parameters.

        """
        worker = cluster_config.worker_type
        r = worker.resources
        driver = cluster_config.driver_type or worker

        # Executor pods are dedicated, so all preset cores go to the executor;
        # node-level daemons are handled by Kubernetes itself.
        executor_cores = r.cpu_cores
        executor_memory = max(1, int(r.memory_gb * (1.0 - self.MEMORY_OVERHEAD_FACTOR)))
        driver_memory = max(1, int(driver.resources.memory_gb * (1.0 - self.MEMORY_OVERHEAD_FACTOR)))

        return {
            # Kubernetes-specific configuration
            "spark.kubernetes.container.image": self._container_image(cluster_config.spark_version),
            "spark.kubernetes.namespace": self.namespace,
            "spark.kubernetes.executor.request.cores": str(executor_cores),
            "spark.kubernetes.executor.limit.cores": str(executor_cores),
            # Executor configuration
            "spark.executor.instances": str(cluster_config.worker_count),
            "spark.executor.cores": str(executor_cores),
            "spark.executor.memory": f"{executor_memory}g",
            "spark.executor.memoryOverheadFactor": str(self.MEMORY_OVERHEAD_FACTOR),
            # Driver configuration
            "spark.driver.cores": str(driver.resources.cpu_cores),
            "spark.driver.memory": f"{driver_memory}g",
            # Dynamic allocation — Kubernetes has no external shuffle service,
            # so shuffle tracking is required instead of spark.shuffle.service.enabled
            "spark.dynamicAllocation.enabled": "true",
            "spark.dynamicAllocation.shuffleTracking.enabled": "true",
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

    def estimate_cost(
        self,
        cluster_config: ClusterConfig,
        duration_hours: float,
    ) -> dict[str, Any]:
        """Estimate cost for a Spark-on-Kubernetes run.

        Self-hosted clusters have no built-in pricing, so the total is
        ``total vCPUs x cost_per_vcpu_hour x hours`` using the user-supplied
        ``cost_per_vcpu_hour`` constructor parameter (0.0 by default, which
        yields a zero estimate).

        Args:
            cluster_config: Cluster configuration.
            duration_hours: Expected runtime in hours.

        Returns:
            Cost breakdown including driver and executor portions.

        """
        worker = cluster_config.worker_type
        driver = cluster_config.driver_type or worker

        billed_hours = max(duration_hours, 0.0)
        executor_vcpus = worker.resources.cpu_cores * cluster_config.worker_count
        driver_vcpus = driver.resources.cpu_cores * cluster_config.driver_count

        worker_cost = self.cost_per_vcpu_hour * executor_vcpus * billed_hours
        driver_cost = self.cost_per_vcpu_hour * driver_vcpus * billed_hours
        total_cost = worker_cost + driver_cost

        return {
            "platform": self.name,
            "currency": "USD",
            "namespace": self.namespace,
            "duration_hours": duration_hours,
            "total_cost": total_cost,
            "breakdown": {
                "driver_pod_preset": driver.name,
                "driver_count": cluster_config.driver_count,
                "driver_cost": driver_cost,
                "worker_pod_preset": worker.name,
                "worker_count": cluster_config.worker_count,
                "worker_cost": worker_cost,
                "total_vcpus": executor_vcpus + driver_vcpus,
                "cost_per_vcpu_hour": self.cost_per_vcpu_hour,
            },
            "notes": (
                "Self-hosted cluster: cost uses the user-supplied cost_per_vcpu_hour "
                "(0.0 by default, which yields a zero estimate)"
            ),
        }

    def get_spark_application_crd(
        self,
        cluster_config: ClusterConfig,
        app_name: str = "spark-optima-app",
        main_application_file: str = "local:///opt/spark/app/main.py",
        spark_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate a SparkApplication custom resource for the Spark Operator.

        The optimized Spark configuration is carried in ``spec.sparkConf`` as
        strings. Keys that the operator derives from the CR spec itself
        (image, namespace, instances, driver/executor cores and memory) are
        omitted from ``sparkConf`` to avoid conflicting definitions.

        Args:
            cluster_config: Cluster configuration.
            app_name: Name for the SparkApplication resource.
            main_application_file: Path to the application entrypoint inside
                the container image (placeholder — replace with your script).
            spark_config: Optional Spark configuration to embed in
                ``spec.sparkConf`` (derived from the cluster config if None).

        Returns:
            Dictionary representing a ``sparkoperator.k8s.io/v1beta2``
            SparkApplication manifest.

        """
        worker = cluster_config.worker_type
        driver = cluster_config.driver_type or worker

        if spark_config is None:
            spark_config = self.translate_to_spark_config(cluster_config)

        # Keys the operator populates from the CR spec itself
        spec_managed_keys = {
            "spark.kubernetes.container.image",
            "spark.kubernetes.namespace",
            "spark.executor.instances",
            "spark.executor.cores",
            "spark.executor.memory",
            "spark.driver.cores",
            "spark.driver.memory",
        }
        spark_conf = {
            k: str(v) for k, v in spark_config.items() if k.startswith("spark.") and k not in spec_managed_keys
        }

        executor_memory = max(1, int(worker.resources.memory_gb * (1.0 - self.MEMORY_OVERHEAD_FACTOR)))
        driver_memory = max(1, int(driver.resources.memory_gb * (1.0 - self.MEMORY_OVERHEAD_FACTOR)))

        return {
            "apiVersion": "sparkoperator.k8s.io/v1beta2",
            "kind": "SparkApplication",
            "metadata": {
                "name": app_name,
                "namespace": self.namespace,
            },
            "spec": {
                "type": "Python",
                "pythonVersion": "3",
                "mode": "cluster",
                "image": self._container_image(cluster_config.spark_version),
                "imagePullPolicy": "IfNotPresent",
                "mainApplicationFile": main_application_file,
                "sparkVersion": cluster_config.spark_version,
                "restartPolicy": {"type": "Never"},
                "driver": {
                    "cores": driver.resources.cpu_cores,
                    "memory": f"{driver_memory}g",
                    "serviceAccount": "spark",  # User should replace
                },
                "executor": {
                    "cores": worker.resources.cpu_cores,
                    "instances": cluster_config.worker_count,
                    "memory": f"{executor_memory}g",
                },
                "sparkConf": spark_conf,
            },
        }
