#!/usr/bin/env python3
"""Example: Sizing a Spark-on-Kubernetes deployment and exporting a CRD.

This example uses the SparkOnK8sPlatform adapter directly (no cluster
needed — everything is a local computation) to:

1. Browse the executor pod size presets (small / medium / large / xlarge).
2. Recommend a pod layout for a resource budget (recommend_config).
3. Translate the layout into Spark-on-K8s settings — dynamic allocation via
   shuffle tracking, memory overhead headroom (translate_to_spark_config).
4. Estimate the run cost from a user-supplied $/vCPU-hour rate
   (estimate_cost; self-hosted clusters default to $0).
5. Export a SparkApplication custom resource for the Spark Operator
   (get_spark_application_crd).
"""

import yaml

from spark_optima.platforms.models import ResourceSpec
from spark_optima.platforms.spark_k8s import SparkOnK8sPlatform


def main() -> None:
    """Recommend, translate, price, and export a Spark-on-K8s deployment."""
    print("=" * 70)
    print("🔥 Spark Optima - Spark on Kubernetes Example")
    print("=" * 70)

    # cost_per_vcpu_hour is user-supplied: self-hosted clusters have no
    # universal price. 0.0 (the default) yields zero-cost estimates.
    platform = SparkOnK8sPlatform(namespace="data-jobs", cost_per_vcpu_hour=0.04)

    # 1. Browse the executor pod size presets
    print("\n🖥️  Executor pod size presets:")
    print("-" * 70)
    for worker in platform.get_worker_types():
        r = worker.resources
        print(f"  {worker.name:8s} {r.cpu_cores:3.0f} CPUs, {r.memory_gb:4.0f}GB RAM, {r.disk_gb:.0f}GB ephemeral")

    # 2. Recommend a pod layout for a total resource budget
    resources = ResourceSpec(cpu_cores=48, memory_gb=192, disk_gb=500)
    cluster = platform.recommend_config(resources=resources, spark_version="3.5.0")

    print("\n📐 Recommended layout for 48 CPUs / 192GB total:")
    print("-" * 70)
    print(f"  Namespace:       {cluster.platform_config['namespace']}")
    print(f"  Driver pod:      1 x {cluster.driver_type.name if cluster.driver_type else 'small'}")
    print(f"  Executor pods:   {cluster.worker_count} x {cluster.worker_type.name}")
    print(f"  Container image: {cluster.platform_config['container_image']}")

    # 3. Translate to Spark configuration. spark.master is NOT set here:
    #    it depends on your API server URL (k8s://https://<api-server>:<port>)
    #    or is implied by the Spark Operator.
    spark_config = platform.translate_to_spark_config(cluster)
    print("\n🔧 Spark configuration for this layout:")
    print("-" * 70)
    for key, value in spark_config.items():
        print(f"  {key:60s} = {value}")

    # 4. Estimate the cost of a 2-hour run at the supplied $/vCPU-hour
    cost = platform.estimate_cost(cluster, duration_hours=2.0)
    breakdown = cost["breakdown"]
    print("\n💰 Cost estimate for a 2-hour run (at $0.04/vCPU-hour):")
    print("-" * 70)
    print(f"  Total:        ${cost['total_cost']:.2f} ({breakdown['total_vcpus']:.0f} vCPUs)")
    print(f"  Driver pod:   ${breakdown['driver_cost']:.2f}")
    print(f"  Executors:    ${breakdown['worker_cost']:.2f}")

    # 5. Export a SparkApplication CRD for the Spark Operator
    crd = platform.get_spark_application_crd(
        cluster,
        app_name="sales-etl",
        main_application_file="local:///opt/spark/app/sales_etl.py",
    )
    print("\n📋 SparkApplication manifest (Spark Operator CRD):")
    print("-" * 70)
    print(yaml.safe_dump(crd, sort_keys=False, default_flow_style=False))

    print("Next step: save the manifest and apply it with")
    print("  kubectl apply -f sales-etl.yaml")


if __name__ == "__main__":
    main()
