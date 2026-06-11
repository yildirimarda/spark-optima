#!/usr/bin/env python3
"""Example: Sizing and pricing a Spark cluster on GCP Dataproc.

This example uses the GCPDataprocPlatform adapter directly (no GCP
credentials needed — everything is a local computation) to:

1. Browse the modeled N2 machine types (n2-standard / n2-highmem).
2. Recommend a cluster layout for a resource budget (recommend_config).
3. Translate the cluster into YARN-oriented Spark settings
   (translate_to_spark_config).
4. Estimate the run cost — Compute Engine + the $0.01/vCPU-hour Dataproc
   fee — and compare on-demand vs. preemptible workers (estimate_cost).
5. Export a cluster definition for the ``clusters.create`` REST API
   (get_dataproc_cluster_config).
"""

import json

from spark_optima.platforms.gcp_dataproc import GCPDataprocPlatform
from spark_optima.platforms.models import ResourceSpec


def main() -> None:
    """Recommend, translate, price, and export a Dataproc cluster config."""
    print("=" * 70)
    print("🔥 Spark Optima - GCP Dataproc Example")
    print("=" * 70)

    platform = GCPDataprocPlatform(region="us-central1")

    # 1. Browse the modeled N2 machine types
    print("\n🖥️  Available machine types (Compute Engine + Dataproc fee):")
    print("-" * 70)
    for worker in platform.get_worker_types():
        r = worker.resources
        print(
            f"  {worker.name:16s} {r.cpu_cores:3.0f} vCPUs, {r.memory_gb:5.0f}GB RAM"
            f"  ${worker.cost.unit_cost_per_hour:.3f}/h",
        )

    # 2. Recommend a cluster for a total resource budget.
    #    Dataproc requires at least 2 primary workers; one master node is
    #    always added on top of the workers.
    resources = ResourceSpec(cpu_cores=64, memory_gb=256, disk_gb=1000)
    cluster = platform.recommend_config(resources=resources, spark_version="3.5.0")

    print("\n📐 Recommended cluster for 64 vCPUs / 256GB total:")
    print("-" * 70)
    print(f"  Image version:   {cluster.platform_config['image_version']}")
    print(f"  Master node:     1 x {cluster.platform_config['master_machine_type']}")
    print(f"  Worker nodes:    {cluster.worker_count} x {cluster.worker_type.name}")
    print(f"  Spark version:   {cluster.spark_version}")

    # 3. Translate to Spark configuration
    spark_config = platform.translate_to_spark_config(cluster)
    print("\n🔧 Spark configuration for this cluster:")
    print("-" * 70)
    for key, value in spark_config.items():
        print(f"  {key:55s} = {value}")

    # 4. Estimate the cost of a 2-hour run (on-demand vs. preemptible)
    cost = platform.estimate_cost(cluster, duration_hours=2.0)
    print("\n💰 Cost estimate for a 2-hour run (on-demand workers):")
    print("-" * 70)
    print(f"  Total:           ${cost['total_cost']:.2f} ({cost['currency']}, {cost['region']})")
    print(f"  Compute Engine:  ${cost['breakdown']['compute_cost']:.2f}")
    print(f"  Dataproc fee:    ${cost['breakdown']['dataproc_fee']:.2f}")
    print(f"  Pricing source:  {cost['pricing_source']}")

    # Preemptible (Spot) workers trade reliability for a large discount on
    # the Compute Engine portion (the Dataproc fee is unchanged).
    spot_platform = GCPDataprocPlatform(region="us-central1", use_preemptible_workers=True)
    spot_cost = spot_platform.estimate_cost(cluster, duration_hours=2.0)
    savings = cost["total_cost"] - spot_cost["total_cost"]
    print(f"\n  With preemptible workers: ${spot_cost['total_cost']:.2f} (saves ${savings:.2f})")

    # 5. Export a clusters.create request body
    cluster_def = platform.get_dataproc_cluster_config(cluster, cluster_name="sales-etl-cluster")
    print("\n📋 Dataproc cluster definition (clusters.create request body):")
    print("-" * 70)
    print(json.dumps(cluster_def, indent=2))

    print("\nNext step: create the cluster with")
    print("  gcloud dataproc clusters create ... or the projects.regions.clusters.create API")


if __name__ == "__main__":
    main()
