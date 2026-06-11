#!/usr/bin/env python3
"""Example: Sizing and pricing a Spark cluster on AWS EMR.

This example uses the AWSEMRPlatform adapter directly (no AWS credentials
needed — everything is a local computation) to:

1. Browse the modeled EC2 instance types (m5 / r5 / c5 families).
2. Recommend a cluster layout for a resource budget (recommend_config).
3. Translate the cluster into YARN-oriented Spark settings
   (translate_to_spark_config).
4. Estimate the run cost, including the EMR surcharge and regional
   multiplier (estimate_cost).
5. Export a ready-to-submit cluster definition for
   ``boto3.client("emr").run_job_flow(**config)`` (get_emr_cluster_config).

Set ``SPARK_OPTIMA_LIVE_PRICING=1`` to replace the static price table with
live EC2 on-demand rates from the AWS Pricing API (opt-in; off by default).
"""

import json

from spark_optima.platforms.aws_emr import AWSEMRPlatform
from spark_optima.platforms.models import ResourceSpec


def main() -> None:
    """Recommend, translate, price, and export an EMR cluster config."""
    print("=" * 70)
    print("🔥 Spark Optima - AWS EMR Example")
    print("=" * 70)

    platform = AWSEMRPlatform(region="us-east-1")

    # 1. Browse the modeled EC2 instance types
    print("\n🖥️  Available EC2 instance types (EC2 price + ~25% EMR surcharge):")
    print("-" * 70)
    for worker in platform.get_worker_types():
        r = worker.resources
        print(
            f"  {worker.name:12s} {r.cpu_cores:3.0f} vCPUs, {r.memory_gb:5.0f}GB RAM"
            f"  ${worker.cost.unit_cost_per_hour:.3f}/h",
        )

    # 2. Recommend a cluster for a total resource budget
    resources = ResourceSpec(cpu_cores=64, memory_gb=256, disk_gb=1000)
    cluster = platform.recommend_config(resources=resources, spark_version="3.5.0")

    print("\n📐 Recommended cluster for 64 vCPUs / 256GB total:")
    print("-" * 70)
    print(f"  Release label:   {cluster.platform_config['release_label']}")
    print(f"  Master node:     1 x {cluster.platform_config['master_instance_type']}")
    print(f"  Core nodes:      {cluster.worker_count} x {cluster.worker_type.name}")
    print(f"  Spark version:   {cluster.spark_version}")

    # 3. Translate to Spark configuration (YARN-oriented: spark.master is
    #    left to the EMR runtime; one core per node reserved for daemons)
    spark_config = platform.translate_to_spark_config(cluster)
    print("\n🔧 Spark configuration for this cluster:")
    print("-" * 70)
    for key, value in spark_config.items():
        print(f"  {key:55s} = {value}")

    # 4. Estimate the cost of a 2-hour run
    cost = platform.estimate_cost(cluster, duration_hours=2.0)
    breakdown = cost["breakdown"]
    print("\n💰 Cost estimate for a 2-hour run:")
    print("-" * 70)
    print(f"  Total:           ${cost['total_cost']:.2f} ({cost['currency']}, {cost['region']})")
    print(f"  EC2 portion:     ${breakdown['ec2_cost']:.2f}")
    print(f"  EMR surcharge:   ${breakdown['emr_surcharge']:.2f} ({breakdown['emr_surcharge_rate']:.0%})")
    print(f"  Pricing source:  {cost['pricing_source']} (set SPARK_OPTIMA_LIVE_PRICING=1 for live rates)")

    # 5. Export a boto3 run_job_flow cluster definition
    job_flow = platform.get_emr_cluster_config(cluster, cluster_name="sales-etl-cluster")
    print("\n📋 EMR cluster definition (boto3 run_job_flow):")
    print("-" * 70)
    print(json.dumps(job_flow, indent=2))

    print("\nNext step: submit with")
    print('  boto3.client("emr").run_job_flow(**job_flow)')


if __name__ == "__main__":
    main()
