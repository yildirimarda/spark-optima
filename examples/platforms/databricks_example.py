#!/usr/bin/env python3
"""Example: Optimize Spark configuration for Databricks."""

from spark_optima import Optimizer
from spark_optima.platforms.models import ResourceSpec


def main():
    """Run optimization for Databricks platform."""
    print("=" * 70)
    print("🔥 Spark Optima - Databricks Example")
    print("=" * 70)

    # Databricks cluster sizes:
    # - Small: 2-4 cores, 8-16GB per node
    # - Medium: 8-16 cores, 32-64GB per node
    # - Large: 32+ cores, 128GB+ per node

    optimizer = Optimizer(
        platform="databricks",
        spark_version="3.5.0",
        optimization_mode="simulation",
    )

    resources = ResourceSpec(
        cpu_cores=16,  # Medium cluster
        memory_gb=64,
        disk_gb=500,
    )

    data_profile = {
        "size_gb": 1000,
        "format": "delta",
        "compression": "snappy",
        "partitioning": {"type": "hash", "columns": ["customer_id"]},
    }

    result = optimizer.optimize(
        code_path="./my_databricks_notebook.py",
        resources=resources,
        data_profile=data_profile,
        use_bayesian=True,
        bayesian_trials=50,
        objectives=["minimize_time"],
    )

    print("\n🔧 Optimal Configuration for Databricks:")
    print("-" * 70)
    for key in result.configuration:
        if "spark.executor" in key or "spark.driver" in key:
            print(f"  {key:45s} = {result.configuration[key]}")

    print(f"\n📊 Estimated Time: {result.estimated_time_minutes:.1f} minutes")

    # Export for Databricks
    print("\n📋 Databricks Cluster Configuration:")
    print("  spark.conf.set for each parameter in result.configuration")


if __name__ == "__main__":
    main()
