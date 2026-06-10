#!/usr/bin/env python3
"""Example: Optimize Spark configuration for Azure Synapse."""

from spark_optima import Optimizer
from spark_optima.platforms.models import ResourceSpec


def main():
    """Run optimization for Azure Synapse platform."""
    print("=" * 70)
    print("🔧 Spark Optima - Azure Synapse Example")
    print("=" * 70)

    # Azure Synapse node sizes:
    # - Small: 4-8 cores, 32GB per node
    # - Medium: 16-32 cores, 64-128GB per node
    # - Large: 48+ cores, 256GB+ per node

    optimizer = Optimizer(
        platform="azure_synapse",
        spark_version="3.4.0",  # Synapse uses older Spark
        optimization_mode="simulation",
    )

    resources = ResourceSpec(
        cpu_cores=16,  # Medium node
        memory_gb=64,
        disk_gb=1000,
    )

    data_profile = {
        "size_gb": 2000,
        "format": "parquet",
        "compression": "snappy",
        "partitioning": {"type": "date", "column": "event_date"},
    }

    result = optimizer.optimize(
        code_path="./my_synapse_job.py",
        resources=resources,
        data_profile=data_profile,
        use_bayesian=True,
        bayesian_trials=40,
        objectives=["minimize_time"],
    )

    print("\n🔧 Optimal Configuration for Azure Synapse:")
    print("-" * 70)
    for key in result.configuration:
        if "spark.executor" in key or "spark.driver" in key:
            print(f"  {key:45s} = {result.configuration[key]}")

    print(f"\n📊 Estimated Time: {result.estimated_time_minutes:.1f} minutes")
    print(f"💰 Estimated Cost: ${result.metadata.get('estimated_cost', 0):.2f}")

    # Export for Synapse pipeline
    print("\n📋 Synapse Pipeline Configuration:")
    print("  Add these to your Synapse Spark pool configuration")


if __name__ == "__main__":
    main()
