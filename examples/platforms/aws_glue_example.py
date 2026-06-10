#!/usr/bin/env python3
"""Example: Optimize Spark configuration for AWS Glue."""

import json

from spark_optima import Optimizer
from spark_optima.platforms.models import ResourceSpec


def main():
    """Run optimization for AWS Glue platform."""
    print("=" * 70)
    print("🔥 Spark Optima - AWS Glue Example")
    print("=" * 70)

    # AWS Glue worker types:
    # - G.1X: 1 DPU (4 vCPU, 16GB RAM)
    # - G.2X: 2 DPU (8 vCPU, 32GB RAM)
    # - G.4X: 4 DPU (16 vCPU, 64GB RAM)

    optimizer = Optimizer(
        platform="aws_glue",
        spark_version="3.5.0",
        optimization_mode="simulation",
    )

    resources = ResourceSpec(
        cpu_cores=8,  # G.2X worker
        memory_gb=32,
        disk_gb=100,
    )

    data_profile = {
        "size_gb": 500,
        "format": "parquet",
        "compression": "snappy",
        "partitioning": {"type": "date", "column": "transaction_date"},
    }

    result = optimizer.optimize(
        code_path="./my_glue_job.py",
        resources=resources,
        data_profile=data_profile,
        use_bayesian=True,
        bayesian_trials=30,
        objectives=["minimize_time", "minimize_cost"],
    )

    print("\n🔧 Optimal Configuration for AWS Glue (G.2X):")
    print("-" * 70)
    for key in ["spark.executor.memory", "spark.executor.cores", "spark.sql.shuffle.partitions"]:
        if key in result.configuration:
            print(f"  {key:45s} = {result.configuration[key]}")

    print(f"\n📊 Estimated Time: {result.estimated_time_minutes:.1f} minutes")
    print(f"💰 Estimated Cost: ${result.metadata.get('estimated_cost', 0):.2f}")

    # Export for Glue job
    from spark_optima.cli.formatters import format_config_for_glue

    glue_config = format_config_for_glue(result.configuration)
    print("\n📋 Glue Job Configuration:")
    print(json.dumps(glue_config, indent=2))


if __name__ == "__main__":
    main()
