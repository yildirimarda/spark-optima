#!/usr/bin/env python3
"""Advanced example: Using simulation mode for fast estimation."""

from spark_optima import Optimizer
from spark_optima.platforms.models import ResourceSpec


def main():
    """Run optimization in simulation mode."""
    print("=" * 70)
    print("🚀 Spark Optima - Simulation Mode Example")
    print("=" * 70)
    print("\nSimulation mode provides fast estimates without running actual Spark jobs.")
    print("Ideal for quick exploration of configuration space.\n")

    # Create optimizer in simulation mode
    optimizer = Optimizer(
        platform="local",
        spark_version="3.5.0",
        optimization_mode="simulation",  # Fast simulation mode
    )

    resources = ResourceSpec(
        cpu_cores=8,
        memory_gb=32,
        disk_gb=200,
    )

    data_profile = {
        "size_gb": 50,
        "format": "parquet",
        "compression": "snappy",
    }

    # Run optimization (simulation is fast!)
    print("Running simulation-based optimization...")
    result = optimizer.optimize(
        code_path="./my_job.py",
        resources=resources,
        data_profile=data_profile,
        use_bayesian=True,
        bayesian_trials=20,
        objectives=["minimize_time"],
    )

    print("\n📊 Simulation Results:")
    print("-" * 70)
    print(f"  Estimated Time: {result.estimated_time_minutes:.1f} minutes")
    print(f"  Confidence Score: {result.confidence_score:.0%}")

    print("\n🔧 Top Configuration Parameters:")
    for key in ["spark.executor.memory", "spark.executor.cores", "spark.sql.shuffle.partitions"]:
        if key in result.configuration:
            print(f"  {key:45s} = {result.configuration[key]}")


if __name__ == "__main__":
    main()
