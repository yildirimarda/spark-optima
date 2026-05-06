#!/usr/bin/env python3
"""Advanced example: Bayesian optimization with custom search space."""

from spark_optima import Optimizer
from spark_optima.core.bayesian import SearchSpace
from spark_optima.platforms.models import ResourceSpec

def main():
    """Run advanced Bayesian optimization."""
    print("=" * 70)
    print("🧠 Advanced Bayesian Optimization Example")
    print("=" * 70)

    # Create optimizer with Bayesian-only mode
    optimizer = Optimizer(
        platform="local",
        spark_version="3.5.0",
        optimization_mode="bayesian",
    )

    # Define custom search space
    custom_space = {
        "spark.executor.memory": SearchSpace(min_value=4, max_value=32, step=4),
        "spark.executor.cores": SearchSpace(min_value=2, max_value=8, step=2),
        "spark.sql.shuffle.partitions": SearchSpace(min_value=200, max_value=2000, step=200),
        "spark.memory.fraction": SearchSpace(min_value=0.4, max_value=0.8, step=0.1),
    }

    resources = ResourceSpec(
        cpu_cores=8,
        memory_gb=32,
        disk_gb=200,
    )

    result = optimizer.optimize(
        code_path="./complex_job.py",
        resources=resources,
        data_profile={"size_gb": 200, "format": "parquet"},
        use_bayesian=True,
        bayesian_trials=100,  # More trials for thorough search
        custom_search_space=custom_space,
        objectives=["minimize_time"],
    )

    print(f"\n📈 Bayesian Optimization Results:")
    print(f"  Trials: 100")
    print(f"  Best time: {result.estimated_time_minutes:.1f} minutes")
    print(f"  Confidence: {result.confidence_score:.0%}")


if __name__ == "__main__":
    main()
