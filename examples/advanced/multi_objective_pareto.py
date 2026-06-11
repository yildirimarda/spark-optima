#!/usr/bin/env python3
"""Example: Multi-objective optimization with a Pareto frontier.

Passing more than one objective to ``Optimizer.optimize()`` switches the
Bayesian phase (Optuna) into multi-objective mode. Instead of a single
"best" trial there is a *Pareto frontier*: the set of trials where no
objective can be improved without worsening another (e.g. faster but more
expensive vs. slower but cheaper).

This example runs a small simulation-mode optimization with
``objectives=["minimize_time", "minimize_cost"]`` and prints the Pareto
frontier persisted in ``result.metadata["pareto_frontier"]``. Each point
carries its trial number, objective values, and configuration.

CLI equivalents:
    spark-optima optimize -c job.py --objective minimize_time --objective minimize_cost
    spark-optima pareto -r result.json
    spark-optima export -r result.json -f pareto-csv
"""

import tempfile
from pathlib import Path

import optuna

from spark_optima import Optimizer
from spark_optima.platforms.models import ResourceSpec

# Silence per-trial Optuna logging so the example output stays readable
optuna.logging.set_verbosity(optuna.logging.WARNING)

SAMPLE_JOB = """
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum

spark = SparkSession.builder.appName("DailyRevenue").getOrCreate()

orders = spark.read.parquet("s3://bucket/orders/")
customers = spark.read.parquet("s3://bucket/customers/")

revenue = (
    orders.join(customers, on="customer_id", how="inner")
    .groupBy("order_date", "segment")
    .agg(spark_sum("amount").alias("revenue"))
    .filter(col("revenue") > 1000)
)

revenue.write.mode("overwrite").parquet("s3://bucket/output/daily_revenue/")
spark.stop()
"""


def main() -> None:
    """Run a time-vs-cost optimization and print the Pareto frontier."""
    print("=" * 70)
    print("⚖️  Spark Optima - Multi-Objective (Pareto) Example")
    print("=" * 70)

    optimizer = Optimizer(
        platform="aws_glue",  # A platform with a cost model makes minimize_cost meaningful
        spark_version="3.5.0",
        optimization_mode="simulation",
    )

    resources = ResourceSpec(cpu_cores=16, memory_gb=64, disk_gb=500)
    data_profile = {"size_gb": 200, "format": "parquet", "compression": "snappy"}

    with tempfile.TemporaryDirectory() as tmp_dir:
        job_file = Path(tmp_dir) / "daily_revenue_job.py"
        job_file.write_text(SAMPLE_JOB)

        print("\nOptimizing for BOTH minimize_time and minimize_cost (15 trials)...")
        result = optimizer.optimize(
            code_path=job_file,
            resources=resources,
            data_profile=data_profile,
            use_bayesian=True,
            bayesian_trials=15,  # Small count to keep the example fast
            objectives=["minimize_time", "minimize_cost"],
        )

    print("\n📊 RECOMMENDED CONFIGURATION (balanced pick from the frontier)")
    print("-" * 70)
    for key in ["spark.executor.memory", "spark.executor.cores", "spark.sql.shuffle.partitions"]:
        if key in result.configuration:
            print(f"  {key:45s} = {result.configuration[key]}")
    print(f"  Estimated time: {result.estimated_time_minutes:.1f} minutes")

    # Multi-objective runs persist the Pareto frontier into result.metadata
    # (single-objective runs do not add these keys).
    objectives = result.metadata.get("objectives", [])
    frontier = result.metadata.get("pareto_frontier", [])

    plural = "s" if len(frontier) != 1 else ""
    print(f"\n🏔️  PARETO FRONTIER ({len(frontier)} non-dominated trial{plural})")
    print(f"  Objectives: {', '.join(objectives)}")
    print("-" * 70)
    for point in frontier:
        values = point["objective_values"]
        rendered = ", ".join(f"{name}={value:.3f}" for name, value in values.items())
        print(f"  Trial #{point['trial_number']:<3d} {rendered}")
        config = point["configuration"]
        for key in ["spark.executor.memory", "spark.executor.cores"]:
            if key in config:
                print(f"      {key} = {config[key]}")

    print("\nEach point is a different time/cost trade-off; no point on the")
    print("frontier is strictly better than another. Pick the one matching")
    print("your SLA and budget, or inspect result.json with:")
    print("  spark-optima pareto -r result.json")


if __name__ == "__main__":
    main()
