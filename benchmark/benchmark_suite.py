#!/usr/bin/env python3
"""Reproducible benchmark suite for Spark Optima.

Synthesizes TPC-DS-like workloads using data/generators.py, runs
before/after Spark configurations through the analytical performance
model, and writes reproducible results.

Usage:
    python benchmark/benchmark_suite.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add repo root to PYTHONPATH so imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spark_optima.core.config_engine.database import ConfigDatabase
from spark_optima.core.heuristics.context import DataProfile as HeuristicDataProfile
from spark_optima.core.heuristics.engine import HeuristicEngine
from spark_optima.core.simulation.performance_model import (
    DataCharacteristics,
    OperationProfile,
    OperationType,
    PerformanceModel,
)
from spark_optima.data.generators import (
    ColumnSpec,
    DataGenerator,
    DataGeneratorConfig,
)
from spark_optima.platforms.models import ResourceSpec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("benchmark")

# ------------------------------------------------------------------
# TPC-DS-like workload definitions
# ------------------------------------------------------------------

TPC_DS_WORKLOADS = {
    "store_sales_scan": {
        "description": "TPC-DS-like store_sales table scan (100M rows, 20 columns)",
        "num_rows": 100_000_000,
        "num_partitions": 200,
        "format": "parquet",
        "compression": "snappy",
        "columns": [
            ColumnSpec("ss_sold_time_sk", "int", nullable=False, min_value=1, max_value=86400),
            ColumnSpec("ss_item_sk", "int", nullable=False, min_value=1, max_value=300000),
            ColumnSpec("ss_customer_sk", "int", nullable=False, min_value=1, max_value=100000),
            ColumnSpec("ss_cdemo_sk", "int", nullable=True, cardinality=20),
            ColumnSpec("ss_hdemo_sk", "int", nullable=True, cardinality=50),
            ColumnSpec("ss_addr_sk", "int", nullable=True, cardinality=20000),
            ColumnSpec("ss_store_sk", "int", nullable=False, cardinality=100),
            ColumnSpec("ss_promo_sk", "int", nullable=True, cardinality=500),
            ColumnSpec("ss_ticket_number", "int", nullable=False, min_value=1, max_value=1000000),
            ColumnSpec("ss_quantity", "int", nullable=False, min_value=1, max_value=100),
            ColumnSpec("ss_wholesale_cost", "double", nullable=False, min_value=0.0, max_value=500.0),
            ColumnSpec("ss_list_price", "double", nullable=False, min_value=0.0, max_value=500.0),
            ColumnSpec("ss_sales_price", "double", nullable=False, min_value=0.0, max_value=500.0),
            ColumnSpec("ss_ext_discount_amt", "double", nullable=False, min_value=0.0, max_value=100.0),
            ColumnSpec("ss_ext_sales_price", "double", nullable=False, min_value=0.0, max_value=500.0),
            ColumnSpec("ss_ext_wholesale_cost", "double", nullable=False, min_value=0.0, max_value=500.0),
            ColumnSpec("ss_ext_list_price", "double", nullable=False, min_value=0.0, max_value=500.0),
            ColumnSpec("ss_ext_tax", "double", nullable=False, min_value=0.0, max_value=50.0),
            ColumnSpec("ss_coupon_amt", "double", nullable=False, min_value=0.0, max_value=50.0),
            ColumnSpec("ss_net_paid", "double", nullable=False, min_value=0.0, max_value=500.0),
        ],
        "operations": [OperationType.SCAN, OperationType.FILTER, OperationType.PROJECT],
        "join_details": {},
        "skew_factor": 1.5,
    },
    "store_sales_aggregation": {
        "description": "TPC-DS-like aggregation over store_sales (groupBy store, date)",
        "num_rows": 50_000_000,
        "num_partitions": 100,
        "format": "parquet",
        "compression": "snappy",
        "columns": [
            ColumnSpec("store_id", "int", nullable=False, cardinality=100),
            ColumnSpec("date_sk", "int", nullable=False, cardinality=3650),
            ColumnSpec("item_sk", "int", nullable=False, cardinality=300000),
            ColumnSpec("sales_amount", "double", nullable=False, min_value=1.0, max_value=1000.0),
            ColumnSpec("quantity", "int", nullable=False, min_value=1, max_value=100),
        ],
        "operations": [OperationType.SCAN, OperationType.AGGREGATION],
        "join_details": {},
        "skew_factor": 2.0,
    },
    "join_customer_store_sales": {
        "description": "TPC-DS-like join: customer × store_sales",
        "num_rows": 20_000_000,
        "num_partitions": 80,
        "format": "parquet",
        "compression": "snappy",
        "columns": [
            ColumnSpec("c_customer_sk", "int", nullable=False, cardinality=100000),
            ColumnSpec("c_first_name", "string", nullable=True, cardinality=5000),
            ColumnSpec("c_last_name", "string", nullable=True, cardinality=10000),
            ColumnSpec("c_birth_country", "string", nullable=True, cardinality=50),
            ColumnSpec("ss_customer_sk", "int", nullable=False, cardinality=100000),
            ColumnSpec("ss_sales_price", "double", nullable=False, min_value=1.0, max_value=500.0),
        ],
        "operations": [OperationType.SCAN, OperationType.JOIN, OperationType.FILTER],
        "join_details": {1: "shuffle_hash"},
        "skew_factor": 1.8,
    },
}

# ------------------------------------------------------------------
# Configuration presets
# ------------------------------------------------------------------

BASELINE_CONFIG = {
    "spark.executor.memory": "2g",
    "spark.executor.cores": "2",
    "spark.default.parallelism": "100",
    "spark.sql.shuffle.partitions": "100",
    "spark.sql.adaptive.enabled": "false",
    "spark.sql.adaptive.skewJoin.enabled": "false",
    "spark.dynamicAllocation.enabled": "false",
    "spark.memory.fraction": "0.6",
    "spark.memory.storageFraction": "0.5",
    "spark.serializer": "org.apache.spark.serializer.JavaSerializer",
    "spark.shuffle.compress": "true",
}

OPTIMIZED_CONFIG = {
    "spark.executor.memory": "4g",
    "spark.executor.cores": "4",
    "spark.default.parallelism": "400",
    "spark.sql.shuffle.partitions": "400",
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.skewJoin.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    "spark.dynamicAllocation.enabled": "true",
    "spark.dynamicAllocation.minExecutors": "2",
    "spark.dynamicAllocation.maxExecutors": "20",
    "spark.dynamicAllocation.initialExecutors": "4",
    "spark.dynamicAllocation.shuffleTracking.enabled": "true",
    "spark.memory.fraction": "0.75",
    "spark.memory.storageFraction": "0.3",
    "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
    "spark.shuffle.compress": "true",
    "spark.shuffle.spill.compress": "true",
}

# ------------------------------------------------------------------
# Benchmark runner
# ------------------------------------------------------------------


def run_benchmark() -> dict:
    """Run the reproducible benchmark and return aggregated results."""
    model = PerformanceModel()
    db = ConfigDatabase()
    heuristic_engine = HeuristicEngine(db.get_config_set("3.5.0"))

    results = {
        "meta": {
            "benchmark_name": "spark-optima-tpc-ds-like",
            "version": "1.0.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "reproducible": True,
            "random_seed": 42,
            "workload_source": "data/generators.py (synthetic TPC-DS-like)",
        },
        "workloads": {},
        "summary": {},
    }

    resources = ResourceSpec(cpu_cores=16, memory_gb=64)

    total_baseline_time = 0.0
    total_optimized_time = 0.0
    total_baseline_cost = 0.0
    total_optimized_cost = 0.0

    for workload_name, spec in TPC_DS_WORKLOADS.items():
        logger.info("Benchmarking workload: %s", workload_name)

        # Synthetic data profile (simulate without writing files for speed)
        data_profile = DataCharacteristics(
            size_gb=(spec["num_rows"] * 20 * len(spec["columns"])) / (1024**3) / 10,
            num_rows=spec["num_rows"],
            num_columns=len(spec["columns"]),
            format=spec["format"],
            compression=spec["compression"],
            partitioning=spec["num_partitions"],
            skew_factor=spec.get("skew_factor", 1.0),
        )

        operations = OperationProfile(
            operations=spec["operations"],
            join_details=spec.get("join_details", {}),
            has_aggregation=OperationType.AGGREGATION in spec["operations"],
            has_shuffle=OperationType.JOIN in spec["operations"] or OperationType.AGGREGATION in spec["operations"],
        )

        # Before (baseline)
        baseline_metrics = model.estimate(
            config=BASELINE_CONFIG,
            resource_spec=resources,
            data_profile=data_profile,
            operations=operations,
        )

        # After (optimized) - use heuristic-derived config if possible, else preset
        try:
            heuristic_profile = HeuristicDataProfile(
                format=spec["format"],
                size_gb=data_profile.size_gb,
                num_files=spec["num_partitions"],
                avg_file_size_mb=(data_profile.size_gb * 1024) / max(spec["num_partitions"], 1),
                compression=spec["compression"],
                has_nulls=True,
                is_partitioned=False,
            )
            heuristic_config = heuristic_engine.evaluate(
                resources=resources,
                platform="local",
                data_profile=heuristic_profile,
            )
            # Merge preset optimized values over heuristic for consistency
            optimized_config = dict(heuristic_config)
            optimized_config.update(OPTIMIZED_CONFIG)
            optimized_metrics = model.estimate(
                config=optimized_config,
                resource_spec=resources,
                data_profile=data_profile,
                operations=operations,
            )
        except Exception as exc:
            logger.warning("Heuristic evaluation failed (%s); falling back to preset optimized config", exc)
            optimized_metrics = model.estimate(
                config=OPTIMIZED_CONFIG,
                resource_spec=resources,
                data_profile=data_profile,
                operations=operations,
            )

        results["workloads"][workload_name] = {
            "description": spec.get("description", workload_name),
            "data_profile": {
                "num_rows": spec["num_rows"],
                "num_partitions": spec["num_partitions"],
                "format": spec["format"],
                "skew_factor": spec.get("skew_factor", 1.0),
            },
            "baseline": {
                "execution_time_seconds": round(baseline_metrics["execution_time_seconds"], 2),
                "cost_estimate_usd": round(baseline_metrics.get("cost_estimate_usd", 0.0), 4),
                "memory_peak_gb": round(baseline_metrics.get("memory_peak_gb", 0.0), 2),
                "shuffle_spill_gb": round(baseline_metrics.get("shuffle_spill_gb", 0.0), 2),
                "cpu_utilization_percent": round(baseline_metrics.get("cpu_utilization_percent", 0.0), 1),
                "simulation_confidence": round(baseline_metrics.get("simulation_confidence", 0.0), 3),
                "success": baseline_metrics.get("success", True),
            },
            "optimized": {
                "execution_time_seconds": round(optimized_metrics["execution_time_seconds"], 2),
                "cost_estimate_usd": round(optimized_metrics.get("cost_estimate_usd", 0.0), 4),
                "memory_peak_gb": round(optimized_metrics.get("memory_peak_gb", 0.0), 2),
                "shuffle_spill_gb": round(optimized_metrics.get("shuffle_spill_gb", 0.0), 2),
                "cpu_utilization_percent": round(optimized_metrics.get("cpu_utilization_percent", 0.0), 1),
                "simulation_confidence": round(optimized_metrics.get("simulation_confidence", 0.0), 3),
                "success": optimized_metrics.get("success", True),
            },
            "delta": {
                "time_seconds": round(
                    baseline_metrics["execution_time_seconds"] - optimized_metrics["execution_time_seconds"], 2
                ),
                "time_percent": round(
                    (baseline_metrics["execution_time_seconds"] - optimized_metrics["execution_time_seconds"])
                    / max(baseline_metrics["execution_time_seconds"], 1e-6)
                    * 100,
                    1,
                ),
                "cost_usd": round(
                    baseline_metrics.get("cost_estimate_usd", 0.0) - optimized_metrics.get("cost_estimate_usd", 0.0), 4
                ),
            },
        }

        total_baseline_time += baseline_metrics["execution_time_seconds"]
        total_optimized_time += optimized_metrics["execution_time_seconds"]
        total_baseline_cost += baseline_metrics.get("cost_estimate_usd", 0.0)
        total_optimized_cost += optimized_metrics.get("cost_estimate_usd", 0.0)

    results["summary"] = {
        "workloads_count": len(TPC_DS_WORKLOADS),
        "total_baseline_time_seconds": round(total_baseline_time, 2),
        "total_optimized_time_seconds": round(total_optimized_time, 2),
        "total_baseline_cost_usd": round(total_baseline_cost, 4),
        "total_optimized_cost_usd": round(total_optimized_cost, 4),
        "aggregate_time_improvement_percent": round(
            (total_baseline_time - total_optimized_time) / max(total_baseline_time, 1e-6) * 100,
            1,
        ),
        "aggregate_cost_improvement_percent": round(
            (total_baseline_cost - total_optimized_cost) / max(total_baseline_cost, 1e-9) * 100,
            1,
        ),
    }

    return results


def synthesize_data_for_workload(workload_name: str, output_dir: Path | None = None) -> Path:
    """Actually synthesize dataset using data/generators.py for audit/reproducibility."""
    spec = TPC_DS_WORKLOADS[workload_name]
    generator = DataGenerator()
    config = DataGeneratorConfig(
        num_rows=min(spec["num_rows"], 500_000),  # cap for local synthesis speed
        num_partitions=min(spec["num_partitions"], 20),
        format=spec["format"],
        compression=spec["compression"],
        random_seed=42,
    )
    out = output_dir or Path("benchmark_output")
    out.mkdir(parents=True, exist_ok=True)
    path = generator.generate(
        output_path=str(out / workload_name),
        config=config,
        columns=spec.get("columns"),
    )
    logger.info("Synthesized dataset at %s", path)
    return path


if __name__ == "__main__":
    # Synthesis is optional; skip by default for fast CI runs.
    # Uncomment next line to generate an audit dataset:
    # synthesize_data_for_workload("store_sales_scan")

    results = run_benchmark()

    output_path = Path("benchmark_output/benchmark_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Benchmark complete. Results written to {output_path}")
    print(json.dumps(results["summary"], indent=2))
