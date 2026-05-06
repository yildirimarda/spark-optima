#!/usr/bin/env python3
"""
Simple Optimization Example

This example demonstrates the basic usage of Spark Optima for optimizing
Spark configurations.

Usage:
    python simple_optimization.py
    python simple_optimization.py --platform databricks --data-size 500
"""

import argparse
import json
from pathlib import Path

from spark_optima import Optimizer
from spark_optima.platforms.models import ResourceSpec


def create_sample_spark_job(output_path: str = "sample_job.py") -> str:
    """Create a sample Spark job for demonstration."""
    job_code = '''
"""
Sample ETL Job - Sales Data Processing

This is a sample Spark job that demonstrates typical ETL operations:
- Reading data from Parquet
- Performing aggregations
- Joining datasets
- Writing results
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, count, avg, max as spark_max


def main():
    # Initialize Spark session
    spark = SparkSession.builder \\
        .appName("SalesETL") \\
        .getOrCreate()
    
    # Read sales data
    print("Reading sales data...")
    sales_df = spark.read.parquet("s3://bucket/sales_data/")
    
    # Read product catalog
    print("Reading product catalog...")
    products_df = spark.read.parquet("s3://bucket/products/")
    
    # Join datasets
    print("Joining sales and products...")
    enriched_df = sales_df.join(
        products_df,
        on="product_id",
        how="inner"
    )
    
    # Perform aggregations
    print("Calculating metrics...")
    daily_metrics = enriched_df.groupBy("date", "category") \\
        .agg(
            spark_sum("amount").alias("total_sales"),
            count("*").alias("transaction_count"),
            avg("amount").alias("avg_transaction"),
            spark_max("amount").alias("max_transaction")
        )
    
    # Filter high-value categories
    high_value = daily_metrics.filter(col("total_sales") > 10000)
    
    # Write results
    print("Writing results...")
    high_value.write \\
        .mode("overwrite") \\
        .parquet("s3://bucket/output/daily_metrics/")
    
    print("ETL job completed successfully!")
    spark.stop()


if __name__ == "__main__":
    main()
'''
    
    with open(output_path, "w") as f:
        f.write(job_code)
    
    return output_path


def run_optimization(
    platform: str = "local",
    spark_version: str = "3.5.0",
    data_size_gb: float = 100.0,
    data_format: str = "parquet",
    max_memory_gb: float = 64.0,
    use_bayesian: bool = True,
    bayesian_trials: int = 30,
) -> None:
    """Run Spark Optima optimization."""
    
    print("=" * 70)
    print("🔥 Spark Optima - Simple Optimization Example")
    print("=" * 70)
    print()
    
    # Create sample job file
    job_file = "sample_etl_job.py"
    if not Path(job_file).exists():
        print(f"Creating sample Spark job: {job_file}")
        create_sample_spark_job(job_file)
    
    print(f"Platform: {platform}")
    print(f"Spark Version: {spark_version}")
    print(f"Data Size: {data_size_gb} GB")
    print(f"Data Format: {data_format}")
    print(f"Max Memory: {max_memory_gb} GB")
    print(f"Bayesian Optimization: {use_bayesian} (trials: {bayesian_trials})")
    print()
    
    # Initialize optimizer
    print("Initializing optimizer...")
    optimizer = Optimizer(
        platform=platform,
        spark_version=spark_version,
        optimization_mode="simulation"
    )
    
    # Define resources
    resources = ResourceSpec(
        cpu_cores=16,
        memory_gb=max_memory_gb,
        disk_gb=500.0
    )
    
    # Define data profile
    data_profile = {
        "size_gb": data_size_gb,
        "format": data_format,
        "compression": "snappy",
        "partitioning": {
            "type": "date",
            "column": "date"
        }
    }
    
    # Define resource constraints
    resource_constraints = {
        "max_memory_gb": max_memory_gb,
        "max_cost_usd": 50.0,
        "max_executors": 20
    }
    
    # Run optimization
    print("Running optimization...")
    print("-" * 70)
    
    result = optimizer.optimize(
        code_path=job_file,
        resources=resources,
        data_profile=data_profile,
        resource_constraints=resource_constraints,
        use_bayesian=use_bayesian,
        bayesian_trials=bayesian_trials,
        objectives=["minimize_time"]
    )
    
    print("-" * 70)
    print()
    
    # Display results
    print("📊 OPTIMIZATION RESULTS")
    print("=" * 70)
    print()
    
    print(f"⏱️  Estimated Execution Time: {result.estimated_time_minutes:.1f} minutes")
    print(f"🎯 Confidence Score: {result.confidence_score:.0%}")
    print(f"💰 Estimated Cost: ${result.metadata.get('estimated_cost', 0):.2f}")
    print()
    
    print("🔧 TOP CONFIGURATION PARAMETERS")
    print("-" * 70)
    
    # Display key configuration parameters
    key_configs = [
        "spark.executor.memory",
        "spark.executor.cores",
        "spark.driver.memory",
        "spark.sql.adaptive.enabled",
        "spark.sql.shuffle.partitions",
        "spark.dynamicAllocation.enabled",
        "spark.dynamicAllocation.maxExecutors",
    ]
    
    for key in key_configs:
        if key in result.configuration:
            print(f"  {key:45s} = {result.configuration[key]}")
    
    print()
    
    # Display code suggestions
    if result.code_suggestions:
        print("💡 CODE SUGGESTIONS")
        print("-" * 70)
        for i, suggestion in enumerate(result.code_suggestions[:5], 1):
            print(f"{i}. Line {suggestion.line_number}: {suggestion.issue_type}")
            print(f"   Severity: {suggestion.severity}")
            print(f"   {suggestion.description}")
            print(f"   💡 {suggestion.suggestion}")
            print()
    
    # Save results to file
    output_file = f"optimization_result_{platform}.json"
    with open(output_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    
    print(f"✅ Results saved to: {output_file}")
    print()
    
    # Display platform-specific info
    if result.platform_specific:
        print("🖥️  PLATFORM-SPECIFIC CONFIGURATION")
        print("-" * 70)
        print(f"Platform: {result.platform_specific.get('platform', 'N/A')}")
        print(f"Spark Version: {result.platform_specific.get('spark_version', 'N/A')}")
        
        if 'cluster_config' in result.platform_specific:
            print("Cluster Config:", result.platform_specific['cluster_config'])
        
        print()
    
    # Display metadata
    print("📈 OPTIMIZATION METADATA")
    print("-" * 70)
    print(f"Platform: {result.metadata.get('platform', 'N/A')}")
    print(f"Spark Version: {result.metadata.get('spark_version', 'N/A')}")
    print(f"Optimization Mode: {result.metadata.get('optimization_mode', 'N/A')}")
    print(f"Bayesian Used: {result.metadata.get('bayesian_used', False)}")
    print(f"Bayesian Trials: {result.metadata.get('bayesian_trials', 0)}")
    print()
    
    print("=" * 70)
    print("✨ Optimization Complete!")
    print("=" * 70)
    
    # Next steps
    print()
    print("Next Steps:")
    print(f"  1. Review the configuration in {output_file}")
    print("  2. Export to your platform using: spark-optima export")
    print("  3. Apply the configuration to your Spark job")
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Spark Optima - Simple Optimization Example"
    )
    
    parser.add_argument(
        "--platform",
        type=str,
        default="local",
        choices=["local", "aws_glue", "databricks", "azure_synapse"],
        help="Target platform (default: local)"
    )
    
    parser.add_argument(
        "--spark-version",
        type=str,
        default="3.5.0",
        help="Spark version (default: 3.5.0)"
    )
    
    parser.add_argument(
        "--data-size",
        type=float,
        default=100.0,
        help="Data size in GB (default: 100)"
    )
    
    parser.add_argument(
        "--data-format",
        type=str,
        default="parquet",
        choices=["parquet", "delta", "json", "csv", "orc"],
        help="Data format (default: parquet)"
    )
    
    parser.add_argument(
        "--max-memory",
        type=float,
        default=64.0,
        help="Maximum memory in GB (default: 64)"
    )
    
    parser.add_argument(
        "--no-bayesian",
        action="store_true",
        help="Disable Bayesian optimization"
    )
    
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Number of Bayesian trials (default: 30)"
    )
    
    args = parser.parse_args()
    
    try:
        run_optimization(
            platform=args.platform,
            spark_version=args.spark_version,
            data_size_gb=args.data_size,
            data_format=args.data_format,
            max_memory_gb=args.max_memory,
            use_bayesian=not args.no_bayesian,
            bayesian_trials=args.trials,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Optimization interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()