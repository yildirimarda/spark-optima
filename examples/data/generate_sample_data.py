#!/usr/bin/env python3
"""Example: Generate sample data for testing."""

from spark_optima.data.generators import DataGenerator, DataGeneratorConfig, ColumnSpec

def main():
    """Generate sample data for Spark optimization testing."""
    print("=" * 70)
    print("📊 Spark Optima - Data Generation Example")
    print("=" * 70)

    # Create generator
    generator = DataGenerator()

    # Define schema
    columns = [
        ColumnSpec(name="id", data_type="int", min_value=1, max_value=1000000, nullable=False),
        ColumnSpec(name="customer_id", data_type="int", cardinality=10000),
        ColumnSpec(name="product_id", data_type="int", cardinality=1000),
        ColumnSpec(name="amount", data_type="double", min_value=10.0, max_value=1000.0),
        ColumnSpec(name="category", data_type="string", cardinality=20),
        ColumnSpec(name="transaction_date", data_type="date"),
        ColumnSpec(name="is_fraud", data_type="boolean"),
    ]

    # Generate Parquet data
    print("\n📊 Generating Parquet data (100K rows)...")
    config = DataGeneratorConfig(
        num_rows=100000,
        num_partitions=4,
        format="parquet",
        compression="snappy",
        null_ratio=0.05,
        random_seed=42,
    )
    result = generator.generate("./sample_data/parquet", config=config, columns=columns)
    print(f"✅ Data saved to: {result}")

    # Generate CSV data
    print("\n📊 Generating CSV data (50K rows)...")
    config = DataGeneratorConfig(
        num_rows=50000,
        format="csv",
        compression="gzip",
    )
    result = generator.generate("./sample_data/csv", config=config, columns=columns)
    print(f"✅ Data saved to: {result}")

    # Estimate size
    print("\n📈 Data Size Estimation:")
    print("-" * 70)
    estimate = generator.estimate_size(num_rows=1000000, num_cols=7)
    print(f"  1M rows, 7 columns: {estimate['estimated_gb']:.2f} GB")
    estimate = generator.estimate_size(num_rows=10000000, num_cols=15)
    print(f"  10M rows, 15 columns: {estimate['estimated_gb']:.2f} GB")


if __name__ == "__main__":
    main()
