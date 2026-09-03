# Spark Optima Benchmark Results

> **Claim under test**: Spark Optima eliminates guesswork by turning synthetic workload synthesis (`data/generators.py`) into measured before/after numbers.

---

## Methodology

### Workload Synthesis

All benchmarks use the synthetic data generators in `src/spark_optima/data/generators.py` (`DataGenerator`, `DataGeneratorConfig`, `ColumnSpec`). Three TPC-DS-like workloads were defined:

| Workload | Rows | Format | Key Pattern |
|---|---|---|---|
| `store_sales_scan` | 100 M | parquet / snappy | Scan + filter + project |
| `store_sales_aggregation` | 50 M | parquet / snappy | Scan + groupBy aggregation |
| `join_customer_store_sales` | 20 M | parquet / snappy | Shuffle join + filter |

Skew is intentionally injected (`skew_factor` 1.5–2.0) so that AQE and shuffle-tuning recommendations have a realistic stress target.

### Configurations Tested

**Baseline (before)** — a minimal, conservative Spark 3.5 config:

- `spark.executor.memory`: 2g
- `spark.executor.cores`: 2
- `spark.sql.adaptive.enabled`: false
- `spark.dynamicAllocation.enabled`: false
- Java serializer
- Low parallelism (100 partitions)

**Optimized (after)** — the heuristic-derived configuration enhanced by the benchmark preset:

- `spark.executor.memory`: 4g
- `spark.executor.cores`: 4
- `spark.sql.adaptive.enabled`: true (with AQE coalesce + skew join)
- `spark.dynamicAllocation.enabled`: true
- Kryo serializer
- Higher parallelism (400 partitions) with adaptive coalesce

### Measurement Engine

Estimates come from `PerformanceModel.estimate()` in `core/simulation/performance_model.py`. The model accounts for:

- Per-format read speeds (`FORMAT_READ_SPEEDS`)
- GC overhead from memory pressure (`GC_PRESSURE_LOW` / `HIGH`)
- Shuffle transfer bounded by per-node network bandwidth (`NETWORK_GBPS_PER_NODE` = 1.25 GB/s) and per-core disk throughput (`SHUFFLE_DISK_GBPS_PER_CORE` = 0.15 GB/s)
- Straggler/skew model with AQE skew-join mitigation (`AQE_SKEW_CAP` = 2.0)

Results include simulation confidence (`simulation_confidence`) and a warning when confidence < 0.7.

### Reproducibility

- Fixed random seed (`random_seed = 42`) inside `DataGeneratorConfig`
- Deterministic column specs and operation profiles
- No external data files required; synthesis is reproducible from the generator code

---

## Running the Benchmark

```bash
python benchmark/benchmark_suite.py
```

This writes `benchmark_output/benchmark_results.json` and prints the aggregate summary to stdout.

To synthesize an audit dataset alongside the simulation:

```python
from benchmark.benchmark_suite import synthesize_data_for_workload
path = synthesize_data_for_workload("store_sales_scan")
```

---

## Results (Sample Run)

Results below are produced by the analytical model using the methodology above. They represent relative improvements rather than absolute wall-clock guarantees; for absolute measurements, switch the optimizer to `execution` mode against a real Spark cluster.

### Aggregate Improvement

| Metric | Baseline | Optimized | Change |
|---|---|---|---|
| Total execution time (sim) | ~X s | ~Y s | +Z% faster |
| Total cost estimate | $A | $B | +W% lower |

*(Exact numbers are written to `benchmark_output/benchmark_results.json` after each run; the JSON format allows CI tracking over time.)*

### Per-Workload Breakdown

Each workload record in the JSON result contains:

- `baseline.execution_time_seconds`
- `optimized.execution_time_seconds`
- `delta.time_percent`
- `delta.cost_usd`
- `delta.time_seconds`
- Simulation confidence and feasibility flag

---

## Interpreting the Numbers

- **Time improvement** comes mainly from:
  1. AQE coalescing reducing task overhead
  2. Skew-join splitting capping straggler time
  3. Dynamic allocation reducing idle executor waste
  4. Higher parallelism matching larger clusters

- **Cost improvement** follows time reduction (same resource spec) plus lower spill from better memory fractions (`memory.fraction` raised to 0.75) and Kryo serialization reducing shuffle bytes.

- **Simulation confidence** stays above 0.7 for scan/filter/project workloads; it drops slightly for complex joins because join overhead depends on cardinality estimates that the model approximates rather than measures.

---

## Limitations

1. **Simulation only** — `PerformanceModel` is analytical, not measured from real Spark runs. For production claims, pair these numbers with `execution` mode using event logs (`core/execution/event_log.py`).
2. **Synthetic data** — `DataGenerator` produces statistically similar but not bit-for-bit identical TPC-DS tables. Schema fidelity is high; value distributions are approximate.
3. **Fixed resource spec** — All runs use `ResourceSpec(cpu_cores=16, memory_gb=64)`. Different cluster sizes will scale results differently.

---

## Growing the Benchmark

New workload profiles should be added to `TPC_DS_WORKLOADS` in `benchmark/benchmark_suite.py` with:

- `columns`: a list of `ColumnSpec`
- `operations`: ordered list of `OperationType`
- `join_details`: dict mapping operation index to join type

New metrics (e.g., network transfer time, GC overhead fraction) can be extracted from `PerformanceModel.estimate()` and added to the JSON result schema.
