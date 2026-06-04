# Architecture Overview

This document describes the system architecture of Spark Optima, including component interactions, data flow, and design patterns.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Spark Optima System                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         User Interfaces                                  ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                     ││
│  │  │    CLI      │  │  Python API │  │  REST API   │                     ││
│  │  │  (Typer)    │  │             │  │  (FastAPI)  │                     ││
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                     ││
│  │         └─────────────────┴─────────────────┘                          ││
│  └───────────────────────────────┬─────────────────────────────────────────┘│
│                                  │                                           │
│  ┌───────────────────────────────▼─────────────────────────────────────────┐│
│  │                        Core Optimization Engine                          ││
│  │  ┌─────────────────────────────────────────────────────────────────┐   ││
│  │  │                      Optimizer (Facade)                          │   ││
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │   ││
│  │  │  │   Heuristic │  │  Bayesian   │  │      Code Analysis      │  │   ││
│  │  │  │   Engine    │──│ Optimizer   │  │    (AST Analysis)       │  │   ││
│  │  │  │             │  │  (Optuna)   │  │                         │  │   ││
│  │  │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │   ││
│  │  └─────────────────────────────────────────────────────────────────┘   ││
│  │                                                                          ││
│  │  ┌─────────────────────────────────────────────────────────────────┐   ││
│  │  │              Simulation & Execution Engines                      │   ││
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │   ││
│  │  │  │ Simulation  │  │  Execution  │  │    Metrics Collector    │  │   ││
│  │  │  │   Model     │  │   Engine    │  │                         │  │   ││
│  │  │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │   ││
│  │  └─────────────────────────────────────────────────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                  │                                           │
│  ┌───────────────────────────────▼─────────────────────────────────────────┐│
│  │                        Platform Adapters                                 ││
│  │  ┌─────────┐ ┌──────────┐ ┌────────────┐ ┌──────────────────┐         ││
│  │  │  Local  │ │AWS Glue  │ │ Databricks │ │  Azure Synapse   │         ││
│  │  │         │ │          │ │            │ │                  │         ││
│  │  └─────────┘ └──────────┘ └────────────┘ └──────────────────┘         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                  │                                           │
│  ┌───────────────────────────────▼─────────────────────────────────────────┐│
│  │                        Data & Configuration                              ││
│  │  ┌─────────────────────────────────────────────────────────────────┐   ││
│  │  │                 Configuration Database (YAML)                      │   ││
│  │  │  Spark 3.0  Spark 3.1  Spark 3.2  Spark 3.3  Spark 3.4  Spark 4.x │   ││
│  │  └─────────────────────────────────────────────────────────────────┘   ││
│  │                                                                          ││
│  │  ┌─────────────────────────────────────────────────────────────────┐   ││
│  │  │                   Sample Data Generators                         │   ││
│  │  └─────────────────────────────────────────────────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. User Interfaces

#### CLI (Command Line Interface)
- **Technology**: Typer
- **Purpose**: Interactive command-line tool
- **Features**:
  - `optimize` command for running optimizations
  - `wizard` for guided configuration
  - `export` for platform-specific formats
  - Rich terminal output with progress indicators

#### Python API
- **Purpose**: Programmatic access
- **Use Cases**:
  - Jupyter notebooks
  - CI/CD pipelines
  - Custom applications
  - Integration with orchestration tools

#### REST API
- **Technology**: FastAPI
- **Purpose**: HTTP-based API
- **Features**:
  - Async endpoints
  - OpenAPI documentation
  - Health checks
  - Platform information endpoints

### 2. Core Optimization Engine

#### Optimizer (Facade Pattern)
The `Optimizer` class serves as the main entry point and facade, coordinating all subsystems:

```python
class Optimizer:
    def __init__(self, platform, spark_version, optimization_mode):
        self.heuristic_engine = HeuristicEngine(...)
        self.bayesian_optimizer = BayesianOptimizer(...)
        self.code_analyzer = RecommendationEngine(...)
    
    def optimize(self, ...):
        # 1. Run heuristics
        # 2. Run Bayesian optimization (if enabled)
        # 3. Analyze code
        # 4. Build result
```

#### Heuristic Engine
- **Purpose**: Generate baseline configuration
- **Approach**: Rule-based system
- **Rules Categories**:
  - Memory heuristics
  - CPU/core heuristics
  - Shuffle optimization
  - Serialization settings
  - Platform-specific rules

```
Heuristic Rule Processing:
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Input     │───▶│ Apply Rules │───▶│   Config    │
│  Context    │    │  (Priority) │    │   Output    │
└─────────────┘    └─────────────┘    └─────────────┘
```

#### Bayesian Optimizer
- **Technology**: Optuna
- **Purpose**: Fine-tune heuristic configuration
- **Features**:
  - Multi-objective optimization
  - Early stopping (pruning)
  - Parallel trial execution
  - Various samplers (TPE, CMA-ES, etc.)

```
Bayesian Optimization Flow:
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Define     │     │    Run       │     │   Update     │
│ Search Space │────▶│   Trials     │────▶│   Model      │
└──────────────┘     └──────────────┘     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │   Converge   │
                     │   to Optimal │
                     └──────────────┘
```

#### Code Analysis
- **Technology**: Python AST (Abstract Syntax Tree)
- **Purpose**: Detect code smells and optimization opportunities
- **Detections**:
  - Missing broadcast hints
  - Unnecessary shuffles
  - Caching issues
  - UDF usage
  - Data skew potential

### 3. Simulation & Execution Engines

#### Simulation Engine
- **Purpose**: Fast performance estimation
- **Approach**: Performance modeling
- **Models**:
  - Execution time estimation
  - Memory usage prediction
  - Cost modeling
  - I/O pattern analysis

#### Execution Engine
- **Purpose**: Real Spark execution
- **Use Case**: Precise measurements
- **Features**:
  - Spark job submission
  - Metrics collection
  - Resource monitoring
  - Result validation

### 4. Platform Adapters

Each platform adapter implements the `Platform` interface:

```python
class Platform(ABC):
    @abstractmethod
    def validate_resources(self, resources: ResourceSpec) -> bool:
        ...
    
    @abstractmethod
    def estimate_cost(self, config: dict, duration: float) -> float:
        ...
    
    @abstractmethod
    def export_config(self, result: OptimizationResult) -> dict:
        ...
```

#### Supported Platforms

| Platform | Key Features |
|----------|--------------|
| **Local** | Standalone Spark, Docker support |
| **AWS Glue** | Worker types, job bookmarks, Data Catalog |
| **Databricks** | DBR versions, node types, Unity Catalog, Photon |
| **Azure Synapse** | Spark pools, ADLS Gen2, auto-pause |

### 5. Data & Configuration

#### Configuration Database
- **Format**: YAML files
- **Content**: Spark configuration parameters by version
- **Structure**:
  ```yaml
  version: "3.5.0"
  parameters:
    spark.executor.memory:
      default: "1g"
      category: "memory"
      description: "Executor memory"
    spark.sql.adaptive.enabled:
      default: "false"
      category: "sql"
      description: "Enable AQE"
  ```

#### Sample Data Generators
- **Purpose**: Create test data for execution mode
- **Formats**: Parquet, Delta, JSON, CSV
- **Features**:
  - Schema generation
  - Data distribution control
  - Partitioning support

## Data Flow

### Optimization Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Optimization Data Flow                                │
└─────────────────────────────────────────────────────────────────────────────┘

Phase 1: Input Collection
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Code Path  │    │   Platform  │    │   Data      │    │  Resources  │
│  (optional) │    │   Selection │    │   Profile   │    │  (optional) │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │                  │
       └──────────────────┴──────────────────┴──────────────────┘
                          │
                          ▼

Phase 2: Code Analysis (if code provided)
┌──────────────────────────────────────────────────────────────────────────┐
│  Parse AST ──▶ Detect Operations ──▶ Identify Smells ──▶ Suggestions    │
└──────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼

Phase 3: Heuristic Configuration
┌──────────────────────────────────────────────────────────────────────────┐
│  Resource Analysis ──▶ Rule Evaluation ──▶ Conflict Resolution ──▶ Config│
└──────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼

Phase 4: Bayesian Optimization (if enabled)
┌──────────────────────────────────────────────────────────────────────────┐
│  Define Search Space ──▶ Run Trials ──▶ Evaluate ──▶ Converge          │
└──────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼

Phase 5: Result Construction
┌──────────────────────────────────────────────────────────────────────────┐
│  Merge Results ──▶ Platform Config ──▶ Validation ──▶ OptimizationResult │
└──────────────────────────────────────────────────────────────────────────┘
```

## Design Patterns

### 1. Facade Pattern
The `Optimizer` class provides a simplified interface to the complex subsystem.

### 2. Strategy Pattern
Different optimization strategies:
- `HeuristicStrategy`
- `BayesianStrategy`
- `HybridStrategy`

### 3. Adapter Pattern
Platform adapters unify different cloud provider APIs.

### 4. Builder Pattern
`OptimizationResult` is constructed step by step.

### 5. Observer Pattern
Metrics collection and monitoring use observer pattern.

## Extension Points

### Adding a New Platform

1. Create new platform adapter:
```python
class NewPlatform(Platform):
    def validate_resources(self, resources):
        # Implementation
        pass
    
    def estimate_cost(self, config, duration):
        # Implementation
        pass
```

2. Register in platform factory:
```python
PLATFORMS = {
    "local": LocalPlatform,
    "new_platform": NewPlatform,
}
```

### Adding Custom Heuristic Rules

1. Create new rule:
```python
class CustomRule(HeuristicRule):
    def evaluate(self, context: HeuristicContext) -> dict:
        return {"spark.custom.config": "value"}
    
    def priority(self) -> int:
        return 100
```

2. Register in rule engine:
```python
engine.register_rule(CustomRule())
```

### Adding Custom Objectives

1. Define objective function:
```python
def custom_objective(trial, config, context):
    # Custom logic
    return score
```

2. Register with optimizer:
```python
optimizer.register_objective("custom", custom_objective)
```

## Performance Considerations

### Optimization Performance

| Phase | Typical Duration | Factors |
|-------|------------------|---------|
| Code Analysis | 1-5 seconds | File size, complexity |
| Heuristics | <1 second | Resource count |
| Bayesian (50 trials) | 30-60 seconds | Trial count, simulation speed |
| Total | 30-120 seconds | All factors |

### Memory Usage

- **Heuristic Engine**: ~50 MB
- **Bayesian Optimizer**: ~100 MB (Optuna)
- **Code Analysis**: ~50 MB (AST)
- **Total**: ~200 MB base + trial overhead

### Scaling

- **Parallel Trials**: Support for n_jobs > 1
- **Distributed**: Can run on multiple machines
- **Caching**: Results cached for identical inputs

## Security Considerations

### Data Handling
- Code is parsed but not executed
- No data leaves the system
- Credentials passed via environment variables

### Platform Credentials
- AWS: IAM roles or access keys
- Databricks: Tokens or OAuth
- Azure: Service principals or managed identity

## Deployment Architecture

### Local Deployment
```
┌─────────────────────────────────────┐
│           Local Machine             │
│  ┌─────────────────────────────┐   │
│  │      Spark Optima CLI       │   │
│  │  ┌───────────────────────┐  │   │
│  │  │   Optimization Engine │  │   │
│  │  └───────────────────────┘  │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

### Docker Deployment
```
┌─────────────────────────────────────┐
│        Docker Container             │
│  ┌─────────────────────────────┐   │
│  │      Spark Optima API       │   │
│  │  ┌───────────────────────┐  │   │
│  │  │   Optimization Engine │  │   │
│  │  └───────────────────────┘  │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

### Kubernetes Deployment
```
┌─────────────────────────────────────────────────────────────┐
│                     Kubernetes Cluster                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  API Pod    │  │  Worker Pod │  │   Worker Pod        │ │
│  │  (FastAPI)  │  │ (Bayesian)  │  │   (Bayesian)        │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## See Also

- [Testing Guide](testing.md) - Testing architecture
- [Contributing Guide](../development/contributing.md) - Development guidelines
- [Configuration Guide](../user-guide/configuration.md) - Configuration options