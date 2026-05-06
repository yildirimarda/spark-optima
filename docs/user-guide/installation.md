# Installation Guide

This guide covers all the ways to install and set up Spark Optima for your environment.

## Prerequisites

### System Requirements

- **Python**: 3.10 or higher
- **Operating System**: Linux, macOS, or Windows (WSL recommended)
- **Memory**: Minimum 4GB RAM (8GB+ recommended)
- **Disk**: 1GB free space for installation

### Optional Dependencies

Depending on your target platform, you may need:

- **AWS Glue**: AWS CLI configured with appropriate credentials
- **Databricks**: Databricks CLI or workspace access
- **Azure Synapse**: Azure CLI configured
- **Local Spark**: Java 8 or 11, Apache Spark 3.x

## Installation Methods

### Method 1: pip (Recommended for Users)

Install the latest stable release from PyPI:

```bash
pip install spark-optima
```

Install with platform-specific extras:

```bash
# For AWS Glue support
pip install spark-optima[aws]

# For Databricks support
pip install spark-optima[databricks]

# For all platforms
pip install spark-optima[all]
```

### Method 2: Poetry (Recommended for Development)

If you're contributing to Spark Optima or need the development tools:

```bash
# Clone the repository
git clone https://github.com/yourusername/spark-optima.git
cd spark-optima

# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install --all-extras

# Activate virtual environment
poetry shell
```

### Method 3: Docker

Run Spark Optima in a containerized environment:

```bash
# Pull the image
docker pull sparkoptima/spark-optima:latest

# Run CLI
docker run --rm sparkoptima/spark-optima spark-optima --help

# Run with mounted volume for your code
docker run --rm -v $(pwd)/my_code:/code sparkoptima/spark-optima \
  spark-optima optimize -c /code/my_job.py
```

Build from source:

```bash
# Build Docker image
docker build -f docker/Dockerfile --target production -t spark-optima:latest .

# Run
docker run --rm -it spark-optima:latest
```

### Method 4: Docker Compose

For a complete development environment:

```bash
# Start services
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f

# Stop services
docker-compose -f docker/docker-compose.yml down
```

## Platform-Specific Setup

### Local/Standalone Spark

1. **Install Java** (required for Spark):

   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install openjdk-11-jdk

   # macOS
   brew install openjdk@11

   # Verify
   java -version
   ```

2. **Install Apache Spark** (optional, for Execution mode):

   ```bash
   # Download Spark
   wget https://archive.apache.org/dist/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz
   
   # Extract
   tar -xzf spark-3.5.0-bin-hadoop3.tgz
   
   # Set environment variables
   export SPARK_HOME=/path/to/spark-3.5.0-bin-hadoop3
   export PATH=$PATH:$SPARK_HOME/bin
   ```

3. **Verify Spark installation**:

   ```bash
   spark-submit --version
   ```

### AWS Glue

1. **Install AWS CLI**:

   ```bash
   pip install awscli
   aws --version
   ```

2. **Configure AWS credentials**:

   ```bash
   aws configure
   # Enter your AWS Access Key ID
   # Enter your AWS Secret Access Key
   # Enter default region (e.g., us-east-1)
   ```

3. **Verify access**:

   ```bash
   aws sts get-caller-identity
   ```

4. **Install boto3** (for programmatic access):

   ```bash
   pip install boto3
   ```

### Databricks

1. **Install Databricks CLI**:

   ```bash
   pip install databricks-cli
   ```

2. **Configure authentication** (choose one method):

   **Method A: Personal Access Token**
   ```bash
   databricks configure --token
   # Enter your Databricks host (e.g., https://myworkspace.cloud.databricks.com)
   # Enter your personal access token
   ```

   **Method B: OAuth (recommended for production)**
   ```bash
   databricks auth login --host <workspace-url>
   ```

3. **Verify connection**:

   ```bash
   databricks clusters list
   ```

4. **Install Databricks SDK**:

   ```bash
   pip install databricks-sdk
   ```

### Azure Synapse

1. **Install Azure CLI**:

   ```bash
   # Windows
   winget install Microsoft.AzureCLI

   # macOS
   brew install azure-cli

   # Ubuntu
   curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
   ```

2. **Login to Azure**:

   ```bash
   az login
   ```

3. **Set default subscription** (if you have multiple):

   ```bash
   az account set --subscription "My Subscription"
   ```

4. **Install Azure Identity** (for programmatic access):

   ```bash
   pip install azure-identity
   ```

## Verifying Installation

### Check CLI Installation

```bash
# Check version
spark-optima --version

# View help
spark-optima --help

# Run optimization wizard
spark-optima wizard
```

### Check Python API

```python
from spark_optima import Optimizer, __version__

print(f"Spark Optima version: {__version__}")

# Test optimizer initialization
optimizer = Optimizer(platform="local", spark_version="3.5.0")
print("Optimizer initialized successfully!")
```

### Check API Server (if using REST API)

```bash
# Start the API server
uvicorn spark_optima.api.main:app --reload

# In another terminal, test the health endpoint
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","version":"0.1.0"}
```

## Troubleshooting

### Common Issues

#### Issue: `ModuleNotFoundError: No module named 'spark_optima'`

**Solution**: Ensure you're in the correct virtual environment:

```bash
# With Poetry
poetry shell

# With venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

#### Issue: `ImportError: cannot import name 'Optimizer'`

**Solution**: Update to the latest version:

```bash
pip install --upgrade spark-optima
```

#### Issue: AWS/Databricks/Azure credentials not found

**Solution**: Verify your credentials are configured:

```bash
# AWS
aws sts get-caller-identity

# Databricks
databricks clusters list

# Azure
az account show
```

#### Issue: Docker permission denied

**Solution**: Add your user to the docker group (Linux):

```bash
sudo usermod -aG docker $USER
# Log out and log back in
```

### Getting Help

If you encounter issues not covered here:

1. Check the [Troubleshooting Guide](../troubleshooting.md)
2. Search [GitHub Issues](https://github.com/yourusername/spark-optima/issues)
3. Join our [Discord community](https://discord.gg/spark-optima)
4. Create a new issue with details about your problem

## Development Installation

For contributing to Spark Optima:

```bash
# Clone the repository
git clone https://github.com/yourusername/spark-optima.git
cd spark-optima

# Install with all dev dependencies
poetry install --all-extras

# Install pre-commit hooks
poetry run pre-commit install

# Run tests to verify setup
poetry run pytest -m unit

# Run all checks
make check-all
```

See [Contributing Guide](../../CONTRIBUTING.md) for more details.

## Next Steps

Now that you have Spark Optima installed:

1. **Learn the CLI** → [CLI Usage Guide](cli.md)
2. **Try the Python API** → [API Usage Guide](api.md)
3. **Optimize your first job** → [Configuration Guide](configuration.md)
4. **Set up your platform** → [Platform Guides](../platforms/)