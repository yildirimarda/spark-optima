# Spark Optima Kubernetes Deployment

Production-ready Kubernetes manifests and Helm charts for deploying Spark Optima.

## 📁 Directory Structure

```
kubernetes/
├── base/                          # Raw Kubernetes manifests
│   ├── namespace.yaml
│   ├── serviceaccount.yaml
│   ├── rbac.yaml
│   ├── resourcequota.yaml
│   ├── configmap.yaml
│   ├── pvc.yaml
│   ├── deployment-api.yaml
│   ├── service-api.yaml
│   ├── ingress.yaml
│   ├── pdb.yaml
│   ├── networkpolicy.yaml
│   ├── hpa.yaml
│   └── job-cli.yaml
├── helm/                          # Helm chart
│   └── spark-optima/
│       ├── Chart.yaml
│       ├── values.yaml
│       ├── values-production.yaml
│       └── templates/
├── monitoring/                    # Prometheus/Grafana configs
│   ├── servicemonitor.yaml
│   └── prometheus-values.yaml
└── local-setup/                   # Local development setup
    ├── minikube-start.sh
    └── kind-config.yaml
```

## 🚀 Quick Start

### Option 1: kubectl (Quick Deploy)

```bash
# Deploy all base manifests
kubectl apply -f kubernetes/base/

# Check deployment status
kubectl get pods -n spark-optima
kubectl get svc -n spark-optima
```

### Option 2: Helm (Recommended for Production)

```bash
# Install Helm chart
helm install spark-optima kubernetes/helm/spark-optima -n spark-optima --create-namespace

# Upgrade
helm upgrade spark-optima kubernetes/helm/spark-optima -n spark-optima

# With production values
helm install spark-optima kubernetes/helm/spark-optima \
  -f kubernetes/helm/spark-optima/values-production.yaml \
  -n spark-optima --create-namespace
```

## 📋 Prerequisites

- Kubernetes 1.24+
- kubectl configured
- (Optional) Helm 3.10+
- (Optional) NGINX Ingress Controller
- (Optional) Prometheus Operator for monitoring

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SPARK_OPTIMA_ENV` | production | Runtime environment |
| `SPARK_OPTIMA_LOG_LEVEL` | INFO | Logging level |
| `SPARK_OPTIMA_DEFAULT_PLATFORM` | local | Default Spark platform |
| `SPARK_OPTIMA_MAX_TRIALS` | 50 | Optimization trials |

See `base/configmap.yaml` for all options.

### Resource Requirements

| Component | CPU Request | Memory Request | CPU Limit | Memory Limit |
|-----------|-------------|----------------|-----------|--------------|
| API | 500m | 1Gi | 1000m | 2Gi |
| CLI Job | 500m | 1Gi | 1000m | 2Gi |

## 🔐 Security Features

- **RBAC**: Least-privilege service accounts
- **NetworkPolicy**: Restricted pod-to-pod traffic
- **PodDisruptionBudget**: Ensures availability during disruptions
- **ResourceQuota**: Namespace resource limits
- **SecurityContext**: Non-root containers

## 📊 Monitoring

Enable Prometheus monitoring:

```bash
kubectl apply -f kubernetes/monitoring/servicemonitor.yaml
```

Install Prometheus/Grafana stack:

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack \
  -f kubernetes/monitoring/prometheus-values.yaml
```

## 🧪 Local Testing

See [LOCAL_SETUP.md](LOCAL_SETUP.md) for Minikube/Kind setup instructions.

## 📖 Production Deployment

See [PRODUCTION.md](PRODUCTION.md) for detailed production deployment guide.

## 🆘 Troubleshooting

```bash
# Check pod status
kubectl describe pods -n spark-optima

# View logs
kubectl logs -n spark-optima -l app.kubernetes.io/component=api

# Check events
kubectl get events -n spark-optima --sort-by='.lastTimestamp'
```
