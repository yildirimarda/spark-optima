# Spark Optima - Production Deployment Guide

This guide covers production deployment of Spark Optima on Kubernetes.

## 📋 Pre-Deployment Checklist

- [ ] Kubernetes cluster 1.24+ with sufficient resources
- [ ] NGINX Ingress Controller installed
- [ ] cert-manager for TLS (recommended)
- [ ] Storage class configured for PVCs
- [ ] Prometheus Operator for monitoring (optional)

## 🚀 Deployment Steps

### 1. Create Namespace

```bash
kubectl apply -f kubernetes/base/namespace.yaml
```

### 2. Configure RBAC

```bash
kubectl apply -f kubernetes/base/serviceaccount.yaml
kubectl apply -f kubernetes/base/rbac.yaml
```

### 3. Set Resource Quotas

```bash
kubectl apply -f kubernetes/base/resourcequota.yaml
```

### 4. Configure Storage

Adjust PVC storage class in `base/pvc.yaml` if needed:

```yaml
storageClassName: fast-ssd  # Change to your storage class
```

Then apply:

```bash
kubectl apply -f kubernetes/base/pvc.yaml
```

### 5. Configure Application

Edit `base/configmap.yaml` with your settings, then apply:

```bash
kubectl apply -f kubernetes/base/configmap.yaml
```

### 6. Deploy API

```bash
kubectl apply -f kubernetes/base/deployment-api.yaml
kubectl apply -f kubernetes/base/service-api.yaml
```

### 7. Configure Ingress

Update `base/ingress.yaml` with your domain:

```yaml
spec:
  rules:
    - host: api.spark-optima.yourdomain.com  # Change this
```

Apply:

```bash
kubectl apply -f kubernetes/base/ingress.yaml
```

### 8. Configure Auto-scaling

```bash
kubectl apply -f kubernetes/base/hpa.yaml
```

### 9. Configure High Availability

```bash
kubectl apply -f kubernetes/base/pdb.yaml
```

### 10. Configure Network Security

```bash
kubectl apply -f kubernetes/base/networkpolicy.yaml
```

## 🔒 TLS Configuration

### Using cert-manager

1. Install cert-manager:

```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
```

2. Create ClusterIssuer:

```yaml
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your-email@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
      - http01:
          ingress:
            class: nginx
```

3. The ingress is already configured to use cert-manager annotations.

## 📊 Monitoring Setup

### Install Prometheus Stack

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install prometheus prometheus-community/kube-prometheus-stack \
  -n monitoring --create-namespace \
  -f kubernetes/monitoring/prometheus-values.yaml
```

### Access Grafana

```bash
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80
# Open http://localhost:3000
# Default credentials: admin / spark-optima-admin
```

## 🔧 Production Values

Use the provided production values file:

```bash
helm install spark-optima kubernetes/helm/spark-optima \
  -f kubernetes/helm/spark-optima/values-production.yaml \
  -n spark-optima
```

Key production settings:
- 3 replicas for HA
- Higher resource limits
- Enabled ingress with TLS
- CronJob for scheduled optimizations
- Larger storage (50Gi)

## 🔄 Upgrading

### Using kubectl

```bash
kubectl apply -f kubernetes/base/
```

### Using Helm

```bash
helm upgrade spark-optima kubernetes/helm/spark-optima -n spark-optima
```

## 🧹 Uninstalling

```bash
# Using kubectl
kubectl delete -f kubernetes/base/
kubectl delete namespace spark-optima

# Using Helm
helm uninstall spark-optima -n spark-optima
kubectl delete namespace spark-optima
```

## 🆘 Troubleshooting

### Pod Not Starting

```bash
kubectl describe pod -n spark-optima <pod-name>
kubectl logs -n spark-optima <pod-name>
```

### PVC Pending

Check storage class:

```bash
kubectl get storageclass
kubectl describe pvc -n spark-optima
```

### Ingress Not Working

```bash
kubectl get ingress -n spark-optima
kubectl describe ingress -n spark-optima
# Check ingress controller logs
kubectl logs -n ingress-nginx deployment/ingress-nginx-controller
```

## 🔐 API Security & Async Jobs (v1.2)

The API supports opt-in security via environment variables (set them in the Deployment/Helm values):

- `SPARK_OPTIMA_API_KEYS` — comma-separated API keys; when set, `/api/v1/*` requires an `X-API-Key` header (health probes stay open).
- `SPARK_OPTIMA_RATE_LIMIT` — requests/minute per client; unset or `0` disables limiting.

**Important:** by default the async job store (`POST /api/v1/optimize/async` + `GET /api/v1/jobs/{id}`) is **in-memory and process-local** — jobs are lost on restart and invisible to other workers/replicas.

For **single-node persistence** (v1.3), switch to the SQLite backend:

- `SPARK_OPTIMA_JOB_STORE` — `memory` (default) or `sqlite`. Any other value logs a warning and falls back to `memory`.
- `SPARK_OPTIMA_JOB_DB` — SQLite database file path (default `~/.spark_optima/jobs.db`). In Kubernetes, point this at a **PVC-backed path** (e.g. mount a PersistentVolumeClaim at `/data` and set `SPARK_OPTIMA_JOB_DB=/data/jobs.db`), otherwise the database disappears with the pod filesystem.

With the SQLite store, job state survives API restarts and multiple uvicorn workers on the *same node* (same DB file, WAL mode) can see each other's jobs. Note the optimization itself still runs in-process: if the process executing a job dies mid-run, the job is reported as **failed with a "worker lost" error** once it has been unfinished for longer than the staleness window (2 hours).

**Multi-replica deployments:** the SQLite store does **not** make the async API safe across replicas — each pod has its own filesystem (or would contend on a shared file over network storage). With multiple replicas, either enable sticky sessions on the ingress, run a single API replica, or keep using the synchronous `POST /api/v1/optimize` endpoint; a true external job store (e.g. Redis/database-backed) remains future work.

## 📞 Support

For issues and questions:
- GitHub Issues: https://github.com/yourusername/spark-optima/issues
- Documentation: https://your-project.readthedocs.io
