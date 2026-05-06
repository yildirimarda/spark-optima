#!/bin/bash
# Copyright 2024 Spark Optima Team
# Licensed under the Apache License, Version 2.0

# Spark Optima - Minikube Local Setup Script
# This script sets up a local Kubernetes cluster with Minikube for testing

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CLUSTER_NAME="spark-optima"
MEMORY="4096"  # 4GB
CPUS="2"
DISK_SIZE="20g"
KUBERNETES_VERSION="v1.28.0"

echo -e "${BLUE}====================================${NC}"
echo -e "${BLUE}  Spark Optima - Minikube Setup    ${NC}"
echo -e "${BLUE}====================================${NC}"
echo ""

# Check if minikube is installed
if ! command -v minikube &> /dev/null; then
    echo -e "${RED}Error: minikube is not installed${NC}"
    echo "Install from: https://minikube.sigs.k8s.io/docs/start/"
    exit 1
fi

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}Error: kubectl is not installed${NC}"
    echo "Install from: https://kubernetes.io/docs/tasks/tools/"
    exit 1
fi

echo -e "${GREEN}✓ Prerequisites met${NC}"

# Check if cluster already exists
if minikube status -p $CLUSTER_NAME &> /dev/null; then
    echo -e "${YELLOW}⚠ Cluster '$CLUSTER_NAME' already exists${NC}"
    read -p "Delete and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Deleting existing cluster...${NC}"
        minikube delete -p $CLUSTER_NAME
    else
        echo -e "${YELLOW}Using existing cluster${NC}"
        minikube start -p $CLUSTER_NAME
        exit 0
    fi
fi

# Start Minikube cluster
echo -e "${BLUE}Starting Minikube cluster...${NC}"
echo "  Name: $CLUSTER_NAME"
echo "  Memory: $MEMORY MB"
echo "  CPUs: $CPUS"
echo "  Disk: $DISK_SIZE"
echo "  Kubernetes: $KUBERNETES_VERSION"
echo ""

minikube start \
    -p $CLUSTER_NAME \
    --memory=$MEMORY \
    --cpus=$CPUS \
    --disk-size=$DISK_SIZE \
    --kubernetes-version=$KUBERNETES_VERSION \
    --driver=docker \
    --addons=ingress,dashboard,storage-provisioner

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Failed to start Minikube cluster${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Cluster started successfully${NC}"

# Verify cluster
echo ""
echo -e "${BLUE}Verifying cluster...${NC}"
kubectl cluster-info
kubectl get nodes

echo ""
echo -e "${GREEN}====================================${NC}"
echo -e "${GREEN}  Minikube Setup Complete!          ${NC}"
echo -e "${GREEN}====================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Deploy Spark Optima:"
echo "     kubectl apply -f kubernetes/base/"
echo ""
echo "  2. Check deployment:"
echo "     kubectl get pods -n spark-optima"
echo ""
echo "  3. Access API (port-forward):"
echo "     kubectl port-forward -n spark-optima svc/spark-optima-api 8000:8000"
echo ""
echo "  4. Or use minikube tunnel for ingress:"
echo "     minikube tunnel -p $CLUSTER_NAME"
echo ""
echo "Useful commands:"
echo "  minikube dashboard -p $CLUSTER_NAME"
echo "  minikube stop -p $CLUSTER_NAME"
echo "  minikube delete -p $CLUSTER_NAME"
echo ""
