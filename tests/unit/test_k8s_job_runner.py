# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for the Kubernetes Job runner manifests, Helm templates, and docs."""

from __future__ import annotations

from pathlib import Path

import yaml


def test_optimization_job_manifest_exists() -> None:
    """The raw Kubernetes Job manifest exists in base/."""
    manifest_path = Path(__file__).parent.parent.parent / "kubernetes" / "base" / "job-optimization.yaml"
    assert manifest_path.exists(), f"Manifest missing: {manifest_path}"


def test_optimization_job_manifest_is_valid_yaml() -> None:
    """The raw manifest is well-formed YAML."""
    manifest_path = Path(__file__).parent.parent.parent / "kubernetes" / "base" / "job-optimization.yaml"
    content = manifest_path.read_text()
    data = yaml.safe_load(content)
    assert data is not None
    assert data.get("kind") == "Job"
    assert data.get("apiVersion") == "batch/v1"
    assert "redis" in content.lower()  # References Redis job store


def test_optimization_job_helm_template_exists() -> None:
    """The Helm template for optimization jobs exists."""
    template_path = (
        Path(__file__).parent.parent.parent
        / "kubernetes"
        / "helm"
        / "spark-optima"
        / "templates"
        / "job-optimization.yaml"
    )
    assert template_path.exists(), f"Helm template missing: {template_path}"


def test_optimization_job_helm_template_has_redis_env() -> None:
    """The Helm template configures SPARK_OPTIMA_JOB_STORE=redis."""
    template_path = (
        Path(__file__).parent.parent.parent
        / "kubernetes"
        / "helm"
        / "spark-optima"
        / "templates"
        / "job-optimization.yaml"
    )
    content = template_path.read_text()
    assert "SPARK_OPTIMA_JOB_STORE" in content
    assert "redis" in content


def test_optimization_values_include_optimization_job() -> None:
    """values.yaml contains the optimizationJob settings."""
    values_path = Path(__file__).parent.parent.parent / "kubernetes" / "helm" / "spark-optima" / "values.yaml"
    content = values_path.read_text()
    assert "optimizationJob:" in content
    assert "enabled:" in content
    assert "redisUrl:" in content


def test_optimization_doc_exists() -> None:
    """Platform-team documentation exists."""
    doc_path = Path(__file__).parent.parent.parent / "docs" / "user-guide" / "k8s-job-runner.md"
    assert doc_path.exists(), f"Doc missing: {doc_path}"


def test_optimization_doc_has_redis_reference() -> None:
    """The platform doc references Redis and the job store."""
    doc_path = Path(__file__).parent.parent.parent / "docs" / "user-guide" / "k8s-job-runner.md"
    content = doc_path.read_text()
    assert "redis" in content.lower()
    assert "SPARK_OPTIMA_REDIS_URL" in content
    assert "SPARK_OPTIMA_JOB_STORE" in content


def test_production_guide_updated() -> None:
    """PRODUCTION.md includes Kubernetes Job runner documentation."""
    prod_path = Path(__file__).parent.parent.parent / "kubernetes" / "PRODUCTION.md"
    content = prod_path.read_text()
    assert "Kubernetes Job Runner" in content
    assert "SPARK_OPTIMA_JOB_STORE=redis" in content
