{{/* Copyright 2024 Spark Optima Team */}}
{{/* Licensed under the Apache License, Version 2.0 */}}

{{/*
Spark Optima Helm Chart Helpers
*/}}

{{/* Expand the name of the chart */}}
{{- define "spark-optima.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/* Create a default fully qualified app name */}}
{{- define "spark-optima.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/* Create chart name and version */}}
{{- define "spark-optima.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/* Common labels */}}
{{- define "spark-optima.labels" -}}
helm.sh/chart: {{ include "spark-optima.chart" . }}
{{ include "spark-optima.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/* Selector labels */}}
{{- define "spark-optima.selectorLabels" -}}
app.kubernetes.io/name: {{ include "spark-optima.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/* API specific labels */}}
{{- define "spark-optima.api.labels" -}}
{{ include "spark-optima.labels" . }}
app.kubernetes.io/component: api
{{- end }}

{{/* API selector labels */}}
{{- define "spark-optima.api.selectorLabels" -}}
{{ include "spark-optima.selectorLabels" . }}
app.kubernetes.io/component: api
{{- end }}

{{/* CLI specific labels */}}
{{- define "spark-optima.cli.labels" -}}
{{ include "spark-optima.labels" . }}
app.kubernetes.io/component: cli-job
{{- end }}

{{/* Create the name of the API service account */}}
{{- define "spark-optima.api.serviceAccountName" -}}
{{- if .Values.serviceAccount.api.create }}
{{- default (printf "%s-api" (include "spark-optima.fullname" .)) .Values.serviceAccount.api.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.api.name }}
{{- end }}
{{- end }}

{{/* Create the name of the CLI service account */}}
{{- define "spark-optima.cli.serviceAccountName" -}}
{{- if .Values.serviceAccount.cli.create }}
{{- default (printf "%s-cli" (include "spark-optima.fullname" .)) .Values.serviceAccount.cli.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.cli.name }}
{{- end }}
{{- end }}

{{/* Get the image tag */}}
{{- define "spark-optima.imageTag" -}}
{{- if .Values.image.tag }}
{{- .Values.image.tag }}
{{- else if .Values.api.image.tag }}
{{- .Values.api.image.tag }}
{{- else }}
{{- .Chart.AppVersion }}
{{- end }}
{{- end }}

{{/* Get the API image repository */}}
{{- define "spark-optima.api.image" -}}
{{- $repo := default .Values.image.repository .Values.api.image.repository }}
{{- $tag := include "spark-optima.imageTag" . }}
{{- printf "%s:%s" $repo $tag }}
{{- end }}

{{/* Get the CLI image repository */}}
{{- define "spark-optima.cli.image" -}}
{{- $repo := default .Values.image.repository .Values.cli.image.repository }}
{{- $tag := include "spark-optima.imageTag" . }}
{{- printf "%s:%s" $repo $tag }}
{{- end }}

{{/* ConfigMap name */}}
{{- define "spark-optima.configMapName" -}}
{{- printf "%s-config" (include "spark-optima.fullname" .) }}
{{- end }}

{{/* PVC names */}}
{{- define "spark-optima.pvc.dataName" -}}
{{- printf "%s-data" (include "spark-optima.fullname" .) }}
{{- end }}

{{- define "spark-optima.pvc.logsName" -}}
{{- printf "%s-logs" (include "spark-optima.fullname" .) }}
{{- end }}
