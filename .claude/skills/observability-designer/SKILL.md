---
name: "observability-designer"
description: "Architect comprehensive monitoring systems for production environments. Use when designing SLIs/SLOs/SLAs, building dashboards, configuring alerts, setting up distributed tracing, or implementing the three pillars of observability (metrics, logs, traces)."
---

# Observability Designer

**Tier:** POWERFUL  
**Category:** Engineering  
**Domain:** Production Monitoring / Site Reliability Engineering

## Overview

Architect comprehensive observability solutions for production systems. Combines metrics, logs, and traces with SLI/SLO design, alert optimization, and dashboard architecture. Includes three production-ready Python scripts: SLO Designer, Alert Optimizer, and Dashboard Generator.

## Core Capabilities

- **SLI/SLO/SLA Design** — define measurable health indicators, set reliability targets, calculate error budget consumption, implement multi-window burn rate alerting
- **Metrics** — Golden Signals and RED/USE methods, Prometheus instrumentation, cardinality management
- **Logs** — structured formatting with correlation IDs, log levels, sampling strategies, centralized aggregation
- **Traces** — distributed tracing with intelligent sampling, trace propagation, instrumentation overhead management
- **Dashboard Architecture** — role-based persona dashboards, 80/20 operational/exploratory split, sub-second rendering
- **Alert Optimization** — severity classification, fatigue prevention, dependent alert suppression, runbook integration

## Quick Start

```bash
python3 scripts/slo_designer.py --service my-api --slo 99.9 --window 30d
python3 scripts/alert_optimizer.py --config alerts.yaml --output optimized-alerts.yaml
python3 scripts/dashboard_generator.py --service my-api --output grafana-dashboard.json
```

## SLI/SLO/SLA Framework

### Defining SLIs (Service Level Indicators)

SLIs must be measurable, meaningful, and actionable:

```yaml
slis:
  availability:
    description: "Percentage of successful HTTP requests"
    formula: "successful_requests / total_requests * 100"
    good_event: "HTTP status < 500"
    valid_event: "All HTTP requests"

  latency:
    description: "Percentage of requests faster than threshold"
    formula: "requests_under_threshold / total_requests * 100"
    threshold_ms: 200
    percentile: p99

  error_rate:
    description: "Percentage of error-free requests"
    formula: "(1 - error_requests / total_requests) * 100"
```

### SLO Target Setting

```yaml
slos:
  availability_slo:
    sli: availability
    target: 99.9%
    window: 30d
    error_budget_minutes: 43.2  # 0.1% of 30 days

  latency_slo:
    sli: latency
    target: 95%  # 95% of requests under 200ms
    window: 30d
```

### Error Budget Calculation

```
Error Budget = (1 - SLO Target) × Window Duration
Example: (1 - 0.999) × 30 days × 24h × 60min = 43.2 minutes/month

Error Budget Consumed = Downtime / Error Budget × 100%
Alert when: Consumed > 50% with burn rate > 2x
```

### Multi-Window Burn Rate Alerting

```yaml
burn_rate_alerts:
  - name: "Critical: Fast burn"
    window_short: 1h
    window_long: 6h
    burn_rate_threshold: 14.4  # Exhausts budget in 2 hours
    severity: critical
    action: page_on_call

  - name: "Warning: Slow burn"
    window_short: 6h
    window_long: 3d
    burn_rate_threshold: 1.0
    severity: warning
    action: ticket
```

## The Three Pillars

### Metrics — Golden Signals (Google SRE)

For every service, instrument these four signals:

```prometheus
# Latency — time to serve requests
histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))

# Traffic — request volume
rate(http_requests_total[5m])

# Errors — error rate
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])

# Saturation — resource utilization
container_memory_usage_bytes / container_spec_memory_limit_bytes
```

### RED Method (Request-Driven Services)

```prometheus
# Rate — requests per second
rate(http_requests_total[1m])

# Errors — failed requests per second
rate(http_requests_total{code=~"5.."}[1m])

# Duration — request latency distribution
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
```

### USE Method (Resource Monitoring)

```prometheus
# Utilization — percent time resource is busy
rate(container_cpu_usage_seconds_total[5m]) / container_spec_cpu_quota * 100

# Saturation — extra work queued
container_memory_working_set_bytes / container_spec_memory_limit_bytes

# Errors — error events
rate(node_disk_io_time_weighted_seconds_total[5m])
```

### Structured Logging

Every log entry must include:

```json
{
  "timestamp": "2024-02-16T13:00:00.000Z",
  "level": "error",
  "service": "payment-api",
  "version": "1.2.3",
  "trace_id": "abc123",
  "span_id": "def456",
  "user_id": "user-789",
  "message": "Payment processing failed",
  "error": {
    "type": "ValidationError",
    "message": "Invalid card number",
    "stack": "..."
  },
  "duration_ms": 45,
  "http": {
    "method": "POST",
    "path": "/api/payments",
    "status": 422
  }
}
```

### Distributed Tracing

```python
# OpenTelemetry instrumentation (Python)
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

tracer_provider = TracerProvider()
tracer_provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter())
)
trace.set_tracer_provider(tracer_provider)

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("process-payment") as span:
    span.set_attribute("payment.amount", amount)
    span.set_attribute("payment.currency", currency)
    # ... processing logic
```

### Sampling Strategies

```yaml
sampling:
  default_rate: 0.1          # 10% of all requests
  error_rate: 1.0            # 100% of error requests
  slow_request_threshold: 500ms
  slow_request_rate: 1.0     # 100% of slow requests
  high_value_endpoints:
    - path: /api/payments
      rate: 1.0              # 100% of payment requests
```

## Dashboard Architecture

### Design Principles

- **80% operational metrics, 20% exploratory metrics**
- **Maximum 7±2 panels per screen** — manage cognitive load
- **Role-based personas** — different dashboards for engineers vs. executives
- **Sub-second rendering** — optimize queries, use recording rules
- **Progressive disclosure** — overview → service → instance

### Dashboard Hierarchy

```
Level 1: Executive Dashboard
├── System availability (SLO status)
├── Business metrics (transactions, revenue)
└── Incident summary

Level 2: Service Dashboard
├── Golden Signals per service
├── Error budget remaining
├── Dependency health

Level 3: Instance Dashboard
├── CPU, memory, disk per instance
├── JVM/runtime metrics
└── Application-specific metrics
```

### Grafana Dashboard Template

```json
{
  "title": "Service Overview",
  "panels": [
    {
      "title": "SLO Status",
      "type": "stat",
      "targets": [{
        "expr": "avg_over_time(slo:availability:ratio[30d]) * 100"
      }],
      "thresholds": {
        "steps": [
          {"value": 0, "color": "red"},
          {"value": 99.5, "color": "yellow"},
          {"value": 99.9, "color": "green"}
        ]
      }
    },
    {
      "title": "Request Rate",
      "type": "graph",
      "targets": [{"expr": "rate(http_requests_total[5m])"}]
    },
    {
      "title": "Error Rate",
      "type": "graph",
      "targets": [{"expr": "rate(http_requests_total{status=~'5..'}[5m])"}]
    },
    {
      "title": "P99 Latency",
      "type": "graph",
      "targets": [{
        "expr": "histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))"
      }]
    }
  ]
}
```

## Alert Optimization

### Severity Classification

```yaml
alert_severities:
  critical:
    description: "Immediate action required — customer impact now"
    response: Page on-call within 5 minutes
    examples:
      - SLO error budget burn rate > 14x
      - Service completely down
      - Data loss in progress

  warning:
    description: "Action required soon — potential customer impact"
    response: Respond within 30 minutes during business hours
    examples:
      - Error budget burn rate > 2x
      - Latency degraded but within SLO
      - Resource saturation approaching threshold

  info:
    description: "Awareness only — no immediate action required"
    response: Review during next business day
    examples:
      - Deployment completed
      - Scheduled maintenance starting
      - Capacity planning threshold crossed
```

### Alert Fatigue Prevention

```yaml
alert_hygiene:
  # Suppress dependent alerts
  inhibit_rules:
    - source_match:
        alertname: ServiceDown
      target_match:
        alertname: HighErrorRate
      equal: [service]

  # Group related alerts
  route:
    group_by: [service, alertname]
    group_wait: 30s        # Wait before sending first notification
    group_interval: 5m     # Wait before sending additional notifications
    repeat_interval: 4h    # Resend if still firing

  # Maintenance windows
  silences:
    - matchers: [{name: env, value: staging}]
      startsAt: "2024-02-16T22:00:00Z"
      endsAt: "2024-02-17T06:00:00Z"
      comment: "Scheduled maintenance"
```

### Runbook Integration

Every alert must link to a runbook:

```yaml
- alert: HighErrorRate
  expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
  labels:
    severity: warning
  annotations:
    summary: "High error rate on {{ $labels.service }}"
    description: "Error rate is {{ $value | humanizePercentage }}"
    runbook: "https://runbooks.company.com/high-error-rate"
    impact: "Users experiencing failures on {{ $labels.service }}"
    investigation: |
      1. Check recent deployments: kubectl rollout history
      2. Review error logs: https://logs.company.com/errors
      3. Check downstream dependencies
```

## Cost Optimization

### Metric Cardinality Management

High cardinality metrics are the primary cost driver:

```yaml
# BAD: unbounded cardinality
http_requests_total{user_id="12345", session_id="abc..."} 

# GOOD: bounded labels
http_requests_total{status="200", method="GET", endpoint="/api/users"}

# Cardinality limits
recording_rules:
  - record: service:http_requests:rate5m
    expr: rate(http_requests_total[5m])
    # Aggregates away high-cardinality labels
```

### Tiered Retention

```yaml
retention_tiers:
  hot:
    duration: 15d
    resolution: raw
    storage: SSD

  warm:
    duration: 90d
    resolution: 5m averages
    storage: HDD

  cold:
    duration: 2y
    resolution: 1h averages
    storage: Object storage (S3/GCS)
```

### Log Sampling

```yaml
log_sampling:
  debug: 0.01     # 1% of debug logs
  info: 0.1       # 10% of info logs
  warning: 1.0    # 100% of warnings
  error: 1.0      # 100% of errors
  # Always sample logs with trace_id for correlation
  traced_requests: 1.0
```

## Success Metrics

**Operational:**
- Mean Time to Detection (MTTD) < 5 minutes
- Mean Time to Resolution (MTTR) reduction of 30%+
- Alert precision > 80% (actionable alerts / total alerts)
- Dashboard load time < 1 second

**Business:**
- System reliability improvement (SLO achievement)
- Engineering velocity improvement through faster incident resolution
- Reduced on-call burden through alert quality

## Platforms Supported

- **Metrics:** Prometheus, Grafana, Datadog, New Relic, CloudWatch
- **Logs:** Elasticsearch/Kibana (ELK), Loki, Splunk, Datadog Logs
- **Traces:** Jaeger, Zipkin, Tempo, Datadog APM, AWS X-Ray
- **Alerting:** Alertmanager, PagerDuty, OpsGenie, Grafana Alerting
