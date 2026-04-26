---
name: "agent-workflow-designer"
description: "Design production-grade multi-agent workflows with clear pattern choice, handoff contracts, failure handling, and cost/context controls. Use when a single prompt is insufficient for task complexity, when you need specialist agents with explicit boundaries, or when building deterministic multi-step agent pipelines."
---

# Agent Workflow Designer

**Tier:** POWERFUL  
**Category:** Engineering  
**Tags:** multi-agent, orchestration, workflow, LLM pipelines, agent patterns

## Overview

Agent Workflow Designer enables teams to design production-grade multi-agent workflows through structured orchestration patterns. It provides skeleton config generation for fast workflow bootstrapping, pattern selection guidance, handoff contract definitions, retry/timeout policies, and cost/context controls.

Use this skill when a single prompt is insufficient for task complexity, or when you require specialist agents with explicit boundaries and deterministic structure before implementation.

## Core Workflow Patterns

### 1. Sequential
Step-by-step dependency chains where each agent hands off to the next.

- **Use Case:** Tasks with strict ordering, where each step depends on the previous output
- **Structure:** Agent A → Agent B → Agent C
- **Handoff:** Explicit output schema passed as input to next step
- **Failure Handling:** Halt on any step failure; retry from failed step

### 2. Parallel (Fan-Out / Fan-In)
Independent work distributed across multiple agents, aggregated at the end.

- **Use Case:** Tasks that can be decomposed into independent subtasks
- **Structure:** Orchestrator → [Agent A, Agent B, Agent C] → Aggregator
- **Handoff:** Merge results with defined aggregation strategy
- **Failure Handling:** Partial results acceptable or require all to succeed

### 3. Router (Intent-Based Dispatch)
Intent classification routes requests to the appropriate specialist agent with fallback.

- **Use Case:** Variable input types requiring different processing paths
- **Structure:** Router Agent → [Specialist A | Specialist B | Fallback]
- **Handoff:** Route decision with confidence score
- **Failure Handling:** Fallback agent handles unclassified or low-confidence inputs

### 4. Orchestrator (Planner + Coordinator)
A planner agent decomposes tasks and coordinates specialist agents dynamically.

- **Use Case:** Complex tasks requiring dynamic planning and resource allocation
- **Structure:** Planner → [Dynamic specialist selection] → Result synthesis
- **Handoff:** Task decomposition plan with assigned agents
- **Failure Handling:** Replanning on failure, escalation on repeated failures

### 5. Evaluator (Generator + Quality Gate)
A generator agent produces output that an evaluator agent checks before acceptance.

- **Use Case:** High-quality output requirements with measurable acceptance criteria
- **Structure:** Generator → Evaluator → [Accept | Regenerate loop]
- **Handoff:** Output + evaluation rubric + pass/fail decision
- **Failure Handling:** Max iteration limit; escalate to human review on loop exhaustion

## Implementation Approach

Follow this progression for building robust workflows:

1. **Select pattern** — choose the smallest pattern that can satisfy requirements
2. **Scaffold config** — generate skeleton configuration files for the chosen pattern
3. **Define handoff contracts** — specify input/output schemas between each step
4. **Add resilience policies** — configure retry logic, timeouts, and fallbacks
5. **Validate with small budget** — test with constrained token/cost limits before scaling

## Handoff Contract Specification

Every agent-to-agent handoff should define:

```yaml
handoff:
  from: agent-name
  to: next-agent-name
  payload:
    required:
      - field: result
        type: string
        description: Primary output of this step
    optional:
      - field: confidence
        type: float
        description: Confidence score 0-1
  on_failure:
    strategy: retry | skip | abort | fallback
    max_retries: 3
    fallback_agent: fallback-agent-name
```

## Resilience Policies

### Timeout Configuration
Every agent step must have an explicit timeout:

```yaml
timeout:
  step_timeout_seconds: 30
  workflow_timeout_seconds: 300
  on_timeout: abort | partial_result
```

### Retry Policy
```yaml
retry:
  max_attempts: 3
  backoff: exponential
  initial_delay_ms: 1000
  max_delay_ms: 30000
  retryable_errors:
    - rate_limit
    - timeout
    - transient_api_error
```

### Circuit Breaker
```yaml
circuit_breaker:
  failure_threshold: 5
  recovery_timeout_seconds: 60
  half_open_attempts: 2
```

## Cost and Context Controls

Track and enforce resource limits across the entire workflow:

```yaml
budget:
  max_total_tokens: 50000
  max_cost_usd: 1.00
  per_step_token_limit: 8000
  on_budget_exceeded: abort | use_cheaper_model | truncate_context
```

### Context Window Management
- Pass only necessary upstream context to each step — avoid forwarding entire conversation history
- Summarize intermediate results before passing to next agent
- Define explicit context boundaries per step

## Common Pitfalls

1. **Over-orchestrating** — don't use multi-agent workflows for tasks solvable by one well-structured prompt
2. **Missing timeouts** — every step must have an explicit timeout; silent hangs will stall entire workflows
3. **Excessive context forwarding** — passing unnecessary upstream context inflates costs and degrades focus
4. **No cost tracking** — failing to monitor cumulative step costs leads to runaway spend
5. **Skipping validation** — always validate intermediate results before passing to downstream agents
6. **Ignoring partial failure** — define explicitly whether partial results are acceptable or require full success

## Best Practices

1. Start with the smallest pattern that satisfies requirements
2. Define explicit payload boundaries for every handoff
3. Validate intermediate results at each step boundary
4. Enforce resource constraints (tokens, cost, time) throughout execution
5. Test with constrained budgets before production scaling
6. Document the blast radius of each agent's failure mode
7. Use structured output schemas (JSON) for all agent outputs
8. Monitor cumulative costs across all workflow steps

## Scaffold Configuration Example

```yaml
workflow:
  name: research-and-summarize
  pattern: sequential
  agents:
    - id: researcher
      role: "Search and retrieve relevant information"
      tools: [web_search, document_retrieval]
      output_schema: {findings: string[], sources: string[]}
      timeout_seconds: 60
      retry:
        max_attempts: 2

    - id: synthesizer
      role: "Synthesize findings into coherent summary"
      input_from: researcher
      context_fields: [findings]  # Only pass findings, not sources
      output_schema: {summary: string, key_points: string[]}
      timeout_seconds: 30

  budget:
    max_total_tokens: 20000
    on_exceeded: abort

  on_failure:
    strategy: abort
    notify: true
```
