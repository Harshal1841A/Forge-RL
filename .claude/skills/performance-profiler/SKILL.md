---
name: "performance-profiler"
description: "Systematically profile and optimize Node.js, Python, and Go applications. Use when the app is slow, P99 latency exceeds SLA, memory grows over time, bundle size increased, or before a traffic spike. Always measures before and after optimizations."
---

# Performance Profiler

**Tier:** POWERFUL  
**Category:** Engineering  
**Domain:** Performance Engineering  

## Overview

Systematic performance profiling for Node.js, Python, and Go applications. Identifies CPU, memory, and I/O bottlenecks; generates flamegraphs; analyzes bundle sizes; optimizes database queries; detects memory leaks; and runs load tests with k6 and Artillery. Always measures before and after.

## Core Capabilities

- **CPU profiling** — flamegraphs for Node.js, py-spy for Python, pprof for Go
- **Memory profiling** — heap snapshots, leak detection, GC pressure
- **Bundle analysis** — webpack-bundle-analyzer, Next.js bundle analyzer
- **Database optimization** — EXPLAIN ANALYZE, slow query log, N+1 detection
- **Load testing** — k6 scripts, Artillery scenarios, ramp-up patterns
- **Before/after measurement** — establish baseline, profile, optimize, verify

## When to Use

- App is slow and you don't know where the bottleneck is
- P99 latency exceeds SLA before a release
- Memory usage grows over time (suspected leak)
- Bundle size increased after adding dependencies
- Preparing for a traffic spike (load test before launch)
- Database queries taking >100ms

## Quick Start

```bash
python3 scripts/performance_profiler.py /path/to/project
python3 scripts/performance_profiler.py /path/to/project --json
python3 scripts/performance_profiler.py /path/to/project --large-file-threshold-kb 256
```

## Golden Rule: Measure First

Establish baseline metrics before optimizations. The prescribed approach:

**Profile → confirm bottleneck → fix → measure again → verify improvement**

Never implement fixes based on assumptions. Always:
1. Establish P50, P95, P99 latency baseline
2. Profile to find the actual bottleneck
3. Apply a single focused fix
4. Re-measure to confirm improvement
5. Document before/after numbers in PRs

## CPU Profiling

### Node.js
```bash
# Built-in profiler
node --prof app.js
node --prof-process isolate-*.log > processed.txt

# Clinic.js (recommended)
npx clinic flame -- node app.js
npx clinic doctor -- node app.js
```

### Python
```bash
# py-spy for low-overhead sampling
py-spy record -o profile.svg --pid <PID>
py-spy top --pid <PID>

# cProfile for function-level detail
python -m cProfile -o output.prof app.py
python -m pstats output.prof
```

### Go
```bash
# pprof built-in
go test -cpuprofile cpu.prof -memprofile mem.prof ./...
go tool pprof cpu.prof

# HTTP endpoint profiling
import _ "net/http/pprof"
go tool pprof http://localhost:6060/debug/pprof/profile
```

## Memory Profiling

### Heap Snapshots (Node.js)
```bash
# Take heap snapshot
node --inspect app.js
# Open Chrome DevTools → Memory → Take snapshot

# Detect leaks with memwatch-next
npm install memwatch-next
```

### Python Memory
```bash
pip install memory-profiler
python -m memory_profiler app.py

# tracemalloc for allocation tracking
import tracemalloc
tracemalloc.start()
# ... code ...
snapshot = tracemalloc.take_snapshot()
```

## Bundle Analysis

### Webpack
```bash
npm install --save-dev webpack-bundle-analyzer
npx webpack-bundle-analyzer stats.json
```

### Next.js
```bash
npm install --save-dev @next/bundle-analyzer
ANALYZE=true npm run build
```

### Common Quick Wins
- Replace `moment.js` with `date-fns` or `dayjs`
- Use selective imports: `import { debounce } from 'lodash'` not `import _ from 'lodash'`
- Dynamic import heavy components: `const Chart = dynamic(() => import('./Chart'))`
- Optimize images with next/image or sharp
- Route-based code splitting

## Database Optimization

### EXPLAIN ANALYZE (PostgreSQL)
```sql
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) 
SELECT * FROM users WHERE email = 'test@example.com';
```

### Common Issues
- **Missing indexes** — check sequential scans on large tables
- **N+1 queries** — use eager loading or DataLoader patterns
- **SELECT *** — select only needed columns
- **Unbounded queries** — always add LIMIT
- **Connection pool exhaustion** — tune pool size to match workload

### Slow Query Log (MySQL/PostgreSQL)
```sql
-- PostgreSQL
ALTER SYSTEM SET log_min_duration_statement = '100ms';
SELECT * FROM pg_stat_statements ORDER BY total_time DESC LIMIT 20;
```

## Load Testing

### k6
```javascript
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  stages: [
    { duration: '2m', target: 100 },   // ramp up
    { duration: '5m', target: 100 },   // stay at 100 users
    { duration: '2m', target: 0 },     // ramp down
  ],
  thresholds: {
    http_req_duration: ['p(99)<500'],   // 99% under 500ms
    http_req_failed: ['rate<0.01'],     // <1% errors
  },
};

export default function () {
  let res = http.get('http://test.k6.io');
  check(res, { 'status was 200': (r) => r.status === 200 });
  sleep(1);
}
```

### Artillery
```yaml
config:
  target: 'http://localhost:3000'
  phases:
    - duration: 60
      arrivalRate: 10
      rampTo: 50
      name: Ramp up

scenarios:
  - name: API test
    flow:
      - get:
          url: '/api/users'
          expect:
            - statusCode: 200
```

## Optimization Checklist

### Database Priorities
- [ ] Missing indexes on filter/join columns
- [ ] N+1 queries (use query logs or ORM debug mode)
- [ ] SELECT * statements on large tables
- [ ] Unbounded queries without LIMIT
- [ ] Connection pooling properly configured

### Node.js Priorities
- [ ] Synchronous I/O in critical paths (fs.readFileSync, etc.)
- [ ] Large object serialization in hot loops
- [ ] Missing computation caching (memoization)
- [ ] Uncompressed responses (enable gzip/brotli)
- [ ] Module-level heavy dependency loading

### Bundle Priorities
- [ ] Heavy library replacements (moment → dayjs)
- [ ] Full vs. selective imports (lodash, etc.)
- [ ] Dynamic loading opportunities for heavy components
- [ ] Image optimization and modern formats (WebP, AVIF)
- [ ] Route-based code splitting

### API Performance
- [ ] Pagination on list endpoints
- [ ] Cache-Control headers on stable resources
- [ ] Parallel data fetching (Promise.all)
- [ ] Response compression
- [ ] CDN for static assets

## Common Pitfalls

- Optimizing without baseline measurement
- Testing against development-scale data (must use production-representative datasets)
- Neglecting P99 percentile analysis (focus only on averages)
- Skipping verification of implemented fixes
- Load testing production environments directly
- Making multiple changes at once (can't isolate the win)

## Best Practices

1. Establish baseline metrics consistently before any change
2. Isolate variables — one change at a time
3. Profile against production-volume datasets
4. Integrate performance thresholds into CI (fail build if P99 regresses)
5. Implement continuous monitoring in production
6. Design deliberate cache invalidation strategies
7. Document optimization results with before/after numbers in PRs
8. Set performance budgets and enforce them

## CI Integration Example

```yaml
# GitHub Actions performance gate
- name: Load test
  run: |
    k6 run --out json=results.json load-test.js
    # Fail if P99 > 500ms
    cat results.json | jq '.metrics.http_req_duration.values["p(99)"] < 500' | grep true
```
