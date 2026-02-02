# AI Agent Instructions for AI Dynamo: The Hard Way

> **Reference**: See [README.md](README.md) for architecture diagrams, learning modules, and hardware setup details.

## Learner Profile

**Background**: Systems engineering (not ML expert)
**Existing Knowledge**: 
- LLM serving basics
- KV cache fundamentals
- Strong in: distributed systems, networking, infrastructure

**Learning Style**: Build progressively, understand each component before moving to the next.

---

## Core Philosophy

> Build everything from scratch. No magic abstractions. Understand the "why" before the "how".

When helping this learner:
1. **Map ML concepts to systems concepts** - e.g., "KV cache is like a session cache in web servers"
2. **Start with working minimal code** - Then add complexity incrementally
3. **Explain distributed systems aspects deeply** - This is their strength
4. **Demystify ML jargon** - Use plain systems engineering language
5. **Establish baselines before showing optimizations** - Can't appreciate "after" without "before"

---

## Baseline Comparison Pedagogy

> You can't appreciate an optimization without seeing what you're optimizing from.

### Why Baselines Matter

This is how systems engineers evaluate any infrastructure change:
- "We added Redis caching → 3x faster"
- "We switched to connection pooling → 50% latency reduction"
- "We implemented disaggregated serving → 2x throughput"

### Making the Baseline Fair

To ensure honest comparison, the baseline should be the *best you can do* without Dynamo:
- Use vLLM's continuous batching (not naive sequential)
- Use a decent load balancer (not just random)
- Tune batch sizes appropriately

This way, when Dynamo wins, it's because of **architectural advantages** (disaggregation, KV-aware routing), not because the baseline was poorly configured.

---

## Dynamo Concept Mappings

When explaining Dynamo components, use these systems engineering equivalents:

| Dynamo Concept | Systems Engineering Equivalent           |
| -------------- | ---------------------------------------- |
| Frontend       | HTTP reverse proxy (like Nginx)          |
| Router         | Load balancer with session affinity      |
| Worker         | Backend service instance                 |
| etcd           | Service registry (like Consul/ZooKeeper) |
| NATS           | Message queue (like Kafka/RabbitMQ)      |
| KV Cache       | In-memory cache (like Redis) per request |
| NIXL           | RDMA-based cache replication             |
| Prefill Phase  | Request parsing + cache warmup           |
| Decode Phase   | Response generation (streaming)          |

---

## Code Creation Guidelines

### When Writing Examples

```python
# ALWAYS include these elements:

# 1. Imports with purpose comments
import dynamo  # Core framework

# 2. Configuration as explicit variables (not hidden)
ETCD_ENDPOINTS = ["http://localhost:2379"]
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# 3. Step-by-step execution with print statements
print("Step 1: Connecting to etcd...")

# 4. Error handling that explains what went wrong
try:
    client.connect()
except ConnectionError as e:
    print(f"Failed to connect. Is etcd running? Error: {e}")

# 5. Cleanup code
finally:
    client.close()
```

### Notebook Structure

Each notebook should follow:

```
1. Learning Objectives (what systems concept we're exploring)
2. Prerequisites Check (verify environment)
3. Concept Explanation (with systems analogy)
4. Minimal Working Example
5. Incremental Additions
6. Troubleshooting Guide
7. Exercises (hands-on tasks)
8. Key Takeaways
```

### Code Comments Style

Prefer systems engineering language:

```python
# BAD: "Initialize the attention mechanism's key-value cache"
# GOOD: "Allocate memory buffer for caching intermediate computations (like connection pooling)"

# BAD: "Set up the inference endpoint"  
# GOOD: "Start HTTP server on port 8000, register with service discovery"
```

---

## Key Concepts to Explain Progressively

### 1. KV Cache (From Systems Perspective)

Explain as:
> "When processing a prompt, the model computes attention scores. These are expensive to compute but reusable. The KV cache stores these like a memoization table. Each token adds ~1MB to the cache for a 7B model."

Build understanding:
1. Show memory growth as prompt length increases
2. Demonstrate cache hit when same prefix is reused
3. Visualize cache as key-value store (position → tensor)

### 2. Disaggregated Serving

Explain as:
> "Like separating web servers (compute-heavy parsing) from API servers (I/O-heavy responses). Prefill is CPU/memory intensive, decode is latency-sensitive."

Build understanding:
1. **First, show baseline** - Two nodes each doing full prefill+decode
2. Measure prefill time vs decode time separately
3. Show how they can run on different hardware
4. Demonstrate KV cache transfer between phases
5. **Compare metrics** - Same hardware, different architecture

Why baseline matters here:
- Without seeing prefill blocking decode, learner won't appreciate specialization
- Without measuring KV cache misses, learner won't understand routing value
- Without observing uneven GPU utilization, learner won't see resource waste

When disaggregation helps most:
- **Long prompts**: Big win (prefill dominates, specialization pays off)
- **Mixed workloads**: Dynamic routing shines
- **High concurrency**: Batching optimization per phase

When disaggregation might not help:
- **Very short prompts**: Transfer overhead might exceed benefit
- **Single user, low concurrency**: Overhead of coordination
- **Latency-critical single requests**: Added network hop

### 3. NIXL and RDMA

Explain as:
> "RDMA bypasses the kernel network stack - GPU memory to GPU memory directly. NIXL is Dynamo's library for this. Like DMA but across network."

Build understanding:
1. Compare TCP transfer speed vs RDMA
2. Show zero-copy semantics
3. Trace a KV cache block transfer

### 4. Service Discovery with etcd

Explain as:
> "Workers register themselves with endpoints and capabilities. Frontend watches for changes. Like DNS but with health checks and metadata."

Build understanding:
1. Manual registration with etcdctl
2. Watch mechanism for dynamic discovery
3. Lease-based TTL for failure detection

### 5. Routing Strategies

Explain as:
> "Router decides which worker handles a request. KV-aware routing sends requests with same prefix to same worker (session affinity for cache hits)."

Build understanding:
1. Random routing (baseline)
2. Round-robin (fair distribution)
3. Prefix-hash routing (cache-aware)
4. Load-based routing (capacity-aware)

---

## Learning Path

See [README.md](README.md#-learning-modules) for the complete module structure and progression.

---

## Troubleshooting Patterns

When things go wrong, check in order:

1. **Network connectivity** - Can services reach each other?
2. **Service registration** - Is the service in etcd?
3. **GPU memory** - Is there OOM killing processes?
4. **CUDA version** - Do all components match CUDA version?
5. **Model loading** - Is the model downloaded and accessible?

### Common Issues

| Symptom            | Likely Cause             | Debug Command                          |
| ------------------ | ------------------------ | -------------------------------------- |
| Worker not found   | etcd registration failed | `etcdctl get --prefix /dynamo`         |
| Slow inference     | KV cache miss            | Check routing logs for prefix matching |
| Connection refused | Service not started      | `ss -tlnp \| grep <port>`              |
| CUDA OOM           | Model too large          | `nvidia-smi` during load               |
| RDMA failed        | InfiniBand misconfigured | `ibstat`, `ibv_devinfo`                |

---

## Preferred Tools and Approaches

### For Exploration
- `curl` for HTTP endpoints
- `etcdctl` for service discovery debugging
- `nats-cli` for messaging debugging
- `nvidia-smi` for GPU monitoring
- `htop` for CPU/memory monitoring

### For Code
- Python with type hints
- Minimal dependencies
- Explicit configuration (no environment variable magic)
- Rich logging with structured output

### For Visualization
- ASCII diagrams for architecture
- Mermaid for sequence diagrams
- Simple matplotlib for metrics

---

## Questions to Ask Before Each Module

1. What systems concept does this map to?
2. What's the minimal code to demonstrate this?
3. What can go wrong and how do we debug it?
4. How does this connect to what we've built before?
5. What performance characteristics should we observe?

---

## Success Metrics

The learner has succeeded when they can:

1. **Explain** each component's role using systems terminology
2. **Debug** a failing Dynamo deployment systematically
3. **Predict** performance characteristics based on configuration
4. **Extend** the system with custom routing logic
5. **Optimize** for their specific hardware setup

---

## Notes for AI Agent

When generating code or explanations:

1. **Always test connectivity before complex operations**
2. **Show the "before" state, not just the "after"**
3. **Include timing measurements** - systems engineers love benchmarks
4. **Explain resource usage** - CPU, memory, GPU, network
5. **Reference official Dynamo source when explaining internals**
6. **Never skip error handling** - real systems fail
7. **Establish baselines before optimizations** - always measure "without" before "with"

### Baseline Comparison Guidelines

When demonstrating Dynamo benefits:
1. **Fair baseline**: Use best-practice vLLM setup (continuous batching, proper tuning)
2. **Same hardware**: Baseline and Dynamo tests must use identical hardware
3. **Same workload**: Use identical request patterns for both tests
4. **Explain the "why"**: Don't just show numbers, explain what causes the difference

When the learner is stuck:
1. Check if it's an environment issue first
2. Provide diagnostic commands
3. Simplify to minimal reproduction
4. Relate back to known systems concepts
