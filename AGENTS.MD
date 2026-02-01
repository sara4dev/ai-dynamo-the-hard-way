# AI Agent Instructions for AI Dynamo: The Hard Way

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

---

## Progressive Code Creation Strategy

### Level 0: Understand the Mental Model

Before any code, establish these mappings:

| Dynamo Concept | Systems Engineering Equivalent |
|----------------|-------------------------------|
| Frontend | HTTP reverse proxy (like Nginx) |
| Router | Load balancer with session affinity |
| Worker | Backend service instance |
| etcd | Service registry (like Consul/ZooKeeper) |
| NATS | Message queue (like Kafka/RabbitMQ) |
| KV Cache | In-memory cache (like Redis) per request |
| NIXL | RDMA-based cache replication |
| Prefill Phase | Request parsing + cache warmup |
| Decode Phase | Response generation (streaming) |

### Level 1: Single Process Understanding

**Goal**: Run inference locally, understand request/response flow.

```python
# Start with the simplest possible inference
# No Dynamo, just raw model serving
```

Build in this order:
1. Load a small model (e.g., TinyLlama)
2. Send a single request, trace what happens
3. Observe GPU memory, understand KV cache growth
4. Add streaming response handling

### Level 2: Component Isolation

**Goal**: Understand each Dynamo component independently.

Order of exploration:
1. **etcd** - Service registration/discovery patterns
2. **NATS** - Pub/sub messaging, JetStream persistence
3. **Frontend** - HTTP handling, OpenAI API translation
4. **Worker** - Model loading, request processing

For each component, create:
- Standalone test script
- Health check utility
- Manual registration/interaction code

### Level 3: Integration Patterns

**Goal**: Connect components manually (no orchestration).

Build connections:
1. Worker → etcd (register itself)
2. Frontend → etcd (discover workers)
3. Router → NATS (receive routing hints)
4. Worker → Worker (KV cache events)

### Level 4: Distributed Patterns

**Goal**: Scale across nodes, understand RDMA benefits.

Focus areas:
1. Why InfiniBand matters for KV cache transfer
2. Disaggregated serving: separate prefill from decode
3. Pipeline parallelism: split model across GPUs
4. Tensor parallelism: parallel matrix operations

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
1. Measure prefill time vs decode time separately
2. Show how they can run on different hardware
3. Demonstrate KV cache transfer between phases

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

## Incremental Build Path

### Week 1: Local Foundations
```
Day 1: Environment setup, first inference with vLLM
Day 2: Understand KV cache through memory monitoring
Day 3: Manual etcd operations
Day 4: Manual NATS operations  
Day 5: Connect vLLM worker to etcd
```

### Week 2: Single-Node Dynamo
```
Day 1: Frontend HTTP handling
Day 2: Basic round-robin routing
Day 3: Add worker health checks
Day 4: Implement KV-aware routing
Day 5: Full single-node deployment
```

### Week 3: Multi-Node Distributed
```
Day 1: InfiniBand verification
Day 2: Cross-node worker registration
Day 3: NIXL setup and testing
Day 4: Disaggregated prefill/decode
Day 5: End-to-end distributed inference
```

---

## Troubleshooting Patterns

When things go wrong, check in order:

1. **Network connectivity** - Can services reach each other?
2. **Service registration** - Is the service in etcd?
3. **GPU memory** - Is there OOM killing processes?
4. **CUDA version** - Do all components match CUDA version?
5. **Model loading** - Is the model downloaded and accessible?

### Common Issues

| Symptom | Likely Cause | Debug Command |
|---------|--------------|---------------|
| Worker not found | etcd registration failed | `etcdctl get --prefix /dynamo` |
| Slow inference | KV cache miss | Check routing logs for prefix matching |
| Connection refused | Service not started | `ss -tlnp \| grep <port>` |
| CUDA OOM | Model too large | `nvidia-smi` during load |
| RDMA failed | InfiniBand misconfigured | `ibstat`, `ibv_devinfo` |

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

When the learner is stuck:
1. Check if it's an environment issue first
2. Provide diagnostic commands
3. Simplify to minimal reproduction
4. Relate back to known systems concepts
