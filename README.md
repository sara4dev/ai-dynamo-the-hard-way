# AI Dynamo: The Hard Way

A progressive, hands-on approach to learning [NVIDIA AI Dynamo](https://github.com/ai-dynamo/dynamo) - a Datacenter Scale Distributed Inference Serving Framework.

> **Philosophy**: No Kubernetes operators. No magic. Just understanding each component from the ground up.
>
> **Key Learning Approach**: First measure baseline performance (without Dynamo), then demonstrate improvements with Dynamo. You can't appreciate optimizations without understanding what you're optimizing from.

## 🖥️ Hardware Setup

| Node             | Hardware   | Purpose                             |
| ---------------- | ---------- | ----------------------------------- |
| **dgx-spark-01** | DGX Spark  | Primary node, Frontend, Workers     |
| **dgx-spark-02** | DGX Spark  | Secondary node, Distributed workers |
| **Network**      | InfiniBand | RDMA for NIXL KV cache transfer     |

This setup is ideal for learning Dynamo because its key innovations (disaggregated serving, NIXL, cross-node inference) require multi-node + InfiniBand.

## 🏗️ Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                        AI Dynamo Architecture                      │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐    │
│  │   Clients    │────▶│   Frontend   │────▶│     Router       │    │
│  │  (OpenAI API)│     │  (Rust HTTP) │     │ (Basic/KV-Aware) │    │
│  └──────────────┘     └──────────────┘     └────────┬─────────┘    │
│                                                     │              │
│                                                     ▼              │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │                          Workers                           │    │
│  │  ┌───────────┐    ┌───────────────┐    ┌───────────────┐   │    │
│  │  │  SGLang   │    │  TensorRT-LLM │    │     vLLM      │   │    │
│  │  └───────────┘    └───────────────┘    └───────────────┘   │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                       Infrastructure                         │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐   │  │
│  │  │    etcd     │    │    NATS     │    │   NIXL (RDMA)   │   │  │
│  │  │  (Service   │    │ (Messaging/ │    │   (KV Cache     │   │  │
│  │  │  Discovery) │    │  JetStream) │    │    Transfer)    │   │  │
│  │  └─────────────┘    └─────────────┘    └─────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

## 📚 Learning Modules

### Part 1: Foundations (Single Node - dgx-spark-01)

| Module | Notebook                                                                           | Description                                  |
| ------ | ---------------------------------------------------------------------------------- | -------------------------------------------- |
| **00** | [00-architecture-overview.ipynb](notebooks/00-architecture-overview.ipynb)         | Understand Dynamo's components and data flow |
| **01** | [01-setup-and-first-inference.ipynb](notebooks/01-setup-and-first-inference.ipynb) | Install Dynamo, first inference request      |
| **02** | [02-frontend-deep-dive.ipynb](notebooks/02-frontend-deep-dive.ipynb)               | Rust HTTP server, OpenAI compatibility       |
| **03** | [03-workers-and-backends.ipynb](notebooks/03-workers-and-backends.ipynb)           | SGLang, vLLM, TensorRT-LLM comparison        |

### Part 2: Infrastructure (Single Node)

| Module | Notebook                                                                     | Description                             |
| ------ | ---------------------------------------------------------------------------- | --------------------------------------- |
| **04** | [04-etcd-service-discovery.ipynb](notebooks/04-etcd-service-discovery.ipynb) | Manual etcd setup, service registration |
| **05** | [05-nats-messaging.ipynb](notebooks/05-nats-messaging.ipynb)                 | NATS JetStream for KV cache events      |
| **06** | [06-kv-aware-routing.ipynb](notebooks/06-kv-aware-routing.ipynb)             | Prefix caching, smart request routing   |

### Part 3: Distributed Inference (Both DGX Spark Nodes)

| Module | Notebook                                                                           | Description                                     |
| ------ | ---------------------------------------------------------------------------------- | ----------------------------------------------- |
| **07** | [07-infiniband-setup.ipynb](notebooks/07-infiniband-setup.ipynb)                   | Verify InfiniBand, RDMA configuration           |
| **08** | [08-multi-node-workers.ipynb](notebooks/08-multi-node-workers.ipynb)               | Workers across nodes, pipeline parallelism      |
| **09** | [09-baseline-two-node-serving.ipynb](notebooks/09-baseline-two-node-serving.ipynb) | **Baseline**: Two vLLM nodes without Dynamo     |
| **10** | [10-disaggregated-serving.ipynb](notebooks/10-disaggregated-serving.ipynb)         | **With Dynamo**: Same nodes, disaggregated mode |
| **11** | [11-nixl-kv-transfer.ipynb](notebooks/11-nixl-kv-transfer.ipynb)                   | RDMA-based KV cache transfer                    |

### Part 4: Production Patterns

| Module | Notebook                                                                     | Description                             |
| ------ | ---------------------------------------------------------------------------- | --------------------------------------- |
| **12** | [12-benchmarking.ipynb](notebooks/12-benchmarking.ipynb)                     | AIPerf, latency analysis, throughput    |
| **13** | [13-large-model-deployment.ipynb](notebooks/13-large-model-deployment.ipynb) | DeepSeek-R1, Llama-3-70B across cluster |

## 🚀 Quick Start

```bash
# Clone this repository
git clone https://github.com/sara4dev/ai-dynamo-the-hard-way.git
cd ai-dynamo-the-hard-way

# Start with Module 00
jupyter lab notebooks/00-architecture-overview.ipynb
```

## 📋 Prerequisites

- **Python**: 3.10+
- **CUDA**: 12.x+
- **Rust**: Latest stable (for building from source)
- **InfiniBand**: Configured between DGX Spark nodes
- **SSH**: Passwordless SSH between nodes

## 🔗 Key Resources

- [AI Dynamo GitHub](https://github.com/ai-dynamo/dynamo)
- [Official Documentation](https://docs.nvidia.com/dynamo/latest)
- [Dynamo v0.8.1 Release](https://github.com/ai-dynamo/dynamo/releases/tag/v0.8.1) (Latest as of Jan 2026)

## 📁 Project Structure

```
ai-dynamo-the-hard-way/
├── README.md
├── notebooks/              # Jupyter notebooks for each module
│   ├── 00-architecture-overview.ipynb
│   ├── 01-setup-and-first-inference.ipynb
│   └── ...
├── scripts/                # Helper scripts
│   ├── install-dynamo.sh
│   ├── start-etcd.sh
│   ├── start-nats.sh
│   └── ...
├── configs/                # Configuration files
│   ├── etcd/
│   ├── nats/
│   └── dynamo/
└── inventory/              # Node inventory
    └── hosts.yaml
```

## 📊 The Baseline Comparison (Modules 09-10)

A key learning experience in this curriculum is the **before/after comparison**:

| Metric              | Module 09 (Baseline)       | Module 10 (Dynamo)     | Why It Matters                |
| ------------------- | -------------------------- | ---------------------- | ----------------------------- |
| **Throughput**      | Two independent vLLM nodes | Same nodes with Dynamo | Shows specialization benefits |
| **TTFT**            | Higher variance            | Lower, consistent      | Dedicated prefill nodes help  |
| **p95 Latency**     | Higher tail latency        | Lower tail latency     | No prefill blocking decode    |
| **GPU Utilization** | Uneven, bursty             | Balanced, efficient    | Better resource allocation    |

This comparison answers the fundamental question: **"Why do we need Dynamo at all?"**

## 🎯 Learning Outcomes

By the end of this journey, you will:

1. **Understand** Dynamo's architecture and how components interact
2. **Deploy** inference workers using multiple backends (SGLang, vLLM, TRT-LLM)
3. **Configure** service discovery with etcd manually
4. **Implement** messaging patterns with NATS JetStream
5. **Enable** KV-aware routing for efficient prefix caching
6. **Scale** inference across multiple nodes using InfiniBand
7. **Measure** baseline performance and demonstrate quantifiable improvements
8. **Optimize** with disaggregated prefill/decode serving
9. **Benchmark** and tune for production workloads

---

*"The hard way is the easy way in the long run."*
