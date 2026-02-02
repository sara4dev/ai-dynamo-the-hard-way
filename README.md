# AI Dynamo: The Hard Way

A progressive, hands-on approach to learning [NVIDIA AI Dynamo](https://github.com/ai-dynamo/dynamo) - a Datacenter Scale Distributed Inference Serving Framework.

> **Philosophy**: No Kubernetes operators. No magic. Just understanding each component from the ground up.

## 🖥️ Hardware Setup

| Component | Hardware | Purpose |
| --------- | -------- | ------- |
| **GPU 0** | NVIDIA RTX 5090 (32GB) | Prefill worker / Primary inference |
| **GPU 1** | NVIDIA RTX 5090 (32GB) | Decode worker / Disaggregated serving |

This dual-GPU setup demonstrates Dynamo's key innovations: disaggregated prefill/decode serving, NIXL GPU-to-GPU transfers via PCIe/cuda_ipc, and KV-aware routing.

> **Coming Soon**: Performance benchmarks comparing baseline vs Dynamo-optimized serving.

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

### Part 1: Core Concepts (Available Now)

| Module | Notebook | Description | Status |
| ------ | -------- | ----------- | ------ |
| **01** | [Setup and First Inference](notebooks/01-setup-and-first-inference.ipynb) | Install Dynamo, start etcd/NATS, run your first inference request with SGLang backend | ✅ |
| **02** | [Disaggregated Serving](notebooks/02-disaggregated-serving.ipynb) | Run prefill on GPU 0 and decode on GPU 1, observe KV cache transfer via NIXL | ✅ |
| **03** | [Exploring NIXL](notebooks/03-exploring-nixl.ipynb) | Deep dive into NVIDIA Inference Xfer Library - GPU-to-GPU transfers | ✅ |
| **04** | [KV-Aware Routing](notebooks/04-kv-aware-routing.ipynb) | Smart routing based on cached KV blocks, NATS JetStream for KV events | ✅ |

### Part 2: Dynamo Ecosystem (Coming Soon)

| Module | Topic | Description | Status |
| ------ | ----- | ----------- | ------ |
| **05** | [Model Express](https://github.com/ai-dynamo/modelexpress) | Accelerate model downloads with Rust-based caching sidecar | 🔜 |
| **06** | [AI Configurator](https://github.com/ai-dynamo/aiconfigurator) | Find optimal prefill/decode configurations for your hardware | 🔜 |
| **07** | [Grove](https://github.com/ai-dynamo/grove) | Kubernetes gang scheduling and topology-aware autoscaling | 🔜 |

---

## 📖 Module Summaries

### Module 01: Setup and First Inference
Get Dynamo running in under 15 minutes:
- Verify GPU and Python environment
- Start etcd (service discovery) and NATS (messaging) via Docker
- Launch Dynamo frontend + SGLang worker
- Send your first OpenAI-compatible inference request
- Understand how workers register in etcd

### Module 02: Disaggregated Prefill-Decode Serving
Separate compute-bound prefill from memory-bound decode:
- Run prefill worker on GPU 0, decode worker on GPU 1
- See how NIXL transfers KV cache between GPUs
- Understand the bootstrap server handshake for RDMA coordination
- Observe both workers registering independently in etcd

### Module 03: Exploring NIXL
Deep dive into NVIDIA's high-performance transfer library:
- Understand `nixl_agent`, memory registration, and UCX backends
- Run a complete multi-process GPU-to-GPU transfer demo
- See how prefill registers its endpoint in etcd for service discovery
- Learn transport selection: NVLink > cuda_ipc > RDMA > TCP fallback
- Transfer 1GB of data and measure actual bandwidth

### Module 04: KV-Aware Routing
Route requests to workers with cached prefixes:
- Launch two workers with `--kv-events-config` for block event publishing
- Enable `--router-mode kv` on the frontend
- Send repeated prompts and observe cache hits
- Monitor KV events in NATS JetStream
- See the router's radix tree pick the optimal worker

---

## 🔮 Upcoming Modules

### Module 05: Model Express
[Model Express](https://github.com/ai-dynamo/modelexpress) is a Rust-based model cache management service designed to accelerate model downloads:

- **What it does**: Acts as a HuggingFace cache, reducing repeated downloads across replicas
- **Deployment modes**: Shared storage (PVC) or distributed (gRPC transfer)
- **Key benefit**: Faster cold starts for multi-node/multi-worker deployments
- **Integration**: Works as a sidecar alongside vLLM, SGLang, TensorRT-LLM

```bash
# CLI usage (HuggingFace replacement)
modelexpress download --model Qwen/Qwen3-32B --cache-dir /models
```

### Module 06: AI Configurator
[AI Configurator](https://github.com/ai-dynamo/aiconfigurator) helps you find optimal configurations for disaggregated serving:

- **What it does**: Searches configuration space to optimize throughput under SLA constraints
- **Inputs**: Model, GPU count, GPU type, TTFT/TPOT targets
- **Outputs**: Prefill/decode worker counts, parallelism settings, Dynamo YAML files
- **Backends**: TensorRT-LLM, vLLM, SGLang

```bash
# Find best configuration for Qwen3-32B on 32x H200
aiconfigurator cli default --model QWEN3_32B --total_gpus 32 --system h200_sxm --ttft 300 --tpot 10
```

### Module 07: Grove
[Grove](https://github.com/ai-dynamo/grove) provides Kubernetes enhancements for AI inference:

- **Hierarchical gang scheduling**: Schedule prefill + decode workers as atomic units
- **Topology-aware placement**: Optimize for NVLink domains and network locality
- **Multi-level autoscaling**: Scale PodCliques (role groups) together
- **Startup ordering**: Ensure workers start in correct order (e.g., MPI leader/worker)

```yaml
# Grove PodCliqueSet example
apiVersion: grove.ai-dynamo.dev/v1alpha1
kind: PodCliqueSet
metadata:
  name: llm-inference
spec:
  scalingGroups:
    - name: prefill-decode
      cliques:
        - name: prefill
          replicas: 2
        - name: decode
          replicas: 4
```

---

## 🚀 Quick Start

```bash
# Clone this repository
git clone https://github.com/sara4dev/ai-dynamo-the-hard-way.git
cd ai-dynamo-the-hard-way

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install Dynamo with SGLang backend
pip install "ai-dynamo[sglang]"

# Start with Module 01
jupyter lab notebooks/01-setup-and-first-inference.ipynb
```

## 📋 Prerequisites

- **Python**: 3.10+
- **CUDA**: 12.x+
- **Docker**: For etcd and NATS containers
- **2+ GPUs**: Required for disaggregated serving (Modules 02-04)

## 🔗 Key Resources

### Core Dynamo
- [AI Dynamo GitHub](https://github.com/ai-dynamo/dynamo)
- [Official Documentation](https://docs.nvidia.com/dynamo/latest)
- [NIXL - NVIDIA Inference Xfer Library](https://github.com/ai-dynamo/nixl)

### Ecosystem Projects
- [Model Express](https://github.com/ai-dynamo/modelexpress) - Model caching sidecar
- [AI Configurator](https://github.com/ai-dynamo/aiconfigurator) - Configuration optimizer
- [Grove](https://github.com/ai-dynamo/grove) - Kubernetes gang scheduling

## 📁 Project Structure

```
ai-dynamo-the-hard-way/
├── README.md
├── notebooks/
│   ├── 01-setup-and-first-inference.ipynb    # ✅ Available
│   ├── 02-disaggregated-serving.ipynb        # ✅ Available
│   ├── 03-exploring-nixl.ipynb               # ✅ Available
│   └── 04-kv-aware-routing.ipynb             # ✅ Available
```

## 🎯 Learning Outcomes

By completing the available modules, you will:

1. **Install and run** Dynamo with etcd/NATS infrastructure
2. **Understand** disaggregated prefill/decode architecture
3. **Experience** NIXL GPU-to-GPU transfers firsthand
4. **Configure** KV-aware routing for prefix cache optimization
5. **Monitor** system state via etcd queries and NATS subscriptions

By completing the upcoming modules, you will also:

6. **Accelerate** model downloads with Model Express
7. **Optimize** configurations with AI Configurator
8. **Deploy** at scale with Grove on Kubernetes

---

*"The hard way is the easy way in the long run."*
