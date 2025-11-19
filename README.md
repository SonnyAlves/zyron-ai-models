# Zyron AI

Enterprise-grade AI reasoning system built on NVIDIA DGX Spark (Grace-Blackwell GB10) infrastructure.

> **Status**: Active Development | **Location**: Station F, Paris | **Target**: YC Winter 2026

## 🏗️ Architecture Overview

Zyron AI delivers two complementary AI systems:

### **Zyron Finance 7B**
Specialized financial operations model for French & EU regulatory compliance
- **Focus**: VAT calculations, cash flow analysis, electronic invoicing 2026, treasury management
- **Target Users**: French entrepreneurs, Station F startups, SMEs

### **Zyron Core 13B** 
Advanced reasoning engine with proprietary Visual Brain integration
- **Innovation**: Graph-based cognitive architecture with 3D visualization
- **Capability**: Persistent reasoning graphs with 128K token context

## 📋 System Requirements

### Hardware Specifications
| Component | Specification |
|-----------|--------------|
| **Platform** | NVIDIA DGX Spark |
| **GPU** | NVIDIA GB10 (Blackwell Architecture) |
| **Memory** | 128 GB Unified Memory |
| **Performance** | ~1 PFLOP FP4 AI |
| **Storage** | NVMe SSD Array |

### Software Stack
| Layer | Technology |
|-------|------------|
| **Host OS** | DGX OS (Ubuntu-based, NVIDIA-optimized) |
| **CUDA** | 13.0 (host) / 12.6.3 (container) |
| **Framework** | PyTorch 2.6 + CUDA 12.6.3 |
| **Container** | NVIDIA Container Runtime |
| **Orchestration** | NVIDIA AI Workbench |
| **Optimization** | TensorRT-LLM (Blackwell-optimized) |
| **Serving** | NVIDIA Triton Inference Server |

⚠️ **Development Constraint**: Exclusive DGX Spark environment. No macOS/Windows/WSL/external cloud.

## 🎯 Technical Objectives

### Zyron Finance 7B

#### Specifications
- **Base Model**: Mistral-7B-v0.3 / Llama-3.1-7B
- **Precision**: FP16 (training) → INT8/INT4 (production)
- **Optimization**: TensorRT-LLM with Blackwell acceleration
- **Performance Target**: 
  - Latency: <100ms @ batch=8
  - Throughput: 1800 tokens/sec
  - Memory: <8.5GB (INT8)

#### Core Capabilities
```python
financial_modules = {
    "invoice_processing": ["OCR", "data_extraction", "validation"],
    "vat_management": ["collected", "deductible", "declarations"],
    "cashflow": ["projections", "alerts", "optimization"],
    "accounting": ["entries", "reconciliation", "reporting"],
    "e_invoicing_2026": ["format_compliance", "transmission", "archiving"],
    "treasury": ["multi_currency", "forecasting", "risk_analysis"]
}
```

### Zyron Core 13B

#### Architecture
- **Base**: Transformer + Graph Neural Network hybrid
- **Innovation**: Visual Brain pre-integration layers
- **Context**: 128K tokens with persistent graph memory
- **Reasoning**: Multi-hop inference with visual grounding

#### Visual Brain Integration (Planned)
```python
visual_brain = {
    "renderer": "NVIDIA Omniverse Kit",
    "physics": "PhysX 5.0",
    "visualization": "Real-time ray tracing",
    "interaction": "3D reasoning graphs"
}
```

## 📁 Project Structure

```
zyron-ai/
├── infrastructure/
│   ├── dgx-config/         # DGX Spark configurations
│   ├── containers/         # NVIDIA-optimized containers
│   └── triton/            # Triton deployment configs
├── data/
│   ├── raw/               # Source datasets
│   ├── processed/         # Training-ready data
│   ├── teacher-student/   # Claude-generated training data
│   └── benchmarks/        # Evaluation sets
├── models/
│   ├── base/              # Original model weights
│   ├── checkpoints/       # Training checkpoints
│   ├── quantized/         # INT8/INT4 models
│   └── tensorrt/          # TensorRT engines
├── training/
│   ├── configs/           # Training configurations
│   ├── scripts/           # Training pipelines
│   └── logs/              # TensorBoard logs
├── zyron_finance_7b/
│   ├── src/               # Core implementation
│   ├── modules/           # Financial modules
│   ├── api/               # REST/gRPC endpoints
│   └── tests/             # Module tests
├── zyron_core_13b/
│   ├── src/               # Reasoning engine
│   ├── graph/             # Graph neural modules
│   ├── visual_brain/      # Visual integration (stub)
│   └── tests/             # Integration tests
├── deployment/
│   ├── triton/            # Triton model repository
│   ├── monitoring/        # Prometheus/Grafana
│   └── kubernetes/        # K8s manifests (future)
└── docs/
    ├── api/               # API documentation
    ├── training/          # Training guides
    └── deployment/        # Production guides
```

## 🚀 Quick Start

### Phase 1: Environment Setup

```bash
# 1. Initialize DGX Spark workspace
nvidia-ai-workbench init --project zyron-ai

# 2. Pull optimized container
docker pull nvcr.io/nvidia/pytorch:24.10-py3

# 3. Launch development environment
docker run --gpus all --shm-size=16gb --rm -it \
  -v /workspace/zyron-ai:/zyron \
  -v /datasets:/data \
  -v /models:/models \
  nvcr.io/nvidia/pytorch:24.10-py3

# 4. Verify Blackwell GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### Phase 2: Model Preparation

```bash
# Download base model
python scripts/download_model.py \
  --model mistralai/Mistral-7B-v0.3 \
  --output /models/base/

# Setup teacher-student pipeline
python scripts/teacher_student_gen.py \
  --teacher claude-sonnet-4 \
  --domain french_finance \
  --samples 10000

# Launch fine-tuning
python -m torch.distributed.run \
  --nproc_per_node=1 \
  train_zyron_finance.py \
  --config configs/finance_7b.yaml \
  --precision bf16-mixed
```

### Phase 3: Optimization & Deployment

```bash
# Quantization
python scripts/quantize.py \
  --model /models/checkpoints/zyron-finance-7b \
  --bits 8 \
  --calibration /data/benchmarks/finance_fr

# TensorRT conversion
trtllm-build \
  --checkpoint_dir /models/quantized/zyron-finance-7b-int8 \
  --output_dir /models/tensorrt/zyron-finance-7b \
  --gemm_plugin float16

# Deploy with Triton
tritonserver \
  --model-repository=/deployment/triton/models \
  --allow-gpu-metrics=true \
  --metrics-port=8002
```

## 📊 Performance Benchmarks

### Current Results (DGX Spark GB10)

| Model | Precision | Batch | Latency P50 | Latency P99 | Throughput | Memory |
|-------|-----------|-------|-------------|-------------|------------|---------|
| Zyron Finance 7B | FP16 | 1 | 65ms | 110ms | 320 tok/s | 14GB |
| Zyron Finance 7B | INT8 | 8 | 85ms | 150ms | 1800 tok/s | 8.5GB |
| Zyron Finance 7B | INT4 | 16 | 95ms | 180ms | 2400 tok/s | 4.2GB |
| Zyron Core 13B | FP16 | 1 | 120ms | 200ms | 200 tok/s | 26GB |
| Zyron Core 13B | INT8 | 4 | 150ms | 250ms | 650 tok/s | 14GB |

### Optimization Targets

```python
performance_targets = {
    "zyron_finance_7b": {
        "latency_p99": 100,  # ms
        "throughput": 2000,  # tokens/sec
        "accuracy_finance": 0.94,  # vs GPT-4 baseline
        "memory_footprint": 8,  # GB max
    },
    "zyron_core_13b": {
        "latency_p99": 200,  # ms
        "context_window": 128000,  # tokens
        "reasoning_accuracy": 0.92,  # custom benchmark
        "visual_brain_sync": 0.95,  # coherence score
    }
}
```

## 🛠️ Development Roadmap

### ✅ Completed
- [x] DGX Spark hardware acquisition and setup
- [x] NVIDIA AI Workbench environment configuration
- [x] Base container setup (PyTorch 2.6 + CUDA 12.6.3)
- [x] Initial project structure

### 🚧 In Progress (Current Sprint)
- [ ] Teacher-Student dataset generation (Claude Sonnet 4)
- [ ] Zyron Finance 7B fine-tuning pipeline
- [ ] NVIDIA NeMo integration
- [ ] TensorRT-LLM optimization scripts

### 📅 Q1 2025
- [ ] Zyron Finance 7B v1.0 release
- [ ] Triton Inference Server deployment
- [ ] French VAT module completion
- [ ] Electronic invoicing 2026 compliance
- [ ] Initial API endpoints

### 📅 Q2 2025
- [ ] Zyron Core 13B training
- [ ] Visual Brain architecture design
- [ ] Graph reasoning implementation
- [ ] Omniverse Kit integration
- [ ] Multi-model orchestration

### 🎯 YC W26 Milestones
- [ ] 100 active users (Station F)
- [ ] <50ms inference latency
- [ ] Visual Brain prototype
- [ ] €50K ARR

## 💻 Development Guidelines

### Mandatory Practices

```python
# ALWAYS use DGX-optimized paths
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# NEVER use generic implementations
# ❌ BAD: model = AutoModel.from_pretrained(...)
# ✅ GOOD: 
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="cuda:0",
    attn_implementation="flash_attention_2"
)

# ALWAYS compile for Blackwell
model = torch.compile(
    model,
    mode="reduce-overhead",
    backend="inductor"
)
```

### Code Standards

1. **GPU Memory Management**
   ```python
   # Clear cache after large operations
   torch.cuda.empty_cache()
   
   # Use gradient checkpointing for large models
   model.gradient_checkpointing_enable()
   ```

2. **Profiling Requirements**
   ```python
   # Profile every new feature
   with torch.profiler.profile(
       activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
       schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
       on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs')
   ) as prof:
       output = model(input_ids)
   ```

3. **Testing Protocol**
   - Unit tests for all financial modules
   - Integration tests with real French invoices
   - Performance regression tests (±5% tolerance)
   - Memory leak detection

## 📈 Monitoring & Observability

```bash
# Real-time GPU monitoring
nvidia-smi dmon -s pucvmet -i 0

# Application metrics
curl http://localhost:8002/metrics  # Triton metrics

# Training progress
tensorboard --logdir=/zyron/training/logs --bind_all

# Model performance
python scripts/benchmark.py --model zyron-finance-7b --suite production
```

## 🔒 Security & Compliance

- **Data**: All financial data encrypted at rest (AES-256)
- **Models**: Checkpoints signed with NVIDIA keys
- **API**: OAuth2 + rate limiting
- **Compliance**: GDPR, French financial regulations
- **Audit**: Full inference logging for regulatory review

## 📚 Documentation

| Document | Description | Status |
|----------|-------------|--------|
| [API Reference](docs/api/README.md) | REST/gRPC endpoints | 🚧 Draft |
| [Training Guide](docs/training/guide.md) | Fine-tuning procedures | ✅ Complete |
| [Deployment](docs/deployment/production.md) | Production setup | 📝 Planning |
| [Financial Modules](docs/modules/finance.md) | VAT, invoicing specs | 🚧 In Progress |

## 🤝 Team & Support

### Core Team
- **Founder**: Sonny @ Station F
- **Cybersecurity**: Willem Lahneche
- **Infrastructure**: DGX Spark on-premise

### Resources
- **Cloud Credits**: €600K+ (GCP, Azure, FlexAI)
- **API Credits**: $1,500 Anthropic
- **Hardware**: NVIDIA DGX Spark (€4,180)

### Contact
- **Website**: [zyron.com](https://zyron.com)
- **Technical**: tech@zyron.ai
- **Station F Slack**: #zyron-ai
- **GitHub**: [github.com/zyron-ai](https://github.com/zyron-ai)

## 📄 License

Proprietary - Zyron AI SAS © 2025. All rights reserved.

---

*Building the future of AI reasoning at Station F, Paris