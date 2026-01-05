---
day: 1
date: 2026-01-05
topic: "GPU Architecture Deep Dive"
focus_area: ML Systems
phase: 1
tags:
  - ml-systems
  - hardware
  - gpu
  - cuda
  - tensor-cores
  - streaming-multiprocessor
prerequisites: []
related:
  - "[[Day 02 - Memory Hierarchy and Bandwidth]]"
  - "[[Day 03 - CUDA Programming Fundamentals]]"
  - "[[Day 04 - FLOPs vs Memory Bound]]"
status: not-started
---

# Day 01: GPU Architecture Deep Dive

> [!summary] TL;DR
> Modern GPUs like the NVIDIA H100 and B200 are massively parallel processors containing thousands of cores organized into Streaming Multiprocessors (SMs). Understanding their architecture—particularly Tensor Cores for matrix operations and the memory hierarchy—is fundamental to optimizing ML workloads.

## Overview & Motivation

### What Is This?

A GPU (Graphics Processing Unit) is a specialized processor designed for parallel computation. Unlike CPUs which excel at sequential tasks with a few powerful cores, GPUs contain thousands of smaller cores that can execute operations simultaneously. This makes them ideal for the matrix multiplications and parallel data processing that dominate machine learning workloads.

### Why Does It Matter?

Understanding GPU architecture is the foundation for:
- **Writing efficient CUDA code** that fully utilizes hardware
- **Identifying bottlenecks** in ML training and inference
- **Making informed decisions** about model architecture and batch sizes
- **Optimizing memory usage** to fit larger models or batches

> [!example] Real-World Impact
> A model that runs at 10% GPU utilization versus 80% GPU utilization represents an 8x difference in cost and time. Understanding architecture is the key to bridging this gap.

### Where Does It Fit?

GPU architecture knowledge enables everything that follows in this roadmap:
- [[Day 02 - Memory Hierarchy and Bandwidth]] builds on memory concepts
- [[Day 03 - CUDA Programming Fundamentals]] shows how to program these architectures
- [[Day 04 - FLOPs vs Memory Bound]] uses architectural knowledge for analysis

### Historical Context

- **1999**: NVIDIA introduces first GPU (GeForce 256)
- **2007**: CUDA 1.0 enables general-purpose GPU computing
- **2012**: AlexNet demonstrates GPU advantage for deep learning
- **2017**: Volta introduces Tensor Cores for matrix operations
- **2022**: Hopper (H100) brings Transformer Engine
- **2024**: Blackwell (B200) doubles Tensor Core throughput

> [!info] Connection to Roadmap
> - **Builds on**: Starting point - no prerequisites
> - **Leads to**: [[Day 02 - Memory Hierarchy and Bandwidth]], [[Day 03 - CUDA Programming Fundamentals]]
> - **Related concepts**: [[Day 17 - Kernel Fusion for Efficiency]], [[Day 47 - Interconnects NVLink vs PCIe]]

---

## Prerequisites Refresher

> [!tip] Before You Continue
> This is Day 1, so there are no formal prerequisites. However, familiarity with these concepts will help:
> - Basic understanding of computer architecture (CPU, RAM, cache)
> - Matrix multiplication (what it means to multiply matrices)
> - Parallelism concept (doing multiple things at once)

---

## Core Concepts

### Concept 1: The CPU vs GPU Paradigm

**CPUs** (Central Processing Units) are designed for:
- Sequential execution with branch prediction
- Complex control flow
- Low-latency single operations
- Typically 8-128 cores

**GPUs** (Graphics Processing Units) are designed for:
- Parallel execution of simple operations
- High throughput on uniform workloads
- Thousands of cores working simultaneously
- Optimized for matrix math

> [!example] Intuitive Analogy
> **CPU**: A single expert chef who can make any dish perfectly but works alone
> **GPU**: 1000 line cooks who each do one simple task, together producing thousands of dishes per hour

**Formal Definition:**
A GPU is a **SIMD** (Single Instruction, Multiple Data) / **SIMT** (Single Instruction, Multiple Thread) processor that executes the same operation across many data elements in parallel.

### Concept 2: Streaming Multiprocessors (SMs)

The **Streaming Multiprocessor (SM)** is the fundamental building block of NVIDIA GPUs. Each SM contains:

| Component       | H100 Count               | Purpose                            |
| --------------- | ------------------------ | ---------------------------------- |
| CUDA Cores      | 128                      | General-purpose floating-point ops |
| Tensor Cores    | 4                        | Matrix multiply-accumulate         |
| Registers       | 256 KB                   | Fast per-thread storage            |
| Shared Memory   | 228 KB (configurable)    | Fast shared storage per block      |
| L1 Cache        | Combined with shared mem | Data caching                       |
| Warp Schedulers | 4                        | Thread scheduling                  |

> [!example] Intuitive Analogy
> An SM is like a factory floor. CUDA cores are the workers, Tensor Cores are specialized machines for bulk work, shared memory is the shared workbench, and registers are each worker's personal toolbelt.

**GPU Scaling:**
| GPU | SMs | Total CUDA Cores | Total Tensor Cores |
|-----|-----|------------------|-------------------|
| A100 | 108 | 6,912 | 432 |
| H100 (SXM) | 132 | 16,896 | 528 |
| B200 | 192 | 18,432 | 768 |

### Concept 3: Tensor Cores ^tensor-cores

**Tensor Cores** are specialized matrix processing units that perform mixed-precision matrix multiply-accumulate operations in a single clock cycle.

**Operation:** Tensor Cores compute:
$$D = A \times B + C$$

Where:
- $A$: Input matrix (FP16, BF16, FP8, or INT8)
- $B$: Input matrix (same precision as A)
- $C$: Accumulator matrix (FP32 or same as inputs)
- $D$: Output matrix

**Matrix Tile Sizes:**
| Generation | Matrix Operation |
|------------|-----------------|
| Volta (V100) | 4×4×4 FP16 |
| Ampere (A100) | 8×8×4 FP16/BF16 |
| Hopper (H100) | 16×8×16 FP16/BF16 |
| Blackwell (B200) | Extended sizes |

> [!warning] Common Misconception
> Tensor Cores don't replace CUDA cores—they complement them. Matrix operations use Tensor Cores; other operations (activations, normalization, etc.) use CUDA cores.

### Concept 4: Thread Hierarchy ^thread-hierarchy

CUDA organizes parallel execution in a hierarchy:

```
Grid (all threads for a kernel launch)
├── Block 0 (up to 1024 threads)
│   ├── Warp 0 (32 threads - execute in lockstep)
│   ├── Warp 1
│   └── ...
├── Block 1
└── ...
```

**Key Concepts:**
- **Thread**: Smallest unit of execution
- **Warp**: 32 threads that execute the same instruction simultaneously
- **Block**: Group of threads that share memory and synchronize (max 1024)
- **Grid**: All blocks for a kernel launch

> [!tip] Performance Insight
> Warps are the actual execution unit. If threads in a warp take different code paths (divergence), both paths execute sequentially, reducing efficiency.

---

## Technical Deep Dive

### Architecture Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                     NVIDIA H100 GPU                          │
├──────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    GPC (8 total)                        │ │
│  │  ┌──────────────────────────────────────────────────┐   │ │
│  │  │              SM (16-18 per GPC)                  │   │ │
│  │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐        │   │ │
│  │  │  │Processing│ │Processing│ │Processing│        │   │ │
│  │  │  │  Block   │ │  Block   │ │  Block   │ x4     │   │ │
│  │  │  │ 32 CUDA  │ │ 32 CUDA  │ │ 32 CUDA  │        │   │ │
│  │  │  │ 1 Tensor │ │ 1 Tensor │ │ 1 Tensor │        │   │ │
│  │  │  └──────────┘ └──────────┘ └──────────┘        │   │ │
│  │  │  ┌────────────────────────────────────────────┐ │   │ │
│  │  │  │    Shared Memory / L1 Cache (228 KB)      │ │   │ │
│  │  │  └────────────────────────────────────────────┘ │   │ │
│  │  └──────────────────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                  L2 Cache (50 MB)                       │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              HBM3 Memory (80 GB, 3.35 TB/s)             │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### H100 Specifications (SXM Version)

| Specification | Value |
|---------------|-------|
| Architecture | Hopper |
| Process Node | TSMC 4N |
| Transistors | 80 billion |
| Die Size | 814 mm² |
| SMs | 132 |
| CUDA Cores | 16,896 |
| Tensor Cores | 528 (4th gen) |
| HBM3 Memory | 80 GB |
| Memory Bandwidth | 3.35 TB/s |
| L2 Cache | 50 MB |
| TDP | 700W |
| FP16 Tensor Core | 1,979 TFLOPS |
| FP8 Tensor Core | 3,958 TFLOPS |

### B200 Specifications

| Specification | Value |
|---------------|-------|
| Architecture | Blackwell |
| Process Node | TSMC 4NP |
| Transistors | 208 billion (2 dies) |
| SMs | 192 |
| CUDA Cores | 18,432 |
| Tensor Cores | 768 (5th gen) |
| HBM3e Memory | 192 GB |
| Memory Bandwidth | 8 TB/s |
| FP8 Tensor Core | 9,000 TFLOPS |
| FP4 Tensor Core | 18,000 TFLOPS |

### Execution Model

**Kernel Launch → Grid → Blocks → Warps → Threads**

1. Host (CPU) launches kernel with grid dimensions
2. GPU schedules blocks onto available SMs
3. Each SM schedules warps from its assigned blocks
4. Warp scheduler issues instructions to 32 threads

```
Time →
Warp 0: [Instruction 1] [Instruction 2] [Instruction 3]...
Warp 1: [Instruction 1] [Instruction 2] [Instruction 3]...
Warp 2: [Instruction 1] [Instruction 2] [Instruction 3]...
...
(Warps interleave to hide latency)
```

---

## Implementation

### Minimal Example: Query GPU Properties

```python
"""
GPU Architecture Query - Day 01
Demonstrates how to query GPU properties programmatically

Prerequisites:
- NVIDIA GPU with CUDA
- PyTorch installed
"""
import torch

def g(p, name):
    return getattr(p, name, None)

def show(label, value, unit=""):
    if value is None:
        print(f"  {label}: N/A")
    else:
        print(f"  {label}: {value}{unit}")

def print_gpu_architecture_info():
    if not torch.cuda.is_available():
        print("No CUDA GPU available!")
        return
    print(f"Number of GPUs: {torch.cuda.device_count()}\n")

    # SM -> CUDA cores mapping
    cuda_cores_per_sm = {
        (7, 0): 64,    # Volta (V100)
        (7, 5): 64,    # Turing (T4, RTX 20xx)
        (8, 0): 64,    # Ampere A100
        (8, 6): 128,   # Ampere RTX 30xx
        (8, 9): 128,   # Ada Lovelace RTX 40xx / L40
        (9, 0): 128,   # Hopper H100
    }

    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)

        print(f"=== GPU {i}: {p.name} ===")
        print(f"Compute Capability: {p.major}.{p.minor}")
        print(f"Total Memory: {p.total_memory / 1e9:.2f} GB")
        print(f"Multiprocessors (SMs): {p.multi_processor_count}")

        cores = cuda_cores_per_sm.get((p.major, p.minor), 64)
        print(f"Estimated CUDA Cores: {p.multi_processor_count * cores}")

        print("\nMemory Properties:")
        show(
            "Shared Memory per Block",
            g(p, "shared_memory_per_block") / 1024 if g(p, "shared_memory_per_block") else None,
            " KB",
        )
        show(
            "Shared Memory per SM",
            g(p, "shared_memory_per_multiprocessor") / 1024 if g(p, "shared_memory_per_multiprocessor") else None,
            " KB",
        )

        print("\nThread / Block Limits:")
        show("Max Threads per Block", g(p, "max_threads_per_block"))
        show("Max Threads per SM", g(p, "max_threads_per_multi_processor"))
        show("Max Block Dimensions", g(p, "max_block_dim"))
        show("Max Grid Dimensions", g(p, "max_grid_dim"))
        show("Warp Size", p.warp_size)

        print("\nClock Rates:")
        show("GPU Clock", g(p, "clock_rate") / 1e6 if g(p, "clock_rate") else None, " GHz")
        show("Memory Clock", g(p, "memory_clock_rate") / 1e6 if g(p, "memory_clock_rate") else None, " GHz")
        show("Memory Bus Width", g(p, "memory_bus_width"), " bits")

        mclk = g(p, "memory_clock_rate")
        bus = g(p, "memory_bus_width")
        if mclk and bus:
            bw = mclk * 1e3 * bus * 2 / 8 / 1e9
            print(f"  Theoretical Memory Bandwidth: {bw:.1f} GB/s")
        else:
            print("  Theoretical Memory Bandwidth: N/A")
        print()


if __name__ == "__main__":
    print_gpu_architecture_info()
```

### Understanding Tensor Core Operations

```python
"""
Tensor Core Usage Demonstration
Shows how Tensor Cores accelerate matrix operations

Prerequisites:
- GPU with Tensor Cores (Volta or newer)
- PyTorch 1.6+
"""

import torch
import torch.nn as nn
import time

def benchmark_matmul(M, N, K, dtype, num_iterations=100):
    """
    Benchmark matrix multiplication.
    
    Args:
        M, N, K: Matrix dimensions (M×K @ K×N = M×N)
        dtype: Data type (float32, float16, bfloat16)
        num_iterations: Number of iterations for timing
    
    Returns:
        Average time in milliseconds
    """
    device = torch.device("cuda")
    
    # Create random matrices
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)
    
    # Warmup
    for _ in range(10):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    avg_time_ms = (end - start) * 1000 / num_iterations
    
    # Calculate TFLOPS
    flops = 2 * M * N * K  # multiply-add = 2 operations
    tflops = flops / (avg_time_ms / 1000) / 1e12
    
    return avg_time_ms, tflops


def demonstrate_tensor_cores():
    """Compare FP32 vs FP16 (Tensor Core) performance."""
    
    print("Matrix Multiplication Benchmark")
    print("=" * 60)
    
    # Typical transformer dimensions
    sizes = [
        (4096, 4096, 4096),    # Square
        (4096, 11008, 4096),   # FFN up-projection (Llama-7B)
        (4096, 4096, 11008),   # FFN down-projection
    ]
    
    for M, N, K in sizes:
        print(f"\nMatrix size: ({M}×{K}) @ ({K}×{N}) = ({M}×{N})")
        print("-" * 50)
        
        # FP32 (uses CUDA cores)
        time_fp32, tflops_fp32 = benchmark_matmul(M, N, K, torch.float32)
        print(f"FP32:   {time_fp32:.3f} ms, {tflops_fp32:.1f} TFLOPS")
        
        # FP16 (uses Tensor Cores when available)
        time_fp16, tflops_fp16 = benchmark_matmul(M, N, K, torch.float16)
        print(f"FP16:   {time_fp16:.3f} ms, {tflops_fp16:.1f} TFLOPS")
        
        # BF16 (uses Tensor Cores when available)
        if torch.cuda.is_bf16_supported():
            time_bf16, tflops_bf16 = benchmark_matmul(M, N, K, torch.bfloat16)
            print(f"BF16:   {time_bf16:.3f} ms, {tflops_bf16:.1f} TFLOPS")
        
        speedup = time_fp32 / time_fp16
        print(f"Speedup (FP16 vs FP32): {speedup:.2f}x")


if __name__ == "__main__":
    demonstrate_tensor_cores()
```

> [!warning] Common Implementation Mistakes
> - **Mistake 1**: Assuming all operations use Tensor Cores. Only matrix multiplications with compatible shapes and dtypes use them.
> - **Mistake 2**: Using matrix dimensions not aligned to 8 or 16. Tensor Cores require specific alignment for peak efficiency.
> - **Mistake 3**: Forgetting `torch.cuda.synchronize()` when benchmarking. GPU operations are asynchronous.

---

## Performance & Benchmarks

### Computational Complexity

| Operation | Time Complexity | Tensor Core Eligible |
|-----------|-----------------|---------------------|
| Matrix Multiply | O(M×N×K) | ✅ Yes |
| Element-wise | O(N) | ❌ No |
| Reduction | O(N) | ❌ No |
| Softmax | O(N) | ❌ No |

### Real-World Benchmarks: H100 vs A100

| Model | A100 (80GB) | H100 (80GB) | Speedup |
|-------|-------------|-------------|---------|
| Llama-2-7B inference | 1,500 tok/s | 3,200 tok/s | 2.1x |
| Llama-2-70B inference | 180 tok/s | 420 tok/s | 2.3x |
| GPT-3 training step | 145 ms | 68 ms | 2.1x |

*Benchmarks vary based on batch size, sequence length, and optimization level*

### GPU Utilization Targets

| Metric | Poor | Acceptable | Good | Excellent |
|--------|------|------------|------|-----------|
| SM Occupancy | <25% | 25-50% | 50-75% | >75% |
| Memory Bandwidth | <30% | 30-50% | 50-70% | >70% |
| Tensor Core Usage | <20% | 20-40% | 40-60% | >60% |

> [!tip] When to Use This Knowledge
> - ✅ Use when: Optimizing ML training/inference, choosing hardware, debugging performance
> - ❌ Avoid when: Writing simple prototypes where correctness matters more than speed
> - 🔄 Consider [[Day 16 - Profiling Tools Introduction]] when: You need to measure actual utilization

---

## Real-World Applications

### Who Uses This?

| Company/Model | How They Use It | Reference |
|---------------|-----------------|-----------|
| OpenAI | Custom GPU configurations for GPT training | Public talks |
| Meta | Large-scale Llama training on H100 clusters | Meta AI blog |
| Google | TPU + GPU hybrid for various models | Google Research |
| NVIDIA | TensorRT optimizations for inference | NVIDIA docs |

### Case Study: LLM Inference Optimization

A company serving a 7B parameter model improved throughput from 500 to 2,000 tokens/second by:
1. Switching from FP32 to FP16 (Tensor Core utilization)
2. Increasing batch size to improve SM occupancy
3. Aligning matrix dimensions to Tensor Core requirements

---

## Common Pitfalls & Debugging

> [!warning] Pitfall 1: Low GPU Utilization
> **Symptom:** GPU shows <30% utilization in nvidia-smi
> **Cause:** Batch size too small, CPU bottleneck, or poor kernel design
> **Fix:** Increase batch size, profile with Nsight, check CPU-GPU transfer

> [!warning] Pitfall 2: Out of Memory Despite "Enough" Memory
> **Symptom:** OOM error when model should fit
> **Cause:** Memory fragmentation, PyTorch caching, or gradients
> **Fix:** Use `torch.cuda.empty_cache()`, gradient checkpointing, or smaller batches

> [!warning] Pitfall 3: Not Using Tensor Cores
> **Symptom:** FP16 not faster than FP32
> **Cause:** Matrix dimensions not aligned, or old GPU without Tensor Cores
> **Fix:** Pad dimensions to multiples of 8 (Ampere) or 16 (Hopper)

### Debugging Checklist

- [ ] Check GPU utilization with `nvidia-smi`
- [ ] Verify Tensor Core usage with Nsight Compute
- [ ] Confirm memory bandwidth utilization
- [ ] Check for CPU-GPU synchronization bottlenecks
- [ ] Verify batch size is large enough for parallelism

---

## Connections to Other Topics

### This Topic Builds On
- No prerequisites (Day 1)

### This Topic Leads To
- [[Day 02 - Memory Hierarchy and Bandwidth]] - Memory architecture in detail
- [[Day 03 - CUDA Programming Fundamentals]] - Programming this hardware
- [[Day 04 - FLOPs vs Memory Bound]] - Analyzing performance limits

### Frequently Combined With
- [[Day 16 - Profiling Tools Introduction]] - Measuring utilization
- [[Day 17 - Kernel Fusion for Efficiency]] - Optimization technique
- [[Day 18 - Mixed Precision FP16 BF16]] - Leveraging Tensor Cores

### See Also
- [[Day 47 - Interconnects NVLink vs PCIe]] - Multi-GPU communication
- [[Day 84 - Future Hardware Groq Cerebras]] - Alternative architectures

---

## Key Takeaways

> [!summary] Must Remember ^key-takeaways
> 1. **GPUs are massively parallel**: Thousands of cores vs CPU's few—design for parallelism
> 2. **SMs are the building blocks**: Each SM has CUDA cores, Tensor Cores, and local memory
> 3. **Tensor Cores accelerate matrix ops**: 10-20x faster than CUDA cores for FP16 matmul
> 4. **Threads organize hierarchically**: Grid → Blocks → Warps (32 threads) → Threads
> 5. **Architecture knowledge enables optimization**: Understanding hardware is key to efficient code

### Quick Reference Cheat Sheet

```
GPU Architecture Quick Reference
================================
SM: Streaming Multiprocessor (fundamental compute unit)
Warp: 32 threads executing in lockstep
Block: Up to 1024 threads sharing memory
Grid: All blocks for a kernel

H100 Key Specs:
- 132 SMs, 528 Tensor Cores
- 80GB HBM3 @ 3.35 TB/s
- 1,979 TFLOPS FP16

Tensor Core Alignment:
- Ampere: Multiples of 8
- Hopper: Multiples of 16
```

---

## Further Reading & Resources

### Essential Papers
- [ ] [NVIDIA H100 Tensor Core GPU Architecture](https://resources.nvidia.com/en-us-tensor-core) - Official whitepaper
- [ ] [Dissecting the NVIDIA Volta GPU Architecture](https://arxiv.org/abs/1804.06826) - Academic analysis

### Official Documentation
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) - Comprehensive reference
- [NVIDIA GPU Architecture Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-implementation)

### Video Resources
- [NVIDIA GTC Hopper Architecture Deep Dive](https://www.nvidia.com/gtc/) - ~1 hour, excellent overview

### Code Repositories
- [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples) - Official examples

### Books
> [!tip] Recommended Reading
> - **Programming Massively Parallel Processors** (Kirk & Hwu) - Chapters 1-4
> - **Professional CUDA C Programming** (Cheng et al.) - Chapters 1-3
> See [[Recommended Books - LLM Inference and ML Systems]] for full details.

---

## Practice Exercises

> [!question] Exercise 1: Conceptual
> Calculate the theoretical peak FP16 TFLOPS for an H100 GPU with 528 Tensor Cores, each capable of 256 FP16 ops per cycle, running at 1.8 GHz.
> 
> <details>
> <summary>Answer</summary>
> 
> Peak TFLOPS = Tensor Cores × Ops per cycle × Clock frequency
> = 528 × 256 × 1.8 GHz
> = 528 × 256 × 1.8 × 10⁹
> = 243.4 × 10¹² = 243.4 TFLOPS
> 
> (Note: Actual advertised specs are higher due to different counting methods)
> </details>

> [!question] Exercise 2: Implementation
> Write a Python script that queries your GPU and calculates what percentage of H100 specs your GPU has.
> 
> <details>
> <summary>Solution</summary>
> 
> ```python
> import torch
> 
> # H100 reference specs
> H100_SPECS = {
>     'sm_count': 132,
>     'memory_gb': 80,
>     'bandwidth_tbps': 3.35
> }
> 
> props = torch.cuda.get_device_properties(0)
> 
> print(f"Your GPU: {props.name}")
> print(f"SMs: {props.multi_processor_count} ({100*props.multi_processor_count/H100_SPECS['sm_count']:.1f}% of H100)")
> print(f"Memory: {props.total_memory/1e9:.1f} GB ({100*props.total_memory/1e9/H100_SPECS['memory_gb']:.1f}% of H100)")
> ```
> </details>

---

## Self-Assessment

Rate your understanding (update after studying):

- [ ] Can explain the difference between CPU and GPU architecture
- [ ] Understand the SM structure (CUDA cores, Tensor Cores, memory)
- [ ] Know the thread hierarchy (Grid → Block → Warp → Thread)
- [ ] Can explain when Tensor Cores are used vs CUDA cores
- [ ] Understand why matrix alignment matters for performance

---

*Next: [[Day 02 - Memory Hierarchy and Bandwidth]]*
*Previous: None (Day 1)*
