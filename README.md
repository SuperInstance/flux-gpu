# FLUX GPU — CUDA Micro-Experiments for Constraint Engine

**Hardware:** NVIDIA GeForce RTX 4050 Laptop (6 GB VRAM, SM 8.9, CUDA 12.6, WSL2 dxg)

## Results (median of runs, 2026-05-19)

### 1. Exact Check Kernel — THE CORE

| N | M | Time (ms) | Throughput (checks/sec) | Violated |
|---:|:-:|----------:|------------------------:|---------:|
| 1K | 8 | 0.018 | 451 M | 757 (75.7%) |
| 10K | 8 | 0.019 | 4.26 B | 7,822 |
| 100K | 8 | 0.048 | 16.5 B | 78,128 |
| 1M | 8 | 0.344 | 23.3 B | 779,918 |
| 5M | 8 | 1.620 | 24.7 B | 3,900,291 |
| **10M** | **8** | **3.213** | **24.9 B** | 7,803,392 |

**Peak throughput: ~25 billion constraint checks per second.** Correctness verified against CPU for all values.

### 2. Batch Check Kernel — BATCH PROCESSING

| Config | Time (ms) | Throughput | Bandwidth |
|--------|----------:|-----------:|----------:|
| 1000 batches × 8 constraints × 1M values | 385 | **20.8 B checks/sec** | 19.3 GB/s |

Each block = one independent constraint group. This IS fracture-coalesce on GPU.

### 3. Sediment Kernel — GPU SEDIMENT

| Values | Constraints | Sediment Layers | Time (ms) | Throughput |
|-------:|:-----------:|:---------------:|----------:|-----------:|
| 5M | 8 | 5 | 1.70 | **2.95 B values/sec** |

Demonstrates frozen core + open edges. Sediment re-checks only previously-violated constraints with relaxed bounds.

### 4. BFS Kernel — GPU vs CPU Crossover

| Graph Size (n) | CPU (µs) | GPU (µs) | CPU/GPU | Winner |
|:--------------:|---------:|---------:|--------:|:------:|
| 8 | 0.0 | 344 | 0.00 | CPU |
| 64 | 1.3 | 353 | 0.00 | CPU |
| 256 | 77.6 | 337 | 0.23 | CPU |
| **1024** | **1694** | **587** | **2.88** | **GPU** |

**Crossover: between n=256 and n=1024.** For constraint dependency graphs (typically 8–256 nodes), CPU BFS wins. GPU wins at n≥1024.

### 5. Hyperbolic Kernel — Poincaré Ball Distances

| Embeddings | Dims | Time (ms) | Pairs | Throughput |
|:----------:|:----:|----------:|------:|-----------:|
| 1024 | 8 | 0.71 | 523,776 | **737 M dist/sec** |

Verification: 10 sample pairs match CPU computation. Distance range [0.115, 5.643], mean 2.040.

## Architecture: Why Error Masks Are the Ideal GPU Workload

1. **1 byte per thread, no divergence** — every thread does the exact same work
2. **Branch elimination** — `mask |= (val < lo) | (val > hi) | isnan(val)` compiles to predicated instructions
3. **Shared memory for constraints** — M=8 doubles = 64 bytes, fits in shared with zero bank conflicts
4. **Coalesced writes** — error masks are consecutive bytes → perfect memory coalescing
5. **Memory-bound, not compute-bound** — ~10 FLOPs per thread, so bandwidth is the bottleneck
6. **Warp-level reduction** — `__ballot_sync()` + `__popc()` counts violations in zero extra memory

## Key Insights

- **25B checks/sec** means the RTX 4050 can validate 250M values against 100 constraints in 1 second
- The error mask is the **natural data structure for GPUs**: no locks, no atomics, no divergence
- Batch check (fracture-coalesce) scales linearly with batch count — each block is independent
- Sediment corrections add ~5% overhead on top of the initial check (re-checking only violations)
- BFS crossover at n≈500: constraint dependency graphs should stay on CPU, but the kernel exists for large graphs
- Hyperbolic distances at 737M/sec make fleet model embedding comparison trivially fast

## Build & Run

```bash
make all              # Build all kernels (requires CUDA 12.6, sm_89)
make bench            # Build + run all benchmarks
make clean            # Clean binaries
bash run_all_benchmarks.sh  # Run all with 3 iterations each
```

## Files

| File | Lines | Purpose |
|------|------:|---------|
| `exact_check_kernel.cu` | ~155 | Core: N values × M constraints, shared mem, warp reduction |
| `batch_check_kernel.cu` | ~180 | Batch: 2D grid, per-batch constraint groups |
| `sediment_kernel.cu` | ~160 | Sediment: re-check with corrected bounds |
| `bfs_kernel.cu` | ~120 | BFS: level-synchronous, CPU vs GPU comparison |
| `hyperbolic_kernel.cu` | ~185 | Poincaré ball: Möbius add, exp/log maps, distance matrix |
| `Makefile` | ~35 | Build system |
| `run_all_benchmarks.sh` | ~40 | Benchmark runner |
