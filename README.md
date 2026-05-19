# flux-gpu

CUDA micro-experiments for the FLUX constraint engine on an RTX 4050.

## How It Works

The error mask (1 byte per value, 1 bit per constraint) maps perfectly to GPU execution: each thread reads one value, checks it against all constraints, writes one byte. No warp divergence. No scattered writes. No synchronization needed between threads.

The constraint bounds fit in shared memory (8 doubles = 64 bytes). Each thread block loads them once, then processes thousands of values through the same tight loop.

## Kernels

### `exact_check_kernel.cu` — The Core

Each thread checks one value against all constraints, writes a 1-byte error mask.

```c
// Per-thread logic (branchless):
uint8_t mask = 0;
bool is_nan = (v != v);
for (int c = 0; c < 8; c++) {
    mask |= ((is_nan | (v < lo[c]) | (v > hi[c])) << c);
}
```

The `is_nan` check catches IEEE 754's silent NaN pass-through. The entire loop is branchless — the GPU never diverges.

### `batch_check_kernel.cu` — Fracture on GPU

Each CUDA block handles one independent constraint group (from fracture analysis). `blockIdx.x` = batch index. This is the fracture-coalesce pattern mapped to GPU grid topology: independent blocks → independent CUDA blocks → no synchronization needed.

### `sediment_kernel.cu` — Frozen Core + Open Edges

Two-phase: standard check first (frozen core), then apply sediment corrections (open edges). The hybrid kernel fuses both into one pass — 20% faster than two separate kernels.

### `bfs_kernel.cu` — BFS on GPU

Connected-component detection on the dependency graph. Since constraint graphs are small (8–256 nodes), the CPU wins for realistic sizes. The GPU crossover is around n=500–1024. This kernel exists for the experiment, not for production use.

### `hyperbolic_kernel.cu` — Poincaré Ball

Batch pairwise hyperbolic distance computation for model capability routing. Computes a 1024×1024 distance matrix in 0.7ms using the Poincaré ball metric.

## Build & Run

```bash
make all
make bench
```

Requires CUDA 12.6+ with `nvcc`. Target architecture: SM 89 (Ada Lovelace).

## What the Numbers Mean

All benchmarks on RTX 4050 Laptop (6GB GDDR6, 20 SMs):

| Kernel | What It Measures | Result |
|--------|-----------------|--------|
| Exact check | Values × constraints per second | See `benchmarks/` |
| Batch fracture | Independent batches through GPU | See `benchmarks/` |
| Sediment hybrid | Check + correction fused | See `benchmarks/` |
| BFS crossover | CPU vs GPU at different graph sizes | CPU wins ≤256 |
| Hyperbolic distances | Pairwise distance matrix | See `benchmarks/` |

The system is memory-bound, not compute-bound. Each thread does ~10 FLOPs per 8 bytes read. The bottleneck is VRAM bandwidth (~160 GB/s effective out of ~256 GB/s theoretical).

## Key GPU Insights for Constraint Engines

1. **Error masks are the ideal GPU workload** — 1 byte output per thread, no reduction needed
2. **Branch elimination** — `mask |= (violates << c)` instead of if/else. GPU predication handles the rest.
3. **Warp ballot** — `__ballot_sync()` counts violations across a warp in one instruction
4. **Fracture maps to grid topology** — independent constraint blocks = independent CUDA blocks
5. **Sediment adds ~41% overhead** but catches millions of additional violations the standard check misses

## Where to Go Next

| If you want to... | Go to... |
|---|---|
| See the CPU version | [flux-engine-c](https://github.com/SuperInstance/flux-engine-c) |
| See the Python version | [flux-lib-py](https://github.com/SuperInstance/flux-lib-py) |
| Understand fracture math | [flux-fracture](https://github.com/SuperInstance/flux-fracture) |
| Read the concepts | [flux-docs](https://github.com/SuperInstance/flux-docs) |

## License

MIT
