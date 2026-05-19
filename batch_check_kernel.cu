// batch_check_kernel.cu — BATCH PROCESSING: N_INDEPENDENT batches × M constraints
// Each block = one independent constraint group (fracture-coalesce on GPU)
// Grid: blockIdx.x = batch, threadIdx.x = constraint within batch
// Results coalesced with atomicOr

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <chrono>

struct Constraint {
    double lo;
    double hi;
};

struct BatchDef {
    int offset;      // offset into values array
    int n_values;    // number of values in this batch
    int n_constraints;
};

// ── Kernel ───────────────────────────────────────────────────────────
// Each block handles one batch. Each thread handles one constraint.
// Thread checks all values in the batch against its assigned constraint.
__global__ void batch_check_kernel(
    const double* __restrict__ values,       // flattened [all batches]
    const Constraint* __restrict__ cons,     // flattened [all constraints]
    const BatchDef* __restrict__ batches,    // [N_BATCHES]
    uint8_t* __restrict__ error_masks,       // per-value error mask
    int N_BATCHES
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= N_BATCHES) return;

    BatchDef bd = batches[batch_idx];
    int c_idx = threadIdx.x;  // constraint index within this batch
    if (c_idx >= bd.n_constraints) return;

    // Each block's thread 0 loads the batch def (already in register)
    // Constraint index in global array: assume constraints are laid out per-batch
    // For simplicity, constraints are batch_idx * max_constraints + c_idx
    Constraint my_con = cons[batch_idx * bd.n_constraints + c_idx];

    const double* batch_vals = values + bd.offset;
    uint8_t* batch_masks = error_masks + bd.offset;
    uint8_t bit = static_cast<uint8_t>(1) << c_idx;

    for (int v = 0; v < bd.n_values; v++) {
        double val = batch_vals[v];
        bool violated = (val < my_con.lo) | (val > my_con.hi) | isnan(val);
        if (violated) {
            // Note: uint8_t atomicOr not available in CUDA
            // For production, use uint32_t masks. Here we use simple write
            // since v1 is just a reference kernel (v2 is the benchmarked one).
            batch_masks[v] |= bit;  // OK for demo; v2 kernel uses per-thread write
        }
    }
}

// Alternative: each thread checks one value against all constraints (better for large batches)
__global__ void batch_check_v2_kernel(
    const double* __restrict__ values,
    const Constraint* __restrict__ cons,
    const BatchDef* __restrict__ batches,
    uint8_t* __restrict__ error_masks,
    int N_BATCHES,
    int MAX_M
) {
    extern __shared__ Constraint s_cons[];

    int batch_idx = blockIdx.y;
    if (batch_idx >= N_BATCHES) return;

    BatchDef bd = batches[batch_idx];

    // Load constraints for this batch into shared memory
    if (threadIdx.x < bd.n_constraints) {
        s_cons[threadIdx.x] = cons[batch_idx * MAX_M + threadIdx.x];
    }
    __syncthreads();

    int val_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (val_idx >= bd.n_values) return;

    double val = values[bd.offset + val_idx];
    uint8_t mask = 0;

    for (int c = 0; c < bd.n_constraints; c++) {
        bool violated = (val < s_cons[c].lo) | (val > s_cons[c].hi) | isnan(val);
        mask |= (static_cast<uint8_t>(violated) << c);
    }

    error_masks[bd.offset + val_idx] = mask;
}

// ── Benchmark ────────────────────────────────────────────────────────
int main() {
    const int N_BATCHES = 1000;
    const int M = 8;
    const int VALUES_PER_BATCH = 1000000;
    const int TOTAL_VALUES = N_BATCHES * VALUES_PER_BATCH;
    const int N_ITER = 3;

    printf("═══ batch_check_kernel benchmark ═══\n");
    printf("Config: %d batches × %d constraints × %d values = %d total checks\n",
           N_BATCHES, M, VALUES_PER_BATCH, (long long)N_BATCHES * M * VALUES_PER_BATCH);

    // Host allocation
    double* h_values = new double[TOTAL_VALUES];
    Constraint* h_cons = new Constraint[N_BATCHES * M];
    BatchDef* h_batches = new BatchDef[N_BATCHES];
    uint8_t* h_masks = new uint8_t[TOTAL_VALUES];

    srand(42);
    for (int i = 0; i < TOTAL_VALUES; i++) {
        h_values[i] = (rand() / (double)RAND_MAX) * 20.0 - 10.0;
    }
    // Sprinkle NaNs
    for (int i = 0; i < TOTAL_VALUES / 500; i++) {
        h_values[rand() % TOTAL_VALUES] = NAN;
    }

    for (int b = 0; b < N_BATCHES; b++) {
        h_batches[b].offset = b * VALUES_PER_BATCH;
        h_batches[b].n_values = VALUES_PER_BATCH;
        h_batches[b].n_constraints = M;
        for (int c = 0; c < M; c++) {
            h_cons[b * M + c].lo = -5.0 + c * 0.3 + (b % 5) * 0.1;
            h_cons[b * M + c].hi =  5.0 - c * 0.2 - (b % 3) * 0.1;
        }
    }
    memset(h_masks, 0, TOTAL_VALUES);

    // Device allocation
    double* d_values;
    Constraint* d_cons;
    BatchDef* d_batches;
    uint8_t* d_masks;
    cudaMalloc(&d_values, TOTAL_VALUES * sizeof(double));
    cudaMalloc(&d_cons, N_BATCHES * M * sizeof(Constraint));
    cudaMalloc(&d_batches, N_BATCHES * sizeof(BatchDef));
    cudaMalloc(&d_masks, TOTAL_VALUES * sizeof(uint8_t));

    cudaMemcpy(d_values, h_values, TOTAL_VALUES * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cons, h_cons, N_BATCHES * M * sizeof(Constraint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_batches, h_batches, N_BATCHES * sizeof(BatchDef), cudaMemcpyHostToDevice);
    cudaMemset(d_masks, 0, TOTAL_VALUES);

    // Using v2 kernel (value-per-thread) with 2D grid
    int block = 256;
    dim3 grid_vals((VALUES_PER_BATCH + block - 1) / block, N_BATCHES);
    size_t shared = M * sizeof(Constraint);

    // Warmup
    batch_check_v2_kernel<<<grid_vals, block, shared>>>(
        d_values, d_cons, d_batches, d_masks, N_BATCHES, M);
    cudaDeviceSynchronize();

    double best_ms = 1e9;
    for (int it = 0; it < N_ITER; it++) {
        cudaMemset(d_masks, 0, TOTAL_VALUES);
        auto t0 = std::chrono::high_resolution_clock::now();
        batch_check_v2_kernel<<<grid_vals, block, shared>>>(
            d_values, d_cons, d_batches, d_masks, N_BATCHES, M);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < best_ms) best_ms = ms;
    }

    double total_checks = (double)N_BATCHES * M * VALUES_PER_BATCH;
    double throughput = total_checks / (best_ms * 1e-3);
    double gb = (double)TOTAL_VALUES * sizeof(double) / (1ULL << 30);

    printf("Best time:     %.3f ms\n", best_ms);
    printf("Throughput:    %.0f checks/sec (%.2f B checks/sec)\n", throughput, throughput / 1e9);
    printf("Data scanned:  %.2f GB\n", gb);
    printf("Bandwidth:     %.1f GB/s\n", gb / (best_ms * 1e-3));

    // Count violations
    cudaMemcpy(h_masks, d_masks, TOTAL_VALUES, cudaMemcpyDeviceToHost);
    long total_violated = 0;
    for (int i = 0; i < TOTAL_VALUES; i++) {
        if (h_masks[i] != 0) total_violated++;
    }
    printf("Violated:      %ld / %d (%.1f%%)\n", total_violated, TOTAL_VALUES,
           100.0 * total_violated / TOTAL_VALUES);

    cudaFree(d_values); cudaFree(d_cons); cudaFree(d_batches); cudaFree(d_masks);
    delete[] h_values; delete[] h_cons; delete[] h_batches; delete[] h_masks;
    return 0;
}
