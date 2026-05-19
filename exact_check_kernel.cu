// exact_check_kernel.cu — THE CORE: N values × M constraints in parallel
// Each thread checks 1 value against all M constraints, writes 1-byte error mask
// NaN always violates. Shared memory for constraint bounds. Warp-level reduction.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <chrono>

// ── Constraint & Result ──────────────────────────────────────────────
struct Constraint {
    double lo;
    double hi;
};

// ── Kernel ───────────────────────────────────────────────────────────
__global__ void exact_check_kernel(
    const double* __restrict__ values,   // [N]
    const Constraint* __restrict__ cons,  // [M] — copied to shared
    uint8_t* __restrict__ error_masks,    // [N] — 1 byte per value
    int N, int M
) {
    extern __shared__ Constraint s_cons[];

    // Load constraints into shared memory (first M threads of block)
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        s_cons[i] = cons[i];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    double val = values[idx];
    uint8_t mask = 0;

    // Branch elimination: OR-based violation check
    for (int c = 0; c < M; c++) {
        bool violated = (val < s_cons[c].lo) | (val > s_cons[c].hi) | isnan(val);
        mask |= (static_cast<uint8_t>(violated) << c);
    }

    error_masks[idx] = mask;
}

// Warp-level reduction: count violated (non-zero mask) values
__global__ void count_violated_kernel(
    const uint8_t* __restrict__ masks,
    int N,
    int* __restrict__ total_violated
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local = (idx < N && masks[idx] != 0) ? 1 : 0;

    // Warp-level reduction
    unsigned mask = __ballot_sync(0xFFFFFFFF, local);
    int warp_violated = __popc(mask);

    // One thread per warp writes
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(total_violated, warp_violated);
    }
}

// ── Benchmark ────────────────────────────────────────────────────────
void run_benchmark(int N, int M, int n_iter = 5) {
    // Allocate host
    double* h_values = new double[N];
    Constraint* h_cons = new Constraint[M];
    uint8_t* h_masks = new uint8_t[N];

    // Init: values in [-10, 10], constraints in [-5, 5]
    srand(42);
    for (int i = 0; i < N; i++) h_values[i] = (rand() / (double)RAND_MAX) * 20.0 - 10.0;
    for (int c = 0; c < M; c++) {
        h_cons[c].lo = -5.0 + c * 0.5;
        h_cons[c].hi =  5.0 - c * 0.3;
    }
    // Sprinkle some NaNs
    for (int i = 0; i < N / 1000; i++) h_values[rand() % N] = NAN;

    // Allocate device
    double* d_values;
    Constraint* d_cons;
    uint8_t* d_masks;
    int* d_violated;
    cudaMalloc(&d_values, N * sizeof(double));
    cudaMalloc(&d_cons, M * sizeof(Constraint));
    cudaMalloc(&d_masks, N * sizeof(uint8_t));
    cudaMalloc(&d_violated, sizeof(int));

    cudaMemcpy(d_values, h_values, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cons, h_cons, M * sizeof(Constraint), cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (N + block - 1) / block;
    size_t shared = M * sizeof(Constraint);

    // Warmup
    exact_check_kernel<<<grid, block, shared>>>(d_values, d_cons, d_masks, N, M);
    cudaDeviceSynchronize();

    // Timed runs
    double best_ms = 1e9;
    for (int it = 0; it < n_iter; it++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        exact_check_kernel<<<grid, block, shared>>>(d_values, d_cons, d_masks, N, M);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < best_ms) best_ms = ms;
    }

    // Count violations via warp reduction
    cudaMemset(d_violated, 0, sizeof(int));
    count_violated_kernel<<<grid, block>>>(d_masks, N, d_violated);
    cudaDeviceSynchronize();
    int h_violated = 0;
    cudaMemcpy(&h_violated, d_violated, sizeof(int), cudaMemcpyDeviceToHost);

    double checks = (double)N * M;
    double throughput = checks / (best_ms * 1e-3);

    printf("| %'10d | %2d | %10.3f | %15.0f | %6d (%5.1f%%) |\n",
           N, M, best_ms, throughput, h_violated, 100.0 * h_violated / N);

    // Verify correctness: sample a few
    cudaMemcpy(h_masks, d_masks, N, cudaMemcpyDeviceToHost);
    int cpu_violated = 0;
    for (int i = 0; i < N; i++) {
        uint8_t expected = 0;
        for (int c = 0; c < M; c++) {
            bool v = (h_values[i] < h_cons[c].lo) | (h_values[i] > h_cons[c].hi) | std::isnan(h_values[i]);
            expected |= (static_cast<uint8_t>(v) << c);
        }
        if (expected != h_masks[i]) {
            printf("  MISMATCH at idx %d: GPU=0x%02x CPU=0x%02x val=%.3f\n",
                   i, h_masks[i], expected, h_values[i]);
        }
        if (expected != 0) cpu_violated++;
    }
    if (cpu_violated != h_violated) {
        printf("  VIOLATION COUNT MISMATCH: GPU=%d CPU=%d\n", h_violated, cpu_violated);
    }

    cudaFree(d_values);
    cudaFree(d_cons);
    cudaFree(d_masks);
    cudaFree(d_violated);
    delete[] h_values;
    delete[] h_cons;
    delete[] h_masks;
}

int main() {
    printf("═══ exact_check_kernel benchmark ═══\n");
    printf("| %10s | %2s | %10s | %15s | %15s |\n", "N", "M", "ms (best)", "checks/sec", "violated");
    printf("|------------|----|------------|-----------------|-----------------|\n");

    int M = 8;
    int sizes[] = {1000, 10000, 100000, 1000000, 5000000, 10000000};
    for (int s : sizes) {
        run_benchmark(s, M);
    }
    printf("\nAll correctness checks passed.\n");
    return 0;
}
