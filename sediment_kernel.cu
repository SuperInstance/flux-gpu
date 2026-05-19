// sediment_kernel.cu — GPU SEDIMENT: frozen core + open edges on GPU
// Input: error masks from batch check + sediment correction table
// Each thread: if sediment layer applies to my constraint, re-check with corrected bounds

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <chrono>

struct Constraint {
    double lo;
    double hi;
};

struct SedimentLayer {
    int constraint_idx;    // which constraint this layer corrects
    double lo_correction;  // additive correction to lower bound
    double hi_correction;  // additive correction to upper bound
    int priority;          // lower = apply first (frozen core)
};

// ── Kernel ───────────────────────────────────────────────────────────
__global__ void sediment_recheck_kernel(
    const double* __restrict__ values,
    const Constraint* __restrict__ cons,        // original constraints [M]
    const SedimentLayer* __restrict__ layers,    // sediment layers [L]
    const uint8_t* __restrict__ initial_masks,   // from batch check [N]
    uint8_t* __restrict__ corrected_masks,       // output [N]
    int N, int M, int L
) {
    extern __shared__ char smem[];
    Constraint* s_cons = reinterpret_cast<Constraint*>(smem);
    SedimentLayer* s_layers = reinterpret_cast<SedimentLayer*>(smem + M * sizeof(Constraint));

    // Load to shared
    for (int i = threadIdx.x; i < M; i += blockDim.x)
        s_cons[i] = cons[i];
    for (int i = threadIdx.x; i < L; i += blockDim.x)
        s_layers[i] = layers[i];
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    double val = values[idx];
    uint8_t mask = initial_masks[idx];

    // Apply sediment corrections: re-check violated constraints with corrected bounds
    for (int l = 0; l < L; l++) {
        int c = s_layers[l].constraint_idx;
        uint8_t bit = static_cast<uint8_t>(1) << c;

        // Only re-check if this constraint was violated
        if (mask & bit) {
            double corrected_lo = s_cons[c].lo + s_layers[l].lo_correction;
            double corrected_hi = s_cons[c].hi + s_layers[l].hi_correction;
            bool still_violated = (val < corrected_lo) | (val > corrected_hi) | isnan(val);
            if (!still_violated) {
                mask &= ~bit;  // Clear violation — sediment absorbed it
            }
        }
    }

    corrected_masks[idx] = mask;
}

// ── Benchmark ────────────────────────────────────────────────────────
int main() {
    const int N = 5000000;
    const int M = 8;
    const int L = 5;  // 5 sediment layers
    const int N_ITER = 5;

    printf("═══ sediment_kernel benchmark ═══\n");
    printf("Config: %d values, %d constraints, %d sediment layers\n", N, M, L);

    // Host
    double* h_values = new double[N];
    Constraint* h_cons = new Constraint[M];
    SedimentLayer* h_layers = new SedimentLayer[L];
    uint8_t* h_initial = new uint8_t[N];
    uint8_t* h_corrected = new uint8_t[N];

    srand(42);
    for (int i = 0; i < N; i++) h_values[i] = (rand() / (double)RAND_MAX) * 20.0 - 10.0;
    for (int i = 0; i < N / 1000; i++) h_values[rand() % N] = NAN;

    for (int c = 0; c < M; c++) {
        h_cons[c].lo = -5.0 + c * 0.5;
        h_cons[c].hi =  5.0 - c * 0.3;
    }

    // Sediment layers: relax bounds for some constraints
    for (int l = 0; l < L; l++) {
        h_layers[l].constraint_idx = l;
        h_layers[l].lo_correction = -1.0 - l * 0.2;  // widen lower bound
        h_layers[l].hi_correction =  1.0 + l * 0.2;  // widen upper bound
        h_layers[l].priority = l;
    }

    // Generate initial masks (simulate batch check output)
    for (int i = 0; i < N; i++) {
        uint8_t mask = 0;
        for (int c = 0; c < M; c++) {
            bool v = (h_values[i] < h_cons[c].lo) | (h_values[i] > h_cons[c].hi) | std::isnan(h_values[i]);
            mask |= (static_cast<uint8_t>(v) << c);
        }
        h_initial[i] = mask;
    }

    // Device
    double* d_values; Constraint* d_cons; SedimentLayer* d_layers;
    uint8_t* d_initial; uint8_t* d_corrected;
    cudaMalloc(&d_values, N * sizeof(double));
    cudaMalloc(&d_cons, M * sizeof(Constraint));
    cudaMalloc(&d_layers, L * sizeof(SedimentLayer));
    cudaMalloc(&d_initial, N);
    cudaMalloc(&d_corrected, N);

    cudaMemcpy(d_values, h_values, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cons, h_cons, M * sizeof(Constraint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layers, h_layers, L * sizeof(SedimentLayer), cudaMemcpyHostToDevice);
    cudaMemcpy(d_initial, h_initial, N, cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (N + block - 1) / block;
    size_t shared = M * sizeof(Constraint) + L * sizeof(SedimentLayer);

    // Warmup
    sediment_recheck_kernel<<<grid, block, shared>>>(
        d_values, d_cons, d_layers, d_initial, d_corrected, N, M, L);
    cudaDeviceSynchronize();

    double best_ms = 1e9;
    for (int it = 0; it < N_ITER; it++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        sediment_recheck_kernel<<<grid, block, shared>>>(
            d_values, d_cons, d_layers, d_initial, d_corrected, N, M, L);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < best_ms) best_ms = ms;
    }

    cudaMemcpy(h_corrected, d_corrected, N, cudaMemcpyDeviceToHost);

    int initial_violated = 0, corrected_violated = 0, absorbed = 0;
    for (int i = 0; i < N; i++) {
        if (h_initial[i]) initial_violated++;
        if (h_corrected[i]) corrected_violated++;
        if (h_initial[i] && !h_corrected[i]) absorbed++;
    }

    printf("Best time:       %.3f ms\n", best_ms);
    printf("Throughput:      %.0f values/sec\n", N / (best_ms * 1e-3));
    printf("Initial violated:  %d (%.1f%%)\n", initial_violated, 100.0 * initial_violated / N);
    printf("After sediment:    %d (%.1f%%)\n", corrected_violated, 100.0 * corrected_violated / N);
    printf("Absorbed by sediment: %d (%.1f%% of violations)\n", absorbed,
           initial_violated > 0 ? 100.0 * absorbed / initial_violated : 0.0);

    cudaFree(d_values); cudaFree(d_cons); cudaFree(d_layers);
    cudaFree(d_initial); cudaFree(d_corrected);
    delete[] h_values; delete[] h_cons; delete[] h_layers;
    delete[] h_initial; delete[] h_corrected;
    return 0;
}
