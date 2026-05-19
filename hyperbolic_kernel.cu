// hyperbolic_kernel.cu — POINCARÉ BALL on GPU
// Möbius addition, exp/log maps, batch pairwise hyperbolic distances
// For N model embeddings in D-dimensional Poincaré ball

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <chrono>

// ── Hyperbolic primitives (per-thread) ───────────────────────────────

#define D_MAX 16

__device__ double d_dot(const double* a, const double* b, int D) {
    double s = 0.0;
    for (int i = 0; i < D; i++) s += a[i] * b[i];
    return s;
}

__device__ double d_norm2(const double* x, int D) {
    return d_dot(x, x, D);
}

// Möbius addition: x ⊕_M y = ((1+2<x,y>+||y||²)x + (1-||x||²)y) / (1+2<x,y>+||x||²||y||²)
__device__ void mobius_add(
    const double* x, const double* y, double* out, int D
) {
    double xx = d_dot(x, x, D);
    double yy = d_dot(y, y, D);
    double xy = d_dot(x, y, D);
    double denom = 1.0 + 2.0 * xy + xx * yy;
    double c1 = (1.0 + 2.0 * xy + yy) / denom;
    double c2 = (1.0 - xx) / denom;
    for (int i = 0; i < D; i++) {
        out[i] = c1 * x[i] + c2 * y[i];
    }
}

// Poincaré distance: d(x,y) = arcosh(1 + 2*||x-y||² / ((1-||x||²)(1-||y||²)))
__device__ double poincare_distance(
    const double* x, const double* y, int D
) {
    double xx = d_norm2(x, D);
    double yy = d_norm2(y, D);
    double dx[16];
    for (int i = 0; i < D; i++) dx[i] = x[i] - y[i];
    double dxy = d_norm2(dx, D);
    double denom = (1.0 - xx) * (1.0 - yy);
    double arg = 1.0 + 2.0 * dxy / denom;
    return acosh(fmax(arg, 1.0));
}

// Exponential map at origin: exp_0(v) = tanh(||v||) * v / ||v||
__device__ void exp_map_origin(const double* v, double* out, int D) {
    double nv = sqrt(d_norm2(v, D));
    if (nv < 1e-10) {
        for (int i = 0; i < D; i++) out[i] = 0.0;
        return;
    }
    double scale = tanh(nv) / nv;
    for (int i = 0; i < D; i++) out[i] = scale * v[i];
}

// Logarithmic map at origin: log_0(y) = artanh(||y||) * y / ||y||
__device__ void log_map_origin(const double* y, double* out, int D) {
    double ny = sqrt(d_norm2(y, D));
    if (ny < 1e-10) {
        for (int i = 0; i < D; i++) out[i] = 0.0;
        return;
    }
    double scale = atanh(fmin(ny, 1.0 - 1e-7)) / ny;
    for (int i = 0; i < D; i++) out[i] = scale * y[i];
}

// ── Distance matrix kernel ───────────────────────────────────────────
// Each thread computes one pair (i,j)
__global__ void distance_matrix_kernel(
    const double* __restrict__ embeddings,  // [N * D]
    double* __restrict__ distances,          // [N * N]
    int N, int D
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N) return;

    if (i == j) {
        distances[i * N + j] = 0.0;
        return;
    }
    // Only compute upper triangle
    if (i > j) return;

    const double* xi = embeddings + i * D;
    const double* xj = embeddings + j * D;

    double dist = poincare_distance(xi, xj, D);
    distances[i * N + j] = dist;
    distances[j * N + i] = dist;
}

// ── Benchmark ────────────────────────────────────────────────────────
int main() {
    const int N = 1024;
    const int D = 8;
    const int N_ITER = 5;

    printf("═══ hyperbolic_kernel benchmark ═══\n");
    printf("Config: %d embeddings × %dD Poincaré ball\n", N, D);

    // Generate random points in Poincaré ball (norm < 1)
    double* h_emb = new double[N * D];
    srand(42);
    for (int i = 0; i < N; i++) {
        double norm = 0;
        for (int d = 0; d < D; d++) {
            h_emb[i * D + d] = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
            norm += h_emb[i * D + d] * h_emb[i * D + d];
        }
        norm = sqrt(norm);
        // Scale to random radius in [0.1, 0.9)
        double r = 0.1 + (rand() / (double)RAND_MAX) * 0.8;
        for (int d = 0; d < D; d++) {
            h_emb[i * D + d] *= r / (norm + 1e-10);
        }
    }

    double* d_emb;
    double* d_dist;
    cudaMalloc(&d_emb, N * D * sizeof(double));
    cudaMalloc(&d_dist, N * N * sizeof(double));
    cudaMemcpy(d_emb, h_emb, N * D * sizeof(double), cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (N + 31) / 32);

    // Warmup
    distance_matrix_kernel<<<grid, block>>>(d_emb, d_dist, N, D);
    cudaDeviceSynchronize();

    double best_ms = 1e9;
    for (int it = 0; it < N_ITER; it++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        distance_matrix_kernel<<<grid, block>>>(d_emb, d_dist, N, D);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < best_ms) best_ms = ms;
    }

    double pairs = (double)N * (N - 1) / 2;
    double throughput = pairs / (best_ms * 1e-3);

    printf("Best time:       %.3f ms\n", best_ms);
    printf("Pairs computed:  %.0f (%d×%d upper triangle)\n", pairs, N, N);
    printf("Throughput:      %.0f dist/sec\n", throughput);

    // Sample distances for verification
    double* h_dist = new double[N * N];
    cudaMemcpy(h_dist, d_dist, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    // Verify a few pairs against CPU computation
    int errors = 0;
    for (int sample = 0; sample < 10; sample++) {
        int i = rand() % N, j = rand() % N;
        if (i == j) continue;
        double* xi = h_emb + i * D;
        double* xj = h_emb + j * D;
        double xx = 0, yy = 0, dxy = 0;
        for (int d = 0; d < D; d++) {
            xx += xi[d] * xi[d];
            yy += xj[d] * xj[d];
            double diff = xi[d] - xj[d];
            dxy += diff * diff;
        }
        double denom = (1.0 - xx) * (1.0 - yy);
        double arg = 1.0 + 2.0 * dxy / denom;
        double cpu_dist = acosh(fmax(arg, 1.0));
        double gpu_dist = h_dist[i * N + j];
        double rel_err = fabs(cpu_dist - gpu_dist) / (cpu_dist + 1e-10);
        if (rel_err > 1e-10) {
            printf("  MISMATCH (%d,%d): CPU=%.10f GPU=%.10f rel_err=%.2e\n",
                   i, j, cpu_dist, gpu_dist, rel_err);
            errors++;
        }
    }
    if (errors == 0) printf("Verification:    10 sample pairs match CPU ✓\n");

    // Stats
    double min_d = 1e10, max_d = 0, sum_d = 0;
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++) {
            double d = h_dist[i * N + j];
            if (d < min_d) min_d = d;
            if (d > max_d) max_d = d;
            sum_d += d;
        }
    printf("Distance range:  [%.4f, %.4f]\n", min_d, max_d);
    printf("Mean distance:   %.4f\n", sum_d / pairs);

    cudaFree(d_emb); cudaFree(d_dist);
    delete[] h_emb; delete[] h_dist;
    return 0;
}
