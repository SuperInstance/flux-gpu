// bfs_kernel.cu — GPU BFS: connected components on constraint dependency graph
// Micro-benchmark: find crossover where GPU beats CPU for n=8,64,256,1024

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

// ── GPU BFS Kernel ───────────────────────────────────────────────────
__global__ void bfs_expand_kernel(
    const int* __restrict__ adj,
    const int* __restrict__ frontier,
    int* __restrict__ visited,
    int* __restrict__ next_frontier,
    int* __restrict__ changed,
    int n
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    if (!frontier[v]) return;

    for (int u = 0; u < n; u++) {
        if (adj[v * n + u] && !visited[u]) {
            int old = atomicExch(&visited[u], 1);
            if (!old) {
                next_frontier[u] = 1;
                *changed = 1;
            }
        }
    }
}

// ── CPU BFS ──────────────────────────────────────────────────────────
int cpu_bfs(const int* adj, int* visited, int n, int start) {
    memset(visited, 0, n * sizeof(int));
    visited[start] = 1;
    int* queue = new int[n];
    int head = 0, tail = 0;
    queue[tail++] = start;
    int count = 1;
    while (head < tail) {
        int v = queue[head++];
        for (int u = 0; u < n; u++) {
            if (adj[v * n + u] && !visited[u]) {
                visited[u] = 1;
                queue[tail++] = u;
                count++;
            }
        }
    }
    delete[] queue;
    return count;
}

// ── GPU BFS (multi-launch) ──────────────────────────────────────────
int gpu_bfs(const int* d_adj, int* d_visited, int* d_frontier, int* d_next,
            int* d_changed, int* h_buf, int n, int start) {
    cudaMemset(d_visited, 0, n * sizeof(int));
    cudaMemset(d_frontier, 0, n * sizeof(int));
    cudaMemset(d_next, 0, n * sizeof(int));

    int one = 1;
    cudaMemcpy((void*)&d_visited[start], &one, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)&d_frontier[start], &one, sizeof(int), cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (n + block - 1) / block;
    int h_changed;

    while (true) {
        cudaMemset(d_changed, 0, sizeof(int));
        cudaMemset(d_next, 0, n * sizeof(int));
        bfs_expand_kernel<<<grid, block>>>(
            d_adj, d_frontier, d_visited, d_next, d_changed, n);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
        if (!h_changed) break;

        // Swap frontiers by pointer swap (device pointers)
        int* tmp = d_frontier;
        d_frontier = d_next;
        d_next = tmp;
    }
    return 0;
}

// ── Benchmark ────────────────────────────────────────────────────────
void bench_size(int n, int n_iter = 200) {
    int* adj = new int[n * n];
    memset(adj, 0, n * n * sizeof(int));
    srand(42);
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (rand() % 5 == 0) {
                adj[i * n + j] = 1;
                adj[j * n + i] = 1;
            }
        }
    }

    int* cpu_visited = new int[n];
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < n_iter; it++) cpu_bfs(adj, cpu_visited, n, 0);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / n_iter;

    int *d_adj, *d_visited, *d_frontier, *d_next, *d_changed;
    int* h_buf = new int[n];
    cudaMalloc(&d_adj, n * n * sizeof(int));
    cudaMalloc(&d_visited, n * sizeof(int));
    cudaMalloc(&d_frontier, n * sizeof(int));
    cudaMalloc(&d_next, n * sizeof(int));
    cudaMalloc(&d_changed, sizeof(int));
    cudaMemcpy(d_adj, adj, n * n * sizeof(int), cudaMemcpyHostToDevice);

    // Warmup
    gpu_bfs(d_adj, d_visited, d_frontier, d_next, d_changed, h_buf, n, 0);
    cudaDeviceSynchronize();

    auto t2 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < n_iter; it++) {
        gpu_bfs(d_adj, d_visited, d_frontier, d_next, d_changed, h_buf, n, 0);
    }
    cudaDeviceSynchronize();
    auto t3 = std::chrono::high_resolution_clock::now();
    double gpu_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / n_iter;

    const char* winner = cpu_us < gpu_us ? "CPU" : "GPU";
    printf("| %6d | %10.1f | %10.1f | %12.2f | %s wins  |\n",
           n, cpu_us, gpu_us, cpu_us / gpu_us, winner);

    cudaFree(d_adj); cudaFree(d_visited); cudaFree(d_frontier);
    cudaFree(d_next); cudaFree(d_changed);
    delete[] adj; delete[] cpu_visited; delete[] h_buf;
}

int main() {
    printf("═══ bfs_kernel benchmark ═══\n");
    printf("| %6s | %10s | %10s | %12s | %s |\n", "n", "CPU (µs)", "GPU (µs)", "CPU/GPU", "Winner");
    printf("|--------|------------|------------|--------------|----------|\n");

    int sizes[] = {8, 64, 256, 1024};
    for (int s : sizes) bench_size(s);
    return 0;
}
