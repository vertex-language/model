// How long ggml's Metal backend takes for one matrix-vector product of a
// given type and shape: the reference the Vertex kernels are timed against
// (model's `ops`), 100 in a graph, as `ops` times 200 launches a wait.
// Weights quantized by ggml from random floats, x of ones.
//
//   c++ -std=c++17 -I$LLAMA/ggml/include mul_mat_time.cpp -L$LLAMA/build/bin -lggml -lggml-base -lggml-metal -o mul_mat_time
//   ./mul_mat_time q4_K 4096 768
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-metal.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 4) { fprintf(stderr, "usage: mul_mat_time type rows cols\n"); return 2; }
    ggml_type type = GGML_TYPE_COUNT;
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const char* n = ggml_type_name((ggml_type)t);
        if (n && strcmp(n, argv[1]) == 0) type = (ggml_type)t;
    }
    int64_t m = atoll(argv[2]), k = atoll(argv[3]);
    ggml_backend_t backend = ggml_backend_metal_init();
    const int per = 100;  // products a graph: timing one would time a submission
    ggml_init_params ip = { (per + 16) * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context* ctx = ggml_init(ip);
    ggml_tensor* w = ggml_new_tensor_2d(ctx, type, k, m);
    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, 1);
    ggml_cgraph* g = ggml_new_graph(ctx);
    for (int i = 0; i < per; i++) ggml_build_forward_expand(g, ggml_mul_mat(ctx, w, x));
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    // Random quants, with every f16 scale small and finite: dequantize
    // random floats through ggml's own quantizer.
    std::vector<float> f(k * m);
    for (auto& v : f) v = (float)rand() / (float)RAND_MAX - 0.5f;
    std::vector<char> q(ggml_nbytes(w));
    ggml_quantize_chunk(type, f.data(), q.data(), 0, m, k, nullptr);
    ggml_backend_tensor_set(w, q.data(), 0, q.size());
    std::vector<float> ones(k, 1.0f);
    ggml_backend_tensor_set(x, ones.data(), 0, k * sizeof(float));
    for (int i = 0; i < 20; i++) ggml_backend_graph_compute(backend, g);
    int runs = 20;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < runs; i++) ggml_backend_graph_compute(backend, g);
    double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / runs / per;
    printf("%s %lldx%lld: %.1fus, %.1f GB/s\n", argv[1], (long long)m, (long long)k, us, ggml_nbytes(w) / us / 1000);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
