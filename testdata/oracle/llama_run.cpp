// The oracle for model/llama: llama.cpp (libllama, CPU only) given a
// prompt one token at a time, as the Vertex decoder takes it, then
// generating greedily. For each step: the token, then the top 5 logits and
// the logits' sum and sum of squares; then the greedy continuation.
//
//   c++ -std=c++17 -I$LLAMA/include -I$LLAMA/ggml/include llama_run.cpp -L$LLAMA/build/bin -lllama -o llama_run
//   ./llama_run model.gguf "Once upon a time" 32 > golden/model.txt
#include "llama.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 4) { fprintf(stderr, "usage: llama_run model.gguf prompt n\n"); return 2; }
    llama_log_set([](ggml_log_level, const char*, void*) {}, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 128; cp.n_batch = 1; cp.n_ubatch = 1; cp.n_threads = 1; cp.no_perf = true;
    // The KV cache in float32, as model/llama keeps it: llama.cpp's f16
    // default rounds every key and value.
    cp.type_k = GGML_TYPE_F32; cp.type_v = GGML_TYPE_F32;
    llama_context* ctx = llama_init_from_model(model, cp);
    std::string prompt = argv[2];
    std::vector<llama_token> toks(prompt.size() + 8);
    toks.resize(llama_tokenize(vocab, prompt.data(), (int)prompt.size(), toks.data(), (int)toks.size(), true, false));
    const int nv = llama_vocab_n_tokens(vocab);
    auto step = [&](llama_token t, int pos) -> llama_token {
        llama_batch b = llama_batch_get_one(&t, 1);
        (void)pos;
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); exit(1); }
        const float* l = llama_get_logits_ith(ctx, -1);
        std::vector<int> ix(nv);
        std::iota(ix.begin(), ix.end(), 0);
        std::partial_sort(ix.begin(), ix.begin() + 5, ix.end(), [&](int a, int b) { return l[a] > l[b] || (l[a] == l[b] && a < b); });
        double sum = 0, sq = 0;
        for (int i = 0; i < nv; i++) { sum += l[i]; sq += (double)l[i] * l[i]; }
        printf("step %d token %d top", pos, t);
        for (int i = 0; i < 5; i++) printf(" %d:%.6f", ix[i], l[ix[i]]);
        printf(" sum %.6f sq %.6f\n", sum, sq);
        return ix[0];
    };
    llama_token next = 0;
    int pos = 0;
    for (llama_token t : toks) next = step(t, pos++);
    std::vector<llama_token> gen;
    int n = atoi(argv[3]);
    for (int i = 0; i < n; i++) {
        gen.push_back(next);
        if (i + 1 < n) next = step(next, pos++);
    }
    printf("greedy");
    for (llama_token t : gen) printf(" %d", t);
    printf("\n");
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
