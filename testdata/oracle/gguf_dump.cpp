// The oracle for model/gguf: what llama.cpp's own reader (ggml's gguf.h)
// makes of a file, in a canonical text form -- floats as their bits,
// arrays as a count and an FNV-1a hash of their elements -- so the test
// can compare it line for line with what gguf.Open reads.
//
//   c++ -std=c++17 -I$LLAMA/ggml/include gguf_dump.cpp -L$LLAMA/build/bin -lggml-base -o gguf_dump
//   ./gguf_dump model.gguf > golden/model.txt
//   ./gguf_dump -dequant model.gguf | grep ^dequant > golden/model.dequant.txt
#include "gguf.h"
#include "ggml.h"
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static uint64_t fnv(uint64_t h, const void* p, size_t n) {
    const unsigned char* b = (const unsigned char*)p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

static size_t elem_size(gguf_type t) {
    switch (t) {
    case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8: case GGUF_TYPE_BOOL: return 1;
    case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: return 2;
    case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: return 4;
    default: return 8;
    }
}

static std::string scalar(const gguf_context* ctx, int64_t k, gguf_type t) {
    char buf[64];
    switch (t) {
    case GGUF_TYPE_UINT8:   snprintf(buf, sizeof buf, "%u", gguf_get_val_u8(ctx, k)); break;
    case GGUF_TYPE_INT8:    snprintf(buf, sizeof buf, "%d", gguf_get_val_i8(ctx, k)); break;
    case GGUF_TYPE_UINT16:  snprintf(buf, sizeof buf, "%u", gguf_get_val_u16(ctx, k)); break;
    case GGUF_TYPE_INT16:   snprintf(buf, sizeof buf, "%d", gguf_get_val_i16(ctx, k)); break;
    case GGUF_TYPE_UINT32:  snprintf(buf, sizeof buf, "%u", gguf_get_val_u32(ctx, k)); break;
    case GGUF_TYPE_INT32:   snprintf(buf, sizeof buf, "%d", gguf_get_val_i32(ctx, k)); break;
    case GGUF_TYPE_UINT64:  snprintf(buf, sizeof buf, "%" PRIu64, gguf_get_val_u64(ctx, k)); break;
    case GGUF_TYPE_INT64:   snprintf(buf, sizeof buf, "%" PRId64, gguf_get_val_i64(ctx, k)); break;
    case GGUF_TYPE_BOOL:    snprintf(buf, sizeof buf, "%s", gguf_get_val_bool(ctx, k) ? "true" : "false"); break;
    case GGUF_TYPE_FLOAT32: { float f = gguf_get_val_f32(ctx, k); uint32_t b; memcpy(&b, &f, 4); snprintf(buf, sizeof buf, "f32:%08x", b); break; }
    case GGUF_TYPE_FLOAT64: { double f = gguf_get_val_f64(ctx, k); uint64_t b; memcpy(&b, &f, 8); snprintf(buf, sizeof buf, "f64:%016" PRIx64, b); break; }
    case GGUF_TYPE_STRING:  return std::string("\"") + gguf_get_val_str(ctx, k) + "\"";
    default: return "?";
    }
    return buf;
}

int main(int argc, char** argv) {
    bool dequant = argc == 3 && strcmp(argv[1], "-dequant") == 0;
    if (dequant) { argv++; argc--; }
    if (argc != 2) { fprintf(stderr, "usage: gguf_dump [-dequant] file.gguf\n"); return 2; }
    gguf_init_params params = { /*no_alloc*/ true, /*ctx*/ nullptr };
    gguf_context* ctx = gguf_init_from_file(argv[1], params);
    if (!ctx) { printf("refused\n"); return 1; }
    printf("version %u\n", gguf_get_version(ctx));
    printf("alignment %zu\n", gguf_get_alignment(ctx));
    printf("data %zu\n", gguf_get_data_offset(ctx));
    for (int64_t k = 0; k < gguf_get_n_kv(ctx); k++) {
        gguf_type t = gguf_get_kv_type(ctx, k);
        printf("kv %s %s ", gguf_get_key(ctx, k), gguf_type_name(t));
        if (t != GGUF_TYPE_ARRAY) { printf("%s\n", scalar(ctx, k, t).c_str()); continue; }
        gguf_type et = gguf_get_arr_type(ctx, k);
        size_t n = gguf_get_arr_n(ctx, k);
        uint64_t h = 1469598103934665603ull;
        if (et == GGUF_TYPE_STRING) {
            for (size_t i = 0; i < n; i++) { const char* s = gguf_get_arr_str(ctx, k, i); h = fnv(h, s, strlen(s) + 1); }
        } else {
            h = fnv(h, gguf_get_arr_data(ctx, k), n * elem_size(et));
        }
        printf("%s x%zu fnv:%016" PRIx64 "\n", gguf_type_name(et), n, h);
    }
    FILE* f = fopen(argv[1], "rb");
    for (int64_t i = 0; i < gguf_get_n_tensors(ctx); i++) {
        size_t at = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, i);
        size_t size = gguf_get_tensor_size(ctx, i);
        std::vector<unsigned char> bytes(size);
        fseek(f, (long)at, SEEK_SET);
        size_t got = fread(bytes.data(), 1, size, f);
        printf("tensor %s %s at %zu size %zu fnv:%016" PRIx64 "\n", gguf_get_tensor_name(ctx, i),
               ggml_type_name(gguf_get_tensor_type(ctx, i)), at, size, fnv(1469598103934665603ull, bytes.data(), got));
        // With -dequant, a quantized tensor's float32 values as ggml's own
        // to_float makes them, hashed: what a device's decode must equal.
        ggml_type type = gguf_get_tensor_type(ctx, i);
        if (dequant && ggml_is_quantized(type)) {
            size_t n = size / ggml_type_size(type) * ggml_blck_size(type);
            std::vector<float> values(n);
            ggml_get_type_traits(type)->to_float(bytes.data(), values.data(), (int64_t)n);
            printf("dequant %s x%zu fnv:%016" PRIx64 "\n", gguf_get_tensor_name(ctx, i), n,
                   fnv(1469598103934665603ull, values.data(), n * sizeof(float)));
        }
    }
    fclose(f);
    gguf_free(ctx);
    return 0;
}
