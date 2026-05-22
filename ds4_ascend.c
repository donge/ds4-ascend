#include <acl/acl_rt.h>

#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_gpu.h"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#define DS4_ASCEND_MAX_DEVICES 2
#define DS4_ASCEND_N_LAYER 43u
#define DS4_ASCEND_N_EXPERT 256u
#define DS4_ASCEND_N_EXPERT_USED 6u
#define DS4_ASCEND_QK_K 256u
#define DS4_ASCEND_TENSOR_Q8_0 8u
#define DS4_ASCEND_TENSOR_Q2_K 10u
#define DS4_ASCEND_TENSOR_Q4_K 12u
#define DS4_ASCEND_TENSOR_IQ2_XXS 16u
#define DS4_ASCEND_QK8_0 32u
#define DS4_ASCEND_BLOCK_Q8_0_BYTES 34u
#define DS4_ASCEND_BLOCK_Q2_K_BYTES 84u
#define DS4_ASCEND_BLOCK_Q4_K_BYTES 144u
#define DS4_ASCEND_BLOCK_Q8_K_BYTES 292u
#define DS4_ASCEND_BLOCK_IQ2_XXS_BYTES 66u
#define DS4_ASCEND_DENSE_CACHE_LIMIT_BYTES (2ull * 1024ull * 1024ull * 1024ull)

#define __device__ static
#define __constant__ const
#include "ds4_iq2_tables_cuda.inc"
#undef __constant__
#undef __device__

typedef struct {
    uint8_t scales[DS4_ASCEND_QK_K / 16];
    uint8_t qs[DS4_ASCEND_QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} ascend_block_q2_K;

typedef struct {
    float d;
    int8_t qs[DS4_ASCEND_QK_K];
    int16_t bsums[DS4_ASCEND_QK_K / 16];
} ascend_block_q8_K;

typedef struct {
    uint16_t d;
    uint16_t qs[DS4_ASCEND_QK_K / 8];
} ascend_block_iq2_xxs;

struct ds4_gpu_tensor {
    void *ptr;
    uint64_t bytes;
    int owner;
    int device;
    void *host_shadow;
    uint64_t host_shadow_bytes;
};

typedef enum {
    DS4_ASCEND_WEIGHT_DENSE,
    DS4_ASCEND_WEIGHT_GATE_EXPERT,
    DS4_ASCEND_WEIGHT_UP_EXPERT,
    DS4_ASCEND_WEIGHT_DOWN_EXPERT,
} ascend_weight_kind;

typedef struct {
    uint64_t model_offset;
    uint64_t bytes;
    uint64_t in_dim;
    uint64_t out_dim;
    uint32_t type;
    uint32_t layer;
    ascend_weight_kind kind;
    ds4_gpu_tensor *device[DS4_ASCEND_MAX_DEVICES];
    uint32_t expert_begin[DS4_ASCEND_MAX_DEVICES];
    uint32_t expert_count[DS4_ASCEND_MAX_DEVICES];
    uint64_t device_offset[DS4_ASCEND_MAX_DEVICES];
    uint64_t device_bytes[DS4_ASCEND_MAX_DEVICES];
} ascend_weight_cache;

static int g_initialized;
static uint32_t g_device_count;
static aclrtContext g_contexts[DS4_ASCEND_MAX_DEVICES];
static int g_context_ready[DS4_ASCEND_MAX_DEVICES];
static int g_default_device;
static const void *g_model_map;
static uint64_t g_model_size;
static int g_model_fd = -1;
static int g_quality_mode;
static uint64_t g_live_bytes[DS4_ASCEND_MAX_DEVICES];
static uint64_t g_peak_bytes[DS4_ASCEND_MAX_DEVICES];
static ascend_weight_cache *g_weight_cache;
static uint64_t g_weight_cache_count;
static uint64_t g_weight_cache_cap;
static uint64_t g_weight_cache_bytes[DS4_ASCEND_MAX_DEVICES];
static uint64_t g_expert_cache_bytes[DS4_ASCEND_MAX_DEVICES];
static uint32_t g_expert_shard_point = DS4_ASCEND_N_EXPERT / 2;
static aclrtStream g_streams[DS4_ASCEND_MAX_DEVICES];
static ds4_gpu_tensor *g_iq2_ksigns_dev[DS4_ASCEND_MAX_DEVICES];
static ds4_gpu_tensor *g_iq2_grid_dev[DS4_ASCEND_MAX_DEVICES];
static ds4_gpu_tensor **g_deferred_free;
static uint64_t g_deferred_free_count;
static uint64_t g_deferred_free_cap;

typedef struct {
    const char *name;
    uint64_t calls;
    uint64_t bytes_to_host;
    uint64_t bytes_to_device;
} ascend_fallback_stat;

static ascend_fallback_stat g_fallback_stats[128];
static uint32_t g_fallback_stat_count;
static uint64_t g_fallback_total_calls;
static uint64_t g_fallback_total_bytes_to_host;
static uint64_t g_fallback_total_bytes_to_device;
static int g_no_host_fallback = -1;

void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor);
static ds4_gpu_tensor *ascend_tensor_alloc_on_device(uint64_t bytes, int device);
extern void ds4_ascend_launch_fill_f32(void *stream, void *out, float value, uint32_t count);
extern void ds4_ascend_launch_add_f32(void *stream, void *out, const void *a, const void *b, uint32_t count);
extern void ds4_ascend_launch_quantize_q8_k(void *stream, void *out, const void *x, uint32_t rows, uint32_t cols);
extern void ds4_ascend_launch_matmul_f16(void *stream, void *out, const void *w, const void *x, uint32_t in_dim, uint32_t out_dim, uint32_t n_tok);
extern void ds4_ascend_launch_matmul_q8_0(void *stream, void *out, const void *w, const void *x, uint32_t in_dim, uint32_t out_dim, uint32_t n_tok);
extern void ds4_ascend_launch_quantize_q8_0(void *stream, void *xq, void *xscale, const void *x, uint32_t in_dim, uint32_t n_tok);
extern void ds4_ascend_launch_matmul_q8_0_prequant(void *stream, void *out, const void *w, const void *xq, const void *xscale, uint32_t in_dim, uint32_t out_dim, uint32_t n_tok);
extern void ds4_ascend_launch_embed_token_hc(void *stream, void *out, const void *w, uint32_t token, uint32_t n_embd, uint32_t n_hc);
extern void ds4_ascend_launch_embed_tokens_hc(void *stream, void *out, const void *tokens, const void *w, uint32_t n_vocab, uint32_t n_tokens, uint32_t n_embd, uint32_t n_hc);
extern void ds4_ascend_launch_rms_norm_plain(void *stream, void *out, const void *x, uint32_t n, uint32_t rows, float eps);
extern void ds4_ascend_launch_rms_norm_weight(void *stream, void *out, const void *x, const void *weight, uint32_t n, uint32_t rows, float eps);
extern void ds4_ascend_launch_rms_norm_inplace(void *stream, void *x, uint32_t n, uint32_t rows, float eps);
extern void ds4_ascend_launch_hc_expand_split(void *stream, void *out, const void *block_out, const void *block_add, const void *residual, const void *split, uint32_t n_embd, uint32_t n_hc, uint32_t rows, uint32_t has_add);
extern void ds4_ascend_launch_hc_expand(void *stream, void *out, const void *block_out, const void *residual, const void *post, const void *comb, uint32_t n_embd, uint32_t n_hc, uint32_t rows);
extern void ds4_ascend_launch_hc_split_sinkhorn4(void *stream, void *out, const void *mix, const void *scale, const void *base, uint32_t rows, uint32_t sinkhorn_iters, float eps);
extern void ds4_ascend_launch_hc_weighted_sum4(void *stream, void *out, const void *residual, const void *weights, uint32_t n_embd, uint32_t rows, uint32_t weight_stride);
extern void ds4_ascend_launch_output_hc_weights(void *stream, void *out, const void *pre, const void *scale, const void *base, uint32_t n_hc, uint32_t n_tokens, float eps);
extern void ds4_ascend_launch_rope_tail_table(void *stream, void *x, const void *table, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot);
extern void ds4_ascend_launch_fp8_kv_quantize(void *stream, void *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot);
extern void ds4_ascend_launch_store_raw_kv_batch(void *stream, void *raw, const void *kv, uint32_t raw_cap, uint32_t pos0, uint32_t n_tokens, uint32_t head_dim);
extern void ds4_ascend_launch_attention_prefill_raw(void *stream, void *heads, const void *sinks, const void *q, const void *raw_kv, uint32_t n_tokens, uint32_t window, uint32_t n_head, uint32_t head_dim, float scale);
extern void ds4_ascend_launch_attention_decode(void *stream, void *heads, const void *sinks, const void *q, const void *raw_kv, const void *comp_kv, const void *comp_mask, uint32_t use_comp_mask, uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start, uint32_t n_comp, uint32_t n_head, uint32_t head_dim, float scale);
extern void ds4_ascend_launch_attention_prefill_mixed(void *stream, void *heads, const void *sinks, const void *q, const void *raw_kv, const void *comp_kv, const void *comp_mask, uint32_t use_comp_mask, uint32_t n_tokens, uint32_t n_comp, uint32_t window, uint32_t ratio, uint32_t n_head, uint32_t head_dim, float scale);
extern void ds4_ascend_launch_attention_output_low_q8(void *stream, void *low, const void *w, const void *heads, uint32_t group_dim, uint32_t rank, uint32_t n_groups, uint32_t n_tokens);
extern void ds4_ascend_launch_attention_output_low_q8_prequant(void *stream, void *low, const void *w, const void *xq, const void *xscale, uint32_t group_dim, uint32_t rank, uint32_t n_groups, uint32_t n_tokens);
extern void ds4_ascend_launch_router_select(void *stream, void *selected, void *weights, void *probs, const void *bias, const void *hash, const void *logits, const void *tokens, int32_t token_scalar, uint32_t hash_rows, uint32_t n_tokens, uint32_t has_bias, uint32_t hash_mode);
extern void ds4_ascend_launch_moe_gate_up_mid_iq2_q8(void *stream, void *gate_out, void *up_out, void *mid_out, const void *gate_w, const void *up_w, const void *xq, const void *selected, const void *weights, const void *ksigns, const void *grid, uint32_t pair_count, uint32_t n_expert, uint32_t expert_begin, uint32_t expert_count, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, float clamp);
extern void ds4_ascend_launch_moe_down_q2_q8(void *stream, void *experts_out, const void *down_w, const void *midq, const void *selected, uint32_t pair_count, uint32_t expert_begin, uint32_t expert_count, uint32_t expert_mid_dim, uint32_t out_dim, uint64_t down_expert_bytes, uint64_t down_row_bytes);
extern void ds4_ascend_launch_moe_merge_width(void *stream, void *dst, const void *src, const void *selected, uint32_t pair_count, uint32_t width, uint32_t expert_begin, uint32_t expert_count);
extern void ds4_ascend_launch_moe_sum_experts(void *stream, void *out, const void *experts, uint32_t n_tokens, uint32_t n_expert, uint32_t out_dim);
extern void ds4_ascend_launch_swiglu(void *stream, void *out, const void *gate, const void *up, uint32_t n, float clamp, float weight);
extern void ds4_ascend_launch_compressor_set_rows(void *stream, void *state_kv, void *state_score, const void *kv, const void *sc, const void *ape, uint32_t ape_type, uint32_t width, uint32_t ratio, uint32_t pos0, uint32_t src0, uint32_t dst0, uint32_t rows);
extern void ds4_ascend_launch_compressor_prefill_pool(void *stream, void *comp, const void *kv, const void *sc, const void *state_kv, const void *state_score, const void *ape, uint32_t ape_type, uint32_t head_dim, uint32_t ratio, uint32_t pos0, uint32_t n_comp, uint32_t replay);
extern void ds4_ascend_launch_compressor_update_pool(void *stream, void *row, const void *state_kv, const void *state_score, uint32_t head_dim, uint32_t ratio);
extern void ds4_ascend_launch_compressor_shift_ratio4(void *stream, void *state_kv, void *state_score, uint32_t width);
extern aclError aclrtMemcpyAsync(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind, aclrtStream stream);
extern aclError aclrtMemset(void *devPtr, size_t maxCount, int32_t value, size_t count);
extern aclError aclrtDeviceCanAccessPeer(int32_t *canAccessPeer, int32_t deviceId, int32_t peerDeviceId);
extern aclError aclrtDeviceEnablePeerAccess(int32_t peerDeviceId, uint32_t flags);
extern aclError aclrtDevicePeerAccessStatus(int32_t deviceId, int32_t peerDeviceId, int32_t *status);

static int ascend_ok(aclError err, const char *what) {
    if (err == ACL_ERROR_NONE) return 1;
    fprintf(stderr, "ds4: AscendCL %s failed: error %d\n", what ? what : "operation", (int)err);
    return 0;
}

static int ascend_set_context(int device) {
    if (device < 0 || device >= (int)g_device_count || !g_context_ready[device]) return 0;
    return ascend_ok(aclrtSetCurrentContext(g_contexts[device]), "set current context");
}

static int ascend_unimplemented(const char *what) {
    fprintf(stderr, "ds4: Ascend backend primitive not implemented yet: %s\n", what ? what : "unknown");
    return 0;
}

static int ascend_no_host_fallback(void) {
    if (g_no_host_fallback < 0) {
        const char *env = getenv("DS4_ASCEND_NO_HOST_FALLBACK");
        g_no_host_fallback = env && env[0] && strcmp(env, "0") != 0;
    }
    return g_no_host_fallback;
}

static void ascend_note_fallback(const char *name, uint64_t bytes_to_host, uint64_t bytes_to_device) {
    if (!name) name = "unknown";
    g_fallback_total_calls++;
    g_fallback_total_bytes_to_host += bytes_to_host;
    g_fallback_total_bytes_to_device += bytes_to_device;
    for (uint32_t i = 0; i < g_fallback_stat_count; i++) {
        if (strcmp(g_fallback_stats[i].name, name) == 0) {
            g_fallback_stats[i].calls++;
            g_fallback_stats[i].bytes_to_host += bytes_to_host;
            g_fallback_stats[i].bytes_to_device += bytes_to_device;
            return;
        }
    }
    if (g_fallback_stat_count < (uint32_t)(sizeof(g_fallback_stats) / sizeof(g_fallback_stats[0]))) {
        ascend_fallback_stat *s = &g_fallback_stats[g_fallback_stat_count++];
        s->name = name;
        s->calls = 1;
        s->bytes_to_host = bytes_to_host;
        s->bytes_to_device = bytes_to_device;
    }
}

static int ascend_host_fallback_begin(const char *name, uint64_t bytes_to_host, uint64_t bytes_to_device) {
    if (ascend_no_host_fallback()) {
        fprintf(stderr, "ds4: Ascend host fallback disabled: %s\n", name ? name : "unknown");
        return 0;
    }
    ascend_note_fallback(name, bytes_to_host, bytes_to_device);
    return 1;
}

static void ascend_report_fallback_stats(void) {
    const char *env = getenv("DS4_ASCEND_TRACE_FALLBACKS");
    if (!env || !env[0] || strcmp(env, "0") == 0) return;
    fprintf(stderr,
            "ds4: Ascend host fallback summary: calls=%" PRIu64 " to_host=%.2f MiB to_device=%.2f MiB\n",
            g_fallback_total_calls,
            (double)g_fallback_total_bytes_to_host / 1048576.0,
            (double)g_fallback_total_bytes_to_device / 1048576.0);
    for (uint32_t i = 0; i < g_fallback_stat_count; i++) {
        const ascend_fallback_stat *s = &g_fallback_stats[i];
        fprintf(stderr,
                "ds4: Ascend host fallback: %s calls=%" PRIu64 " to_host=%.2f MiB to_device=%.2f MiB\n",
                s->name,
                s->calls,
                (double)s->bytes_to_host / 1048576.0,
                (double)s->bytes_to_device / 1048576.0);
    }
}

static int ascend_create_streams(void) {
    for (uint32_t i = 0; i < g_device_count; i++) {
        if (!ascend_set_context((int)i)) return 0;
        if (!ascend_ok(aclrtCreateStream(&g_streams[i]), "create stream")) return 0;
    }
    return ascend_set_context(g_default_device);
}

static int ascend_enable_peer_access(void) {
    for (uint32_t d = 0; d < g_device_count; d++) {
        if (!ascend_set_context((int)d)) return 0;
        for (uint32_t p = 0; p < g_device_count; p++) {
            if (p == d) continue;
            int32_t can_access = 0;
            if (!ascend_ok(aclrtDeviceCanAccessPeer(&can_access, (int32_t)d, (int32_t)p), "check peer access")) return 0;
            if (!can_access) continue;
            int32_t status = 0;
            aclError status_err = aclrtDevicePeerAccessStatus((int32_t)d, (int32_t)p, &status);
            if (status_err == ACL_ERROR_NONE && status) continue;
            aclError err = aclrtDeviceEnablePeerAccess((int32_t)p, 0);
            if (err != ACL_ERROR_NONE) {
                int32_t after_status = 0;
                if (aclrtDevicePeerAccessStatus((int32_t)d, (int32_t)p, &after_status) != ACL_ERROR_NONE || !after_status) {
                    return ascend_ok(err, "enable peer access");
                }
            }
        }
    }
    return ascend_set_context(g_default_device);
}

static void ascend_destroy_streams(void) {
    for (uint32_t i = 0; i < DS4_ASCEND_MAX_DEVICES; i++) {
        if (g_streams[i]) {
            (void)ascend_set_context((int)i);
            (void)aclrtSynchronizeStream(g_streams[i]);
            (void)aclrtDestroyStream(g_streams[i]);
            g_streams[i] = NULL;
        }
    }
}

static void ascend_iq2_tables_reset(void) {
    for (uint32_t i = 0; i < DS4_ASCEND_MAX_DEVICES; i++) {
        ds4_gpu_tensor_free(g_iq2_ksigns_dev[i]);
        ds4_gpu_tensor_free(g_iq2_grid_dev[i]);
        g_iq2_ksigns_dev[i] = NULL;
        g_iq2_grid_dev[i] = NULL;
    }
}

static int ascend_require_iq2_tables(int device) {
    if (device < 0 || device >= (int)g_device_count) return 0;
    if (g_iq2_ksigns_dev[device] && g_iq2_grid_dev[device]) return 1;
    g_iq2_ksigns_dev[device] = ascend_tensor_alloc_on_device(sizeof(cuda_ksigns_iq2xs), device);
    g_iq2_grid_dev[device] = ascend_tensor_alloc_on_device(sizeof(cuda_iq2xxs_grid), device);
    if (!g_iq2_ksigns_dev[device] || !g_iq2_grid_dev[device]) return 0;
    if (!ds4_gpu_tensor_write(g_iq2_ksigns_dev[device], 0, cuda_ksigns_iq2xs, sizeof(cuda_ksigns_iq2xs))) return 0;
    if (!ds4_gpu_tensor_write(g_iq2_grid_dev[device], 0, cuda_iq2xxs_grid, sizeof(cuda_iq2xxs_grid))) return 0;
    return 1;
}

static int ascend_defer_free_tensor(ds4_gpu_tensor *tensor) {
    if (!tensor) return 1;
    if (g_deferred_free_count == g_deferred_free_cap) {
        uint64_t cap = g_deferred_free_cap ? g_deferred_free_cap * 2 : 256;
        ds4_gpu_tensor **p = realloc(g_deferred_free, (size_t)(cap * sizeof(*p)));
        if (!p) return 0;
        g_deferred_free = p;
        g_deferred_free_cap = cap;
    }
    g_deferred_free[g_deferred_free_count++] = tensor;
    return 1;
}

static void ascend_drain_deferred_frees(void) {
    for (uint64_t i = 0; i < g_deferred_free_count; i++) ds4_gpu_tensor_free(g_deferred_free[i]);
    g_deferred_free_count = 0;
}

static void ascend_deferred_frees_reset(void) {
    ascend_drain_deferred_frees();
    free(g_deferred_free);
    g_deferred_free = NULL;
    g_deferred_free_cap = 0;
}

static int ascend_launch_fill_f32(ds4_gpu_tensor *tensor, float value, uint32_t count) {
    if (!tensor || count == 0) return tensor != NULL;
    if ((uint64_t)count * sizeof(float) > tensor->bytes) return 0;
    if (!ascend_set_context(tensor->device) || !g_streams[tensor->device]) return 0;
    ds4_ascend_launch_fill_f32(g_streams[tensor->device], tensor->ptr, value, count);
    return 1;
}

static int ascend_parse_device_env(void) {
    const char *env = getenv("DS4_ASCEND_ALLOC_DEVICE");
    if (!env || !env[0]) return 0;
    char *end = NULL;
    long v = strtol(env, &end, 10);
    if (end == env || *end != '\0' || v < 0 || v >= DS4_ASCEND_MAX_DEVICES) return 0;
    return (int)v;
}

static uint32_t ascend_parse_shard_point(void) {
    const char *env = getenv("DS4_ASCEND_EXPERT_SHARD_POINT");
    if (!env || !env[0]) return DS4_ASCEND_N_EXPERT / 2;
    char *end = NULL;
    unsigned long v = strtoul(env, &end, 10);
    if (end == env || *end != '\0' || v == 0 || v >= DS4_ASCEND_N_EXPERT) return DS4_ASCEND_N_EXPERT / 2;
    return (uint32_t)v;
}

static uint64_t ascend_quant_block_bytes(uint32_t type) {
    switch (type) {
    case DS4_ASCEND_TENSOR_IQ2_XXS: return DS4_ASCEND_BLOCK_IQ2_XXS_BYTES;
    case DS4_ASCEND_TENSOR_Q2_K: return DS4_ASCEND_BLOCK_Q2_K_BYTES;
    case DS4_ASCEND_TENSOR_Q4_K: return DS4_ASCEND_BLOCK_Q4_K_BYTES;
    default: return 0;
    }
}

static uint64_t ascend_routed_row_bytes(uint32_t type, uint64_t in_dim) {
    const uint64_t block = ascend_quant_block_bytes(type);
    if (block == 0 || in_dim == 0 || (in_dim % DS4_ASCEND_QK_K) != 0) return 0;
    return (in_dim / DS4_ASCEND_QK_K) * block;
}

static ds4_gpu_tensor *ascend_tensor_alloc_on_device(uint64_t bytes, int device) {
    if (!g_initialized && !ds4_gpu_init()) return NULL;
    if (device < 0 || device >= (int)g_device_count) return NULL;
    if (bytes == 0) bytes = 1;
    if (!ascend_set_context(device)) return NULL;

    ds4_gpu_tensor *t = calloc(1, sizeof(*t));
    if (!t) return NULL;
    aclError err = aclrtMalloc(&t->ptr, (size_t)bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (err != ACL_ERROR_NONE) {
        fprintf(stderr, "ds4: Ascend tensor alloc failed on device %d (%.2f MiB): error %d\n",
                device, (double)bytes / 1048576.0, (int)err);
        free(t);
        return NULL;
    }
    t->bytes = bytes;
    t->owner = 1;
    t->device = device;
    g_live_bytes[device] += bytes;
    if (g_live_bytes[device] > g_peak_bytes[device]) g_peak_bytes[device] = g_live_bytes[device];
    return t;
}

static void ascend_weight_cache_reset(void) {
    for (uint64_t i = 0; i < g_weight_cache_count; i++) {
        for (uint32_t d = 0; d < DS4_ASCEND_MAX_DEVICES; d++) {
            ds4_gpu_tensor_free(g_weight_cache[i].device[d]);
            g_weight_cache[i].device[d] = NULL;
        }
    }
    free(g_weight_cache);
    g_weight_cache = NULL;
    g_weight_cache_count = 0;
    g_weight_cache_cap = 0;
    memset(g_weight_cache_bytes, 0, sizeof(g_weight_cache_bytes));
    memset(g_expert_cache_bytes, 0, sizeof(g_expert_cache_bytes));
}

static ascend_weight_cache *ascend_weight_cache_find(uint64_t offset, uint64_t bytes) {
    for (uint64_t i = 0; i < g_weight_cache_count; i++) {
        if (g_weight_cache[i].model_offset == offset && g_weight_cache[i].bytes == bytes) return &g_weight_cache[i];
    }
    return NULL;
}

static ascend_weight_cache *ascend_weight_cache_add(uint64_t offset, uint64_t bytes) {
    ascend_weight_cache *existing = ascend_weight_cache_find(offset, bytes);
    if (existing) return existing;
    if (g_weight_cache_count == g_weight_cache_cap) {
        uint64_t cap = g_weight_cache_cap ? g_weight_cache_cap * 2 : 256;
        ascend_weight_cache *p = realloc(g_weight_cache, (size_t)(cap * sizeof(*p)));
        if (!p) return NULL;
        g_weight_cache = p;
        g_weight_cache_cap = cap;
    }
    ascend_weight_cache *w = &g_weight_cache[g_weight_cache_count++];
    memset(w, 0, sizeof(*w));
    w->model_offset = offset;
    w->bytes = bytes;
    return w;
}

static bool ascend_parse_expert_label(const char *label, uint32_t *layer, ascend_weight_kind *kind) {
    if (!label || !layer || !kind) return false;
    const char *name = label;
    if (strncmp(name, "tensor:", 7) == 0) name += 7;
    unsigned long il = 0;
    char suffix[96];
    if (sscanf(name, "blk.%lu.%95s", &il, suffix) != 2) return false;
    if (il >= DS4_ASCEND_N_LAYER) return false;
    if (strcmp(suffix, "ffn_gate_exps.weight") == 0) *kind = DS4_ASCEND_WEIGHT_GATE_EXPERT;
    else if (strcmp(suffix, "ffn_up_exps.weight") == 0) *kind = DS4_ASCEND_WEIGHT_UP_EXPERT;
    else if (strcmp(suffix, "ffn_down_exps.weight") == 0) *kind = DS4_ASCEND_WEIGHT_DOWN_EXPERT;
    else return false;
    *layer = (uint32_t)il;
    return true;
}

static inline float ascend_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
        }
    } else if (exp == 31u) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static inline uint16_t ascend_f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        uint32_t hmant = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) hmant++;
        return (uint16_t)(sign | hmant);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint32_t h = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x00001000u) h++;
    return (uint16_t)h;
}

static inline float ascend_f16_round_f32(float f) {
    return ascend_f16_to_f32(ascend_f32_to_f16(f));
}

static inline float ascend_silu(float x) {
    return x / (1.0f + expf(-x));
}

static int32_t ascend_dot_q2_16(const uint8_t *q2, const int8_t *q8, int shift) {
    int32_t s = 0;
    for (uint32_t i = 0; i < 16; i++) s += (int32_t)((q2[i] >> shift) & 3u) * (int32_t)q8[i];
    return s;
}

static float ascend_vec_dot_q2_K_q8_K(uint32_t n, const ascend_block_q2_K *x, const ascend_block_q8_K *y) {
    const uint32_t nb = n / DS4_ASCEND_QK_K;
    float sumf = 0.0f;
    for (uint32_t i = 0; i < nb; i++) {
        const uint8_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        const uint8_t *sc = x[i].scales;
        int summs = 0;
        for (uint32_t j = 0; j < 16; j++) summs += y[i].bsums[j] * (sc[j] >> 4);
        const float dall = y[i].d * ascend_f16_to_f32(x[i].d);
        const float dmin = y[i].d * ascend_f16_to_f32(x[i].dmin);
        int isum = 0;
        int is = 0;
        for (uint32_t k = 0; k < DS4_ASCEND_QK_K / 128; k++) {
            int shift = 0;
            for (uint32_t j = 0; j < 4; j++) {
                int d = sc[is++] & 0x0f;
                isum += d * ascend_dot_q2_16(q2, q8, shift);
                d = sc[is++] & 0x0f;
                isum += d * ascend_dot_q2_16(q2 + 16, q8 + 16, shift);
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
        sumf += dall * (float)isum - dmin * (float)summs;
    }
    return sumf;
}

static float ascend_vec_dot_iq2_xxs_q8_K(uint32_t n, const ascend_block_iq2_xxs *x, const ascend_block_q8_K *y) {
    const uint32_t nb = n / DS4_ASCEND_QK_K;
    float sumf = 0.0f;
    for (uint32_t i = 0; i < nb; i++) {
        const float d = ascend_f16_to_f32(x[i].d) * y[i].d;
        const uint16_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        int32_t bsum = 0;
        for (uint32_t ib32 = 0; ib32 < DS4_ASCEND_QK_K / 32; ib32++) {
            const uint32_t aux_g = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
            const uint32_t aux_s = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
            q2 += 4;
            const uint32_t ls = 2u * (aux_s >> 28) + 1u;
            int32_t sumi = 0;
            for (uint32_t l = 0; l < 4; l += 2) {
                const uint8_t grid0 = (uint8_t)((aux_g >> (8u * l)) & 0xffu);
                const uint8_t grid1 = (uint8_t)((aux_g >> (8u * (l + 1))) & 0xffu);
                const uint32_t sign0 = (aux_s >> (7u * l)) & 127u;
                const uint32_t sign1 = (aux_s >> (7u * (l + 1))) & 127u;
                const uint64_t g0 = cuda_iq2xxs_grid[grid0];
                const uint64_t g1 = cuda_iq2xxs_grid[grid1];
                const uint8_t s0 = cuda_ksigns_iq2xs[sign0];
                const uint8_t s1 = cuda_ksigns_iq2xs[sign1];
                for (uint32_t j = 0; j < 8; j++) {
                    int32_t w0 = (int32_t)((g0 >> (8u * j)) & 0xffu);
                    int32_t w1 = (int32_t)((g1 >> (8u * j)) & 0xffu);
                    if (s0 & (1u << j)) w0 = -w0;
                    if (s1 & (1u << j)) w1 = -w1;
                    sumi += w0 * (int32_t)q8[j] + w1 * (int32_t)q8[8u + j];
                }
                q8 += 16;
            }
            bsum += sumi * (int32_t)ls;
        }
        sumf += d * (float)bsum;
    }
    return 0.125f * sumf;
}

static const char *ascend_model_range_ptr(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, const char *label) {
    if (!model_map || offset > model_size || bytes > model_size - offset) {
        fprintf(stderr, "ds4: Ascend invalid model range for %s\n", label ? label : "tensor");
        return NULL;
    }
    return (const char *)model_map + offset;
}

static int ascend_cache_kind_matches(uint64_t offset, uint64_t bytes, ascend_weight_kind kind) {
    ascend_weight_cache *w = ascend_weight_cache_find(offset, bytes);
    return !w || w->kind == DS4_ASCEND_WEIGHT_DENSE || w->kind == kind;
}

static int ascend_cache_tensor_to_device(ascend_weight_cache *w, const void *model_map, uint64_t model_size, int device, uint64_t offset, uint64_t bytes);

static int ascend_weight_cache_is_expert(const ascend_weight_cache *w) {
    return w && (w->kind == DS4_ASCEND_WEIGHT_GATE_EXPERT ||
                 w->kind == DS4_ASCEND_WEIGHT_UP_EXPERT ||
                 w->kind == DS4_ASCEND_WEIGHT_DOWN_EXPERT);
}

static void ascend_evict_dense_weight_cache_for_device(int device, uint64_t needed) {
    if (device < 0 || device >= (int)g_device_count) return;
    while (g_weight_cache_bytes[device] + needed > DS4_ASCEND_DENSE_CACHE_LIMIT_BYTES) {
        ascend_weight_cache *victim = NULL;
        for (uint64_t i = 0; i < g_weight_cache_count; i++) {
            ascend_weight_cache *w = &g_weight_cache[i];
            if (!w->device[device] || ascend_weight_cache_is_expert(w)) continue;
            victim = w;
            break;
        }
        if (!victim) return;
        if (g_streams[device]) {
            (void)ascend_set_context(device);
            (void)aclrtSynchronizeStream(g_streams[device]);
        }
        const uint64_t bytes = victim->device_bytes[device];
        const int expert = ascend_weight_cache_is_expert(victim);
        ds4_gpu_tensor_free(victim->device[device]);
        victim->device[device] = NULL;
        victim->device_offset[device] = 0;
        victim->device_bytes[device] = 0;
        if (!expert) {
            if (g_weight_cache_bytes[device] >= bytes) g_weight_cache_bytes[device] -= bytes;
            else g_weight_cache_bytes[device] = 0;
        }
    }
}

static void *ascend_model_range_device_ptr(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, int device, const char *label) {
    if (!model_map || offset > model_size || bytes > model_size - offset || device < 0 || device >= (int)g_device_count) {
        fprintf(stderr, "ds4: Ascend invalid device model range for %s\n", label ? label : "tensor");
        return NULL;
    }
    ascend_weight_cache *w = ascend_weight_cache_add(offset, bytes);
    if (!w) return NULL;
    if (w->device[device]) {
        if (w->device_offset[device] == offset && w->device_bytes[device] == bytes) return w->device[device]->ptr;
        fprintf(stderr, "ds4: Ascend overlapping cached model range for %s\n", label ? label : "tensor");
        return NULL;
    }
    ascend_evict_dense_weight_cache_for_device(device, bytes);
    if (!ascend_cache_tensor_to_device(w, model_map, model_size, device, offset, bytes)) return NULL;
    return w->device[device]->ptr;
}

static int ascend_cache_tensor_to_device(ascend_weight_cache *w, const void *model_map, uint64_t model_size, int device, uint64_t offset, uint64_t bytes) {
    if (!w || !model_map || offset > model_size || bytes > model_size - offset) return 0;
    ds4_gpu_tensor *t = ascend_tensor_alloc_on_device(bytes, device);
    if (!t) return 0;
    if (!ds4_gpu_tensor_write(t, 0, (const uint8_t *)model_map + offset, bytes)) {
        ds4_gpu_tensor_free(t);
        return 0;
    }
    w->device[device] = t;
    w->device_offset[device] = offset;
    w->device_bytes[device] = bytes;
    if (w->kind == DS4_ASCEND_WEIGHT_GATE_EXPERT ||
        w->kind == DS4_ASCEND_WEIGHT_UP_EXPERT ||
        w->kind == DS4_ASCEND_WEIGHT_DOWN_EXPERT) {
        g_expert_cache_bytes[device] += bytes;
    } else {
        g_weight_cache_bytes[device] += bytes;
    }
    return 1;
}

static int ascend_cache_routed_expert_tensor(
        ascend_weight_cache *w,
        const void *model_map,
        uint64_t model_size,
        uint64_t offset,
        uint64_t bytes,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t type,
        const char *label) {
    if (!w || !model_map || offset > model_size || bytes > model_size - offset) return 0;
    if (g_device_count < 2) {
        fprintf(stderr, "ds4: Ascend routed expert cache needs two visible devices for %s\n", label ? label : "tensor");
        return 0;
    }
    const uint64_t row_bytes = ascend_routed_row_bytes(type, in_dim);
    if (row_bytes == 0) {
        fprintf(stderr, "ds4: Ascend unsupported routed expert type/dim for %s (type=%u, in_dim=%" PRIu64 ")\n",
                label ? label : "tensor", type, in_dim);
        return 0;
    }
    const uint64_t expert_bytes = out_dim * row_bytes;
    if (expert_bytes == 0 || bytes != expert_bytes * DS4_ASCEND_N_EXPERT) {
        fprintf(stderr,
                "ds4: Ascend routed expert size mismatch for %s: bytes=%" PRIu64 " expert_bytes=%" PRIu64 "\n",
                label ? label : "tensor", bytes, expert_bytes);
        return 0;
    }

    w->in_dim = in_dim;
    w->out_dim = out_dim;
    w->type = type;

    const uint32_t split = g_expert_shard_point;
    const uint32_t begin[DS4_ASCEND_MAX_DEVICES] = {0, split};
    const uint32_t count[DS4_ASCEND_MAX_DEVICES] = {split, DS4_ASCEND_N_EXPERT - split};
    for (uint32_t d = 0; d < 2; d++) {
        const uint64_t dev_offset = offset + (uint64_t)begin[d] * expert_bytes;
        const uint64_t dev_bytes = (uint64_t)count[d] * expert_bytes;
        w->expert_begin[d] = begin[d];
        w->expert_count[d] = count[d];
        if (!ascend_cache_tensor_to_device(w, model_map, model_size, (int)d, dev_offset, dev_bytes)) return 0;
    }
    return 1;
}

int ds4_gpu_init(void) {
    if (g_initialized) return 1;

    const char *requested_visible = getenv("DS4_ASCEND_VISIBLE_DEVICES");
    if (requested_visible && requested_visible[0]) {
        setenv("ASCEND_RT_VISIBLE_DEVICES", requested_visible, 1);
    } else {
        const char *visible = getenv("ASCEND_RT_VISIBLE_DEVICES");
        if (!visible || !visible[0] || strcmp(visible, "0") == 0) {
            setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1", 1);
        }
    }

    if (!ascend_ok(aclInit(NULL), "init")) return 0;

    uint32_t count = 0;
    if (!ascend_ok(aclrtGetDeviceCount(&count), "get device count")) {
        (void)aclFinalize();
        return 0;
    }
    if (count == 0) {
        fprintf(stderr, "ds4: Ascend backend found no visible devices\n");
        (void)aclFinalize();
        return 0;
    }

    g_device_count = count > DS4_ASCEND_MAX_DEVICES ? DS4_ASCEND_MAX_DEVICES : count;
    g_default_device = ascend_parse_device_env();
    if (g_default_device >= (int)g_device_count) g_default_device = 0;
    g_expert_shard_point = ascend_parse_shard_point();

    for (uint32_t i = 0; i < g_device_count; i++) {
        if (!ascend_ok(aclrtSetDevice((int32_t)i), "set device")) {
            ds4_gpu_cleanup();
            return 0;
        }
        if (!ascend_ok(aclrtCreateContext(&g_contexts[i], (int32_t)i), "create context")) {
            ds4_gpu_cleanup();
            return 0;
        }
        g_context_ready[i] = 1;
    }
    if (!ascend_set_context(g_default_device)) {
        ds4_gpu_cleanup();
        return 0;
    }
    if (!ascend_create_streams()) {
        ds4_gpu_cleanup();
        return 0;
    }
    if (!ascend_enable_peer_access()) {
        ds4_gpu_cleanup();
        return 0;
    }

    fprintf(stderr,
            "ds4: Ascend backend initialized with %u visible device%s (default=%d, expert split=%u/%u, ASCEND_RT_VISIBLE_DEVICES=%s)\n",
            g_device_count,
            g_device_count == 1 ? "" : "s",
            g_default_device,
            g_expert_shard_point,
            DS4_ASCEND_N_EXPERT - g_expert_shard_point,
            getenv("ASCEND_RT_VISIBLE_DEVICES") ? getenv("ASCEND_RT_VISIBLE_DEVICES") : "unset");
    if (g_device_count < 2) {
        fprintf(stderr, "ds4: warning: DeepSeek V4 Flash q2 on Atlas 300I Duo needs both 310P3 chips visible\n");
    }

    g_initialized = 1;
    return 1;
}

void ds4_gpu_cleanup(void) {
    if (!g_initialized && g_device_count == 0) return;
    ascend_report_fallback_stats();
    ascend_deferred_frees_reset();
    ascend_weight_cache_reset();
    ascend_iq2_tables_reset();
    ascend_destroy_streams();
    for (uint32_t i = 0; i < g_device_count; i++) {
        if (g_context_ready[i]) {
            (void)aclrtSetCurrentContext(g_contexts[i]);
            (void)aclrtSynchronizeDevice();
            (void)aclrtDestroyContext(g_contexts[i]);
            g_contexts[i] = NULL;
            g_context_ready[i] = 0;
        }
        (void)aclrtResetDevice((int32_t)i);
    }
    (void)aclFinalize();
    g_initialized = 0;
    g_device_count = 0;
    g_model_map = NULL;
    g_model_size = 0;
    g_model_fd = -1;
    memset(g_live_bytes, 0, sizeof(g_live_bytes));
    memset(g_peak_bytes, 0, sizeof(g_peak_bytes));
}

ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    return ascend_tensor_alloc_on_device(bytes, g_default_device);
}

ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed(uint64_t bytes) {
    return ds4_gpu_tensor_alloc(bytes);
}

ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes) {
    if (!base || offset > base->bytes || bytes > base->bytes - offset) return NULL;
    ds4_gpu_tensor *t = calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->ptr = (char *)base->ptr + offset;
    t->bytes = bytes;
    t->owner = 0;
    t->device = base->device;
    return t;
}

void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor) {
    if (!tensor) return;
    if (tensor->owner && tensor->ptr) {
        if (ascend_set_context(tensor->device)) {
            (void)aclrtFree(tensor->ptr);
        }
        if (tensor->device >= 0 && tensor->device < DS4_ASCEND_MAX_DEVICES) {
            if (g_live_bytes[tensor->device] >= tensor->bytes) g_live_bytes[tensor->device] -= tensor->bytes;
            else g_live_bytes[tensor->device] = 0;
        }
    }
    free(tensor->host_shadow);
    free(tensor);
}

uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor) {
    return tensor ? tensor->bytes : 0;
}

void *ds4_gpu_tensor_contents(ds4_gpu_tensor *tensor) {
    if (!tensor) return NULL;
    if (tensor->host_shadow_bytes < tensor->bytes) {
        void *p = realloc(tensor->host_shadow, (size_t)tensor->bytes);
        if (!p) return NULL;
        tensor->host_shadow = p;
        tensor->host_shadow_bytes = tensor->bytes;
    }
    if (ds4_gpu_tensor_read(tensor, 0, tensor->host_shadow, tensor->bytes) == 0) return NULL;
    return tensor->host_shadow;
}

int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value, uint64_t count) {
    if (!tensor || count > tensor->bytes / sizeof(float) || count > UINT32_MAX) return 0;
    if (count == 0) return 1;
    return ascend_launch_fill_f32(tensor, value, (uint32_t)count);
}

int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes) {
    if (!tensor || !data || offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    if (bytes == 0) return 1;
    if (!ascend_set_context(tensor->device)) return 0;
    return ascend_ok(aclrtMemcpy((char *)tensor->ptr + offset, (size_t)(tensor->bytes - offset), data, (size_t)bytes, ACL_MEMCPY_HOST_TO_DEVICE), "tensor write");
}

int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes) {
    if (!tensor || !data || offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    if (bytes == 0) return 1;
    if (!ascend_set_context(tensor->device)) return 0;
    return ascend_ok(aclrtMemcpy(data, (size_t)bytes, (const char *)tensor->ptr + offset, (size_t)bytes, ACL_MEMCPY_DEVICE_TO_HOST), "tensor read");
}

int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset, const ds4_gpu_tensor *src, uint64_t src_offset, uint64_t bytes) {
    if (!dst || !src || dst_offset > dst->bytes || src_offset > src->bytes ||
        bytes > dst->bytes - dst_offset || bytes > src->bytes - src_offset) return 0;
    if (bytes == 0) return 1;
    if (dst->device != src->device) return ascend_unimplemented("cross-device tensor copy");
    if (!ascend_set_context(dst->device)) return 0;
    return ascend_ok(aclrtMemcpy((char *)dst->ptr + dst_offset,
                                (size_t)(dst->bytes - dst_offset),
                                (const char *)src->ptr + src_offset,
                                (size_t)bytes,
                                ACL_MEMCPY_DEVICE_TO_DEVICE),
                     "tensor copy");
}

int ds4_gpu_ascend_quantize_q8_k_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, uint32_t rows, uint32_t cols) {
    if (!out || !x || rows == 0 || cols == 0 || (cols % DS4_ASCEND_QK_K) != 0) return 0;
    const uint64_t blocks = (uint64_t)rows * (cols / DS4_ASCEND_QK_K);
    if (x->bytes < (uint64_t)rows * cols * sizeof(float) || out->bytes < blocks * sizeof(ascend_block_q8_K)) return 0;
    if (out->device != x->device) return ascend_unimplemented("cross-device q8 quantize");
    if (!ascend_set_context(out->device) || !g_streams[out->device]) return 0;
    ds4_ascend_launch_quantize_q8_k(g_streams[out->device], out->ptr, x->ptr, rows, cols);
    return 1;
}

int ds4_gpu_begin_commands(void) { return (!g_initialized && !ds4_gpu_init()) ? 0 : 1; }
int ds4_gpu_flush_commands(void) { return ds4_gpu_synchronize(); }
int ds4_gpu_end_commands(void) { return ds4_gpu_synchronize(); }

int ds4_gpu_synchronize(void) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    for (uint32_t i = 0; i < g_device_count; i++) {
        if (!ascend_set_context((int)i)) return 0;
        if (g_streams[i]) {
            if (!ascend_ok(aclrtSynchronizeStream(g_streams[i]), "synchronize stream")) return 0;
        } else if (!ascend_ok(aclrtSynchronizeDevice(), "synchronize device")) return 0;
    }
    ascend_drain_deferred_frees();
    return ascend_set_context(g_default_device);
}

int ds4_gpu_set_model_map(const void *model_map, uint64_t model_size) {
    if (!model_map || model_size == 0) return 0;
    g_model_map = model_map;
    g_model_size = model_size;
    return 1;
}

int ds4_gpu_set_model_fd(int fd) {
    g_model_fd = fd;
    return 1;
}

int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size) {
    if (map_offset > model_size || map_size > model_size - map_offset) return 0;
    return ds4_gpu_set_model_map(model_map, model_size);
}

int ds4_gpu_cache_model_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, const char *label) {
    if (!model_map || offset > model_size || bytes > model_size - offset) return 0;
    uint32_t layer = 0;
    ascend_weight_kind kind = DS4_ASCEND_WEIGHT_DENSE;
    ascend_weight_cache *w = ascend_weight_cache_add(offset, bytes);
    if (!w) return 0;
    if (ascend_parse_expert_label(label, &layer, &kind)) {
        w->kind = kind;
        w->layer = layer;
    }
    return 1;
}

int ds4_gpu_cache_q8_f16_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, uint64_t in_dim, uint64_t out_dim, const char *label) {
    if (!model_map || offset > model_size || bytes > model_size - offset) return 0;
    uint32_t layer = 0;
    ascend_weight_kind kind = DS4_ASCEND_WEIGHT_DENSE;
    ascend_weight_cache *w = ascend_weight_cache_add(offset, bytes);
    if (!w) return 0;
    w->in_dim = in_dim;
    w->out_dim = out_dim;
    if (!ascend_parse_expert_label(label, &layer, &kind)) return 1;
    w->kind = kind;
    w->layer = layer;

    uint32_t type = 0;
    switch (kind) {
    case DS4_ASCEND_WEIGHT_GATE_EXPERT:
    case DS4_ASCEND_WEIGHT_UP_EXPERT:
        type = DS4_ASCEND_TENSOR_IQ2_XXS;
        break;
    case DS4_ASCEND_WEIGHT_DOWN_EXPERT:
        type = DS4_ASCEND_TENSOR_Q2_K;
        break;
    default:
        return 1;
    }
    return ascend_cache_routed_expert_tensor(w, model_map, model_size, offset, bytes, in_dim, out_dim, type, label);
}

int ds4_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes, uint64_t context_bytes) {
    return 0;
}

void ds4_gpu_set_quality(bool quality) {
    g_quality_mode = quality ? 1 : 0;
}

void ds4_gpu_print_memory_report(const char *label) {
    if (!g_initialized && !ds4_gpu_init()) return;
    for (uint32_t i = 0; i < g_device_count; i++) {
        size_t free_b = 0;
        size_t total_b = 0;
        if (!ascend_set_context((int)i)) continue;
        (void)aclrtGetMemInfo(ACL_MEM_NORMAL, &free_b, &total_b);
        fprintf(stderr,
                "ds4: Ascend memory report %s device %u: free %.2f MiB total %.2f MiB live %.2f MiB peak %.2f MiB weights %.2f MiB experts %.2f MiB\n",
                label ? label : "",
                i,
                (double)free_b / 1048576.0,
                (double)total_b / 1048576.0,
                (double)g_live_bytes[i] / 1048576.0,
                (double)g_peak_bytes[i] / 1048576.0,
                (double)g_weight_cache_bytes[i] / 1048576.0,
                (double)g_expert_cache_bytes[i] / 1048576.0);
    }
    (void)ascend_set_context(g_default_device);
}

static int ascend_matmul_f16_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !model_map || !x || in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;
    const uint64_t weight_bytes = in_dim * out_dim * sizeof(uint16_t);
    const uint64_t x_count = n_tok * in_dim;
    const uint64_t out_count = n_tok * out_dim;
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset ||
        x->bytes < x_count * sizeof(float) || out->bytes < out_count * sizeof(float) ||
        x->device != out->device) return 0;
    void *w_dev = ascend_model_range_device_ptr(model_map, model_size, weight_offset, weight_bytes, x->device, "matmul_f16");
    if (!w_dev) return 0;
    if (!ascend_set_context(x->device) || !g_streams[x->device]) return 0;
    ds4_ascend_launch_matmul_f16(g_streams[x->device], out->ptr, w_dev, x->ptr, (uint32_t)in_dim, (uint32_t)out_dim, (uint32_t)n_tok);
    return 1;
}

int ds4_gpu_embed_token_hc_tensor(ds4_gpu_tensor *out_hc, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n_vocab, uint32_t token, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !model_map || n_vocab == 0 || n_embd == 0 || n_hc == 0) return 0;
    if (token >= n_vocab) return 0;
    const uint64_t weight_bytes = (uint64_t)n_vocab * n_embd * sizeof(uint16_t);
    const uint64_t out_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset || out_hc->bytes < out_bytes) return 0;
    void *w_dev = ascend_model_range_device_ptr(model_map, model_size, weight_offset, weight_bytes, out_hc->device, "token_embd");
    if (!w_dev) return 0;
    if (!ascend_set_context(out_hc->device) || !g_streams[out_hc->device]) return 0;
    ds4_ascend_launch_embed_token_hc(g_streams[out_hc->device], out_hc->ptr, w_dev, token, n_embd, n_hc);
    return 1;
}

int ds4_gpu_embed_tokens_hc_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *tokens, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n_vocab, uint32_t n_tokens, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !tokens || !model_map || n_vocab == 0 || n_tokens == 0 || n_embd == 0 || n_hc == 0) return 0;
    const uint64_t weight_bytes = (uint64_t)n_vocab * n_embd * sizeof(uint16_t);
    const uint64_t out_bytes = (uint64_t)n_tokens * n_hc * n_embd * sizeof(float);
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset ||
        tokens->bytes < (uint64_t)n_tokens * sizeof(int32_t) || out_hc->bytes < out_bytes ||
        tokens->device != out_hc->device) return 0;
    void *w_dev = ascend_model_range_device_ptr(model_map, model_size, weight_offset, weight_bytes, out_hc->device, "token_embd");
    if (!w_dev) return 0;
    if (!ascend_set_context(out_hc->device) || !g_streams[out_hc->device]) return 0;
    ds4_ascend_launch_embed_tokens_hc(g_streams[out_hc->device], out_hc->ptr, tokens->ptr, w_dev, n_vocab, n_tokens, n_embd, n_hc);
    return 1;
}
int ds4_gpu_indexer_score_one_tensor(ds4_gpu_tensor *scores, const ds4_gpu_tensor *q, const ds4_gpu_tensor *weights, const ds4_gpu_tensor *index_comp, uint32_t n_comp, uint32_t n_head, uint32_t head_dim, float scale) { return ascend_unimplemented(__func__); }
int ds4_gpu_indexer_scores_prefill_tensor(ds4_gpu_tensor *scores, const ds4_gpu_tensor *q, const ds4_gpu_tensor *weights, const ds4_gpu_tensor *index_comp, uint32_t n_comp, uint32_t n_tokens, uint32_t n_head, uint32_t head_dim, uint32_t ratio, float scale) { return ascend_unimplemented(__func__); }
int ds4_gpu_indexer_scores_decode_batch_tensor(ds4_gpu_tensor *scores, const ds4_gpu_tensor *q, const ds4_gpu_tensor *weights, const ds4_gpu_tensor *index_comp, uint32_t n_comp, uint32_t n_tokens, uint32_t pos0, uint32_t n_head, uint32_t head_dim, uint32_t ratio, float scale) { return ascend_unimplemented(__func__); }
int ds4_gpu_indexer_topk_tensor(ds4_gpu_tensor *selected, const ds4_gpu_tensor *scores, uint32_t n_comp, uint32_t n_tokens, uint32_t top_k) { return ascend_unimplemented(__func__); }
int ds4_gpu_dsv4_topk_mask_tensor(ds4_gpu_tensor *mask, const ds4_gpu_tensor *topk, uint32_t n_comp, uint32_t n_tokens, uint32_t top_k) { return ascend_unimplemented(__func__); }
static int ascend_matmul_q8_0_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !model_map || !x || in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;
    const uint64_t blocks = (in_dim + DS4_ASCEND_QK8_0 - 1) / DS4_ASCEND_QK8_0;
    const uint64_t row_bytes = blocks * DS4_ASCEND_BLOCK_Q8_0_BYTES;
    const uint64_t weight_bytes = out_dim * row_bytes;
    const uint64_t x_count = n_tok * in_dim;
    const uint64_t out_count = n_tok * out_dim;
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset ||
        x->bytes < x_count * sizeof(float) || out->bytes < out_count * sizeof(float) ||
        x->device != out->device) return 0;
    void *w_dev = ascend_model_range_device_ptr(model_map, model_size, weight_offset, weight_bytes, x->device, "matmul_q8_0");
    if (!w_dev) return 0;
    if (!ascend_set_context(x->device) || !g_streams[x->device]) return 0;
    const uint64_t xq_bytes = n_tok * blocks * DS4_ASCEND_QK8_0;
    const uint64_t xscale_bytes = n_tok * blocks * sizeof(float);
    ds4_gpu_tensor *xq = ascend_tensor_alloc_on_device(xq_bytes, x->device);
    ds4_gpu_tensor *xscale = ascend_tensor_alloc_on_device(xscale_bytes, x->device);
    if (!xq || !xscale) {
        ds4_gpu_tensor_free(xq);
        ds4_gpu_tensor_free(xscale);
        return 0;
    }
    ds4_ascend_launch_quantize_q8_0(g_streams[x->device], xq->ptr, xscale->ptr, x->ptr, (uint32_t)in_dim, (uint32_t)n_tok);
    ds4_ascend_launch_matmul_q8_0_prequant(g_streams[x->device], out->ptr, w_dev, xq->ptr, xscale->ptr, (uint32_t)in_dim, (uint32_t)out_dim, (uint32_t)n_tok);
    int ok = ascend_defer_free_tensor(xq);
    ok = ascend_defer_free_tensor(xscale) && ok;
    return ok;
}

int ds4_gpu_matmul_q8_0_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    return ascend_matmul_q8_0_tensor(out, model_map, model_size, weight_offset, in_dim, out_dim, x, n_tok);
}

static int ascend_swiglu_device(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up, uint64_t n, float clamp, float weight) {
    if (!out || !gate || !up || n > UINT32_MAX ||
        out->bytes < n * sizeof(float) || gate->bytes < n * sizeof(float) || up->bytes < n * sizeof(float) ||
        out->device != gate->device || out->device != up->device) return 0;
    if (!ascend_set_context(out->device) || !g_streams[out->device]) return 0;
    ds4_ascend_launch_swiglu(g_streams[out->device], out->ptr, gate->ptr, up->ptr, (uint32_t)n, clamp, weight);
    return 1;
}

static int ascend_swiglu_host(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up, uint64_t n, float clamp, float weight) {
    if (!out || !gate || !up || out->bytes < n * sizeof(float) || gate->bytes < n * sizeof(float) || up->bytes < n * sizeof(float)) return 0;
    if (!ascend_host_fallback_begin("swiglu", 2u * n * sizeof(float), n * sizeof(float))) return 0;
    float *oh = malloc((size_t)(n * sizeof(float)));
    float *gh = malloc((size_t)(n * sizeof(float)));
    float *uh = malloc((size_t)(n * sizeof(float)));
    if (!oh || !gh || !uh) {
        free(oh); free(gh); free(uh);
        return 0;
    }
    int ok = ds4_gpu_tensor_read(gate, 0, gh, n * sizeof(float)) &&
             ds4_gpu_tensor_read(up, 0, uh, n * sizeof(float));
    if (ok) {
        for (uint64_t i = 0; i < n; i++) {
            float g = gh[i];
            float u = uh[i];
            if (clamp > 1.0e-6f) {
                if (g > clamp) g = clamp;
                if (u > clamp) u = clamp;
                if (u < -clamp) u = -clamp;
            }
            oh[i] = (g / (1.0f + expf(-g))) * u * weight;
        }
        ok = ds4_gpu_tensor_write(out, 0, oh, n * sizeof(float));
    }
    free(oh); free(gh); free(uh);
    return ok;
}

static int ascend_shared_gate_up_swiglu_q8_0_tensor(ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!gate || !up || !mid || !model_map || !x || in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;
    const uint64_t blocks = (in_dim + DS4_ASCEND_QK8_0 - 1) / DS4_ASCEND_QK8_0;
    const uint64_t row_bytes = blocks * DS4_ASCEND_BLOCK_Q8_0_BYTES;
    const uint64_t weight_bytes = out_dim * row_bytes;
    const uint64_t x_count = n_tok * in_dim;
    const uint64_t out_count = n_tok * out_dim;
    if (gate_offset > model_size || up_offset > model_size ||
        weight_bytes > model_size - gate_offset || weight_bytes > model_size - up_offset ||
        x->bytes < x_count * sizeof(float) || gate->bytes < out_count * sizeof(float) ||
        up->bytes < out_count * sizeof(float) || mid->bytes < out_count * sizeof(float) ||
        x->device != gate->device || x->device != up->device || x->device != mid->device) return 0;
    void *gate_dev = ascend_model_range_device_ptr(model_map, model_size, gate_offset, weight_bytes, x->device, "shared_gate_q8_0");
    void *up_dev = ascend_model_range_device_ptr(model_map, model_size, up_offset, weight_bytes, x->device, "shared_up_q8_0");
    if (!gate_dev || !up_dev) return 0;
    if (!ascend_set_context(x->device) || !g_streams[x->device]) return 0;
    const uint64_t xq_bytes = n_tok * blocks * DS4_ASCEND_QK8_0;
    const uint64_t xscale_bytes = n_tok * blocks * sizeof(float);
    ds4_gpu_tensor *xq = ascend_tensor_alloc_on_device(xq_bytes, x->device);
    ds4_gpu_tensor *xscale = ascend_tensor_alloc_on_device(xscale_bytes, x->device);
    if (!xq || !xscale) {
        ds4_gpu_tensor_free(xq);
        ds4_gpu_tensor_free(xscale);
        return 0;
    }
    ds4_ascend_launch_quantize_q8_0(g_streams[x->device], xq->ptr, xscale->ptr, x->ptr, (uint32_t)in_dim, (uint32_t)n_tok);
    ds4_ascend_launch_matmul_q8_0_prequant(g_streams[x->device], gate->ptr, gate_dev, xq->ptr, xscale->ptr, (uint32_t)in_dim, (uint32_t)out_dim, (uint32_t)n_tok);
    ds4_ascend_launch_matmul_q8_0_prequant(g_streams[x->device], up->ptr, up_dev, xq->ptr, xscale->ptr, (uint32_t)in_dim, (uint32_t)out_dim, (uint32_t)n_tok);
    int ok = ascend_swiglu_device(mid, gate, up, out_count, 10.0f, 1.0f);
    int defer_ok = ascend_defer_free_tensor(xq);
    defer_ok = ascend_defer_free_tensor(xscale) && defer_ok;
    return ok && defer_ok;
}

int ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x) {
    if (ascend_shared_gate_up_swiglu_q8_0_tensor(gate, up, mid, model_map, model_size, gate_offset, up_offset, in_dim, out_dim, x, 1)) return 1;
    return ascend_matmul_q8_0_tensor(gate, model_map, model_size, gate_offset, in_dim, out_dim, x, 1) &&
           ascend_matmul_q8_0_tensor(up, model_map, model_size, up_offset, in_dim, out_dim, x, 1) &&
           (ascend_swiglu_device(mid, gate, up, out_dim, 10.0f, 1.0f) || ascend_swiglu_host(mid, gate, up, out_dim, 10.0f, 1.0f));
}
int ds4_gpu_matmul_f16_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    return ascend_matmul_f16_tensor(out, model_map, model_size, weight_offset, in_dim, out_dim, x, n_tok);
}

int ds4_gpu_matmul_f16_pair_tensor(ds4_gpu_tensor *out_a, ds4_gpu_tensor *out_b, const void *model_map, uint64_t model_size, uint64_t weight_a_offset, uint64_t weight_b_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    return ascend_matmul_f16_tensor(out_a, model_map, model_size, weight_a_offset, in_dim, out_dim, x, n_tok) &&
           ascend_matmul_f16_tensor(out_b, model_map, model_size, weight_b_offset, in_dim, out_dim, x, n_tok);
}
int ds4_gpu_matmul_f32_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) { return ascend_unimplemented(__func__); }
int ds4_gpu_repeat_hc_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *row, uint32_t n_embd, uint32_t n_hc) { return ascend_unimplemented(__func__); }
static int ascend_rms_norm_plain_rows_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, uint32_t n, uint32_t rows, float eps) {
    if (!out || !x || n == 0 || rows == 0) return 0;
    if ((uint64_t)n * rows > UINT32_MAX) return 0;
    const uint64_t count = (uint64_t)n * rows;
    if (x->bytes < count * sizeof(float) || out->bytes < count * sizeof(float) || out->device != x->device) return 0;
    if (!ascend_set_context(x->device) || !g_streams[x->device]) return 0;
    ds4_ascend_launch_rms_norm_plain(g_streams[x->device], out->ptr, x->ptr, n, rows, eps);
    return 1;
}

int ds4_gpu_rms_norm_plain_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, uint32_t n, float eps) {
    return ascend_rms_norm_plain_rows_tensor(out, x, n, 1, eps);
}

int ds4_gpu_rms_norm_plain_rows_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, uint32_t n, uint32_t rows, float eps) {
    return ascend_rms_norm_plain_rows_tensor(out, x, n, rows, eps);
}

static int ascend_rms_norm_weight_rows_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n, uint32_t rows, float eps) {
    if (!out || !x || !model_map || n == 0 || rows == 0) return 0;
    if ((uint64_t)n * rows > UINT32_MAX) return 0;
    const uint64_t count = (uint64_t)n * rows;
    const uint64_t weight_bytes = (uint64_t)n * sizeof(float);
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset ||
        x->bytes < count * sizeof(float) || out->bytes < count * sizeof(float) || out->device != x->device) return 0;
    void *w_dev = ascend_model_range_device_ptr(model_map, model_size, weight_offset, weight_bytes, x->device, "rms_norm_weight");
    if (!w_dev) return 0;
    if (!ascend_set_context(x->device) || !g_streams[x->device]) return 0;
    ds4_ascend_launch_rms_norm_weight(g_streams[x->device], out->ptr, x->ptr, w_dev, n, rows, eps);
    return 1;
}

int ds4_gpu_rms_norm_weight_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n, float eps) {
    return ascend_rms_norm_weight_rows_tensor(out, x, model_map, model_size, weight_offset, n, 1, eps);
}

int ds4_gpu_rms_norm_weight_rows_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n, uint32_t rows, float eps) {
    return ascend_rms_norm_weight_rows_tensor(out, x, model_map, model_size, weight_offset, n, rows, eps);
}
int ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(ds4_gpu_tensor *q_out, const ds4_gpu_tensor *q, const void *model_map, uint64_t model_size, uint64_t q_weight_offset, uint32_t q_n, ds4_gpu_tensor *kv_out, const ds4_gpu_tensor *kv, uint64_t kv_weight_offset, uint32_t kv_n, uint32_t rows, float eps) {
    return ascend_rms_norm_weight_rows_tensor(q_out, q, model_map, model_size, q_weight_offset, q_n, rows, eps) &&
           ascend_rms_norm_weight_rows_tensor(kv_out, kv, model_map, model_size, kv_weight_offset, kv_n, rows, eps);
}
int ds4_gpu_head_rms_norm_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, float eps) {
    if (!x || n_tok == 0 || n_head == 0 || head_dim == 0) return 0;
    const uint64_t rows = (uint64_t)n_tok * n_head;
    const uint64_t count = rows * head_dim;
    if (rows > UINT32_MAX || count > UINT32_MAX || x->bytes < count * sizeof(float)) return 0;
    if (!ascend_set_context(x->device) || !g_streams[x->device]) return 0;
    ds4_ascend_launch_rms_norm_inplace(g_streams[x->device], x->ptr, head_dim, (uint32_t)rows, eps);
    return 1;
}
int ds4_gpu_dsv4_fp8_kv_quantize_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot) {
    if (!x || n_tok == 0 || head_dim == 0 || n_rot > head_dim) return 0;
    const uint32_t n_nope = head_dim - n_rot;
    if (n_nope == 0) return 1;
    const uint64_t count = (uint64_t)n_tok * head_dim;
    if (count > UINT32_MAX || x->bytes < count * sizeof(float)) return 0;
    if (!ascend_set_context(x->device) || !g_streams[x->device]) return 0;
    ds4_ascend_launch_fp8_kv_quantize(g_streams[x->device], x->ptr, n_tok, head_dim, n_rot);
    return 1;
}
int ds4_gpu_rope_tail_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    if (!x || n_tok == 0 || n_head == 0 || head_dim == 0 || n_rot > head_dim || (n_rot & 1u)) return 0;
    if (n_rot == 0) return 1;
    const uint64_t count = (uint64_t)n_tok * n_head * head_dim;
    const uint64_t table_count = (uint64_t)n_tok * n_rot;
    if (x->bytes < count * sizeof(float) || count > UINT32_MAX || table_count > UINT32_MAX) return 0;
    float *table = malloc((size_t)(table_count * sizeof(float)));
    if (!table) return 0;

    float corr0 = 0.0f;
    float corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        const float pi2 = 6.2831853071795864769f;
        corr0 = floorf((float)n_rot * logf((float)n_ctx_orig / (beta_fast * pi2)) / (2.0f * logf(freq_base)));
        corr1 = ceilf((float)n_rot * logf((float)n_ctx_orig / (beta_slow * pi2)) / (2.0f * logf(freq_base)));
        if (corr0 < 0.0f) corr0 = 0.0f;
        if (corr1 > (float)(n_rot - 1)) corr1 = (float)(n_rot - 1);
    }
    const uint32_t half_rot = n_rot / 2u;
    for (uint32_t t = 0; t < n_tok; t++) {
        const float pos = (float)(pos0 + t);
        for (uint32_t pair = 0; pair < half_rot; pair++) {
            const uint32_t i = pair * 2u;
            const float theta_extrap = pos * powf(freq_base, -((float)i) / (float)n_rot);
            const float theta_interp = freq_scale * theta_extrap;
            float theta = theta_interp;
            float mscale = attn_factor;
            if (ext_factor != 0.0f) {
                const float denom = fmaxf(0.001f, corr1 - corr0);
                float y = ((float)pair - corr0) / denom;
                if (y < 0.0f) y = 0.0f;
                if (y > 1.0f) y = 1.0f;
                const float ramp_mix = (1.0f - y) * ext_factor;
                theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
                mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
            }
            const uint64_t base = ((uint64_t)t * half_rot + pair) * 2u;
            table[base] = cosf(theta) * mscale;
            table[base + 1u] = (inverse ? -sinf(theta) : sinf(theta)) * mscale;
        }
    }

    ds4_gpu_tensor *table_dev = ascend_tensor_alloc_on_device(table_count * sizeof(float), x->device);
    int ok = table_dev && ds4_gpu_tensor_write(table_dev, 0, table, table_count * sizeof(float));
    if (ok && (!ascend_set_context(x->device) || !g_streams[x->device])) ok = 0;
    int launched = 0;
    if (ok) {
        ds4_ascend_launch_rope_tail_table(g_streams[x->device], x->ptr, table_dev->ptr, n_tok, n_head, head_dim, n_rot);
        launched = 1;
    }
    if (ok) ok = ascend_defer_free_tensor(table_dev);
    if (!ok) {
        if (launched) (void)aclrtSynchronizeStream(g_streams[x->device]);
        ds4_gpu_tensor_free(table_dev);
    }
    free(table);
    return ok;
}
int ds4_gpu_store_raw_kv_tensor(ds4_gpu_tensor *raw_cache, const ds4_gpu_tensor *kv, uint32_t raw_cap, uint32_t row, uint32_t head_dim) {
    if (!raw_cache || !kv || raw_cap == 0 || head_dim == 0 || row >= raw_cap) return 0;
    const uint64_t row_bytes = (uint64_t)head_dim * sizeof(float);
    if (raw_cache->bytes < (uint64_t)raw_cap * row_bytes || kv->bytes < row_bytes || raw_cache->device != kv->device) return 0;
    if (!ascend_set_context(raw_cache->device) || !g_streams[raw_cache->device]) return 0;
    ds4_ascend_launch_store_raw_kv_batch(g_streams[raw_cache->device], raw_cache->ptr, kv->ptr, raw_cap, row, 1, head_dim);
    return 1;
}

int ds4_gpu_store_raw_kv_batch_tensor(ds4_gpu_tensor *raw_cache, const ds4_gpu_tensor *kv, uint32_t raw_cap, uint32_t pos0, uint32_t n_tokens, uint32_t head_dim) {
    if (!raw_cache || !kv || raw_cap == 0 || n_tokens == 0 || head_dim == 0) return 0;
    const uint64_t row_bytes = (uint64_t)head_dim * sizeof(float);
    const uint64_t count = (uint64_t)n_tokens * head_dim;
    if (count > UINT32_MAX || raw_cache->bytes < (uint64_t)raw_cap * row_bytes || kv->bytes < (uint64_t)n_tokens * row_bytes || raw_cache->device != kv->device) return 0;
    if (!ascend_set_context(raw_cache->device) || !g_streams[raw_cache->device]) return 0;
    ds4_ascend_launch_store_raw_kv_batch(g_streams[raw_cache->device], raw_cache->ptr, kv->ptr, raw_cap, pos0, n_tokens, head_dim);
    return 1;
}

int ds4_gpu_kv_fp8_store_raw_tensor(ds4_gpu_tensor *kv, ds4_gpu_tensor *raw_cache, uint32_t raw_cap, uint32_t row, uint32_t head_dim, uint32_t n_rot) {
    return ds4_gpu_dsv4_fp8_kv_quantize_tensor(kv, 1, head_dim, n_rot) &&
           ds4_gpu_store_raw_kv_tensor(raw_cache, kv, raw_cap, row, head_dim);
}
static float ascend_model_scalar(const void *base, uint64_t offset, uint32_t type, uint64_t idx) {
    const uint8_t *p = (const uint8_t *)base + offset;
    if (type == 1u) return ascend_f16_to_f32(((const uint16_t *)p)[idx]);
    return ((const float *)p)[idx];
}

static int ascend_compressor_store_rows_host(float *state_kv, float *state_score, const float *kv, const float *sc, const void *model_map, uint64_t ape_offset, uint32_t ape_type, uint32_t width, uint32_t ratio, uint32_t pos0, uint32_t src0, uint32_t dst0, uint32_t rows) {
    for (uint32_t r = 0; r < rows; r++) {
        const uint32_t src = src0 + r;
        const uint32_t dst = dst0 + r;
        const uint32_t phase = (pos0 + src) % ratio;
        for (uint32_t j = 0; j < width; j++) {
            state_kv[(uint64_t)dst * width + j] = kv[(uint64_t)src * width + j];
            state_score[(uint64_t)dst * width + j] = sc[(uint64_t)src * width + j] + ascend_model_scalar(model_map, ape_offset, ape_type, (uint64_t)phase * width + j);
        }
    }
    return 1;
}

static int ascend_compressor_pool_host(float *comp, const float *kv, const float *sc, const float *state_kv, const float *state_score, const void *model_map, uint64_t ape_offset, uint32_t ape_type, uint32_t head_dim, uint32_t ratio, uint32_t pos0, uint32_t n_comp, uint32_t replay) {
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    for (uint32_t c = 0; c < n_comp; c++) {
        for (uint32_t d = 0; d < head_dim; d++) {
            float vals[128];
            float scores[128];
            float max_s = -INFINITY;
            uint32_t n_cand = 0;
            if (ratio == 4u) {
                if (replay && c == 0) {
                    for (uint32_t r = 0; r < 4u; r++) {
                        vals[n_cand] = state_kv[(uint64_t)r * width + d];
                        scores[n_cand] = state_score[(uint64_t)r * width + d];
                        if (scores[n_cand] > max_s) max_s = scores[n_cand];
                        n_cand++;
                    }
                } else if (c > 0) {
                    const uint32_t base = (c - 1u) * ratio;
                    for (uint32_t r = 0; r < 4u; r++) {
                        const uint32_t t = base + r;
                        vals[n_cand] = kv[(uint64_t)t * width + d];
                        scores[n_cand] = sc[(uint64_t)t * width + d] + ascend_model_scalar(model_map, ape_offset, ape_type, (uint64_t)((pos0 + t) % ratio) * width + d);
                        if (scores[n_cand] > max_s) max_s = scores[n_cand];
                        n_cand++;
                    }
                }
                const uint32_t base = c * ratio;
                for (uint32_t r = 0; r < 4u; r++) {
                    const uint32_t t = base + r;
                    vals[n_cand] = kv[(uint64_t)t * width + head_dim + d];
                    scores[n_cand] = sc[(uint64_t)t * width + head_dim + d] + ascend_model_scalar(model_map, ape_offset, ape_type, (uint64_t)((pos0 + t) % ratio) * width + head_dim + d);
                    if (scores[n_cand] > max_s) max_s = scores[n_cand];
                    n_cand++;
                }
            } else {
                const uint32_t base = c * ratio;
                for (uint32_t r = 0; r < ratio; r++) {
                    const uint32_t t = base + r;
                    vals[n_cand] = kv[(uint64_t)t * width + d];
                    scores[n_cand] = sc[(uint64_t)t * width + d] + ascend_model_scalar(model_map, ape_offset, ape_type, (uint64_t)((pos0 + t) % ratio) * width + d);
                    if (scores[n_cand] > max_s) max_s = scores[n_cand];
                    n_cand++;
                }
            }
            float den = 0.0f;
            float acc = 0.0f;
            for (uint32_t i = 0; i < n_cand; i++) {
                const float w = expf(scores[i] - max_s);
                den += w;
                acc += vals[i] * w;
            }
            comp[(uint64_t)c * head_dim + d] = den != 0.0f ? acc / den : 0.0f;
        }
    }
    return 1;
}

int ds4_gpu_compressor_update_tensor(const ds4_gpu_tensor *kv_cur, const ds4_gpu_tensor *sc_cur, ds4_gpu_tensor *state_kv, ds4_gpu_tensor *state_score, ds4_gpu_tensor *comp_cache, const void *model_map, uint64_t model_size, uint64_t ape_offset, uint32_t ape_type, uint64_t norm_offset, uint32_t norm_type, uint32_t head_dim, uint32_t ratio, uint32_t pos, uint32_t comp_row, uint32_t n_rot, uint32_t n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow, float rms_eps) {
    if (!kv_cur || !sc_cur || !state_kv || !state_score || !comp_cache || !model_map || head_dim == 0 || ratio == 0 || n_rot > head_dim || (n_rot & 1u) != 0 || (ape_type != 0u && ape_type != 1u) || norm_type != 0u) return 0;
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t emit = ((pos + 1u) % ratio) == 0u ? 1u : 0u;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        norm_offset > model_size || norm_bytes > model_size - norm_offset ||
        kv_cur->bytes < kv_bytes || sc_cur->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        kv_cur->device != sc_cur->device || kv_cur->device != state_kv->device || kv_cur->device != state_score->device || kv_cur->device != comp_cache->device ||
        (emit && comp_cache->bytes < (uint64_t)(comp_row + 1u) * head_dim * sizeof(float))) return 0;
    void *ape_dev = ascend_model_range_device_ptr(model_map, model_size, ape_offset, ape_bytes, kv_cur->device, "compressor_ape");
    if (!ape_dev || !ascend_set_context(kv_cur->device) || !g_streams[kv_cur->device]) return 0;
    const uint32_t dst = ratio == 4u ? ratio + (pos % ratio) : (pos % ratio);
    ds4_ascend_launch_compressor_set_rows(g_streams[kv_cur->device], state_kv->ptr, state_score->ptr, kv_cur->ptr, sc_cur->ptr, ape_dev, ape_type, width, ratio, pos, 0, dst, 1);
    if (!emit) return 1;
    ds4_gpu_tensor *comp_row_view = ds4_gpu_tensor_view(comp_cache, (uint64_t)comp_row * head_dim * sizeof(float), (uint64_t)head_dim * sizeof(float));
    if (!comp_row_view) return 0;
    ds4_ascend_launch_compressor_update_pool(g_streams[kv_cur->device], comp_row_view->ptr, state_kv->ptr, state_score->ptr, head_dim, ratio);
    int ok = ds4_gpu_rms_norm_weight_rows_tensor(comp_row_view, comp_row_view, model_map, model_size, norm_offset, head_dim, 1, rms_eps);
    if (ok && n_rot != 0) ok = ds4_gpu_rope_tail_tensor(comp_row_view, 1, 1, head_dim, n_rot, pos + 1u - ratio, n_ctx_orig, false, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    ds4_gpu_tensor_free(comp_row_view);
    if (ok && ratio == 4u) {
        if (!ascend_set_context(kv_cur->device) || !g_streams[kv_cur->device]) return 0;
        ds4_ascend_launch_compressor_shift_ratio4(g_streams[kv_cur->device], state_kv->ptr, state_score->ptr, width);
    }
    return ok;
}

int ds4_gpu_compressor_store_batch_tensor(const ds4_gpu_tensor *kv, const ds4_gpu_tensor *sc, ds4_gpu_tensor *state_kv, ds4_gpu_tensor *state_score, const void *model_map, uint64_t model_size, uint64_t ape_offset, uint32_t ape_type, uint32_t head_dim, uint32_t ratio, uint32_t pos0, uint32_t n_tokens) {
    if (!kv || !sc || !state_kv || !state_score || !model_map || head_dim == 0 || ratio == 0 || n_tokens == 0 || (ape_type != 0u && ape_type != 1u)) return 0;
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_count = (uint64_t)n_tokens * width;
    const uint64_t state_count = (uint64_t)state_rows * width;
    if (ape_offset > model_size || (uint64_t)width * ratio * elem_ape > model_size - ape_offset || kv->bytes < kv_count * sizeof(float) || sc->bytes < kv_count * sizeof(float) || state_kv->bytes < state_count * sizeof(float) || state_score->bytes < state_count * sizeof(float)) return 0;
    if (!ascend_host_fallback_begin("compressor_store_batch", 2u * kv_count * sizeof(float), 2u * state_count * sizeof(float))) return 0;
    float *kvh = malloc((size_t)(kv_count * sizeof(float)));
    float *sch = malloc((size_t)(kv_count * sizeof(float)));
    float *skv = calloc((size_t)state_count, sizeof(float));
    float *ssc = malloc((size_t)(state_count * sizeof(float)));
    if (!kvh || !sch || !skv || !ssc) { free(kvh); free(sch); free(skv); free(ssc); return 0; }
    for (uint64_t i = 0; i < state_count; i++) ssc[i] = -INFINITY;
    int ok = ds4_gpu_tensor_read(kv, 0, kvh, kv_count * sizeof(float)) && ds4_gpu_tensor_read(sc, 0, sch, kv_count * sizeof(float));
    if (ok) {
        for (uint32_t t = 0; t < n_tokens; t++) {
            const uint32_t pos_mod = (pos0 + t) % ratio;
            const uint32_t dst = ratio == 4u ? ratio + pos_mod : pos_mod;
            ascend_compressor_store_rows_host(skv, ssc, kvh, sch, model_map, ape_offset, ape_type, width, ratio, pos0, t, dst, 1);
        }
        ok = ds4_gpu_tensor_write(state_kv, 0, skv, state_count * sizeof(float)) && ds4_gpu_tensor_write(state_score, 0, ssc, state_count * sizeof(float));
    }
    free(kvh); free(sch); free(skv); free(ssc);
    return ok;
}

static int ascend_compressor_prefill_device(ds4_gpu_tensor *comp_cache, ds4_gpu_tensor *state_kv, ds4_gpu_tensor *state_score, const ds4_gpu_tensor *kv, const ds4_gpu_tensor *sc, const void *model_map, uint64_t model_size, uint64_t ape_offset, uint32_t ape_type, uint64_t norm_offset, uint32_t norm_type, uint32_t head_dim, uint32_t ratio, uint32_t pos0, uint32_t n_tokens, uint32_t n_rot, uint32_t n_ctx_orig, bool quantize_fp8, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow, float rms_eps, uint32_t replay) {
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map || head_dim == 0 || ratio == 0 || n_tokens == 0 || n_rot > head_dim || (n_rot & 1u) != 0 || (ape_type != 0u && ape_type != 1u) || norm_type != 0u) return 0;
    if (replay && (ratio != 4u || (n_tokens & 3u) != 0 || (pos0 & 3u) != 0)) return 0;
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t n_comp = n_tokens / ratio;
    const uint32_t cutoff = n_comp * ratio;
    const uint32_t rem = n_tokens - cutoff;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_count = (uint64_t)n_tokens * width;
    const uint64_t state_count = (uint64_t)state_rows * width;
    const uint64_t comp_count = (uint64_t)n_comp * head_dim;
    if (ape_offset > model_size || (uint64_t)width * ratio * elem_ape > model_size - ape_offset ||
        norm_offset > model_size || (uint64_t)head_dim * sizeof(float) > model_size - norm_offset ||
        kv->bytes < kv_count * sizeof(float) || sc->bytes < kv_count * sizeof(float) ||
        state_kv->bytes < state_count * sizeof(float) || state_score->bytes < state_count * sizeof(float) ||
        (n_comp != 0 && comp_cache->bytes < comp_count * sizeof(float)) ||
        kv->device != sc->device || kv->device != state_kv->device || kv->device != state_score->device || kv->device != comp_cache->device) return 0;
    void *ape_dev = ascend_model_range_device_ptr(model_map, model_size, ape_offset, (uint64_t)width * ratio * elem_ape, kv->device, "compressor_ape");
    if (!ape_dev || !ascend_set_context(kv->device) || !g_streams[kv->device]) return 0;
    const uint64_t state_bytes = state_count * sizeof(float);
    if (!ascend_ok(aclrtMemset(state_kv->ptr, (size_t)state_bytes, 0, (size_t)state_bytes), "compressor state kv zero")) return 0;
    ds4_ascend_launch_fill_f32(g_streams[kv->device], state_score->ptr, -INFINITY, (uint32_t)state_count);
    if (!replay) {
        if (ratio == 4u) {
            if (cutoff >= ratio) ds4_ascend_launch_compressor_set_rows(g_streams[kv->device], state_kv->ptr, state_score->ptr, kv->ptr, sc->ptr, ape_dev, ape_type, width, ratio, pos0, cutoff - ratio, 0, ratio);
            if (rem != 0) ds4_ascend_launch_compressor_set_rows(g_streams[kv->device], state_kv->ptr, state_score->ptr, kv->ptr, sc->ptr, ape_dev, ape_type, width, ratio, pos0, cutoff, ratio, rem);
        } else if (rem != 0) {
            ds4_ascend_launch_compressor_set_rows(g_streams[kv->device], state_kv->ptr, state_score->ptr, kv->ptr, sc->ptr, ape_dev, ape_type, width, ratio, pos0, cutoff, 0, rem);
        }
    }
    if (n_comp != 0) {
        ds4_ascend_launch_compressor_prefill_pool(g_streams[kv->device], comp_cache->ptr, kv->ptr, sc->ptr, state_kv->ptr, state_score->ptr, ape_dev, ape_type, head_dim, ratio, pos0, n_comp, replay);
        if (!ds4_gpu_rms_norm_weight_rows_tensor(comp_cache, comp_cache, model_map, model_size, norm_offset, head_dim, n_comp, rms_eps)) return 0;
        if (n_rot != 0 && !ds4_gpu_rope_tail_tensor(comp_cache, n_comp, 1, head_dim, n_rot, pos0, n_ctx_orig, false, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)) return 0;
        if (quantize_fp8 && !ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_cache, n_comp, head_dim, n_rot)) return 0;
    }
    if (replay) {
        if (!ascend_set_context(kv->device)) return 0;
        if (!ascend_ok(aclrtMemset(state_kv->ptr, (size_t)state_bytes, 0, (size_t)state_bytes), "compressor replay state kv zero")) return 0;
        ds4_ascend_launch_fill_f32(g_streams[kv->device], state_score->ptr, -INFINITY, (uint32_t)state_count);
        ds4_ascend_launch_compressor_set_rows(g_streams[kv->device], state_kv->ptr, state_score->ptr, kv->ptr, sc->ptr, ape_dev, ape_type, width, ratio, pos0, n_tokens - ratio, 0, ratio);
    }
    return 1;
}

static int ascend_compressor_prefill_common(ds4_gpu_tensor *comp_cache, ds4_gpu_tensor *state_kv, ds4_gpu_tensor *state_score, const ds4_gpu_tensor *kv, const ds4_gpu_tensor *sc, const void *model_map, uint64_t model_size, uint64_t ape_offset, uint32_t ape_type, uint64_t norm_offset, uint32_t norm_type, uint32_t head_dim, uint32_t ratio, uint32_t pos0, uint32_t n_tokens, uint32_t n_rot, uint32_t n_ctx_orig, bool quantize_fp8, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow, float rms_eps, uint32_t replay) {
    if (ascend_compressor_prefill_device(comp_cache, state_kv, state_score, kv, sc, model_map, model_size, ape_offset, ape_type, norm_offset, norm_type, head_dim, ratio, pos0, n_tokens, n_rot, n_ctx_orig, quantize_fp8, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, rms_eps, replay)) return 1;
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map || head_dim == 0 || ratio == 0 || n_tokens == 0 || n_rot > head_dim || (n_rot & 1u) != 0 || (ape_type != 0u && ape_type != 1u) || norm_type != 0u) return 0;
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t n_comp = n_tokens / ratio;
    const uint32_t cutoff = n_comp * ratio;
    const uint32_t rem = n_tokens - cutoff;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_count = (uint64_t)n_tokens * width;
    const uint64_t state_count = (uint64_t)state_rows * width;
    const uint64_t comp_count = (uint64_t)n_comp * head_dim;
    if (ape_offset > model_size || (uint64_t)width * ratio * elem_ape > model_size - ape_offset ||
        norm_offset > model_size || (uint64_t)head_dim * sizeof(float) > model_size - norm_offset ||
        kv->bytes < kv_count * sizeof(float) || sc->bytes < kv_count * sizeof(float) ||
        state_kv->bytes < state_count * sizeof(float) || state_score->bytes < state_count * sizeof(float) ||
        (n_comp != 0 && comp_cache->bytes < comp_count * sizeof(float))) return 0;
    const uint64_t prefill_to_host = 2u * kv_count * sizeof(float) + (replay ? 2u * state_count * sizeof(float) : 0u);
    const uint64_t prefill_to_device = comp_count * sizeof(float) + 2u * state_count * sizeof(float);
    if (!ascend_host_fallback_begin(replay ? "compressor_prefill_replay" : "compressor_prefill", prefill_to_host, prefill_to_device)) return 0;

    float *kvh = malloc((size_t)(kv_count * sizeof(float)));
    float *sch = malloc((size_t)(kv_count * sizeof(float)));
    float *skv = calloc((size_t)state_count, sizeof(float));
    float *ssc = malloc((size_t)(state_count * sizeof(float)));
    float *comp = n_comp ? malloc((size_t)(comp_count * sizeof(float))) : NULL;
    if (!kvh || !sch || !skv || !ssc || (n_comp && !comp)) { free(kvh); free(sch); free(skv); free(ssc); free(comp); return 0; }
    for (uint64_t i = 0; i < state_count; i++) ssc[i] = -INFINITY;
    int ok = ds4_gpu_tensor_read(kv, 0, kvh, kv_count * sizeof(float)) && ds4_gpu_tensor_read(sc, 0, sch, kv_count * sizeof(float));
    if (ok && replay) ok = ds4_gpu_tensor_read(state_kv, 0, skv, state_count * sizeof(float)) && ds4_gpu_tensor_read(state_score, 0, ssc, state_count * sizeof(float));
    if (ok && !replay) {
        if (ratio == 4u) {
            if (cutoff >= ratio) ascend_compressor_store_rows_host(skv, ssc, kvh, sch, model_map, ape_offset, ape_type, width, ratio, pos0, cutoff - ratio, 0, ratio);
            if (rem != 0) ascend_compressor_store_rows_host(skv, ssc, kvh, sch, model_map, ape_offset, ape_type, width, ratio, pos0, cutoff, ratio, rem);
        } else if (rem != 0) {
            ascend_compressor_store_rows_host(skv, ssc, kvh, sch, model_map, ape_offset, ape_type, width, ratio, pos0, cutoff, 0, rem);
        }
    }
    if (ok && n_comp != 0) {
        ascend_compressor_pool_host(comp, kvh, sch, skv, ssc, model_map, ape_offset, ape_type, head_dim, ratio, pos0, n_comp, replay);
        ok = ds4_gpu_tensor_write(comp_cache, 0, comp, comp_count * sizeof(float)) &&
             ds4_gpu_rms_norm_weight_rows_tensor(comp_cache, comp_cache, model_map, model_size, norm_offset, head_dim, n_comp, rms_eps);
        if (ok && n_rot != 0) ok = ds4_gpu_rope_tail_tensor(comp_cache, n_comp, 1, head_dim, n_rot, pos0, n_ctx_orig, false, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        if (ok && quantize_fp8) ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_cache, n_comp, head_dim, n_rot);
    }
    if (ok) {
        if (replay && ratio == 4u) {
            memset(skv, 0, (size_t)(state_count * sizeof(float)));
            for (uint64_t i = 0; i < state_count; i++) ssc[i] = -INFINITY;
            ascend_compressor_store_rows_host(skv, ssc, kvh, sch, model_map, ape_offset, ape_type, width, ratio, pos0, n_tokens - ratio, 0, ratio);
        }
        ok = ds4_gpu_tensor_write(state_kv, 0, skv, state_count * sizeof(float)) && ds4_gpu_tensor_write(state_score, 0, ssc, state_count * sizeof(float));
    }
    free(kvh); free(sch); free(skv); free(ssc); free(comp);
    return ok;
}

int ds4_gpu_compressor_prefill_tensor(ds4_gpu_tensor *comp_cache, ds4_gpu_tensor *state_kv, ds4_gpu_tensor *state_score, const ds4_gpu_tensor *kv, const ds4_gpu_tensor *sc, const void *model_map, uint64_t model_size, uint64_t ape_offset, uint32_t ape_type, uint64_t norm_offset, uint32_t norm_type, uint32_t head_dim, uint32_t ratio, uint32_t pos0, uint32_t n_tokens, uint32_t n_rot, uint32_t n_ctx_orig, bool quantize_fp8, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow, float rms_eps) {
    return ascend_compressor_prefill_common(comp_cache, state_kv, state_score, kv, sc, model_map, model_size, ape_offset, ape_type, norm_offset, norm_type, head_dim, ratio, pos0, n_tokens, n_rot, n_ctx_orig, quantize_fp8, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, rms_eps, 0);
}

int ds4_gpu_compressor_prefill_ratio4_replay_tensor(ds4_gpu_tensor *comp_cache, ds4_gpu_tensor *state_kv, ds4_gpu_tensor *state_score, const ds4_gpu_tensor *kv, const ds4_gpu_tensor *sc, const void *model_map, uint64_t model_size, uint64_t ape_offset, uint32_t ape_type, uint64_t norm_offset, uint32_t norm_type, uint32_t head_dim, uint32_t pos0, uint32_t n_tokens, uint32_t n_rot, uint32_t n_ctx_orig, bool quantize_fp8, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow, float rms_eps) {
    return ascend_compressor_prefill_common(comp_cache, state_kv, state_score, kv, sc, model_map, model_size, ape_offset, ape_type, norm_offset, norm_type, head_dim, 4u, pos0, n_tokens, n_rot, n_ctx_orig, quantize_fp8, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, rms_eps, 1);
}

int ds4_gpu_compressor_prefill_state_ratio4_tensor(ds4_gpu_tensor *state_kv, ds4_gpu_tensor *state_score, const ds4_gpu_tensor *kv_tail, const ds4_gpu_tensor *sc_tail, const void *model_map, uint64_t model_size, uint64_t ape_offset, uint32_t ape_type, uint32_t head_dim, uint32_t pos0) {
    if (!state_kv || !state_score || !kv_tail || !sc_tail || !model_map || head_dim == 0 || (ape_type != 0u && ape_type != 1u)) return 0;
    const uint32_t ratio = 4u;
    const uint32_t width = 2u * head_dim;
    const uint32_t state_rows = 8u;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t tail_count = (uint64_t)ratio * width;
    const uint64_t state_count = (uint64_t)state_rows * width;
    const uint64_t state_bytes = state_count * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)ratio * width * elem_ape;
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset || kv_tail->bytes < tail_count * sizeof(float) || sc_tail->bytes < tail_count * sizeof(float) || state_kv->bytes < state_count * sizeof(float) || state_score->bytes < state_count * sizeof(float)) return 0;
    if (state_kv->device == state_score->device && state_kv->device == kv_tail->device && state_kv->device == sc_tail->device) {
        void *ape_dev = ascend_model_range_device_ptr(model_map, model_size, ape_offset, ape_bytes, state_kv->device, "compressor_ape");
        if (ape_dev && ascend_set_context(state_kv->device) && g_streams[state_kv->device] &&
            ascend_ok(aclrtMemset(state_kv->ptr, (size_t)state_bytes, 0, (size_t)state_bytes), "compressor state kv zero")) {
            ds4_ascend_launch_fill_f32(g_streams[state_kv->device], state_score->ptr, -INFINITY, (uint32_t)state_count);
            ds4_ascend_launch_compressor_set_rows(g_streams[state_kv->device], state_kv->ptr, state_score->ptr, kv_tail->ptr, sc_tail->ptr, ape_dev, ape_type, width, ratio, pos0, 0, 0, ratio);
            return 1;
        }
    }
    if (!ascend_host_fallback_begin("compressor_prefill_state_ratio4", 2u * tail_count * sizeof(float), 2u * state_count * sizeof(float))) return 0;
    float *kvh = malloc((size_t)(tail_count * sizeof(float)));
    float *sch = malloc((size_t)(tail_count * sizeof(float)));
    float *skv = calloc((size_t)state_count, sizeof(float));
    float *ssc = malloc((size_t)(state_count * sizeof(float)));
    if (!kvh || !sch || !skv || !ssc) { free(kvh); free(sch); free(skv); free(ssc); return 0; }
    for (uint64_t i = 0; i < state_count; i++) ssc[i] = -INFINITY;
    int ok = ds4_gpu_tensor_read(kv_tail, 0, kvh, tail_count * sizeof(float)) && ds4_gpu_tensor_read(sc_tail, 0, sch, tail_count * sizeof(float));
    if (ok) {
        ascend_compressor_store_rows_host(skv, ssc, kvh, sch, model_map, ape_offset, ape_type, width, ratio, pos0, 0, 0, ratio);
        ok = ds4_gpu_tensor_write(state_kv, 0, skv, state_count * sizeof(float)) && ds4_gpu_tensor_write(state_score, 0, ssc, state_count * sizeof(float));
    }
    free(kvh); free(sch); free(skv); free(ssc);
    return ok;
}
int ds4_gpu_attention_decode_heads_tensor(ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size, uint64_t sinks_offset, const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw_kv, uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start, const ds4_gpu_tensor *comp_kv, uint32_t n_comp, const ds4_gpu_tensor *comp_mask, uint32_t use_mask, uint32_t n_head, uint32_t head_dim) {
    if (!heads || !model_map || !q || !raw_kv || n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap || n_head == 0 || head_dim == 0 || (n_comp != 0 && !comp_kv) || (use_mask && !comp_mask)) return 0;
    const uint64_t q_count = (uint64_t)n_head * head_dim;
    const uint64_t raw_count = (uint64_t)raw_cap * head_dim;
    const uint64_t comp_count = (uint64_t)n_comp * head_dim;
    const uint64_t sink_bytes = (uint64_t)n_head * sizeof(float);
    if (sinks_offset > model_size || sink_bytes > model_size - sinks_offset ||
        heads->bytes < q_count * sizeof(float) || q->bytes < q_count * sizeof(float) || raw_kv->bytes < raw_count * sizeof(float) ||
        (n_comp != 0 && comp_kv->bytes < comp_count * sizeof(float)) ||
        (use_mask && comp_mask->bytes < (uint64_t)n_comp * sizeof(float)) ||
        heads->device != q->device || heads->device != raw_kv->device ||
        (n_comp != 0 && heads->device != comp_kv->device) ||
        (use_mask && heads->device != comp_mask->device) ||
        n_raw + n_comp > 1024u) return 0;
    void *sinks_dev = ascend_model_range_device_ptr(model_map, model_size, sinks_offset, sink_bytes, heads->device, "attn_sinks");
    if (!sinks_dev) return 0;
    if (!ascend_set_context(heads->device) || !g_streams[heads->device]) return 0;
    ds4_ascend_launch_attention_decode(g_streams[heads->device], heads->ptr, sinks_dev, q->ptr, raw_kv->ptr, n_comp ? comp_kv->ptr : raw_kv->ptr, use_mask ? comp_mask->ptr : NULL, use_mask, n_raw, raw_cap, raw_start, n_comp, n_head, head_dim, 1.0f / sqrtf((float)head_dim));
    return 1;
}
int ds4_gpu_attention_prefill_raw_heads_tensor(ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size, uint64_t sinks_offset, const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw_kv, uint32_t n_tokens, uint32_t window, uint32_t n_head, uint32_t head_dim) {
    if (!heads || !model_map || !q || !raw_kv || n_tokens == 0 || n_head == 0 || head_dim == 0 || window > 512u || (window == 0 && n_tokens > 512u)) return 0;
    const uint64_t q_count = (uint64_t)n_tokens * n_head * head_dim;
    const uint64_t kv_count = (uint64_t)n_tokens * head_dim;
    const uint64_t sink_bytes = (uint64_t)n_head * sizeof(float);
    if (q_count > UINT32_MAX || kv_count > UINT32_MAX || sinks_offset > model_size || sink_bytes > model_size - sinks_offset ||
        q->bytes < q_count * sizeof(float) || heads->bytes < q_count * sizeof(float) || raw_kv->bytes < kv_count * sizeof(float) ||
        heads->device != q->device || heads->device != raw_kv->device) return 0;
    void *sinks_dev = ascend_model_range_device_ptr(model_map, model_size, sinks_offset, sink_bytes, heads->device, "attn_sinks");
    if (!sinks_dev) return 0;
    if (!ascend_set_context(heads->device) || !g_streams[heads->device]) return 0;
    ds4_ascend_launch_attention_prefill_raw(g_streams[heads->device], heads->ptr, sinks_dev, q->ptr, raw_kv->ptr, n_tokens, window, n_head, head_dim, 1.0f / sqrtf((float)head_dim));
    return 1;
}
int ds4_gpu_attention_decode_raw_batch_heads_tensor(ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size, uint64_t sinks_offset, const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw_kv, uint32_t n_tokens, uint32_t pos0, uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start, uint32_t window, uint32_t n_head, uint32_t head_dim) { return ascend_unimplemented(__func__); }
int ds4_gpu_attention_decode_mixed_batch_heads_tensor(ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size, uint64_t sinks_offset, const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw_kv, const ds4_gpu_tensor *comp_kv, const ds4_gpu_tensor *comp_mask, uint32_t use_comp_mask, uint32_t n_tokens, uint32_t pos0, uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start, uint32_t n_comp, uint32_t window, uint32_t ratio, uint32_t n_head, uint32_t head_dim) { return ascend_unimplemented(__func__); }
int ds4_gpu_attention_indexed_mixed_batch_heads_tensor(ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size, uint64_t sinks_offset, const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw_kv, const ds4_gpu_tensor *comp_kv, const ds4_gpu_tensor *topk, uint32_t n_tokens, uint32_t pos0, uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start, uint32_t n_comp, uint32_t top_k, uint32_t window, uint32_t ratio, uint32_t n_head, uint32_t head_dim) { return ascend_unimplemented(__func__); }

static int ascend_attention_prefill_mixed_device(ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size, uint64_t sinks_offset, const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw_kv, const ds4_gpu_tensor *comp_kv, const ds4_gpu_tensor *comp_mask, uint32_t use_comp_mask, uint32_t n_tokens, uint32_t n_comp, uint32_t window, uint32_t ratio, uint32_t n_head, uint32_t head_dim) {
    if (!heads || !model_map || !q || !raw_kv || n_tokens == 0 || ratio == 0 || n_head == 0 || head_dim == 0 || (n_comp != 0 && !comp_kv) || (use_comp_mask && !comp_mask)) return 0;
    const uint64_t q_count = (uint64_t)n_tokens * n_head * head_dim;
    const uint64_t raw_count_all = (uint64_t)n_tokens * head_dim;
    const uint64_t comp_count_all = (uint64_t)n_comp * head_dim;
    if (sinks_offset > model_size || (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        q->bytes < q_count * sizeof(float) || heads->bytes < q_count * sizeof(float) || raw_kv->bytes < raw_count_all * sizeof(float) ||
        (n_comp != 0 && comp_kv->bytes < comp_count_all * sizeof(float)) ||
        (use_comp_mask && comp_mask->bytes < (uint64_t)n_tokens * n_comp * sizeof(float)) ||
        heads->device != q->device || heads->device != raw_kv->device ||
        (n_comp != 0 && heads->device != comp_kv->device) ||
        (use_comp_mask && heads->device != comp_mask->device) ||
        n_tokens + n_comp > 1024u) return 0;
    void *sinks_dev = ascend_model_range_device_ptr(model_map, model_size, sinks_offset, (uint64_t)n_head * sizeof(float), heads->device, "attn_sinks");
    if (!sinks_dev || !ascend_set_context(heads->device) || !g_streams[heads->device]) return 0;
    ds4_ascend_launch_attention_prefill_mixed(g_streams[heads->device], heads->ptr, sinks_dev, q->ptr, raw_kv->ptr, n_comp ? comp_kv->ptr : raw_kv->ptr, use_comp_mask ? comp_mask->ptr : NULL, use_comp_mask, n_tokens, n_comp, window, ratio, n_head, head_dim, 1.0f / sqrtf((float)head_dim));
    return 1;
}

static int ascend_attention_prefill_mixed_host(ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size, uint64_t sinks_offset, const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw_kv, const ds4_gpu_tensor *comp_kv, const ds4_gpu_tensor *comp_mask, uint32_t use_comp_mask, uint32_t n_tokens, uint32_t n_comp, uint32_t window, uint32_t ratio, uint32_t n_head, uint32_t head_dim) {
    if (ascend_attention_prefill_mixed_device(heads, model_map, model_size, sinks_offset, q, raw_kv, comp_kv, comp_mask, use_comp_mask, n_tokens, n_comp, window, ratio, n_head, head_dim)) return 1;
    if (!heads || !model_map || !q || !raw_kv || n_tokens == 0 || ratio == 0 || n_head == 0 || head_dim == 0 || (n_comp != 0 && !comp_kv) || (use_comp_mask && !comp_mask)) return 0;
    const uint64_t q_count = (uint64_t)n_tokens * n_head * head_dim;
    const uint64_t raw_count_all = (uint64_t)n_tokens * head_dim;
    const uint64_t comp_count_all = (uint64_t)n_comp * head_dim;
    if (sinks_offset > model_size || (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        q->bytes < q_count * sizeof(float) || heads->bytes < q_count * sizeof(float) || raw_kv->bytes < raw_count_all * sizeof(float) ||
        (n_comp != 0 && comp_kv->bytes < comp_count_all * sizeof(float)) ||
        (use_comp_mask && comp_mask->bytes < (uint64_t)n_tokens * n_comp * sizeof(float))) return 0;
    const uint64_t mask_count = use_comp_mask ? (uint64_t)n_tokens * n_comp : 0u;
    if (!ascend_host_fallback_begin(use_comp_mask ? "attention_prefill_masked_mixed" : "attention_prefill_static_mixed",
                                    (q_count + raw_count_all + comp_count_all + mask_count) * sizeof(float),
                                    q_count * sizeof(float))) return 0;
    float *qh = malloc((size_t)(q_count * sizeof(float)));
    float *raw = malloc((size_t)(raw_count_all * sizeof(float)));
    float *comp = n_comp ? malloc((size_t)(comp_count_all * sizeof(float))) : NULL;
    float *mask = use_comp_mask ? malloc((size_t)((uint64_t)n_tokens * n_comp * sizeof(float))) : NULL;
    float *out = malloc((size_t)(q_count * sizeof(float)));
    float *scores = malloc((size_t)(n_tokens + n_comp) * sizeof(float));
    if (!qh || !raw || (n_comp && !comp) || (use_comp_mask && !mask) || !out || !scores) {
        free(qh); free(raw); free(comp); free(mask); free(out); free(scores);
        return 0;
    }
    int ok = ds4_gpu_tensor_read(q, 0, qh, q_count * sizeof(float)) && ds4_gpu_tensor_read(raw_kv, 0, raw, raw_count_all * sizeof(float));
    if (ok && n_comp) ok = ds4_gpu_tensor_read(comp_kv, 0, comp, comp_count_all * sizeof(float));
    if (ok && use_comp_mask) ok = ds4_gpu_tensor_read(comp_mask, 0, mask, (uint64_t)n_tokens * n_comp * sizeof(float));
    const float *sinks = (const float *)((const uint8_t *)model_map + sinks_offset);
    if (ok) {
        const float scale = 1.0f / sqrtf((float)head_dim);
        for (uint32_t t = 0; t < n_tokens; t++) {
            const uint32_t raw_start = (window != 0 && t + 1u > window) ? t + 1u - window : 0u;
            const uint32_t raw_count = t + 1u - raw_start;
            uint32_t visible_comp = (t + 1u) / ratio;
            if (visible_comp > n_comp) visible_comp = n_comp;
            for (uint32_t h = 0; h < n_head; h++) {
                const float *qrow = qh + ((uint64_t)t * n_head + h) * head_dim;
                float max_s = sinks[h];
                for (uint32_t r = 0; r < raw_count; r++) {
                    const float *kvrow = raw + (uint64_t)(raw_start + r) * head_dim;
                    float dot = 0.0f;
                    for (uint32_t d = 0; d < head_dim; d++) dot += qrow[d] * kvrow[d];
                    const float s = dot * scale;
                    scores[r] = s;
                    if (s > max_s) max_s = s;
                }
                for (uint32_t c = 0; c < visible_comp; c++) {
                    float s = -INFINITY;
                    const float add = use_comp_mask ? mask[(uint64_t)t * n_comp + c] : 0.0f;
                    if (add > -1.0e20f) {
                        const float *kvrow = comp + (uint64_t)c * head_dim;
                        float dot = 0.0f;
                        for (uint32_t d = 0; d < head_dim; d++) dot += qrow[d] * kvrow[d];
                        s = dot * scale + add;
                    }
                    scores[raw_count + c] = s;
                    if (s > max_s) max_s = s;
                }
                float *oh = out + ((uint64_t)t * n_head + h) * head_dim;
                for (uint32_t d = 0; d < head_dim; d++) oh[d] = 0.0f;
                float denom = expf(sinks[h] - max_s);
                for (uint32_t r = 0; r < raw_count; r++) {
                    const float w = expf(scores[r] - max_s);
                    const float *kvrow = raw + (uint64_t)(raw_start + r) * head_dim;
                    denom += w;
                    for (uint32_t d = 0; d < head_dim; d++) oh[d] += kvrow[d] * w;
                }
                for (uint32_t c = 0; c < visible_comp; c++) {
                    if (!isfinite(scores[raw_count + c])) continue;
                    const float w = expf(scores[raw_count + c] - max_s);
                    const float *kvrow = comp + (uint64_t)c * head_dim;
                    denom += w;
                    for (uint32_t d = 0; d < head_dim; d++) oh[d] += kvrow[d] * w;
                }
                const float inv = denom != 0.0f ? 1.0f / denom : 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) oh[d] *= inv;
            }
        }
        ok = ds4_gpu_tensor_write(heads, 0, out, q_count * sizeof(float));
    }
    free(qh); free(raw); free(comp); free(mask); free(out); free(scores);
    return ok;
}

int ds4_gpu_attention_prefill_static_mixed_heads_tensor(ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size, uint64_t sinks_offset, const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw_kv, const ds4_gpu_tensor *comp_kv, uint32_t n_tokens, uint32_t n_comp, uint32_t window, uint32_t ratio, uint32_t n_head, uint32_t head_dim) {
    return ascend_attention_prefill_mixed_host(heads, model_map, model_size, sinks_offset, q, raw_kv, comp_kv, NULL, 0, n_tokens, n_comp, window, ratio, n_head, head_dim);
}

int ds4_gpu_attention_prefill_masked_mixed_heads_tensor(ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size, uint64_t sinks_offset, const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw_kv, const ds4_gpu_tensor *comp_kv, const ds4_gpu_tensor *comp_mask, uint32_t n_tokens, uint32_t n_comp, uint32_t window, uint32_t ratio, uint32_t n_head, uint32_t head_dim) {
    return ascend_attention_prefill_mixed_host(heads, model_map, model_size, sinks_offset, q, raw_kv, comp_kv, comp_mask, 1, n_tokens, n_comp, window, ratio, n_head, head_dim);
}
static int ascend_attention_output_low_q8(ds4_gpu_tensor *low, const void *model_map, uint64_t model_size, uint64_t out_a_offset, uint64_t group_dim, uint64_t rank, uint32_t n_groups, const ds4_gpu_tensor *heads, uint32_t n_tokens) {
    if (!low || !model_map || !heads || group_dim == 0 || rank == 0 || n_groups == 0 || n_tokens == 0) return 0;
    if (group_dim > UINT32_MAX || rank > UINT32_MAX) return 0;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint64_t blocks = (group_dim + DS4_ASCEND_QK8_0 - 1) / DS4_ASCEND_QK8_0;
    const uint64_t row_bytes = blocks * DS4_ASCEND_BLOCK_Q8_0_BYTES;
    const uint64_t weight_bytes = low_dim * row_bytes;
    const uint64_t heads_count = (uint64_t)n_tokens * n_groups * group_dim;
    const uint64_t low_count = (uint64_t)n_tokens * low_dim;
    if (heads_count > UINT32_MAX || low_count > UINT32_MAX || out_a_offset > model_size || weight_bytes > model_size - out_a_offset ||
        heads->bytes < heads_count * sizeof(float) || low->bytes < low_count * sizeof(float) || low->device != heads->device) return 0;
    void *w_dev = ascend_model_range_device_ptr(model_map, model_size, out_a_offset, weight_bytes, heads->device, "attn_out_a");
    if (!w_dev) return 0;
    if (!ascend_set_context(heads->device) || !g_streams[heads->device]) return 0;
    /* Quantize heads once (n_tokens*n_groups rows of group_dim), reuse for all rank outputs */
    const uint64_t xq_rows = (uint64_t)n_tokens * n_groups;
    if (xq_rows > UINT32_MAX) return 0;
    const uint64_t xq_bytes = xq_rows * blocks * DS4_ASCEND_QK8_0;
    const uint64_t xscale_bytes = xq_rows * blocks * sizeof(float);
    ds4_gpu_tensor *xq = ascend_tensor_alloc_on_device(xq_bytes, heads->device);
    ds4_gpu_tensor *xscale = ascend_tensor_alloc_on_device(xscale_bytes, heads->device);
    if (!xq || !xscale) {
        ds4_gpu_tensor_free(xq);
        ds4_gpu_tensor_free(xscale);
        /* fall back to inline-quantize kernel */
        ds4_ascend_launch_attention_output_low_q8(g_streams[heads->device], low->ptr, w_dev, heads->ptr, (uint32_t)group_dim, (uint32_t)rank, n_groups, n_tokens);
        return 1;
    }
    ds4_ascend_launch_quantize_q8_0(g_streams[heads->device], xq->ptr, xscale->ptr, heads->ptr, (uint32_t)group_dim, (uint32_t)xq_rows);
    ds4_ascend_launch_attention_output_low_q8_prequant(g_streams[heads->device], low->ptr, w_dev, xq->ptr, xscale->ptr, (uint32_t)group_dim, (uint32_t)rank, n_groups, n_tokens);
    int ok = ascend_defer_free_tensor(xq);
    ok = ascend_defer_free_tensor(xscale) && ok;
    return ok;
}

int ds4_gpu_attention_output_q8_batch_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *low, ds4_gpu_tensor *group_tmp, ds4_gpu_tensor *low_tmp, const void *model_map, uint64_t model_size, uint64_t out_a_offset, uint64_t out_b_offset, uint64_t group_dim, uint64_t rank, uint32_t n_groups, uint64_t out_dim, const ds4_gpu_tensor *heads, uint32_t n_tokens) {
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    return ascend_attention_output_low_q8(low, model_map, model_size, out_a_offset, group_dim, rank, n_groups, heads, n_tokens) &&
           ascend_matmul_q8_0_tensor(out, model_map, model_size, out_b_offset, low_dim, out_dim, low, n_tokens);
}

int ds4_gpu_attention_output_low_q8_tensor(ds4_gpu_tensor *low, const void *model_map, uint64_t model_size, uint64_t out_a_offset, uint64_t group_dim, uint64_t rank, uint32_t n_groups, const ds4_gpu_tensor *heads) {
    return ascend_attention_output_low_q8(low, model_map, model_size, out_a_offset, group_dim, rank, n_groups, heads, 1);
}
int ds4_gpu_swiglu_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up, uint32_t n, float clamp, float weight) {
    return ascend_swiglu_device(out, gate, up, n, clamp, weight) || ascend_swiglu_host(out, gate, up, n, clamp, weight);
}

int ds4_gpu_add_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *a, const ds4_gpu_tensor *b, uint32_t n) {
    if (!out || !a || !b || out->bytes < (uint64_t)n * sizeof(float) || a->bytes < (uint64_t)n * sizeof(float) || b->bytes < (uint64_t)n * sizeof(float)) return 0;
    if (out->device != a->device || out->device != b->device) return ascend_unimplemented("cross-device add");
    if (!ascend_set_context(out->device) || !g_streams[out->device]) return 0;
    ds4_ascend_launch_add_f32(g_streams[out->device], out->ptr, a->ptr, b->ptr, n);
    return 1;
}
int ds4_gpu_directional_steering_project_tensor(ds4_gpu_tensor *x, const ds4_gpu_tensor *directions, uint32_t layer, uint32_t width, uint32_t rows, float scale) { return ascend_unimplemented(__func__); }

static int ascend_router_select(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, ds4_gpu_tensor *probs, const void *model_map, uint64_t model_size, uint64_t bias_offset, uint64_t hash_offset, uint32_t hash_rows, int32_t token_scalar, uint32_t n_expert_groups, uint32_t n_group_used, bool has_bias, bool hash_mode, const ds4_gpu_tensor *logits, const ds4_gpu_tensor *tokens, uint32_t n_tokens) {
    if (!selected || !weights || !probs || !logits || !model_map || n_tokens == 0 ||
        n_expert_groups > 1u || n_group_used > 0u || n_tokens > UINT32_MAX ||
        logits->bytes < (uint64_t)n_tokens * DS4_ASCEND_N_EXPERT * sizeof(float) ||
        probs->bytes < (uint64_t)n_tokens * DS4_ASCEND_N_EXPERT * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * DS4_ASCEND_N_EXPERT_USED * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tokens * DS4_ASCEND_N_EXPERT_USED * sizeof(float) ||
        selected->device != logits->device || selected->device != weights->device || selected->device != probs->device ||
        (tokens && (tokens->bytes < (uint64_t)n_tokens * sizeof(int32_t) || tokens->device != selected->device))) return 0;

    void *bias = NULL;
    void *hash = NULL;
    if (has_bias && !hash_mode) {
        bias = ascend_model_range_device_ptr(model_map, model_size, bias_offset, DS4_ASCEND_N_EXPERT * sizeof(float), logits->device, "router_bias");
        if (!bias) return 0;
    }
    if (hash_mode) {
        if (hash_rows == 0) return 0;
        hash = ascend_model_range_device_ptr(model_map, model_size, hash_offset, (uint64_t)hash_rows * DS4_ASCEND_N_EXPERT_USED * sizeof(int32_t), logits->device, "router_hash");
        if (!hash) return 0;
    }

    if (!ascend_set_context(selected->device) || !g_streams[selected->device]) return 0;
    ds4_ascend_launch_router_select(g_streams[selected->device], selected->ptr, weights->ptr, probs->ptr, bias, hash, logits->ptr, tokens ? tokens->ptr : NULL, token_scalar, hash_rows, n_tokens, has_bias && !hash_mode ? 1u : 0u, hash_mode ? 1u : 0u);
    return 1;
}

int ds4_gpu_router_select_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, ds4_gpu_tensor *probs, const void *model_map, uint64_t model_size, uint64_t bias_offset, uint64_t hash_offset, uint32_t hash_rows, uint32_t token, uint32_t n_expert_groups, uint32_t n_group_used, bool has_bias, bool hash_mode, const ds4_gpu_tensor *logits) {
    return ascend_router_select(selected, weights, probs, model_map, model_size, bias_offset, hash_offset, hash_rows, (int32_t)token, n_expert_groups, n_group_used, has_bias, hash_mode, logits, NULL, 1);
}

int ds4_gpu_router_select_batch_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, ds4_gpu_tensor *probs, const void *model_map, uint64_t model_size, uint64_t bias_offset, uint64_t hash_offset, uint32_t hash_rows, uint32_t n_expert_groups, uint32_t n_group_used, bool has_bias, bool hash_mode, const ds4_gpu_tensor *logits, const ds4_gpu_tensor *tokens, uint32_t n_tokens) {
    return ascend_router_select(selected, weights, probs, model_map, model_size, bias_offset, hash_offset, hash_rows, 0, n_expert_groups, n_group_used, has_bias, hash_mode, logits, tokens, n_tokens);
}

static ascend_weight_cache *ascend_require_expert_cache(uint64_t offset, uint64_t bytes, ascend_weight_kind kind, uint32_t type, uint64_t in_dim, uint64_t out_dim) {
    ascend_weight_cache *w = ascend_weight_cache_find(offset, bytes);
    if (!w || w->kind != kind || w->type != type || w->in_dim != in_dim || w->out_dim != out_dim) return NULL;
    if (g_device_count < 2 || bytes % DS4_ASCEND_N_EXPERT != 0) return NULL;
    const uint64_t expert_bytes = bytes / DS4_ASCEND_N_EXPERT;
    for (uint32_t d = 0; d < 2; d++) {
        if (!w->device[d] || !w->device[d]->ptr || w->device[d]->device != (int)d ||
            w->expert_count[d] == 0 || w->expert_begin[d] + w->expert_count[d] > DS4_ASCEND_N_EXPERT ||
            w->device_offset[d] != offset + (uint64_t)w->expert_begin[d] * expert_bytes ||
            w->device_bytes[d] != (uint64_t)w->expert_count[d] * expert_bytes) return NULL;
    }
    return w;
}

static int ascend_device_to_device_copy(void *dst, int dst_device, const void *src, int src_device, uint64_t bytes, const char *what) {
    if (!dst || !src || bytes == 0) return bytes == 0;
    if (dst_device < 0 || src_device < 0 || dst_device >= (int)g_device_count || src_device >= (int)g_device_count) return 0;
    if (g_streams[src_device]) {
        if (!ascend_set_context(src_device)) return 0;
        if (!ascend_ok(aclrtSynchronizeStream(g_streams[src_device]), what ? what : "sync source before device copy")) return 0;
    }
    if (!ascend_set_context(dst_device)) return 0;
    if (dst_device == src_device) {
        return ascend_ok(aclrtMemcpy(dst, (size_t)bytes, src, (size_t)bytes, ACL_MEMCPY_INNER_DEVICE_TO_DEVICE), what ? what : "device copy");
    }
    if (!g_streams[dst_device]) return 0;
    if (!ascend_ok(aclrtSynchronizeStream(g_streams[dst_device]), what ? what : "sync destination before device copy")) return 0;
    if (!ascend_ok(aclrtMemcpyAsync(dst, (size_t)bytes, src, (size_t)bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, g_streams[dst_device]), what ? what : "device copy async")) return 0;
    return ascend_ok(aclrtSynchronizeStream(g_streams[dst_device]), what ? what : "sync device copy async");
}

static int ascend_all_moe_tensors_on_device(int device, const ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up, const ds4_gpu_tensor *mid, const ds4_gpu_tensor *experts, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, const ds4_gpu_tensor *x) {
    return out->device == device && gate->device == device && up->device == device && mid->device == device &&
           experts->device == device && selected->device == device && weights->device == device && x->device == device;
}

static int ascend_expert_cache_shards_match(const ascend_weight_cache *a, const ascend_weight_cache *b) {
    if (!a || !b) return 0;
    for (uint32_t d = 0; d < 2; d++) {
        if (a->expert_begin[d] != b->expert_begin[d] || a->expert_count[d] != b->expert_count[d]) return 0;
    }
    return 1;
}

static int ascend_routed_moe_device_iq2_q2(ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *experts, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_expert, float clamp, const ds4_gpu_tensor *x, uint32_t n_tokens, bool *mid_is_f16) {
    const char *disable = getenv("DS4_ASCEND_DISABLE_DEVICE_MOE");
    if (disable && disable[0] && strcmp(disable, "0") != 0) return 0;
    if (g_device_count < 2 || n_tokens == 0 || n_expert != DS4_ASCEND_N_EXPERT_USED) return 0;
    if (expert_in_dim != 4096u || expert_mid_dim != 2048u || out_dim != 4096u) return 0;
    const int canonical = x->device;
    if (canonical < 0 || canonical >= (int)g_device_count || !ascend_all_moe_tensors_on_device(canonical, out, gate, up, mid, experts, selected, weights, x)) return 0;
    if (canonical > 1) return 0;
    const int remote = canonical == 0 ? 1 : 0;

    const uint64_t gate_bytes = DS4_ASCEND_N_EXPERT * gate_expert_bytes;
    const uint64_t down_bytes = DS4_ASCEND_N_EXPERT * down_expert_bytes;
    ascend_weight_cache *gate_cache = ascend_require_expert_cache(gate_offset, gate_bytes, DS4_ASCEND_WEIGHT_GATE_EXPERT, DS4_ASCEND_TENSOR_IQ2_XXS, expert_in_dim, expert_mid_dim);
    ascend_weight_cache *up_cache = ascend_require_expert_cache(up_offset, gate_bytes, DS4_ASCEND_WEIGHT_UP_EXPERT, DS4_ASCEND_TENSOR_IQ2_XXS, expert_in_dim, expert_mid_dim);
    ascend_weight_cache *down_cache = ascend_require_expert_cache(down_offset, down_bytes, DS4_ASCEND_WEIGHT_DOWN_EXPERT, DS4_ASCEND_TENSOR_Q2_K, expert_mid_dim, out_dim);
    if (!gate_cache || !up_cache || !down_cache ||
        !ascend_expert_cache_shards_match(gate_cache, up_cache) ||
        !ascend_expert_cache_shards_match(gate_cache, down_cache)) return 0;

    const uint64_t pair_count64 = (uint64_t)n_tokens * n_expert;
    if (pair_count64 > UINT32_MAX ||
        pair_count64 * expert_mid_dim > UINT32_MAX ||
        pair_count64 * out_dim > UINT32_MAX ||
        (uint64_t)n_tokens * out_dim > UINT32_MAX) return 0;
    const uint32_t pair_count = (uint32_t)pair_count64;
    const uint32_t in_blocks = expert_in_dim / DS4_ASCEND_QK_K;
    const uint32_t mid_blocks = expert_mid_dim / DS4_ASCEND_QK_K;
    const uint64_t xq_bytes = (uint64_t)n_tokens * in_blocks * sizeof(ascend_block_q8_K);
    const uint64_t midq_bytes = pair_count64 * mid_blocks * sizeof(ascend_block_q8_K);
    const uint64_t selected_bytes = pair_count64 * sizeof(int32_t);
    const uint64_t weights_bytes = pair_count64 * sizeof(float);
    const uint64_t mid_float_bytes = pair_count64 * expert_mid_dim * sizeof(float);
    const uint64_t expert_float_bytes = pair_count64 * out_dim * sizeof(float);

    ds4_gpu_tensor *xq_canon = ascend_tensor_alloc_on_device(xq_bytes, canonical);
    ds4_gpu_tensor *midq_canon = ascend_tensor_alloc_on_device(midq_bytes, canonical);
    ds4_gpu_tensor *merge_tmp = ascend_tensor_alloc_on_device(expert_float_bytes > mid_float_bytes ? expert_float_bytes : mid_float_bytes, canonical);
    ds4_gpu_tensor *selected_remote = ascend_tensor_alloc_on_device(selected_bytes, remote);
    ds4_gpu_tensor *weights_remote = ascend_tensor_alloc_on_device(weights_bytes, remote);
    ds4_gpu_tensor *xq_remote = ascend_tensor_alloc_on_device(xq_bytes, remote);
    ds4_gpu_tensor *gate_remote = ascend_tensor_alloc_on_device(mid_float_bytes, remote);
    ds4_gpu_tensor *up_remote = ascend_tensor_alloc_on_device(mid_float_bytes, remote);
    ds4_gpu_tensor *mid_remote = ascend_tensor_alloc_on_device(mid_float_bytes, remote);
    ds4_gpu_tensor *midq_remote = ascend_tensor_alloc_on_device(midq_bytes, remote);
    ds4_gpu_tensor *experts_remote = ascend_tensor_alloc_on_device(expert_float_bytes, remote);
    int ok = xq_canon && midq_canon && merge_tmp && selected_remote && weights_remote && xq_remote && gate_remote && up_remote && mid_remote && midq_remote && experts_remote;

    if (ok) ok = ds4_gpu_ascend_quantize_q8_k_tensor(xq_canon, x, n_tokens, expert_in_dim);
    if (ok) ok = ascend_device_to_device_copy(xq_remote->ptr, remote, xq_canon->ptr, canonical, xq_bytes, "copy moe xq");
    if (ok) ok = ascend_device_to_device_copy(selected_remote->ptr, remote, selected->ptr, canonical, selected_bytes, "copy moe selected");
    if (ok) ok = ascend_device_to_device_copy(weights_remote->ptr, remote, weights->ptr, canonical, weights_bytes, "copy moe weights");

    if (ok) ok = ascend_require_iq2_tables(canonical) && ascend_require_iq2_tables(remote);
    if (ok) {
        if (!ascend_set_context(canonical) || !g_streams[canonical]) ok = 0;
        else ds4_ascend_launch_moe_gate_up_mid_iq2_q8(g_streams[canonical], gate->ptr, up->ptr, mid->ptr, gate_cache->device[canonical]->ptr, up_cache->device[canonical]->ptr, xq_canon->ptr, selected->ptr, weights->ptr, g_iq2_ksigns_dev[canonical]->ptr, g_iq2_grid_dev[canonical]->ptr, pair_count, n_expert, gate_cache->expert_begin[canonical], gate_cache->expert_count[canonical], expert_in_dim, expert_mid_dim, gate_expert_bytes, gate_row_bytes, clamp);
    }
    if (ok) {
        if (!ascend_set_context(remote) || !g_streams[remote]) ok = 0;
        else ds4_ascend_launch_moe_gate_up_mid_iq2_q8(g_streams[remote], gate_remote->ptr, up_remote->ptr, mid_remote->ptr, gate_cache->device[remote]->ptr, up_cache->device[remote]->ptr, xq_remote->ptr, selected_remote->ptr, weights_remote->ptr, g_iq2_ksigns_dev[remote]->ptr, g_iq2_grid_dev[remote]->ptr, pair_count, n_expert, gate_cache->expert_begin[remote], gate_cache->expert_count[remote], expert_in_dim, expert_mid_dim, gate_expert_bytes, gate_row_bytes, clamp);
    }

    if (ok) ok = ascend_device_to_device_copy(merge_tmp->ptr, canonical, gate_remote->ptr, remote, mid_float_bytes, "copy moe gate remote");
    if (ok) {
        if (!ascend_set_context(canonical) || !g_streams[canonical]) ok = 0;
        else ds4_ascend_launch_moe_merge_width(g_streams[canonical], gate->ptr, merge_tmp->ptr, selected->ptr, pair_count, expert_mid_dim, gate_cache->expert_begin[remote], gate_cache->expert_count[remote]);
    }
    if (ok) ok = ascend_device_to_device_copy(merge_tmp->ptr, canonical, up_remote->ptr, remote, mid_float_bytes, "copy moe up remote");
    if (ok) {
        if (!ascend_set_context(canonical) || !g_streams[canonical]) ok = 0;
        else ds4_ascend_launch_moe_merge_width(g_streams[canonical], up->ptr, merge_tmp->ptr, selected->ptr, pair_count, expert_mid_dim, up_cache->expert_begin[remote], up_cache->expert_count[remote]);
    }
    if (ok) ok = ascend_device_to_device_copy(merge_tmp->ptr, canonical, mid_remote->ptr, remote, mid_float_bytes, "copy moe mid remote");
    if (ok) {
        if (!ascend_set_context(canonical) || !g_streams[canonical]) ok = 0;
        else ds4_ascend_launch_moe_merge_width(g_streams[canonical], mid->ptr, merge_tmp->ptr, selected->ptr, pair_count, expert_mid_dim, up_cache->expert_begin[remote], up_cache->expert_count[remote]);
    }

    if (ok) ok = ds4_gpu_ascend_quantize_q8_k_tensor(midq_canon, mid, pair_count, expert_mid_dim);
    if (ok) ok = ds4_gpu_ascend_quantize_q8_k_tensor(midq_remote, mid_remote, pair_count, expert_mid_dim);

    if (ok) {
        if (!ascend_set_context(canonical) || !g_streams[canonical]) ok = 0;
        else ds4_ascend_launch_moe_down_q2_q8(g_streams[canonical], experts->ptr, down_cache->device[canonical]->ptr, midq_canon->ptr, selected->ptr, pair_count, down_cache->expert_begin[canonical], down_cache->expert_count[canonical], expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
    }
    if (ok) {
        if (!ascend_set_context(remote) || !g_streams[remote]) ok = 0;
        else ds4_ascend_launch_moe_down_q2_q8(g_streams[remote], experts_remote->ptr, down_cache->device[remote]->ptr, midq_remote->ptr, selected_remote->ptr, pair_count, down_cache->expert_begin[remote], down_cache->expert_count[remote], expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
    }
    if (ok) ok = ascend_device_to_device_copy(merge_tmp->ptr, canonical, experts_remote->ptr, remote, expert_float_bytes, "copy moe experts remote");
    if (ok) {
        if (!ascend_set_context(canonical) || !g_streams[canonical]) ok = 0;
        else ds4_ascend_launch_moe_merge_width(g_streams[canonical], experts->ptr, merge_tmp->ptr, selected->ptr, pair_count, out_dim, down_cache->expert_begin[remote], down_cache->expert_count[remote]);
    }
    if (ok) {
        if (!ascend_set_context(canonical) || !g_streams[canonical]) ok = 0;
        else ds4_ascend_launch_moe_sum_experts(g_streams[canonical], out->ptr, experts->ptr, n_tokens, n_expert, out_dim);
    }
    if (ok && mid_is_f16) *mid_is_f16 = false;

    ds4_gpu_tensor_free(experts_remote);
    ds4_gpu_tensor_free(midq_remote);
    ds4_gpu_tensor_free(mid_remote);
    ds4_gpu_tensor_free(up_remote);
    ds4_gpu_tensor_free(gate_remote);
    ds4_gpu_tensor_free(xq_remote);
    ds4_gpu_tensor_free(weights_remote);
    ds4_gpu_tensor_free(selected_remote);
    ds4_gpu_tensor_free(merge_tmp);
    ds4_gpu_tensor_free(midq_canon);
    ds4_gpu_tensor_free(xq_canon);
    return ok;
}

int ds4_gpu_routed_moe_one_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *experts, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_expert, float clamp, const ds4_gpu_tensor *x) {
    return ds4_gpu_routed_moe_batch_tensor(out, gate, up, mid, experts, model_map, model_size,
                                           gate_offset, up_offset, down_offset,
                                           gate_type, down_type,
                                           gate_expert_bytes, gate_row_bytes,
                                           down_expert_bytes, down_row_bytes,
                                           expert_in_dim, expert_mid_dim, out_dim,
                                           selected, weights, n_expert, clamp, x, 1, NULL);
}

int ds4_gpu_routed_moe_batch_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *experts, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_expert, float clamp, const ds4_gpu_tensor *x, uint32_t n_tokens, bool *mid_is_f16) {
    if (!out || !gate || !up || !mid || !experts || !model_map || !selected || !weights || !x ||
        n_tokens == 0 || n_expert == 0 || n_expert > 16 ||
        gate_type != DS4_ASCEND_TENSOR_IQ2_XXS || down_type != DS4_ASCEND_TENSOR_Q2_K ||
        expert_in_dim == 0 || expert_mid_dim == 0 || out_dim == 0 ||
        expert_in_dim % DS4_ASCEND_QK_K != 0 || expert_mid_dim % DS4_ASCEND_QK_K != 0) {
        return 0;
    }

    const uint64_t pair_count = (uint64_t)n_tokens * n_expert;
    const uint64_t x_count = (uint64_t)n_tokens * expert_in_dim;
    const uint64_t mid_count = pair_count * expert_mid_dim;
    const uint64_t down_count = pair_count * out_dim;
    const uint64_t out_count = (uint64_t)n_tokens * out_dim;
    const uint64_t gate_bytes = DS4_ASCEND_N_EXPERT * gate_expert_bytes;
    const uint64_t down_bytes = DS4_ASCEND_N_EXPERT * down_expert_bytes;
    if (gate_offset > model_size || up_offset > model_size || down_offset > model_size ||
        gate_bytes > model_size - gate_offset || gate_bytes > model_size - up_offset ||
        down_bytes > model_size - down_offset ||
        gate_row_bytes != (uint64_t)(expert_in_dim / DS4_ASCEND_QK_K) * DS4_ASCEND_BLOCK_IQ2_XXS_BYTES ||
        down_row_bytes != (uint64_t)(expert_mid_dim / DS4_ASCEND_QK_K) * DS4_ASCEND_BLOCK_Q2_K_BYTES ||
        gate_expert_bytes != (uint64_t)expert_mid_dim * gate_row_bytes ||
        down_expert_bytes != (uint64_t)out_dim * down_row_bytes ||
        x->bytes < x_count * sizeof(float) ||
        selected->bytes < pair_count * sizeof(int32_t) ||
        weights->bytes < pair_count * sizeof(float) ||
        gate->bytes < mid_count * sizeof(float) ||
        up->bytes < mid_count * sizeof(float) ||
        mid->bytes < mid_count * sizeof(float) ||
        experts->bytes < down_count * sizeof(float) ||
        out->bytes < out_count * sizeof(float)) {
        return 0;
    }
    if (!ascend_cache_kind_matches(gate_offset, gate_bytes, DS4_ASCEND_WEIGHT_GATE_EXPERT) ||
        !ascend_cache_kind_matches(up_offset, gate_bytes, DS4_ASCEND_WEIGHT_UP_EXPERT) ||
        !ascend_cache_kind_matches(down_offset, down_bytes, DS4_ASCEND_WEIGHT_DOWN_EXPERT)) {
        return 0;
    }
    if (ascend_routed_moe_device_iq2_q2(out, gate, up, mid, experts,
                                        gate_offset, up_offset, down_offset,
                                        gate_expert_bytes, gate_row_bytes,
                                        down_expert_bytes, down_row_bytes,
                                        expert_in_dim, expert_mid_dim, out_dim,
                                        selected, weights, n_expert, clamp, x,
                                        n_tokens, mid_is_f16)) {
        return 1;
    }
    const uint64_t routed_to_host = pair_count * (sizeof(int32_t) + sizeof(float)) + x_count * sizeof(ascend_block_q8_K) / DS4_ASCEND_QK_K + mid_count * sizeof(ascend_block_q8_K) / DS4_ASCEND_QK_K;
    const uint64_t routed_to_device = (3u * mid_count + down_count + out_count) * sizeof(float);
    if (!ascend_host_fallback_begin("routed_moe_iq2_q2", routed_to_host, routed_to_device)) return 0;

    const char *gate_w = ascend_model_range_ptr(model_map, model_size, gate_offset, gate_bytes, "moe_gate");
    const char *up_w = ascend_model_range_ptr(model_map, model_size, up_offset, gate_bytes, "moe_up");
    const char *down_w = ascend_model_range_ptr(model_map, model_size, down_offset, down_bytes, "moe_down");
    if (!gate_w || !up_w || !down_w) return 0;

    const char *trace_env = getenv("DS4_ASCEND_TRACE_MOE");
    const int trace_moe = trace_env && trace_env[0] && strcmp(trace_env, "0") != 0;
    uint64_t trace_iq2_dot_calls = 0;
    uint64_t trace_q2_dot_calls = 0;

    int32_t *selected_host = malloc((size_t)(pair_count * sizeof(int32_t)));
    float *weights_host = malloc((size_t)(pair_count * sizeof(float)));
    float *gate_host = malloc((size_t)(mid_count * sizeof(float)));
    float *up_host = malloc((size_t)(mid_count * sizeof(float)));
    float *mid_host = malloc((size_t)(mid_count * sizeof(float)));
    float *experts_host = malloc((size_t)(down_count * sizeof(float)));
    float *out_host = calloc((size_t)out_count, sizeof(float));
    const uint32_t in_blocks = expert_in_dim / DS4_ASCEND_QK_K;
    const uint32_t mid_blocks = expert_mid_dim / DS4_ASCEND_QK_K;
    const uint64_t xq_bytes = (uint64_t)n_tokens * in_blocks * sizeof(ascend_block_q8_K);
    const uint64_t midq_bytes = pair_count * mid_blocks * sizeof(ascend_block_q8_K);
    ascend_block_q8_K *xq_host = malloc((size_t)xq_bytes);
    ascend_block_q8_K *midq_host = malloc((size_t)midq_bytes);
    ds4_gpu_tensor *xq_dev = ascend_tensor_alloc_on_device(xq_bytes, x->device);
    ds4_gpu_tensor *midq_dev = ascend_tensor_alloc_on_device(midq_bytes, mid->device);
    int ok = selected_host && weights_host && gate_host && up_host && mid_host && experts_host && out_host && xq_host && midq_host && xq_dev && midq_dev;
    if (ok) ok = ds4_gpu_tensor_read(selected, 0, selected_host, pair_count * sizeof(int32_t));
    if (ok) ok = ds4_gpu_tensor_read(weights, 0, weights_host, pair_count * sizeof(float));
    if (ok) ok = ds4_gpu_ascend_quantize_q8_k_tensor(xq_dev, x, n_tokens, expert_in_dim);
    if (ok) ok = ds4_gpu_synchronize();
    if (ok) ok = ds4_gpu_tensor_read(xq_dev, 0, xq_host, xq_bytes);

    if (ok) {
        for (uint64_t pair = 0; pair < pair_count && ok; pair++) {
            const uint32_t tok = (uint32_t)(pair / n_expert);
            const uint32_t slot = (uint32_t)(pair - (uint64_t)tok * n_expert);
            int32_t expert_i = selected_host[pair];
            if (expert_i < 0) expert_i = 0;
            if (expert_i >= (int32_t)DS4_ASCEND_N_EXPERT) {
                ok = 0;
                break;
            }
            const uint32_t expert = (uint32_t)expert_i;
            const ascend_block_q8_K *xq = xq_host + (uint64_t)tok * in_blocks;
            const float router_weight = weights_host[(uint64_t)tok * n_expert + slot];
            const char *gate_base = gate_w + (uint64_t)expert * gate_expert_bytes;
            const char *up_base = up_w + (uint64_t)expert * gate_expert_bytes;
            for (uint32_t row = 0; row < expert_mid_dim; row++) {
                const ascend_block_iq2_xxs *gr = (const ascend_block_iq2_xxs *)(gate_base + (uint64_t)row * gate_row_bytes);
                const ascend_block_iq2_xxs *ur = (const ascend_block_iq2_xxs *)(up_base + (uint64_t)row * gate_row_bytes);
                float gv = ascend_vec_dot_iq2_xxs_q8_K(expert_in_dim, gr, xq);
                float uv = ascend_vec_dot_iq2_xxs_q8_K(expert_in_dim, ur, xq);
                if (trace_moe) trace_iq2_dot_calls += 2;
                if (clamp > 1.0e-6f) {
                    if (gv > clamp) gv = clamp;
                    if (uv > clamp) uv = clamp;
                    if (uv < -clamp) uv = -clamp;
                }
                const uint64_t off = pair * expert_mid_dim + row;
                gate_host[off] = gv;
                up_host[off] = uv;
                mid_host[off] = ascend_silu(gv) * uv * router_weight;
            }
        }
    }
    if (ok) ok = ds4_gpu_tensor_write(mid, 0, mid_host, mid_count * sizeof(float));
    if (ok) ok = ds4_gpu_ascend_quantize_q8_k_tensor(midq_dev, mid, (uint32_t)pair_count, expert_mid_dim);
    if (ok) ok = ds4_gpu_synchronize();
    if (ok) ok = ds4_gpu_tensor_read(midq_dev, 0, midq_host, midq_bytes);
    if (ok) {
        for (uint64_t pair = 0; pair < pair_count && ok; pair++) {
            const uint32_t tok = (uint32_t)(pair / n_expert);
            int32_t expert_i = selected_host[pair];
            if (expert_i < 0) expert_i = 0;
            if (expert_i >= (int32_t)DS4_ASCEND_N_EXPERT) {
                ok = 0;
                break;
            }
            const char *down_base = down_w + (uint64_t)(uint32_t)expert_i * down_expert_bytes;
            const ascend_block_q8_K *midq = midq_host + pair * mid_blocks;
            for (uint32_t row = 0; row < out_dim; row++) {
                const ascend_block_q2_K *dr = (const ascend_block_q2_K *)(down_base + (uint64_t)row * down_row_bytes);
                const float v = ascend_vec_dot_q2_K_q8_K(expert_mid_dim, dr, midq);
                if (trace_moe) trace_q2_dot_calls++;
                experts_host[pair * out_dim + row] = v;
                out_host[(uint64_t)tok * out_dim + row] += v;
            }
        }
    }

    if (ok) ok = ds4_gpu_tensor_write(gate, 0, gate_host, mid_count * sizeof(float));
    if (ok) ok = ds4_gpu_tensor_write(up, 0, up_host, mid_count * sizeof(float));
    if (ok) ok = ds4_gpu_tensor_write(mid, 0, mid_host, mid_count * sizeof(float));
    if (ok) ok = ds4_gpu_tensor_write(experts, 0, experts_host, down_count * sizeof(float));
    if (ok) ok = ds4_gpu_tensor_write(out, 0, out_host, out_count * sizeof(float));
    if (ok && mid_is_f16) *mid_is_f16 = false;
    if (trace_moe) {
        fprintf(stderr,
                "ds4: Ascend MoE fallback trace: tokens=%u experts_per_token=%u iq2_xxs_dot_calls=%" PRIu64 " q2_k_dot_calls=%" PRIu64 "\n",
                n_tokens,
                n_expert,
                trace_iq2_dot_calls,
                trace_q2_dot_calls);
    }

    ds4_gpu_tensor_free(midq_dev);
    ds4_gpu_tensor_free(xq_dev);
    free(selected_host);
    free(weights_host);
    free(gate_host);
    free(up_host);
    free(mid_host);
    free(experts_host);
    free(out_host);
    free(xq_host);
    free(midq_host);
    return ok;
}
static uint32_t ascend_hc_mix_count(uint32_t n_hc) {
    return 2u * n_hc + n_hc * n_hc;
}

static float ascend_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static void ascend_hc_split_one(float *out, const float *mix, const float *scale, const float *base, uint32_t n_hc, uint32_t sinkhorn_iters, float eps) {
    const float pre_scale = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    for (uint32_t i = 0; i < n_hc; i++) out[i] = ascend_sigmoid(mix[i] * pre_scale + base[i]) + eps;
    for (uint32_t i = 0; i < n_hc; i++) {
        const uint32_t off = n_hc + i;
        out[off] = 2.0f * ascend_sigmoid(mix[off] * post_scale + base[off]);
    }

    float c[16 * 16];
    for (uint32_t dst = 0; dst < n_hc; dst++) {
        float row_max = -3.402823466e+38f;
        for (uint32_t src = 0; src < n_hc; src++) {
            const uint32_t idx = src + dst * n_hc;
            const uint32_t off = 2u * n_hc + idx;
            const float v = mix[off] * comb_scale + base[off];
            c[idx] = v;
            if (v > row_max) row_max = v;
        }
        float row_sum = 0.0f;
        for (uint32_t src = 0; src < n_hc; src++) {
            const uint32_t idx = src + dst * n_hc;
            const float v = expf(c[idx] - row_max);
            c[idx] = v;
            row_sum += v;
        }
        const float inv = 1.0f / row_sum;
        for (uint32_t src = 0; src < n_hc; src++) {
            const uint32_t idx = src + dst * n_hc;
            c[idx] = c[idx] * inv + eps;
        }
    }

    for (uint32_t src = 0; src < n_hc; src++) {
        float sum = 0.0f;
        for (uint32_t dst = 0; dst < n_hc; dst++) sum += c[src + dst * n_hc];
        const float inv = 1.0f / (sum + eps);
        for (uint32_t dst = 0; dst < n_hc; dst++) c[src + dst * n_hc] *= inv;
    }

    for (uint32_t iter = 1; iter < sinkhorn_iters; iter++) {
        for (uint32_t dst = 0; dst < n_hc; dst++) {
            float sum = 0.0f;
            for (uint32_t src = 0; src < n_hc; src++) sum += c[src + dst * n_hc];
            const float inv = 1.0f / (sum + eps);
            for (uint32_t src = 0; src < n_hc; src++) c[src + dst * n_hc] *= inv;
        }
        for (uint32_t src = 0; src < n_hc; src++) {
            float sum = 0.0f;
            for (uint32_t dst = 0; dst < n_hc; dst++) sum += c[src + dst * n_hc];
            const float inv = 1.0f / (sum + eps);
            for (uint32_t dst = 0; dst < n_hc; dst++) c[src + dst * n_hc] *= inv;
        }
    }

    for (uint32_t i = 0; i < n_hc * n_hc; i++) out[2u * n_hc + i] = c[i];
}

int ds4_gpu_hc_split_sinkhorn_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *mix, const void *model_map, uint64_t model_size, uint64_t scale_offset, uint64_t base_offset, uint32_t n_hc, uint32_t sinkhorn_iters, float eps) {
    if (!out || !mix || !model_map || n_hc == 0 || n_hc > 16) return 0;
    const uint32_t mix_hc = ascend_hc_mix_count(n_hc);
    const uint64_t row_bytes = (uint64_t)mix_hc * sizeof(float);
    const uint64_t rows = mix->bytes / row_bytes;
    if (rows == 0 || out->bytes < rows * row_bytes || scale_offset > model_size || base_offset > model_size ||
        3u * sizeof(float) > model_size - scale_offset || row_bytes > model_size - base_offset) return 0;
    if (n_hc == 4 && rows <= UINT32_MAX && out->device == mix->device) {
        void *scale_dev = ascend_model_range_device_ptr(model_map, model_size, scale_offset, 3u * sizeof(float), mix->device, "hc_split_scale");
        void *base_dev = ascend_model_range_device_ptr(model_map, model_size, base_offset, row_bytes, mix->device, "hc_split_base");
        if (!scale_dev || !base_dev) return 0;
        if (!ascend_set_context(mix->device) || !g_streams[mix->device]) return 0;
        ds4_ascend_launch_hc_split_sinkhorn4(g_streams[mix->device], out->ptr, mix->ptr, scale_dev, base_dev, (uint32_t)rows, sinkhorn_iters, eps);
        return 1;
    }
    if (!ascend_host_fallback_begin("hc_split_sinkhorn", rows * row_bytes, rows * row_bytes)) return 0;
    float *mix_host = malloc((size_t)(rows * row_bytes));
    float *out_host = malloc((size_t)(rows * row_bytes));
    if (!mix_host || !out_host) {
        free(mix_host);
        free(out_host);
        return 0;
    }
    int ok = ds4_gpu_tensor_read(mix, 0, mix_host, rows * row_bytes);
    const float *scale = (const float *)((const uint8_t *)model_map + scale_offset);
    const float *base = (const float *)((const uint8_t *)model_map + base_offset);
    if (ok) {
        for (uint64_t r = 0; r < rows; r++) ascend_hc_split_one(out_host + r * mix_hc, mix_host + r * mix_hc, scale, base, n_hc, sinkhorn_iters, eps);
        ok = ds4_gpu_tensor_write(out, 0, out_host, rows * row_bytes);
    }
    free(mix_host);
    free(out_host);
    return ok;
}

static int ascend_hc_weighted_sum_host(ds4_gpu_tensor *out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *weights, uint32_t n_embd, uint32_t n_hc, uint32_t weight_stride) {
    if (!out || !residual_hc || !weights || n_embd == 0 || n_hc == 0) return 0;
    const uint64_t rows = out->bytes / ((uint64_t)n_embd * sizeof(float));
    if (rows == 0 || residual_hc->bytes < rows * n_hc * (uint64_t)n_embd * sizeof(float) || weights->bytes < ((rows - 1) * (uint64_t)weight_stride + n_hc) * sizeof(float)) return 0;
    const uint64_t weights_count = (rows - 1) * (uint64_t)weight_stride + n_hc;
    const uint64_t residual_count = rows * n_hc * (uint64_t)n_embd;
    const uint64_t out_count = rows * (uint64_t)n_embd;
    if (n_hc == 4 && rows <= UINT32_MAX && out_count <= UINT32_MAX && out->device == residual_hc->device && out->device == weights->device) {
        if (!ascend_set_context(out->device) || !g_streams[out->device]) return 0;
        ds4_ascend_launch_hc_weighted_sum4(g_streams[out->device], out->ptr, residual_hc->ptr, weights->ptr, n_embd, (uint32_t)rows, weight_stride);
        return 1;
    }
    if (!ascend_host_fallback_begin("hc_weighted_sum", (residual_count + weights_count) * sizeof(float), out_count * sizeof(float))) return 0;
    float *res_host = malloc((size_t)(rows * n_hc * (uint64_t)n_embd * sizeof(float)));
    float *w_host = malloc((size_t)(((rows - 1) * (uint64_t)weight_stride + n_hc) * sizeof(float)));
    float *out_host = malloc((size_t)(rows * (uint64_t)n_embd * sizeof(float)));
    if (!res_host || !w_host || !out_host) {
        free(res_host);
        free(w_host);
        free(out_host);
        return 0;
    }
    int ok = ds4_gpu_tensor_read(residual_hc, 0, res_host, rows * n_hc * (uint64_t)n_embd * sizeof(float)) &&
             ds4_gpu_tensor_read(weights, 0, w_host, ((rows - 1) * (uint64_t)weight_stride + n_hc) * sizeof(float));
    if (ok) {
        for (uint64_t r = 0; r < rows; r++) {
            const float *res = res_host + r * n_hc * (uint64_t)n_embd;
            const float *w = w_host + r * weight_stride;
            float *orow = out_host + r * n_embd;
            for (uint32_t d = 0; d < n_embd; d++) {
                float acc = 0.0f;
                for (uint32_t h = 0; h < n_hc; h++) acc += res[(uint64_t)h * n_embd + d] * w[h];
                orow[d] = acc;
            }
        }
        ok = ds4_gpu_tensor_write(out, 0, out_host, rows * (uint64_t)n_embd * sizeof(float));
    }
    free(res_host);
    free(w_host);
    free(out_host);
    return ok;
}

int ds4_gpu_hc_weighted_sum_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *weights, uint32_t n_embd, uint32_t n_hc) {
    return ascend_hc_weighted_sum_host(out, residual_hc, weights, n_embd, n_hc, n_hc);
}

int ds4_gpu_hc_weighted_sum_split_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    return ascend_hc_weighted_sum_host(out, residual_hc, split, n_embd, n_hc, ascend_hc_mix_count(n_hc));
}

int ds4_gpu_hc_split_weighted_sum_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *split, const ds4_gpu_tensor *mix, const ds4_gpu_tensor *residual_hc, const void *model_map, uint64_t model_size, uint64_t scale_offset, uint64_t base_offset, uint32_t n_embd, uint32_t n_hc, uint32_t sinkhorn_iters, float eps) {
    return ds4_gpu_hc_split_sinkhorn_tensor(split, mix, model_map, model_size, scale_offset, base_offset, n_hc, sinkhorn_iters, eps) &&
           ds4_gpu_hc_weighted_sum_split_tensor(out, residual_hc, split, n_embd, n_hc);
}

int ds4_gpu_hc_split_weighted_sum_norm_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *norm_out, ds4_gpu_tensor *split, const ds4_gpu_tensor *mix, const ds4_gpu_tensor *residual_hc, const void *model_map, uint64_t model_size, uint64_t scale_offset, uint64_t base_offset, uint64_t norm_weight_offset, uint32_t n_embd, uint32_t n_hc, uint32_t sinkhorn_iters, float eps, float norm_eps) {
    if (!ds4_gpu_hc_split_weighted_sum_tensor(out, split, mix, residual_hc, model_map, model_size, scale_offset, base_offset, n_embd, n_hc, sinkhorn_iters, eps)) return 0;
    return ds4_gpu_rms_norm_weight_rows_tensor(norm_out, out, model_map, model_size, norm_weight_offset, n_embd, (uint32_t)(out->bytes / ((uint64_t)n_embd * sizeof(float))), norm_eps);
}

int ds4_gpu_output_hc_weights_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *pre, const void *model_map, uint64_t model_size, uint64_t scale_offset, uint64_t base_offset, uint32_t n_hc, float eps) {
    if (!out || !pre || !model_map || n_hc == 0) return 0;
    const uint64_t row_bytes = (uint64_t)n_hc * sizeof(float);
    if (out->bytes < row_bytes || out->bytes % row_bytes != 0 || pre->bytes < out->bytes ||
        scale_offset > model_size || sizeof(float) > model_size - scale_offset ||
        base_offset > model_size || row_bytes > model_size - base_offset) return 0;
    const uint64_t n_tokens = out->bytes / row_bytes;
    const uint64_t count = n_tokens * n_hc;
    if (out->device == pre->device && n_tokens <= UINT32_MAX) {
        void *scale_dev = ascend_model_range_device_ptr(model_map, model_size, scale_offset, sizeof(float), out->device, "output_hc_scale");
        void *base_dev = ascend_model_range_device_ptr(model_map, model_size, base_offset, row_bytes, out->device, "output_hc_base");
        if (scale_dev && base_dev && ascend_set_context(out->device) && g_streams[out->device]) {
            ds4_ascend_launch_output_hc_weights(g_streams[out->device], out->ptr, pre->ptr, scale_dev, base_dev, n_hc, (uint32_t)n_tokens, eps);
            return 1;
        }
    }
    if (!ascend_host_fallback_begin("output_hc_weights", out->bytes, out->bytes)) return 0;
    float *pre_host = malloc((size_t)(count * sizeof(float)));
    float *out_host = malloc((size_t)(count * sizeof(float)));
    if (!pre_host || !out_host) {
        free(pre_host); free(out_host);
        return 0;
    }
    const float scale = *(const float *)((const uint8_t *)model_map + scale_offset);
    const float *base = (const float *)((const uint8_t *)model_map + base_offset);
    int ok = ds4_gpu_tensor_read(pre, 0, pre_host, count * sizeof(float));
    if (ok) {
        for (uint64_t i = 0; i < count; i++) {
            const uint32_t h = (uint32_t)(i % n_hc);
            out_host[i] = ascend_sigmoid(pre_host[i] * scale + base[h]) + eps;
        }
        ok = ds4_gpu_tensor_write(out, 0, out_host, count * sizeof(float));
    }
    free(pre_host); free(out_host);
    return ok;
}

static int ascend_hc_expand_split(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *block_add, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !block_out || !residual_hc || !split || n_embd == 0 || n_hc == 0) return 0;
    const uint32_t mix_hc = ascend_hc_mix_count(n_hc);
    const uint64_t rows = block_out->bytes / ((uint64_t)n_embd * sizeof(float));
    const uint64_t out_count = rows * n_hc * (uint64_t)n_embd;
    const uint64_t block_count = rows * (uint64_t)n_embd;
    const uint64_t split_count = rows * (uint64_t)mix_hc;
    if (rows == 0 || rows > UINT32_MAX || out_count > UINT32_MAX ||
        out_hc->bytes < out_count * sizeof(float) || residual_hc->bytes < out_count * sizeof(float) || split->bytes < split_count * sizeof(float) ||
        (block_add && block_add->bytes < block_count * sizeof(float)) || out_hc->device != block_out->device || out_hc->device != residual_hc->device ||
        out_hc->device != split->device || (block_add && out_hc->device != block_add->device)) return 0;
    if (!ascend_set_context(out_hc->device) || !g_streams[out_hc->device]) return 0;
    ds4_ascend_launch_hc_expand_split(g_streams[out_hc->device], out_hc->ptr, block_out->ptr, block_add ? block_add->ptr : NULL, residual_hc->ptr, split->ptr, n_embd, n_hc, (uint32_t)rows, block_add ? 1u : 0u);
    return 1;
}

int ds4_gpu_hc_expand_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *post, const ds4_gpu_tensor *comb, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !block_out || !residual_hc || !post || !comb || n_embd == 0 || n_hc == 0) return 0;
    const uint64_t rows = block_out->bytes / ((uint64_t)n_embd * sizeof(float));
    const uint64_t out_count = rows * n_hc * (uint64_t)n_embd;
    const uint64_t post_count = rows * (uint64_t)n_hc;
    const uint64_t comb_count = rows * (uint64_t)n_hc * n_hc;
    if (rows == 0 || rows > UINT32_MAX || out_count > UINT32_MAX ||
        block_out->bytes < rows * (uint64_t)n_embd * sizeof(float) ||
        residual_hc->bytes < out_count * sizeof(float) ||
        out_hc->bytes < out_count * sizeof(float) ||
        post->bytes < post_count * sizeof(float) ||
        comb->bytes < comb_count * sizeof(float) ||
        out_hc->device != block_out->device || out_hc->device != residual_hc->device ||
        out_hc->device != post->device || out_hc->device != comb->device) return 0;
    if (!ascend_set_context(out_hc->device) || !g_streams[out_hc->device]) return 0;
    ds4_ascend_launch_hc_expand(g_streams[out_hc->device], out_hc->ptr, block_out->ptr, residual_hc->ptr, post->ptr, comb->ptr, n_embd, n_hc, (uint32_t)rows);
    return 1;
}
int ds4_gpu_hc_expand_split_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) { return ascend_hc_expand_split(out_hc, block_out, NULL, residual_hc, split, n_embd, n_hc); }
int ds4_gpu_hc_expand_add_split_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *block_add, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) { return ascend_hc_expand_split(out_hc, block_out, block_add, residual_hc, split, n_embd, n_hc); }
int ds4_gpu_shared_down_hc_expand_q8_0_tensor(ds4_gpu_tensor *out_hc, ds4_gpu_tensor *shared_out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *shared_mid, const ds4_gpu_tensor *routed_out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !shared_out || !model_map || !shared_mid || !routed_out || !residual_hc || !split || n_embd == 0 || n_hc == 0 || out_dim != n_embd) return 0;
    if (shared_out->bytes < (uint64_t)n_embd * sizeof(float) || routed_out->bytes < (uint64_t)n_embd * sizeof(float) ||
        out_hc->device != shared_out->device || out_hc->device != shared_mid->device || out_hc->device != routed_out->device ||
        out_hc->device != residual_hc->device || out_hc->device != split->device) return 0;
    return ascend_matmul_q8_0_tensor(shared_out, model_map, model_size, weight_offset, in_dim, out_dim, shared_mid, 1) &&
           ascend_hc_expand_split(out_hc, routed_out, shared_out, residual_hc, split, n_embd, n_hc);
}
int ds4_gpu_matmul_q8_0_hc_expand_tensor(ds4_gpu_tensor *out_hc, ds4_gpu_tensor *block_out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !block_out || !model_map || !x || !residual_hc || !split || n_embd == 0 || n_hc == 0 || out_dim != n_embd) return 0;
    if (block_out->bytes < (uint64_t)n_embd * sizeof(float) || out_hc->device != block_out->device || out_hc->device != x->device || out_hc->device != residual_hc->device || out_hc->device != split->device) return 0;
    return ascend_matmul_q8_0_tensor(block_out, model_map, model_size, weight_offset, in_dim, out_dim, x, 1) &&
           ascend_hc_expand_split(out_hc, block_out, NULL, residual_hc, split, n_embd, n_hc);
}
