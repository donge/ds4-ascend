#include "kernel_operator.h"

#include <stdint.h>
#include <math.h>
#include <stdlib.h>

using namespace AscendC;

#define DS4_QK_K 256u
#define DS4_QK8_0 32u
#define DS4_BLOCK_Q8_0_BYTES 34u
#define DS4_N_EXPERT 256u
#define DS4_N_EXPERT_USED 6u
#define DS4_ASCEND_DEFAULT_KERNEL_PARTS 32u
#define DS4_ASCEND_MAX_KERNEL_PARTS 64u

static uint32_t ds4_ascend_parse_parts(const char *name, uint32_t fallback) {
    const char *env = getenv(name);
    if (!env || !env[0]) return fallback;
    char *end = nullptr;
    unsigned long v = strtoul(env, &end, 10);
    if (end == env || v == 0) v = fallback;
    if (v > DS4_ASCEND_MAX_KERNEL_PARTS) v = DS4_ASCEND_MAX_KERNEL_PARTS;
    return (uint32_t)v;
}

static uint32_t ds4_ascend_kernel_parts(void) {
    static uint32_t cached = 0;
    if (!cached) cached = ds4_ascend_parse_parts("DS4_ASCEND_KERNEL_PARTS", DS4_ASCEND_DEFAULT_KERNEL_PARTS);
    return cached;
}

static uint32_t ds4_ascend_matmul_q8_0_parts(void) {
    static uint32_t cached = 0;
    if (!cached) cached = ds4_ascend_parse_parts("DS4_ASCEND_MATMUL_Q8_0_PARTS", ds4_ascend_kernel_parts());
    return cached;
}

typedef struct {
    uint8_t scales[DS4_QK_K / 16];
    uint8_t qs[DS4_QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} ds4_block_q2_K;

typedef struct {
    float d;
    int8_t qs[DS4_QK_K];
    int16_t bsums[DS4_QK_K / 16];
} ds4_block_q8_K;

typedef struct {
    uint16_t d;
    uint16_t qs[DS4_QK_K / 8];
} ds4_block_iq2_xxs;

static __aicore__ inline float ds4_abs_f32(float x) {
    return x < 0.0f ? -x : x;
}

static __aicore__ inline int32_t ds4_round_f32_to_i32(float x) {
    const int32_t i = (int32_t)x;
    const float frac = x - (float)i;
    if (frac > 0.5f || (frac == 0.5f && (i & 1) != 0)) return i + 1;
    if (frac < -0.5f || (frac == -0.5f && (i & 1) != 0)) return i - 1;
    return i;
}

static __aicore__ inline float ds4_rsqrt_f32(float x) {
    union { float f; uint32_t u; } v;
    v.f = x;
    v.u = 0x5f3759dfu - (v.u >> 1);
    float y = v.f;
    const float half = 0.5f * x;
    y = y * (1.5f - half * y * y);
    y = y * (1.5f - half * y * y);
    y = y * (1.5f - half * y * y);
    return y;
}

static __aicore__ inline float ds4_exp_f32(float x) {
    if (x < -20.0f) return 0.0f;
    if (x > 20.0f) x = 20.0f;
    float y = 1.0f + x * 0.00048828125f;
    y *= y; y *= y; y *= y; y *= y;
    y *= y; y *= y; y *= y; y *= y;
    y *= y; y *= y; y *= y;
    return y;
}

static __aicore__ inline float ds4_sigmoid_f32(float x) {
    if (x >= 0.0f) {
        const float e = ds4_exp_f32(-x);
        return 1.0f / (1.0f + e);
    }
    const float e = ds4_exp_f32(x);
    return e / (1.0f + e);
}

static __aicore__ inline float ds4_sqrt_f32(float x) {
    if (x <= 0.0f) return 0.0f;
    return x * ds4_rsqrt_f32(x);
}

static __aicore__ inline float ds4_log_f32(float x) {
    if (x <= 0.0f) return -3.402823466e+38f;
    union { float f; uint32_t u; } v;
    v.f = x;
    const int32_t exp = (int32_t)((v.u >> 23) & 0xffu) - 127;
    v.u = (v.u & 0x007fffffu) | 0x3f800000u;
    const float m = v.f;
    const float z = (m - 1.0f) / (m + 1.0f);
    const float z2 = z * z;
    float term = z;
    float sum = term;
    term *= z2; sum += term * 0.3333333333333333f;
    term *= z2; sum += term * 0.2f;
    term *= z2; sum += term * 0.14285714285714285f;
    term *= z2; sum += term * 0.1111111111111111f;
    term *= z2; sum += term * 0.09090909090909091f;
    return 0.6931471805599453f * (float)exp + 2.0f * sum;
}

static __aicore__ inline float ds4_softplus_f32(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return ds4_exp_f32(x);
    return ds4_log_f32(1.0f + ds4_exp_f32(x));
}

static __aicore__ inline bool ds4_router_score_better(float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

static __aicore__ inline float ds4_pow2_ceil_scale_for_fp8(float amax) {
    if (amax < 1.0e-4f) amax = 1.0e-4f;
    union { float f; uint32_t u; } v;
    v.f = amax / 448.0f;
    int32_t exp = (int32_t)((v.u >> 23) & 0xffu) - 127;
    if ((v.u & 0x007fffffu) != 0) exp++;
    int32_t bexp = exp + 127;
    if (bexp < 1) bexp = 1;
    if (bexp > 254) bexp = 254;
    union { uint32_t u; float f; } out;
    out.u = (uint32_t)bexp << 23;
    return out.f;
}

static __aicore__ inline float ds4_e4m3fn_value_i32(int32_t code) {
    const int32_t exp = (code >> 3) & 15;
    const int32_t mant = code & 7;
    const float m = (float)mant;
    if (exp == 0) return m * 0.001953125f;
    float scale = 1.0f;
    if (exp == 1) scale = 0.015625f;
    else if (exp == 2) scale = 0.03125f;
    else if (exp == 3) scale = 0.0625f;
    else if (exp == 4) scale = 0.125f;
    else if (exp == 5) scale = 0.25f;
    else if (exp == 6) scale = 0.5f;
    else if (exp == 7) scale = 1.0f;
    else if (exp == 8) scale = 2.0f;
    else if (exp == 9) scale = 4.0f;
    else if (exp == 10) scale = 8.0f;
    else if (exp == 11) scale = 16.0f;
    else if (exp == 12) scale = 32.0f;
    else if (exp == 13) scale = 64.0f;
    else if (exp == 14) scale = 128.0f;
    else scale = 256.0f;
    return (1.0f + m * 0.125f) * scale;
}

static __aicore__ inline float ds4_e4m3fn_roundtrip(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = ds4_abs_f32(x);
    if (ax > 448.0f) ax = 448.0f;
    int32_t lo = 0;
    int32_t hi = 126;
    while (lo < hi) {
        const int32_t mid = (lo + hi + 1) >> 1;
        if (ds4_e4m3fn_value_i32(mid) <= ax) lo = mid;
        else hi = mid - 1;
    }
    int32_t best = lo;
    if (best < 126) {
        const float bd = ds4_abs_f32(ax - ds4_e4m3fn_value_i32(best));
        const float nd = ds4_abs_f32(ax - ds4_e4m3fn_value_i32(best + 1));
        if (nd < bd || (nd == bd && (((best + 1) & 1) == 0) && ((best & 1) != 0))) best++;
    }
    return sign * ds4_e4m3fn_value_i32(best);
}

static __aicore__ inline float ds4_f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
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
    union { uint32_t u; float f; } v;
    v.u = bits;
    return v.f;
}

static __aicore__ inline uint16_t ds4_f32_to_f16(float f) {
    union { float f; uint32_t u; } v;
    v.f = f;
    const uint32_t sign = (v.u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((v.u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = v.u & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        uint32_t hmant = mant >> shift;
        if ((mant >> (shift - 1u)) & 1u) hmant++;
        return (uint16_t)(sign | hmant);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint32_t h = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x00001000u) h++;
    return (uint16_t)h;
}

static __aicore__ inline float ds4_f16_round_f32(float f) {
    return ds4_f16_to_f32(ds4_f32_to_f16(f));
}

static __aicore__ inline int32_t ds4_dot_q2_16_gm(const __gm__ ds4_block_q2_K *x, const __gm__ ds4_block_q8_K *y, uint32_t block, uint32_t q2_off, uint32_t q8_off, int shift) {
    int32_t s = 0;
    for (uint32_t i = 0; i < 16; i++) s += (int32_t)((x[block].qs[q2_off + i] >> shift) & 3u) * (int32_t)y[block].qs[q8_off + i];
    return s;
}

static __aicore__ inline float ds4_dot_q2_K_q8_K(uint32_t n, const __gm__ ds4_block_q2_K *x, const __gm__ ds4_block_q8_K *y) {
    const uint32_t nb = n / DS4_QK_K;
    float sumf = 0.0f;
    for (uint32_t i = 0; i < nb; i++) {
        int summs = 0;
        for (uint32_t j = 0; j < 16; j++) summs += y[i].bsums[j] * (x[i].scales[j] >> 4);
        const float dall = y[i].d * ds4_f16_to_f32(x[i].d);
        const float dmin = y[i].d * ds4_f16_to_f32(x[i].dmin);
        int isum = 0;
        int is = 0;
        uint32_t q2_off = 0;
        uint32_t q8_off = 0;
        for (uint32_t k = 0; k < DS4_QK_K / 128; k++) {
            int shift = 0;
            for (uint32_t j = 0; j < 4; j++) {
                int d = x[i].scales[is++] & 0x0f;
                isum += d * ds4_dot_q2_16_gm(x, y, i, q2_off, q8_off, shift);
                d = x[i].scales[is++] & 0x0f;
                isum += d * ds4_dot_q2_16_gm(x, y, i, q2_off + 16u, q8_off + 16u, shift);
                shift += 2;
                q8_off += 32u;
            }
            q2_off += 32u;
        }
        sumf += dall * (float)isum - dmin * (float)summs;
    }
    return sumf;
}

static __aicore__ inline float ds4_dot_iq2_xxs_q8_K(uint32_t n, const __gm__ ds4_block_iq2_xxs *x, const __gm__ ds4_block_q8_K *y, const __gm__ uint8_t *ksigns, const __gm__ uint64_t *grid) {
    const uint32_t nb = n / DS4_QK_K;
    float sumf = 0.0f;
    for (uint32_t i = 0; i < nb; i++) {
        const float d = ds4_f16_to_f32(x[i].d) * y[i].d;
        int32_t bsum = 0;
        uint32_t q2_off = 0;
        uint32_t q8_off = 0;
        for (uint32_t ib32 = 0; ib32 < DS4_QK_K / 32; ib32++) {
            const uint32_t aux_g = (uint32_t)x[i].qs[q2_off] | ((uint32_t)x[i].qs[q2_off + 1u] << 16);
            const uint32_t aux_s = (uint32_t)x[i].qs[q2_off + 2u] | ((uint32_t)x[i].qs[q2_off + 3u] << 16);
            q2_off += 4u;
            const uint32_t ls = 2u * (aux_s >> 28) + 1u;
            int32_t sumi = 0;
            for (uint32_t l = 0; l < 4; l += 2) {
                const uint8_t grid0 = (uint8_t)((aux_g >> (8u * l)) & 0xffu);
                const uint8_t grid1 = (uint8_t)((aux_g >> (8u * (l + 1u))) & 0xffu);
                const uint32_t sign0 = (aux_s >> (7u * l)) & 127u;
                const uint32_t sign1 = (aux_s >> (7u * (l + 1u))) & 127u;
                const uint64_t g0 = grid[grid0];
                const uint64_t g1 = grid[grid1];
                const uint8_t s0 = ksigns[sign0];
                const uint8_t s1 = ksigns[sign1];
                for (uint32_t j = 0; j < 8; j++) {
                    int32_t w0 = (int32_t)((g0 >> (8u * j)) & 0xffu);
                    int32_t w1 = (int32_t)((g1 >> (8u * j)) & 0xffu);
                    if (s0 & (1u << j)) w0 = -w0;
                    if (s1 & (1u << j)) w1 = -w1;
                    sumi += w0 * (int32_t)y[i].qs[q8_off + j] + w1 * (int32_t)y[i].qs[q8_off + 8u + j];
                }
                q8_off += 16u;
            }
            bsum += sumi * (int32_t)ls;
        }
        sumf += d * (float)bsum;
    }
    return 0.125f * sumf;
}

extern "C" __global__ __aicore__ __attribute__((aiv)) void ds4_fill_f32(GM_ADDR out_gm, float value, uint32_t count) {
    GlobalTensor<float> out;
    out.SetGlobalBuffer((__gm__ float *)out_gm, count);
    for (uint32_t i = 0; i < count; i++) out.SetValue(i, value);
}

extern "C" __global__ __aicore__ __attribute__((aiv)) void ds4_matmul_f16(GM_ADDR out_gm, GM_ADDR w_gm, GM_ADDR x_gm, uint32_t in_dim, uint32_t out_dim, uint32_t n_tok, uint32_t start_gid, uint32_t stride) {
    GlobalTensor<float> out;
    GlobalTensor<float> x;
    GlobalTensor<uint16_t> w;
    out.SetGlobalBuffer((__gm__ float *)out_gm, n_tok * out_dim);
    x.SetGlobalBuffer((__gm__ float *)x_gm, n_tok * in_dim);
    w.SetGlobalBuffer((__gm__ uint16_t *)w_gm, out_dim * in_dim);

    const uint32_t total = n_tok * out_dim;
    if (stride == 0) stride = 1;
    for (uint32_t gid = start_gid; gid < total; gid += stride) {
        const uint32_t t = gid / out_dim;
        const uint32_t r = gid - t * out_dim;
        const uint32_t x_base = t * in_dim;
        const uint32_t w_base = r * in_dim;
        float sum = 0.0f;
        for (uint32_t c = 0; c < in_dim; c++) {
            sum += ds4_f16_to_f32(w.GetValue(w_base + c)) * x.GetValue(x_base + c);
        }
        out.SetValue(gid, sum);
    }
}

extern "C" __global__ __aicore__ __attribute__((aiv)) void ds4_embed_token_hc(GM_ADDR out_gm, GM_ADDR w_gm, uint32_t token, uint32_t n_embd, uint32_t n_hc, uint32_t start_gid, uint32_t stride) {
    GlobalTensor<float> out;
    GlobalTensor<uint16_t> w;
    out.SetGlobalBuffer((__gm__ float *)out_gm, n_hc * n_embd);
    w.SetGlobalBuffer((__gm__ uint16_t *)w_gm, (token + 1u) * n_embd);

    const uint32_t total = n_hc * n_embd;
    const uint32_t base = token * n_embd;
    if (stride == 0) stride = 1;
    for (uint32_t gid = start_gid; gid < total; gid += stride) {
        const uint32_t d = gid % n_embd;
        out.SetValue(gid, ds4_f16_to_f32(w.GetValue(base + d)));
    }
}

extern "C" __global__ __aicore__ __attribute__((aiv)) void ds4_embed_tokens_hc(GM_ADDR out_gm, GM_ADDR tokens_gm, GM_ADDR w_gm, uint32_t n_vocab, uint32_t n_tokens, uint32_t n_embd, uint32_t n_hc, uint32_t start_gid, uint32_t stride) {
    GlobalTensor<float> out;
    GlobalTensor<int32_t> tokens;
    GlobalTensor<uint16_t> w;
    out.SetGlobalBuffer((__gm__ float *)out_gm, n_tokens * n_hc * n_embd);
    tokens.SetGlobalBuffer((__gm__ int32_t *)tokens_gm, n_tokens);
    w.SetGlobalBuffer((__gm__ uint16_t *)w_gm, n_vocab * n_embd);

    const uint32_t total = n_tokens * n_hc * n_embd;
    if (stride == 0) stride = 1;
    for (uint32_t gid = start_gid; gid < total; gid += stride) {
        const uint32_t d = gid % n_embd;
        const uint32_t t = gid / (n_hc * n_embd);
        int32_t tok = tokens.GetValue(t);
        if (tok < 0 || tok >= (int32_t)n_vocab) tok = 0;
        out.SetValue(gid, ds4_f16_to_f32(w.GetValue((uint32_t)tok * n_embd + d)));
    }
}

extern "C" __global__ __aicore__ __attribute__((aiv)) void ds4_quantize_q8_0(GM_ADDR xq_gm, GM_ADDR xscale_gm, GM_ADDR x_gm, uint32_t in_dim, uint32_t n_tok, uint32_t start_block, uint32_t stride) {
    GlobalTensor<float> x;
    GlobalTensor<float> xscale;
    GlobalTensor<uint8_t> xq;
    x.SetGlobalBuffer((__gm__ float *)x_gm, n_tok * in_dim);
    const uint32_t blocks = (in_dim + DS4_QK8_0 - 1u) / DS4_QK8_0;
    xq.SetGlobalBuffer((__gm__ uint8_t *)xq_gm, n_tok * blocks * DS4_QK8_0);
    xscale.SetGlobalBuffer((__gm__ float *)xscale_gm, n_tok * blocks);

    const uint32_t total = n_tok * blocks;
    if (stride == 0) stride = 1;
    for (uint32_t gid = start_block; gid < total; gid += stride) {
        const uint32_t t = gid / blocks;
        const uint32_t b = gid - t * blocks;
        const uint32_t i0 = b * DS4_QK8_0;
        const uint32_t remain = in_dim - i0;
        const uint32_t bn = remain < DS4_QK8_0 ? remain : DS4_QK8_0;
        const uint32_t x_base = t * in_dim + i0;
        const uint32_t q_base = (t * blocks + b) * DS4_QK8_0;
        float amax = 0.0f;
        for (uint32_t i = 0; i < bn; i++) {
            const float av = ds4_abs_f32(x.GetValue(x_base + i));
            if (av > amax) amax = av;
        }
        const float xd = amax / 127.0f;
        const float xid = xd != 0.0f ? 1.0f / xd : 0.0f;
        xscale.SetValue(gid, xd);
        for (uint32_t i = 0; i < DS4_QK8_0; i++) {
            int32_t aq = 0;
            if (i < bn) {
                aq = ds4_round_f32_to_i32(x.GetValue(x_base + i) * xid);
                if (aq > 127) aq = 127;
                if (aq < -128) aq = -128;
            }
            xq.SetValue(q_base + i, (uint8_t)(int8_t)aq);
        }
    }
}

extern "C" __global__ __aicore__ __attribute__((aiv)) void ds4_matmul_q8_0_prequant(GM_ADDR out_gm, GM_ADDR w_gm, GM_ADDR xq_gm, GM_ADDR xscale_gm, uint32_t in_dim, uint32_t out_dim, uint32_t n_tok, uint32_t start_gid, uint32_t stride) {
    GlobalTensor<float> out;
    GlobalTensor<float> xscale;
    GlobalTensor<uint8_t> w;
    GlobalTensor<uint8_t> xq;
    out.SetGlobalBuffer((__gm__ float *)out_gm, n_tok * out_dim);
    const uint32_t blocks = (in_dim + DS4_QK8_0 - 1u) / DS4_QK8_0;
    const uint32_t row_bytes = blocks * DS4_BLOCK_Q8_0_BYTES;
    w.SetGlobalBuffer((__gm__ uint8_t *)w_gm, out_dim * row_bytes);
    xq.SetGlobalBuffer((__gm__ uint8_t *)xq_gm, n_tok * blocks * DS4_QK8_0);
    xscale.SetGlobalBuffer((__gm__ float *)xscale_gm, n_tok * blocks);

    const uint32_t total = n_tok * out_dim;
    if (stride == 0) stride = 1;
    for (uint32_t gid = start_gid; gid < total; gid += stride) {
        const uint32_t t = gid / out_dim;
        const uint32_t r = gid - t * out_dim;
        const uint32_t xq_base = t * blocks * DS4_QK8_0;
        const uint32_t scale_base = t * blocks;
        const uint32_t w_base = r * row_bytes;
        float acc = 0.0f;
        for (uint32_t b = 0; b < blocks; b++) {
            const uint32_t w_block_base = w_base + b * DS4_BLOCK_Q8_0_BYTES;
            const uint16_t wscale_bits = (uint16_t)w.GetValue(w_block_base) | ((uint16_t)w.GetValue(w_block_base + 1u) << 8);
            const float xd = xscale.GetValue(scale_base + b);
            const float wd = ds4_f16_to_f32(wscale_bits);
            int32_t dot = 0;
            for (uint32_t i = 0; i < DS4_QK8_0; i++) {
                dot += (int32_t)(int8_t)w.GetValue(w_block_base + 2u + i) * (int32_t)(int8_t)xq.GetValue(xq_base + b * DS4_QK8_0 + i);
            }
            acc += wd * xd * (float)dot;
        }
        out.SetValue(gid, acc);
    }
}

extern "C" __global__ __aicore__ __attribute__((aiv)) void ds4_matmul_q8_0(GM_ADDR out_gm, GM_ADDR w_gm, GM_ADDR x_gm, uint32_t in_dim, uint32_t out_dim, uint32_t n_tok, uint32_t start_gid, uint32_t stride) {
    GlobalTensor<float> out;
    GlobalTensor<float> x;
    GlobalTensor<uint8_t> w;
    out.SetGlobalBuffer((__gm__ float *)out_gm, n_tok * out_dim);
    x.SetGlobalBuffer((__gm__ float *)x_gm, n_tok * in_dim);
    const uint32_t blocks = (in_dim + DS4_QK8_0 - 1u) / DS4_QK8_0;
    const uint32_t row_bytes = blocks * DS4_BLOCK_Q8_0_BYTES;
    w.SetGlobalBuffer((__gm__ uint8_t *)w_gm, out_dim * row_bytes);

    const uint32_t total = n_tok * out_dim;
    if (stride == 0) stride = 1;
    for (uint32_t gid = start_gid; gid < total; gid += stride) {
        const uint32_t t = gid / out_dim;
        const uint32_t r = gid - t * out_dim;
        const uint32_t x_base = t * in_dim;
        const uint32_t w_base = r * row_bytes;
        float acc = 0.0f;
        for (uint32_t b = 0; b < blocks; b++) {
            const uint32_t i0 = b * DS4_QK8_0;
            const uint32_t remain = in_dim - i0;
            const uint32_t bn = remain < DS4_QK8_0 ? remain : DS4_QK8_0;
            float amax = 0.0f;
            for (uint32_t i = 0; i < bn; i++) {
                const float av = ds4_abs_f32(x.GetValue(x_base + i0 + i));
                if (av > amax) amax = av;
            }
            const float xd = amax / 127.0f;
            const float xid = xd != 0.0f ? 1.0f / xd : 0.0f;
            const uint32_t block_base = w_base + b * DS4_BLOCK_Q8_0_BYTES;
            const uint16_t wscale_bits = (uint16_t)w.GetValue(block_base) | ((uint16_t)w.GetValue(block_base + 1u) << 8);
            const float wd = ds4_f16_to_f32(wscale_bits);
            int32_t dot = 0;
            for (uint32_t i = 0; i < bn; i++) {
                int32_t aq = ds4_round_f32_to_i32(x.GetValue(x_base + i0 + i) * xid);
                if (aq > 127) aq = 127;
                if (aq < -128) aq = -128;
                dot += (int32_t)(int8_t)w.GetValue(block_base + 2u + i) * aq;
            }
            acc += wd * xd * (float)dot;
        }
        out.SetValue(gid, acc);
    }
}

extern "C" __global__ __aicore__ __attribute__((aiv)) void ds4_rms_norm_plain(GM_ADDR out_gm, GM_ADDR x_gm, uint32_t n, uint32_t rows, float inv_n, float eps, uint32_t start_row, uint32_t stride) {
    GlobalTensor<float> out;
    GlobalTensor<float> x;
    out.SetGlobalBuffer((__gm__ float *)out_gm, n * rows);
    x.SetGlobalBuffer((__gm__ float *)x_gm, n * rows);

    if (stride == 0) stride = 1;
    for (uint32_t r = start_row; r < rows; r += stride) {
        const uint32_t base = r * n;
        float ss = 0.0f;
        for (uint32_t i = 0; i < n; i++) {
            const float v = x.GetValue(base + i);
            ss += v * v;
        }
        const float scale = ds4_rsqrt_f32(ss * inv_n + eps);
        for (uint32_t i = 0; i < n; i++) out.SetValue(base + i, x.GetValue(base + i) * scale);
    }
}

extern "C" __global__ __aicore__ __attribute__((aiv)) void ds4_rms_norm_weight(GM_ADDR out_gm, GM_ADDR x_gm, GM_ADDR weight_gm, uint32_t n, uint32_t rows, float inv_n, float eps, uint32_t start_row, uint32_t stride) {
    GlobalTensor<float> out;
    GlobalTensor<float> x;
    GlobalTensor<float> weight;
    out.SetGlobalBuffer((__gm__ float *)out_gm, n * rows);
    x.SetGlobalBuffer((__gm__ float *)x_gm, n * rows);
    weight.SetGlobalBuffer((__gm__ float *)weight_gm, n);

    if (stride == 0) stride = 1;
    for (uint32_t r = start_row; r < rows; r += stride) {
        const uint32_t base = r * n;
        float ss = 0.0f;
        for (uint32_t i = 0; i < n; i++) {
            const float v = x.GetValue(base + i);
            ss += v * v;
        }
        const float scale = ds4_rsqrt_f32(ss * inv_n + eps);
        for (uint32_t i = 0; i < n; i++) out.SetValue(base + i, x.GetValue(base + i) * scale * weight.GetValue(i));
    }
}

extern "C" __global__ __aicore__ __attribute__((aiv)) void ds4_rms_norm_inplace(GM_ADDR x_gm, uint32_t n, uint32_t rows, float inv_n, float eps, uint32_t start_row, uint32_t stride) {
    GlobalTensor<float> x;
    x.SetGlobalBuffer((__gm__ float *)x_gm, n * rows);

    if (stride == 0) stride = 1;
    for (uint32_t r = start_row; r < rows; r += stride) {
        const uint32_t base = r * n;
        float ss = 0.0f;
        for (uint32_t i = 0; i < n; i++) {
            const float v = x.GetValue(base + i);
            ss += v * v;
        }
        const float scale = ds4_rsqrt_f32(ss * inv_n + eps);
        for (uint32_t i = 0; i < n; i++) x.SetValue(base + i, x.GetValue(base + i) * scale);
    }
}

extern "C" __global__ __aicore__ __attribute__((aiv)) void ds4_hc_weighted_sum4(GM_ADDR out_gm, GM_ADDR residual_gm, GM_ADDR weights_gm, uint32_t n_embd, uint32_t rows, uint32_t weight_stride, uint32_t start_gid, uint32_t stride) {
    GlobalTensor<float> out;
    GlobalTensor<float> residual;
    GlobalTensor<float> weights;
    out.SetGlobalBuffer((__gm__ float *)out_gm, rows * n_embd);
    residual.SetGlobalBuffer((__gm__ float *)residual_gm, rows * 4u * n_embd);
    weights.SetGlobalBuffer((__gm__ float *)weights_gm, rows * weight_stride);

    const uint32_t total = rows * n_embd;
    if (stride == 0) stride = 1;
    for (uint32_t gid = start_gid; gid < total; gid += stride) {
        const uint32_t r = gid / n_embd;
        const uint32_t d = gid - r * n_embd;
        const uint32_t res_base = r * 4u * n_embd + d;
        const uint32_t w_base = r * weight_stride;
        const float acc = residual.GetValue(res_base) * weights.GetValue(w_base) +
                          residual.GetValue(res_base + n_embd) * weights.GetValue(w_base + 1u) +
                          residual.GetValue(res_base + 2u * n_embd) * weights.GetValue(w_base + 2u) +
                          residual.GetValue(res_base + 3u * n_embd) * weights.GetValue(w_base + 3u);
        out.SetValue(gid, acc);
    }
}

extern "C" __global__ __aicore__ __attribute__((aiv)) void ds4_output_hc_weights(GM_ADDR out_gm, GM_ADDR pre_gm, GM_ADDR scale_gm, GM_ADDR base_gm, uint32_t n_hc, uint32_t n_tokens, float eps, uint32_t start_gid, uint32_t stride) {
    GlobalTensor<float> out;
    GlobalTensor<float> pre;
    GlobalTensor<float> scale;
    GlobalTensor<float> base;
    out.SetGlobalBuffer((__gm__ float *)out_gm, n_tokens * n_hc);
    pre.SetGlobalBuffer((__gm__ float *)pre_gm, n_tokens * n_hc);
    scale.SetGlobalBuffer((__gm__ float *)scale_gm, 1u);
    base.SetGlobalBuffer((__gm__ float *)base_gm, n_hc);
    const float s = scale.GetValue(0);
    const uint32_t total = n_tokens * n_hc;
    if (stride == 0) stride = 1;
    for (uint32_t gid = start_gid; gid < total; gid += stride) {
        const uint32_t h = gid % n_hc;
        out.SetValue(gid, ds4_sigmoid_f32(pre.GetValue(gid) * s + base.GetValue(h)) + eps);
    }
}

extern "C" __global__ __aicore__ __attribute__((aiv)) void ds4_router_select(GM_ADDR selected_gm, GM_ADDR weights_gm, GM_ADDR probs_gm, GM_ADDR bias_gm, GM_ADDR hash_gm, GM_ADDR logits_gm, GM_ADDR tokens_gm, int32_t token_scalar, uint32_t hash_rows, uint32_t n_tokens, uint32_t has_bias, uint32_t hash_mode, uint32_t start_row, uint32_t stride) {
    GlobalTensor<int32_t> selected;
    GlobalTensor<float> weights;
    GlobalTensor<float> probs;
    GlobalTensor<float> bias;
    GlobalTensor<int32_t> hash;
    GlobalTensor<float> logits;
    GlobalTensor<int32_t> tokens;
    selected.SetGlobalBuffer((__gm__ int32_t *)selected_gm, n_tokens * DS4_N_EXPERT_USED);
    weights.SetGlobalBuffer((__gm__ float *)weights_gm, n_tokens * DS4_N_EXPERT_USED);
    probs.SetGlobalBuffer((__gm__ float *)probs_gm, n_tokens * DS4_N_EXPERT);
    if (has_bias) bias.SetGlobalBuffer((__gm__ float *)bias_gm, DS4_N_EXPERT);
    if (hash_mode) hash.SetGlobalBuffer((__gm__ int32_t *)hash_gm, hash_rows * DS4_N_EXPERT_USED);
    logits.SetGlobalBuffer((__gm__ float *)logits_gm, n_tokens * DS4_N_EXPERT);
    if (tokens_gm) tokens.SetGlobalBuffer((__gm__ int32_t *)tokens_gm, n_tokens);

    if (stride == 0) stride = 1;
    for (uint32_t t = start_row; t < n_tokens; t += stride) {
        float prob_local[DS4_N_EXPERT];
        const uint32_t prob_base = t * DS4_N_EXPERT;
        for (uint32_t e = 0; e < DS4_N_EXPERT; e++) {
            const float p = ds4_sqrt_f32(ds4_softplus_f32(logits.GetValue(prob_base + e)));
            prob_local[e] = p;
            probs.SetValue(prob_base + e, p);
        }

        int32_t sel[DS4_N_EXPERT_USED];
        if (hash_mode) {
            int32_t tok = tokens_gm ? tokens.GetValue(t) : token_scalar;
            if (tok < 0 || (uint32_t)tok >= hash_rows) tok = 0;
            const uint32_t hash_base = (uint32_t)tok * DS4_N_EXPERT_USED;
            for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) sel[i] = hash.GetValue(hash_base + i);
        } else {
            float scores[DS4_N_EXPERT_USED];
            for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
                sel[i] = -1;
                scores[i] = -3.402823466e+38f;
            }
            for (uint32_t e = 0; e < DS4_N_EXPERT; e++) {
                const float score = prob_local[e] + (has_bias ? bias.GetValue(e) : 0.0f);
                for (uint32_t j = 0; j < DS4_N_EXPERT_USED; j++) {
                    const uint32_t cur = sel[j] >= 0 ? (uint32_t)sel[j] : UINT32_MAX;
                    if (sel[j] < 0 || ds4_router_score_better(score, e, scores[j], cur)) {
                        for (uint32_t k = DS4_N_EXPERT_USED - 1u; k > j; k--) {
                            sel[k] = sel[k - 1u];
                            scores[k] = scores[k - 1u];
                        }
                        sel[j] = (int32_t)e;
                        scores[j] = score;
                        break;
                    }
                }
            }
        }

        float sum = 0.0f;
        float w[DS4_N_EXPERT_USED];
        for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
            const int32_t e = sel[i];
            const float v = (e >= 0 && (uint32_t)e < DS4_N_EXPERT) ? prob_local[(uint32_t)e] : 0.0f;
            w[i] = v;
            sum += v;
        }
        if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
        const uint32_t out_base = t * DS4_N_EXPERT_USED;
        for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
            selected.SetValue(out_base + i, sel[i]);
            weights.SetValue(out_base + i, w[i] / sum * 1.5f);
        }
    }
}

extern "C" __global__ __aicore__ __attribute__((aiv)) void ds4_hc_expand_split(GM_ADDR out_gm, GM_ADDR block_out_gm, GM_ADDR block_add_gm, GM_ADDR residual_gm, GM_ADDR split_gm, uint32_t n_embd, uint32_t n_hc, uint32_t rows, uint32_t has_add, uint32_t start_gid, uint32_t stride) {
    GlobalTensor<float> out;
    GlobalTensor<float> block_out;
    GlobalTensor<float> block_add;
    GlobalTensor<float> residual;
    GlobalTensor<float> split;
    const uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
    out.SetGlobalBuffer((__gm__ float *)out_gm, rows * n_hc * n_embd);
    block_out.SetGlobalBuffer((__gm__ float *)block_out_gm, rows * n_embd);
    if (has_add) block_add.SetGlobalBuffer((__gm__ float *)block_add_gm, rows * n_embd);
    residual.SetGlobalBuffer((__gm__ float *)residual_gm, rows * n_hc * n_embd);
    split.SetGlobalBuffer((__gm__ float *)split_gm, rows * mix_hc);

    const uint32_t total = rows * n_hc * n_embd;
    if (stride == 0) stride = 1;
    for (uint32_t gid = start_gid; gid < total; gid += stride) {
        const uint32_t d = gid % n_embd;
        const uint32_t tmp = gid / n_embd;
        const uint32_t dst = tmp % n_hc;
        const uint32_t r = tmp / n_hc;
        const uint32_t block_idx = r * n_embd + d;
        const uint32_t split_base = r * mix_hc;
        float block_v = block_out.GetValue(block_idx);
        if (has_add) block_v += block_add.GetValue(block_idx);
        float acc = block_v * split.GetValue(split_base + n_hc + dst);
        const uint32_t res_base = r * n_hc * n_embd + d;
        const uint32_t comb_base = split_base + 2u * n_hc + dst;
        for (uint32_t src = 0; src < n_hc; src++) {
            acc += split.GetValue(comb_base + src * n_hc) * residual.GetValue(res_base + src * n_embd);
        }
        out.SetValue(gid, acc);
    }
}

extern "C" __global__ __aicore__ __attribute__((aiv)) void ds4_hc_expand(GM_ADDR out_gm, GM_ADDR block_out_gm, GM_ADDR residual_gm, GM_ADDR post_gm, GM_ADDR comb_gm, uint32_t n_embd, uint32_t n_hc, uint32_t rows, uint32_t start_gid, uint32_t stride) {
    GlobalTensor<float> out;
    GlobalTensor<float> block_out;
    GlobalTensor<float> residual;
    GlobalTensor<float> post;
    GlobalTensor<float> comb;
    out.SetGlobalBuffer((__gm__ float *)out_gm, rows * n_hc * n_embd);
    block_out.SetGlobalBuffer((__gm__ float *)block_out_gm, rows * n_embd);
    residual.SetGlobalBuffer((__gm__ float *)residual_gm, rows * n_hc * n_embd);
    post.SetGlobalBuffer((__gm__ float *)post_gm, rows * n_hc);
    comb.SetGlobalBuffer((__gm__ float *)comb_gm, rows * n_hc * n_hc);

    const uint32_t total = rows * n_hc * n_embd;
    if (stride == 0) stride = 1;
    for (uint32_t gid = start_gid; gid < total; gid += stride) {
        const uint32_t d = gid % n_embd;
        const uint32_t tmp = gid / n_embd;
        const uint32_t dst = tmp % n_hc;
        const uint32_t r = tmp / n_hc;
        const uint32_t block_idx = r * n_embd + d;
        float acc = block_out.GetValue(block_idx) * post.GetValue(r * n_hc + dst);
        const uint32_t res_base = r * n_hc * n_embd + d;
        const uint32_t comb_base = r * n_hc * n_hc + dst;
        for (uint32_t src = 0; src < n_hc; src++) {
            acc += comb.GetValue(comb_base + src * n_hc) * residual.GetValue(res_base + src * n_embd);
        }
        out.SetValue(gid, acc);
    }
}

extern "C" __global__ __aicore__ __attribute__((aiv)) void ds4_hc_split_sinkhorn4(GM_ADDR out_gm, GM_ADDR mix_gm, GM_ADDR scale_gm, GM_ADDR base_gm, uint32_t rows, uint32_t sinkhorn_iters, float eps, uint32_t start_row, uint32_t stride) {
    GlobalTensor<float> out;
    GlobalTensor<float> mix;
    GlobalTensor<float> scale;
    GlobalTensor<float> base;
    out.SetGlobalBuffer((__gm__ float *)out_gm, rows * 24u);
    mix.SetGlobalBuffer((__gm__ float *)mix_gm, rows * 24u);
    scale.SetGlobalBuffer((__gm__ float *)scale_gm, 3u);
    base.SetGlobalBuffer((__gm__ float *)base_gm, 24u);

    const float pre_scale = scale.GetValue(0);
    const float post_scale = scale.GetValue(1);
    const float comb_scale = scale.GetValue(2);
    if (stride == 0) stride = 1;
    for (uint32_t r = start_row; r < rows; r += stride) {
        const uint32_t row = r * 24u;
        for (uint32_t i = 0; i < 4u; i++) {
            const float z = mix.GetValue(row + i) * pre_scale + base.GetValue(i);
            out.SetValue(row + i, ds4_sigmoid_f32(z) + eps);
        }
        for (uint32_t i = 0; i < 4u; i++) {
            const uint32_t off = 4u + i;
            const float z = mix.GetValue(row + off) * post_scale + base.GetValue(off);
            out.SetValue(row + off, 2.0f * ds4_sigmoid_f32(z));
        }

        float c[16];
        for (uint32_t dst = 0; dst < 4u; dst++) {
            float row_max = -3.402823466e+38f;
            for (uint32_t src = 0; src < 4u; src++) {
                const uint32_t idx = src + dst * 4u;
                const uint32_t off = 8u + idx;
                const float v = mix.GetValue(row + off) * comb_scale + base.GetValue(off);
                c[idx] = v;
                if (v > row_max) row_max = v;
            }
            float row_sum = 0.0f;
            for (uint32_t src = 0; src < 4u; src++) {
                const uint32_t idx = src + dst * 4u;
                const float v = ds4_exp_f32(c[idx] - row_max);
                c[idx] = v;
                row_sum += v;
            }
            const float inv = 1.0f / row_sum;
            for (uint32_t src = 0; src < 4u; src++) {
                const uint32_t idx = src + dst * 4u;
                c[idx] = c[idx] * inv + eps;
            }
        }

        for (uint32_t src = 0; src < 4u; src++) {
            float sum = 0.0f;
            for (uint32_t dst = 0; dst < 4u; dst++) sum += c[src + dst * 4u];
            const float inv = 1.0f / (sum + eps);
            for (uint32_t dst = 0; dst < 4u; dst++) c[src + dst * 4u] *= inv;
        }

        for (uint32_t iter = 1; iter < sinkhorn_iters; iter++) {
            for (uint32_t dst = 0; dst < 4u; dst++) {
                float sum = 0.0f;
                for (uint32_t src = 0; src < 4u; src++) sum += c[src + dst * 4u];
                const float inv = 1.0f / (sum + eps);
                for (uint32_t src = 0; src < 4u; src++) c[src + dst * 4u] *= inv;
            }
            for (uint32_t src = 0; src < 4u; src++) {
                float sum = 0.0f;
                for (uint32_t dst = 0; dst < 4u; dst++) sum += c[src + dst * 4u];
                const float inv = 1.0f / (sum + eps);
                for (uint32_t dst = 0; dst < 4u; dst++) c[src + dst * 4u] *= inv;
            }
        }

        for (uint32_t i = 0; i < 16u; i++) out.SetValue(row + 8u + i, c[i]);
    }
}

extern "C" __global__ __aicore__ __attribute__((aiv)) void ds4_fp8_kv_quantize(GM_ADDR x_gm, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot, uint32_t start_block, uint32_t stride) {
    GlobalTensor<float> x;
    x.SetGlobalBuffer((__gm__ float *)x_gm, n_tok * head_dim);

    const uint32_t n_nope = head_dim - n_rot;
    const uint32_t blocks_per_row = (n_nope + 63u) / 64u;
    const uint32_t total_blocks = n_tok * blocks_per_row;
    if (stride == 0) stride = 1;
    for (uint32_t gid = start_block; gid < total_blocks; gid += stride) {
        const uint32_t t = gid / blocks_per_row;
        const uint32_t b = gid - t * blocks_per_row;
        const uint32_t off = b * 64u;
        const uint32_t block = n_nope - off < 64u ? n_nope - off : 64u;
        const uint32_t base = t * head_dim + off;
        float amax = 0.0f;
        for (uint32_t i = 0; i < block; i++) {
            const float av = ds4_abs_f32(x.GetValue(base + i));
            if (av > amax) amax = av;
        }
        const float scale = ds4_pow2_ceil_scale_for_fp8(amax);
        for (uint32_t i = 0; i < block; i++) {
            float v = x.GetValue(base + i) / scale;
            if (v > 448.0f) v = 448.0f;
            if (v < -448.0f) v = -448.0f;
            x.SetValue(base + i, ds4_e4m3fn_roundtrip(v) * scale);
        }
    }
}

extern "C" __global__ __aicore__ __attribute__((aiv)) void ds4_rope_tail_table(GM_ADDR x_gm, GM_ADDR table_gm, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t start_gid, uint32_t stride) {
    GlobalTensor<float> x;
    GlobalTensor<float> table;
    x.SetGlobalBuffer((__gm__ float *)x_gm, n_tok * n_head * head_dim);
    table.SetGlobalBuffer((__gm__ float *)table_gm, n_tok * n_rot);

    const uint32_t half_rot = n_rot / 2u;
    const uint32_t n_nope = head_dim - n_rot;
    const uint32_t total = n_tok * n_head * half_rot;
    if (stride == 0) stride = 1;
    for (uint32_t gid = start_gid; gid < total; gid += stride) {
        const uint32_t pair = gid % half_rot;
        const uint32_t tmp = gid / half_rot;
        const uint32_t h = tmp % n_head;
        const uint32_t t = tmp / n_head;
        const uint32_t table_base = (t * half_rot + pair) * 2u;
        const float c = table.GetValue(table_base);
        const float s = table.GetValue(table_base + 1u);
        const uint32_t x_base = (t * n_head + h) * head_dim + n_nope + pair * 2u;
        const float x0 = x.GetValue(x_base);
        const float x1 = x.GetValue(x_base + 1u);
        x.SetValue(x_base, x0 * c - x1 * s);
        x.SetValue(x_base + 1u, x0 * s + x1 * c);
    }
}

extern "C" __global__ __aicore__ __attribute__((aiv)) void ds4_store_raw_kv_batch(GM_ADDR raw_gm, GM_ADDR kv_gm, uint32_t raw_cap, uint32_t pos0, uint32_t n_tokens, uint32_t head_dim, uint32_t start_gid, uint32_t stride) {
    GlobalTensor<float> raw;
    GlobalTensor<float> kv;
    raw.SetGlobalBuffer((__gm__ float *)raw_gm, raw_cap * head_dim);
    kv.SetGlobalBuffer((__gm__ float *)kv_gm, n_tokens * head_dim);

    const uint32_t total = n_tokens * head_dim;
    if (stride == 0) stride = 1;
    for (uint32_t gid = start_gid; gid < total; gid += stride) {
        const uint32_t d = gid % head_dim;
        const uint32_t t = gid / head_dim;
        const uint32_t row = (pos0 + t) % raw_cap;
        raw.SetValue(row * head_dim + d, ds4_f16_round_f32(kv.GetValue(gid)));
    }
}

extern "C" __global__ __aicore__ __attribute__((aiv)) void ds4_attention_prefill_raw(GM_ADDR heads_gm, GM_ADDR sinks_gm, GM_ADDR q_gm, GM_ADDR raw_kv_gm, uint32_t n_tokens, uint32_t window, uint32_t n_head, uint32_t head_dim, float scale, uint32_t start_gid, uint32_t stride) {
    GlobalTensor<float> heads;
    GlobalTensor<float> sinks;
    GlobalTensor<float> q;
    GlobalTensor<float> raw_kv;
    heads.SetGlobalBuffer((__gm__ float *)heads_gm, n_tokens * n_head * head_dim);
    sinks.SetGlobalBuffer((__gm__ float *)sinks_gm, n_head);
    q.SetGlobalBuffer((__gm__ float *)q_gm, n_tokens * n_head * head_dim);
    raw_kv.SetGlobalBuffer((__gm__ float *)raw_kv_gm, n_tokens * head_dim);

    float scores[512];
    const uint32_t total = n_tokens * n_head;
    if (stride == 0) stride = 1;
    for (uint32_t gid = start_gid; gid < total; gid += stride) {
        const uint32_t h = gid % n_head;
        const uint32_t t = gid / n_head;
        const uint32_t raw_start = (window != 0 && t + 1u > window) ? t + 1u - window : 0u;
        const uint32_t raw_count = t + 1u - raw_start;
        const uint32_t q_base = (t * n_head + h) * head_dim;
        float max_s = sinks.GetValue(h);
        for (uint32_t r = 0; r < raw_count; r++) {
            const uint32_t kv_base = (raw_start + r) * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += q.GetValue(q_base + d) * raw_kv.GetValue(kv_base + d);
            const float s = dot * scale;
            scores[r] = s;
            if (s > max_s) max_s = s;
        }
        float denom = ds4_exp_f32(sinks.GetValue(h) - max_s);
        for (uint32_t r = 0; r < raw_count; r++) {
            scores[r] = ds4_exp_f32(scores[r] - max_s);
            denom += scores[r];
        }
        const float inv_denom = 1.0f / denom;
        for (uint32_t d = 0; d < head_dim; d++) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv.GetValue((raw_start + r) * head_dim + d) * scores[r];
            heads.SetValue(q_base + d, acc * inv_denom);
        }
    }
}

extern "C" __global__ __aicore__ __attribute__((aiv)) void ds4_attention_decode(GM_ADDR heads_gm, GM_ADDR sinks_gm, GM_ADDR q_gm, GM_ADDR raw_kv_gm, GM_ADDR comp_kv_gm, GM_ADDR comp_mask_gm, uint32_t use_comp_mask, uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start, uint32_t n_comp, uint32_t n_head, uint32_t head_dim, float scale, uint32_t start_gid, uint32_t stride) {
    GlobalTensor<float> heads;
    GlobalTensor<float> sinks;
    GlobalTensor<float> q;
    GlobalTensor<float> raw_kv;
    GlobalTensor<float> comp_kv;
    GlobalTensor<float> comp_mask;
    heads.SetGlobalBuffer((__gm__ float *)heads_gm, n_head * head_dim);
    sinks.SetGlobalBuffer((__gm__ float *)sinks_gm, n_head);
    q.SetGlobalBuffer((__gm__ float *)q_gm, n_head * head_dim);
    raw_kv.SetGlobalBuffer((__gm__ float *)raw_kv_gm, raw_cap * head_dim);
    comp_kv.SetGlobalBuffer((__gm__ float *)comp_kv_gm, n_comp * head_dim);
    comp_mask.SetGlobalBuffer((__gm__ float *)comp_mask_gm, n_comp);

    float scores[1024];
    if (stride == 0) stride = 1;
    for (uint32_t h = start_gid; h < n_head; h += stride) {
        const uint32_t q_base = h * head_dim;
        float max_s = sinks.GetValue(h);
        for (uint32_t r = 0; r < n_raw; r++) {
            const uint32_t raw_idx = (raw_start + r) % raw_cap;
            const uint32_t kv_base = raw_idx * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += q.GetValue(q_base + d) * raw_kv.GetValue(kv_base + d);
            const float s = dot * scale;
            scores[r] = s;
            if (s > max_s) max_s = s;
        }
        for (uint32_t c = 0; c < n_comp; c++) {
            float s = -3.402823466e+38f;
            const float add = use_comp_mask ? comp_mask.GetValue(c) : 0.0f;
            if (add > -1.0e20f) {
                const uint32_t kv_base = c * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) dot += q.GetValue(q_base + d) * comp_kv.GetValue(kv_base + d);
                s = dot * scale + add;
            }
            scores[n_raw + c] = s;
            if (s > max_s) max_s = s;
        }
        float denom = ds4_exp_f32(sinks.GetValue(h) - max_s);
        for (uint32_t i = 0; i < n_raw + n_comp; i++) {
            scores[i] = ds4_exp_f32(scores[i] - max_s);
            denom += scores[i];
        }
        const float inv_denom = denom != 0.0f ? 1.0f / denom : 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < n_raw; r++) acc += raw_kv.GetValue(((raw_start + r) % raw_cap) * head_dim + d) * scores[r];
            for (uint32_t c = 0; c < n_comp; c++) acc += comp_kv.GetValue(c * head_dim + d) * scores[n_raw + c];
            heads.SetValue(q_base + d, acc * inv_denom);
        }
    }
}

extern "C" __global__ __aicore__ __attribute__((aiv)) void ds4_attention_prefill_mixed(GM_ADDR heads_gm, GM_ADDR sinks_gm, GM_ADDR q_gm, GM_ADDR raw_kv_gm, GM_ADDR comp_kv_gm, GM_ADDR comp_mask_gm, uint32_t use_comp_mask, uint32_t n_tokens, uint32_t n_comp, uint32_t window, uint32_t ratio, uint32_t n_head, uint32_t head_dim, float scale, uint32_t start_gid, uint32_t stride) {
    GlobalTensor<float> heads;
    GlobalTensor<float> sinks;
    GlobalTensor<float> q;
    GlobalTensor<float> raw_kv;
    GlobalTensor<float> comp_kv;
    GlobalTensor<float> comp_mask;
    heads.SetGlobalBuffer((__gm__ float *)heads_gm, n_tokens * n_head * head_dim);
    sinks.SetGlobalBuffer((__gm__ float *)sinks_gm, n_head);
    q.SetGlobalBuffer((__gm__ float *)q_gm, n_tokens * n_head * head_dim);
    raw_kv.SetGlobalBuffer((__gm__ float *)raw_kv_gm, n_tokens * head_dim);
    comp_kv.SetGlobalBuffer((__gm__ float *)comp_kv_gm, n_comp * head_dim);
    comp_mask.SetGlobalBuffer((__gm__ float *)comp_mask_gm, n_tokens * n_comp);

    float scores[1024];
    const uint32_t total = n_tokens * n_head;
    if (stride == 0) stride = 1;
    for (uint32_t gid = start_gid; gid < total; gid += stride) {
        const uint32_t h = gid % n_head;
        const uint32_t t = gid / n_head;
        const uint32_t raw_start = (window != 0 && t + 1u > window) ? t + 1u - window : 0u;
        const uint32_t raw_count = t + 1u - raw_start;
        uint32_t visible_comp = n_comp != 0 ? (t + 1u) / ratio : 0u;
        if (visible_comp > n_comp) visible_comp = n_comp;
        const uint32_t q_base = (t * n_head + h) * head_dim;
        float max_s = sinks.GetValue(h);
        for (uint32_t r = 0; r < raw_count; r++) {
            const uint32_t kv_base = (raw_start + r) * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += q.GetValue(q_base + d) * raw_kv.GetValue(kv_base + d);
            const float s = dot * scale;
            scores[r] = s;
            if (s > max_s) max_s = s;
        }
        for (uint32_t c = 0; c < visible_comp; c++) {
            float s = -3.402823466e+38f;
            const float add = use_comp_mask ? comp_mask.GetValue(t * n_comp + c) : 0.0f;
            if (add > -1.0e20f) {
                const uint32_t kv_base = c * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) dot += q.GetValue(q_base + d) * comp_kv.GetValue(kv_base + d);
                s = dot * scale + add;
            }
            scores[raw_count + c] = s;
            if (s > max_s) max_s = s;
        }
        float denom = ds4_exp_f32(sinks.GetValue(h) - max_s);
        for (uint32_t r = 0; r < raw_count + visible_comp; r++) {
            scores[r] = ds4_exp_f32(scores[r] - max_s);
            denom += scores[r];
        }
        const float inv_denom = denom != 0.0f ? 1.0f / denom : 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv.GetValue((raw_start + r) * head_dim + d) * scores[r];
            for (uint32_t c = 0; c < visible_comp; c++) acc += comp_kv.GetValue(c * head_dim + d) * scores[raw_count + c];
            heads.SetValue(q_base + d, acc * inv_denom);
        }
    }
}

extern "C" __global__ __aicore__ __attribute__((aiv)) void ds4_attention_output_low_q8(GM_ADDR low_gm, GM_ADDR w_gm, GM_ADDR heads_gm, uint32_t group_dim, uint32_t rank, uint32_t n_groups, uint32_t n_tokens, uint32_t start_gid, uint32_t stride) {
    GlobalTensor<float> low;
    GlobalTensor<float> heads;
    GlobalTensor<uint8_t> w;
    const uint32_t blocks = (group_dim + DS4_QK8_0 - 1u) / DS4_QK8_0;
    const uint32_t row_bytes = blocks * DS4_BLOCK_Q8_0_BYTES;
    const uint32_t low_dim = n_groups * rank;
    low.SetGlobalBuffer((__gm__ float *)low_gm, n_tokens * low_dim);
    heads.SetGlobalBuffer((__gm__ float *)heads_gm, n_tokens * n_groups * group_dim);
    w.SetGlobalBuffer((__gm__ uint8_t *)w_gm, n_groups * rank * row_bytes);

    const uint32_t total = n_tokens * low_dim;
    if (stride == 0) stride = 1;
    for (uint32_t gid = start_gid; gid < total; gid += stride) {
        const uint32_t low_col = gid % low_dim;
        const uint32_t t = gid / low_dim;
        const uint32_t g = low_col / rank;
        const uint32_t r = low_col - g * rank;
        const uint32_t x_base = (t * n_groups + g) * group_dim;
        const uint32_t w_base = (g * rank + r) * row_bytes;
        float acc = 0.0f;
        for (uint32_t b = 0; b < blocks; b++) {
            const uint32_t i0 = b * DS4_QK8_0;
            const uint32_t remain = group_dim - i0;
            const uint32_t bn = remain < DS4_QK8_0 ? remain : DS4_QK8_0;
            float amax = 0.0f;
            for (uint32_t i = 0; i < bn; i++) {
                const float av = ds4_abs_f32(heads.GetValue(x_base + i0 + i));
                if (av > amax) amax = av;
            }
            const float xd = amax / 127.0f;
            const float xid = xd != 0.0f ? 1.0f / xd : 0.0f;
            const uint32_t block_base = w_base + b * DS4_BLOCK_Q8_0_BYTES;
            const uint16_t wscale_bits = (uint16_t)w.GetValue(block_base) | ((uint16_t)w.GetValue(block_base + 1u) << 8);
            const float wd = ds4_f16_to_f32(wscale_bits);
            int32_t dot = 0;
            for (uint32_t i = 0; i < bn; i++) {
                int32_t aq = ds4_round_f32_to_i32(heads.GetValue(x_base + i0 + i) * xid);
                if (aq > 127) aq = 127;
                if (aq < -128) aq = -128;
                dot += (int32_t)(int8_t)w.GetValue(block_base + 2u + i) * aq;
            }
            acc += wd * xd * (float)dot;
        }
        low.SetValue(gid, acc);
    }
}

extern "C" __global__ __aicore__ __attribute__((aiv)) void ds4_quantize_q8_k(GM_ADDR out_gm, GM_ADDR x_gm, uint32_t rows, uint32_t cols, uint32_t start_block, uint32_t stride) {
    GlobalTensor<float> x;
    x.SetGlobalBuffer((__gm__ float *)x_gm, rows * cols);

    __gm__ ds4_block_q8_K *out = (__gm__ ds4_block_q8_K *)out_gm;
    const uint32_t blocks = rows * (cols / DS4_QK_K);
    if (stride == 0) stride = 1;
    for (uint32_t b = start_block; b < blocks; b += stride) {
        const uint32_t base = b * DS4_QK_K;
        float max = 0.0f;
        float amax = 0.0f;
        for (uint32_t j = 0; j < DS4_QK_K; j++) {
            const float v = x.GetValue(base + j);
            const float av = ds4_abs_f32(v);
            if (av > amax) {
                amax = av;
                max = v;
            }
        }

        if (amax == 0.0f) {
            out[b].d = 0.0f;
            for (uint32_t j = 0; j < DS4_QK_K; j++) out[b].qs[j] = 0;
            for (uint32_t j = 0; j < DS4_QK_K / 16; j++) out[b].bsums[j] = 0;
            continue;
        }

        const float iscale = -127.0f / max;
        for (uint32_t j = 0; j < DS4_QK_K; j++) {
            int32_t v = ds4_round_f32_to_i32(iscale * x.GetValue(base + j));
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            out[b].qs[j] = (int8_t)v;
        }
        for (uint32_t j = 0; j < DS4_QK_K / 16; j++) {
            int32_t sum = 0;
            for (uint32_t i = 0; i < 16; i++) sum += (int32_t)out[b].qs[j * 16 + i];
            out[b].bsums[j] = (int16_t)sum;
        }
        out[b].d = 1.0f / iscale;
    }
}

extern "C" __global__ __aicore__ __attribute__((aiv)) void ds4_moe_gate_up_mid_iq2_q8(GM_ADDR gate_out_gm, GM_ADDR up_out_gm, GM_ADDR mid_out_gm, GM_ADDR gate_w_gm, GM_ADDR up_w_gm, GM_ADDR xq_gm, GM_ADDR selected_gm, GM_ADDR weights_gm, GM_ADDR ksigns_gm, GM_ADDR grid_gm, uint32_t pair_count, uint32_t n_expert, uint32_t expert_begin, uint32_t expert_count, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t gate_expert_blocks, uint32_t gate_row_blocks, float clamp, uint32_t start_gid, uint32_t stride) {
    GlobalTensor<float> gate_out;
    GlobalTensor<float> up_out;
    GlobalTensor<float> mid_out;
    GlobalTensor<int32_t> selected;
    GlobalTensor<float> weights;
    gate_out.SetGlobalBuffer((__gm__ float *)gate_out_gm, pair_count * expert_mid_dim);
    up_out.SetGlobalBuffer((__gm__ float *)up_out_gm, pair_count * expert_mid_dim);
    mid_out.SetGlobalBuffer((__gm__ float *)mid_out_gm, pair_count * expert_mid_dim);
    selected.SetGlobalBuffer((__gm__ int32_t *)selected_gm, pair_count);
    weights.SetGlobalBuffer((__gm__ float *)weights_gm, pair_count);
    const __gm__ ds4_block_iq2_xxs *gate_w = (const __gm__ ds4_block_iq2_xxs *)gate_w_gm;
    const __gm__ ds4_block_iq2_xxs *up_w = (const __gm__ ds4_block_iq2_xxs *)up_w_gm;
    const __gm__ ds4_block_q8_K *xq = (const __gm__ ds4_block_q8_K *)xq_gm;
    const __gm__ uint8_t *ksigns = (const __gm__ uint8_t *)ksigns_gm;
    const __gm__ uint64_t *grid = (const __gm__ uint64_t *)grid_gm;
    const uint32_t in_blocks = expert_in_dim / DS4_QK_K;
    const uint32_t total = pair_count * expert_mid_dim;
    if (stride == 0) stride = 1;
    for (uint32_t gid = start_gid; gid < total; gid += stride) {
        const uint32_t pair = gid / expert_mid_dim;
        const uint32_t row = gid - pair * expert_mid_dim;
        int32_t expert_i = selected.GetValue(pair);
        if (expert_i < 0) expert_i = 0;
        if (expert_i < (int32_t)expert_begin || expert_i >= (int32_t)(expert_begin + expert_count)) continue;
        const uint32_t expert = (uint32_t)expert_i - expert_begin;
        const uint32_t tok = pair / n_expert;
        const __gm__ ds4_block_q8_K *xq_row = xq + tok * in_blocks;
        const __gm__ ds4_block_iq2_xxs *gr = gate_w + expert * gate_expert_blocks + row * gate_row_blocks;
        const __gm__ ds4_block_iq2_xxs *ur = up_w + expert * gate_expert_blocks + row * gate_row_blocks;
        float gv = ds4_dot_iq2_xxs_q8_K(expert_in_dim, gr, xq_row, ksigns, grid);
        float uv = ds4_dot_iq2_xxs_q8_K(expert_in_dim, ur, xq_row, ksigns, grid);
        if (clamp > 1.0e-6f) {
            if (gv > clamp) gv = clamp;
            if (uv > clamp) uv = clamp;
            if (uv < -clamp) uv = -clamp;
        }
        const float mid = gv * ds4_sigmoid_f32(gv) * uv * weights.GetValue(pair);
        gate_out.SetValue(gid, gv);
        up_out.SetValue(gid, uv);
        mid_out.SetValue(gid, mid);
    }
}

extern "C" __global__ __aicore__ __attribute__((aiv)) void ds4_moe_down_q2_q8(GM_ADDR experts_out_gm, GM_ADDR down_w_gm, GM_ADDR midq_gm, GM_ADDR selected_gm, uint32_t pair_count, uint32_t expert_begin, uint32_t expert_count, uint32_t expert_mid_dim, uint32_t out_dim, uint32_t down_expert_blocks, uint32_t down_row_blocks, uint32_t start_gid, uint32_t stride) {
    GlobalTensor<float> experts_out;
    GlobalTensor<int32_t> selected;
    experts_out.SetGlobalBuffer((__gm__ float *)experts_out_gm, pair_count * out_dim);
    selected.SetGlobalBuffer((__gm__ int32_t *)selected_gm, pair_count);
    const __gm__ ds4_block_q2_K *down_w = (const __gm__ ds4_block_q2_K *)down_w_gm;
    const __gm__ ds4_block_q8_K *midq = (const __gm__ ds4_block_q8_K *)midq_gm;
    const uint32_t mid_blocks = expert_mid_dim / DS4_QK_K;
    const uint32_t total = pair_count * out_dim;
    if (stride == 0) stride = 1;
    for (uint32_t gid = start_gid; gid < total; gid += stride) {
        const uint32_t pair = gid / out_dim;
        const uint32_t row = gid - pair * out_dim;
        int32_t expert_i = selected.GetValue(pair);
        if (expert_i < 0) expert_i = 0;
        if (expert_i < (int32_t)expert_begin || expert_i >= (int32_t)(expert_begin + expert_count)) continue;
        const uint32_t expert = (uint32_t)expert_i - expert_begin;
        const __gm__ ds4_block_q8_K *midq_row = midq + pair * mid_blocks;
        const __gm__ ds4_block_q2_K *dr = down_w + expert * down_expert_blocks + row * down_row_blocks;
        experts_out.SetValue(gid, ds4_dot_q2_K_q8_K(expert_mid_dim, dr, midq_row));
    }
}

extern "C" __global__ __aicore__ __attribute__((aiv)) void ds4_moe_merge_width(GM_ADDR dst_gm, GM_ADDR src_gm, GM_ADDR selected_gm, uint32_t pair_count, uint32_t width, uint32_t expert_begin, uint32_t expert_count, uint32_t start_gid, uint32_t stride) {
    GlobalTensor<float> dst;
    GlobalTensor<float> src;
    GlobalTensor<int32_t> selected;
    dst.SetGlobalBuffer((__gm__ float *)dst_gm, pair_count * width);
    src.SetGlobalBuffer((__gm__ float *)src_gm, pair_count * width);
    selected.SetGlobalBuffer((__gm__ int32_t *)selected_gm, pair_count);
    const uint32_t total = pair_count * width;
    if (stride == 0) stride = 1;
    for (uint32_t gid = start_gid; gid < total; gid += stride) {
        const uint32_t pair = gid / width;
        int32_t expert_i = selected.GetValue(pair);
        if (expert_i < 0) expert_i = 0;
        if (expert_i < (int32_t)expert_begin || expert_i >= (int32_t)(expert_begin + expert_count)) continue;
        dst.SetValue(gid, src.GetValue(gid));
    }
}

extern "C" __global__ __aicore__ __attribute__((aiv)) void ds4_moe_sum_experts(GM_ADDR out_gm, GM_ADDR experts_gm, uint32_t n_tokens, uint32_t n_expert, uint32_t out_dim, uint32_t start_gid, uint32_t stride) {
    GlobalTensor<float> out;
    GlobalTensor<float> experts;
    out.SetGlobalBuffer((__gm__ float *)out_gm, n_tokens * out_dim);
    experts.SetGlobalBuffer((__gm__ float *)experts_gm, n_tokens * n_expert * out_dim);
    const uint32_t total = n_tokens * out_dim;
    if (stride == 0) stride = 1;
    for (uint32_t gid = start_gid; gid < total; gid += stride) {
        const uint32_t tok = gid / out_dim;
        const uint32_t row = gid - tok * out_dim;
        float acc = 0.0f;
        for (uint32_t slot = 0; slot < n_expert; slot++) {
            const uint32_t off = (tok * n_expert + slot) * out_dim + row;
            acc += experts.GetValue(off);
        }
        out.SetValue(gid, acc);
    }
}

extern "C" __global__ __aicore__ __attribute__((aiv)) void ds4_swiglu(GM_ADDR out_gm, GM_ADDR gate_gm, GM_ADDR up_gm, uint32_t n, float clamp, float weight, uint32_t start_gid, uint32_t stride) {
    GlobalTensor<float> out;
    GlobalTensor<float> gate;
    GlobalTensor<float> up;
    out.SetGlobalBuffer((__gm__ float *)out_gm, n);
    gate.SetGlobalBuffer((__gm__ float *)gate_gm, n);
    up.SetGlobalBuffer((__gm__ float *)up_gm, n);
    if (stride == 0) stride = 1;
    for (uint32_t gid = start_gid; gid < n; gid += stride) {
        float g = gate.GetValue(gid);
        float u = up.GetValue(gid);
        if (clamp > 1.0e-6f) {
            if (g > clamp) g = clamp;
            if (u > clamp) u = clamp;
            if (u < -clamp) u = -clamp;
        }
        out.SetValue(gid, g * ds4_sigmoid_f32(g) * u * weight);
    }
}

extern "C" __global__ __aicore__ __attribute__((aiv)) void ds4_compressor_set_rows(GM_ADDR state_kv_gm, GM_ADDR state_score_gm, GM_ADDR kv_gm, GM_ADDR sc_gm, GM_ADDR ape_gm, uint32_t ape_type, uint32_t width, uint32_t ratio, uint32_t pos0, uint32_t src0, uint32_t dst0, uint32_t rows, uint32_t start_gid, uint32_t stride) {
    GlobalTensor<float> state_kv;
    GlobalTensor<float> state_score;
    GlobalTensor<float> kv;
    GlobalTensor<float> sc;
    GlobalTensor<float> ape_f32;
    GlobalTensor<uint16_t> ape_f16;
    state_kv.SetGlobalBuffer((__gm__ float *)state_kv_gm, rows * width);
    state_score.SetGlobalBuffer((__gm__ float *)state_score_gm, rows * width);
    kv.SetGlobalBuffer((__gm__ float *)kv_gm, (src0 + rows) * width);
    sc.SetGlobalBuffer((__gm__ float *)sc_gm, (src0 + rows) * width);
    ape_f32.SetGlobalBuffer((__gm__ float *)ape_gm, ratio * width);
    ape_f16.SetGlobalBuffer((__gm__ uint16_t *)ape_gm, ratio * width);
    const uint32_t total = rows * width;
    if (stride == 0) stride = 1;
    for (uint32_t gid = start_gid; gid < total; gid += stride) {
        const uint32_t r = gid / width;
        const uint32_t j = gid - r * width;
        const uint32_t src = src0 + r;
        const uint32_t dst = dst0 + r;
        const uint32_t phase = (pos0 + src) % ratio;
        const uint32_t src_idx = src * width + j;
        const uint32_t dst_idx = dst * width + j;
        const uint32_t ape_idx = phase * width + j;
        const float ape = ape_type == 1u ? ds4_f16_to_f32(ape_f16.GetValue(ape_idx)) : ape_f32.GetValue(ape_idx);
        state_kv.SetValue(dst_idx, kv.GetValue(src_idx));
        state_score.SetValue(dst_idx, sc.GetValue(src_idx) + ape);
    }
}

extern "C" __global__ __aicore__ __attribute__((aiv)) void ds4_compressor_prefill_pool(GM_ADDR comp_gm, GM_ADDR kv_gm, GM_ADDR sc_gm, GM_ADDR state_kv_gm, GM_ADDR state_score_gm, GM_ADDR ape_gm, uint32_t ape_type, uint32_t head_dim, uint32_t ratio, uint32_t pos0, uint32_t n_comp, uint32_t replay, uint32_t start_gid, uint32_t stride) {
    GlobalTensor<float> comp;
    GlobalTensor<float> kv;
    GlobalTensor<float> sc;
    GlobalTensor<float> state_kv;
    GlobalTensor<float> state_score;
    GlobalTensor<float> ape_f32;
    GlobalTensor<uint16_t> ape_f16;
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    comp.SetGlobalBuffer((__gm__ float *)comp_gm, n_comp * head_dim);
    kv.SetGlobalBuffer((__gm__ float *)kv_gm, n_comp * ratio * width);
    sc.SetGlobalBuffer((__gm__ float *)sc_gm, n_comp * ratio * width);
    state_kv.SetGlobalBuffer((__gm__ float *)state_kv_gm, coff * ratio * width);
    state_score.SetGlobalBuffer((__gm__ float *)state_score_gm, coff * ratio * width);
    ape_f32.SetGlobalBuffer((__gm__ float *)ape_gm, ratio * width);
    ape_f16.SetGlobalBuffer((__gm__ uint16_t *)ape_gm, ratio * width);
    const uint32_t total = n_comp * head_dim;
    if (stride == 0) stride = 1;
    for (uint32_t gid = start_gid; gid < total; gid += stride) {
        const uint32_t c = gid / head_dim;
        const uint32_t d = gid - c * head_dim;
        float vals[128];
        float scores[128];
        float max_s = -3.402823466e+38f;
        uint32_t n_cand = 0;
        if (ratio == 4u) {
            if (replay && c == 0) {
                for (uint32_t r = 0; r < 4u; r++) {
                    vals[n_cand] = state_kv.GetValue(r * width + d);
                    scores[n_cand] = state_score.GetValue(r * width + d);
                    if (scores[n_cand] > max_s) max_s = scores[n_cand];
                    n_cand++;
                }
            } else if (c > 0) {
                const uint32_t base = (c - 1u) * ratio;
                for (uint32_t r = 0; r < 4u; r++) {
                    const uint32_t t = base + r;
                    const uint32_t phase = (pos0 + t) % ratio;
                    const uint32_t ape_idx = phase * width + d;
                    const float ape = ape_type == 1u ? ds4_f16_to_f32(ape_f16.GetValue(ape_idx)) : ape_f32.GetValue(ape_idx);
                    vals[n_cand] = kv.GetValue(t * width + d);
                    scores[n_cand] = sc.GetValue(t * width + d) + ape;
                    if (scores[n_cand] > max_s) max_s = scores[n_cand];
                    n_cand++;
                }
            }
            const uint32_t base = c * ratio;
            for (uint32_t r = 0; r < 4u; r++) {
                const uint32_t t = base + r;
                const uint32_t phase = (pos0 + t) % ratio;
                const uint32_t ape_idx = phase * width + head_dim + d;
                const float ape = ape_type == 1u ? ds4_f16_to_f32(ape_f16.GetValue(ape_idx)) : ape_f32.GetValue(ape_idx);
                vals[n_cand] = kv.GetValue(t * width + head_dim + d);
                scores[n_cand] = sc.GetValue(t * width + head_dim + d) + ape;
                if (scores[n_cand] > max_s) max_s = scores[n_cand];
                n_cand++;
            }
        } else {
            const uint32_t base = c * ratio;
            for (uint32_t r = 0; r < ratio; r++) {
                const uint32_t t = base + r;
                const uint32_t phase = (pos0 + t) % ratio;
                const uint32_t ape_idx = phase * width + d;
                const float ape = ape_type == 1u ? ds4_f16_to_f32(ape_f16.GetValue(ape_idx)) : ape_f32.GetValue(ape_idx);
                vals[n_cand] = kv.GetValue(t * width + d);
                scores[n_cand] = sc.GetValue(t * width + d) + ape;
                if (scores[n_cand] > max_s) max_s = scores[n_cand];
                n_cand++;
            }
        }
        float den = 0.0f;
        float acc = 0.0f;
        for (uint32_t i = 0; i < n_cand; i++) {
            const float w = ds4_exp_f32(scores[i] - max_s);
            den += w;
            acc += vals[i] * w;
        }
        comp.SetValue(c * head_dim + d, den != 0.0f ? acc / den : 0.0f);
    }
}

extern "C" __global__ __aicore__ __attribute__((aiv)) void ds4_add_f32(GM_ADDR out_gm, GM_ADDR a_gm, GM_ADDR b_gm, uint32_t count, uint32_t start_gid, uint32_t stride) {
    GlobalTensor<float> out;
    GlobalTensor<float> a;
    GlobalTensor<float> b;
    out.SetGlobalBuffer((__gm__ float *)out_gm, count);
    a.SetGlobalBuffer((__gm__ float *)a_gm, count);
    b.SetGlobalBuffer((__gm__ float *)b_gm, count);
    if (stride == 0) stride = 1;
    for (uint32_t i = start_gid; i < count; i += stride) out.SetValue(i, a.GetValue(i) + b.GetValue(i));
}

extern "C" __global__ __aicore__ __attribute__((aiv)) void ds4_compressor_update_pool(GM_ADDR row_gm, GM_ADDR state_kv_gm, GM_ADDR state_score_gm, uint32_t head_dim, uint32_t ratio, uint32_t start_gid, uint32_t stride) {
    GlobalTensor<float> row;
    GlobalTensor<float> state_kv;
    GlobalTensor<float> state_score;
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    row.SetGlobalBuffer((__gm__ float *)row_gm, head_dim);
    state_kv.SetGlobalBuffer((__gm__ float *)state_kv_gm, coff * ratio * width);
    state_score.SetGlobalBuffer((__gm__ float *)state_score_gm, coff * ratio * width);
    if (stride == 0) stride = 1;
    for (uint32_t d = start_gid; d < head_dim; d += stride) {
        float vals[128];
        float scores[128];
        float max_s = -3.402823466e+38f;
        uint32_t n_cand = 0;
        if (ratio == 4u) {
            for (uint32_t r = 0; r < 4u; r++) {
                vals[n_cand] = state_kv.GetValue(r * width + d);
                scores[n_cand] = state_score.GetValue(r * width + d);
                if (scores[n_cand] > max_s) max_s = scores[n_cand];
                n_cand++;
            }
            for (uint32_t r = 0; r < 4u; r++) {
                vals[n_cand] = state_kv.GetValue((ratio + r) * width + head_dim + d);
                scores[n_cand] = state_score.GetValue((ratio + r) * width + head_dim + d);
                if (scores[n_cand] > max_s) max_s = scores[n_cand];
                n_cand++;
            }
        } else {
            for (uint32_t r = 0; r < ratio; r++) {
                vals[n_cand] = state_kv.GetValue(r * width + d);
                scores[n_cand] = state_score.GetValue(r * width + d);
                if (scores[n_cand] > max_s) max_s = scores[n_cand];
                n_cand++;
            }
        }
        float den = 0.0f;
        float acc = 0.0f;
        for (uint32_t i = 0; i < n_cand; i++) {
            const float w = ds4_exp_f32(scores[i] - max_s);
            den += w;
            acc += vals[i] * w;
        }
        row.SetValue(d, den != 0.0f ? acc / den : 0.0f);
    }
}

extern "C" __global__ __aicore__ __attribute__((aiv)) void ds4_compressor_shift_ratio4(GM_ADDR state_kv_gm, GM_ADDR state_score_gm, uint32_t width, uint32_t start_gid, uint32_t stride) {
    GlobalTensor<float> state_kv;
    GlobalTensor<float> state_score;
    state_kv.SetGlobalBuffer((__gm__ float *)state_kv_gm, 8u * width);
    state_score.SetGlobalBuffer((__gm__ float *)state_score_gm, 8u * width);
    const uint32_t half = 4u * width;
    if (stride == 0) stride = 1;
    for (uint32_t i = start_gid; i < half; i += stride) {
        const float v = state_kv.GetValue(half + i);
        const float s = state_score.GetValue(half + i);
        state_kv.SetValue(i, v);
        state_score.SetValue(i, s);
        state_kv.SetValue(half + i, v);
        state_score.SetValue(half + i, s);
    }
}

extern "C" void ds4_ascend_launch_fill_f32(void *stream, void *out, float value, uint32_t count) {
    ds4_fill_f32<<<1, nullptr, stream>>>((GM_ADDR)out, value, count);
}

extern "C" void ds4_ascend_launch_add_f32(void *stream, void *out, const void *a, const void *b, uint32_t count) {
    const uint32_t parts = ds4_ascend_kernel_parts();
    for (uint32_t p = 0; p < parts; p++) {
        ds4_add_f32<<<1, nullptr, stream>>>((GM_ADDR)out, (GM_ADDR)a, (GM_ADDR)b, count, p, parts);
    }
}

extern "C" void ds4_ascend_launch_quantize_q8_k(void *stream, void *out, const void *x, uint32_t rows, uint32_t cols) {
    const uint32_t parts = ds4_ascend_kernel_parts();
    for (uint32_t p = 0; p < parts; p++) {
        ds4_quantize_q8_k<<<1, nullptr, stream>>>((GM_ADDR)out, (GM_ADDR)x, rows, cols, p, parts);
    }
}

extern "C" void ds4_ascend_launch_moe_gate_up_mid_iq2_q8(void *stream, void *gate_out, void *up_out, void *mid_out, const void *gate_w, const void *up_w, const void *xq, const void *selected, const void *weights, const void *ksigns, const void *grid, uint32_t pair_count, uint32_t n_expert, uint32_t expert_begin, uint32_t expert_count, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, float clamp) {
    const uint32_t parts = ds4_ascend_kernel_parts();
    const uint32_t gate_expert_blocks = (uint32_t)(gate_expert_bytes / sizeof(ds4_block_iq2_xxs));
    const uint32_t gate_row_blocks = (uint32_t)(gate_row_bytes / sizeof(ds4_block_iq2_xxs));
    for (uint32_t p = 0; p < parts; p++) {
        ds4_moe_gate_up_mid_iq2_q8<<<1, nullptr, stream>>>((GM_ADDR)gate_out, (GM_ADDR)up_out, (GM_ADDR)mid_out, (GM_ADDR)gate_w, (GM_ADDR)up_w, (GM_ADDR)xq, (GM_ADDR)selected, (GM_ADDR)weights, (GM_ADDR)ksigns, (GM_ADDR)grid, pair_count, n_expert, expert_begin, expert_count, expert_in_dim, expert_mid_dim, gate_expert_blocks, gate_row_blocks, clamp, p, parts);
    }
}

extern "C" void ds4_ascend_launch_moe_down_q2_q8(void *stream, void *experts_out, const void *down_w, const void *midq, const void *selected, uint32_t pair_count, uint32_t expert_begin, uint32_t expert_count, uint32_t expert_mid_dim, uint32_t out_dim, uint64_t down_expert_bytes, uint64_t down_row_bytes) {
    const uint32_t parts = ds4_ascend_kernel_parts();
    const uint32_t down_expert_blocks = (uint32_t)(down_expert_bytes / sizeof(ds4_block_q2_K));
    const uint32_t down_row_blocks = (uint32_t)(down_row_bytes / sizeof(ds4_block_q2_K));
    for (uint32_t p = 0; p < parts; p++) {
        ds4_moe_down_q2_q8<<<1, nullptr, stream>>>((GM_ADDR)experts_out, (GM_ADDR)down_w, (GM_ADDR)midq, (GM_ADDR)selected, pair_count, expert_begin, expert_count, expert_mid_dim, out_dim, down_expert_blocks, down_row_blocks, p, parts);
    }
}

extern "C" void ds4_ascend_launch_moe_merge_width(void *stream, void *dst, const void *src, const void *selected, uint32_t pair_count, uint32_t width, uint32_t expert_begin, uint32_t expert_count) {
    const uint32_t parts = ds4_ascend_kernel_parts();
    for (uint32_t p = 0; p < parts; p++) {
        ds4_moe_merge_width<<<1, nullptr, stream>>>((GM_ADDR)dst, (GM_ADDR)src, (GM_ADDR)selected, pair_count, width, expert_begin, expert_count, p, parts);
    }
}

extern "C" void ds4_ascend_launch_moe_sum_experts(void *stream, void *out, const void *experts, uint32_t n_tokens, uint32_t n_expert, uint32_t out_dim) {
    const uint32_t parts = ds4_ascend_kernel_parts();
    for (uint32_t p = 0; p < parts; p++) {
        ds4_moe_sum_experts<<<1, nullptr, stream>>>((GM_ADDR)out, (GM_ADDR)experts, n_tokens, n_expert, out_dim, p, parts);
    }
}

extern "C" void ds4_ascend_launch_swiglu(void *stream, void *out, const void *gate, const void *up, uint32_t n, float clamp, float weight) {
    const uint32_t parts = ds4_ascend_kernel_parts();
    for (uint32_t p = 0; p < parts; p++) {
        ds4_swiglu<<<1, nullptr, stream>>>((GM_ADDR)out, (GM_ADDR)gate, (GM_ADDR)up, n, clamp, weight, p, parts);
    }
}

extern "C" void ds4_ascend_launch_compressor_set_rows(void *stream, void *state_kv, void *state_score, const void *kv, const void *sc, const void *ape, uint32_t ape_type, uint32_t width, uint32_t ratio, uint32_t pos0, uint32_t src0, uint32_t dst0, uint32_t rows) {
    const uint32_t parts = ds4_ascend_kernel_parts();
    for (uint32_t p = 0; p < parts; p++) {
        ds4_compressor_set_rows<<<1, nullptr, stream>>>((GM_ADDR)state_kv, (GM_ADDR)state_score, (GM_ADDR)kv, (GM_ADDR)sc, (GM_ADDR)ape, ape_type, width, ratio, pos0, src0, dst0, rows, p, parts);
    }
}

extern "C" void ds4_ascend_launch_compressor_prefill_pool(void *stream, void *comp, const void *kv, const void *sc, const void *state_kv, const void *state_score, const void *ape, uint32_t ape_type, uint32_t head_dim, uint32_t ratio, uint32_t pos0, uint32_t n_comp, uint32_t replay) {
    const uint32_t parts = ds4_ascend_kernel_parts();
    for (uint32_t p = 0; p < parts; p++) {
        ds4_compressor_prefill_pool<<<1, nullptr, stream>>>((GM_ADDR)comp, (GM_ADDR)kv, (GM_ADDR)sc, (GM_ADDR)state_kv, (GM_ADDR)state_score, (GM_ADDR)ape, ape_type, head_dim, ratio, pos0, n_comp, replay, p, parts);
    }
}

extern "C" void ds4_ascend_launch_compressor_update_pool(void *stream, void *row, const void *state_kv, const void *state_score, uint32_t head_dim, uint32_t ratio) {
    const uint32_t parts = ds4_ascend_kernel_parts();
    for (uint32_t p = 0; p < parts; p++) {
        ds4_compressor_update_pool<<<1, nullptr, stream>>>((GM_ADDR)row, (GM_ADDR)state_kv, (GM_ADDR)state_score, head_dim, ratio, p, parts);
    }
}

extern "C" void ds4_ascend_launch_compressor_shift_ratio4(void *stream, void *state_kv, void *state_score, uint32_t width) {
    const uint32_t parts = ds4_ascend_kernel_parts();
    for (uint32_t p = 0; p < parts; p++) {
        ds4_compressor_shift_ratio4<<<1, nullptr, stream>>>((GM_ADDR)state_kv, (GM_ADDR)state_score, width, p, parts);
    }
}

extern "C" void ds4_ascend_launch_matmul_f16(void *stream, void *out, const void *w, const void *x, uint32_t in_dim, uint32_t out_dim, uint32_t n_tok) {
    const uint32_t parts = ds4_ascend_kernel_parts();
    for (uint32_t p = 0; p < parts; p++) {
        ds4_matmul_f16<<<1, nullptr, stream>>>((GM_ADDR)out, (GM_ADDR)w, (GM_ADDR)x, in_dim, out_dim, n_tok, p, parts);
    }
}

extern "C" void ds4_ascend_launch_embed_token_hc(void *stream, void *out, const void *w, uint32_t token, uint32_t n_embd, uint32_t n_hc) {
    const uint32_t parts = ds4_ascend_kernel_parts();
    for (uint32_t p = 0; p < parts; p++) {
        ds4_embed_token_hc<<<1, nullptr, stream>>>((GM_ADDR)out, (GM_ADDR)w, token, n_embd, n_hc, p, parts);
    }
}

extern "C" void ds4_ascend_launch_embed_tokens_hc(void *stream, void *out, const void *tokens, const void *w, uint32_t n_vocab, uint32_t n_tokens, uint32_t n_embd, uint32_t n_hc) {
    const uint32_t parts = ds4_ascend_kernel_parts();
    for (uint32_t p = 0; p < parts; p++) {
        ds4_embed_tokens_hc<<<1, nullptr, stream>>>((GM_ADDR)out, (GM_ADDR)tokens, (GM_ADDR)w, n_vocab, n_tokens, n_embd, n_hc, p, parts);
    }
}

extern "C" void ds4_ascend_launch_matmul_q8_0(void *stream, void *out, const void *w, const void *x, uint32_t in_dim, uint32_t out_dim, uint32_t n_tok) {
    const uint32_t parts = ds4_ascend_matmul_q8_0_parts();
    for (uint32_t p = 0; p < parts; p++) {
        ds4_matmul_q8_0<<<1, nullptr, stream>>>((GM_ADDR)out, (GM_ADDR)w, (GM_ADDR)x, in_dim, out_dim, n_tok, p, parts);
    }
}

extern "C" void ds4_ascend_launch_quantize_q8_0(void *stream, void *xq, void *xscale, const void *x, uint32_t in_dim, uint32_t n_tok) {
    const uint32_t parts = ds4_ascend_kernel_parts();
    for (uint32_t p = 0; p < parts; p++) {
        ds4_quantize_q8_0<<<1, nullptr, stream>>>((GM_ADDR)xq, (GM_ADDR)xscale, (GM_ADDR)x, in_dim, n_tok, p, parts);
    }
}

extern "C" void ds4_ascend_launch_matmul_q8_0_prequant(void *stream, void *out, const void *w, const void *xq, const void *xscale, uint32_t in_dim, uint32_t out_dim, uint32_t n_tok) {
    const uint32_t parts = ds4_ascend_matmul_q8_0_parts();
    for (uint32_t p = 0; p < parts; p++) {
        ds4_matmul_q8_0_prequant<<<1, nullptr, stream>>>((GM_ADDR)out, (GM_ADDR)w, (GM_ADDR)xq, (GM_ADDR)xscale, in_dim, out_dim, n_tok, p, parts);
    }
}

extern "C" void ds4_ascend_launch_rms_norm_plain(void *stream, void *out, const void *x, uint32_t n, uint32_t rows, float eps) {
    const uint32_t parts = ds4_ascend_kernel_parts();
    const float inv_n = 1.0f / (float)n;
    for (uint32_t p = 0; p < parts; p++) {
        ds4_rms_norm_plain<<<1, nullptr, stream>>>((GM_ADDR)out, (GM_ADDR)x, n, rows, inv_n, eps, p, parts);
    }
}

extern "C" void ds4_ascend_launch_rms_norm_weight(void *stream, void *out, const void *x, const void *weight, uint32_t n, uint32_t rows, float eps) {
    const uint32_t parts = ds4_ascend_kernel_parts();
    const float inv_n = 1.0f / (float)n;
    for (uint32_t p = 0; p < parts; p++) {
        ds4_rms_norm_weight<<<1, nullptr, stream>>>((GM_ADDR)out, (GM_ADDR)x, (GM_ADDR)weight, n, rows, inv_n, eps, p, parts);
    }
}

extern "C" void ds4_ascend_launch_rms_norm_inplace(void *stream, void *x, uint32_t n, uint32_t rows, float eps) {
    const uint32_t parts = ds4_ascend_kernel_parts();
    const float inv_n = 1.0f / (float)n;
    for (uint32_t p = 0; p < parts; p++) {
        ds4_rms_norm_inplace<<<1, nullptr, stream>>>((GM_ADDR)x, n, rows, inv_n, eps, p, parts);
    }
}

extern "C" void ds4_ascend_launch_router_select(void *stream, void *selected, void *weights, void *probs, const void *bias, const void *hash, const void *logits, const void *tokens, int32_t token_scalar, uint32_t hash_rows, uint32_t n_tokens, uint32_t has_bias, uint32_t hash_mode) {
    const uint32_t parts = ds4_ascend_kernel_parts();
    for (uint32_t p = 0; p < parts; p++) {
        ds4_router_select<<<1, nullptr, stream>>>((GM_ADDR)selected, (GM_ADDR)weights, (GM_ADDR)probs, (GM_ADDR)bias, (GM_ADDR)hash, (GM_ADDR)logits, (GM_ADDR)tokens, token_scalar, hash_rows, n_tokens, has_bias, hash_mode, p, parts);
    }
}

extern "C" void ds4_ascend_launch_hc_expand_split(void *stream, void *out, const void *block_out, const void *block_add, const void *residual, const void *split, uint32_t n_embd, uint32_t n_hc, uint32_t rows, uint32_t has_add) {
    const uint32_t parts = ds4_ascend_kernel_parts();
    for (uint32_t p = 0; p < parts; p++) {
        ds4_hc_expand_split<<<1, nullptr, stream>>>((GM_ADDR)out, (GM_ADDR)block_out, (GM_ADDR)block_add, (GM_ADDR)residual, (GM_ADDR)split, n_embd, n_hc, rows, has_add, p, parts);
    }
}

extern "C" void ds4_ascend_launch_hc_expand(void *stream, void *out, const void *block_out, const void *residual, const void *post, const void *comb, uint32_t n_embd, uint32_t n_hc, uint32_t rows) {
    const uint32_t parts = ds4_ascend_kernel_parts();
    for (uint32_t p = 0; p < parts; p++) {
        ds4_hc_expand<<<1, nullptr, stream>>>((GM_ADDR)out, (GM_ADDR)block_out, (GM_ADDR)residual, (GM_ADDR)post, (GM_ADDR)comb, n_embd, n_hc, rows, p, parts);
    }
}

extern "C" void ds4_ascend_launch_hc_split_sinkhorn4(void *stream, void *out, const void *mix, const void *scale, const void *base, uint32_t rows, uint32_t sinkhorn_iters, float eps) {
    const uint32_t parts = ds4_ascend_kernel_parts();
    for (uint32_t p = 0; p < parts; p++) {
        ds4_hc_split_sinkhorn4<<<1, nullptr, stream>>>((GM_ADDR)out, (GM_ADDR)mix, (GM_ADDR)scale, (GM_ADDR)base, rows, sinkhorn_iters, eps, p, parts);
    }
}

extern "C" void ds4_ascend_launch_hc_weighted_sum4(void *stream, void *out, const void *residual, const void *weights, uint32_t n_embd, uint32_t rows, uint32_t weight_stride) {
    const uint32_t parts = ds4_ascend_kernel_parts();
    for (uint32_t p = 0; p < parts; p++) {
        ds4_hc_weighted_sum4<<<1, nullptr, stream>>>((GM_ADDR)out, (GM_ADDR)residual, (GM_ADDR)weights, n_embd, rows, weight_stride, p, parts);
    }
}

extern "C" void ds4_ascend_launch_output_hc_weights(void *stream, void *out, const void *pre, const void *scale, const void *base, uint32_t n_hc, uint32_t n_tokens, float eps) {
    const uint32_t parts = ds4_ascend_kernel_parts();
    for (uint32_t p = 0; p < parts; p++) {
        ds4_output_hc_weights<<<1, nullptr, stream>>>((GM_ADDR)out, (GM_ADDR)pre, (GM_ADDR)scale, (GM_ADDR)base, n_hc, n_tokens, eps, p, parts);
    }
}

extern "C" void ds4_ascend_launch_rope_tail_table(void *stream, void *x, const void *table, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot) {
    const uint32_t parts = ds4_ascend_kernel_parts();
    for (uint32_t p = 0; p < parts; p++) {
        ds4_rope_tail_table<<<1, nullptr, stream>>>((GM_ADDR)x, (GM_ADDR)table, n_tok, n_head, head_dim, n_rot, p, parts);
    }
}

extern "C" void ds4_ascend_launch_fp8_kv_quantize(void *stream, void *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot) {
    const uint32_t parts = ds4_ascend_kernel_parts();
    for (uint32_t p = 0; p < parts; p++) {
        ds4_fp8_kv_quantize<<<1, nullptr, stream>>>((GM_ADDR)x, n_tok, head_dim, n_rot, p, parts);
    }
}

extern "C" void ds4_ascend_launch_store_raw_kv_batch(void *stream, void *raw, const void *kv, uint32_t raw_cap, uint32_t pos0, uint32_t n_tokens, uint32_t head_dim) {
    const uint32_t parts = ds4_ascend_kernel_parts();
    for (uint32_t p = 0; p < parts; p++) {
        ds4_store_raw_kv_batch<<<1, nullptr, stream>>>((GM_ADDR)raw, (GM_ADDR)kv, raw_cap, pos0, n_tokens, head_dim, p, parts);
    }
}

extern "C" void ds4_ascend_launch_attention_prefill_raw(void *stream, void *heads, const void *sinks, const void *q, const void *raw_kv, uint32_t n_tokens, uint32_t window, uint32_t n_head, uint32_t head_dim, float scale) {
    const uint32_t parts = ds4_ascend_kernel_parts();
    for (uint32_t p = 0; p < parts; p++) {
        ds4_attention_prefill_raw<<<1, nullptr, stream>>>((GM_ADDR)heads, (GM_ADDR)sinks, (GM_ADDR)q, (GM_ADDR)raw_kv, n_tokens, window, n_head, head_dim, scale, p, parts);
    }
}

extern "C" void ds4_ascend_launch_attention_decode(void *stream, void *heads, const void *sinks, const void *q, const void *raw_kv, const void *comp_kv, const void *comp_mask, uint32_t use_comp_mask, uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start, uint32_t n_comp, uint32_t n_head, uint32_t head_dim, float scale) {
    const uint32_t parts = ds4_ascend_kernel_parts();
    for (uint32_t p = 0; p < parts; p++) {
        ds4_attention_decode<<<1, nullptr, stream>>>((GM_ADDR)heads, (GM_ADDR)sinks, (GM_ADDR)q, (GM_ADDR)raw_kv, (GM_ADDR)comp_kv, (GM_ADDR)comp_mask, use_comp_mask, n_raw, raw_cap, raw_start, n_comp, n_head, head_dim, scale, p, parts);
    }
}

extern "C" void ds4_ascend_launch_attention_prefill_mixed(void *stream, void *heads, const void *sinks, const void *q, const void *raw_kv, const void *comp_kv, const void *comp_mask, uint32_t use_comp_mask, uint32_t n_tokens, uint32_t n_comp, uint32_t window, uint32_t ratio, uint32_t n_head, uint32_t head_dim, float scale) {
    const uint32_t parts = ds4_ascend_kernel_parts();
    for (uint32_t p = 0; p < parts; p++) {
        ds4_attention_prefill_mixed<<<1, nullptr, stream>>>((GM_ADDR)heads, (GM_ADDR)sinks, (GM_ADDR)q, (GM_ADDR)raw_kv, (GM_ADDR)comp_kv, (GM_ADDR)comp_mask, use_comp_mask, n_tokens, n_comp, window, ratio, n_head, head_dim, scale, p, parts);
    }
}

extern "C" void ds4_ascend_launch_attention_output_low_q8(void *stream, void *low, const void *w, const void *heads, uint32_t group_dim, uint32_t rank, uint32_t n_groups, uint32_t n_tokens) {
    const uint32_t parts = ds4_ascend_kernel_parts();
    for (uint32_t p = 0; p < parts; p++) {
        ds4_attention_output_low_q8<<<1, nullptr, stream>>>((GM_ADDR)low, (GM_ADDR)w, (GM_ADDR)heads, group_dim, rank, n_groups, n_tokens, p, parts);
    }
}
