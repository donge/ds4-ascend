#include "kernel_operator.h"

#include <stdint.h>
#include <math.h>

using namespace AscendC;

#define DS4_QK_K 256u
#define DS4_QK8_0 32u
#define DS4_BLOCK_Q8_0_BYTES 34u

typedef struct {
    float d;
    int8_t qs[DS4_QK_K];
    int16_t bsums[DS4_QK_K / 16];
} ds4_block_q8_K;

static __aicore__ inline float ds4_abs_f32(float x) {
    return x < 0.0f ? -x : x;
}

static __aicore__ inline int32_t ds4_round_f32_to_i32(float x) {
    return (int32_t)(x >= 0.0f ? x + 0.5f : x - 0.5f);
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

extern "C" __global__ __aicore__ __attribute__((aiv)) void ds4_quantize_q8_k(GM_ADDR out_gm, GM_ADDR x_gm, uint32_t rows, uint32_t cols) {
    GlobalTensor<float> x;
    x.SetGlobalBuffer((__gm__ float *)x_gm, rows * cols);

    __gm__ ds4_block_q8_K *out = (__gm__ ds4_block_q8_K *)out_gm;
    const uint32_t blocks = rows * (cols / DS4_QK_K);
    for (uint32_t b = 0; b < blocks; b++) {
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

extern "C" void ds4_ascend_launch_fill_f32(void *stream, void *out, float value, uint32_t count) {
    ds4_fill_f32<<<1, nullptr, stream>>>((GM_ADDR)out, value, count);
}

extern "C" void ds4_ascend_launch_quantize_q8_k(void *stream, void *out, const void *x, uint32_t rows, uint32_t cols) {
    ds4_quantize_q8_k<<<1, nullptr, stream>>>((GM_ADDR)out, (GM_ADDR)x, rows, cols);
}

extern "C" void ds4_ascend_launch_matmul_f16(void *stream, void *out, const void *w, const void *x, uint32_t in_dim, uint32_t out_dim, uint32_t n_tok) {
    const uint32_t parts = 8u;
    for (uint32_t p = 0; p < parts; p++) {
        ds4_matmul_f16<<<1, nullptr, stream>>>((GM_ADDR)out, (GM_ADDR)w, (GM_ADDR)x, in_dim, out_dim, n_tok, p, parts);
    }
}

extern "C" void ds4_ascend_launch_matmul_q8_0(void *stream, void *out, const void *w, const void *x, uint32_t in_dim, uint32_t out_dim, uint32_t n_tok) {
    const uint32_t parts = 8u;
    for (uint32_t p = 0; p < parts; p++) {
        ds4_matmul_q8_0<<<1, nullptr, stream>>>((GM_ADDR)out, (GM_ADDR)w, (GM_ADDR)x, in_dim, out_dim, n_tok, p, parts);
    }
}

extern "C" void ds4_ascend_launch_rms_norm_plain(void *stream, void *out, const void *x, uint32_t n, uint32_t rows, float eps) {
    const uint32_t parts = 8u;
    const float inv_n = 1.0f / (float)n;
    for (uint32_t p = 0; p < parts; p++) {
        ds4_rms_norm_plain<<<1, nullptr, stream>>>((GM_ADDR)out, (GM_ADDR)x, n, rows, inv_n, eps, p, parts);
    }
}

extern "C" void ds4_ascend_launch_rms_norm_weight(void *stream, void *out, const void *x, const void *weight, uint32_t n, uint32_t rows, float eps) {
    const uint32_t parts = 8u;
    const float inv_n = 1.0f / (float)n;
    for (uint32_t p = 0; p < parts; p++) {
        ds4_rms_norm_weight<<<1, nullptr, stream>>>((GM_ADDR)out, (GM_ADDR)x, (GM_ADDR)weight, n, rows, inv_n, eps, p, parts);
    }
}

extern "C" void ds4_ascend_launch_rms_norm_inplace(void *stream, void *x, uint32_t n, uint32_t rows, float eps) {
    const uint32_t parts = 8u;
    const float inv_n = 1.0f / (float)n;
    for (uint32_t p = 0; p < parts; p++) {
        ds4_rms_norm_inplace<<<1, nullptr, stream>>>((GM_ADDR)x, n, rows, inv_n, eps, p, parts);
    }
}

extern "C" void ds4_ascend_launch_hc_split_sinkhorn4(void *stream, void *out, const void *mix, const void *scale, const void *base, uint32_t rows, uint32_t sinkhorn_iters, float eps) {
    const uint32_t parts = 8u;
    for (uint32_t p = 0; p < parts; p++) {
        ds4_hc_split_sinkhorn4<<<1, nullptr, stream>>>((GM_ADDR)out, (GM_ADDR)mix, (GM_ADDR)scale, (GM_ADDR)base, rows, sinkhorn_iters, eps, p, parts);
    }
}

extern "C" void ds4_ascend_launch_hc_weighted_sum4(void *stream, void *out, const void *residual, const void *weights, uint32_t n_embd, uint32_t rows, uint32_t weight_stride) {
    const uint32_t parts = 8u;
    for (uint32_t p = 0; p < parts; p++) {
        ds4_hc_weighted_sum4<<<1, nullptr, stream>>>((GM_ADDR)out, (GM_ADDR)residual, (GM_ADDR)weights, n_embd, rows, weight_stride, p, parts);
    }
}

extern "C" void ds4_ascend_launch_rope_tail_table(void *stream, void *x, const void *table, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot) {
    const uint32_t parts = 8u;
    for (uint32_t p = 0; p < parts; p++) {
        ds4_rope_tail_table<<<1, nullptr, stream>>>((GM_ADDR)x, (GM_ADDR)table, n_tok, n_head, head_dim, n_rot, p, parts);
    }
}

extern "C" void ds4_ascend_launch_fp8_kv_quantize(void *stream, void *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot) {
    const uint32_t parts = 8u;
    for (uint32_t p = 0; p < parts; p++) {
        ds4_fp8_kv_quantize<<<1, nullptr, stream>>>((GM_ADDR)x, n_tok, head_dim, n_rot, p, parts);
    }
}

extern "C" void ds4_ascend_launch_store_raw_kv_batch(void *stream, void *raw, const void *kv, uint32_t raw_cap, uint32_t pos0, uint32_t n_tokens, uint32_t head_dim) {
    const uint32_t parts = 8u;
    for (uint32_t p = 0; p < parts; p++) {
        ds4_store_raw_kv_batch<<<1, nullptr, stream>>>((GM_ADDR)raw, (GM_ADDR)kv, raw_cap, pos0, n_tokens, head_dim, p, parts);
    }
}
