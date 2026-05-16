#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QK_K 256u

typedef struct {
    float d;
    int8_t qs[QK_K];
    int16_t bsums[QK_K / 16];
} block_q8_K;

typedef struct {
    uint16_t d;
    int8_t qs[32];
} block_q8_0;

static uint16_t f32_to_f16(float f) {
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

static float f16_to_f32(uint16_t h) {
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

static int32_t round_f32_to_i32(float x) {
    return (int32_t)(x >= 0.0f ? x + 0.5f : x - 0.5f);
}

static void quantize_row_q8_0(const float *x, block_q8_0 *y, uint32_t k) {
    const uint32_t qk = 32;
    const uint32_t nb = (k + qk - 1) / qk;
    for (uint32_t b = 0; b < nb; b++) {
        const uint32_t i0 = b * qk;
        const uint32_t bn = k - i0 < qk ? k - i0 : qk;
        float amax = 0.0f;
        for (uint32_t i = 0; i < bn; i++) {
            const float av = fabsf(x[i0 + i]);
            if (av > amax) amax = av;
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        y[b].d = f32_to_f16(d);
        memset(y[b].qs, 0, sizeof(y[b].qs));
        for (uint32_t i = 0; i < bn; i++) {
            int v = round_f32_to_i32(x[i0 + i] * id);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            y[b].qs[i] = (int8_t)v;
        }
    }
}

static void matmul_q8_0_ref(float *out, const block_q8_0 *w, const float *x, uint32_t in_dim, uint32_t out_dim, uint32_t n_tok) {
    const uint32_t qk = 32;
    const uint32_t blocks = (in_dim + qk - 1) / qk;
    int8_t *xq = calloc((size_t)n_tok * blocks * qk, sizeof(*xq));
    float *xscale = calloc((size_t)n_tok * blocks, sizeof(*xscale));
    if (!xq || !xscale) {
        free(xq); free(xscale);
        return;
    }
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t b = 0; b < blocks; b++) {
            const uint32_t i0 = b * qk;
            const uint32_t bn = in_dim - i0 < qk ? in_dim - i0 : qk;
            float amax = 0.0f;
            for (uint32_t i = 0; i < bn; i++) {
                const float av = fabsf(x[(uint64_t)t * in_dim + i0 + i]);
                if (av > amax) amax = av;
            }
            const float xd = amax / 127.0f;
            const float xid = xd != 0.0f ? 1.0f / xd : 0.0f;
            xscale[(uint64_t)t * blocks + b] = xd;
            for (uint32_t i = 0; i < bn; i++) {
                int v = round_f32_to_i32(x[(uint64_t)t * in_dim + i0 + i] * xid);
                if (v > 127) v = 127;
                if (v < -128) v = -128;
                xq[((uint64_t)t * blocks + b) * qk + i] = (int8_t)v;
            }
        }
    }
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t r = 0; r < out_dim; r++) {
            float acc = 0.0f;
            for (uint32_t b = 0; b < blocks; b++) {
                const uint32_t i0 = b * qk;
                const uint32_t bn = in_dim - i0 < qk ? in_dim - i0 : qk;
                const block_q8_0 *wb = w + (uint64_t)r * blocks + b;
                const int8_t *xb = xq + ((uint64_t)t * blocks + b) * qk;
                int32_t dot = 0;
                for (uint32_t i = 0; i < bn; i++) dot += (int32_t)wb->qs[i] * (int32_t)xb[i];
                acc += f16_to_f32(wb->d) * xscale[(uint64_t)t * blocks + b] * (float)dot;
            }
            out[(uint64_t)t * out_dim + r] = acc;
        }
    }
    free(xscale);
    free(xq);
}

static void quantize_row_q8_K(const float *x, block_q8_K *y, uint32_t k) {
    const uint32_t nb = k / QK_K;
    for (uint32_t b = 0; b < nb; b++) {
        float max = 0.0f;
        float amax = 0.0f;
        for (uint32_t j = 0; j < QK_K; j++) {
            const float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax;
                max = x[j];
            }
        }
        if (amax == 0.0f) {
            y[b].d = 0.0f;
            memset(y[b].qs, 0, sizeof(y[b].qs));
            memset(y[b].bsums, 0, sizeof(y[b].bsums));
            x += QK_K;
            continue;
        }
        const float iscale = -127.0f / max;
        for (uint32_t j = 0; j < QK_K; j++) {
            int v = round_f32_to_i32(iscale * x[j]);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            y[b].qs[j] = (int8_t)v;
        }
        for (uint32_t j = 0; j < QK_K / 16; j++) {
            int sum = 0;
            for (uint32_t i = 0; i < 16; i++) sum += y[b].qs[j * 16 + i];
            y[b].bsums[j] = (int16_t)sum;
        }
        y[b].d = 1.0f / iscale;
        x += QK_K;
    }
}

static void rms_norm_ref(float *out, const float *x, const float *weight, uint32_t n, uint32_t rows, float eps) {
    for (uint32_t r = 0; r < rows; r++) {
        const uint64_t base = (uint64_t)r * n;
        float ss = 0.0f;
        for (uint32_t i = 0; i < n; i++) ss += x[base + i] * x[base + i];
        const float scale = 1.0f / sqrtf(ss / (float)n + eps);
        for (uint32_t i = 0; i < n; i++) out[base + i] = x[base + i] * scale * (weight ? weight[i] : 1.0f);
    }
}

static void hc_weighted_sum_ref(float *out, const float *residual, const float *weights, uint32_t n_embd, uint32_t rows, uint32_t weight_stride) {
    for (uint32_t r = 0; r < rows; r++) {
        const uint64_t res_base = (uint64_t)r * 4u * n_embd;
        const uint64_t w_base = (uint64_t)r * weight_stride;
        const uint64_t out_base = (uint64_t)r * n_embd;
        for (uint32_t d = 0; d < n_embd; d++) {
            out[out_base + d] = residual[res_base + d] * weights[w_base] +
                                residual[res_base + n_embd + d] * weights[w_base + 1u] +
                                residual[res_base + 2u * n_embd + d] * weights[w_base + 2u] +
                                residual[res_base + 3u * n_embd + d] * weights[w_base + 3u];
        }
    }
}

static void hc_split_ref(float *out, const float *mix, const float *scale, const float *base, uint32_t rows, uint32_t sinkhorn_iters, float eps) {
    const uint32_t n_hc = 4;
    const uint32_t mix_hc = 24;
    const float pre_scale = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    for (uint32_t r = 0; r < rows; r++) {
        const float *m = mix + (uint64_t)r * mix_hc;
        float *o = out + (uint64_t)r * mix_hc;
        for (uint32_t i = 0; i < n_hc; i++) o[i] = 1.0f / (1.0f + expf(-(m[i] * pre_scale + base[i]))) + eps;
        for (uint32_t i = 0; i < n_hc; i++) {
            const uint32_t off = n_hc + i;
            o[off] = 2.0f / (1.0f + expf(-(m[off] * post_scale + base[off])));
        }

        float c[16];
        for (uint32_t dst = 0; dst < n_hc; dst++) {
            float row_max = -3.402823466e+38f;
            for (uint32_t src = 0; src < n_hc; src++) {
                const uint32_t idx = src + dst * n_hc;
                const uint32_t off = 2u * n_hc + idx;
                const float v = m[off] * comb_scale + base[off];
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
        for (uint32_t i = 0; i < n_hc * n_hc; i++) o[2u * n_hc + i] = c[i];
    }
}

static float e4m3fn_value(int code) {
    const int exp = (code >> 3) & 0x0f;
    const int mant = code & 0x07;
    if (exp == 0) return (float)mant * 0.001953125f;
    return (1.0f + (float)mant * 0.125f) * exp2f((float)exp - 7.0f);
}

static float e4m3fn_roundtrip(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 448.0f);
    int lo = 0;
    int hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (e4m3fn_value(mid) <= ax) lo = mid;
        else hi = mid - 1;
    }
    int best = lo;
    if (best < 126) {
        const float bd = fabsf(ax - e4m3fn_value(best));
        const float nd = fabsf(ax - e4m3fn_value(best + 1));
        if (nd < bd || (nd == bd && (((best + 1) & 1) == 0) && ((best & 1) != 0))) best++;
    }
    return sign * e4m3fn_value(best);
}

static void fp8_kv_quantize_ref(float *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot) {
    const uint32_t n_nope = head_dim - n_rot;
    for (uint32_t t = 0; t < n_tok; t++) {
        float *row = x + (uint64_t)t * head_dim;
        for (uint32_t off = 0; off < n_nope; off += 64) {
            const uint32_t block = n_nope - off < 64 ? n_nope - off : 64;
            float amax = 0.0f;
            for (uint32_t i = 0; i < block; i++) {
                const float av = fabsf(row[off + i]);
                if (av > amax) amax = av;
            }
            if (amax < 1.0e-4f) amax = 1.0e-4f;
            const float scale = exp2f(ceilf(log2f(amax / 448.0f)));
            for (uint32_t i = 0; i < block; i++) {
                float v = row[off + i] / scale;
                if (v > 448.0f) v = 448.0f;
                if (v < -448.0f) v = -448.0f;
                row[off + i] = e4m3fn_roundtrip(v) * scale;
            }
        }
    }
}

static void rope_tail_ref(float *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    const uint32_t n_nope = head_dim - n_rot;
    float corr0 = 0.0f;
    float corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        const float pi2 = 6.2831853071795864769f;
        corr0 = floorf((float)n_rot * logf((float)n_ctx_orig / (beta_fast * pi2)) / (2.0f * logf(freq_base)));
        corr1 = ceilf((float)n_rot * logf((float)n_ctx_orig / (beta_slow * pi2)) / (2.0f * logf(freq_base)));
        if (corr0 < 0.0f) corr0 = 0.0f;
        if (corr1 > (float)(n_rot - 1)) corr1 = (float)(n_rot - 1);
    }
    for (uint32_t t = 0; t < n_tok; t++) {
        const float pos = (float)(pos0 + t);
        for (uint32_t h = 0; h < n_head; h++) {
            float *tail = x + ((uint64_t)t * n_head + h) * head_dim + n_nope;
            for (uint32_t i = 0; i < n_rot; i += 2) {
                const float theta_extrap = pos * powf(freq_base, -((float)i) / (float)n_rot);
                const float theta_interp = freq_scale * theta_extrap;
                float theta = theta_interp;
                float mscale = attn_factor;
                if (ext_factor != 0.0f) {
                    const float denom = fmaxf(0.001f, corr1 - corr0);
                    float y = ((float)(i / 2) - corr0) / denom;
                    if (y < 0.0f) y = 0.0f;
                    if (y > 1.0f) y = 1.0f;
                    const float ramp_mix = (1.0f - y) * ext_factor;
                    theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
                    mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
                }
                const float c = cosf(theta) * mscale;
                float s = sinf(theta) * mscale;
                if (inverse) s = -s;
                const float x0 = tail[i];
                const float x1 = tail[i + 1];
                tail[i] = x0 * c - x1 * s;
                tail[i + 1] = x0 * s + x1 * c;
            }
        }
    }
}

static void store_raw_kv_batch_ref(float *raw, const float *kv, uint32_t raw_cap, uint32_t pos0, uint32_t n_tokens, uint32_t head_dim) {
    for (uint32_t t = 0; t < n_tokens; t++) {
        const uint32_t row = (pos0 + t) % raw_cap;
        for (uint32_t d = 0; d < head_dim; d++) {
            raw[(uint64_t)row * head_dim + d] = f16_to_f32(f32_to_f16(kv[(uint64_t)t * head_dim + d]));
        }
    }
}

static int check_close(const char *name, const float *want, const float *got, uint32_t count, float tol) {
    for (uint32_t i = 0; i < count; i++) {
        if (fabsf(want[i] - got[i]) > tol) {
            fprintf(stderr, "ascend_fill_smoke: %s mismatch idx %u want=%.9g got=%.9g\n", name, i, want[i], got[i]);
            return 0;
        }
    }
    return 1;
}

static int check_fill(void) {
    const unsigned n = 1024;
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc((uint64_t)n * sizeof(float));
    if (!t) return 0;
    if (!ds4_gpu_tensor_fill_f32(t, 3.25f, n) || !ds4_gpu_synchronize()) {
        ds4_gpu_tensor_free(t);
        return 0;
    }

    float *host = malloc((size_t)n * sizeof(float));
    if (!host || !ds4_gpu_tensor_read(t, 0, host, (uint64_t)n * sizeof(float))) {
        free(host);
        ds4_gpu_tensor_free(t);
        return 0;
    }

    int ok = 1;
    for (unsigned i = 0; i < n; i++) {
        if (fabsf(host[i] - 3.25f) > 0.0f) {
            fprintf(stderr, "ascend_fill_smoke: mismatch at %u: %f\n", i, host[i]);
            ok = 0;
            break;
        }
    }

    free(host);
    ds4_gpu_tensor_free(t);
    return ok;
}

static int check_matmul_f16(void) {
    const uint32_t n_tok = 3;
    const uint32_t in_dim = 17;
    const uint32_t out_dim = 11;
    float *x = malloc((size_t)n_tok * in_dim * sizeof(float));
    uint16_t *w = malloc((size_t)out_dim * in_dim * sizeof(uint16_t));
    float *want = malloc((size_t)n_tok * out_dim * sizeof(float));
    float *got = malloc((size_t)n_tok * out_dim * sizeof(float));
    if (!x || !w || !want || !got) {
        free(x); free(w); free(want); free(got);
        return 0;
    }
    for (uint32_t i = 0; i < n_tok * in_dim; i++) x[i] = ((int32_t)(i % 23) - 11) * 0.0625f;
    for (uint32_t i = 0; i < out_dim * in_dim; i++) w[i] = f32_to_f16(((int32_t)(i % 19) - 9) * 0.03125f);
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t r = 0; r < out_dim; r++) {
            float sum = 0.0f;
            for (uint32_t c = 0; c < in_dim; c++) sum += f16_to_f32(w[(uint64_t)r * in_dim + c]) * x[(uint64_t)t * in_dim + c];
            want[(uint64_t)t * out_dim + r] = sum;
        }
    }

    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc((uint64_t)n_tok * in_dim * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)n_tok * out_dim * sizeof(float));
    int ok = xt && out &&
             ds4_gpu_tensor_write(xt, 0, x, (uint64_t)n_tok * in_dim * sizeof(float)) &&
             ds4_gpu_matmul_f16_tensor(out, w, (uint64_t)out_dim * in_dim * sizeof(uint16_t), 0, in_dim, out_dim, xt, n_tok) &&
             ds4_gpu_synchronize() &&
             ds4_gpu_tensor_read(out, 0, got, (uint64_t)n_tok * out_dim * sizeof(float));
    if (ok) {
        for (uint32_t i = 0; i < n_tok * out_dim; i++) {
            if (fabsf(want[i] - got[i]) > 1e-4f) {
                fprintf(stderr, "ascend_fill_smoke: f16 matmul mismatch idx %u want=%.9g got=%.9g\n", i, want[i], got[i]);
                ok = 0;
                break;
            }
        }
    }
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(xt);
    free(x); free(w); free(want); free(got);
    return ok;
}

static int check_matmul_q8_0(void) {
    const uint32_t n_tok = 2;
    const uint32_t in_dim = 45;
    const uint32_t out_dim = 7;
    const uint32_t blocks = (in_dim + 31) / 32;
    float *x = malloc((size_t)n_tok * in_dim * sizeof(float));
    float *w_f32 = malloc((size_t)out_dim * in_dim * sizeof(float));
    block_q8_0 *w = calloc((size_t)out_dim * blocks, sizeof(*w));
    float *want = malloc((size_t)n_tok * out_dim * sizeof(float));
    float *got = malloc((size_t)n_tok * out_dim * sizeof(float));
    if (!x || !w_f32 || !w || !want || !got) {
        free(x); free(w_f32); free(w); free(want); free(got);
        return 0;
    }
    for (uint32_t i = 0; i < n_tok * in_dim; i++) x[i] = ((int32_t)(i % 29) - 14) * 0.04375f;
    for (uint32_t i = 0; i < out_dim * in_dim; i++) w_f32[i] = ((int32_t)(i % 31) - 15) * 0.0375f;
    for (uint32_t r = 0; r < out_dim; r++) quantize_row_q8_0(w_f32 + (uint64_t)r * in_dim, w + (uint64_t)r * blocks, in_dim);
    matmul_q8_0_ref(want, w, x, in_dim, out_dim, n_tok);

    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc((uint64_t)n_tok * in_dim * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)n_tok * out_dim * sizeof(float));
    int ok = xt && out &&
             ds4_gpu_tensor_write(xt, 0, x, (uint64_t)n_tok * in_dim * sizeof(float)) &&
             ds4_gpu_matmul_q8_0_tensor(out, w, (uint64_t)out_dim * blocks * sizeof(block_q8_0), 0, in_dim, out_dim, xt, n_tok) &&
             ds4_gpu_synchronize() &&
             ds4_gpu_tensor_read(out, 0, got, (uint64_t)n_tok * out_dim * sizeof(float));
    if (ok) {
        for (uint32_t i = 0; i < n_tok * out_dim; i++) {
            if (fabsf(want[i] - got[i]) > 1e-2f) {
                fprintf(stderr, "ascend_fill_smoke: q8_0 matmul mismatch idx %u want=%.9g got=%.9g\n", i, want[i], got[i]);
                ok = 0;
                break;
            }
        }
    }
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(xt);
    free(x); free(w_f32); free(w); free(want); free(got);
    return ok;
}

static int check_rms_norm_plain(void) {
    const uint32_t rows = 5;
    const uint32_t n = 19;
    const uint32_t count = rows * n;
    float *x = malloc((size_t)count * sizeof(float));
    float *want = malloc((size_t)count * sizeof(float));
    float *got = malloc((size_t)count * sizeof(float));
    if (!x || !want || !got) {
        free(x); free(want); free(got);
        return 0;
    }
    for (uint32_t i = 0; i < count; i++) x[i] = ((int32_t)(i % 37) - 18) * 0.03125f;
    rms_norm_ref(want, x, NULL, n, rows, 1.0e-6f);

    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc((uint64_t)count * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)count * sizeof(float));
    int ok = xt && out &&
             ds4_gpu_tensor_write(xt, 0, x, (uint64_t)count * sizeof(float)) &&
             ds4_gpu_rms_norm_plain_rows_tensor(out, xt, n, rows, 1.0e-6f) &&
             ds4_gpu_synchronize() &&
             ds4_gpu_tensor_read(out, 0, got, (uint64_t)count * sizeof(float));
    if (ok) ok = check_close("rms plain", want, got, count, 1e-5f);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(xt);
    free(x); free(want); free(got);
    return ok;
}

static int check_rms_norm_weight(void) {
    const uint32_t rows = 4;
    const uint32_t n = 23;
    const uint32_t count = rows * n;
    float *x = malloc((size_t)count * sizeof(float));
    float *weight = malloc((size_t)n * sizeof(float));
    float *want = malloc((size_t)count * sizeof(float));
    float *got = malloc((size_t)count * sizeof(float));
    if (!x || !weight || !want || !got) {
        free(x); free(weight); free(want); free(got);
        return 0;
    }
    for (uint32_t i = 0; i < count; i++) x[i] = ((int32_t)(i % 41) - 20) * 0.02734375f;
    for (uint32_t i = 0; i < n; i++) weight[i] = 0.75f + (float)(i % 11) * 0.03125f;
    rms_norm_ref(want, x, weight, n, rows, 1.0e-5f);

    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc((uint64_t)count * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)count * sizeof(float));
    int ok = xt && out &&
             ds4_gpu_tensor_write(xt, 0, x, (uint64_t)count * sizeof(float)) &&
             ds4_gpu_rms_norm_weight_rows_tensor(out, xt, weight, (uint64_t)n * sizeof(float), 0, n, rows, 1.0e-5f) &&
             ds4_gpu_synchronize() &&
             ds4_gpu_tensor_read(out, 0, got, (uint64_t)count * sizeof(float));
    if (ok) ok = check_close("rms weight", want, got, count, 1e-5f);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(xt);
    free(x); free(weight); free(want); free(got);
    return ok;
}

static int check_head_rms_norm(void) {
    const uint32_t n_tok = 3;
    const uint32_t n_head = 5;
    const uint32_t head_dim = 13;
    const uint32_t rows = n_tok * n_head;
    const uint32_t count = rows * head_dim;
    float *x = malloc((size_t)count * sizeof(float));
    float *want = malloc((size_t)count * sizeof(float));
    float *got = malloc((size_t)count * sizeof(float));
    if (!x || !want || !got) {
        free(x); free(want); free(got);
        return 0;
    }
    for (uint32_t i = 0; i < count; i++) x[i] = ((int32_t)(i % 43) - 21) * 0.01953125f;
    rms_norm_ref(want, x, NULL, head_dim, rows, 1.0e-6f);

    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc((uint64_t)count * sizeof(float));
    int ok = xt &&
             ds4_gpu_tensor_write(xt, 0, x, (uint64_t)count * sizeof(float)) &&
             ds4_gpu_head_rms_norm_tensor(xt, n_tok, n_head, head_dim, 1.0e-6f) &&
             ds4_gpu_synchronize() &&
             ds4_gpu_tensor_read(xt, 0, got, (uint64_t)count * sizeof(float));
    if (ok) ok = check_close("head rms", want, got, count, 1e-5f);
    ds4_gpu_tensor_free(xt);
    free(x); free(want); free(got);
    return ok;
}

static int check_hc_weighted_sum(void) {
    const uint32_t rows = 5;
    const uint32_t n_embd = 37;
    const uint32_t weight_stride = 24;
    const uint32_t out_count = rows * n_embd;
    const uint32_t residual_count = rows * 4u * n_embd;
    float *residual = malloc((size_t)residual_count * sizeof(float));
    float *weights = malloc((size_t)rows * weight_stride * sizeof(float));
    float *want = malloc((size_t)out_count * sizeof(float));
    float *got = malloc((size_t)out_count * sizeof(float));
    if (!residual || !weights || !want || !got) {
        free(residual); free(weights); free(want); free(got);
        return 0;
    }
    for (uint32_t i = 0; i < residual_count; i++) residual[i] = ((int32_t)(i % 47) - 23) * 0.015625f;
    for (uint32_t i = 0; i < rows * weight_stride; i++) weights[i] = ((int32_t)(i % 29) - 14) * 0.021484375f;
    hc_weighted_sum_ref(want, residual, weights, n_embd, rows, weight_stride);

    ds4_gpu_tensor *rt = ds4_gpu_tensor_alloc((uint64_t)residual_count * sizeof(float));
    ds4_gpu_tensor *wt = ds4_gpu_tensor_alloc((uint64_t)rows * weight_stride * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)out_count * sizeof(float));
    int ok = rt && wt && out &&
             ds4_gpu_tensor_write(rt, 0, residual, (uint64_t)residual_count * sizeof(float)) &&
             ds4_gpu_tensor_write(wt, 0, weights, (uint64_t)rows * weight_stride * sizeof(float)) &&
             ds4_gpu_hc_weighted_sum_split_tensor(out, rt, wt, n_embd, 4) &&
             ds4_gpu_synchronize() &&
             ds4_gpu_tensor_read(out, 0, got, (uint64_t)out_count * sizeof(float));
    if (ok) ok = check_close("hc weighted sum", want, got, out_count, 1e-6f);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(wt);
    ds4_gpu_tensor_free(rt);
    free(residual); free(weights); free(want); free(got);
    return ok;
}

static int check_hc_split_sinkhorn(void) {
    const uint32_t rows = 6;
    const uint32_t mix_hc = 24;
    const uint32_t count = rows * mix_hc;
    float scale[3] = {0.25f, -0.375f, 0.1875f};
    float base[24];
    float *mix = malloc((size_t)count * sizeof(float));
    float *want = malloc((size_t)count * sizeof(float));
    float *got = malloc((size_t)count * sizeof(float));
    if (!mix || !want || !got) {
        free(mix); free(want); free(got);
        return 0;
    }
    for (uint32_t i = 0; i < 24; i++) base[i] = ((int32_t)(i % 13) - 6) * 0.0625f;
    for (uint32_t i = 0; i < count; i++) mix[i] = ((int32_t)(i % 31) - 15) * 0.0234375f;
    hc_split_ref(want, mix, scale, base, rows, 3, 1.0e-6f);

    ds4_gpu_tensor *mt = ds4_gpu_tensor_alloc((uint64_t)count * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)count * sizeof(float));
    uint8_t model[27 * sizeof(float)];
    memcpy(model, scale, sizeof(scale));
    memcpy(model + sizeof(scale), base, sizeof(base));
    int ok = mt && out &&
             ds4_gpu_tensor_write(mt, 0, mix, (uint64_t)count * sizeof(float)) &&
             ds4_gpu_hc_split_sinkhorn_tensor(out, mt, model, sizeof(model), 0, sizeof(scale), 4, 3, 1.0e-6f) &&
             ds4_gpu_synchronize() &&
             ds4_gpu_tensor_read(out, 0, got, (uint64_t)count * sizeof(float));
    if (ok) ok = check_close("hc split", want, got, count, 2e-3f);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(mt);
    free(mix); free(want); free(got);
    return ok;
}

static int check_fp8_kv_quantize(void) {
    const uint32_t n_tok = 5;
    const uint32_t head_dim = 93;
    const uint32_t n_rot = 16;
    const uint32_t count = n_tok * head_dim;
    float *x = malloc((size_t)count * sizeof(float));
    float *want = malloc((size_t)count * sizeof(float));
    float *got = malloc((size_t)count * sizeof(float));
    if (!x || !want || !got) {
        free(x); free(want); free(got);
        return 0;
    }
    for (uint32_t i = 0; i < count; i++) x[i] = ((int32_t)(i % 67) - 33) * 0.01953125f;
    x[7] = 0.000001f;
    x[89] = -7.25f;
    memcpy(want, x, (size_t)count * sizeof(float));
    fp8_kv_quantize_ref(want, n_tok, head_dim, n_rot);

    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc((uint64_t)count * sizeof(float));
    int ok = xt &&
             ds4_gpu_tensor_write(xt, 0, x, (uint64_t)count * sizeof(float)) &&
             ds4_gpu_dsv4_fp8_kv_quantize_tensor(xt, n_tok, head_dim, n_rot) &&
             ds4_gpu_synchronize() &&
             ds4_gpu_tensor_read(xt, 0, got, (uint64_t)count * sizeof(float));
    if (ok) ok = check_close("fp8 kv", want, got, count, 1e-6f);
    ds4_gpu_tensor_free(xt);
    free(x); free(want); free(got);
    return ok;
}

static int check_rope_tail(void) {
    const uint32_t n_tok = 4;
    const uint32_t n_head = 3;
    const uint32_t head_dim = 14;
    const uint32_t n_rot = 8;
    const uint32_t count = n_tok * n_head * head_dim;
    float *x = malloc((size_t)count * sizeof(float));
    float *want = malloc((size_t)count * sizeof(float));
    float *got = malloc((size_t)count * sizeof(float));
    if (!x || !want || !got) {
        free(x); free(want); free(got);
        return 0;
    }
    for (uint32_t i = 0; i < count; i++) x[i] = ((int32_t)(i % 53) - 26) * 0.017578125f;
    memcpy(want, x, (size_t)count * sizeof(float));
    rope_tail_ref(want, n_tok, n_head, head_dim, n_rot, 7, 4096, false, 10000.0f, 0.25f, 1.0f, 0.875f, 32.0f, 1.0f);

    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc((uint64_t)count * sizeof(float));
    int ok = xt &&
             ds4_gpu_tensor_write(xt, 0, x, (uint64_t)count * sizeof(float)) &&
             ds4_gpu_rope_tail_tensor(xt, n_tok, n_head, head_dim, n_rot, 7, 4096, false, 10000.0f, 0.25f, 1.0f, 0.875f, 32.0f, 1.0f) &&
             ds4_gpu_synchronize() &&
             ds4_gpu_tensor_read(xt, 0, got, (uint64_t)count * sizeof(float));
    if (ok) ok = check_close("rope tail", want, got, count, 1e-6f);
    ds4_gpu_tensor_free(xt);
    free(x); free(want); free(got);
    return ok;
}

static int check_store_raw_kv(void) {
    const uint32_t raw_cap = 4;
    const uint32_t head_dim = 17;
    const uint32_t n_tokens = 4;
    const uint32_t raw_count = raw_cap * head_dim;
    const uint32_t kv_count = n_tokens * head_dim;
    float *raw = malloc((size_t)raw_count * sizeof(float));
    float *kv = malloc((size_t)kv_count * sizeof(float));
    float *want = malloc((size_t)raw_count * sizeof(float));
    float *got = malloc((size_t)raw_count * sizeof(float));
    if (!raw || !kv || !want || !got) {
        free(raw); free(kv); free(want); free(got);
        return 0;
    }
    for (uint32_t i = 0; i < raw_count; i++) raw[i] = -1000.0f - (float)i;
    for (uint32_t i = 0; i < kv_count; i++) kv[i] = ((int32_t)(i % 71) - 35) * 0.013671875f;
    kv[3] = 0.33325195f;
    kv[29] = -6.75f;
    memcpy(want, raw, (size_t)raw_count * sizeof(float));
    store_raw_kv_batch_ref(want, kv, raw_cap, 3, n_tokens, head_dim);

    ds4_gpu_tensor *raw_t = ds4_gpu_tensor_alloc((uint64_t)raw_count * sizeof(float));
    ds4_gpu_tensor *kv_t = ds4_gpu_tensor_alloc((uint64_t)kv_count * sizeof(float));
    int ok = raw_t && kv_t &&
             ds4_gpu_tensor_write(raw_t, 0, raw, (uint64_t)raw_count * sizeof(float)) &&
             ds4_gpu_tensor_write(kv_t, 0, kv, (uint64_t)kv_count * sizeof(float)) &&
             ds4_gpu_store_raw_kv_batch_tensor(raw_t, kv_t, raw_cap, 3, n_tokens, head_dim) &&
             ds4_gpu_synchronize() &&
             ds4_gpu_tensor_read(raw_t, 0, got, (uint64_t)raw_count * sizeof(float));
    if (ok) ok = check_close("store raw kv batch", want, got, raw_count, 0.0f);

    if (ok) {
        const uint32_t one_count = head_dim;
        for (uint32_t i = 0; i < raw_count; i++) raw[i] = 500.0f + (float)i;
        for (uint32_t i = 0; i < one_count; i++) kv[i] = ((int32_t)(i % 19) - 9) * 0.021484375f;
        memcpy(want, raw, (size_t)raw_count * sizeof(float));
        store_raw_kv_batch_ref(want, kv, raw_cap, 2, 1, head_dim);
        ok = ds4_gpu_tensor_write(raw_t, 0, raw, (uint64_t)raw_count * sizeof(float)) &&
             ds4_gpu_tensor_write(kv_t, 0, kv, (uint64_t)one_count * sizeof(float)) &&
             ds4_gpu_store_raw_kv_tensor(raw_t, kv_t, raw_cap, 2, head_dim) &&
             ds4_gpu_synchronize() &&
             ds4_gpu_tensor_read(raw_t, 0, got, (uint64_t)raw_count * sizeof(float));
        if (ok) ok = check_close("store raw kv single", want, got, raw_count, 0.0f);
    }

    ds4_gpu_tensor_free(kv_t);
    ds4_gpu_tensor_free(raw_t);
    free(raw); free(kv); free(want); free(got);
    return ok;
}

static int check_quantize(void) {
    const uint32_t rows = 2;
    const uint32_t cols = 512;
    const uint32_t blocks = rows * (cols / QK_K);
    float *host = malloc((size_t)rows * cols * sizeof(float));
    block_q8_K *want = calloc(blocks, sizeof(*want));
    block_q8_K *got = calloc(blocks, sizeof(*got));
    if (!host || !want || !got) {
        free(host);
        free(want);
        free(got);
        return 0;
    }
    for (uint32_t i = 0; i < rows * cols; i++) {
        host[i] = ((int32_t)(i % 97) - 48) * 0.03125f;
    }
    host[17] = -9.0f;
    host[411] = 7.5f;

    for (uint32_t r = 0; r < rows; r++) {
        quantize_row_q8_K(host + (uint64_t)r * cols, want + (uint64_t)r * (cols / QK_K), cols);
    }

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)rows * cols * sizeof(float));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc((uint64_t)blocks * sizeof(block_q8_K));
    int ok = x && q &&
             ds4_gpu_tensor_write(x, 0, host, (uint64_t)rows * cols * sizeof(float)) &&
             ds4_gpu_ascend_quantize_q8_k_tensor(q, x, rows, cols) &&
             ds4_gpu_synchronize() &&
             ds4_gpu_tensor_read(q, 0, got, (uint64_t)blocks * sizeof(block_q8_K));
    if (ok) {
        for (uint32_t b = 0; b < blocks; b++) {
            if (fabsf(want[b].d - got[b].d) > 1e-7f) {
                fprintf(stderr, "ascend_fill_smoke: q8 d mismatch block %u want=%.9g got=%.9g\n", b, want[b].d, got[b].d);
                ok = 0;
                break;
            }
            for (uint32_t i = 0; i < QK_K; i++) {
                if (want[b].qs[i] != got[b].qs[i]) {
                    fprintf(stderr, "ascend_fill_smoke: q8 qs mismatch block %u idx %u want=%d got=%d\n", b, i, want[b].qs[i], got[b].qs[i]);
                    ok = 0;
                    break;
                }
            }
            if (!ok) break;
            for (uint32_t i = 0; i < QK_K / 16; i++) {
                if (want[b].bsums[i] != got[b].bsums[i]) {
                    fprintf(stderr, "ascend_fill_smoke: q8 bsums mismatch block %u idx %u want=%d got=%d\n", b, i, want[b].bsums[i], got[b].bsums[i]);
                    ok = 0;
                    break;
                }
            }
            if (!ok) break;
        }
    }

    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(x);
    free(host);
    free(want);
    free(got);
    return ok;
}

int main(void) {
    if (!ds4_gpu_init()) return 1;

    int ok = check_fill() && check_matmul_f16() && check_matmul_q8_0() &&
             check_rms_norm_plain() && check_rms_norm_weight() && check_head_rms_norm() &&
             check_hc_split_sinkhorn() && check_hc_weighted_sum() && check_fp8_kv_quantize() &&
             check_rope_tail() && check_store_raw_kv() && check_quantize();
    if (ok) printf("ascend_fill_smoke: ok\n");

    ds4_gpu_cleanup();
    return ok ? 0 : 1;
}
