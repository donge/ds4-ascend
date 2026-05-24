# Ascend backend notes

This document summarizes the current Atlas 300I Duo / Ascend 310P3 work for the DS4 DeepSeek V4 Flash backend. The implementation is deliberately narrow: it targets the current IQ2/Q2 GGUF used by this repository, not a general Ascend runtime or arbitrary GGUF support.

## Current status

The Ascend path can now run the current DeepSeek V4 Flash IQ2/Q2 model on Atlas
300I Duo / Ascend 310P3 with host fallback disabled:

```sh
DS4_ASCEND_VISIBLE_DEVICES=0,1 DS4_ASCEND_NO_HOST_FALLBACK=1 \
./ds4_ascend --ascend \
  -m models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  -c 512 --temp 0 --nothink -p 'Hello'
```

Validated runnable results so far:

- `--metal-graph-test --ascend -n 1 -p Hello` exits with status 0.
- Layer-0 graph diagnostics are back in the expected smoke range:
  - `hc_post_w=0.0001559`
  - `hc_comb=0.000204027`
  - `q_rope=0.00925317`
  - `after_attn_hc=0.00128685`
  - `logits=0.0489826`
- A no-host greedy `Hello -n8` smoke completed previously and produced readable
  text: `Hello! How can I assist you today`.
- A later repeat of `Hello -n8` was paused externally during prefill at layer
  7/43 and exited 137, so it is not a model correctness failure or a completed
  validation result.

This proves the no-host Ascend execution path can reach semantically readable
output for a short deterministic smoke. It does not yet prove broad stability for
longer contexts, multiple prompts, sampling, batching, server usage, or future
performance-optimized kernels.

Observed startup behavior on the Atlas server:

- Two visible devices: `ASCEND_RT_VISIBLE_DEVICES=0,1`.
- Expert split: `128/128` across the two devices.
- Routed expert sharding prepares about `72.56 GiB` of q2 tensors per process startup.
- Startup sharding currently takes about `135-146s` before graph prefill begins.

## Design constraints

The Ascend backend follows the existing DS4 backend API in `ds4_gpu.h`; the public interface is not changed for Ascend-specific work.

The current implementation assumes the repository's DeepSeek V4 Flash quantization mix:

- Routed expert gate/up weights: `IQ2_XXS`.
- Routed expert down weights: `Q2_K`.
- Activation and intermediate quantization: `Q8_K`.
- Routed expert dimensions: `4096 -> 2048 -> 4096`.
- 256 routed experts, with 6 selected experts per token.

`DS4_ASCEND_NO_HOST_FALLBACK=1` is the correctness gate. If an Ascend primitive is not implemented on device, the run must fail instead of silently reading tensors back to the host.

## Device memory and sharding

`ds4_ascend.c` keeps per-device weight and expert caches. Routed expert tensors are cached in sharded form instead of replicating all experts on every device. For the current two-device Atlas target, experts are split into `[0,128)` and `[128,256)`.

The MoE path uses the existing expert cache metadata to find gate, up, and down expert shards. It validates the quantization type and current-model dimensions before using the device path. The implementation intentionally avoids full expert replication because the q2 expert cache is already about `72.56 GiB`.

Cross-device data movement is handled with device-to-device copies. On the tested Atlas system, peer access reports unavailable and `ACL_MEMCPY_INTER_DEVICE_TO_DEVICE` failed, while `aclrtMemcpyAsync(..., ACL_MEMCPY_DEVICE_TO_DEVICE, ...)` works for cross-device copies when launched on the destination device stream and synchronized.

## No-host frontiers cleared

The no-host bring-up progressed by replacing each host fallback frontier with an Ascend device path. The major cleared frontiers are:

1. `routed_moe_iq2_q2`
   - Uses sharded expert caches across the two devices.
   - Quantizes token activations to Q8_K on device.
   - Computes IQ2_XXS gate/up dot products, SwiGLU mid activations, Q8_K mid quantization, Q2_K down projections, remote shard merges, and 6-expert accumulation on device.
   - Uses device IQ2 lookup tables rather than host-side IQ2 table access.

2. `swiglu`
   - Added a small AscendC elementwise SwiGLU kernel.
   - Shared expert swiglu paths try the device kernel before host fallback.

3. `compressor_prefill`
   - Added AscendC compressor row setup and prefill pooling kernels.
   - Reuses existing device RMSNorm, RoPE tail, and FP8 KV quantization paths.

4. `compressor_prefill_state_ratio4`
   - Refreshes compressor state on device using device memset, fill, and compressor row setup.

5. `attention_prefill_static_mixed`
   - Added a correctness-first mixed raw/compressed attention prefill kernel.
   - This is a direct implementation, not the final optimized online/vectorized form.

6. `output_hc_weights`
   - Added an AscendC sigmoid weight kernel for the final output head.
   - Loads scale and base tensors through the existing device weight cache path.

## Important implementation files

- `ds4_ascend.c`
  - Ascend runtime setup, tensor allocation, streams, weight caches, fallback tracing, and high-level device orchestration.
  - Contains the device paths that decide whether an operation can run without host fallback.
  - Contains the cross-device copy helper used by sharded MoE.

- `ds4_ascend_kernels.cpp`
  - AscendC kernels and launch wrappers.
  - Includes current correctness-first kernels for MoE pieces, SwiGLU, compressor prefill, mixed attention prefill, and output HC weights.

- `ds4_cuda.cu`
  - Reference for several GPU algorithms and formulas during Ascend porting.

- `ds4_iq2_tables_cuda.inc`
  - Source of IQ2_XXS table semantics used by the Ascend MoE implementation.

## Build notes

On the Atlas server the build currently needs `libprofapi` in addition to the usual Ascend libraries because the CANN 9.0 `ccec` output references profiling symbols.

Example build command used on Atlas:

```sh
make ds4_ascend ASCEND_LDLIBS='-lm -pthread \
  -L/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64 \
  -L/usr/local/Ascend/driver/lib64/driver \
  -Wl,-rpath,/usr/local/Ascend/driver/lib64/driver \
  -Wl,-rpath,/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64 \
  -lruntime -lascendcl -lascend_hal -lprofapi -lstdc++'
```

The build has been validated on the remote Atlas path `/home/sw/ds4-ascend`.

## Validation workflow

Use `DS4_ASCEND_NO_HOST_FALLBACK=1` to ensure no primitive silently falls back to CPU:

```sh
DS4_ASCEND_VISIBLE_DEVICES=0,1 DS4_ASCEND_NO_HOST_FALLBACK=1 \
./ds4_ascend --ascend \
  -m models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  -c 512 -n 32 --temp 0 --nothink -p 'Hello'
```

A successful correctness/progress run should:

- initialize two Ascend devices,
- prepare routed expert sharding,
- complete `gpu prefill layer 43/43`,
- emit generated tokens,
- print `prefill` and `generation` throughput,
- exit with status 0,
- never print `Ascend host fallback disabled: ...`.

`-n 1` is useful for confirming that execution reaches output generation, but its `generation t/s` should not be used for performance decisions. Use `-n 32` or `-n 64` for steadier decode measurement.

## Current performance interpretation

For short prompts, the visible wall-clock bottleneck is currently startup plus
prefill, not steady-state decode:

- Expert sharding startup prepares about `72.56 GiB` of q2 tensors and has been
  observed at roughly `135-146s`.
- Layer-major prefill still runs all 43 layers even for a tiny prompt.
- Profiling before the first optimization pass showed the largest costs in Q8_0
  projections: attention output projection, attention `q_b`, and shared expert
  projections.
- The first validated optimization prequantizes each Q8_0 activation once and
  reuses the int8 activation plus float scale for all output rows. This keeps the
  layer-0 graph diff at the established baseline while reducing graph smoke wall
  time from roughly `207-215s` to `161s` on the tested Atlas system.
- Because many kernels are still direct correctness ports, printed `0.00 t/s`
  values on very short runs are not useful steady-state throughput numbers.

Decode throughput must be interpreted from longer generation runs after startup
and prefill complete.

### Optimization history

The following optimizations have been validated and committed in order:

1. **Q8_0 shared activation prequantization** (`2fc0819`): shared gate/up
   projection activations are quantized once and reused across all output rows
   instead of re-quantizing per row.

2. **Attention output Q8_0 prequantization** (`7f1d55b`): attention output
   projection activation is quantized once and reused across all rank output rows,
   eliminating redundant per-row quantization.

3. **Routed MoE gate/up/mid return copy merge** (`c702f4f`): gate, up, and mid
   remote-to-canonical copies are merged into a single combined cross-device
   transfer, reducing stream synchronization overhead by roughly 4 syncs × 43
   layers per decode step.

4. **Routed MoE forward bundle copy merge** (`4155212`): xq activation, selected
   expert indices, and routing weights are packed into a single canonical buffer
   and transferred in one cross-device copy, reducing forward D2D syncs by 2 × 43
   layers per decode step.

### Hotspot analysis conclusion

Wall-clock timing instrumented at each MoE phase (43 layers of prefill) shows:

| Phase | Time | Share |
|-------|------|-------|
| `gum_copy` (wait for remote gate_up_mid kernel) | 439.5 s | 56 % |
| `down_copy` (wait for remote down kernel) | 166.3 s | 21 % |
| `fwd_copy` (forward D2D transfer) | 163.7 s | 21 % |
| All other phases | 16.8 s | 2 % |

98 % of decode time is AICore kernel execution on the Ascend 310P3. The kernels
(`ds4_moe_gate_up_mid_iq2_q8`, `ds4_moe_down_q2_q8`) are tagged
`__attribute__((aiv))` but are implemented as scalar loops reading directly from
global memory. The 310P3 has 256-bit vector hardware that is not yet used.

The dominant cost is global-memory (GM) access latency rather than arithmetic:
each dot product accesses roughly 5 KB from GM with random-access lookup patterns.
Estimated AICore utilization is well under 0.01 % of the 128 TOPS peak.

Further optimization of this bottleneck requires either AscendC SIMD vectorization
of the IQ2_XXS and Q2_K dot products (complex due to lookup-table structure) or a
fundamentally different quantization format that maps better to the hardware.
Cross-device copy overhead has now been reduced to a secondary cost and does not
represent the primary optimization target.

## Known non-goals for this stage

- No support for arbitrary models, arbitrary GGUFs, or arbitrary quantization mixes.
- No full expert replication across devices.
- No generic Ascend runtime abstraction.
- No final performance-tuned fused MoE or attention implementation yet.
- No claim that one-token generation throughput represents steady-state speed.

## Next work

Correctness plan:

1. Re-run `Hello -n8` with a remote `nohup`-style wrapper so the validation is not
   killed when the controlling SSH/background task is stopped.
2. If `Hello -n8` passes, extend to `Hello -n16` and `Hello -n32` with
   `DS4_ASCEND_NO_HOST_FALLBACK=1`.
3. If those pass, run one short prompt variation, for example
   `Write one short sentence about the sky.` with `-n 16`.
4. Stop on the first semantic, fallback, NaN, or runtime failure and return to
   graph diagnostics.

Optimization notes:

1. Cross-device copy overhead has been reduced by four optimization passes (see
   history above). Further D2D savings are marginal — the remaining copies are
   load-bearing and cannot be avoided without a topology change.
2. AICore kernel compute is now the dominant bottleneck (98 % of decode time).
   Vectorizing `ds4_dot_iq2_xxs_q8_K` and `ds4_dot_q2_K_q8_K` with AscendC
   `LocalTensor` SIMD APIs is the primary remaining optimization lever, but
   requires nontrivial kernel rewriting due to IQ2_XXS lookup-table structure.
3. Startup cost (expert sharding ~140s) could be reduced by persisting sharded
   tensors to disk or by reducing the sharding granularity.
4. Replace correctness-first prefill attention with a more vectorized or online
   implementation once decode throughput is the governing constraint.
