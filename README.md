# DS4 Ascend backend

这是 DwarfStar 4（DS4）的 Ascend 后端 bring-up 仓库，目标是在 Atlas 300I Duo / Ascend 310P3 上运行 **DeepSeek V4 Flash** 的 q2-imatrix GGUF。

本仓库不是通用 GGUF 推理框架，也不是完整的 Ascend 通用运行时。当前实现只面向本项目使用的 DeepSeek V4 Flash IQ2/Q2 量化布局，用于验证 Ascend no-host 执行路径、恢复语义正确性，并继续推进性能优化。

详细状态和实现 notes 见 [ASCEND.md](ASCEND.md)。

## 基于原 DS4 项目的主要改动

这个仓库基于原始 DS4 项目增加和调整了 Ascend 相关实现，主要改动包括：

- 增加 `ds4_ascend.c` 中的 Ascend runtime、device tensor、stream、weight cache 和 expert cache 管理。
- 增加 Atlas 双卡 expert sharding：当前把 256 个 routed experts 按 `128/128` 分到两个 Ascend device。
- 补齐 no-host 执行路径，尽量让 graph prefill、decode、MoE、compressor、attention、RoPE、HC split/expand、output head 等关键步骤在 Ascend device 上执行。
- 增加 `DS4_ASCEND_NO_HOST_FALLBACK=1` correctness gate；如果某个 primitive 没有 Ascend device 实现，程序会失败，而不是静默回退到 CPU。
- 在 `ds4_ascend_kernels.cpp` 中增加一批 AscendC correctness-first kernels，包括 MoE IQ2/Q2、SwiGLU、compressor prefill、mixed attention prefill、RoPE tail、HC split/expand 和 output HC weights 等。
- 扩展 `ds4.c` 的 graph diagnostics，对 Ascend 和 CPU reference 的中间 stage 做 diff，便于定位语义错误。
- 修复 bring-up 过程中发现的 correctness 问题，包括 expert cache / dense cache 统计导致的小权重被错误淘汰，以及 RoPE 临时 table 过早释放导致的异步 kernel 读悬空指针。
- 新增 `ASCEND.md` 和本 README，用中文记录 Atlas/Ascend 构建、运行、验证、性能现状和后续优化计划。

这些改动的当前目标是先让 DeepSeek V4 Flash 在 Ascend 上 no-host 跑通并恢复短路径语义正确性；性能仍在后续优化阶段。

## 当前状态

Ascend 路径已经可以在 `DS4_ASCEND_NO_HOST_FALLBACK=1` 下运行当前 DeepSeek V4 Flash q2-imatrix 模型。

已经验证的结果：

- `--metal-graph-test --ascend -n 1 -p Hello` 可以成功退出。
- layer-0 graph diagnostics 已恢复到当前 smoke baseline 范围：
  - `hc_post_w=0.0001559`
  - `hc_comb=0.000204027`
  - `q_rope=0.00925317`
  - `after_attn_hc=0.00128685`
  - `logits=0.0489826`
- no-host greedy `Hello -n8` 曾完成并生成可读文本：
  ```text
  Hello! How can I assist you today
  ```
- 后续一次重复 `Hello -n8` 在 prefill `layer 7/43` 被外部暂停，退出码为 137；这不是模型 correctness 失败，也不是完整验证结果。

当前结论：Ascend no-host 路径已经能在短 deterministic smoke 中到达可读输出，但还不能声称覆盖了长上下文、多 prompt、采样、batch、server 或性能优化后的回归稳定性。

## 模型要求

当前验证使用的模型文件路径为：

```text
models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
```

当前 Ascend 实现假设以下 DeepSeek V4 Flash 量化布局：

- routed expert gate/up weights: `IQ2_XXS`
- routed expert down weights: `Q2_K`
- activation / intermediate quantization: `Q8_K`
- routed expert dimensions: `4096 -> 2048 -> 4096`
- 256 routed experts，每 token 选择 6 个 expert

不要把当前 Ascend backend 当作任意 DeepSeek/GGUF 文件的通用加载器使用。

## 构建步骤

在 Atlas 300I Duo / Ascend 310P3 机器上克隆仓库：

```sh
git clone https://github.com/donge/ds4-ascend.git
cd ds4-ascend
```

确认模型文件存在：

```sh
ls models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
```

构建 Ascend 版本：

```sh
make ds4_ascend
```

远端验证过的路径是：

```text
/home/sw/ds4-ascend
```

如果 CANN 9.0 链接环境需要显式库路径，可使用：

```sh
make ds4_ascend ASCEND_LDLIBS='-lm -pthread \
  -L/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64 \
  -L/usr/local/Ascend/driver/lib64/driver \
  -Wl,-rpath,/usr/local/Ascend/driver/lib64/driver \
  -Wl,-rpath,/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64 \
  -lruntime -lascendcl -lascend_hal -lprofapi -lstdc++'
```

## 运行 graph correctness smoke

先运行最短 graph correctness 验证：

```sh
DS4_ASCEND_VISIBLE_DEVICES=0,1 DS4_ASCEND_NO_HOST_FALLBACK=1 \
./ds4_ascend --metal-graph-test --ascend \
  -m models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  -c 512 -n 1 --temp 0 --nothink -p Hello
```

成功标准：

- exit code 为 0。
- 没有 `Ascend host fallback disabled: ...`。
- 没有 primitive missing / NaN / runtime error。
- graph diff 与当前 baseline 同量级，例如 `logits` 约 `0.05`。

## 运行短生成 smoke

graph smoke 通过后，再运行短生成：

```sh
DS4_ASCEND_VISIBLE_DEVICES=0,1 DS4_ASCEND_NO_HOST_FALLBACK=1 \
./ds4_ascend --ascend \
  -m models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  -c 512 -n 8 --temp 0 --nothink -p Hello
```

预期结果是完成全部 43 个 prefill layer，生成可读英文文本，并正常退出。

为了避免 SSH 或本地后台任务中断导致远端进程被杀，长时间验证建议使用独立日志或 `nohup` 包装，例如：

```sh
nohup sh -c '
DS4_ASCEND_VISIBLE_DEVICES=0,1 DS4_ASCEND_NO_HOST_FALLBACK=1 \
./ds4_ascend --ascend \
  -m models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  -c 512 -n 8 --temp 0 --nothink -p Hello
' > /tmp/ds4_ascend_n8_hello.log 2>&1 &
```

查看日志：

```sh
tail -f /tmp/ds4_ascend_n8_hello.log
```

## 当前性能指标

当前 Ascend backend 还处于 correctness-first 阶段，不是性能完成状态。

已观察到的性能特征：

- 两个可见 Ascend device：`ASCEND_RT_VISIBLE_DEVICES=0,1`。
- expert split: `128/128`。
- routed expert sharding 每次进程启动会准备约 `72.56 GiB` q2 tensors。
- startup sharding 目前约 `135-146s`。
- layer-major prefill 即使是很短 prompt，也会跑完整 43 层。
- correctness-first prefill attention profile 约 `97-105s/layer`。
- FFN profile 约 `23s/layer`。
- 很短 run 中打印的 `0.00 t/s` 不能代表 steady-state decode 性能。

因此当前性能瓶颈主要是：

1. 启动阶段 expert sharding 成本过高。
2. prefill attention kernel 是直接 correctness port，尚未优化。
3. FFN/MoE orchestration 仍有大量 scratch/copy/launch 开销。
4. RoPE 和临时 buffer 仍需要进一步复用，减少分配和同步成本。

## 后续计划

### Correctness plan

1. 用更可靠的远端 `nohup`/独立日志方式重跑 `Hello -n8`。
2. `Hello -n8` 通过后，继续跑 `Hello -n16`。
3. `Hello -n16` 通过后，继续跑 `Hello -n32`。
4. `Hello -n32` 通过后，跑一个 prompt variation，例如：
   ```text
   Write one short sentence about the sky.
   ```
   使用 `-n 16`。
5. 如果出现语义崩坏、fallback、NaN、assert 或 device runtime error，停止更长验证，回到 graph dump / diff 诊断。

### Optimization plan

1. 优化 startup expert sharding，避免每次启动重复完整准备 72.56 GiB expert tensors。
2. 重写 prefill attention，从 correctness-first direct kernel 转向更向量化或 online 的实现。
3. 优化 FFN/MoE orchestration，减少 scratch allocation、cross-device copy 和 kernel launch count。
4. 复用 RoPE table 和其他短生命周期临时 buffer。
5. 在 attention 和 FFN 不再是绝对瓶颈后，继续 profile compressor prefill、output head、router/MoE kernels。
6. 如继续使用 CANN 9.0，考虑把 Ascend link dependency 固化到 `Makefile`。

## 常用环境变量

```sh
export DS4_ASCEND_VISIBLE_DEVICES=0,1
export DS4_ASCEND_NO_HOST_FALLBACK=1
```

`DS4_ASCEND_NO_HOST_FALLBACK=1` 是 correctness gate：如果某个 primitive 还没有 Ascend device 实现，程序应该失败，而不是静默把 tensor 读回 host 执行。

## 重要文件

- `ds4.c`
  - graph generation、graph diagnostics、CPU reference 和 diff 输出。
- `ds4_ascend.c`
  - Ascend runtime、tensor、stream、weight/expert cache、no-host fallback gate 和高层 device orchestration。
- `ds4_ascend_kernels.cpp`
  - AscendC kernels 和 launch wrappers。
- `ASCEND.md`
  - 更详细的 Ascend backend 状态、实现约束、验证方式和优化方向。
