# Local LLM Serving & Benchmarking on a Single Workstation (llama.cpp)

Central, project-agnostic reference: how to serve and benchmark local LLMs with llama.cpp on
one machine, with everything below measured on the hardware in §2. Structure is deliberately
split into **common parts** (hardware facts, measured laws, benchmarking strategy — §2–§4)
and **extensible parts** (one subsection per goal in §1, one per model in §5). Add new goals
or models as new subsections; do not rewrite the common parts.

Projects reference goals here by **goal ID** and link relatively (`../llm-benchmarks/README.md`);
each project keeps only its own `SERVING.md` (goal → model → serve command → port).

---

## 1. Goal registry (one `###` section per workload)

| Goal ID            | Shape                                                                                       | Metrics              | Best known model                                   |
| ------------------ | ------------------------------------------------------------------------------------------- | -------------------- | -------------------------------------------------- |
| `notebook-codegen` | `--parallel 1`; prompt 0.5–2k tok; output 100–2048 tok (typ. 100–500); temp 0, thinking off | tg primary, pp floor | `unsloth-Qwen3-Coder-30B-A3B-Instruct-Q4_K_M` (§5) |

Each goal section states: concurrency (`--parallel`), prompt token range, max output tokens,
sampling settings, endpoint, and which metric decides "good".

### Goal `notebook-codegen`: interactive single-prompt code generation in notebooks

- One request at a time → `--parallel 1`.
- Prompts ~0.5–2k tokens; outputs typically 100–500 tokens, capped at `max_tokens` 1024–2048;
  `temperature=0`, thinking disabled.
- Served OpenAI-compatible at `http://127.0.0.1:8080/v1`; clients probe `GET /v1/models`,
  `GET /props`, `POST /tokenize` at startup.
- Metric: decode speed **tg (t/s) is primary**, prefill **pp (t/s) is a floor**.
  Interactive latency ≈ prompt_tokens/pp + output_tokens/tg.
- Measured 2026-11-14: bench r3 pp512 102 / pp2048 132 / tg256 11.4 / tg512 10.3 t/s;
  live 1209-tok prompt + 128 gen = 20.5 s cold / 11.4 s warm wall, pp 124.7 t/s, tg 11.8 t/s.
  Raw: `results/bench-2026-11-14-unsloth-Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.jsonl`

---

## 2. Hardware (measured)

| Component       | Fact                                                                                                                   |
| --------------- | ---------------------------------------------------------------------------------------------------------------------- |
| GPU             | NVIDIA Quadro P1000, 4095 MiB VRAM, Pascal (CC 6.1), **no tensor cores**                                               |
| CPU             | Intel Xeon E-2176M, 6C/12T @ 2.7 GHz, AVX2, 12 MB L3 — **cache-limited**                                               |
| RAM             | 32 GB DDR4 dual-channel (~35–40 GB/s effective bandwidth)                                                              |
| Storage         | 1 TB NVMe (model load from HF cache ≈ 30–60 s)                                                                         |
| OS / shell      | Windows 11, PowerShell 7 (`pwsh`)                                                                                      |
| Inference stack | llama.cpp build 11669 (0.7.0-dev, commit 620d2225c); backends `ggml-cuda.dll`, `ggml-cpu-haswell.dll` (AVX2, REPACK=1) |
| Entry points    | `llama serve` (server) and `llama bench` (**dispatcher** — there is no separate `llama-bench` binary)                  |

---

## 3. Measured hardware laws (binding — do not re-measure)

1. **Context length, not model size, is the decode-speed lever** on a cache-limited CPU:
   the KV cache grows linearly and becomes latency-bound (non-sequential access). Measured:
   an 8B dense model dropped 4.2 → 2.9 t/s when ctx went `-c 8192` → `16384` (pure CPU at
   16384: 1.9 t/s). Not a flash-attention issue — pure CPU (no FA at all) was slower still.
2. **Dense models cap at ~8B for interactive use on this box**: a 14B dense model lands at
   ~1.5–2 t/s no matter the placement. Fine for overnight batch, never interactive.
3. **Split offload balances the two chips**: prefill saturates the GPU (100% GPU / 0% CPU),
   decode uses both (~50% GPU / 100% CPU). A 4B fully offloads; an 8B at 22/37 layers splits.
4. **Judge by `tg`/`pp` in the server log, not by utilization percentages** — latency-bound
   runs show misleadingly low CPU% in Task Manager/WMI.
5. **MoE decode reads only the active experts**: a 30B-A3B MoE decodes at ~3B-param cost.
6. **OpenAI SDK default client timeout is 600 s** — any longer request dies with
   `APITimeoutError`. Set `timeout=3600` (or worst-case-based) on the client constructor.

---

## 4. Benchmarking strategy

### 4.1 Tooling facts (`llama bench`, build 11669)

- Model from HF cache: `-hf <user>/<repo>:<quant> --offline`.
- Tests: `-p <n>` prefill-only, `-n <n>` decode-only, `-pg <pp,tg>` combined, `-d <depth>`
  KV-cache depth. Comma-separated values and repeated flags = cartesian sweep **inside one
  process** (one model load per invocation, ~30–60 s from NVMe).
- `-r <reps>`, `--no-warmup`, `--delay <s>`, `--progress` (goes to stderr),
  `-o jsonl` → results on **stdout** (redirect stdout only; keep stderr for logs).
- There is **no `-c` flag** in bench (context auto-sizes from the largest test) and **no
  `-lv` flag** (serve-only).
- Sweepable flags: `-ngl`, `-ncmoe`, `-b` (default 2048), `-ub` (default 512),
  `-ctk/-ctv` (default f16), `-nkvo` (KV offload, default 0), `-fa` (default auto),
  `-lm/--load-mode` (default auto = mmap for CPU tensors), `--repack` (default 1), `-t`.
- Serve-only flags (validate live, not in bench): `--fit/-fitt/-fitc`,
  `-cmoe/--moe-cache-mib`, `-np/--parallel`, `-lv`, `--log-file`, `--host/--port`,
  `--offline`, `-c`.

### 4.2 Golden run — parameters derived, not measured

Run once per model:

```powershell
llama serve -hf <user>/<repo>:<quant> --offline --fit on --parallel 1 -t <logical cores> -lv 4 --host 127.0.0.1 --port 8080 --log-file <golden.log>
```

`--fit on` **derives** placement from the hardware and writes it to the log — that log is the
source of truth for the sweep center. Observed on the §5 model: ngl=49 with 47 layers
overflowing (dense weights on GPU, MoE experts mmap'd in RAM), ctx auto-reduced to the fit
floor 4096, KV f16 on GPU, ubatch 512, flash-attn auto-on, plus a warning that CPU-mapped
tensors + mmap suggest `--load-mode none`. `-lv 4` is the minimum for per-request
`slot print_timing` pp/tg lines later. **Never overwrite the golden log** — serve experiment
variants with separate `--log-file`s.

### 4.3 Sweep plan (phases, ≈2 h on this hardware)

| Phase               | What                                                                                                                                                                  | Decision rule                                    |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| P0 baselines        | golden-style placement vs `-ngl 0` pure CPU, at workload sizes (`-p 512,2048 -n 128`, r2)                                                                             | pick placement direction                         |
| P1 placement        | `-ngl 0,2,4,6,8…` full layers on GPU vs `-ngl <max> -ncmoe <layers>` (dense GPU + experts RAM)                                                                        | watch 4 GB VRAM; OOM = reject                    |
| P2 batch sizes      | `-ub 256…2048` at b 2048, then `-b 512…4096` at best ub                                                                                                               | pp + tg both flat ⇒ keep defaults                |
| P3 KV cache         | `-ctk/-ctv` pairs f16 / q8_0 / q4_0; `-nkvo 1`                                                                                                                        | tg decides                                       |
| P4 load & attention | `-lm none` vs default mmap; `-fa 0` vs auto                                                                                                                           | log warning is a hint, verify                    |
| P5 context          | `-d 0,2048,4096,8192` (tg vs KV depth) → pick server `-c`; `-pg` at real prompt/gen sizes                                                                             | smallest `-c` that fits prompt+gen with headroom |
| P6 validation       | winner at r3 on workload sizes, then live A/B on the port: 2 real requests per variant, wall-clock + log pp/tg (covers serve-only flags like `--moe-cache-mib`, `-c`) | confirm bench winner end-to-end                  |

Fixed across all runs (derived from hardware/goal, not measured): `-t <logical cores>`,
`--parallel` from the goal, port 8080. Metric policy: **tg primary for interactive goals,
pp as floor**; a config that OOMs or halves tg is rejected.

### 4.4 Live validation pattern

Start each variant detached with its own `--log-file`, poll `GET /props` until up, POST
`/v1/chat/completions` with UTF-8 body bytes
(`Invoke-RestMethod -Body ([System.Text.Encoding]::UTF8.GetBytes($json))` — quoted JSON in
`curl -d` gets mangled by PowerShell), measure wall time per request, read `slot
print_timing` pp/tg from the log, stop the process, wait for the port to free before the
next variant.

---

## 5. Model results (one `###` subsection per model)

### `unsloth-Qwen3-Coder-30B-A3B-Instruct-Q4_K_M` — measured 2026-11-14 (build 11669)

Arch: qwen3moe, 30.53B params / 3.3B active per token, 48 layers, 32 heads / 4 KV heads,
head dim 128, MoE 128 experts × 8 active, 17.28 GiB GGUF, n_ctx_train 262144.
≈0.36 GiB/layer ⇒ only a few full layers fit in 4 GB VRAM.

**Winner** (bench r3 + live serve):

```powershell
llama serve -hf unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M --offline --parallel 1 -t 12 -lv 4 -ngl 49 -ncmoe 48 -b 2048 -ub 1024 -ctk f16 -ctv f16 -lm none -c 8192 --host 127.0.0.1 --port 8080 --log-file .\llama.log
```

| Metric                          | Value                                                     |
| ------------------------------- | --------------------------------------------------------- |
| pp512 / pp2048 (bench, r3)      | 102 / 132 t/s                                             |
| tg256 / tg512 (bench, r3)       | 11.4 / 10.3 t/s                                           |
| tg vs KV depth                  | 12.5 @ d0, 11.7 @ d2048, 11.0 @ d4096, 9.9 @ d8192 t/s    |
| live: 1209-tok prompt + 128 gen | 20.5 s cold / 11.4 s warm wall; pp 124.7 t/s, tg 11.8 t/s |

Rejected configs (measured): `-ngl 0` pure CPU (tg 8.4), full-layer `-ngl 2…8` on GPU
(tg 7.9–10.4), `-ctk/-ctv q4_0` (tg 10.7), `-nkvo 1` KV on CPU (tg 9.3), `-fa 0` (tg 10.5),
`-ub 256` (pp512 60), live `--moe-cache-mib 1024` (tg 5.3 — the expert cache thrashes a
4 GB card). Old hand-made recipe (`-ngl 0 -c 8192 -ctk q4_0 -ctv q4_0 -b 2048 -ub 2048`):
tg256/512 8.55 t/s, live wall 30.1/21.1 s — the winner is +21–33% tg in bench, ~+70% tg and
−32%/−46% wall time live.

Raw data: `results/bench-2026-11-14-unsloth-Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.jsonl`
(one JSONL line per test with all params + avg/stddev t/s) and `.md` summary next to it.

### MoE-specific lessons (measured on a 4 GB VRAM card)

- **Expert placement is the dominant MoE lever.** Dense weights on GPU + MoE experts in RAM
  (`-ngl <max> -ncmoe <layers>` — exactly what `--fit on` derives automatically) beat pure
  CPU by ~40% tg (11.7 vs 8.4 t/s) and beat every full-layer GPU offload (`-ngl 2…8`,
  tg 7.9–10.4).
- **`--moe-cache-mib` (expert cache on GPU) backfired on 4 GB VRAM**: live tg 5.3 vs 11.8 t/s —
  the cache thrashes the small card. Re-evaluate only on larger VRAM.
- **`-lm none` matters precisely because experts are CPU-mapped**: the server log warns
  "tensor overrides to CPU are used with mmap enabled"; `-lm none` lifted tg 11.5 → 12.0.
- Rule of thumb for MoE on a small GPU: trust `--fit on` for placement, add `-lm none`,
  skip `--moe-cache-mib`, and verify tg with `llama bench` before serving.

### Other measured generation recipes (same hardware, older build 10441, port 8080)

| Model                   | Recipe highlights                                      | Measured                                    |
| ----------------------- | ------------------------------------------------------ | ------------------------------------------- |
| Qwen3-8B Q4_K_M         | `-c 8192 -ctk q4_0 -ctv q4_0 -ngl 22 -b 2048 -ub 2048` | ~4.0 t/s gen, ~92 t/s prefill               |
| Qwen3-4B Q4_K_M         | `-c 8192 -ngl 37` (fully offloaded)                    | ~11–12 t/s                                  |
| Qwen2.5-Coder-7B Q4_K_M | `-c 8192 -ngl 28` (≈4.7 GB, full GPU; trim if OOM)     | simple-coding slot                          |
| Qwen3-14B Q4_K_M        | `-c 8192 -ngl 12` (batch only)                         | ~1.7–2 t/s; `-c 16384` variant ~1.3–1.5 t/s |

### Infrastructure models (stable ports, not benchmarked here)

**Embeddings — bge-m3, CPU, port 8081:**

```powershell
llama serve -hf CompendiumLabs/bge-m3-gguf:F16 --parallel 1 -lv 4 -t 12 -c 1024 --embeddings --pooling cls --fit off -ngl 0 -b 1024 -ub 1024 --port 8081
```

No `query:`/`passage:` prefix needed (unlike e5). `-b` must be ≥ max input length.

**Reranker — bge-reranker-v2-m3, CPU, port 8082:**

```powershell
llama serve -hf gpustack/bge-reranker-v2-m3-GGUF:Q8_0 --parallel 1 -lv 4 -t 12 -c 2048 --embedding --pooling rank --rerank --fit off -ngl 0 -b 1024 -ub 1024 --port 8082
```

`POST /v1/rerank` with `{query, documents, top_n}` → Jina-style
`results[{index, relevance_score}]`; scores are raw logits (less negative = more relevant).

---

## 6. Extending this document

- **New model** → new `###` under §5 with model ID `<org>-<name>-<quant>`: arch facts,
  golden-run params read from its `--fit on` log, the §4.3 phase sweep, winner command +
  results table + rejected options with numbers.
- **New goal** → new row in the §1 registry + new `### Goal:` section: concurrency,
  prompt/output token ranges, sampling, and the metric weights (tg vs pp vs `-pg` combined) —
  then pick the §5 model whose measured numbers fit. Common parts (§2–§4) do not change.
- **Raw results** → `results/bench-<date>-<model-id>.jsonl` (+ `.md` summary). Projects
  reference goals by ID and link here relatively; they never copy benchmark content.
