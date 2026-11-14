# Bench: Qwen3-Coder-30B-A3B-Instruct Q4_K_M on Quadro P1000 — 2026-11-14

Build 11669 (0.7.0-dev 620d2225c), CUDA backend, Xeon E-2176M `-t 12`, P1000 4095 MiB.
Fixed: `-t 12`, `--parallel 1`, port 8080. Raw runs: `bench-30b-a3b-2026-11-14.jsonl`.
All runs `--offline`, `-o jsonl`, `--no-warmup`; pp/tg in t/s (avg of reps).

## Winner

```
llama serve -hf unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M --offline --parallel 1 -t 12 -lv 4 -ngl 49 -ncmoe 48 -b 2048 -ub 1024 -ctk f16 -ctv f16 -lm none -c 8192 --host 127.0.0.1 --port 8080 --log-file .\llama.log
```

Measured (r3): pp512 102, pp2048 132, tg256 11.4, tg512 10.3 t/s.
Live on 8080 (1209-tok prompt + 128 gen): wall 20.5 s cold / 11.4 s warm, pp 124.7 t/s, tg 11.8/11.2 t/s.

## Phase results

| Phase | Config                                  | pp512          | pp2048 | tg128               | tg256/512              |
| ----- | --------------------------------------- | -------------- | ------ | ------------------- | ---------------------- |
| P0a   | `-ngl 49 -ncmoe 48`, ub 512 (fit-style) | 77.9           | 91.9   | 11.7                | —                      |
| P0b   | `-ngl 0` pure CPU (old recipe path)     | 70.4           | 92.0   | 8.4                 | —                      |
| P1    | `-ngl 0,2,4,6,8` full layers on GPU     | 66–71          | —      | 7.9–10.4            | —                      |
| P2a   | ub 256/512/1024/2048 @ b 2048           | 60/108/110/109 | —      | 11.7/11.2/11.5/11.2 | —                      |
| P2b   | b 512/1024/2048/4096 @ ub 1024          | 80/108/110/109 | —      | 11.8/11.3/11.2/11.0 | —                      |
| P3    | KV q8_0 / q4_0 / f16+`-nkvo 1`          | 80/81/67       | —      | 11.3 / 10.7 / 9.3   | —                      |
| P4    | `-lm none` / `-fa 0`                    | 103 / 73       | —      | **12.0** / 10.5     | —                      |
| P5a   | depth d 0/2048/4096/8192                | 82/90/73/55    | —      | 12.5/11.7/11.0/9.9  | —                      |
| P5b   | `-pg 512,128` / `-pg 2048,256`          | 103            | —      | 11.7                | pg 38.6 / 57.7         |
| P6    | winner r3 workload sizes                | 102            | 132    | —                   | tg256 11.4, tg512 10.3 |

Live serve variants (2 requests each, logs in temp, `-lv 4`):

- a) winner `-c 8192`: wall 20.5/11.4 s, tg 11.8/11.2 t/s
- b) a + `--moe-cache-mib 1024`: wall 33.1/23.8 s, tg **5.3** t/s → rejected (cache thrashes the 4 GB card)
- c) winner `-c 4096`: wall 20.2/11.1 s, tg 11.8/11.7 t/s → equal to a; 8192 kept for headroom

## Conclusions

- Placement: dense weights on GPU + MoE experts in RAM (`-ngl 49 -ncmoe 48`, what `--fit on` picks) beats pure CPU on tg (11.7 vs 8.4) and any full-layer offload. The old recipe's `-ngl 0` was the wrong path.
- `-lm none` beats mmap when experts are CPU-mapped (log warning confirmed): tg 12.0 vs 11.5.
- KV: f16 on GPU (default) best; q8_0 ≈, q4_0 worse tg, `-nkvo 1` clearly worse.
- Batch: defaults fine — b 2048, ub 512–1024 (ub 256 hurts pp).
- Flash-attn auto (on) helps: `-fa 0` drops tg to 10.5.
- Context: tg falls with KV depth (12.5 @ d0 → 9.9 @ d8192); `-c 8192` is the right headroom for ≤2k prompt + ≤2048 gen.
- `--moe-cache-mib` is counterproductive on 4 GB VRAM — do not use here.
