# Spec Sheet: Analog Linear Backend for llama.cpp

Version: v0.3 draft (v0.2 base plus MoE additions). Companion: Guide Sheet.

## 1. Purpose

Measure accuracy and conversion cost of running static-weight linear layers through a simulated analog crossbar, compared with the same model run digitally. Output is a go/no-go signal for a hardware path (M6).

## 2. Pinned versions

- llama.cpp commit: `2f160ba59ae7a84e014d7c0e67690d110058aa03` (tag b11565). Record in every run.
- Model: one small Llama GGUF for development (Q8\_0 preferred), one small MoE GGUF for the MoE track. Record SHA256.
- Python 3 with numpy and gguf-py from the llama.cpp tree.

## 3. Scope

In:

- Dense linear layers: attn\_q, attn\_k, attn\_v, attn\_output, ffn\_gate, ffn\_up, ffn\_down.
- MoE routed experts: `blk.N.ffn_{gate,up,down}_exps` via `GGML_OP_MUL_MAT_ID`.
- Shared experts (`ffn_*_shexp`) treated as a dense FFN, if present.

Out:

- Attention score math (Q K^T), softmax, KV cache, RoPE, norms, activations, residuals. Stay digital.
- Router (`ffn_gate_inp`). Stays digital in v0.x.
- Hardware drivers. Training or noise-aware fine-tuning.
- Speed claims. Wall-clock numbers measure the simulator only.

## 4. Code layout (new backend)

```
ggml/include/ggml-analog.h
ggml/src/ggml-analog/
  CMakeLists.txt
  ggml-analog.cpp      # registration, buffer type, supports_op
  analog-core.h/.cpp   # DAC/ADC, differential tiling, noise, IR profile
  analog-config.cpp    # config parse and validate
  analog-metrics.cpp   # per-layer and per-expert stats, JSON output
```

## 5. Interfaces

- Verified at pinned commit: `ggml_backend_dev_supports_op`, `ggml_backend_dev_supports_buft`.
- Verified: `-ot` / `--override-tensor` (`common/arg.cpp`), format `<pattern>=<buffer type>`.
- Confirm before M1: backend registry entry point, buffer type struct layout, `init_tensor` hook.

## 6. supports\_op contract

Accept `GGML_OP_MUL_MAT` and `GGML_OP_MUL_MAT_ID` only when:

- src\[0\] is in the analog buffer type.
- Weight name matches the allowlist.
- Weight type is one of F16, F32, Q8\_0, Q4\_K.

Everything else returns false.

Invariant at load: allowlisted weights are in analog buffer; nothing else is.

## 7. Converter and device model

- Tiles: default 256 x 256 (configurable). Per-expert tiling for MoE.
- Differential pair G+ and G- per tile, per-tile scale.
- DAC: `dac_bits` (default 8), static or dynamic range. Dynamic cost is counted.
- ADC: `adc_bits` (default 8), full scale from calibration at `adc_fs_percentile` (default 99.9).
- Device variation: sigma\_dev applied to G+ and G- independently, seeded.
- IR drop: separable profile A(r,c) = a\_r(r) a\_c(c), precomputed.
- Partial sums accumulate digitally per input tile.

## 8. Modes

- Isolated: per-layer error from same digital input. Used for ranking.
- Cascaded: analog propagates end to end. Used for perplexity and token agreement.

## 9. Metrics

Per layer and per expert:

- MAC count, DAC count, ADC count
- Clip rate (static, calibration, eval)
- Relative L2 error and max abs error vs digital
- Activation count (MoE)
- Host transfer bytes

Per run:

- Perplexity (digital and analog)
- Greedy token agreement and first divergence
- Top-k routing agreement (MoE)
- Energy estimates, shown with a sensitivity sweep. Constants are inputs, not results.

Output: one schema-versioned JSON per run.

## 10. Milestones and pass gates

| ID | Scope | Gate |
| --- | --- | --- |
| M1 | Backend skeleton, supports\_op false everywhere, model runs unchanged | Output identical to baseline |
| M2 | Tiling, differential mapping, exact mode, batched inputs | Exact-mode error < 1e-5 |
| M3 | DAC/ADC, calibration, noise, IR profile | Monte Carlo agreement |
| M4 | Instrumentation, isolated mode | JSON schema valid, per-layer metrics present |
| M5 | Sweeps on target models, cascaded mode | Curves and cliff ranking |
| M6 | Go/no-go | Accuracy holds at target bits AND conversion energy < digital MAC energy across sensitivity sweep |

Numpy-only tests (sim harness) cover M2 and M3 before the backend exists. Full llama.cpp tests need M1 onward.

## 11. Test suite

See Spec v0.2 section 10 and the test procedure section of the design doc (Steps 1 to 10). Pass criteria there are placeholders until set before running.

## 12. Known limits

- Memory: mapped float16 weights plus retained quantized bytes is several times GGUF size. Full 7B MoE will not fit in 3 GB.
- Attention stays digital, capping best-case gain.
- Energy conclusions depend on constants that are not yet sourced.
- Synthetic numpy results show mechanics only, not Llama accuracy.

## 13. Open decisions

1. Target dense model and MoE model.
2. Router digital or analog (default digital).
3. Shared experts in scope for first result (default yes if present).
4. Per-token vs per-expert batched reads for MoE accounting.
5. Energy constant sources.
