# Translating the Analog Linear Backend to MoE Models

Draft notes for extending the llama.cpp analog simulation spec (v0.2) to Mixture-of-Experts models. Verified against llama.cpp commit 2f160ba59ae7a84e014d7c0e67690d110058aa03.

## 1. Op interception

The v0.2 `supports_op` accepts only `GGML_OP_MUL_MAT`. Expert matmuls use `GGML_OP_MUL_MAT_ID`, where `src[0]` is the 3D expert tensor `[in, out, n_expert]` and `src[2]` is the id tensor that selects the experts each token uses. Add the op, and treat each expert slice as its own matrix for tiling. Without this, expert layers silently stay digital and the results look fine.

Tensor names at this commit: `blk.N.ffn_gate_exps`, `blk.N.ffn_up_exps`, `blk.N.ffn_down_exps` for experts, and `blk.N.ffn_gate_inp` for the router.

## 2. Active versus stored experts

For top-k routing, each token touches k of N experts per layer. Conversion counts and energy should scale with k. Storage scales with N: every expert's tiles must be resident, or reloaded. The mapped footprint is N times a dense FFN of the same width. This is the first place the memory picture gets worse.

## 3. The router is a discrete decision

`ffn_gate_inp` is a small dense matmul. If analog error flips a top-k choice, the error is no longer a small L2 perturbation. It sends the token through a different expert. Keep the router digital in v0.1. If it is analogued later, add a metric: top-k set agreement versus digital, per layer. Also log routing entropy and expert load.

## 4. Calibration per expert

Each expert sees a different activation distribution, and rarely used experts may see too few calibration tokens to set ADC full scale. Calibration needs a minimum token count per expert. Experts below the minimum are reported as uncalibrated, not silently given a bad range. Per-expert clip rate becomes a primary output, and the outlier analysis from section 6.2 of the spec applies per expert.

## 5. Shared experts

Models with shared experts (for example DeepSeek-style, `ffn_*_shexp`) have a dense path that behaves like a normal FFN. Treat it as an ordinary layer in the allowlist, separate from the routed experts. Report its share of total MACs.

## 6. Batching changes the dataflow

Tokens in a prefill batch go to different experts. On a weight-stationary crossbar, tokens would be grouped by expert before the analog read. The spec must state whether the simulator models per-token reads (simple, pessimistic) or per-expert batched reads (what hardware would do). Conversion counts differ a lot between the two, so pick one and state it.

## 7. Energy and sensitivity

Report expert-level conversion energy, and compare it with the digital cost of the same active experts, not all experts. An MoE can look better on MACs per token while costing more in stored weights. That is a separate line in the M6 go/no-go decision.

## 8. Practical constraints

Model size is the blocker. Candidate small MoE models are around 7B total parameters (for example OLMoE-1B-7B). Mapping all experts as float16 needs roughly 14 GB, and the current container has 3 GB. A full-model sim will not run here. Options: run a reduced MoE (fewer layers or experts), or run a subset of layers in isolated mode with the limitation stated.

Start with isolated mode on one expert layer. Pick one MoE block, map its experts, and check per-expert clip and error against digital. That is the MoE version of M4 and fits in memory.

## 9. Spec additions for v0.3

- Section 3: `supports_op` accepts `GGML_OP_MUL_MAT_ID` with 3D expert weights.
- Section 5: tiling is per expert.
- Section 6: active-expert accounting, and a batching mode (per-token or per-expert grouping).
- Section 9: per-expert clip and error counts, plus routing agreement.
- Section 12: memory scales with total experts; compute scales with active experts.
- Section 13 open questions: target MoE model, router digital or analog, and whether shared experts are in scope.

## Open questions

1. Which MoE model is the target?
2. Is the router digital or analog?
3. Are shared experts in scope for the first result?

## 10. Test procedure

Steps 1 to 5 run in numpy without a model. Steps 6 to 10 need a GGUF MoE model.

### Prerequisites

- llama.cpp at commit `2f160ba59ae7a84e014d7c0e67690d110058aa03`, CPU-only build.
- Python with numpy and the `gguf` package from the llama.cpp tree (`gguf-py`).
- A small MoE GGUF that fits in 3 GB RAM, or a reduced-layer version. Record its SHA256.
- Fixed seed (42), and fixed calibration and eval text files, version-controlled.

### Step 1: Confirm the model's MoE structure

Dump GGUF metadata and tensor list. Record `n_expert` (E) and `n_expert_used` (k), the shapes of `blk.N.ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps` (`[in, out, E]`), and whether `blk.N.ffn_*_shexp` exists.

Pass: values written to the run record. If there are no `_exps` tensors, the model is not MoE; stop.

### Step 2: Confirm op dispatch

Run one forward pass with debug logging. Confirm expert layers use `GGML_OP_MUL_MAT_ID`, not `MUL_MAT`.

Pass: expert layers appear as `MUL_MAT_ID` in the graph dump. Record the op count per layer.

### Step 3: Per-expert tiling, exact mode (numpy)

Generate a random `[in, out, E]` tensor with `in` and `out` not multiples of the tile size (e.g. 1000 x 700, E = 8). For each expert, map to tiles with `dac_bits = adc_bits = 32`, sigma = 0, no IR drop. Compare analog output to digital `W_e @ x`. Repeat for `n_tokens` = 1 and 37, and tile sizes 128 and 256.

Pass: max relative error < 1e-5 for every expert, shape, and tile size.

### Step 4: Routing agreement, exact mode

Feed random router logits through top-k selection, once with digital logits and once with analog logits (exact mode). Compare selected expert sets per token.

Pass: top-k set agreement = 100%. Any mismatch in exact mode is a bug, since there is no noise to cause it.

### Step 5: Noise and converter sweep (numpy)

Using the Step 3 setup, sweep `adc_bits` in {4, 5, 6, 8}, `dac_bits` in {4, 6, 8}, `sigma_dev` in {0, 0.02, 0.05}, and outlier magnitude in {1, 40} on 4 random input channels. Record per-expert relative L2 error, per-expert clip rate, and top-k agreement for each config.

Pass criteria (placeholders, set before running): clip rate < 1% on the calibration set for every expert; top-k agreement >= 99%.

Expect the outlier case to fail DAC clipping first. If it does not, the outlier injection is not working; check it.

### Step 6: Placement and copy test (llama.cpp)

Run with `-ot "ffn_.*_exps=analog"` to pin expert tensors to the analog buffer type. Confirm from the load log that every `_exps` tensor is in the analog buffer and the router is not. Run 64 decode tokens and check the weight-copy counter.

Pass: all expert tensors in analog buffer; zero per-token weight copies. Activation host-transfer bytes logged separately.

### Step 7: Isolated mode, one MoE block

Pick one block (e.g. block 0). For each eval token, compute digital and analog outputs from the same digital input, route with the digital router, and log per-expert L2 error, clip rate, and activation count. Each expert needs a minimum calibration token count (e.g. 256). Experts below it are marked uncalibrated.

Pass: every expert is calibrated or explicitly flagged. No silent defaults.

### Step 8: Conversion count check

Expected counts per token per MoE layer:

- ADC = k x R x T x C
- DAC = k x C x T

where R and C are output and input tile counts per expert, and T is the tile size.

Pass: logged counts match the formulas exactly. A mismatch means the accounting is wrong, not the hardware.

### Step 9: Cascaded mode, perplexity

Run full cascaded inference on the eval set, digital vs analog config. Log perplexity, greedy-decode token agreement, and first-divergence position.

Pass: numbers reported. The pass/fail threshold comes from the M6 go/no-go criteria, not this test.

### Step 10: Record and reproduce

Write one JSON per run containing llama.cpp commit, model SHA256, config hash, seed, calibration and eval file hashes, and all metrics above. Run the full suite twice.

Pass: byte-identical JSON on re-run.
