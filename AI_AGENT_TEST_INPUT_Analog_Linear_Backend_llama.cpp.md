# AI Agent Test Input: Analog Linear Backend for llama.cpp

**Document type:** Executable engineering test plan and agent operating contract  
**Merged from:** Analog Linear Backend Spec v0.3 draft, Running the Analog Simulation Test guide, and MoE translation/test notes  
**Status:** Consolidated draft. Numerical exact-mode gates are firm; noisy-model and energy gates require a frozen acceptance policy before final evaluation.

## 1. Mission

Validate, with reproducible evidence, whether selected dense linear layers and Mixture-of-Experts (MoE) expert linear layers can be represented by a simulated analogue crossbar while keeping the rest of llama.cpp digital. Measure numerical error, clipping, converter activity, expert routing behaviour, host transfers, memory requirements, and model-level quality against a digital baseline.

The outcome is an evidence-backed engineering assessment and, only when all prerequisites and acceptance thresholds are defensible, a go/no-go recommendation for a possible hardware path. A NumPy simulation is evidence about the mathematical model, not proof of real hardware performance or Llama quality.

## 2. Instructions to the AI agent

Act as a careful test engineer, not as a demo generator.

1. Inspect the actual repository, branch, current commit, build system, tensor names, backend registration APIs, and existing tests before changing anything. Verify API assumptions against the checked-out source.
2. Treat the pinned llama.cpp commit and the rules in this document as the reference target. If the working tree is on a different commit, report the mismatch first. Do not silently reset, discard, or overwrite user changes to force the pin.
3. Separate **implemented**, **tested**, **failed**, **blocked**, and **not applicable**. A missing feature is not a passing test. A test that could not run is not a failure of numerical correctness, but it remains an unresolved gate.
4. Do not fabricate logs, metrics, model files, hashes, benchmark results, source citations, or completed implementation status. Every PASS must point to the command or test, captured output, and relevant artefact.
5. Run tests in order. Stop the dependent phases when an upstream gate fails. Independent tests may continue only when doing so cannot mask or invalidate the failure.
6. Do not weaken tolerances or change acceptance thresholds after seeing the results. Freeze all provisional thresholds and configuration before the evaluated run. Record any proposed change as a separate decision, then rerun the affected tests.
7. Keep the router digital for this v0.x specification. Do not imply that router approximation has been tested when no analogue router exists.
8. Preserve a digital baseline and use the same model, inputs, tokenisation, seeds, and relevant runtime settings for comparisons.
9. Make every result traceable to the exact commit, model SHA256, configuration hash, seed, calibration file hash, evaluation file hash, and test code revision.
10. Report limitations plainly. Simulator throughput is not hardware throughput. Estimated energy is not measured energy. A reduced-layer or one-block MoE result must not be described as a full-model result.
11. Prefer a small, testable change over unrelated refactoring. Do not add placeholder implementations merely to make an interface appear complete.
12. If repository evidence contradicts this document, do not silently choose one. Record the exact contradiction, inspect the pinned source/API, state the impact, and use the conservative behaviour that prevents a false PASS.

## 3. Consolidation decisions and conflicts resolved

The source sheets overlap, and some instructions conflict or are incomplete. Apply the following resolutions for this test plan.

| Topic | Consolidated rule |
|---|---|
| llama.cpp revision | Reference commit is `2f160ba59ae7a84e014d7c0e67690d110058aa03` (reported as tag `b11565`). Record the actual commit in every result. A different revision is a compatibility deviation, not an equivalent run by assumption. |
| Scope of analogue execution | Dense linear layers and routed expert linear layers only. Attention scores, softmax, KV cache, RoPE, normalisation, activations, residuals, and the router remain digital. |
| MoE operator | Expert weights use `GGML_OP_MUL_MAT_ID` in the source-described design. Verify graph dispatch at the pinned commit and for the selected model. Never assume expert layers were tested merely because `MUL_MAT` works. |
| Router agreement contradiction | One source test suggests comparing digital and analogue router logits, while the main spec keeps `ffn_gate_inp` digital. For v0.x, test routing determinism and expert-dispatch consistency with the same digital router output. Report analogue-router top-k agreement as **NOT APPLICABLE**. If a future analogue-router experiment is added, label it experimental and outside this plan's pass gates. |
| Exact mode | Use an explicit ideal/bypass path for DAC, ADC, device variation, and IR drop. Merely setting converters to 32 bits is not enough unless the implementation proves that this is an exact bypass. |
| Model output identity | Compare generated token IDs or the exact generated payload under deterministic, matched settings. Ignore expected differences in diagnostic logs, elapsed-time fields, and process metadata when checking M1 output identity. |
| Reproducibility | Canonical result JSON must be byte-identical across repeat runs when its fields are deterministic. Volatile wall-clock timings and timestamps must be stored separately or excluded from the canonical comparison. Never claim byte-identical full output when volatile fields differ. |
| MoE conversion accounting | Canonical v0.3 mode is **per-token expert reads**, explicitly labelled as the simple, conservative accounting mode. If per-expert batching is implemented, test it as a separate mode and do not compare its counts as though they were per-token counts. |
| Shared experts | Include `ffn_*_shexp` paths when present, as a separate dense FFN path. Report their MAC share separately. If excluded for a specific run, make the exclusion explicit in the run manifest. |
| Calibration coverage | Use at least 256 calibration tokens per expert as the initial rule. Experts below the threshold must be marked `uncalibrated`; never silently substitute a default range and call the run calibrated. The threshold is a test-policy default and must be recorded. |
| Noisy acceptance thresholds | The source proposes clip rate below 1% and top-k agreement at least 99% as placeholders, not settled acceptance criteria. Since the router remains digital, analogue-router top-k agreement does not apply. Freeze the clip threshold and any model-quality thresholds before the formal sweep. |
| Energy decision | Energy constants are inputs, not measured facts. M6 cannot be certified until constants have traceable sources or explicitly approved assumptions and the sensitivity range has been frozen before evaluation. If not, report energy as exploratory and M6 as BLOCKED. |
| Memory claims | The source sheets describe a 3 GB environment and estimate roughly 14 GB for a 7B model's float16 weights. Verify current available RAM and model footprint on the actual test host. Do not assume the old environment limit still applies. |
| Tensor override | `-ot` / `--override-tensor` uses a pattern-to-buffer-type mapping. Run the override only after the `analog` buffer type is genuinely registered. Verify the exact tensor names and matching pattern from the loaded model. |

## 4. Pinned reference and test scope

### 4.1 Reference environment

- llama.cpp commit: `2f160ba59ae7a84e014d7c0e67690d110058aa03`.
- CPU-only reference build for the baseline. BLAS and Vulkan are not prerequisites for this test plan.
- Python 3, NumPy, and `gguf-py` from the checked-out llama.cpp tree.
- Fixed random seed: `42` for the reference test suite unless an individual test declares and records another seed.
- Separate, version-controlled calibration and evaluation text files. Their SHA256 values must be recorded. Never use the evaluation file for calibration.

Suggested dense development model from the source notes: `hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF`. Choose and record an actual small MoE GGUF separately. Check memory availability before loading or mapping all expert weights. If no suitable MoE model is present, complete the NumPy MoE tests and mark model-dependent tests BLOCKED rather than inventing a model result.

### 4.2 In-scope tensors and operations

Dense linear layers, subject to actual tensor names in the selected GGUF:

- `attn_q`, `attn_k`, `attn_v`, `attn_output`
- `ffn_gate`, `ffn_up`, `ffn_down`

Routed MoE experts:

- `blk.N.ffn_gate_exps`
- `blk.N.ffn_up_exps`
- `blk.N.ffn_down_exps`
- operation: `GGML_OP_MUL_MAT_ID`

Shared experts, if present:

- `ffn_*_shexp`, as a separate dense FFN path

Router, kept digital:

- `blk.N.ffn_gate_inp`

Weight types eligible under the source contract: F16, F32, Q8_0, and Q4_K. Verify exact type names and enum support in the pinned source. Unsupported types and all non-allowlisted operations must remain unsupported by the analogue backend.

### 4.3 Explicitly out of scope

- Attention score computation (`Q K^T`), softmax, KV cache, RoPE, normalisation, activations, and residuals.
- Analogue router execution in v0.x.
- Hardware drivers or claims about physical silicon speed or energy.
- Training and noise-aware fine-tuning.
- Full-model MoE testing where the host cannot hold the mapped weights. A one-block/reduced-layer test must be labelled as such.

## 5. Expected implementation boundary

This is the target layout, not evidence that these files already exist:

```text
ggml/include/ggml-analog.h
ggml/src/ggml-analog/
  CMakeLists.txt
  ggml-analog.cpp       # registration, buffer type, supports_op
  analog-core.h/.cpp    # DAC/ADC, differential tiling, noise, IR profile
  analog-config.cpp     # parse and validate configuration
  analog-metrics.cpp    # per-layer/per-expert statistics and JSON output
```

The agent must inspect and verify the pinned commit's backend registry entry point, buffer-type structure, `init_tensor` hook, and `ggml_backend_dev_supports_op` / `ggml_backend_dev_supports_buft` usage before implementing the backend. The source sheets identify these support functions as verified but still require confirmation of the registration and buffer hooks before M1.

### `supports_op` contract

Return true only when all of the following are true:

1. The operation is `GGML_OP_MUL_MAT` or `GGML_OP_MUL_MAT_ID`.
2. The weight source (`src[0]` under the source-described contract) belongs to the analogue buffer type.
3. The tensor name matches an explicit allowlist.
4. The weight type is one of F16, F32, Q8_0, or Q4_K.
5. The operation shape and source layout are handled by a tested implementation.

Everything else must return false and be handled by an appropriate digital backend. At load time, enforce and test the invariant that only allowlisted tensors are placed in the analogue buffer.

## 6. Converter and device model

Use an explicit, serialisable configuration. Source defaults are:

| Parameter | Default | Required behaviour |
|---|---:|---|
| Tile dimensions | 256 x 256 | Configurable; handle non-multiple dimensions and boundary tiles. |
| Differential mapping | G+ and G- | Differential conductance representation with scale recorded per tile. |
| `dac_bits` | 8 | Support static or dynamic input range. Count dynamic-range cost when enabled. |
| `adc_bits` | 8 | ADC full scale derived from calibration at `adc_fs_percentile = 99.9` by default. |
| `sigma_dev` | Explicitly configured | Apply independently to G+ and G-; use deterministic seeded draws. Do not invent a non-zero default without recording it. |
| IR drop | Disabled in exact mode | Use separable precomputed profile `A(r,c) = a_r(r) * a_c(c)` in noisy/device mode. |
| Partial sums | Digital accumulation | Accumulate partial sums digitally per input tile. |
| Random seed | 42 | Seed once at simulator/device initialisation according to one documented policy; do not reseed per token unless the configuration explicitly requires it. |
| MoE read accounting | `per_token` | Grouping mode must be explicit in configuration and metrics. |

Configuration validation must reject invalid bit widths, invalid percentile values, negative or non-finite noise values, unsupported tile dimensions, missing calibration inputs when required, unknown enum values, and contradictory exact-mode/noise settings. Error messages must identify the invalid field.

## 7. Test execution phases

Run the phases in order. Every phase must end with a test record containing status, commands, environment, evidence paths, and any failure details.

### Phase 0: Repository reconnaissance and safety checks

1. Record `git rev-parse HEAD`, branch, working-tree status, and submodule state where applicable.
2. Confirm the current revision matches the pinned commit. If not, do not silently change branches or discard edits. Either test the current tree as a clearly labelled deviation or stop pinned-reference tests as BLOCKED.
3. Read the actual backend registration and tensor override code, and inspect the graph/operator code for `GGML_OP_MUL_MAT` and `GGML_OP_MUL_MAT_ID`.
4. Inventory existing analogue backend files, test files, scripts, model files, and prior outputs. Distinguish existing implementation from the target layout in Section 5.
5. Record operating system, compiler, CMake, Python, NumPy, available RAM, and free disk space.

**Gate P0:** No source edits before the initial state is recorded. Any incompatible API assumption becomes an explicit finding.

### Phase 1: Build and digital baseline

From the llama.cpp checkout:

```sh
cmake -B build -DGGML_NATIVE=ON
cmake --build build --config Release -j"$(nproc)"
./build/bin/llama-cli --version
python3 -m venv .venv
. .venv/bin/activate
pip install numpy
pip install -e gguf-py
python -c "import numpy, gguf; print('ok')"
./build/bin/llama-cli --help | grep -i override-tensor
```

Use the platform-appropriate equivalent if the shell is not POSIX. Do not treat a shell syntax issue as a backend failure; record the adapted command.

1. Confirm the commit with `git rev-parse HEAD`.
2. Select a small dense GGUF and record its SHA256.
3. Run a baseline generation with fixed prompt, token count, seed, thread count, context, and relevant flags. The source command is:

```sh
./build/bin/llama-cli -m models/<file>.gguf -p "The capital of France is" -n 32 --no-warmup
```

Add explicit deterministic settings supported by this revision and record them. Save generated token IDs or an exact generated payload separately from logs.
4. Confirm `-ot` / `--override-tensor` is present in help. This only proves the CLI flag exists, not that the analogue buffer type exists.

**Gate P1:** CPU baseline runs successfully and can be repeated under the recorded settings. If a model or required dependency is absent, mark the dependent tests BLOCKED with the missing item recorded.

### Phase 2: NumPy exact-mode harness, no model required

Create or inspect `tile_sim.py`. It must support dense matrices and expert tensors with logical shape `[in, out, E]`, with tiling performed independently for each expert. Ensure the test exercises non-divisible tile boundaries.

Run the exact-mode matrix:

- `in = 1000`, `out = 700`, `E = 8`.
- Tile sizes: 128 and 256.
- Token counts: 1 and 37.
- `dac_bits = 32`, `adc_bits = 32` only if those settings invoke the documented ideal bypass. Otherwise enable the explicit exact bypass.
- `sigma_dev = 0`; IR drop disabled; all stochastic/non-ideal effects bypassed.
- Compare each expert result with the digital reference `W_e @ x` using identical inputs.

Collect max absolute error, relative L2 error, and maximum relative error. Define and document the denominator/epsilon used for maximum relative error before running, so near-zero reference elements do not silently cause divide-by-zero or inconsistent scoring.

**Gate P2:** Maximum relative error is less than `1e-5` for every tested expert, shape, token count, and tile size. An exact-mode mismatch is an implementation defect. Stop noisy sweeps that depend on the faulty conversion path until it is fixed.

### Phase 3: Routing and dispatch integrity

Because the router remains digital in v0.x:

1. Feed fixed router logits into the repository's digital top-k selection.
2. Verify that the same selected expert IDs are passed to the expert dispatch path and that the `src[2]` IDs for `GGML_OP_MUL_MAT_ID` are interpreted correctly.
3. Test empty/invalid IDs, repeated expert IDs if accepted by the operator contract, top-k boundaries, and deterministic repeatability as supported by the actual model/operator semantics.
4. Do not calculate an analogue-router accuracy score. Record router analogue parity as NOT APPLICABLE.

**Gate P3:** Digital route IDs and expert dispatch agree exactly for all applicable deterministic fixtures. Any exact-mode route/dispatch mismatch is a defect, not noise.

### Phase 4: Converter, noise, and clipping sweep

Use the Phase 2 dimensions and fixed data generation. The source sweep is:

- `adc_bits` in `{4, 5, 6, 8}`.
- `dac_bits` in `{4, 6, 8}`.
- `sigma_dev` in `{0, 0.02, 0.05}`.
- Outlier magnitude in `{1, 40}` on four fixed input channels.

Record one canonical result JSON per configuration. For each expert and layer, record relative L2 error, max absolute error, max relative error, static/calibration/evaluation clip rate, DAC/ADC counts, and the number/location of injected outliers. Use fixed seeds and record them.

The source expects the magnitude-40 outlier case to expose DAC clipping as a dominant error contributor. Treat that as a diagnostic expectation to investigate, not permission to alter the injection until it passes. Prove that the intended outliers were actually inserted and that clipping metrics identify them.

Initial noisy-test policy from the source notes, pending pre-run confirmation:

- Calibration-set clip rate target: below 1% per expert.
- The source's top-k agreement target of at least 99% is **not an analogue-router gate** while the router is digital. Do not report it as such.

Freeze any required clip/error thresholds before the formal sweep. If thresholds have not been approved, report the sweep as exploratory and mark its acceptance gate BLOCKED, while still reporting raw metrics honestly.

**Gate P4:** The sweep is complete and deterministic; expected clipping behaviour is explained by evidence. A surprising result is investigated, not forcibly tuned away.

### Phase 5: Backend M1, no analogue operations accepted

Only run this phase after the backend is registered. Implement or verify the target backend layout and expose one analogue buffer type. `supports_op` must return false for every operation at M1. Add `--analog-config <file>` parsing and validation, but do not imply that this enables analogue computation.

1. Build the backend.
2. Load and run the same model under the same deterministic settings as Phase 1.
3. Compare generated token IDs/payload with the baseline, not diagnostic logs.
4. Test configuration parsing, invalid values, and helpful failure messages.

**Gate M1:** Model behaviour is identical to baseline under matched deterministic settings. The backend has not yet been proven to execute analogue matmuls.

### Phase 6: Backend M2, tiling and ideal mode

1. Implement dequantisation, per-tensor conversion, differential mapping, tile boundary handling, and storage initialisation at the verified `init_tensor`/appropriate lifecycle hook.
2. Enable `GGML_OP_MUL_MAT` and `GGML_OP_MUL_MAT_ID` only for allowlisted weights that meet the operation, buffer, type, and shape contract.
3. Keep all other operations on a digital backend.
4. Mirror Phase 2 numerical tests in C++ and test at least the first applicable real-model layer against the digital reference.
5. Validate dense and MoE paths separately. A dense-only pass does not pass MoE support.

**Gate M2:** Exact-mode error remains below `1e-5` on all synthetic fixtures and the tested real-model layer. Unsupported operations remain rejected by `supports_op`.

### Phase 7: Backend M3, DAC/ADC, calibration, variation, and IR drop

Implement the non-ideal effects from Section 6. Use distinct, version-controlled calibration and evaluation text files. Record how calibration tokens are collected, which activations are calibrated, each expert's token count, and the full-scale value used.

Test:

- DAC and ADC quantisation, static/dynamic range, and clipping.
- Calibration at the configured percentile.
- Independent seeded device variation on G+ and G-.
- The precomputed separable IR profile.
- Repeat-run determinism with the same seed and configuration.
- Invalid/missing calibration data and under-calibrated experts.

**Gate M3:** Numerical behaviour matches independent reference calculations/Monte Carlo expectations within a tolerance established before the run. State that tolerance and its rationale in the test record. Every uncalibrated expert is explicitly flagged.

### Phase 8: Backend M4, instrumentation and placement

Implement schema-versioned JSON reporting. Verify the loaded tensor placement and that the dispatch actually reaches the analogue operation path.

After the backend is registered, the source example is:

```sh
./build/bin/llama-cli -m models/<file>.gguf -ot "ffn_.*_exps=analog" -p "test" -n 64
```

Use this command only if `analog` is the registered buffer type's actual name and the selected model has those tensor names. For dense testing, define an explicit dense allowlist pattern from actual GGUF names rather than reusing an MoE-only pattern.

Required checks:

1. Every intended allowlisted weight is in the analogue buffer.
2. Router weights and unrelated tensors are not in that buffer.
3. Graph logs show the intended operations and no unsupported operation is falsely reported as analogue-executed.
4. Weight-copy counter is zero after initial loading / during the measured decode path. If the implementation deliberately reloads weights, count and report it rather than suppressing the counter.
5. Activation host-transfer bytes are logged separately from weight-copy bytes.
6. Each layer has operation counts, MACs, DAC/ADC counts, clipping rates, error statistics, and clear missing/NA markers.
7. Each MoE expert has activation count, calibration count/status, clipping and error statistics.
8. Each run writes one schema-versioned canonical JSON file.

**Gate M4:** Placement is correct, the active path is evidenced, metrics exist for every applicable layer/expert, and the JSON conforms to its schema. Merely seeing tensors assigned to a buffer does not prove that the operation ran on the analogue path.

### Phase 9: MoE-specific structural and isolated tests

Before converting experts, confirm the selected GGUF is genuinely MoE.

1. Dump GGUF metadata and tensor list. Record `n_expert` (`E`), `n_expert_used` (`k`), shapes of gate/up/down expert tensors, and whether shared-expert tensors exist. Source notes expect expert tensor shape `[in, out, E]`; verify actual stored dimensions and stride semantics rather than assuming the printed order.
2. If there are no `_exps` tensors or equivalent verified MoE expert tensors, stop the MoE model track as NOT APPLICABLE for that model.
3. Run a graph dump/forward pass and verify expert layers dispatch through `GGML_OP_MUL_MAT_ID`, not merely `GGML_OP_MUL_MAT`. Record per-layer operator counts.
4. Run the placement and copy checks from Phase 8 for all experts.
5. Run isolated mode for one MoE block, initially block 0 unless there is a documented reason to choose another. For each evaluation token, feed the same digital input to the digital and analogue expert calculations and route both through the same digital router result.
6. Report per-expert activation count, calibration count, clip rate, relative L2 error, max absolute error, and calibrated/uncalibrated status.
7. Include shared experts as a separate dense path if present and enabled; report their share of total MACs.

**Gate M-MoE-isolated:** Every expert is calibrated or explicitly flagged. No expert silently inherits an undocumented default. All dispatch and placement assertions pass. The report states clearly that a single-block isolated result is not a full-model result.

### Phase 10: Conversion-count accounting

For the canonical per-token accounting mode, use the source formulas per MoE layer:

- `ADC = k * R * T * C`
- `DAC = k * C * T`

where `k` is active experts per token, `R` is the number of output tiles, `C` is the number of input tiles, and `T` is the tile size. Verify these meanings against the implemented dataflow before using the formulas. If the implementation's crossbar/converter mapping implies a different count, stop, document the derivation from the implementation and correct the specification before testing. Do not adjust code counters to match a formula that does not describe the implementation.

Test edge cases including partial boundary tiles, different expert dimensions, and different `k` values. Separately account for shared experts. If per-expert batched reads are implemented, use and validate an independent count formula for that mode and label the dataflow clearly.

**Gate M-Counts:** Logged converter counts match a verified independent derivation exactly for every fixture. Any mismatch is an accounting bug and blocks energy conclusions.

### Phase 11: Cascaded evaluation and quality comparison

Run only after exact mode, noisy conversion, placement, and isolated layer tests pass.

1. Use a fixed, version-controlled evaluation text file that was not used for calibration.
2. Run the digital and analogue configurations over exactly the same model and evaluation tokens.
3. Record digital and analogue perplexity, greedy decode token agreement, first divergence position, and relevant per-layer/per-expert diagnostics.
4. Record host transfer bytes and weight storage/reload behaviour.
5. Distinguish full-model cascaded runs from reduced-layer or partial execution. Do not combine results from different scopes into one unlabeled score.

Source command for the digital perplexity baseline:

```sh
./build/bin/llama-perplexity -m models/<file>.gguf -f <eval.txt>
```

Run the equivalent supported analogue-config path only if implemented in the pinned tree. Verify the CLI option and resulting dispatch before interpreting the output.

The sources do not define the final acceptable perplexity delta or token-agreement threshold. Freeze these thresholds before evaluation. If no approved thresholds exist, report numeric outcomes and mark the quality acceptance decision BLOCKED, not PASS.

**Gate M5:** Reproducible evaluation curves and any accuracy cliff are reported, with per-layer/per-expert data sufficient to rank error sources. A report of measurements alone is not an M6 pass.

### Phase 12: Reproduction and M6 go/no-go

Run the exact same suite twice with the same pinned inputs, environment, model, config, and seed.

The canonical result JSON must contain no random run identifier, timestamp, or wall-clock measurement that changes between otherwise identical runs. Store volatile runtime/performance observations separately. Compare canonical JSON byte-for-byte and compare volatile measurements using an explicitly described method.

For M6, assess both:

1. Accuracy/quality meets the thresholds frozen before the run at the declared target DAC/ADC bits and noise settings.
2. Estimated conversion energy is less than the equivalent digital MAC energy across the predeclared sensitivity sweep, comparing the same active experts and same operation scope.

Energy constants must have a source, unit, and uncertainty/sensitivity range. Unsourced constants must be labelled assumptions. If source quality or ranges are inadequate, M6 is BLOCKED; do not manufacture a confident go/no-go recommendation. Memory/storage costs and the still-digital portions of the model must be reported alongside the conversion estimate.

**Gate M6:** GO only if every mandatory accuracy and energy condition passes with defensible inputs and reproducible records. NO-GO if a mandatory criterion fails. BLOCKED if target thresholds, energy inputs, target bit settings, or required evidence are missing. Do not relabel BLOCKED as a provisional GO.

## 8. Required metrics

### Per layer and per expert, where applicable

- Layer/tensor name, operation, backend/device path, and weight type.
- MAC count.
- DAC and ADC counts.
- Static-range, calibration-set, and evaluation-set clip rates, with denominator definitions.
- Relative L2 error, maximum absolute error, and maximum relative error versus the digital reference.
- MoE activation count, calibration-token count, and calibration status.
- Host transfer bytes, separated into activation and weight traffic where available.
- Exact/noisy mode and converter configuration.

### Per run

- Digital and analogue perplexity when cascaded evaluation has been run.
- Greedy token agreement and first divergence position.
- Digital route IDs / dispatch integrity for MoE. Analogue-router top-k agreement is NA in v0.x.
- Energy estimates for analogue conversion and comparable digital MAC work, including units, constants, provenance, and sensitivity range.
- Model mapped-weight footprint, retained quantised-weight footprint, peak memory use if measurable, and scope of layers/experts tested.
- Seed, configuration hash, model SHA256, calibration/evaluation hashes, pinned commit, toolchain and test revision.

## 9. Canonical result artefacts

Produce one canonical JSON per run, plus a human-readable test report. Keep bulky logs and volatile runtime measurements as separate artefacts referenced by relative path.

At minimum, the canonical JSON must include these groups:

```json
{
  "schema_version": "1.0",
  "test_status": "PASS|FAIL|BLOCKED|NOT_RUN|NOT_APPLICABLE",
  "source_revision": {
    "llama_cpp_commit": "...",
    "test_code_revision": "..."
  },
  "environment": {
    "os": "...",
    "compiler": "...",
    "cmake_version": "...",
    "python_version": "...",
    "numpy_version": "...",
    "available_ram_bytes": 0
  },
  "model": {
    "path_or_id": "...",
    "sha256": "...",
    "track": "dense|moe",
    "scope": "full_model|single_block|reduced_layers|synthetic"
  },
  "inputs": {
    "seed": 42,
    "config_sha256": "...",
    "calibration_sha256": "...",
    "evaluation_sha256": "..."
  },
  "configuration": {},
  "placement_and_dispatch": {},
  "layers": [],
  "experts": [],
  "model_quality": {},
  "energy_estimate": {},
  "acceptance_policy": {
    "policy_version": "...",
    "frozen_before_run": true,
    "thresholds": {}
  },
  "tests": [],
  "limitations": []
}
```

This is a minimum field contract, not permission to emit fabricated values. Use `null` with a reason or a status such as NOT_APPLICABLE / NOT_MEASURED where data is not available. Define the final JSON Schema in the repository and validate every generated result against it. Keep deterministic canonical fields separate from non-deterministic wall-clock measurements.

## 10. Test IDs and status contract

Every test record must use one of the statuses below:

- `PASS`: executed with evidence and all predeclared conditions met.
- `FAIL`: executed and a predeclared condition was not met.
- `BLOCKED`: could not validly execute because a prerequisite, threshold, model, API, or input is missing.
- `NOT_RUN`: eligible but not executed.
- `NOT_APPLICABLE`: explicitly outside the selected test scope, with rationale.

Recommended stable IDs:

| ID | Test | Required evidence |
|---|---|---|
| `ENV-001` | Pinned revision and toolchain | Commit, branch, dirty-tree state, compiler/build versions |
| `ENV-002` | CPU build and Python imports | Build log, binary version, import result |
| `BASE-001` | Dense digital baseline | Model SHA256, fixed run command/settings, generated token payload |
| `MOE-001` | Model really contains MoE tensors | Metadata/tensor dump, E, k, tensor shapes, shared-expert presence |
| `MOE-002` | Expert op dispatch | Graph dump, `MUL_MAT_ID` occurrences per layer |
| `NUM-001` | NumPy exact mode | Full shape/tile/token matrix and error metrics |
| `NUM-002` | Exact-mode routing/dispatch integrity | Fixed logits, selected IDs, dispatched IDs |
| `NUM-003` | DAC/ADC/noise/outlier sweep | One JSON per config, seeds, clip and error metrics |
| `BACKEND-001` | M1 no-op backend | Build log and baseline token-payload comparison |
| `BACKEND-002` | M2 dense exact mode | Unit fixtures and a real-model layer comparison |
| `BACKEND-003` | M2 MoE exact mode | Per-expert fixtures and real `MUL_MAT_ID` validation |
| `BACKEND-004` | M3 converter/noise model | Reference comparisons, calibration evidence, deterministic replay |
| `METRIC-001` | M4 schema and required metrics | JSON Schema validation and field-coverage report |
| `PLACE-001` | Buffer placement and copy accounting | Per-tensor placement log, operation path, copy counters |
| `MOE-003` | One-block isolated expert test | Per-expert metrics and calibration/uncalibrated statuses |
| `COUNT-001` | Converter-count identities | Independent count derivation and exact comparison |
| `EVAL-001` | Cascaded model quality | Digital/analogue perplexity, token agreement, first divergence |
| `REPRO-001` | Repeatability | Two canonical JSON files and byte-comparison result |
| `ENERGY-001` | Energy sensitivity and M6 decision | Sources, units, assumptions, uncertainty ranges, frozen criteria |

## 11. Stop conditions and troubleshooting

Stop the dependent phase and record the smallest reproducible failure when:

- The pinned commit/API contract cannot be established.
- The digital baseline fails or the model does not load.
- Exact-mode error is at or above `1e-5`.
- Exact-mode dispatch differs from the expected digital route IDs.
- The purported MoE model lacks verified expert tensors.
- Intended tensors are not in the analogue buffer or non-allowlisted tensors are in it.
- Unsupported operations appear to be accepted by `supports_op`.
- Per-expert calibration is silently missing or defaults are undocumented.
- Converter-count formula and actual dataflow disagree.
- Required canonical JSON fields are absent or invalid.
- A run cannot be reproduced under supposedly identical deterministic inputs.

Use these source-derived checks:

- Build failures mentioning BLAS or Vulkan: this plan requires the CPU build only; first confirm the build command and dependency configuration.
- Out-of-memory: record measured available memory, mapped-weight estimate, and peak use; reduce model scope only with the report labelled accordingly.
- Exact-mode error: verify true DAC/ADC bypass, zero variation, disabled IR drop, tile edges, scaling, dequantisation, and accumulation order.
- Output changes between repeat runs: verify seed initialisation and draw order, thread/parallel scheduling assumptions, input hashes, and uninitialised state.
- Expert placement misses: compare actual GGUF tensor names against the exact override pattern and confirm the buffer type name is registered.
- Weight-copy counter is non-zero: identify whether copies happen at load, once per session, per batch, or per token; report the observed behaviour rather than hiding it.

When blocked, include the failing test ID, exact command, return code, last 40 lines of relevant output, commit hash, model/config hashes, and the concrete next action needed to unblock it.

## 12. Required final report from the agent

Return a concise but evidence-rich report in this order:

1. **Verdict:** PASS / FAIL / BLOCKED for each milestone M1-M6 and the overall outcome.
2. **What was actually run:** test IDs, commands, commit, model and input hashes.
3. **Evidence:** paths to logs, canonical JSON, schema validation, graph dumps, and token comparisons.
4. **Numerical results:** exact-mode errors, per-layer/per-expert error, clipping, converter counts, activation/weight transfer bytes, perplexity and token agreement where tested.
5. **MoE-specific findings:** E, k, expert dispatch path, calibration coverage, shared experts, memory footprint, and whether scope was full model or a subset.
6. **Energy assessment:** source and uncertainty for each constant, sensitivity outcome, and any reason M6 is blocked.
7. **Failures and gaps:** root cause supported by evidence, impacted gates, and the next smallest corrective action. Distinguish observed facts from hypotheses.
8. **Reproducibility:** whether canonical JSON files were byte-identical and which volatile measurements were excluded.
9. **Known limitations:** simulator-only conclusions, remaining digital computation, model/memory scope, and unresolved open decisions.

Never report the project as hardware-ready solely because the NumPy harness passes. Do not claim a performance improvement from simulator wall time. The final decision must follow the evidence and the predeclared acceptance policy, not the desired outcome.
