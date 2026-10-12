# Guide Sheet: Running the Analog Simulation Test

Companion to the Spec Sheet. Follow in order. Each part ends with a check. Do not skip checks.

## Part A: Environment

### A1. Get llama.cpp at the pinned commit

```
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 2f160ba59ae7a84e014d7c0e67690d110058aa03
git rev-parse HEAD
```

Check: `rev-parse` prints `2f160ba59ae7a84e014d7c0e67690d110058aa03`.

### A2. Build the CPU reference

```
cmake -B build -DGGML_NATIVE=ON
cmake --build build --config Release -j$(nproc)
```

The first build takes a while on a small machine. Binaries land in `build/bin/`.

Check: `./build/bin/llama-cli --version` runs.

### A3. Python environment

```
python3 -m venv .venv
source .venv/bin/activate
pip install numpy
pip install -e gguf-py
```

Check: `python -c "import numpy, gguf; print('ok')"` prints `ok`.

### A4. Get a model

Download the GGUF file yourself from Hugging Face. Suggested dense model: `hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF`. For the MoE track, pick a small MoE GGUF that fits in RAM.

Place it in a `models/` folder inside the llama.cpp directory. Record its hash:

```
sha256sum models/<file>.gguf
```

Save the hash in your run notes.

### A5. Confirm the model runs

```
./build/bin/llama-cli -m models/<file>.gguf -p "The capital of France is" -n 32 --no-warmup
```

Check: coherent text is generated, no crash.

### A6. Confirm the override flag exists

```
./build/bin/llama-cli --help | grep -i override-tensor
```

Check: `-ot` / `--override-tensor` is listed.

## Part B: Numpy harness (runs now, no model needed)

This covers Steps 3 to 5 of the test procedure.

1. Write `tile_sim.py` (the tiling, DAC, ADC, and noise model from Spec section 7). Start from the sim already written for the dense case, then add an expert dimension: shape `[in, out, E]`, tiles per expert.
2. Run exact mode first:
   - `dac_bits = adc_bits = 32`, `sigma_dev = 0`, no IR drop.
   - Compare each expert against `W_e @ x`.
   - Shapes: in=1000, out=700, E=8. Tile sizes 128 and 256. n\_tokens 1 and 37.

   Check: max relative error < 1e-5 everywhere. If not, stop and fix. Exact-mode error means a bug, not noise.
3. Routing agreement, exact mode (Step 4). Top-k sets must match 100%.
4. Sweep (Step 5): adc\_bits {4,5,6,8}, dac\_bits {4,6,8}, sigma\_dev {0, 0.02, 0.05}, outlier magnitude {1, 40}. Save results to JSON, one file per config.

   Check: outlier case shows DAC clipping as the largest error source. If it does not, the outlier injection is wrong.

## Part C: llama.cpp analog backend (after M1 exists)

Nothing in this part can run until the backend skeleton is written. Order of work:

### C1. M1: backend skeleton

- Add `ggml/include/ggml-analog.h` and `ggml/src/ggml-analog/` per Spec section 4.
- Register the device with `ggml_backend_reg`. Expose one buffer type.
- `supports_op` returns false for everything.
- Hook CLI flag `--analog-config <file>` in `common/arg.cpp`.

Check: model loads and output is byte-identical to the Part A5 baseline.

### C2. M2: tiling and exact mode

- Implement dequantize, tile, and differential mapping at `init_tensor`.
- Enable `supports_op` for `GGML_OP_MUL_MAT` and `GGML_OP_MUL_MAT_ID` on allowlisted weights.
- Exact-mode tests in C++ mirror Part B step 2.

Check: exact-mode output matches digital within 1e-5 on the real model's first layer.

### C3. M3: converters and noise

- DAC/ADC, calibration, sigma\_dev, IR profile per Spec section 7.
- Calibration set and eval set must be different text files, both version-controlled.

### C4. M4: metrics

- JSON output per run, schema-versioned.

## Part D: Run the test suite

Run Steps 6 to 10 from the test procedure.

### D1. Placement (Step 6)

```
./build/bin/llama-cli -m models/<file>.gguf -ot "ffn_.*_exps=analog" -p "test" -n 64
```

Check: the load log shows expert tensors in the analog buffer. The router does not. Weight-copy counter is zero after load.

### D2. Isolated mode, one block (Step 7)

Run on block 0 only. Each expert needs at least 256 calibration tokens. Experts below that are flagged uncalibrated in the output.

### D3. Conversion counts (Step 8)

Compare logged counts with:

- ADC per token per layer = k x R x T x C
- DAC per token per layer = k x C x T

Check: exact match. A mismatch is an accounting bug.

### D4. Perplexity and token agreement (Step 9)

```
./build/bin/llama-perplexity -m models/<file>.gguf -f <eval.txt>
```

Run the same command on the analog config. Record both numbers.

### D5. Reproduce (Step 10)

Run the full suite twice. Diff the JSON outputs. They must be byte-identical.

## Troubleshooting

- **Build fails on missing BLAS or Vulkan:** you don't need them. Use the CPU build from A2 only.
- **Out of memory on load:** use a smaller model or reduce layers in the analog allowlist. See Spec section 12.
- **Exact-mode error nonzero:** check that ADC and DAC bypass at 32 bits, and that sigma is zero.
- **Output differs run to run:** check seed handling and that noise draws are per-device, seeded once at load.
- **Expert tensors not in analog buffer:** check the `-ot` pattern matches `blk.N.ffn_*_exps` exactly.

## What to send back when stuck

The failing step number, the exact command, the last 40 lines of output, and the llama.cpp commit hash.
