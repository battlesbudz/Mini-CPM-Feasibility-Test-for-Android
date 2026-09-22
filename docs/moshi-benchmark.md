# Moshi Voice Test — first Android prototype

Separate application ID: `com.battlesbudz.moshitest`. Installs alongside Jarvis, Liquid and MiniCPM. This is a release APK with the repository's existing feasibility-test signing key, not a debug build or a production Jarvis replacement.

## Current phone test

Build 11 passed the Fold6 precision gate. Use the existing APK for the next test: select **Vulkan model · CPU codec** and **2. Moshi load-only · 750-frame context**. Install/verify the full model pack if not already present; the existing Mimi file is reused. Export diagnostics after loading. If the result is `LOAD_OK`, run **3. Moshi voice replay · 20 sec**, listen to the output and export another ZIP. A load failure is a diagnostic result, not a reason to retry repeatedly. No new APK is needed for these modes.

## Earlier phone tests

### Build 2: isolate the observed Vulkan audio failure

Install build 2 over build 1 to retain the downloaded codec and recording. Select **4. Compare CPU/GPU codec · all four routes** and run once; it ignores the backend selector and performs the complete comparison automatically. It needs only the existing 347 MB Mimi file, not the full model. Allow a few minutes, then export one ZIP. Four playback buttons select the reference and each comparison recording.

The run encodes identical PCM on CPU and GPU, then decodes both token streams on both backends using four independent decoder states. `tokens.csv` contains each codebook's CPU/GPU tokens. `comparison.json` includes token agreement, first disagreement, per-codebook mismatches, waveform normalized RMSE, signal-level ratios, correlation, clipping counts and component timing. `comparison-frames.csv` records component timings for each frame.

For **CPU encoder → GPU decoder**, the tokens are exactly those used by the CPU reference; waveform divergence therefore isolates a decoder/backend problem independently of GPU encoding. **GPU encoder → CPU decoder** shows how GPU token differences affect CPU reconstruction. Token differences alone can result from rounding and are not automatically labeled an encoding bug. A waveform NRMSE over 0.1 is flagged for investigation; it is a diagnostic threshold, not an official model tolerance. A reference RMS below 0.01 is reported as inconclusive. These four-route timings must not be treated as single-pipeline throughput.

Codec replay now reports `AUDIO_LEVEL_FAILURE` if a non-quiet input produces output more than 100 times quieter. Full-model near-silence is labeled `QUIET_OUTPUT_REVIEW`, since a conversational model can legitimately choose silence. The comparison saves its final status to both its report and partial checkpoint.

Observed build-1 Fold6 evidence: identical ten-second input; CPU 6.06 fps and RMS 0.182, Vulkan 4.15 fps and RMS 0.000117, with normal reported thermal status. Build 2 instruments this unresolved issue; it does not claim to fix the GPU kernels.

### Original smoke tests

1. Install the Moshi APK and choose **Download Mimi codec only · 347 MB**. Keep the screen open during setup. Downloads resume and are verified by SHA-256.
2. Record a ten-second question. Choose **CPU · 4 threads** and **1. Mimi codec replay**, then run it. Play the output: this test reconstructs your recording; it does not answer the question.
3. Export the diagnostics/audio ZIP. Optionally repeat codec replay with **Vulkan model + codec** to test the GPU codec path. The middle backend option keeps the codec on CPU.
4. Download the full **4.68 GB Moshika Q4_K** pack. Allow about 6 GB free storage. Already verified codec files are reused. The reference model has a female voice.
5. Choose **Vulkan model · CPU codec**, then **2. Moshi load-only**. This checks weights, codec and initial streaming-state allocation without generating a response. Lazy first-frame allocations are still untested at this point.
6. If loading succeeds, select **3. Moshi voice replay**. It feeds the recorded ten seconds followed by ten seconds of silence through the persistent model. Playback is available after the test, including partial output after many failures.
7. Export the ZIP after each configuration, before starting another test. A new run replaces the detailed files. The ZIP contains the input recording, generated PCM, timing CSV, native log and memory history. It is saved only to the destination you select.

The test has a ten-minute wall-clock limit and a five-minute no-progress watchdog. Stop terminates the dedicated native worker process, which also releases its allocations. It does not kill the app UI. Native aborts and Android low-memory exits can be inspected through the copied diagnostics; use **Save native crash trace** when Android provides a tombstone.

## Interpreting results

- `LOAD_OK`: the load stage completed. It does not prove inference correctness or real-time operation.
- `REPLAY_COMPLETE`: the bounded replay completed. Listen to the audio; completion alone does not certify quality.
- `processingFps`: audio input frames processed per second, including encode/model/decode work, excluding model load, disk checkpoints and playback. Full replay needs at least 12.5 fps, with headroom.
- `replayWallFps`: includes replay-loop overhead and checkpoints. This is still an unpaced offline test, not measured microphone-to-speaker latency.
- `frameP50Ms`, `frameP95Ms`, `frameP99Ms`, `framesOver80Ms`: frame-processing distribution. Slow frames can cause live backlog even when average throughput looks acceptable.
- `encodeMs`, `modelMs`, `decodeMs`: cumulative component time. Mimi-only results must not be interpreted as full-model speed.
- `firstPcmMs`: starts at replay, excluding model loading. Initial output may be silence.
- `peakWorkerPssKb`: one-second memory samples, not an exact allocator peak; GPU/driver accounting can differ.

All Moshi modes currently use a 750-frame main context (60 seconds of audio history) to establish a memory baseline. The replay retains state between every 80 ms / 1,920-sample frame. It never resets at speech pauses. No thermal rejection policy is added; thermal status is recorded and Android retains its normal controls.

## Implementation

- Native C++20, NDK 27.2.12479018, ARM64, CPU and Vulkan.
- Moshi: `f1fabbd14a506076d4d0a9755811598220ee9e13`.
- Compatible GGML: `8cf09e9cd3c227ecb42aefc544c820b6c63a28f3`.
- SentencePiece: `17d7580d6407802f85855d2cc9190634e2c95624`.
- Vulkan headers: `19725e4d48082fe78e26622b15d3080ccd54112b`.
- Exact model revisions, sizes and SHA-256 hashes: `moshi/src/main/assets/models.json`.

`scripts/prepare_moshi.py` preserves pinned upstream checkouts and generates an Android integration copy. Repairs include byte-sized loader buffers, 8 MiB GGUF upload chunks, allocation error handling, a release-build missing-return path, and explicit graph-operation/status checks. Unhandled upstream assertions can still abort the worker and are reported as failures. No automatic CPU fallback is presented as GPU success.

No existing Liquid Q4 GPU workaround was copied: those patches target a different runtime and quantization path. OpenCL, Qualcomm NPU execution, live microphone/speaker streaming, echo cancellation, Jarvis tools and persona integration remain later work.

## Build

```sh
python3 scripts/prepare_moshi.py
export MOSHI_GLSLC="$ANDROID_HOME/ndk/27.2.12479018/shader-tools/linux-x86_64/glslc"
export PATH="$ANDROID_HOME/cmake/3.22.1/bin:$PATH"
./gradlew --no-daemon :moshi:assembleRelease :moshi:lintRelease
```

Use a complete JDK 17, Android platform/build tools 35, CMake 3.22.1 and the pinned NDK. `MOSHI_GLSLC` selects a host shader compiler; CI uses the NDK-bundled version. Output: `moshi/build/outputs/apk/release/moshi-release.apk`. The Moshi workflow uploads an APK artifact on its experiment branch; it does not open a PR or merge anything.

Host codec smoke test, using the manifest-verified Mimi file:

```sh
cmake -S moshi/src/main/cpp -B /tmp/moshi-host -DMOSHI_HOST_SMOKE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/moshi-host --target moshi_codec_smoke -j 4
/tmp/moshi-host/moshi_codec_smoke /path/to/mimi-e351c8d8-125.gguf
```

This test exercises 20 tone/silence frames, validates token ranges and finite nonzero decoded PCM, then resets and reuses the codec. It is a host CPU integration check, not Android performance or parity against Kyutai's official implementation.

## Plan status

- Implemented: pinned native build, loader repairs, JNI benchmark, resumable/verified setup, codec/load/full-replay modes, CPU/Vulkan selection, isolated worker, playback, stop, memory/timing/crash diagnostics, ZIP export and model deletion.
- Verified during development: Android ARM64 native compilation and host CPU Mimi encode/decode/reset smoke test with the pinned codec checksum.
- Verified on Fold6 through build 9: IM2COL repair passes all 112,480 element checks; GPU audio no longer collapses and tokens are no longer constant. CPU output remains byte-identical to earlier working CPU builds.
- Verified on Fold6 in build 11: opt-in FP32 Mimi accumulation matches all 1,000 CPU encoder tokens across 125 frames; identical-token GPU decoder NRMSE 0.0007614; no level collapse. CPU PCM remains byte-identical to build 9.
- Next device gate: full-model load and bounded replay using the existing Vulkan-model/CPU-codec configuration. CPU codec remains the faster measured reference; FP32 is the supported accuracy choice for future GPU-codec integration, still opt-in in this APK.
- Unverified: full Moshi model load, coherent full replay, official-reference parity and real-time throughput.
- Later gates: deterministic official-reference parity, sustained ten-minute inference, OpenCL comparison if justified, then live duplex and echo-control integration.

## Build 3: first-frame tensor tracing

Build 2 on SM-F956U completed all four routes. The CPU PCM matched build 1 byte
for byte. All 1,000 GPU encoder tokens differed from CPU; each GPU codebook
repeated one token across all 125 frames. GPU decoding collapsed with either
CPU or GPU tokens. This establishes separate encoder and decoder path failures,
not a particular faulty kernel.

Select **5. Trace CPU/GPU operations · first frame** (the new default). Keep the
existing verified Mimi model and recording. The backend selection is automatic.
The test runs a normal CPU reference, a normal Vulkan reference, then a fresh
Vulkan codec with verified uploads and operation-by-operation CPU comparison.
Only the first 80 ms frame is used because the failure already appears there.
Decoder tests always consume identical CPU tokens. Allow a few minutes and export
the diagnostic ZIP. There is no new model download and no playable full response
from this diagnostic mode.

Exported `tensor-trace.jsonl` records each operation before execution, then its
shape, strides, type, sampled CPU/tested RMS, error and nonfinite counts. It also
records pre-dispatch input comparisons for the first divergent operation in each
stage, including initialization. `tensor-summary.json` contains first divergences
by stage, byte-for-byte upload verification counts, token agreement and normal vs
traced output metrics. `trace-tokens.csv` contains all eight first-frame tokens.

The trace wraps the existing GGML graph-copy API, executes matching one-node graph
views on CPU and Vulkan, and checks compute statuses. Complete explicit uploads
(weights, inputs, state writes and cross-backend copies) are read back in 1 MiB
chunks and compared byte-for-byte. Numerical comparison samples at most 4,096
logical elements per tensor with stride-aware F32/F16/BF16/I32 decoding, absolute
tolerance 1e-4 plus relative tolerance 1e-3, and an explicit skip for tensors over
64 MiB or other types. Execution is capped at 64 graphs / 20,000 nodes in addition
to the service watchdog. Regular replay modes use the unchanged whole-graph path.

Interpretation: each CPU graph clone starts from the tested graph's current data.
Earlier state divergence can therefore carry into both copies; examine the earliest
initialization/encoder/decoder difference and upload checks. Splitting graphs also
changes GPU fusion/synchronization. A difference that disappears under tracing is
useful evidence of that sensitivity, not proof of correctness. Sampling and numeric
tolerances identify candidate operations; this is not official-runtime parity or a
throughput benchmark. No GPU kernel fix is claimed in build 3.


## Build 5 onward: portable IM2COL correction and device correctness gate

Build 4 on SM-F956U verified all 356,462,404 uploaded bytes with zero mismatches.
Encoder node 3 (IM2COL, F16 [7,1920,1,1]) diverged despite exact inputs. Decoder
node 625 (IM2COL, F16 [3584,2,1,1]) became all zeros; earlier matrix multiplication
differences were much smaller. Normal and traced GPU paths both reproduced the
collapse. This isolates an operation, not a proven Adreno compiler root cause.

The 1D/2D shader now writes one output per invocation through a normal storage
buffer, using fixed 64-thread groups. It avoids the old specialization-sized
arrays, unrolled scatter and physical-address writes. Host dispatch sizing and
full output descriptor ranges change together. 3D and other operators retain
their existing pipelines. `prepare_moshi.py` applies a checked overlay to an
integration copy; upstream pinned checkouts remain pristine.

Every Vulkan test first runs 16 operator cases / 32 dispatches without model
weights. It checks every element against an independent scalar oracle with exactly
representable data, including both failing Mimi shapes, single timestep, padding,
stride, dilation, batches, partial groups, 2D and F16/F32. Changed inputs and poisoned
outputs catch stale data and missing writes. A failure stops the GPU run explicitly;
there is no silent CPU fallback. `im2col-check.jsonl` and `im2col-summary.json` are
included in the diagnostic ZIP. Passing this gate does not prove whole-codec parity.

The default is now **4. Compare CPU/GPU codec paths**. Upgrade the release APK,
keep the verified model and recording, run the comparison and export diagnostics.
Check that the GPU no longer produces constant tokens or collapsed audio, and
compare CPU-token/GPU-decoder PCM against the CPU reference. If differences remain,
run **5. Trace CPU/GPU operations · first frame** to locate the next divergence.
No additional model download is required.

Validation gates: host CPU operator oracle and trace tests; 20 actual Mimi CPU
frames with independent-state/reset checks; Mesa software-Vulkan operator oracle;
Android ARM64 release build and lint. CI logs identify the source revision.
**Build 9 Fold6 diagnostics subsequently confirmed the IM2COL repair.** Software Vulkan
cannot validate Qualcomm's driver. Performance may change because the simpler
shader trades the old batching optimization for easier-to-verify writes. Full
Moshi 7B execution, real-time throughput and live duplex remain separate gates.


## Build 10 onward: controlled GPU precision comparison

Build 9 on Fold6 passed all IM2COL checks and verified 356,462,404 uploaded bytes
without a mismatch. Over 125 frames, default GPU token agreement was 88.2%; GPU
decoding of identical CPU tokens had waveform NRMSE 0.00808. The first-frame trace
found the first decoder difference at F16 matrix multiplication with matching
sampled inputs (NRMSE 0.00198); final first-frame decoder NRMSE was 0.01069. Encoder
rounding accumulated before later codebook decisions. These results motivate a
precision experiment; they do not prove that accumulation explains every difference.

Mode 6 requests `GGML_PREC_F32` for matrix multiplications only while executing
higher-precision GPU states. F16 model weights and normal backend input conversions
remain unchanged. Some kernels already accumulate in F32. The report counts graphs
and matrix nodes receiving the request; it does not claim hardware instruction
tracing. Scope and graph flags are restored after each operation. CPU and previous
GPU modes retain their normal settings; no environment-wide precision override is
used and no new model download is required.

Two loaded codec weight instances support three independent encoder histories and
seven independent decoder histories:

| Encoder | Decoder | Purpose |
| --- | --- | --- |
| CPU | CPU | Reference |
| Default GPU | CPU | Baseline encoder difference |
| CPU | Default GPU | Baseline decoder difference |
| Default GPU | Default GPU | Baseline combined difference |
| FP32 GPU | CPU | Higher-precision encoder difference |
| CPU | FP32 GPU | Higher-precision decoder difference |
| FP32 GPU | FP32 GPU | Higher-precision combined difference |

`precision.json`, `precision-tokens.csv`, `precision-frames.csv`, and seven PCM
files are exported alongside existing memory, thermal and native logs. Metrics use
native floating-point audio before PCM clipping. First-frame timings are separate;
GPU variant order alternates per frame. Total seven-route timing is not standalone
throughput. `PRECISION_COMPARISON_COMPLETE` means all routes completed, not numerical
parity or a working conversational model. Compare token agreement, identical-token
decoder NRMSE, combined audio, thermal history and component time before choosing a
precision policy. A quiet reference limits interpretation.

CI runs an independent double-accumulation matrix oracle with the actual stored
half-precision inputs, including the first divergent decoder shape, the encoder
convolution shape and a partial/vector shape. It verifies backend isolation, nested
scope restoration and bit-identical default results before/after FP32. The same
codec experiment used by Android processes eight tone/silence frames on CPU and
Mesa software Vulkan with the pinned real Mimi model. It rejects invalid tokens,
nonfinite or collapsed audio, missing precision dispatches, FP32 identical-token
decoder NRMSE >= 0.05 and FP32 encoder/combined NRMSE >= 0.1. These are regression
bounds, not official model tolerances. Existing CPU, IM2COL and release/lint gates
remain required. Software Vulkan cannot establish Adreno correctness or speed.

The Fold6 precision gate is now complete (results below). Next validate full-model
load and replay before standalone throughput and live duplex work. Do not infer
full 7B model success from Mimi-only results.


The first precision CI run (build 10) correctly blocked the release on nonfinite
matrix results. Local reproduction exposed an upstream scalar small-tile layout
error on eight-lane software Vulkan: 16 invocations were divided into two virtual
warps, each sized for the whole 32x32 output tile. The second warp accessed beyond
that tile and raced adjacent output groups. The integration overlay makes that
scalar tile use a single 16-lane virtual warp. Hardware subgroup operations are
not used by this scalar kernel. Cooperative-matrix layouts and devices with at
least 16 lanes, including the tested Adreno, retain their previous configuration.
Matrix tests poison outputs, compare every result, and require repeatable baseline
outputs across precision changes. This host-path correction is separate from the
Fold6 precision experiment; it is not evidence that the Adreno had the same bug.


## Fold6 build 11 results — precision gate passed

Evidence: user-provided `moshi-test (6).zip`, build `0.1.0-build.11`, source
`cc8228fa63c9ca1c0552048ccb4188a30be0802c`, SM-F956U / SDK 36 / Adreno 750,
64-lane groups. All 125 frames (10 seconds) completed across seven routes.
Input SHA-256: `4c73c0e8e1020f14a48e41140a95ee83a1f42fc9ae2401c47bd8e1b803de2fd3`,
identical to the previous build-9 comparison and trace. Counts below were checked
against the token CSV and PCM files, not only the report's completion status.

| Check | Default Vulkan | FP32 Vulkan |
| --- | ---: | ---: |
| Encoder tokens matching CPU | 882 / 1,000 (88.2%) | 1,000 / 1,000 (100%) |
| Decoder NRMSE using identical CPU tokens | 0.00807655 (0.808%) | 0.000761403 (0.0761%) |
| Combined encoder/decoder NRMSE | 0.1553958 (15.54%) | 0.000761403 (0.0761%) |
| Encoder mean ms/frame, excluding first frame | 66.844 | 71.394 |
| Decoder mean ms/frame, own encoder tokens, excluding first frame | 158.905 | 160.094 |

The precision request reached 23,125 matrix nodes in 750 graphs. All encoder tokens
were in range. FP32-encoder/CPU-decoder PCM is byte-identical to CPU/CPU PCM;
both FP32-decoder routes are byte-identical to each other. CPU reference PCM SHA-256
`daf88d6a1f3a0d9cc5ed8ea983bc58af0b3e48b8305253bcb7b76e3796c2283a`
matches build 9. The IM2COL gate again passed all 112,480 elements.

The recording's GPU token divergence disappears under FP32 accumulation. Decoder
waveform error improves about 10.6 times, and combined error about 204 times.
This establishes a useful GPU-codec correctness baseline for this recording;
remaining small waveform differences and official-runtime parity are not resolved.
Native float output exceeds PCM range on 850 CPU samples and 851 FP32 samples
(about 0.35%); export clipping is shared with the CPU reference and is not the
previous GPU collapse. Listening quality and generated-answer relevance were not
established by this numeric analysis.

Thermal status stayed 0 in all 140 memory samples; peak worker PSS was 1,268,186 KiB
(about 1.21 GiB), with no low-memory flag. The complete diagnostic took 144.8 seconds.
Steady CPU encoder/decoder means were 61.241 + 97.505 ms/frame; default GPU
66.844 + 158.905 and FP32 GPU 71.394 + 160.094. FP32 adds about 2.5% to this summed
GPU component time after excluding first-use costs. This is a seven-route interleaved
comparison, not a standalone throughput benchmark. CPU remains faster here; neither
this run nor the prior standalone CPU replay proves real-time 80 ms frame processing.
Do not interpret the diagnostic wall time as the speed of one inference pipeline.

Decision: preserve build 11 and the existing model files. Test full Moshi first with
Vulkan model / CPU codec using modes 2 then 3 as described above. Those modes do not
silently enable FP32 for the language model: Mimi precision evidence does not validate
quantized language-model kernels. Full-model fit, first-frame allocations, coherent
speech and sustained real-time operation remain separate gates. The next required
evidence is a Fold6 load report, followed by full replay only if loading succeeds.
