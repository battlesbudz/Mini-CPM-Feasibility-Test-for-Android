# Moshi Voice Test — first Android prototype

Separate application ID: `com.battlesbudz.moshitest`. Installs alongside Jarvis, Liquid and MiniCPM. This is a release APK with the repository's existing feasibility-test signing key, not a debug build or a production Jarvis replacement.

## First phone test

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
- Pending device evidence: APK installation on Fold6; codec CPU/Vulkan output; full model load; coherent full replay; actual memory and throughput.
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
