# Moshi Voice Test — first Android prototype

Separate application ID: `com.battlesbudz.moshitest`. Installs alongside Jarvis, Liquid and MiniCPM. This is a release APK with the repository's existing feasibility-test signing key, not a debug build or a production Jarvis replacement.

## First phone test

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
