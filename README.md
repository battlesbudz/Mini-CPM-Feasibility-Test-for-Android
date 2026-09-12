# Mini CPM Feasibility Test for Android

A separate Android experiment for MiniCPM-o 4.5. Initial target: Samsung Galaxy Z Fold6, Snapdragon 8 Gen 3, 12 GB RAM. Jarvis V2 remains independent.

**This is a CPU compute screen, not a finished voice assistant.** It exercises the audio encoder, language model, TTS and native Token2Wav through the upstream duplex API. It does not yet prove live acoustic full duplex, echo rejection, tool calling, useful answer quality, or accelerated performance. A CPU failure does not rule out a GPU implementation.

## Install and test without a computer

Successful main-branch builds publish a directly downloadable APK in [GitHub Releases](https://github.com/battlesbudz/Mini-CPM-Feasibility-Test-for-Android/releases). No ADB or Termux is needed. The APK targets ARM64 Android 10 or newer; Android 16 is the primary device target. Device operation still needs validation.

1. Install the APK. Download the voice pack in the app or import an existing folder matching `app/src/main/assets/models.json`. The nine files total **7,777,183,904 bytes**. Allow at least 9 GB free storage. Keep the app open during installation. Interrupted downloads can resume; all files are checked against pinned SHA-256 hashes.
2. Record a ten-second question that invites a spoken answer. The app appends twenty seconds of silence and repeats that thirty-second input during the benchmark.
3. Optionally enable airplane mode. Run the sixty-second smoke test first; progress to five and thirty minutes only after it completes.
4. Use **Copy diagnostics** and **Play generated speech**. Listen for intelligible, relevant speech; passing numeric thresholds alone does not establish quality.
5. **Stop current operation** terminates the isolated benchmark worker. **Delete recording and test output** removes the recording and output while retaining the model pack.

Recordings and output remain in app-private storage and are never uploaded. Diagnostics export timings and system metadata, not raw audio. Upstream debug files may include generated text or audio under `last-run`; deleting test output removes them. Internet permission is used for the explicitly requested model download. There is no telemetry.

## Interpretation

Frames are submitted on a one-second schedule independently of inference completion. Decision latency includes queue backlog. Reports include frame completion, SPEAK decisions, PCM samples and RMS, estimated PCM supply gaps, p95 decision latency, loading time, worker peak proportional memory, thermal status, device and source revision.

`PASS_CPU_SCREEN` requires all scheduled frames, at least one SPEAK decision, at least 24,000 non-silent generated samples, a drained audio pipeline, finite output, p95 decision latency below 1,000 ms and estimated PCM supply gaps below 100 ms. These are project screening thresholds, not published model guarantees. Supply gaps are estimated from callback arrivals/final flags and do not measure acoustic playback or question-end-to-first-speech latency. Long runs may encounter context-window limitations independently of compute speed.

Missing speech or partial completion is `INCOMPLETE`, never a pass. Native crashes or process kills retain a last checkpoint; Android exit metadata helps identify their cause. A watchdog bounds hangs. `FAIL_CPU_SCREEN` calls for investigation or acceleration testing before rejecting the model.

## Build

Use JDK 17, Android SDK platform/build-tools 35, NDK 27.2.12479018 and CMake 3.22.1. Set `ANDROID_HOME`, or `sdk.dir` in an untracked `local.properties`.

```sh
python3 scripts/prepare_native.py
./gradlew testDebugUnitTest assembleRelease lintRelease
```

APK: `app/build/outputs/apk/release/app-release.apk`. Model weights are not bundled or downloaded at build time. The GitHub workflow builds, tests and publishes an APK asset only on success.

The checked-in signing key is a **public development key**, with standard Android debug alias/passwords. It enables consistent experimental updates and provides no production publisher authenticity. Never use it or this application ID for production distribution.

## Pinned sources

- [MiniCPM-o 4.5 GGUF](https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf), revision `db25077c33951fe163b42986fba0132e279872a2`: Q4_K_M language model plus eight audio/TTS/Token2Wav support files. Hashes and sizes are in the manifest. The model repository lists Apache-2.0 licensing; model terms apply separately.
- [llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni), commit `64d092c60db4b4ee45768476bd752f03fdcc98ea`. Its MIT license and bundled default reference WAV are copied into assets. This app does not modify upstream sources.
- [Upstream duplex profiling](https://github.com/tc-mb/llama.cpp-omni/blob/64d092c60db4b4ee45768476bd752f03fdcc98ea/tools/omni/perf/DUPLEX_PROFILING.md).

## Next milestones

1. Run the CPU screen on the phone and inspect actual speech, memory and failure diagnostics.
2. Compare a supported Android GPU backend using the same recorded input and metrics. Do not assume every submodel has NPU support.
3. Measure live microphone/speaker interaction, echo rejection, natural interruption and first audible response latency.
4. Add deterministic Android tool dispatch and test selection and continuation after tool results.
5. Reuse proven Jarvis capabilities behind a separate backend interface if these results justify V3.

## Reconstruction note

The initial local project built and passed nine tests, but workspace maintenance removed its files and the saved chat archive could not be accessed. This repository reconstructs that implementation from the conversation, including a newly generated experimental signing key. Current validation is provided by this repository's GitHub Actions run; the prior build is not claimed as validation of these reconstructed bytes. No phone performance results have been collected.
