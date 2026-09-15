# Liquid voice feasibility: phase 1

Goal: determine whether LFM2.5-Audio-1.5B offers a better Android speech engine starting point than the existing V2 stack. V2 remains unchanged. MiniCPM builds 1 and 2 and their low-memory findings remain available.

The separate `liquid` Android module installs as Liquid Voice Feasibility (`com.battlesbudz.liquidtest`), uses the existing development signing key, and has its own model storage and diagnostics. The Liquid workflow publishes APKs under `liquid-build-N`; its version sequence is independent of MiniCPM.

## Pins and execution

- Official GGUF pack: LiquidAI/LFM2.5-Audio-1.5B-GGUF at 7d525f883a077e20afb782f2ff618edcae0e39e4.
- All four Q4_0 components, 1,074,794,688 bytes total, SHA-256 verified. Model weights are downloaded only on request, never during APK build.
- Official runner branch linked from the model card: tdakhran/llama.cpp at ec9d1fdd9cc18643c5e161b65b8053f6f6f34a4b (upstream PR 18641). This is a development implementation, not evidence that stock llama.cpp supports the entire stack.
- CPU, four threads, 4096 context, batch 256, microbatch 64. The runner API handles speech encoding, language generation, audio tokenization and vocoding. No model components are substituted.
- LFM Open License v1.0 is bundled, alongside the native runtime license. No contributions to upstream are made.

## First phone test

1. Install the Liquid APK alongside MiniCPM. Download the approximately 1.07 GB voice pack, or import a folder containing its four exact files.
2. Let the phone cool, stop competing model sessions, and record a ten-second question. Suggested fixed prompt: "Explain why the sky looks blue. Give me a clear explanation in a few sentences."
3. Run the one-question CPU smoke test. It passes the complete recorded WAV once to the model, without external ASR. A three-minute watchdog stops a hung process.
4. Play generated speech, then Copy diagnostics. Report accuracy, naturalness, repetitions and completeness. All audio remains local; the delete button removes recording and output.
5. Repeat the same spoken question in V2. Compare model loading separately from inference and playback latency.

## Evidence and interpretation

Persisted per-second PSS, sampled peak, available system RAM and thermal status survive worker termination. Exit identity is matched by PID and time. Native stderr stays in last-run/native.log for debugging.

The result includes model load time, first text and first PCM callback times after submission, total generation time, sample rate/count/RMS, output text and audio callback timestamps. Playback is deliberately after generation for this first compute test; first PCM is NOT first audible playback latency. A successful callback run is OUTPUT_GENERATED_REVIEW_REQUIRED, never a full-duplex or real-time pass. The runner does not expose EOS versus reaching the 2048-step generation cap, so human review of answer completeness is required.

## Remaining gates before adoption

- Stream PCM to live AudioTrack and measure first audible speech and underruns.
- Repeated and sustained tests with cold/warm loading separated, monitoring heat and RAM.
- Multi-turn follow-ups and corrections.
- Live mic overlap, echo rejection, stop latency and retention of the interruption. Interleaved text/audio output does not prove native full duplex.
- Simulated tool selection and arguments before connecting real device actions.
- Compare all metrics and answer quality to V2 before replacing any voice engine.

## Hardware comparison update

Select CPU 2/4/6 threads, Vulkan main model, or Vulkan main + audio. Reuse the same recording and default voice for hardware comparisons. Run each separately after cooling; initial/peak thermal status and comparison history are included in copied diagnostics. GPU discovery is logged; inspect native layer/buffer placement for actual offload. An unavailable GPU is an error, not a silent CPU comparison. All runs currently reload the model; warm resident-session tests remain a later step.

Voice selector: default conversation preserves the baseline. UK male conversation adds a voice instruction experimentally; it is not a guaranteed speaker control. UK male TTS uses the runtime's explicit TTS prompt on a fixed sample without needing a recording. Do not compare TTS-only latency directly with a recorded question. Listen to verify the voice; no automatic gender/accent verdict is made.

The benchmark compiles an app-local copy of audio-decoder.cpp with explicit detokenizer GPU/thread selection, retaining the pinned upstream checkout unchanged. Vulkan headers use SDK tag 1.4.328.1. Vulkan may increase memory or be slower on this device; measure rather than assume. Comparison history retains result summaries until Delete recording and test output is used.

## Build 4 correction
UK male fixed TTS now requests AUDIO only, matching the pinned runner CLI. Conversational trials continue to request AUDIO+TEXT. Once per second of callback activity the worker flushes PCM, atomically checkpoints partial metadata, and reports elapsed time/audio duration. After cancellation or timeout, Play uses partial metadata when a final report is absent; this is unfinished audio, not a completed benchmark. Build 5 removes the app-level thermal start restriction at the user’s request. Thermal status remains diagnostic only; the app does not gate tests on temperature. Android’s own thermal management is unchanged.

## Integrated GPU detection correction
Build 7 accepts both GGML GPU and IGPU device types. Adreno 750 was discovered as Vulkan0 in Build 6 but rejected by the dedicated-GPU-only preflight check before model loading. Devices now report raw type and integrated status. Actual offloading and performance still require a phone run.


## Build 8: isolate CPU audio operations and export crash traces
Build 7 detected the Adreno 750, but the main-model Vulkan run crashed with native
signal 11 after audio prompt evaluation. Its detokenizer had zero GPU layers yet
allocated Vulkan compute buffers. Zero layers alone does not disable automatic
operation offload. CPU audio now uses an explicit empty device list and disables
both operation and KV offload. The main model keeps Vulkan enabled; the all-audio
GPU profile remains separately selectable. This corrects placement, but does not
prove the signal 11 cause without an on-device retest/backtrace.

After a native crash, use **Save native crash trace** and attach the exported
Android tombstone protobuf (.pb), alongside **Copy diagnostics**. Export matches
the current worker PID and run start time; Android may return no trace or provide
it later. Export runs off the UI thread. No thermal test blocker is added.


## Build 9: Adreno compiler compatibility baseline
The build 8 tombstone (PID 560 / TID 802) reports SIGSEGV at address 0x2d0
inside Qualcomm libllvm-qgl.so through CreateQGLCProgram and
vkCreateComputePipelines. CPU isolation worked, but did not eliminate this crash.
The exact shader was not logged and the proprietary compiler frames are unnamed.

Vulkan profiles now disable shader FP16, cooperative matrices, integer dot-product
extensions, fusion, graph optimization and main-model flash attention. We retain
Q4 model weights and request GPU layer offload; this is not CPU fallback.
Pipeline creation is serialized in an app-local copy of the pinned Vulkan source,
and flushed begin/end records name the shader being compiled. The dependency
checkout remains unmodified. This intentionally conservative baseline can be
slower and does not establish which feature caused the original crash.

Validation: build/test/lint in CI; on-device success remains pending. Retry
Vulkan main model with default conversation voice and the same recording.
If it fails, send copied diagnostics and Save native crash trace output again.
No thermal blocker has been added. CPU profiles retain their previous settings.


## Build 10: bypass the identified Q4 matvec pipeline
Build 9's serialized log ends at mul_mat_vec_q4_0_f32_f32 (subgroup=64),
with the same Qualcomm compiler SIGSEGV. Several general matmul pipelines had
already compiled, and thermal status was 0 throughout the run.

The app-local Vulkan dispatch now routes Qualcomm Q4_0 x F32 small-column
multiplications through the existing general GPU matmul implementation.
This includes N=1; it retains the same tensors, strides, output shape, and
general dispatch bounds handling. Other vendors/types retain their old route.
The workaround requires an explicit app environment flag and no fused follow-on
operations; the compatibility profile still disables fusion.
A once-per-process vulkan_q4_route message confirms the branch was taken.
Weights remain Q4 and the operation remains on GPU. No CPU fallback is labelled
as GPU success. Retest required for output correctness, stability, and speed.
