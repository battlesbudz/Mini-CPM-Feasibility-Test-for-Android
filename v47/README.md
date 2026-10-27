# MiniCPM-V 4.7 phone trial (`:v47`)

Feasibility screen: can MiniCPM-V 4.7 (35B-A3B) run on a 12 GB Android phone
through OpenBMB's own phone stack, and what does it answer?

- **Runtime:** `tc-mb/llama.cpp-omni` pinned at
  `96bbf61710d44421ea54e63f5194f9b17c20d877` (the exact submodule commit
  OpenBMB's MiniCPM-V-Apps demos build against), fetched by
  `scripts/prepare_v47.py` into `vendor/llama.cpp-omni-v47/`. Its LES /
  MoE-SSD core streams the routed experts from SSD during generation
  (`llama_model_params.moe_experts_path`).
- **Model pack:** the phone-only `-les` split from
  `tc-mb/MiniCPM-V-Apps-gguf` @ `8716e5209f308fd53f197ccd7d50c6fdcc2dc37d` —
  main gguf (1.66 GB) + F16 mmproj (1.12 GB) + `experts/` (40 shards +
  manifest, 13.84 GB) = **16,623,585,508 bytes**, every file SHA-256 verified
  by the in-app store (`src/main/assets/models.json`) before a run is
  allowed. The official inline GGUF is *not* used: per the vendor it is
  killed by the OS on phones.
- **Trial:** one bundled street photo (`test-bus.jpg`, the same photo used
  by the desktop step-1 rig and the Faraday vision tests), three fixed
  questions — "Describe this image.", "How many people are in this image?",
  "How many cars are in this image?" — greedy sampling, thinking disabled,
  512-token cap, 4 threads, 4096-token context (vendor demo defaults).
- **Recorded per case:** tokenize time, vision-encode time, prefill time,
  time to first token, decode wall time and tokens/sec, generated tokens,
  peak RSS, and the answer text verbatim. The service also samples worker
  PSS and thermal status once per second, following the `:app` module's
  benchmark pattern. Results land in the app's `report.json` and are shown
  on screen; "Copy diagnostics" exports them.
- **Ground truth for the photo:** 3 prominent foreground people plus ~8
  distant pedestrians; 3 clear parked cars plus several distant ones; 1 bus.
  Faraday's pipelines on the same photo: native ONNX 5 detections
  (3 people / 2 cars, ~0.3 s); phone-inference server 17 raw detections.

## Live stage checkpoints (build 9)

Build 8 fixed model load but a device run then sat in Case 1 for the full
30 minutes with no signal past the case-start line, so a hang and an
extreme crawl were indistinguishable. From build 9 the native benchmark
emits live stage checkpoints and the service merges them into
`status.json` (and the current stage into the memory sample), so the
status always shows where a run is and when it last made progress:

- `stage` / `stageEvent` / `caseIndex`: current stage (`model_load`,
  `mmproj_load`, `context_create`, then per case `tokenize`,
  `prefill`, `vision_encode`, `decode`, `failed`) and whether it
  started, is in progress, produced its first token, or finished.
- `stageStartedAtMs`: when the current stage began; `lastProgressAtMs`:
  when the last checkpoint arrived. A stage that started long ago with
  no later progress is a hang in that stage; a growing
  `generatedTokens` with a fresh `lastProgressAtMs` is a crawl.
- Per-stage measurements as they complete: `modelLoadMs`, `mmprojLoadMs`,
  `contextMs`, `tokenizeMs`, `visionMs`, `prefillMs`, `promptTokens`,
  `imageTokens`, and during decode a ~1 Hz `generatedTokens` counter
  with `decodeElapsedMs` / `decodeTokensPerSec`.

Instrumentation only: model files, prompts, context/batch/token caps,
thread count and the watchdog are unchanged, and the final report format
is untouched.

## Build 12: streaming levers, quick mode, per-case reports

Build 11 (decode-window expert cache + vendor w16 prewarm table) got
decode to persist experts across tokens inside a 16-slot-per-layer
window. Build 12 stacks the next levers from the streaming research
(`~/workspace/evals/minicpm-v47/STREAMING_RESEARCH.md` in the trial
record) plus the diagnostic-loop fixes:

- **Lookahead prefetch ON** (`LLAMA_MOE_LOOKAHEAD_PREFETCH=1`, default
  depth 8): the runtime pre-reads the predicted next experts during the
  current layer's compute. Report field `moeLookaheadPrefetch`.
- **Prefill bank 256 → 1024 slots** (~1.29 GiB resident; fork cap is
  32 × 40 layers = 1280). Total expert residency with the decode
  windows is ~2.50 GiB, toward the vendor's ~3 GB working set. Honest
  bound: the fork clears each layer's cache per call (that clear is
  load-bearing — the shared bank is time-multiplexed across layers),
  so the bank is within-call capacity, not cross-chunk persistence.
  Retaining the cache across image chunks is **not** in this build:
  it needs per-layer prefill windows in the fork, a redesign.
- **Own prewarm table, mined from our own routing:** every run records
  its decode routing to `moe_sweep.txt` in the model directory
  (`LLAMA_MOE_SWEEP`). At startup the previous run's trace is mined
  (top-16 experts per layer by observed frequency) and used as the
  prewarm table; with no usable trace the vendor w16 table is the
  fallback, so the first build 12 run prewarms with w16 and the mined
  table takes over from the second run. Report field `moePrewarmTable`
  says which table was used.
- **[moe-ssd] counters in status and report:** the runtime's own
  expert-install / cache-hit / prefetch / I/O counters are parsed out
  of its log stream after every case (`moeExpertInstalls`,
  `moeCacheHitRatePct`, `moePrefetchHits`, `moeLookaheadHitRatePct`,
  `moeIoReadMiB`, …), so each lever above is provable from one run.
- **Per-case partial report:** every finished case is written to
  `report-partial.json` as it completes (and shown in Copy
  diagnostics). A watchdog timeout now loses at most the in-progress
  case — build 10's 100-minute timeout produced no per-case results
  at all. Cumulative run totals (sums across cases) ride along in
  `status.json` under `totals`; the per-stage status fields remain
  latest-case snapshots.
- **Quick decode check:** the "Quick decode check · Case 3 only"
  button runs just the long bicycle text case with a 32-token decode
  cap under a 10-minute watchdog — a trustworthy tok/s in minutes
  instead of a full trial. The full trial's cases, prompts and
  defaults are unchanged.

## Build 13: lookahead prediction accuracy

Builds 11/12 ran lookahead prefetch in the runtime's window mode,
where its hits counter can never increment (§21 of the trial record:
the only increment site is in the non-window branch), so both device
runs reported armed 614/629 with 0 hits while prediction accuracy
itself was measured by nothing. Build 13 changes no configuration
and no behavior — it only scores the predictions. The trial patch
now also adds two counters to the slot runtime: when a layer
consumes the prediction the previous layer stored for it, the raw
predicted set (top-8 per layer) is compared with the set the layer
actually requests in the same decode step. Scored = distinct
predicted experts compared; correct = those actually requested.
The totals print at runtime teardown on a `lookahead-accuracy:`
line and land in the report as `moeLookaheadScored`,
`moeLookaheadCorrect` and `moeLookaheadAccuracyPct`. They come
from the runtime's atomic counters read whole at teardown, so
they cover the entire run (the per-step flush-line sums are
throttled by the runtime and only ever described the early steps).
One quick-mode run produces the number under exactly the build 12
runs' conditions.

## Build

CI (`.github/workflows/v47.yml`) builds `:v47:assembleRelease` on GitHub
Actions and uploads the APK as a workflow artifact (no GitHub Release).
Locally the same steps are: `python3 scripts/prepare_v47.py` then
`./gradlew :v47:assembleRelease`. The APK is signed with the repo's
feasibility-test key, like the other modules.

## Run

1. Install the APK, open **MiniCPM-V 4.7 Trial**.
2. Tap **Download model pack · 16.6 GB** on Wi-Fi with ≥ 18 GB free.
3. For a fast decode-speed verdict, tap **Quick decode check ·
   Case 3 only** (32-token cap, 10-minute limit). For the full run,
   tap **Run phone trial**. Keep the phone awake and on the charger;
   a slow first run is expected (expert paging is storage-bound).
4. Read the verdict and answers on screen, or copy diagnostics.

## Known unknowns this trial is designed to resolve

- Whether the fork's LES slot runtime behaves on Android's CPU backend
  as it does on the vendor's validated iOS/Metal path (all published
  numbers are iPhone numbers; no Android result exists upstream).
- Whether Android's per-app memory limits let the load + streaming
  steady-state survive on a 12 GB device (the vendor floor for this
  model) — peak PSS/RSS in the report answers this with data.
