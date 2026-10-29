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

## Build 14: the lookahead A/B test

Build 13's quick run scored the predictor at 84.0% accuracy
(8,128 of 9,672), but accuracy alone does not prove the prefetch is
a net speed win on this CPU path — its worker shares the same I/O
threads as demand installs. Build 14 adds a second quick button,
**Quick decode check · Lookahead OFF (A/B)**, which runs the exact
same Case 3 / 32-token quick run with one value changed:
`LLAMA_MOE_LOOKAHEAD_PREFETCH` is set to `0` instead of `1` before
load — the fork's own off switch, no fork patch. The decode window
(16 slots/layer), slot bank (1024), prewarm, threads and prompt are
identical in both arms; with prefetch off the fork simply reserves
no prefetch staging slots and never computes predictions. The OFF
run cannot be confused with the ON run: its status and report carry
`mode: "quick_off"`, `moeLookaheadPrefetch: "off"` and
`abArm: "lookahead_off"`, and its report has no
`moeLookaheadScored/Correct/AccuracyPct` fields at all (nothing is
predicted, so the runtime prints no accuracy line — the fields are
absent, not zero). Run the normal quick check and the OFF check
back to back and compare `decodeTokensPerSec`.

## Build 15: the DSpark A/B arm

The lookahead file closed with no consistent win (two A/B pairs,
§§25–26 of the trial record), and the direction settled on the 35B
stack with DSpark first: speculative decoding with the model's own
trained draft heads, the strongest published lever left for this
model on this phone. Build 15 adds a third quick button, **Quick
decode check · DSpark (A/B)**, running the same Case 3 / 32-token
quick run with generation driven by the fork's `draft-dspark`
speculative implementation instead of the plain greedy loop.

- **Draft model:** `dspark-MiniCPM-V-4_7-35B-A3B-Q8_0.gguf`
  (424,572,928 B, SHA-256
  `8c0ee456b4b6fb3b67eee42dd515d97f7c8f7353bc4be9aafe9c4ff68a780c16`)
  from `openbmb/MiniCPM-V-4.7-35B-A3B-DSpark-gguf` @
  `2c8f6931e5fba8a01bcc74b2964f330c62056b5e` — a separate repo from
  the rest of the pack, so `models.json` carries a per-file source
  override and `ModelStore` resolves it (size and hash verification
  unchanged). The GGUF is a dense 5-layer `dflash` model
  (block_size 15, target layers 4/12/20/28/36, confidence head);
  it lands beside the main gguf in the model directory. Pack total
  grows to 17,048,158,436 bytes (~17.0 GB). An existing install
  re-runs Download once: the changed manifest makes the pack "not
  ready" until the one new file is fetched and verified; every
  existing file is hash-checked and skipped.
- **Configuration (all before target load, vendor iOS bridge
  pattern):** lookahead prefetch OFF (the baseline for this arm is
  the build 14 OFF arm), decode window widened 16 → 32 with
  `LLAMA_MOE_WINDOW_MAX_ROWS=4` so a verify batch of draft n_max+1
  = 4 rows stays in window mode (worst-case union 4 × 8 = 32 slots,
  exactly the window width, so the runtime's clamp leaves it at 4 —
  the report's `moeWindowMaxRowsEffective` proves it), target
  context `n_rs_seq = 3`, spec draft n_max = 3, draft context
  created with `ctx_other` = target context. The draft model loads
  with a fresh default params struct (no MoE fields: the fork
  server clears them for the dense draft the same way).
- **Driver:** the fork's speculative-simple flow — first token
  sampled greedily from the prefill logits; per round, draft 3,
  verify as one 4-row target batch, feed the batch to
  `common_speculative_process`, accept greedily
  (`common_sampler_sample_and_accept_n` with a top_k=1 sampler —
  accepted ids are the target's own tokens, so acceptance is
  exact), then `seq_rm` both contexts back to the new anchor.
  Rollback is probed once at empty KV
  (`common_context_can_seq_rm`): unless it returns `rs` or `part`
  the arm fails safe — the report says DSpark disabled itself and
  why, and no decode number is produced.
- **Telemetry:** the case carries `dsparkRounds`,
  `dsparkDraftedTokens`, `dsparkAcceptedTokens`,
  `dsparkAcceptRatePct` and `dsparkAvgAcceptanceLength` (accepted
  draft tokens per verify round) from the driver, plus the
  runtime's own `common_speculative_print_stats` line captured
  from the redirected common log (`dsparkSpecStatsLine` and its
  parsed counters) as a cross-check, and `dsparkSeqRmType` with
  the probed rollback mode. Every arm now also records
  `generatedTokenIds` (additive field, no behavior change).
- **Correctness gate:** the DSpark run's `generatedTokenIds` must
  match the baseline arms' token for token over the 32-token cap.
  Any divergence is a correctness failure, not a speed result.
  Compare against the lookahead OFF arm (identical prefetch-off
  configuration), run back to back.

## Build 17: DSpark widened to 7 draft tokens

Build 16 proved the DSpark arm end to end on device (12 rounds,
52.8% acceptance at n_max = 3, decode 1.538 tok/s), and the
token-for-token gate above was then retired: batched verification
does not guarantee bit-identical logits, so DSpark output — the
target's own batched argmax at every position — is judged on
fluency and speed instead. Build 17 changes only the DSpark arm's
depth: draft n_max 3 → 7, decode window 32 → 64 with
`LLAMA_MOE_WINDOW_MAX_ROWS` 4 → 8 (an 8-row verify batch's
worst-case union is 8 × 8 = 64 slots, exactly the window width),
target `n_rs_seq` 3 → 7. Window residency grows to ~3.3 GiB.
Everything else — draft model, prefetch off, prewarm, fail-safe
probe, telemetry, the other two quick arms — is unchanged from
build 16, and `models.json` is byte-identical, so installing over
build 16 needs no new download.

## Build 18: a safety catch around the DSpark rounds

Build 17's widened arm died silently on device: the `:benchmark`
worker process was killed natively during an early verify round
and no report was ever written (PHONE_TRIAL.md section 41). The
diagnosis named the leading vehicle an uncaught C++ exception —
an MoE slot-runtime throw with no handler anywhere from the ggml
custom op through `llama_decode` to the app's driver. Build 18
changes no configuration (the arm stays at n_max = 7, window 64,
rows 8, `n_rs_seq` = 7); it wraps the DSpark round loop in
benchmark.cpp in a try/catch that converts a throw into the
existing fatal report path with errorPhase `dspark_exception`,
keeping the exception's `what()` text in the case as
`dsparkCaughtException`. A catchable throw now ends the run with
a finished report naming the throwing site; if a build 18 run
still dies silently, that convicts the uncatchable GGML_ASSERT
branch instead. `models.json` is byte-identical, so installing
over build 17 needs no new download.

## Build 19 — the crash recorder (no configuration change)

Build 18's DSpark run died silently exactly like build 17's
(PHONE_TRIAL.md section 44): the safety catch never fired, so
the killer is an uncatchable native abort — a GGML_ASSERT /
GGML_ABORT, a slot-runtime throw whose unwind never reaches
the driver through ggml's C dispatch frames, or a hard fault —
and the source hunt (ABORT_HUNT.md) could not convict one
specific line by reading. Build 19 changes no configuration
and no behavior (the DSpark arm stays widened at n_max = 7,
window 64, rows 8, `n_rs_seq` = 7, under build 18's catch);
it adds an in-process **native crash recorder** in
benchmark.cpp, installed at the very top of the JNI entry,
before model load, so the next death names itself from the
phone (Android's own tombstones are not readable from an app
without adb). Three channels, all landing in the app files
dir:

- **Signal handlers** for SIGABRT, SIGSEGV, SIGBUS, SIGILL
  and SIGFPE append a record to `native_crash.txt` using
  async-signal-safe calls only (open/write/close, hand-rolled
  formatting): signal number and name, si_code, fault address,
  timestamp, and the PC/LR/SP registers from the ucontext plus
  a frame-pointer return-address chain. (`<execinfo.h>`'s
  `backtrace()` is best-effort at best across NDK/API levels,
  so the record carries registers and raw addresses instead;
  they resolve against the shipped `.so`.) The handler then
  restores the default disposition and re-raises — the process
  still dies exactly as it would have without the recorder.
- **A ggml abort callback** (`ggml_set_abort_callback`)
  preserves a GGML assert's own `file:line: GGML_ASSERT(...)`
  message into `native_crash.txt` and onto the captured stderr
  before `ggml_abort()` raises the SIGABRT the handler records.
  A `std::terminate` handler (installed after
  `llama_backend_init`, so it is not overwritten by ggml's own)
  likewise keeps an uncaught exception's `what()` text.
- **stderr redirected** to `native_stderr.log` (fresh per
  run), so ggml's assert printout, a terminate line, and the
  runtime's `[moe-ssd]` stderr lines survive the process. The
  trial's existing captures are not disturbed: the loader log
  arrives through the llama log callback, not fd 2 (and
  `moe_log` mirrors every line through `LLAMA_LOG_INFO`, so
  the parsed counters keep flowing), and the DSpark common log
  already has its own file.

Surfacing: **Copy diagnostics** gains a `NATIVE CRASH`
section (the record, with its file timestamp) and a
`NATIVE STDERR TAIL` section (the last 32 KiB of the latest
run's stderr — a crash's assert/terminate text is the last
thing the process wrote). The app also shows a "native crash
recorded" line on screen when a record exists, and
BenchmarkService carries a previous run's record (bounded to
4 KiB) into the next run's status.json as
`previousNativeCrash`, so the evidence survives in pastes even
after a new run starts. The recorder never deletes a record
at run start; only a new crash overwrites it. `models.json`
is byte-identical, so installing over build 18 needs no new
download.

**Device protocol for build 19:** install over build 18, do
NOT tap Download, tap **Quick decode check · DSpark (A/B)**
once, let it run its course (it is expected to die again,
silently, inside the first minute of decode), then **Copy
diagnostics** and paste. The paste should now carry the crash
record and/or the assert's stderr text naming the site: a
`file:line: GGML_ASSERT` line convicts the assert family and
names the line; a `terminate called` / `exception what():`
line naming a MoE slot error convicts the throw family and
names the throw; a NATIVE CRASH section with signal 11
(SIGSEGV) and no assert/terminate text points at a raw memory
fault; no crash record at all points at an external kill
(LMKD). No OFF-baseline run is needed.

## Build 20 — the widening is called; DSpark back to depth 3

Build 19's device run settled the widening question the other
way from what builds 17-18 suggested: the widened DSpark arm
**completed** all 32 tokens for the first time (10 rounds, 70
drafted, 21 accepted, average acceptance length 2.1, answer
token-for-token identical to build 16's DSpark output, no crash
for the recorder to record) — but it decoded at **0.640 tok/s**,
slower than build 16's depth-3 DSpark (1.538 tok/s) and slower
than the plain lookahead-OFF arms (1.325-1.480 tok/s). The
round economics are decisive: at depth 7 each round costs
~5.0 s (~505 ms of draft generation plus an 8-row verify batch
that pulls a much larger expert union from flash) against
~1.73 s at depth 3, and the acceptance-length gain (2.1 vs
1.583) does not cover a ~3x round cost. Peak worker PSS was
also the highest yet (~6.2 GB). Battles called the widening on
that evidence (2026-10-29).

Build 20 therefore reverts **only** the four widening settings
to build 16's values, which are the standing DSpark
configuration again: draft `n_max` 7 → 3,
`LLAMA_MOE_DECODE_WINDOW` 64 → 32, `LLAMA_MOE_WINDOW_MAX_ROWS`
8 → 4, target `n_rs_seq` 7 → 3. Build 18's try/catch and
build 19's crash recorder stay exactly as shipped — both are
purely additive instrumentation and remain as standing safety
nets. The quick and quick_off arms are unchanged, the fork
patch and `scripts/prepare_v47.py` are untouched, and
`models.json` is byte-identical, so installing over build 19
needs no new download.

**Device protocol for build 20:** install over build 19, do
NOT tap Download, tap **Quick decode check · DSpark (A/B)**
once, let it finish, then **Copy diagnostics** and paste. The
expected result is a completed depth-3 run at build 16's speed
class (~1.5 tok/s); the report's `dsparkWindowMaxRows` /
`moeWindowMaxRowsEffective` should read 4 and
`dsparkDraftNMax` 3.

## Build

CI (`.github/workflows/v47.yml`) builds `:v47:assembleRelease` on GitHub
Actions and uploads the APK as a workflow artifact (no GitHub Release).
Locally the same steps are: `python3 scripts/prepare_v47.py` then
`./gradlew :v47:assembleRelease`. The APK is signed with the repo's
feasibility-test key, like the other modules.

## Run

1. Install the APK, open **MiniCPM-V 4.7 Trial**.
2. Tap **Download model pack · 17.0 GB** on Wi-Fi with ≥ 19 GB free.
   (Updating from build 14 or earlier: tap Download again — only
   the new 425 MB DSpark draft file is fetched.)
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
