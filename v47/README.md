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

## Build 21 — the per-layer window A/B arm

Build 21 adds one new quick arm, **Quick decode check · DSpark
per-layer windows (A/B)** (mode `quick_dspark_layerwin`), Battles
approved 2026-10-29. It is the stock DSpark arm in every respect
(depth 3, base decode window 32, 4 verify rows, `n_rs_seq` 3,
lookahead prefetch off, same prewarm, threads, context and slot
bank) except that `LLAMA_MOE_LAYER_WINDOWS` is set before the
target model loads, distributing the decode window unevenly
across the 40 layers: floor 32 slots per layer (the slot runtime
clamps `LLAMA_MOE_WINDOW_MAX_ROWS` against the narrowest layer,
and 4 rows need width ≥ 32 to keep a whole verify batch in
window mode) and exactly 1600 slots in total, against the stock
arm's uniform 40 × 32 = 1280.

**Width source:** mined from the app's own previous-run sweep
trace (`moe_sweep.txt`) by extending the same trace parsing the
prewarm miner uses. Each layer's score is a saturation proxy —
the number of experts, in descending routing-frequency order,
needed to cover 90% of that layer's observed decode routing —
and the slots above the floor are allocated in proportion to
the scores. If no usable trace exists (the prewarm miner's own
trust rule: ≥ 400 decode lines, every layer seen), the scores
fall back to the fork header's documented saturation curves
(layer 0 ≈ 200, most layers ≈ 96), normalized to the same floor
and total. The report states which source produced the widths
actually used (`moeLayerWindowsSource`), alongside the full
vector (`moeLayerWindows`, `moeLayerWindowsWidths`), the total
(`moeLayerWindowsTotalSlots`), floor, min and max — so a device
paste shows the distribution itself, not only the runtime's
per-layer-widths-on log line. The stock `quick_dspark` arm is
unchanged and remains in the same build as the paired baseline;
the arm is judged on `decodeTokensPerSec`, answer fluency, and
the installs/hits counters against it.

**Device protocol for build 21:** install over build 20, do
NOT tap Download (`models.json` is byte-identical), tap
**Quick decode check · DSpark (A/B)** and paste, then tap
**Quick decode check · DSpark per-layer windows (A/B)** and
paste. In a good layerwin paste expect `abArm:
"dspark_layerwin"`, `moeLayerWindowsTotalSlots: 1600`,
`moeWindowMaxRowsEffective: 4`, and the runtime log's
per-layer-widths line reading on.

## Build 22 — miner fix, the 1920 probe, split and I/O arms, KleidiAI arch floor

Build 22 keeps every build 21 arm and adds three, from the
Strata/Avalon scoping plan (items 1, 3, 4 and 6). Battles
approved 2026-11-05. The pinned fork and its patch stack are
byte-identical to build 21; everything below is app-side.

**The layerwin miner is fixed.** Build 21's device run fell
back to the documented curves. The cause is structural in the
fork: the slot runtime's sweep dump takes its decode branch
only for single-row steps (`decode = n_rows == 1` in
`llama-moe-slot.cpp`), and a DSpark verify batch is 4 rows
wide, so a DSpark run records its decode routing as
prefill-format `P<layer>` lines — each carrying the batch's
last-token routed experts — and leaves ~zero `L<layer>`
decode lines. The build 21 miner's ≥ 400 L-line trust rule
could therefore never pass after a DSpark run.
`compute_layer_windows()` now mines those batch samples when
the L-line rule fails, under their own trust rule (≥ 400
batch samples in total, ≥ 8 per layer; a 32-token quick
DSpark run leaves ~13 per layer), and `moeLayerWindowsSource`
states the basis, with the trace counts in the report's
`moeLayerWindowsTrace*` fields. The prewarm miner is
deliberately unchanged, so the stock arm's prewarm behaviour
is identical to builds 12–21.

**New arms** (all are the stock DSpark configuration — depth
3, base window 32, 4 verify rows, lookahead off — plus the
stated difference):

- **Quick decode check · DSpark per-layer windows 1920
  (A/B)** (`quick_dspark_layerwin1920`): the layerwin
  distribution at a 1920-slot total instead of 1600, floor 32
  unchanged. Probes whether window memory is still buying
  misses at ~2.4 GiB of window residency; watch the run's
  peak PSS.
- **Quick decode check · DSpark matmul split (A/B)**
  (`quick_dspark_split`): the 1600-slot layerwin
  configuration plus `LLAMA_MOE_MATMUL_SPLIT=2` — the last 2
  rank positions' missing experts install deferred, so their
  flash reads overlap the matmul of resident experts. A/B
  against the plain layerwin arm.
- **Quick decode check · DSpark I/O threads 8 (A/B)**
  (`quick_dspark_io8`): the 1600-slot layerwin configuration
  plus `LLAMA_MOE_IO_THREADS=8` (fork default 4) for install
  read parallelism. The runtime reads the value once per
  process and the benchmark process exits after every run, so
  the per-run setting applies cleanly; its init log line
  prints the engaged `io_threads` value.

Both env variables are explicitly unset in every other arm,
so no stale value can leak into a baseline run.

**KleidiAI arch floor.** The trial CMake now sets
`GGML_CPU_ARM_ARCH=armv8.2-a+dotprod+i8mm` for ggml-cpu.
Until now the backend compiled at plain armv8-a, so the
fork's compile-time feature macros read 0 and its KleidiAI
kernels never engaged (the loader log's "mask 0" line). The
fork's KleidiAI covers Q4_0/Q8_0 matmuls only, so the
expected effect is confined to the Q8_0 DSpark draft path;
the check is the loader log printing a real kernel feature
line instead of mask 0. **Caveat: the ggml-cpu objects will
SIGILL on a pre-armv8.2 phone.** Fine for this trial's Fold6
(Snapdragon 8 Gen 3); do not install this build on an older
phone.

**Device protocol for build 22:** install over build 21, do
NOT tap Download (`models.json` is byte-identical). Run each
A/B pair twice with the order reversed: stock DSpark vs
per-layer windows first (in a good layerwin paste
`moeLayerWindowsSource` now reads "sweep-mined" with the
batch-sample basis, and `moeLayerWindowsTotalSlots: 1600`),
then the 1920 arm against the 1600 arm, then matmul split
and I/O threads 8 against the plain layerwin arm. A win must
beat the ±20% single-run noise band on both pairs.

## Build 23 — the prune: non-winning arms removed, io8 rebased on stock

Battles directed 2026-11-05, after build 22's full device
protocol (PHONE_TRIAL.md section 52): remove every build 22
setting that improved no stage of the pipeline, and keep any
setting that improved any stage, not only end-result decode
speed. Removed: **per-layer windows** at both totals (1600 and
1920) — neither cut installs or decode time against stock —
and **matmul split**, whose install reduction never became
speed anywhere and whose decode ran slowest of the session.
Their modes, buttons, env wiring and the sweep width miner
are gone; the stock arm's prewarm miner is untouched. Kept:
the stock **DSpark** arm (build 22's decode winner, on the
KleidiAI arch floor, which stays) and **I/O threads 8**, whose
two build 22 prefills were the fastest of the session (~29%
under stock) even though its decode showed no proven gain.

One consequence to state plainly: build 22's io8 arm ran
stacked on the 1600-slot per-layer window configuration, so
with that configuration removed, build 23's io8 arm is the
stock DSpark configuration plus `LLAMA_MOE_IO_THREADS=8` —
the io8-on-stock combination, untested until a device run
measures it. `models.json` is byte-identical again, so
installing over build 22 needs no new download.

**Device protocol for build 23:** install over build 22, do
NOT tap Download. The mode list should now show only the
full trial, quick, Lookahead OFF, DSpark, and DSpark I/O
threads 8. Optional check: run **Quick decode check · DSpark
I/O threads 8 (A/B)** twice and paste — its readings are the
first measurement of io8 on the stock configuration (prefill
is the number to watch; decode is judged against stock's
2.340 tok/s build 22 mean under the ±20% band).

## Build 24 — item 2: frequency-aware admission (pin the prewarm set, LFU probationary admission)

The ranking plan's item 2, the one build 22 deliberately
sequenced out until the cheaper levers set a clean baseline
(scope: `ITEM2_SCOPE.md`; the base is build 23's stock DSpark
configuration — depth 3, KleidiAI floor, uniform 32-slot
windows, bank 1024, I/O threads at the fork default of 4; the
plan's original layerwin base was pruned in build 23). Two
eviction policies for the MoE slot runtime, each env-gated
and default off, in a new fork patch
(`v47/patches/moe-slot-admission.patch`, applied by
`prepare_v47.py` after the existing patch; it touches only
`llama-moe-slot.cpp`, only victim selection and install
bookkeeping):

- **2a, pin the prewarm set** (`LLAMA_MOE_PIN_PREWARM`): the
  prewarm table's 640 experts (16/layer × 40) are
  eviction-protected in `process()`'s victim scan. If every
  candidate in a layer's window were protected, the scan
  falls back to plain LRU inside the protected set — an
  install is never refused.
- **2b, LFU admission** (`LLAMA_MOE_LFU_ADMIT`): one demand
  counter per (layer, expert), counted once per unique expert
  per decode step and aged by halving every 1,024 demand
  events. The adaptation from Avalon's rule, stated plainly:
  Avalon installs a miss only if its count beats the victim,
  because it keeps a second resident copy to compute from.
  LES has no second copy, so a refused install has nothing to
  compute with. Here a miss **always installs and always
  computes**; admission chooses only the residency class. A
  miss lands in a probationary segment (2 of the layer's 32
  window slots) unless its counter beats the weakest general
  resident's counter, in which case it installs as a general
  resident. Victim selection prefers probationary slots, then
  the lowest-count general resident, LRU breaks ties; a
  probationary expert demanded again while resident is
  promoted to general. Policy changes residency, never math,
  so greedy token IDs must match the stock arm exactly — that
  equality is a hard gate on every device paste.

Three new arms, each the stock DSpark configuration plus its
flag(s): **DSpark pin prewarm**, **DSpark LFU admission**,
**DSpark pin + LFU**. io8 is deliberately not crossed with
either policy (one new variable per verdict). Every DSpark
report, control included, names both flags (`moePinPrewarm`,
`moeLfuAdmit`), and the runtime logs an `admission:` line at
init naming the live flags. With both flags off the runtime
must reproduce build 23's stock counters exactly — the stock
arm in this same build is the regression check. `models.json`
is byte-identical again, so installing over build 23 needs
no new download.

**Device protocol for build 24:** install over build 23, do
NOT tap Download. Run pairs in this order, two runs each,
order reversed on the second pass: **DSpark** (control,
establishes this build's stock mean), **DSpark pin
prewarm**, **DSpark LFU admission**, **DSpark pin + LFU**;
paste each run. Pass bar (the plan's): an arm's decode pair
mean beats the same build's stock pair mean by more than
20%, both runs above the stock mean, AND installs per token
down at least 15%. Prefill is read as a secondary stage under
Battles's any-stage criterion. Case-3 flattery warning for
2a: the prewarm table is mined from this same case, so a pin
pass proves the mechanism under the kindest conditions; a
pin fail kills 2a outright.

## Build 25 — the LFU scan fix (build 24's LFU crash)

Build 24's device protocol died on its third arm: **DSpark LFU
admission** engaged correctly and finished prefill, then aborted
on the first DSpark verify step with the slot runtime's own
guard, "layer 18 requires more than 32 unique experts in one
graph". Stock and pin ran the same graph fine, so the defect was
in the LFU allocation path, and the shipped patch text shows the
two departures from stock that could strand an install on a
graph that fits the 32-slot window:

- A probationary install arriving when the probationary quota
  (2/layer) was already full **skipped the free-slot scan** and
  went straight to victim selection — so free slots could sit
  unused while the victim scan found nobody it was allowed to
  evict (every resident protected by the graph's own demand set)
  and the guard fired.
- The policy victim scan's passes were **all class-filtered**,
  with no final class-blind fallback, so it could return no
  victim while a non-requested resident existed.

The fix, confined to `v47/patches/moe-slot-admission.patch`
(no app code changes): probationary installs **never skip the
free-slot scan** — free slots are tried first, exactly like
stock, and the quota is enforced by the victim scan's class
preference once the window is actually full; and the policy
victim scan gains a **final class-blind LRU fallback pass**
that picks the plain-LRU resident excluding only the graph's
own `requested[]` set (and pending prefetch slots), regardless
of probationary/general class. Together these restore stock's
fit guarantee: any graph whose unique experts per layer are
≤ the window always places, and the guard fires only when a
graph genuinely exceeds the window. Everything else about the
LFU contract is unchanged — probationary quota 2 of 32 as the
steady state, promotion on re-demand while resident, demand
counters with halving aging, pin behavior, and the combined
pin + LFU arm through the same fixed path. With both flags off
the path is still the original stock scan verbatim, so the
flags-off counter regression check from build 24 must still
reproduce exactly. `models.json` is byte-identical again, so
installing over build 24 needs no new download.

**Device protocol for build 25:** install over build 24, do
NOT tap Download. The build 24 protocol resumes where it
stopped: **DSpark** control run 2 (also re-proves the flags-off
regression check on the fixed patch), **DSpark pin prewarm**
run 2, then the **DSpark LFU admission** pair — run 1 is the
crash retest, and merely surviving the first verify step is the
fix's first proof; a full 32-token decode with token IDs
matching the control exactly is the fix's pass. Then **DSpark
pin + LFU**, two runs, which rides the same fixed LFU path.
Pass bars are build 24's, unchanged.

## Build 26 — the run export bundle (no more diagnostics pastes)

Device pastes of the full diagnostics were tens of thousands of
tokens, almost all of it the loader log embedded in
`report.json`, and they flooded the working chat. Build 26
changes no inference behavior, no arm, and no counter; it adds
one thing: **every run exports itself as a single zip file.**

- When a run reaches a terminal state (completed, error, user
  stop, watchdog timeout), the service writes the bundle
  automatically. The hook is `BenchmarkService.retire()`, the
  one funnel every terminal path passes through before the
  process is killed, and it fires at most once per process.
- An **Export last run** button in the main screen bundles
  whatever run state is on disk at any time, so a run whose
  process died outright or was killed (build 25's LFU run 1
  stopped with no completed report) still leaves a sendable
  file. The saved bundle's filename shows in the status text.
- The zip lands in the public **Downloads** folder via
  MediaStore (no extra permission under scoped storage), with a
  second copy in the app's external files `exports/` dir.
  Filename: `minicpm-build<build>-<arm>-<yyyyMMdd-HHmmss>.zip`,
  a fresh file per run, never overwritten.
- Inside the zip, as separate files: `status.json`,
  `report.json` (full), `report-slim.json` (the same report
  with the `loaderLog` field removed), `summary.txt` (the
  small verdict summary: build, commit, arm, pin/LFU flags,
  verdict, tokens, decode tok/s, prefill, DSpark
  rounds/drafted/accepted/rate, installs, hits, hit rate, IO,
  peak thermal, peak PSS, and the full generated token ID
  list), `memory.json` (the MEMORY snapshot), the last 32 KiB
  of native stderr, `native_crash.txt` when one exists, and
  `loader_log.txt` — the loader log in its own file, present
  for crash chasing and never needed for a verdict.
- The existing Copy diagnostics flow is unchanged. On the
  analysis side, `extract_run.py` (in the trial records) reads
  a bundle and prints only the summary fields, never the
  loader log.

**Device protocol for build 26:** install over build 25, do
NOT tap Download. Run one quick arm (any arm) and confirm a
bundle named for that arm appears in Downloads, open the app
again and confirm the status text names the bundle, then attach
the zip in chat instead of pasting diagnostics. After that,
the build 25 protocol resumes on this build: control run 2
(flags-off regression re-check on the fixed patch), a
cool-phone LFU pair, and the pin + LFU pair, all reported as
bundles. Pin run 2 already landed on build 25 (see the trial
record), so pin is closed: its counters repeated exactly and
installs were up, not down.

## Build 27 — item 5: tail-skip (the last ranked lever)

Item 2 closed on device (build 26's record): pin failed on
counters, LFU failed on counters and speed. This build tests
the scoping plan's item 5, **tail-skip**, the last untested
lever in the plan (scope: `TAILSKIP_SCOPE.md` in the trial
records). One new arm, one new fork patch, nothing else moves.

- **New fork patch** `v47/patches/moe-slot-tail-skip.patch`
  (third in `prepare_v47.py`'s list; touches only
  `src/llama-moe-slot.cpp`, in regions disjoint from both
  earlier patches — it applies onto the pristine pin
  standalone and in either order relative to the admission
  patch, and the byte-exact integrity check covers all
  three). Env `LLAMA_MOE_TAIL_SKIP`, default off.
- **The rule:** in a DSpark verify batch (the decode-window
  path, multi-row), a missing expert routed **only at rank 7**
  — the last of the top-8 picks — by every row of the batch is
  skipped for that step: not installed, and its contribution
  is exactly zero. Any row routing it at ranks 0–6 installs it
  normally. Prefill is never skipped (it does not run in
  window mode) and neither is single-row decode.
- **How zero is realized:** rank is read in the slot runtime
  from the flattened id tensor (`rank = pos % n_used`, the
  same identity the fork's A2 deferral uses). The graph reads
  expert weights only through the runtime's slot ids, so a
  skipped position's slot id is pointed at a **reserved zero
  slot** — the last slot of the layer's window, evicted and
  memset to zero (gate/up/down) when first armed in a window
  phase, then excluded from every allocation scan until the
  phase ends. `silu(0)*0 = 0` and `down @ 0 = 0`, so the
  router weight multiplies an exact zero. A skipped expert is
  temporarily marked resident at that slot for the demand
  loop and unmarked right after, and it is subtracted from
  the unique count so the hit/miss accounting stays truthful.
- **Disclosed cost:** while the zero slot is armed the window
  has 31 usable slots, not 32. A later verify batch in the
  same generation needing 32 installs with no skippable
  expert would hit the runtime's fit guard; observed verify
  batches demand about 17–24 unique experts per layer.
- **New arm** `quick_dspark_tailskip` (button "DSpark
  tail-skip"): the stock DSpark configuration plus
  `LLAMA_MOE_TAIL_SKIP=1`. Reports carry `moeTailSkip`
  (on/off) in every DSpark arm, and the runtime's teardown
  line adds `moeTailSkipBatches` (verify batches with at
  least one skip), `moeTailSkipSkipped` (experts skipped)
  and `moeTailSkipBytesNotReadMiB` (skips × per-expert
  bytes). summary.txt carries the flag, the telemetry, and
  the probe verdict.
- **Structured-output probe (this arm only):** after Case 3
  the arm runs a seventh case (96-token cap) demanding a
  strict JSON-only object about the bicycle case (keys
  `forces`, `balance_factors`, `summary`). The service
  validates the answer with org.json and records
  `moeTailSkipProbeJsonValid`; the answer text and token
  count ride along as `moeTailSkipProbeAnswer` /
  `moeTailSkipProbeTokens`. No other arm runs the probe, and
  the full trial is unchanged.
- **Judging:** this lever changes outputs by design, so there
  is no token-ID gate on this arm — Case 3's token IDs are
  expected to differ from the stock sequence. The bars are
  the plan's: the decode pair mean must beat the stock pair
  references by more than 20% (clean stock pair mean
  2.340 tok/s, builds 22/23), the JSON probe must be valid,
  and Case 3's answer gets a fluency read. With the flag off
  the runtime is behavior-identical, so the control arm
  (pair already complete: 6,099 installs / 2,729 hits) stays
  the regression check if it is run again.

**Device protocol for build 27:** install over build 26, do
NOT tap Download (the model pack is unchanged). On a cool
phone, run the **DSpark tail-skip** quick check twice; each
run exports its bundle as usual — attach the zips. No new
control runs are needed.

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
