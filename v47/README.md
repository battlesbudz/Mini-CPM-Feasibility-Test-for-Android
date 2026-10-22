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

## Build

CI (`.github/workflows/v47.yml`) builds `:v47:assembleRelease` on GitHub
Actions and uploads the APK as a workflow artifact (no GitHub Release).
Locally the same steps are: `python3 scripts/prepare_v47.py` then
`./gradlew :v47:assembleRelease`. The APK is signed with the repo's
feasibility-test key, like the other modules.

## Run

1. Install the APK, open **MiniCPM-V 4.7 Trial**.
2. Tap **Download model pack · 16.6 GB** on Wi-Fi with ≥ 18 GB free.
3. Tap **Run phone trial**. Keep the phone awake and on the charger;
   a slow first run is expected (expert paging is storage-bound).
4. Read the verdict and answers on screen, or copy diagnostics.

## Known unknowns this trial is designed to resolve

- Whether the fork's LES slot runtime behaves on Android's CPU backend
  as it does on the vendor's validated iOS/Metal path (all published
  numbers are iPhone numbers; no Android result exists upstream).
- Whether Android's per-app memory limits let the load + streaming
  steady-state survive on a 12 GB device (the vendor floor for this
  model) — peak PSS/RSS in the report answers this with data.
