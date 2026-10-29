// MiniCPM-V 4.7 phone trial — native benchmark.
//
// Loads the phone-only `-les` split of MiniCPM-V 4.7 35B-A3B through the
// llama.cpp-omni fork's LES / MoE-SSD runtime (routed experts streamed from
// the sibling experts/ directory via llama_model_params.moe_experts_path),
// then runs fixed benchmark cases and reports measured timings, token
// rates, peak RSS and the verbatim answers as JSON. Cases, in run order:
// three text-only questions, then the original three questions about one
// bundled street photo (build 10). A text-only case skips the bitmap load
// and the whole mmproj/mtmd path: its prompt is tokenized with
// llama_tokenize and prefilled through llama_decode, then shares the
// greedy generation loop with the image cases. Text cases run first
// because on device the first image case alone can consume the whole
// watchdog window (PHONE_TRIAL.md section 13), so their measurements must
// land before that. The vision projector is therefore loaded lazily, at
// the first image case. Build 10 also emits one progress checkpoint per
// evaluated chunk in the image path (chunk index / total), so a future
// image run shows whether chunks advance (a crawl) or freeze on a single
// chunk (a deadlock).
//
// Build 11 changes exactly one thing: the MoE decode-window expert cache
// is ON, with prewarm. The §16 diagnosis (PHONE_TRIAL.md) proved build
// 10's decode re-read ~412 MB of experts from flash for every generated
// token because the fork's decode cache was off by default and nothing
// was ever prewarmed (0.274 tok/s). Build 11 sets LLAMA_MOE_DECODE_WINDOW
// before model load and calls llama_moe_prewarm_experts() before every
// case's prefill, using the vendor app's own static w16 table. Prompts,
// cases, sampling, threads, context, the vision path and the model pack
// are all untouched.
//
// Build 12 stacks the streaming-research additions on top of build 11
// (STREAMING_RESEARCH.md sections 2.1, 2.2 and 3): lookahead prefetch
// ON (LLAMA_MOE_LOOKAHEAD_PREFETCH=1, default depth 8), the shared
// prefill bank raised 256 -> 1024 slots, a prewarm table mined from
// this app's own previous-run routing trace (LLAMA_MOE_SWEEP) when one
// exists, with the vendor w16 table kept as the fallback, the runtime's
// own [moe-ssd] counters parsed out of the captured log into the
// status and the report, a per-case CASE_DONE event the service turns
// into an incremental partial report, and a quick mode (Case 3 only,
// 32-token decode cap) the service selects with the run-mode argument.
// One research item is deliberately NOT here: retaining the slot cache
// across image chunks. In this runtime the prefill bank is a single
// pool time-multiplexed across layers, and the per-call layer-cache
// clear is what keeps a layer's residency map from going stale while
// other layers overwrite the same physical slots; removing the clear
// would silently corrupt weights (the fork documents exactly that bug
// class at the prefill-to-decode transition). Cross-chunk reuse needs
// per-layer prefill windows, a fork redesign, not a trial patch.
//
// Build 13 changes no configuration and no behavior; it scores the
// lookahead predictions. The section 21 diagnosis (PHONE_TRIAL.md)
// proved the runtime's hits counter is structurally unreachable in
// window mode, so prediction accuracy was measured by nothing. The
// trial patch now also carries two atomic counters in the slot
// runtime (patch part 2): in process(), where the previous layer's
// stored prediction for a layer is consumed, the raw predicted set is
// scored against the set that layer actually requests in the same
// decode step. The runtime prints the counts whole at teardown on a
// "lookahead-accuracy:" line, parsed below into the report as
// moeLookaheadScored / moeLookaheadCorrect / moeLookaheadAccuracyPct.
// Because the numbers come from that one teardown line (the atomic
// counters read directly), they cover the whole run — unlike the
// per-step flush-line sums, which the runtime throttles (section 21.5).
//
// Build 14 adds the A/B test build 13 left open: is lookahead
// prefetch a net speed win on this CPU path, or does its I/O and
// thread contention cost more than the (now proven 84%-accurate)
// guesses save? A second quick-mode entry, mode "quick_off", runs
// Case 3 with the same 32-token cap and the identical configuration
// except one value: LLAMA_MOE_LOOKAHEAD_PREFETCH is set to "0"
// instead of "1" before load — the fork's own off switch, no patch.
// The report carries moeLookaheadPrefetch "off" and abArm
// "lookahead_off" so the two arms can never be confused in a pasted
// diagnostics dump; the ON arm is byte-for-byte build 13's quick
// behavior (abArm "lookahead_on" is the only added field). In the
// OFF arm the runtime prints no lookahead-accuracy line at all
// (nothing is predicted, so nothing is scored), and the report
// simply has no moeLookaheadScored/Correct/AccuracyPct fields —
// absent, not zero-filled.
//
// Build 15 adds the DSpark A/B arm: does speculative decoding with
// the model's own DSpark draft heads beat plain greedy decode on
// this CPU path? A third quick-mode entry, mode "quick_dspark",
// runs the exact same Case 3 / 32-token quick run, but generation
// goes through the fork's common speculative machinery
// (common/speculative.cpp, type draft-dspark) with the vendor's
// DSpark draft GGUF (openbmb/MiniCPM-V-4.7-35B-A3B-DSpark-gguf,
// Q8_0; dflash arch, block_size 15, target layers 4/12/20/28/36).
// Configuration follows the vendor iOS bridge pattern, all set
// before the target model loads: lookahead prefetch OFF (the
// bridge disables MoE prediction whenever DSpark is on), decode
// window widened 16 -> 64 with LLAMA_MOE_WINDOW_MAX_ROWS=8 so a
// verify batch of n_max+1 = 8 rows stays in window mode, target
// context created with n_rs_seq = 7 so recurrent-state rollback
// covers a full round of rejected drafts, draft n_max = 7.
// (Builds 15-16 ran this arm at n_max = 3 with window 32 / rows 4;
// build 17 widens only the draft depth and its window, nothing
// else in the arm changes.)
// Rollback is probed once at empty KV (common_context_can_seq_rm,
// the fork server's own probe): unless it returns RS or PART, the
// arm fails safe — the report says DSpark disabled itself and why,
// and no decode number is produced. Acceptance is greedy-exact
// (top_k=1 sampler + common_sampler_sample_and_accept_n: every
// emitted token is the target's own batched argmax), and every
// arm records generatedTokenIds. DSpark output is judged on
// fluency and decode speed, not token-for-token identity with
// the baseline arms — the runtime does not guarantee bit-identical
// logits across batch sizes (Battles's call, 2026-10-28). The
// case also carries the driver's own spec counters (rounds /
// drafted / accepted / accept rate / average
// acceptance length) plus the runtime's print_stats line captured
// from the redirected common log. The fork patch is untouched;
// this is app-side only, linking the already-built llama-common.
//
// Build 18 changes no configuration: the DSpark arm stays at
// build 17's widened settings (n_max = 7, window 64, rows 8,
// n_rs_seq = 7). It wraps the DSpark round loop in a try/catch.
// The section 41 diagnosis (PHONE_TRIAL.md) proved build 17's
// widened arm kills the :benchmark process natively during an
// early verify round — the in-process memory sampler and the
// 10-minute watchdog froze with it, and no report was ever
// written — and named the leading vehicle an uncaught C++
// exception (an MoE slot-runtime throw): there is no handler
// anywhere from the ggml custom op through llama_decode to this
// driver. The catch converts such a throw into the driver's
// existing fatal path (errorPhase "dspark_exception"), with the
// exception's what() text kept in the case as
// dsparkCaughtException, so the run ends with a finished report
// naming the throwing site instead of a silent process death.
// A GGML_ASSERT abort cannot be caught this way: if a build 18
// DSpark run still dies silently, that convicts the assert
// branch instead. The quick and quick_off arms are untouched.
//
// Configuration mirrors the vendor demo defaults for this fork
// (OpenBMB MiniCPM-V-demo-Android): 4 threads, 4096-token context,
// image_max_slice_nums 9. Sampling is greedy (temperature 0) to match the
// desktop step-1 scorecard methodology, and thinking is disabled by closing
// the think block in the prompt, which is exactly what the model's chat
// template does with enable_thinking=false (documented in the upstream
// llama.cpp MiniCPM-V 4.7 notes).
#include <jni.h>
#include <android/log.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <sys/resource.h>
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "nlohmann/json.hpp"
// Build 15: the fork's common speculative-decoding machinery (the
// trial CMake already builds llama-common; build 15 links it).
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"

using Clock = std::chrono::steady_clock;
using json = nlohmann::ordered_json;

static const char * RUNTIME_COMMIT = "96bbf61710d44421ea54e63f5194f9b17c20d877";
static const char * MODEL_SUBDIR   = "MiniCPM-V-4.7-35B-A3B";
static const char * MAIN_GGUF      = "MiniCPM-V-4_7-35B-A3B-mix_q3gud_im-les.gguf";
static const char * MMPROJ_GGUF    = "mmproj-model-f16.gguf";
static const int    N_THREADS      = 4;
// LES contract in the pinned fork: llama_model_params.moe_experts_path and
// moe_slot_bank MUST be set together (include/llama.h: "Both fields must be
// set together"). With the path set and the bank left at its default of 0,
// the llama_model constructor throws std::invalid_argument, which
// llama_model_load catches and converts into a null model return — that was
// the build-5 device failure (null in ~170 ms, before any tensor loads).
// Builds 5-11 used 256, the vendor demo's own default for this model
// (MiniCPMV47Const.defaultSlotBank). Build 12 raises the bank to 1024:
// - The fork's hard cap is 32 x n_moe_layers = 32 x 40 = 1280 slots
//   (llama-model.cpp, kMoeSlotBankRamBaseline); 1024 stays under it.
// - The bank is ONE shared pool, resident cost = slots x 1,351,680 B
//   per-slot-per-layer bytes: 1024 x 1,351,680 B = ~1.29 GiB (was
//   ~330 MiB at 256). With build 11/12's decode windows (below) total
//   expert residency is ~2.50 GiB, toward the vendor's ~3 GB working
//   set for this model.
// - Headroom check against build 10's device run: worker PSS peaked
//   ~3.02 GB with the 256 bank and no windows. Swapping in this bank
//   (+~0.97 GiB) and the windows with prefetch staging (+~1.21 GiB)
//   estimates a worst-case worker PSS of ~5.2 GB on the 12 GB Fold6,
//   leaving ~1.3 GB of system-free margin by build 10's observed
//   split; the run's memory.json lowMemory flag is the on-device
//   check. 1280 was rejected for a first run of this combination: it
//   buys capacity current workloads cannot use (a layer call routes
//   ~84 unique experts per image chunk, ~200 max on the text prompts)
//   for ~0.3 GiB more margin loss.
// - Honest bound: with the fork's per-call layer-cache clear left in
//   place (see the build 12 header note), the bank is within-call
//   capacity — a single layer call can now hold up to 1024 unique
//   experts without eviction or the fork's "requires more than N
//   unique experts" throw. It does not by itself create cross-chunk
//   cache hits.
static const int    MOE_SLOT_BANK  = 1024;
// Build 11: per-layer decode-window width, in expert slots
// (LLAMA_MOE_DECODE_WINDOW, read by the fork at model load to build the
// per-layer decode tensor layout, and again by the slot runtime; the
// fork clamps it to [n_expert_used, 1024] = [8, 1024] for this model).
// The decode windows are physically separate from the prefill bank, so
// the resident cost is additive. 16 is the vendor app's matched pairing
// with its w16 prewarm table (below). Build 12 turns lookahead
// prefetch ON (LLAMA_MOE_LOOKAHEAD_PREFETCH=1, default depth 8): the
// model-side tensor layout then adds the fork's prefetch staging of
// depth slots per layer (moe_prefetch_slots()), so the physical width
// is 16 + 8 = 24 slots/layer and the resident cost is 40 layers x 24 x
// 1,351,680 B = ~1.21 GiB (was ~825 MiB in build 11 with prefetch off).
// The cache portion a token can hit stays 16/layer either way.
static const int    MOE_DECODE_WINDOW = 16;
// Build 12: lookahead prefetch depth is left at the fork's default (8)
// — LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH is never set. Depth is also the
// per-layer prefetch staging width counted in the sizing above.
// Build 12 quick mode: run only this case (1-based display index; the
// long bicycle text case, build 10/11's decode measurement) with the
// decode cap below. Selected by the service via the run-mode argument;
// the full trial's case list and defaults are unchanged.
static const int    QUICK_CASE_DISPLAY = 3;
static const int    QUICK_MAX_TOKENS   = 32;
// Build 15 (DSpark arm): the draft GGUF sits beside the main gguf in
// the model directory (downloaded from the DSpark repo via the
// per-file source override in models.json / ModelStore).
static const char * DSPARK_GGUF    = "dspark-MiniCPM-V-4_7-35B-A3B-Q8_0.gguf";
// Draft tokens per verify round. The draft GGUF's trained block is
// 15 wide (dflash.block_size), so 7 is inside the trained range;
// it also keeps a round's worst-case rollback at 7 tokens, exactly
// the target context's n_rs_seq below. The vendor's published
// acceptance length (3.501 at block 15) is a general-text average,
// not a promise for this prompt. (Builds 15-16: n_max = 3.)
static const int    DSPARK_N_MAX   = 7;
// DSpark arm decode window: widened from 16 to 64 so the clamp in
// the slot runtime leaves room for LLAMA_MOE_WINDOW_MAX_ROWS=8
// (n_max + 1 verify rows) — worst-case union for 8 rows is
// 8 rows x 8 routed experts = 64 slots, exactly the window width,
// so no clamp applies. Resident cost grows to 40 layers x 64 x
// 1,351,680 B = ~3.3 GiB of window (from ~1.65 GiB at width 32 in
// builds 15-16, ~825 MiB at width 16 with prefetch off); lookahead
// prefetch is OFF in this arm, so no prefetch staging is added on
// top.
static const int    DSPARK_DECODE_WINDOW   = 64;
static const int    DSPARK_WINDOW_MAX_ROWS = DSPARK_N_MAX + 1;
// The vendor app's static prewarm table for this model, top-16 experts
// per layer, ranked by observed routing frequency (k_mb_prewarm_w16 in
// OpenBMB/MiniCPM-V-Apps, MTMDWrapper/Bridge/MBPrewarmTable.h, generated
// from their tables_w.json). The slot runtime's parser skips '#' lines;
// every other line is one model layer, in order. Written to the model
// directory at run start and handed to the runtime via LLAMA_MOE_PREWARM,
// exactly the vendor app's own mechanism.
static const char * MOE_PREWARM_TABLE_W16 =
R"PREWARM(# Prewarm table: one line per model layer, top-16 experts per layer (vendor tables_w.json)
151,43,203,231,222,70,106,187,217,31,55,30,33,66,147,177
233,211,128,19,236,123,141,48,70,107,242,238,124,146,190,232
49,3,39,58,164,169,188,90,11,43,217,221,20,60,133,23
172,199,159,125,44,167,62,94,163,110,177,223,117,16,90,92
154,192,54,80,160,204,166,79,109,58,88,243,206,30,33,34
165,202,102,158,155,205,69,54,77,147,178,46,182,218,139,52
15,56,232,79,37,236,196,100,36,65,84,149,108,138,191,227
35,14,202,76,147,32,173,38,171,116,23,36,191,130,166,121
180,194,183,61,77,110,173,232,3,4,51,116,215,84,227,46
86,197,98,198,0,138,59,104,44,196,102,93,161,218,120,15
196,234,187,111,239,81,39,142,242,99,246,146,247,157,194,228
107,3,225,69,235,106,111,37,145,218,7,56,93,181,147,160
35,15,128,1,39,193,46,219,139,172,220,49,223,107,191,108
32,249,160,125,99,19,13,175,219,134,224,113,28,191,182,154
52,51,64,172,137,22,160,106,167,174,131,212,19,32,129,130
213,4,162,54,165,242,135,87,122,58,5,104,120,250,94,118
150,32,252,222,170,199,28,80,102,185,98,116,243,53,7,205
48,64,230,87,200,16,173,147,20,201,212,214,232,41,242,15
84,115,155,251,135,13,204,134,158,43,192,142,174,4,122,203
30,230,112,168,117,42,26,109,86,228,20,35,53,85,229,4
51,194,4,110,115,183,61,180,191,3,116,227,185,232,19,63
86,161,197,102,138,198,200,93,208,196,120,34,98,104,66,245
234,196,187,111,239,39,242,155,157,146,99,142,228,106,159,194
107,3,111,145,7,93,106,218,194,56,70,235,225,69,195,6
35,1,15,237,39,46,193,219,81,49,108,107,11,125,96,220
32,249,130,13,125,66,134,245,113,191,219,175,19,229,29,124
137,131,52,159,54,106,51,67,118,16,130,77,19,172,174,212
162,4,94,54,135,5,104,165,120,58,63,41,110,242,204,28
170,252,222,150,199,98,179,102,181,73,185,26,80,53,149,152
230,48,200,212,147,20,24,201,19,41,188,220,242,21,173,101
84,135,251,177,192,155,115,174,158,196,204,134,29,248,4,122
30,230,86,112,42,229,228,117,14,168,253,177,82,76,187,226
239,72,178,207,65,176,126,141,50,249,55,169,69,153,253,200
149,16,46,120,1,249,27,67,216,230,8,125,229,222,65,154
189,145,79,127,178,102,33,148,60,251,40,47,119,190,78,35
175,28,92,247,112,29,158,243,10,1,63,134,205,50,25,159
184,38,108,104,182,175,176,209,221,18,113,215,84,76,162,90
247,203,157,155,48,79,41,97,20,251,197,229,231,249,217,47
26,221,51,53,208,2,240,97,170,1,178,211,244,75,79,157
156,135,51,172,82,59,214,143,188,3,26,207,54,121,43,76
)PREWARM";
static const int    N_CTX          = 4096;
static const int    N_BATCH        = 2048;
static const int    MAX_NEW_TOKENS = 512;

static double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
static long peak_rss_kb() {
    struct rusage ru;
    return getrusage(RUSAGE_SELF, &ru) == 0 ? (long) ru.ru_maxrss : -1;
}
static std::string jstr(JNIEnv * env, jstring value) {
    const char * p = env->GetStringUTFChars(value, nullptr);
    std::string out(p);
    env->ReleaseStringUTFChars(value, p);
    return out;
}
// The llama/ggml loader explains its failures only through this log stream
// (llama_model_load_from_file itself just returns null), so accumulate the
// stream and ship it in the report: a failed run must say where it died in
// the loader's own words. Capped so a noisy run cannot bloat the report.
static std::string g_loader_log;
static std::mutex g_loader_log_mutex;
static const size_t LOADER_LOG_MAX = 256 * 1024;

static void log_callback(ggml_log_level level, const char * text, void *) {
    __android_log_print(level == GGML_LOG_LEVEL_ERROR ? ANDROID_LOG_ERROR : ANDROID_LOG_INFO,
                        "MiniCPMV47", "%s", text);
    if (text) {
        std::lock_guard<std::mutex> lock(g_loader_log_mutex);
        if (g_loader_log.size() < LOADER_LOG_MAX) {
            g_loader_log += text;
        }
    }
}
static std::string loader_log_snapshot() {
    std::lock_guard<std::mutex> lock(g_loader_log_mutex);
    if (g_loader_log.size() >= LOADER_LOG_MAX) {
        return g_loader_log + "\n[loader log truncated at 256 KiB]\n";
    }
    return g_loader_log;
}

// Build 12: mine a prewarm table from a previous run's routing trace.
// The slot runtime's LLAMA_MOE_SWEEP dump writes one line per decode
// step per layer: "L<layer> E:<experts actually used> P:<predicted>".
// Routing depends only on the real weights and prompts, so the
// per-layer frequency of the E: ids over a previous run IS this app's
// own routing distribution; the top-16 per layer is exactly what the
// width-16 decode window wants prewarmed. Returns the table text in
// the runtime's parser format (one line per layer, '#' lines skipped),
// or "" when the trace is missing or too thin to trust: fewer than
// 400 decode lines (10 steps x 40 layers) or any layer unseen. The
// "P<layer>" prefill lines the runtime also writes are ignored — only
// decode routing seeds a decode cache.
static std::string mine_prewarm_table(const std::string & sweep_path, long & lines_out) {
    lines_out = 0;
    FILE * fp = std::fopen(sweep_path.c_str(), "r");
    if (!fp) return "";
    static const int N_LAYERS = 40, N_EXPERTS = 256;
    long counts[N_LAYERS][N_EXPERTS] = {};
    long layer_lines[N_LAYERS] = {};
    char buf[8192];
    long total = 0;
    while (std::fgets(buf, sizeof(buf), fp)) {
        if (buf[0] != 'L') continue;
        char * p = buf + 1;
        char * end = nullptr;
        const long layer = std::strtol(p, &end, 10);
        if (end == p || layer < 0 || layer >= N_LAYERS) continue;
        const char * e = std::strstr(end, "E:");
        if (!e) continue;
        e += 2;
        layer_lines[layer]++;
        total++;
        while (*e) {
            while (*e == ' ' || *e == ',' || *e == '\t') e++;
            if (*e == 'P' && e[1] == ':') break; // predictions start here
            if (!*e || *e == '\n' || *e == '\r') break;
            char * idend = nullptr;
            const long id = std::strtol(e, &idend, 10);
            if (idend == e) break;
            if (id >= 0 && id < N_EXPERTS) counts[layer][id]++;
            e = idend;
        }
    }
    std::fclose(fp);
    if (total < 400) return "";
    for (int l = 0; l < N_LAYERS; l++) if (layer_lines[l] == 0) return "";
    lines_out = total;
    std::string out = "# Prewarm table mined from this app's previous run (LLAMA_MOE_SWEEP trace):\n"
                      "# one line per model layer, top-16 experts by observed decode routing frequency.\n";
    for (int l = 0; l < N_LAYERS; l++) {
        std::array<int, N_EXPERTS> ids;
        for (int i = 0; i < N_EXPERTS; i++) ids[i] = i;
        std::sort(ids.begin(), ids.end(), [&](int a, int b) {
            return counts[l][a] != counts[l][b] ? counts[l][a] > counts[l][b] : a < b;
        });
        for (int i = 0; i < 16; i++) {
            if (i) out += ",";
            out += std::to_string(ids[i]);
        }
        out += "\n";
    }
    return out;
}

// Build 12: the runtime's own counters, recovered from its [moe-ssd]
// log lines (the trial already captures the full llama log stream).
// The per-ubatch flush lines carry DELTAS (miss/hit/pfhit/pfread/
// readall/calls per flush window; the runtime resets its accumulator
// after each flush), so summing the lines yields run totals: misses
// are main-path expert installs, hits are slot-cache hits, and the
// lookahead report line ("armed=.. hits=.. (X% of predictions)") is
// cumulative and taken whole. Approximation by design: the captured
// log is capped (LOADER_LOG_MAX), so on a very long run the totals
// undercount — the report flags that via moeCountersTruncated.
struct MoeCounters {
    long long miss = 0, hit = 0, pfRead = 0, pfHit = 0, ioCalls = 0;
    double ioMiB = 0.0;
    bool lookaheadSeen = false;
    long long laArmed = 0, laHits = 0;
    double laPct = 0.0;
    // Build 13: prediction accuracy, from the teardown
    // "lookahead-accuracy:" line (whole-run atomic counters, not
    // summed flush deltas, so no section 21.5 throttling applies).
    // The percentage is recomputed here from the raw counts.
    bool lookaheadAccuracySeen = false;
    long long laScored = 0, laCorrect = 0;
    double laAccuracyPct = 0.0;
    // Build 15: effective decode-window batch width after the slot
    // runtime's clamp, from its "decode-cache: 窗口覆盖批宽 1..N"
    // line (printed once at slot-runtime init). -1 = line not seen.
    long long windowRows = -1;
};
static bool log_field_ll(const std::string & line, const char * key, long long & out) {
    const std::string k = std::string(key) + "=";
    const size_t pos = line.find(k);
    if (pos == std::string::npos) return false;
    out = std::strtoll(line.c_str() + pos + k.size(), nullptr, 10);
    return true;
}
static bool log_field_d(const std::string & line, const char * key, double & out) {
    const std::string k = std::string(key) + "=";
    const size_t pos = line.find(k);
    if (pos == std::string::npos) return false;
    out = std::strtod(line.c_str() + pos + k.size(), nullptr);
    return true;
}
static MoeCounters parse_moe_counters(const std::string & log) {
    MoeCounters c;
    size_t pos = 0;
    while (pos < log.size()) {
        const size_t eol = log.find('\n', pos);
        const std::string line = log.substr(pos, eol == std::string::npos ? std::string::npos : eol - pos);
        pos = eol == std::string::npos ? log.size() : eol + 1;
        if (line.find("[moe-ssd]") == std::string::npos) continue;
        long long v;
        double d;
        if (line.find(" tok=") != std::string::npos && line.find(" span=") != std::string::npos) {
            if (log_field_ll(line, "miss", v)) c.miss += v;
            if (log_field_ll(line, "hit", v)) c.hit += v;
            if (log_field_ll(line, "pfhit", v)) c.pfHit += v;
            if (log_field_ll(line, "pfread", v)) c.pfRead += v;
            if (log_field_ll(line, "calls", v)) c.ioCalls += v;
            if (log_field_d(line, "readall", d)) c.ioMiB += d;
        }
        if (line.find("lookahead-prefetch: armed=") != std::string::npos) {
            long long armed = 0, hits = 0;
            if (log_field_ll(line, "armed", armed) && log_field_ll(line, "hits", hits)) {
                c.lookaheadSeen = true;
                c.laArmed = armed;
                c.laHits = hits;
                const size_t paren = line.find('(', line.find("hits="));
                c.laPct = paren == std::string::npos ? 0.0 : std::strtod(line.c_str() + paren + 1, nullptr);
            }
        }
        if (line.find("lookahead-accuracy: scored=") != std::string::npos) {
            long long scored = 0, correct = 0;
            if (log_field_ll(line, "scored", scored) && log_field_ll(line, "correct", correct)) {
                c.lookaheadAccuracySeen = true;
                c.laScored = scored;
                c.laCorrect = correct;
                c.laAccuracyPct = scored > 0 ? 100.0 * (double) correct / (double) scored : 0.0;
            }
        }
        if (line.find("decode-cache:") != std::string::npos) {
            const size_t dots = line.find("1..");
            if (dots != std::string::npos) {
                const long long rows = std::strtoll(line.c_str() + dots + 3, nullptr, 10);
                if (rows > 0) c.windowRows = rows;
            }
        }
    }
    return c;
}

static llama_token greedy_token(llama_context * ctx, const llama_vocab * vocab) {
    const float * logits = llama_get_logits_ith(ctx, -1);
    const int n = llama_vocab_n_tokens(vocab);
    int best = 0;
    float best_value = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > best_value) { best_value = logits[i]; best = i; }
    }
    return (llama_token) best;
}
static std::string token_piece(const llama_vocab * vocab, llama_token token) {
    char buf[256];
    int n = llama_token_to_piece(vocab, token, buf, (int32_t) sizeof(buf), 0, true);
    if (n < 0) {
        std::vector<char> big((size_t) -n);
        n = llama_token_to_piece(vocab, token, big.data(), (int32_t) big.size(), 0, true);
        return std::string(big.data(), (size_t) n);
    }
    return std::string(buf, (size_t) n);
}

extern "C" JNIEXPORT jstring JNICALL Java_com_battlesbudz_minicpmv47_NativeBench_run(
        JNIEnv * env, jclass, jstring jroot, jstring jimage, jstring jout, jstring jmode, jobject listener) {
    const std::string root  = jstr(env, jroot);
    const std::string image = jstr(env, jimage);
    const std::string out   = jstr(env, jout);
    (void) out; // report is returned to the service, which persists it
    // Build 12: run mode from the service. "quick" runs only Case 3
    // (QUICK_CASE_DISPLAY) with a QUICK_MAX_TOKENS decode cap; anything
    // else is the full six-case trial with untouched defaults.
    const std::string mode  = jmode ? jstr(env, jmode) : std::string("full");
    // Build 14: "quick_off" is the A/B lookahead-OFF arm of quick mode
    // (see the build 14 header note). Everything else about the run —
    // case, cap, watchdog (service side) — is quick mode unchanged.
    const bool lookahead_off = mode == "quick_off";
    // Build 15: "quick_dspark" is the DSpark A/B arm of quick mode
    // (see the build 15 header note): same case, cap and watchdog,
    // generation through the speculative driver instead of the
    // greedy loop, with its own pre-load configuration below.
    const bool dspark = mode == "quick_dspark";
    const bool quick = mode == "quick" || lookahead_off || dspark;
    const int max_tokens = quick ? QUICK_MAX_TOKENS : MAX_NEW_TOKENS;
    auto listener_cls = env->GetObjectClass(listener);
    auto on_event = env->GetMethodID(listener_cls, "onEvent", "(Ljava/lang/String;)V");
    auto emit = [&](const std::string & s) {
        jstring js = env->NewStringUTF(s.c_str());
        env->CallVoidMethod(listener, on_event, js);
        env->DeleteLocalRef(js);
    };
    // Build 9 diagnostic instrumentation (observability only, no model
    // behavior changes): live stage checkpoints. The native run is one
    // blocking JNI call, and before build 9 the only in-case events were
    // the case-start line and a token count every 32 tokens, so a stall
    // inside vision encode, prefill or a slow decode was indistinguishable
    // from a hard hang (build 8, Case 1, 30 minutes, no further events).
    // Checkpoints go out as "PROGRESS {json}" events; the service parses
    // them and merges the fields into status.json, stamping wall-clock
    // times itself (stageStartedAtMs on a stage start, lastProgressAtMs
    // on every checkpoint). Payload keys: stage, event (start / progress
    // / first_token / done), caseIndex, label (readable phase text), plus
    // per-stage measurements (modelLoadMs, mmprojLoadMs, contextMs,
    // tokenizeMs, visionMs, prefillMs, promptTokens, imageTokens,
    // generatedTokens, decodeElapsedMs, decodeTokensPerSec). The plain
    // phase events above are unchanged.
    auto emit_progress = [&](const json & fields) {
        emit("PROGRESS " + fields.dump());
    };

    // Build 12 (research section 3, item 5): surface the runtime's own
    // [moe-ssd] counters so every optimization above is provable from a
    // single run. Parsed from the captured loader log (parse_moe_counters)
    // and emitted as a checkpoint after every case, so status.json shows
    // the running totals; the same totals land in the final report.
    auto moe_counters_json = [&]() {
        const MoeCounters c = parse_moe_counters(loader_log_snapshot());
        const double rate = (c.hit + c.miss) > 0 ? 100.0 * (double) c.hit / (double) (c.hit + c.miss) : 0.0;
        json fields = {
            {"moeExpertInstalls", c.miss},
            {"moePrefetchReads", c.pfRead},
            {"moeCacheHits", c.hit},
            {"moeCacheMisses", c.miss},
            {"moeCacheHitRatePct", rate},
            {"moePrefetchHits", c.pfHit},
            {"moeIoReadMiB", c.ioMiB},
            {"moeIoCalls", c.ioCalls},
        };
        if (c.lookaheadSeen) {
            fields["moeLookaheadArmed"] = c.laArmed;
            fields["moeLookaheadHits"] = c.laHits;
            fields["moeLookaheadHitRatePct"] = c.laPct;
        }
        if (c.lookaheadAccuracySeen) {
            fields["moeLookaheadScored"] = c.laScored;
            fields["moeLookaheadCorrect"] = c.laCorrect;
            fields["moeLookaheadAccuracyPct"] = c.laAccuracyPct;
        }
        if (c.windowRows >= 0) {
            fields["moeWindowMaxRowsEffective"] = c.windowRows;
        }
        return fields;
    };
    auto emit_counters = [&]() {
        json fields = moe_counters_json();
        fields["stage"] = "moe_counters";
        fields["event"] = "progress";
        fields["label"] = "MoE expert counters";
        emit_progress(fields);
    };

    json report = {
        {"schemaVersion", 2},
        {"profile", "minicpm-v47-phone-trial-v1"},
        {"runtime", "llama.cpp-omni (tc-mb fork, LES / MoE-SSD)"},
        {"runtimeCommit", RUNTIME_COMMIT},
        {"runtimePatch", "v47/patches/ggml-cpu-alloc-only-buft.patch (CPU alloc-only buft + llama-model.cpp loader fix; build 13 adds lookahead-accuracy telemetry counters in llama-moe-slot.cpp)"},
        {"modelRepo", "tc-mb/MiniCPM-V-Apps-gguf"},
        {"modelRevision", "8716e5209f308fd53f197ccd7d50c6fdcc2dc37d"},
        {"modelFiles", "main -les gguf + mmproj f16 + experts/ (40 shards + manifest.json)"},
        {"threads", N_THREADS},
        {"contextTokens", N_CTX},
        {"batchTokens", N_BATCH},
        {"mode", quick ? "quick" : "full"},
        {"maxNewTokens", max_tokens},
        {"sampling", "greedy (temperature 0)"},
        {"thinking", "disabled via closed think block (enable_thinking=false equivalent)"},
    };
    const auto trial_start = Clock::now();
    auto fail = [&](const std::string & phase, const std::string & message) -> jstring {
        emit_progress({{"stage", "failed"}, {"event", "done"}, {"errorPhase", phase},
                       {"label", "Failed at " + phase}});
        report["errorPhase"] = phase;
        report["error"] = message;
        const json counters = moe_counters_json();
        for (auto it = counters.begin(); it != counters.end(); ++it) report[it.key()] = it.value();
        report["moeCountersTruncated"] = loader_log_snapshot().size() >= LOADER_LOG_MAX;
        report["loaderLog"] = loader_log_snapshot();
        report["peakRssKb"] = peak_rss_kb();
        report["totalMs"] = ms_between(trial_start, Clock::now());
        emit("FAILED at " + phase + ": " + message);
        return env->NewStringUTF(report.dump(2).c_str());
    };

    llama_log_set(log_callback, nullptr);
    llama_backend_init();

    const std::string model_dir    = root + "/" + MODEL_SUBDIR;
    const std::string main_path    = model_dir + "/" + MAIN_GGUF;
    const std::string mmproj_path  = model_dir + "/" + MMPROJ_GGUF;
    const std::string experts_path = model_dir + "/experts";
    report["moeExpertsPath"] = experts_path;
    report["moeSlotBank"] = MOE_SLOT_BANK;

    // Builds 11/12: enable the decode-window expert cache, prewarm,
    // lookahead prefetch and the routing-trace dump BEFORE the model
    // loads — the fork reads LLAMA_MOE_DECODE_WINDOW and
    // LLAMA_MOE_LOOKAHEAD_PREFETCH on the model side at load time to
    // size the per-layer decode tensors, LLAMA_MOE_PREWARM when the
    // slot runtime initializes, and LLAMA_MOE_SWEEP at context
    // creation. Overwrite unconditionally, as the vendor app does, so
    // no stale value can leak in.
    // Build 15: the DSpark arm widens the decode window to 32 (see
    // DSPARK_DECODE_WINDOW above); the other arms keep build 11's 16.
    const int decode_window = dspark ? DSPARK_DECODE_WINDOW : MOE_DECODE_WINDOW;
    setenv("LLAMA_MOE_DECODE_WINDOW", std::to_string(decode_window).c_str(), 1);
    // Build 12 (research 2.1): lookahead prefetch ON, depth left at
    // the fork default of 8. The runtime predicts the next layer's
    // routing from the current hidden state and pre-reads those
    // experts during the current layer's compute; hits remap for free.
    // Build 14 A/B: the OFF arm flips the fork's own switch to "0".
    // That is the fork's documented off state: the slot runtime's
    // lookahead init reads depth 0 and disables the mechanism
    // (llama-moe-slot.cpp), the model graph then never computes
    // predictions (qwen35moe.cpp gates them on
    // lookahead_prefetch_enabled()), and moe_prefetch_slots()
    // (llama-model.cpp) returns 0, so no prefetch staging slots are
    // reserved in the decode tensors. The decode window (16 cache
    // slots/layer), the slot bank, prewarm, threads and every other
    // configuration value are set exactly as in the ON arm.
    // Build 15: the DSpark arm also sets the switch to "0" — the
    // vendor iOS bridge disables MoE lookahead prediction whenever
    // DSpark is on (the two predictors would fight over the same
    // I/O), and the DSpark baseline is the lookahead-OFF arm.
    setenv("LLAMA_MOE_LOOKAHEAD_PREFETCH", (lookahead_off || dspark) ? "0" : "1", 1);
    if (dspark) {
        // Widen the window-mode batch coverage to n_max + 1 rows so
        // a whole verify batch stays in the decode window (the slot
        // runtime clamps this against the window width; width 32
        // leaves it at 4, see the counters' moeWindowMaxRowsEffective).
        setenv("LLAMA_MOE_WINDOW_MAX_ROWS", std::to_string(DSPARK_WINDOW_MAX_ROWS).c_str(), 1);
    }
    report["moeLookaheadPrefetch"] = dspark ? "off" : (lookahead_off ? "off" : "on (depth default 8)");
    report["abArm"] = dspark ? "dspark_on" : (lookahead_off ? "lookahead_off" : "lookahead_on");
    if (dspark) {
        report["dsparkDraftModel"] = std::string(DSPARK_GGUF) +
            " (openbmb/MiniCPM-V-4.7-35B-A3B-DSpark-gguf @ 2c8f6931e5fba8a01bcc74b2964f330c62056b5e)";
        report["dsparkSpecType"] = "draft-dspark";
        report["dsparkDraftNMax"] = DSPARK_N_MAX;
        report["dsparkWindowMaxRows"] = DSPARK_WINDOW_MAX_ROWS;
        report["dsparkTargetNRsSeq"] = DSPARK_N_MAX;
    }
    // Build 12 (research 2.2/3): always record this run's routing
    // trace. The runtime truncates the file at context creation, so
    // the previous run's trace must be mined BEFORE the model loads
    // (the prewarm selection below) — after this run it holds this
    // run's trace for the next one.
    const std::string sweep_path = model_dir + "/moe_sweep.txt";
    setenv("LLAMA_MOE_SWEEP", sweep_path.c_str(), 1);
    report["moeSweepTrace"] = sweep_path;
    // Prewarm table selection. First choice: a table mined from the
    // previous run's sweep trace (this app's own routing on this
    // phone). No previous trace (or a too-thin one) falls back to the
    // vendor w16 table exactly as build 11 shipped it — so build 12's
    // FIRST run prewarms with w16 and records the trace; the mined
    // table takes over from the second run on. If neither file can
    // be written the window still turns on; prewarm then no-ops
    // inside the runtime and the report says so.
    std::string prewarm_path;
    std::string prewarm_desc;
    long sweep_lines = 0;
    const std::string mined = mine_prewarm_table(sweep_path, sweep_lines);
    if (!mined.empty()) {
        const std::string ours_path = model_dir + "/moe_prewarm_ours.txt";
        if (FILE * pf = std::fopen(ours_path.c_str(), "w")) {
            if (std::fputs(mined.c_str(), pf) >= 0) {
                prewarm_path = ours_path;
                prewarm_desc = "ours (mined from previous run sweep: " +
                               std::to_string(sweep_lines) + " decode-step layer lines)";
            }
            std::fclose(pf);
        }
    }
    if (prewarm_path.empty()) {
        const std::string w16_path = model_dir + "/moe_prewarm_w16.txt";
        if (FILE * pf = std::fopen(w16_path.c_str(), "w")) {
            if (std::fputs(MOE_PREWARM_TABLE_W16, pf) >= 0) {
                prewarm_path = w16_path;
                prewarm_desc = "w16 (vendor top-16/layer)";
            }
            std::fclose(pf);
        }
    }
    if (!prewarm_path.empty()) {
        setenv("LLAMA_MOE_PREWARM", prewarm_path.c_str(), 1);
    }
    report["moeDecodeWindow"] = decode_window;
    report["moePrewarmTable"] = prewarm_desc.empty() ? "WRITE FAILED - prewarm off" : prewarm_desc;
    emit("MoE decode window on: " + std::to_string(decode_window) +
         " slots/layer, lookahead prefetch " + std::string((lookahead_off || dspark) ? "off" : "on") + ", prewarm table " +
         (prewarm_desc.empty() ? std::string("FAILED to write") : prewarm_desc));

    emit("Loading MiniCPM-V 4.7 main model (LES expert streaming)");
    emit_progress({{"stage", "model_load"}, {"event", "start"}, {"label", "Loading main model"}});
    const auto load_start = Clock::now();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.use_mmap = true;
    mparams.moe_experts_path = experts_path.c_str();
    mparams.moe_slot_bank = MOE_SLOT_BANK;
    llama_model * model = llama_model_load_from_file(main_path.c_str(), mparams);
    if (!model) return fail("model_load", "llama_model_load_from_file returned null for " + main_path);
    const double load_model_ms = ms_between(load_start, Clock::now());
    report["loadModelMs"] = load_model_ms;
    emit_progress({{"stage", "model_load"}, {"event", "done"}, {"label", "Main model loaded"},
                   {"modelLoadMs", load_model_ms}, {"stageElapsedMs", load_model_ms}});
    {
        char desc[256];
        llama_model_desc(model, desc, sizeof(desc));
        report["modelDesc"] = desc;
        report["modelParams"] = (double) llama_model_n_params(model);
    }
    emit("Main model loaded");

    // The vision projector (mmproj) is NOT loaded here: build 10 runs the
    // text-only cases first, and they must not depend on it. It loads
    // lazily at the first image case (see ensure_vision below).

    emit("Creating context");
    emit_progress({{"stage", "context_create"}, {"event", "start"}, {"label", "Creating context"}});
    const auto ctx_start = Clock::now();
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = N_CTX;
    cparams.n_batch = N_BATCH;
    cparams.n_ubatch = 512;
    cparams.n_threads = N_THREADS;
    cparams.n_threads_batch = N_THREADS;
    // Build 15: only the DSpark arm asks for recurrent-state
    // snapshots (n_rs_seq = draft n_max), so a verify round's
    // rollback never exceeds what the context can restore. The
    // other arms keep the default 0, byte-for-byte as before.
    if (dspark) cparams.n_rs_seq = DSPARK_N_MAX;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { llama_model_free(model); return fail("context", "llama_init_from_model returned null"); }
    const double context_ms = ms_between(ctx_start, Clock::now());
    report["contextMs"] = context_ms;
    emit_progress({{"stage", "context_create"}, {"event", "done"}, {"label", "Context created"},
                   {"contextMs", context_ms}, {"stageElapsedMs", context_ms}});

    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_batch batch = llama_batch_init(1, 0, 1);

    // Build 15: DSpark arm setup — draft model, draft context,
    // rollback probe, speculative init. Everything here happens
    // after the target context exists and before any case runs, in
    // the fork server's own order (tools/server/server-context.cpp).
    // Any failure is a fail-safe stop: the report names the phase
    // and the reason, and no case (hence no decode number) is ever
    // produced. The other arms skip this block entirely.
    const std::string dspark_path = model_dir + "/" + DSPARK_GGUF;
    std::string dspark_spec_log_path;
    llama_model * dspark_model = nullptr;
    llama_context * dspark_ctx = nullptr;
    common_speculative * spec = nullptr;
    common_sampler * dspark_sampler = nullptr;
    auto free_dspark = [&]() {
        if (dspark_sampler) { common_sampler_free(dspark_sampler); dspark_sampler = nullptr; }
        if (spec) { common_speculative_free(spec); spec = nullptr; }
        if (dspark_ctx) { llama_free(dspark_ctx); dspark_ctx = nullptr; }
        if (dspark_model) { llama_model_free(dspark_model); dspark_model = nullptr; }
    };
    if (dspark) {
        emit("Loading DSpark draft model");
        emit_progress({{"stage", "dspark_setup"}, {"event", "start"}, {"label", "Loading DSpark draft model"}});
        // The runtime's speculative stats are printed through the
        // common log (common_speculative_print_stats), which writes
        // to stderr by default and would be lost on Android.
        // Redirect the main common log to a file in the model
        // directory so the stats line can be read back at teardown
        // and parsed into the report; colors/prefix/timestamps off
        // so the line is plain text.
        dspark_spec_log_path = model_dir + "/dspark_common.log";
        common_log_set_colors(common_log_main(), LOG_COLORS_DISABLED);
        common_log_set_prefix(common_log_main(), false);
        common_log_set_timestamps(common_log_main(), false);
        common_log_set_file(common_log_main(), dspark_spec_log_path.c_str());
        // Draft model load, the fork server's way: a fresh default
        // params struct, i.e. no MoE fields at all — the server
        // explicitly clears the MoE params for the draft because
        // draft models are dense (the DSpark GGUF is a 5-layer
        // dflash model), so there is no experts path and no slot
        // bank interaction.
        const auto dspark_load_start = Clock::now();
        llama_model_params dparams = llama_model_default_params();
        dparams.n_gpu_layers = 0;
        dparams.use_mmap = true;
        dspark_model = llama_model_load_from_file(dspark_path.c_str(), dparams);
        if (!dspark_model) {
            free_dspark();
            return fail("dspark_draft_load", "DSpark disabled itself: llama_model_load_from_file returned null for " + dspark_path);
        }
        report["dsparkDraftLoadMs"] = ms_between(dspark_load_start, Clock::now());
        // Draft context with ctx_other = target context: the
        // dflash/DSpark implementation reads the target's extracted
        // layer embeddings through it (common/speculative.cpp).
        llama_context_params dcparams = llama_context_default_params();
        dcparams.n_ctx = N_CTX;
        dcparams.n_batch = N_BATCH;
        dcparams.n_ubatch = 512;
        dcparams.n_threads = N_THREADS;
        dcparams.n_threads_batch = N_THREADS;
        dcparams.ctx_other = ctx;
        dspark_ctx = llama_init_from_model(dspark_model, dcparams);
        if (!dspark_ctx) {
            free_dspark();
            return fail("dspark_draft_context", "DSpark disabled itself: llama_init_from_model returned null for the DSpark draft model");
        }
        // Rollback probe, once, at empty KV, exactly the fork
        // server's gate: speculative decoding is only attempted
        // when the target context can remove partial sequences.
        // RS (bounded by n_rs_seq) is the expected answer for
        // qwen35moe; PART is equally usable by this driver. FULL
        // would need the server's checkpoint save/restore path,
        // which this build deliberately does not port, and NO
        // means no rollback at all — both fail safe here.
        const common_context_seq_rm_type rm_type = common_context_can_seq_rm(ctx);
        const std::string rm_name = rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS   ? "rs"
                                  : rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_PART ? "part"
                                  : rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ? "full" : "no";
        report["dsparkSeqRmType"] = rm_name;
        if (rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO || rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            free_dspark();
            return fail("dspark_probe", "DSpark disabled itself: target context seq_rm probe returned \"" + rm_name +
                        "\"; speculative rollback needs \"rs\" (bounded by n_rs_seq) or \"part\" on this build, and the checkpoint path for \"full\" is not ported");
        }
        common_params_speculative sparams;
        sparams.types = { COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK };
        sparams.draft.n_max = DSPARK_N_MAX;
        sparams.draft.n_min = 0;
        sparams.draft.p_min = 0.0f; // confidence-head truncation off: always draft the full n_max
        sparams.draft.mparams.path = dspark_path;
        sparams.draft.ctx_tgt = ctx;
        sparams.draft.ctx_dft = dspark_ctx;
        try {
            spec = common_speculative_init(sparams, 1);
        } catch (...) {
            spec = nullptr;
        }
        if (!spec) {
            free_dspark();
            return fail("dspark_init", "DSpark disabled itself: common_speculative_init failed for spec type draft-dspark (see loaderLog)");
        }
        // Greedy-exact verify sampler: a top_k=1 chain, the same
        // construction the fork's own draft implementations use.
        common_params_sampling verify_sampling;
        verify_sampling.top_k = 1;
        verify_sampling.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
        dspark_sampler = common_sampler_init(model, verify_sampling);
        if (!dspark_sampler) {
            free_dspark();
            return fail("dspark_init", "DSpark disabled itself: common_sampler_init failed for the verify sampler");
        }
        report["dsparkEnabled"] = true;
        emit_progress({{"stage", "dspark_setup"}, {"event", "done"}, {"label", "DSpark ready"},
                       {"dsparkSeqRmType", rm_name}, {"dsparkDraftNMax", DSPARK_N_MAX}});
        emit("DSpark ready: draft-dspark, n_max " + std::to_string(DSPARK_N_MAX) +
             ", seq_rm probe \"" + rm_name + "\"");
    }

    // Build 10 case list. Text-only cases first (indices 1-3), then the
    // original image cases (indices 4-6, prompts and image unchanged).
    // A text-only prompt needs no image: plain general knowledge questions,
    // the third written to produce a long answer so decode speed is
    // measured over a decent token count.
    struct TrialCase { const char * question; bool has_image; };
    const TrialCase trial_cases[] = {
        { "What is the capital of France?", false },
        { "Explain in a few sentences why the sky appears blue.", false },
        { "Write a detailed explanation of how a bicycle stays upright while it is moving, covering the main physical ideas in several paragraphs.", false },
        { "Describe this image.", true },
        { "How many people are in this image?", true },
        { "How many cars are in this image?", true },
    };
    const int n_cases = (int) (sizeof(trial_cases) / sizeof(trial_cases[0]));
    json cases = json::array();
    std::string fatal;
    std::string fatal_phase;

    // Extra batch for text-only prefill (the generation batch above holds one
    // token). Capacity N_BATCH; llama_decode splits into n_ubatch pieces
    // internally, exactly as the mtmd helper's eval calls do.
    llama_batch prefill_batch = llama_batch_init(N_BATCH, 0, 1);

    // Lazy vision load (build 10): mtmd enters only here, at the first
    // image case. Measurements and checkpoints match build 9's upfront
    // load (stage mmproj_load; report loadMmprojMs / visionSupported).
    mtmd_context * vision = nullptr;
    auto ensure_vision = [&](int display) -> bool {
        if (vision) return true;
        emit("Loading vision projector (mmproj)");
        emit_progress({{"stage", "mmproj_load"}, {"event", "start"}, {"caseIndex", display},
                       {"label", "Loading vision projector"}});
        const auto mmproj_start = Clock::now();
        mtmd_context_params vparams = mtmd_context_params_default();
        vparams.use_gpu = false;
        vparams.print_timings = false;
        vparams.n_threads = N_THREADS;
        vparams.image_max_slice_nums = 9; // vendor demo default for MiniCPM-V
        vision = mtmd_init_from_file(mmproj_path.c_str(), model, vparams);
        if (!vision) {
            fatal_phase = "mmproj_load";
            fatal = "mtmd_init_from_file returned null for " + mmproj_path;
            return false;
        }
        const double load_mmproj_ms = ms_between(mmproj_start, Clock::now());
        report["loadMmprojMs"] = load_mmproj_ms;
        emit_progress({{"stage", "mmproj_load"}, {"event", "done"}, {"caseIndex", display},
                       {"label", "Vision projector loaded"},
                       {"mmprojLoadMs", load_mmproj_ms}, {"stageElapsedMs", load_mmproj_ms}});
        report["visionSupported"] = mtmd_support_vision(vision);
        return true;
    };

    // Greedy generation, shared verbatim by the text-only and image
    // cases (build 9 behavior, including the decode checkpoints: start,
    // first_token, ~1 Hz live token counter, done). On a decode failure
    // it sets fatal/fatal_phase and returns with the partial counts
    // recorded, exactly as the inlined build 9 loop did.
    auto run_generation = [&](int display, const Clock::time_point & case_start,
                              json & kase, llama_pos & n_past) {
        const std::string tag = "Case " + std::to_string(display);
        const auto gen_start = Clock::now();
        std::string answer;
        // Build 15: the counted token ids, recorded in every arm
        // (additive field only) so a DSpark run's output can be
        // checked token-for-token against the baseline arms.
        std::vector<llama_token> gen_ids;
        int generated = 0;
        bool ended_eos = false;
        double decode_calls_ms = 0.0;
        emit_progress({{"stage", "decode"}, {"event", "start"}, {"caseIndex", display},
                       {"generatedTokens", 0},
                       {"label", tag + ": generating answer"}});
        auto last_progress_emit = Clock::now();
        while (generated < max_tokens) {
            const llama_token token = greedy_token(ctx, vocab);
            if (llama_vocab_is_eog(vocab, token)) { ended_eos = true; break; }
            if (generated == 0) kase["ttftMs"] = ms_between(case_start, Clock::now());
            answer += token_piece(vocab, token);
            gen_ids.push_back(token);
            generated++;
            if (generated == 1) {
                emit_progress({{"stage", "decode"}, {"event", "first_token"}, {"caseIndex", display},
                               {"generatedTokens", generated},
                               {"ttftMs", kase["ttftMs"].get<double>()},
                               {"label", tag + ": generating answer"}});
                last_progress_emit = Clock::now();
            } else if (ms_between(last_progress_emit, Clock::now()) >= 1000.0) {
                // Live token counter, ~1 Hz: status.json always shows the
                // latest count and (via the service stamp) when the last
                // token arrived, so a crawl reads as a crawl, not a hang.
                const double decode_elapsed_ms = ms_between(gen_start, Clock::now());
                emit_progress({{"stage", "decode"}, {"event", "progress"}, {"caseIndex", display},
                               {"generatedTokens", generated},
                               {"decodeElapsedMs", decode_elapsed_ms},
                               {"decodeTokensPerSec", decode_elapsed_ms > 0 ? generated / (decode_elapsed_ms / 1000.0) : 0.0},
                               {"label", tag + ": generating answer"}});
                last_progress_emit = Clock::now();
            }
            if (generated % 32 == 0) emit(tag + ": " + std::to_string(generated) + " tokens");
            if (generated >= max_tokens) break;
            batch.n_tokens = 1;
            batch.token[0] = token;
            batch.pos[0] = n_past;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = 1;
            const auto dec_start = Clock::now();
            const int dres = llama_decode(ctx, batch);
            decode_calls_ms += ms_between(dec_start, Clock::now());
            if (dres != 0) { fatal_phase = "decode"; fatal = "llama_decode failed with code " + std::to_string(dres); break; }
            n_past++;
        }
        const double gen_wall_ms = ms_between(gen_start, Clock::now());
        kase["decodeWallMs"] = gen_wall_ms;
        kase["decodeCallsMs"] = decode_calls_ms;
        kase["generatedTokens"] = generated;
        kase["decodeTokensPerSec"] = gen_wall_ms > 0 ? generated / (gen_wall_ms / 1000.0) : 0.0;
        kase["endedByEos"] = ended_eos;
        emit_progress({{"stage", "decode"}, {"event", "done"}, {"caseIndex", display},
                       {"generatedTokens", generated}, {"decodeWallMs", gen_wall_ms},
                       {"decodeTokensPerSec", kase["decodeTokensPerSec"].get<double>()},
                       {"label", tag + " complete"}});
        kase["peakRssKb"] = peak_rss_kb();
        kase["answer"] = answer;
        json id_array = json::array();
        for (const llama_token t : gen_ids) id_array.push_back(t);
        kase["generatedTokenIds"] = id_array;
    };

    // Build 15: speculative generation for the DSpark arm, following
    // the fork's own speculative-simple example flow (the server loop
    // with a single slot reduces to the same steps):
    //   - the first token is sampled greedily from the prefill
    //     logits, exactly the baseline loop's first token;
    //   - each round drafts n_max tokens with the DSpark heads,
    //     rolls the draft context's drafting scratch block back
    //     (seq_rm from the anchor's position — the step both of the
    //     fork's reference drivers take right after drafting;
    //     build 15 missed it and process() collided with the scratch
    //     block's positions, see the build 16 note in the loop),
    //     re-decodes [anchor, draft...] on the target as one verify
    //     batch, feeds that batch to common_speculative_process (it
    //     refreshes the draft context's injected features from the
    //     target's extracted layer embeddings), and accepts greedily
    //     with common_sampler_sample_and_accept_n — result ids are
    //     the target's own tokens, so acceptance is exact;
    //   - both contexts are then rolled back with seq_rm from the
    //     new anchor's position: rejected draft KV (and the anchor's
    //     own verify-row KV, which the next round's verify batch
    //     re-decodes) is dropped. Worst-case rollback per round is
    //     n_max tokens, the n_rs_seq the context was created with.
    // Case fields mirror run_generation's, plus the spec counters.
    // On any runtime failure it sets fatal/fatal_phase and returns
    // with the partial counts recorded, like the greedy loop does.
    // Build 18: the round loop runs under a try/catch (see the
    // header note); a caught exception takes the same fatal path
    // with errorPhase "dspark_exception".
    auto run_generation_dspark = [&](int display, const Clock::time_point & case_start,
                                     json & kase, const std::vector<llama_token> & prompt_tokens) {
        const std::string tag = "Case " + std::to_string(display);
        const auto gen_start = Clock::now();
        std::string answer;
        std::vector<llama_token> gen_ids;
        int generated = 0;
        bool ended_eos = false;
        double decode_calls_ms = 0.0;
        long long rounds = 0, drafted = 0, accepted_tokens = 0, draft_seq_rm_failures = 0;
        std::string caught_exception; // build 18: what() of a caught round exception, "" if none
        emit_progress({{"stage", "decode"}, {"event", "start"}, {"caseIndex", display},
                       {"generatedTokens", 0},
                       {"label", tag + ": generating answer (DSpark)"}});
        auto last_progress_emit = Clock::now();
        // Emit one accepted token with the baseline loop's semantics:
        // an EOG token ends generation and is never counted; the cap
        // is checked by the callers. Returns false when generation
        // must stop after this token.
        auto emit_token = [&](llama_token token) -> bool {
            if (llama_vocab_is_eog(vocab, token)) { ended_eos = true; return false; }
            if (generated == 0) kase["ttftMs"] = ms_between(case_start, Clock::now());
            answer += token_piece(vocab, token);
            gen_ids.push_back(token);
            generated++;
            if (generated == 1) {
                emit_progress({{"stage", "decode"}, {"event", "first_token"}, {"caseIndex", display},
                               {"generatedTokens", generated},
                               {"ttftMs", kase["ttftMs"].get<double>()},
                               {"label", tag + ": generating answer (DSpark)"}});
                last_progress_emit = Clock::now();
            } else if (ms_between(last_progress_emit, Clock::now()) >= 1000.0) {
                const double decode_elapsed_ms = ms_between(gen_start, Clock::now());
                emit_progress({{"stage", "decode"}, {"event", "progress"}, {"caseIndex", display},
                               {"generatedTokens", generated},
                               {"decodeElapsedMs", decode_elapsed_ms},
                               {"decodeTokensPerSec", decode_elapsed_ms > 0 ? generated / (decode_elapsed_ms / 1000.0) : 0.0},
                               {"label", tag + ": generating answer (DSpark)"}});
                last_progress_emit = Clock::now();
            }
            return generated < max_tokens;
        };

        // prompt_tgt tracks the committed tokens EXCLUDING the
        // anchor (the DSpark draft params' prompt pointer; the
        // dflash implementation itself only needs n_past/id_last,
        // but the API contract is kept whole).
        llama_tokens prompt_tgt(prompt_tokens.begin(), prompt_tokens.end());
        common_speculative_begin(spec, 0, prompt_tgt);
        llama_token anchor = greedy_token(ctx, vocab);
        llama_pos anchor_pos = (llama_pos) prompt_tokens.size();
        bool keep_going = emit_token(anchor);

        llama_batch verify_batch = llama_batch_init(DSPARK_N_MAX + 1, 0, 1);
        llama_tokens draft;
        // Build 18 safety catch: every native call in a round —
        // the speculative draft, the draft-context seq_rm (the
        // build 16 fix), the target verify decode,
        // common_speculative_process and the sampler accept —
        // runs under this catch. Build 17's widened arm died
        // silently mid-round with no report (section 41
        // diagnosis): an uncaught slot-runtime throw has no
        // handler anywhere below this driver, so it terminated
        // the process. Here it becomes a named fatal with the
        // exception text preserved, and the normal report path
        // runs. A GGML_ASSERT abort is not catchable and would
        // still kill the process — that outcome is itself the
        // discriminator the diagnosis named.
        try {
            while (keep_going && !ended_eos && fatal.empty()) {
                const auto round_start = Clock::now();
                draft.clear();
                common_speculative_draft_params & dp = common_speculative_get_draft_params(spec, 0);
                dp.drafting = true;
                dp.n_max = DSPARK_N_MAX;
                dp.n_past = anchor_pos;
                dp.id_last = anchor;
                dp.prompt = &prompt_tgt;
                dp.result = &draft;
                common_speculative_draft(spec);
                // Build 16 fix: drafting leaves scratch KV in the DRAFT
                // context. The dflash draft() decodes its noise block
                // ([anchor, mask, mask] for DSpark: n_max tokens) into
                // ctx_dft at positions anchor_pos .. anchor_pos + n_max - 1, and
                // process() below injects the real target features at
                // those same positions via llama_decode(ctx_dft, ...).
                // Both of the fork's reference drivers remove the scratch
                // block the moment drafting returns, before the verify
                // batch: the server (server-context.cpp, right after
                // common_speculative_draft) and speculative-simple both
                // call seq_rm(ctx_dft, ckpt.pos_max + 1, -1), where
                // pos_max is the target's last committed position, so
                // pos_max + 1 is exactly the anchor's position. Build 15
                // skipped this rollback: the noise block's last position
                // (36 + 3 - 1 = 38) was still stored in the draft memory
                // when process() tried to inject starting at 36, llama's
                // M-RoPE batch check (stored last position must be <= the
                // batch's starting position) rejected the injection
                // decode with ret -1, and process() returned false
                // (device: errorPhase dspark_process, X = 38, Y = 36).
                if (!llama_memory_seq_rm(llama_get_memory(dspark_ctx), 0, anchor_pos, -1)) {
                    draft_seq_rm_failures++;
                }
                if (draft.empty()) {
                    fatal_phase = "dspark_draft";
                    fatal = "common_speculative_draft produced no draft tokens";
                    break;
                }
                rounds++;
                drafted += (long long) draft.size();
                common_batch_clear(verify_batch);
                common_batch_add(verify_batch, anchor, anchor_pos, { 0 }, true);
                for (size_t i = 0; i < draft.size(); i++) {
                    common_batch_add(verify_batch, draft[i], anchor_pos + 1 + (llama_pos) i, { 0 }, true);
                }
                if (llama_decode(ctx, verify_batch) != 0) {
                    fatal_phase = "decode";
                    fatal = "llama_decode failed during DSpark verify";
                    break;
                }
                if (!common_speculative_process(spec, verify_batch)) {
                    fatal_phase = "dspark_process";
                    fatal = "common_speculative_process failed on the DSpark verify batch";
                    break;
                }
                const std::vector<llama_token> ids =
                    common_sampler_sample_and_accept_n(dspark_sampler, ctx, draft);
                if (ids.empty()) {
                    fatal_phase = "dspark_accept";
                    fatal = "common_sampler_sample_and_accept_n returned no tokens";
                    break;
                }
                const long long n_accepted = (long long) ids.size() - 1;
                accepted_tokens += n_accepted;
                common_speculative_accept(spec, 0, (uint16_t) n_accepted);
                // Commit: the new anchor is the last accepted id, at
                // position anchor_pos + ids.size(); everything before it
                // joins prompt_tgt. Roll both contexts back to the
                // anchor's position (the anchor itself is re-decoded as
                // the next verify batch's first row).
                prompt_tgt.push_back(anchor);
                for (size_t i = 0; i + 1 < ids.size(); i++) prompt_tgt.push_back(ids[i]);
                anchor = ids.back();
                anchor_pos += (llama_pos) ids.size();
                if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, anchor_pos, -1)) {
                    fatal_phase = "dspark_rollback";
                    fatal = "llama_memory_seq_rm failed on the target context after DSpark acceptance";
                    break;
                }
                if (!llama_memory_seq_rm(llama_get_memory(dspark_ctx), 0, anchor_pos, -1)) {
                    draft_seq_rm_failures++;
                }
                decode_calls_ms += ms_between(round_start, Clock::now());
                for (const llama_token token : ids) {
                    if (generated >= max_tokens) { keep_going = false; break; }
                    if (!emit_token(token)) { keep_going = false; break; }
                }
            }
        } catch (const std::exception & e) {
            fatal_phase = "dspark_exception";
            caught_exception = e.what();
            fatal = "DSpark decode round threw an exception: " + caught_exception;
        } catch (...) {
            fatal_phase = "dspark_exception";
            caught_exception = "unknown non-std exception";
            fatal = "DSpark decode round threw an unknown non-std exception";
        }
        llama_batch_free(verify_batch);
        const double gen_wall_ms = ms_between(gen_start, Clock::now());
        kase["decodeWallMs"] = gen_wall_ms;
        kase["decodeCallsMs"] = decode_calls_ms;
        kase["generatedTokens"] = generated;
        kase["decodeTokensPerSec"] = gen_wall_ms > 0 ? generated / (gen_wall_ms / 1000.0) : 0.0;
        kase["endedByEos"] = ended_eos;
        kase["dsparkRounds"] = rounds;
        kase["dsparkDraftedTokens"] = drafted;
        kase["dsparkAcceptedTokens"] = accepted_tokens;
        kase["dsparkAcceptRatePct"] = drafted > 0 ? 100.0 * (double) accepted_tokens / (double) drafted : 0.0;
        kase["dsparkAvgAcceptanceLength"] = rounds > 0 ? (double) accepted_tokens / (double) rounds : 0.0;
        kase["dsparkDraftSeqRmFailures"] = draft_seq_rm_failures;
        // Build 18: the caught round exception's text ("" when
        // the catch never fired), so it survives into the report
        // and the pasted Copy diagnostics bundle.
        kase["dsparkCaughtException"] = caught_exception;
        emit_progress({{"stage", "decode"}, {"event", "done"}, {"caseIndex", display},
                       {"generatedTokens", generated}, {"decodeWallMs", gen_wall_ms},
                       {"decodeTokensPerSec", kase["decodeTokensPerSec"].get<double>()},
                       {"label", tag + " complete"}});
        emit_progress({{"stage", "dspark_stats"}, {"event", "done"}, {"caseIndex", display},
                       {"dsparkRounds", rounds}, {"dsparkDraftedTokens", drafted},
                       {"dsparkAcceptedTokens", accepted_tokens},
                       {"dsparkAcceptRatePct", kase["dsparkAcceptRatePct"].get<double>()},
                       {"dsparkAvgAcceptanceLength", kase["dsparkAvgAcceptanceLength"].get<double>()},
                       {"label", tag + ": DSpark speculative stats"}});
        kase["peakRssKb"] = peak_rss_kb();
        kase["answer"] = answer;
        json id_array = json::array();
        for (const llama_token t : gen_ids) id_array.push_back(t);
        kase["generatedTokenIds"] = id_array;
    };

    for (int qi = 0; qi < n_cases && fatal.empty(); qi++) {
        // Build 12 quick mode: only the decode measurement case runs.
        if (quick && qi + 1 != QUICK_CASE_DISPLAY) continue;
        const int display = qi + 1;
        const std::string tag = "Case " + std::to_string(display);
        const std::string question = trial_cases[qi].question;
        const bool has_image = trial_cases[qi].has_image;
        emit(tag + "/" + std::to_string(n_cases) + ": " + question);
        json kase = {{"index", display}, {"type", has_image ? "image" : "text"}, {"prompt", question}};
        const auto case_start = Clock::now();
        llama_memory_clear(llama_get_memory(ctx), true);
        // Build 11: refill the decode windows from the prewarm table
        // before this case's prefill, the vendor app's per-round path
        // (the runtime clears window mappings at every prefill-to-decode
        // transition, and its own comments require a fresh prewarm call
        // each round). The flash read happens here, off the decode
        // clock; at the transition the runtime only republishes the
        // slot mappings. No-op if the table was not written.
        llama_moe_prewarm_experts(ctx);
        llama_pos n_past = 0;

        if (!has_image) {
            // ---- Text-only case (build 10): no bitmap, no mmproj, no
            // mtmd. Same chat wrapper as the image cases minus the image
            // marker; tokenize with llama_tokenize (add_special and
            // parse_special mirror the mtmd_input_text flags) and prefill
            // the tokens through llama_decode in N_BATCH pieces.
            emit_progress({{"stage", "tokenize"}, {"event", "start"}, {"caseIndex", display},
                           {"generatedTokens", 0},
                           {"label", tag + " (text): tokenizing prompt"}});
            const std::string prompt = "<|im_start|>user\n" + question +
                "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
            const auto tok_start = Clock::now();
            int32_t n_needed = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(),
                                              nullptr, 0, true, true);
            std::vector<llama_token> tokens;
            int32_t n_tok = -1;
            if (n_needed != INT32_MIN && n_needed != 0) {
                tokens.assign((size_t) (n_needed < 0 ? -n_needed : n_needed), 0);
                n_tok = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(),
                                       tokens.data(), (int32_t) tokens.size(), true, true);
            }
            const double tokenize_ms = ms_between(tok_start, Clock::now());
            kase["tokenizeMs"] = tokenize_ms;
            emit_progress({{"stage", "tokenize"}, {"event", "done"}, {"caseIndex", display},
                           {"tokenizeMs", tokenize_ms}, {"stageElapsedMs", tokenize_ms},
                           {"label", tag + " (text): evaluating prompt"}});
            if (n_tok < 0) {
                fatal_phase = "tokenize";
                fatal = "llama_tokenize failed for text case " + std::to_string(display);
                break;
            }
            tokens.resize((size_t) n_tok);

            emit_progress({{"stage", "text_prefill"}, {"event", "start"}, {"caseIndex", display},
                           {"label", tag + " (text): evaluating prompt"}});
            const auto prefill_start = Clock::now();
            bool prefill_ok = true;
            bool dspark_prefill_process_failed = false;
            for (size_t off = 0; off < tokens.size(); ) {
                const int32_t chunk_n = (int32_t) std::min<size_t>((size_t) N_BATCH, tokens.size() - off);
                prefill_batch.n_tokens = chunk_n;
                for (int32_t i = 0; i < chunk_n; i++) {
                    prefill_batch.token[i]  = tokens[off + (size_t) i];
                    prefill_batch.pos[i]    = (llama_pos) (off + (size_t) i);
                    prefill_batch.n_seq_id[i] = 1;
                    prefill_batch.seq_id[i][0] = 0;
                    // logits only needed from the final prompt token
                    prefill_batch.logits[i] = (off + (size_t) i + 1 == tokens.size()) ? 1 : 0;
                }
                if (llama_decode(ctx, prefill_batch) != 0) { prefill_ok = false; break; }
                // Build 15: feed the DSpark draft context from this
                // chunk's target embeddings before any later decode
                // overwrites them — process() gathers the extracted
                // layer inputs of the decode that just finished.
                if (dspark && !common_speculative_process(spec, prefill_batch)) {
                    prefill_ok = false;
                    dspark_prefill_process_failed = true;
                    break;
                }
                off += (size_t) chunk_n;
            }
            const double prefill_ms = ms_between(prefill_start, Clock::now());
            if (!prefill_ok) {
                if (dspark_prefill_process_failed) {
                    fatal_phase = "dspark_process";
                    fatal = "common_speculative_process failed on the text prefill batch in case " + std::to_string(display);
                } else {
                    fatal_phase = "prefill";
                    fatal = "llama_decode failed during text prefill in case " + std::to_string(display);
                }
                break;
            }
            n_past = (llama_pos) tokens.size();
            const double prefill_tps = prefill_ms > 0 ? (double) tokens.size() / (prefill_ms / 1000.0) : 0.0;
            kase["promptTokens"] = tokens.size();
            kase["imageTokens"] = 0;
            kase["visionEvalMs"] = 0.0;
            kase["prefillMs"] = prefill_ms;
            kase["prefillTokensPerSec"] = prefill_tps;
            emit_progress({{"stage", "text_prefill"}, {"event", "done"}, {"caseIndex", display},
                           {"prefillMs", prefill_ms}, {"prefillTokensPerSec", prefill_tps},
                           {"promptTokens", tokens.size()}, {"stageElapsedMs", prefill_ms},
                           {"label", tag + " (text): generating answer"}});

            if (dspark) run_generation_dspark(display, case_start, kase, tokens);
            else run_generation(display, case_start, kase, n_past);
            cases.push_back(kase);
            emit(tag + " complete: " + std::to_string(kase["generatedTokens"].get<int>()) + " tokens");
            // Build 12: hand the finished case to the service at once,
            // so it lands in the incremental partial report even if the
            // run is later killed by the watchdog; then refresh the
            // [moe-ssd] counter totals in status.json.
            emit("CASE_DONE " + kase.dump());
            emit_counters();
            continue;
        }

        // ---- Image case: build 9 flow, plus per-chunk progress.
        if (!ensure_vision(display)) break;
        emit_progress({{"stage", "tokenize"}, {"event", "start"}, {"caseIndex", display},
                       {"generatedTokens", 0},
                       {"label", tag + ": reading image"}});

        mtmd_bitmap * bitmap = mtmd_helper_bitmap_init_from_file(vision, image.c_str());
        if (!bitmap) { fatal_phase = "image_load"; fatal = "mtmd_helper_bitmap_init_from_file failed for " + image; break; }

        std::string prompt = "<|im_start|>user\n";
        prompt += mtmd_default_marker();
        prompt += "\n" + question + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        mtmd_input_text text;
        text.text = prompt.c_str();
        text.add_special = true;
        text.parse_special = true;
        mtmd_input_chunks * chunks = mtmd_input_chunks_init();
        const mtmd_bitmap * bitmaps[] = { bitmap };
        const auto tok_start = Clock::now();
        int32_t tres = mtmd_tokenize(vision, chunks, &text, bitmaps, 1);
        const double tokenize_ms = ms_between(tok_start, Clock::now());
        kase["tokenizeMs"] = tokenize_ms;
        emit_progress({{"stage", "tokenize"}, {"event", "done"}, {"caseIndex", display},
                       {"tokenizeMs", tokenize_ms}, {"stageElapsedMs", tokenize_ms},
                       {"label", tag + ": evaluating prompt"}});
        mtmd_bitmap_free(bitmap);
        if (tres != 0) {
            mtmd_input_chunks_free(chunks);
            fatal_phase = "tokenize"; fatal = "mtmd_tokenize failed with code " + std::to_string(tres); break;
        }

        double vision_ms = 0.0, prefill_ms = 0.0;
        size_t image_tokens = 0, prompt_tokens = 0;
        bool eval_ok = true;
        const size_t n_chunks = mtmd_input_chunks_size(chunks);
        // Pre-scan so the vision_encode stage can close the moment the
        // last image chunk finishes (image chunks sit inside the prefill
        // loop; visionMs below is the image-chunk share of prefillMs,
        // matching the report's existing visionEvalMs/prefillMs split).
        size_t last_image_chunk = n_chunks; // n_chunks == no image chunk
        for (size_t ci = 0; ci < n_chunks; ci++) {
            if (mtmd_input_chunk_get_type(mtmd_input_chunks_get(chunks, ci)) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                last_image_chunk = ci;
            }
        }
        bool vision_started = false;
        size_t image_chunks_done = 0;
        emit_progress({{"stage", "prefill"}, {"event", "start"}, {"caseIndex", display},
                       {"label", tag + ": evaluating prompt"}});
        for (size_t ci = 0; ci < n_chunks; ci++) {
            const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, ci);
            const bool is_image = mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE;
            if (is_image && !vision_started) {
                vision_started = true;
                emit_progress({{"stage", "vision_encode"}, {"event", "start"}, {"caseIndex", display},
                               {"label", tag + ": vision encode"}});
            }
            const size_t n_tok = mtmd_input_chunk_get_n_tokens(chunk);
            prompt_tokens += n_tok;
            if (is_image) image_tokens += n_tok;
            const auto chunk_start = Clock::now();
            llama_pos new_n_past = n_past;
            // logits only needed from the final chunk
            const bool logits_last = (ci + 1 == n_chunks);
            if (mtmd_helper_eval_chunk_single(vision, ctx, chunk, n_past, 0, N_BATCH, logits_last, &new_n_past) != 0) {
                eval_ok = false;
                break;
            }
            const double chunk_ms = ms_between(chunk_start, Clock::now());
            prefill_ms += chunk_ms;
            if (is_image) { vision_ms += chunk_ms; image_chunks_done++; }
            n_past = new_n_past;
            // Build 10: per-chunk sign of life. One checkpoint after
            // every evaluated chunk, image or text, carrying the chunk's
            // position in the sequence. In status.json this advances
            // chunkIndex (and lastProgressAtMs) chunk by chunk: a crawl
            // shows steady movement, a deadlock freezes on one index.
            emit_progress({{"stage", "chunk_progress"}, {"event", "progress"}, {"caseIndex", display},
                           {"chunkIndex", ci + 1}, {"chunkTotal", n_chunks},
                           {"chunkIsImage", is_image}, {"chunkMs", chunk_ms},
                           {"imageChunksDone", image_chunks_done},
                           {"label", tag + ": chunk " + std::to_string(ci + 1) + "/" +
                                     std::to_string(n_chunks) + (is_image ? " (image)" : " (text)")}});
            if (is_image && ci == last_image_chunk) {
                emit_progress({{"stage", "vision_encode"}, {"event", "done"}, {"caseIndex", display},
                               {"visionMs", vision_ms}, {"stageElapsedMs", vision_ms},
                               {"label", tag + ": vision encode done"}});
            }
        }
        mtmd_input_chunks_free(chunks);
        kase["promptTokens"] = prompt_tokens;
        kase["imageTokens"] = image_tokens;
        kase["visionEvalMs"] = vision_ms;
        kase["prefillMs"] = prefill_ms;
        if (!eval_ok) { fatal_phase = "prefill"; fatal = "mtmd_helper_eval_chunk_single failed in case " + std::to_string(display); break; }
        emit_progress({{"stage", "prefill"}, {"event", "done"}, {"caseIndex", display},
                       {"prefillMs", prefill_ms}, {"visionMs", vision_ms}, {"stageElapsedMs", prefill_ms},
                       {"promptTokens", prompt_tokens}, {"imageTokens", image_tokens},
                       {"label", tag + ": generating answer"}});

        run_generation(display, case_start, kase, n_past);
        cases.push_back(kase);
        emit(tag + " complete: " + std::to_string(kase["generatedTokens"].get<int>()) + " tokens");
        emit("CASE_DONE " + kase.dump());
        emit_counters();
    }

    report["cases"] = cases;
    // Build 15: DSpark teardown. Print the runtime's own speculative
    // stats (common_speculative_print_stats) into the redirected
    // common log, read the "statistics draft-dspark" line back and
    // parse its counters into the report as the runtime's own
    // cross-check of the driver's counters in the case. The common
    // log is asynchronous (a worker thread drains a ring buffer),
    // so poll the file briefly for the line; if it never lands, the
    // fields are simply absent and the driver's counters stand.
    if (dspark && spec) {
        common_speculative_print_stats(spec);
        std::string stats_line;
        for (int attempt = 0; attempt < 40 && stats_line.empty(); attempt++) {
            if (FILE * lf = std::fopen(dspark_spec_log_path.c_str(), "r")) {
                std::string content;
                char buf[8192];
                size_t n;
                while ((n = std::fread(buf, 1, sizeof(buf), lf)) > 0) content.append(buf, n);
                std::fclose(lf);
                size_t pos = 0;
                while (pos < content.size()) {
                    const size_t eol = content.find('\n', pos);
                    const std::string line = content.substr(pos, eol == std::string::npos ? std::string::npos : eol - pos);
                    pos = eol == std::string::npos ? content.size() : eol + 1;
                    if (line.find("statistics") != std::string::npos &&
                        line.find("draft-dspark") != std::string::npos) {
                        stats_line = line;
                        break;
                    }
                }
            }
            if (stats_line.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
        if (!stats_line.empty()) {
            report["dsparkSpecStatsLine"] = stats_line;
            long long v;
            if (log_field_ll(stats_line, "#gen drafts ", v)) report["dsparkSpecGenDrafts"] = v;
            if (log_field_ll(stats_line, "#acc drafts ", v)) report["dsparkSpecAccDrafts"] = v;
            if (log_field_ll(stats_line, "#gen tokens ", v)) report["dsparkSpecGenTokens"] = v;
            if (log_field_ll(stats_line, "#acc tokens ", v)) report["dsparkSpecAccTokens"] = v;
        }
    }
    free_dspark();
    llama_batch_free(batch);
    llama_batch_free(prefill_batch);
    llama_free(ctx);
    if (vision) mtmd_free(vision);
    llama_model_free(model);
    if (!fatal.empty()) return fail(fatal_phase, fatal);
    report["peakRssKb"] = peak_rss_kb();
    report["totalMs"] = ms_between(trial_start, Clock::now());
    // Final [moe-ssd] totals. Parsed after llama_free above on purpose:
    // the slot runtime prints its cumulative lookahead report line at
    // teardown, and the log callback is still registered to capture it.
    const json counters = moe_counters_json();
    for (auto it = counters.begin(); it != counters.end(); ++it) report[it.key()] = it.value();
    report["moeCountersSource"] = "runtime [moe-ssd] lines parsed from the captured loader log";
    report["moeCountersTruncated"] = loader_log_snapshot().size() >= LOADER_LOG_MAX;
    report["loaderLog"] = loader_log_snapshot();
    emit("Trial complete");
    return env->NewStringUTF(report.dump(2).c_str());
}
