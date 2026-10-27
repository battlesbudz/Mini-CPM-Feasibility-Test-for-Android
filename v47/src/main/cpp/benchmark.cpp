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
#include <mutex>
#include <string>
#include <vector>
#include <sys/resource.h>
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "nlohmann/json.hpp"

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
    const bool quick = mode == "quick";
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
    setenv("LLAMA_MOE_DECODE_WINDOW", std::to_string(MOE_DECODE_WINDOW).c_str(), 1);
    // Build 12 (research 2.1): lookahead prefetch ON, depth left at
    // the fork default of 8. The runtime predicts the next layer's
    // routing from the current hidden state and pre-reads those
    // experts during the current layer's compute; hits remap for free.
    setenv("LLAMA_MOE_LOOKAHEAD_PREFETCH", "1", 1);
    report["moeLookaheadPrefetch"] = "on (depth default 8)";
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
    report["moeDecodeWindow"] = MOE_DECODE_WINDOW;
    report["moePrewarmTable"] = prewarm_desc.empty() ? "WRITE FAILED - prewarm off" : prewarm_desc;
    emit("MoE decode window on: " + std::to_string(MOE_DECODE_WINDOW) +
         " slots/layer, lookahead prefetch on, prewarm table " +
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
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { llama_model_free(model); return fail("context", "llama_init_from_model returned null"); }
    const double context_ms = ms_between(ctx_start, Clock::now());
    report["contextMs"] = context_ms;
    emit_progress({{"stage", "context_create"}, {"event", "done"}, {"label", "Context created"},
                   {"contextMs", context_ms}, {"stageElapsedMs", context_ms}});

    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_batch batch = llama_batch_init(1, 0, 1);

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
                off += (size_t) chunk_n;
            }
            const double prefill_ms = ms_between(prefill_start, Clock::now());
            if (!prefill_ok) {
                fatal_phase = "prefill";
                fatal = "llama_decode failed during text prefill in case " + std::to_string(display);
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

            run_generation(display, case_start, kase, n_past);
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
