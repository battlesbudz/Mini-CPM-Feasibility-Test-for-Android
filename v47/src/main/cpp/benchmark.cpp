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
#include <chrono>
#include <cstdint>
#include <cstdio>
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
// 256 is the vendor demo's own default for this model
// (MiniCPMV47Const.defaultSlotBank; min = expert_used_count 8,
// max = expert_count 256). Resident cost on this pack:
// 256 slots x 1,351,680 B per-slot-per-layer bytes = ~330 MiB.
static const int    MOE_SLOT_BANK  = 256;
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
        JNIEnv * env, jclass, jstring jroot, jstring jimage, jstring jout, jobject listener) {
    const std::string root  = jstr(env, jroot);
    const std::string image = jstr(env, jimage);
    const std::string out   = jstr(env, jout);
    (void) out; // report is returned to the service, which persists it
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

    json report = {
        {"schemaVersion", 2},
        {"profile", "minicpm-v47-phone-trial-v1"},
        {"runtime", "llama.cpp-omni (tc-mb fork, LES / MoE-SSD)"},
        {"runtimeCommit", RUNTIME_COMMIT},
        {"runtimePatch", "v47/patches/ggml-cpu-alloc-only-buft.patch (CPU alloc-only buft + llama-model.cpp loader fix)"},
        {"modelRepo", "tc-mb/MiniCPM-V-Apps-gguf"},
        {"modelRevision", "8716e5209f308fd53f197ccd7d50c6fdcc2dc37d"},
        {"modelFiles", "main -les gguf + mmproj f16 + experts/ (40 shards + manifest.json)"},
        {"threads", N_THREADS},
        {"contextTokens", N_CTX},
        {"batchTokens", N_BATCH},
        {"maxNewTokens", MAX_NEW_TOKENS},
        {"sampling", "greedy (temperature 0)"},
        {"thinking", "disabled via closed think block (enable_thinking=false equivalent)"},
    };
    const auto trial_start = Clock::now();
    auto fail = [&](const std::string & phase, const std::string & message) -> jstring {
        emit_progress({{"stage", "failed"}, {"event", "done"}, {"errorPhase", phase},
                       {"label", "Failed at " + phase}});
        report["errorPhase"] = phase;
        report["error"] = message;
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
        while (generated < MAX_NEW_TOKENS) {
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
            if (generated >= MAX_NEW_TOKENS) break;
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
        const int display = qi + 1;
        const std::string tag = "Case " + std::to_string(display);
        const std::string question = trial_cases[qi].question;
        const bool has_image = trial_cases[qi].has_image;
        emit(tag + "/" + std::to_string(n_cases) + ": " + question);
        json kase = {{"index", display}, {"type", has_image ? "image" : "text"}, {"prompt", question}};
        const auto case_start = Clock::now();
        llama_memory_clear(llama_get_memory(ctx), true);
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
    report["loaderLog"] = loader_log_snapshot();
    emit("Trial complete");
    return env->NewStringUTF(report.dump(2).c_str());
}
