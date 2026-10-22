// MiniCPM-V 4.7 phone trial — native benchmark.
//
// Loads the phone-only `-les` split of MiniCPM-V 4.7 35B-A3B through the
// llama.cpp-omni fork's LES / MoE-SSD runtime (routed experts streamed from
// the sibling experts/ directory via llama_model_params.moe_experts_path),
// then runs three fixed questions about one bundled street photo and reports
// measured timings, token rates, peak RSS and the verbatim answers as JSON.
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
#include <chrono>
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

    json report = {
        {"schemaVersion", 2},
        {"profile", "minicpm-v47-phone-trial-v1"},
        {"runtime", "llama.cpp-omni (tc-mb fork, LES / MoE-SSD)"},
        {"runtimeCommit", RUNTIME_COMMIT},
        {"runtimePatch", "v47/patches/ggml-cpu-alloc-only-buft.patch (CPU alloc-only buft for the LES load path)"},
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
    const auto load_start = Clock::now();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.use_mmap = true;
    mparams.moe_experts_path = experts_path.c_str();
    mparams.moe_slot_bank = MOE_SLOT_BANK;
    llama_model * model = llama_model_load_from_file(main_path.c_str(), mparams);
    if (!model) return fail("model_load", "llama_model_load_from_file returned null for " + main_path);
    report["loadModelMs"] = ms_between(load_start, Clock::now());
    {
        char desc[256];
        llama_model_desc(model, desc, sizeof(desc));
        report["modelDesc"] = desc;
        report["modelParams"] = (double) llama_model_n_params(model);
    }
    emit("Main model loaded");

    emit("Loading vision projector (mmproj)");
    const auto mmproj_start = Clock::now();
    mtmd_context_params vparams = mtmd_context_params_default();
    vparams.use_gpu = false;
    vparams.print_timings = false;
    vparams.n_threads = N_THREADS;
    vparams.image_max_slice_nums = 9; // vendor demo default for MiniCPM-V
    mtmd_context * vision = mtmd_init_from_file(mmproj_path.c_str(), model, vparams);
    if (!vision) { llama_model_free(model); return fail("mmproj_load", "mtmd_init_from_file returned null for " + mmproj_path); }
    report["loadMmprojMs"] = ms_between(mmproj_start, Clock::now());
    report["visionSupported"] = mtmd_support_vision(vision);

    emit("Creating context");
    const auto ctx_start = Clock::now();
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = N_CTX;
    cparams.n_batch = N_BATCH;
    cparams.n_ubatch = 512;
    cparams.n_threads = N_THREADS;
    cparams.n_threads_batch = N_THREADS;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { mtmd_free(vision); llama_model_free(model); return fail("context", "llama_init_from_model returned null"); }
    report["contextMs"] = ms_between(ctx_start, Clock::now());

    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_batch batch = llama_batch_init(1, 0, 1);

    const char * questions[] = {
        "Describe this image.",
        "How many people are in this image?",
        "How many cars are in this image?",
    };
    json cases = json::array();
    std::string fatal;
    std::string fatal_phase;

    for (int qi = 0; qi < 3 && fatal.empty(); qi++) {
        const std::string question = questions[qi];
        emit("Case " + std::to_string(qi + 1) + "/3: " + question);
        json kase = {{"index", qi + 1}, {"prompt", question}};
        const auto case_start = Clock::now();
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_pos n_past = 0;

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
        kase["tokenizeMs"] = ms_between(tok_start, Clock::now());
        mtmd_bitmap_free(bitmap);
        if (tres != 0) {
            mtmd_input_chunks_free(chunks);
            fatal_phase = "tokenize"; fatal = "mtmd_tokenize failed with code " + std::to_string(tres); break;
        }

        double vision_ms = 0.0, prefill_ms = 0.0;
        size_t image_tokens = 0, prompt_tokens = 0;
        bool eval_ok = true;
        const size_t n_chunks = mtmd_input_chunks_size(chunks);
        for (size_t ci = 0; ci < n_chunks; ci++) {
            const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, ci);
            const bool is_image = mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE;
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
            if (is_image) vision_ms += chunk_ms;
            n_past = new_n_past;
        }
        mtmd_input_chunks_free(chunks);
        kase["promptTokens"] = prompt_tokens;
        kase["imageTokens"] = image_tokens;
        kase["visionEvalMs"] = vision_ms;
        kase["prefillMs"] = prefill_ms;
        if (!eval_ok) { fatal_phase = "prefill"; fatal = "mtmd_helper_eval_chunk_single failed in case " + std::to_string(qi + 1); break; }

        // Greedy generation.
        const auto gen_start = Clock::now();
        std::string answer;
        int generated = 0;
        bool ended_eos = false;
        double decode_calls_ms = 0.0;
        while (generated < MAX_NEW_TOKENS) {
            const llama_token token = greedy_token(ctx, vocab);
            if (llama_vocab_is_eog(vocab, token)) { ended_eos = true; break; }
            if (generated == 0) kase["ttftMs"] = ms_between(case_start, Clock::now());
            answer += token_piece(vocab, token);
            generated++;
            if (generated % 32 == 0) emit("Case " + std::to_string(qi + 1) + ": " + std::to_string(generated) + " tokens");
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
        kase["peakRssKb"] = peak_rss_kb();
        kase["answer"] = answer;
        cases.push_back(kase);
        emit("Case " + std::to_string(qi + 1) + " complete: " + std::to_string(generated) + " tokens");
    }

    report["cases"] = cases;
    llama_batch_free(batch);
    llama_free(ctx);
    mtmd_free(vision);
    llama_model_free(model);
    if (!fatal.empty()) return fail(fatal_phase, fatal);
    report["peakRssKb"] = peak_rss_kb();
    report["totalMs"] = ms_between(trial_start, Clock::now());
    report["loaderLog"] = loader_log_snapshot();
    emit("Trial complete");
    return env->NewStringUTF(report.dump(2).c_str());
}
