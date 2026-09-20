#include <jni.h>
#include <moshi/moshi.h>
#include <ggml-vulkan.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
std::string utf(JNIEnv* env, jstring value) {
    const char* data = env->GetStringUTFChars(value, nullptr);
    if (!data) throw std::runtime_error("Cannot read JNI argument");
    std::string result(data); env->ReleaseStringUTFChars(value, data); return result;
}
// JSON text can contain SentencePiece's UTF-8; NewStringUTF expects modified UTF-8.
jstring javaString(JNIEnv* env, const std::string& value) {
    jbyteArray bytes = env->NewByteArray(static_cast<jsize>(value.size()));
    if (!bytes) return nullptr;
    env->SetByteArrayRegion(bytes, 0, static_cast<jsize>(value.size()), reinterpret_cast<const jbyte*>(value.data()));
    jclass cls = env->FindClass("java/lang/String");
    jmethodID ctor = env->GetMethodID(cls, "<init>", "([BLjava/lang/String;)V");
    jstring charset = env->NewStringUTF("UTF-8");
    auto result = static_cast<jstring>(env->NewObject(cls, ctor, bytes, charset));
    env->DeleteLocalRef(bytes); env->DeleteLocalRef(cls); env->DeleteLocalRef(charset);
    return result;
}
std::string quoted(const std::string& text) {
    std::ostringstream out; out << '"';
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else out << c;
    }
    out << '"'; return out.str();
}
void save(const std::string& path, const std::string& data) {
    auto temp = path + ".tmp";
    FILE* file = fopen(temp.c_str(), "wb");
    if (!file) throw std::runtime_error("Cannot create checkpoint");
    bool ok = fwrite(data.data(), 1, data.size(), file) == data.size();
    ok = fflush(file) == 0 && ok;
    ok = fsync(fileno(file)) == 0 && ok;
    ok = fclose(file) == 0 && ok;
    if (!ok || rename(temp.c_str(), path.c_str()) != 0) throw std::runtime_error("Cannot persist checkpoint");
}
uint32_t le(const unsigned char* p, int n) {
    uint32_t value = 0; for (int i=0; i<n; ++i) value |= uint32_t(p[i]) << (8*i); return value;
}
std::vector<float> readInput(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    unsigned char h[44]{};
    if (!in.read(reinterpret_cast<char*>(h), 44) || std::string(reinterpret_cast<char*>(h),4)!="RIFF" ||
        std::string(reinterpret_cast<char*>(h+8),8)!="WAVEfmt " || le(h+16,4)!=16 || le(h+20,2)!=1 ||
        le(h+22,2)!=1 || le(h+24,4)!=24000 || le(h+34,2)!=16 || std::string(reinterpret_cast<char*>(h+36),4)!="data")
        throw std::runtime_error("Record a 24 kHz mono PCM question in this app first");
    uint32_t count = le(h+40,4);
    if (count == 0 || count > 24000*2*30 || count%2) throw std::runtime_error("Invalid recording length");
    std::vector<unsigned char> pcm(count);
    if (!in.read(reinterpret_cast<char*>(pcm.data()),count)) throw std::runtime_error("Truncated recording");
    std::vector<float> result(count/2);
    for (size_t i=0;i<result.size();++i) result[i]=static_cast<int16_t>(le(pcm.data()+i*2,2))/32768.f;
    return result;
}
struct Backends {
    ggml_backend_t cpu = nullptr, gpu = nullptr;
    ~Backends() { if(gpu) ggml_backend_free(gpu); if(cpu) ggml_backend_free(cpu); }
};
struct Stats {
    std::vector<double> frames;
    double encode=0, model=0, decode=0, load=0, firstPcm=-1, sumSquares=0;
    int outputFrames=0, slowFrames=0; int64_t samples=0;
    std::string text, backendName, codecBackendName, error;
    double percentile(double q) const {
        if(frames.empty()) return 0;
        auto sorted=frames; std::sort(sorted.begin(),sorted.end());
        return sorted[std::min(sorted.size()-1,static_cast<size_t>(std::ceil(q*sorted.size())-1))];
    }
    std::string json(int mode, int context, double wall, const std::string& status) const {
        double total=0; for(double t:frames) total+=t;
        std::ostringstream out; out << std::setprecision(10);
        out << "{\"status\":" << quoted(status) << ",\"mode\":" << mode
            << ",\"backend\":" << quoted(backendName) << ",\"codecBackend\":" << quoted(codecBackendName)
            << ",\"contextFrames\":" << context << ",\"sampleRate\":24000,\"frameSamples\":1920"
            << ",\"inputFrames\":" << frames.size() << ",\"outputFrames\":" << outputFrames
            << ",\"audioSamples\":" << samples << ",\"audioRms\":" << (samples?std::sqrt(sumSquares/samples):0)
            << ",\"loadMs\":" << load << ",\"firstPcmMs\":" << firstPcm
            << ",\"processingMs\":" << total << ",\"replayWallMs\":" << wall
            << ",\"processingFps\":" << (total>0?frames.size()*1000.0/total:0)
            << ",\"replayWallFps\":" << (wall>0?frames.size()*1000.0/wall:0)
            << ",\"encodeMs\":" << encode << ",\"modelMs\":" << model << ",\"decodeMs\":" << decode
            << ",\"frameP50Ms\":" << percentile(.50) << ",\"frameP95Ms\":" << percentile(.95)
            << ",\"frameP99Ms\":" << percentile(.99) << ",\"framesOver80Ms\":" << slowFrames
            << ",\"text\":" << quoted(text)
            << ",\"limitations\":\"Unpaced recorded-audio smoke test; no live duplex, echo cancellation, reference parity or sustained thermal validation. Processing time excludes model load and playback.\"";
        if(!error.empty()) out << ",\"error\":" << quoted(error);
        out << "}"; return out.str();
    }
};
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_battlesbudz_moshitest_NativeBench_run(JNIEnv* env, jclass, jstring modelsArg,
    jstring inputArg, jstring outputArg, jint mode, jint backend, jint context, jobject listener) {
    Stats stats; auto started=Clock::now(); double replayWall=0;
    try {
        const std::string root=utf(env,modelsArg), inputPath=utf(env,inputArg), outDir=utf(env,outputArg);
        if(mode<0 || mode>2 || backend<0 || backend>2 || (context!=750 && context!=1000 && context!=3000))
            throw std::runtime_error("Invalid benchmark configuration");
        if(!freopen((outDir+"/native.log").c_str(),"w",stderr)) throw std::runtime_error("Cannot open native log");
        setvbuf(stderr,nullptr,_IONBF,0);
        dup2(fileno(stderr),fileno(stdout)); setvbuf(stdout,nullptr,_IONBF,0);
        jclass cls=env->GetObjectClass(listener);
        jmethodID callback=env->GetMethodID(cls,"onEvent","(Ljava/lang/String;)V");
        env->DeleteLocalRef(cls);
        auto event=[&](const std::string& message) {
            fprintf(stderr,"moshi_phase %s\n",message.c_str());
            jstring s=javaString(env,message); env->CallVoidMethod(listener,callback,s); env->DeleteLocalRef(s);
            if(env->ExceptionCheck()) { env->ExceptionClear(); throw std::runtime_error("Progress callback failed"); }
        };
        std::vector<float> input;
        if(mode!=1) input=readInput(inputPath);
        event("Initializing selected backend");
        Backends devices;
        devices.cpu=ggml_backend_cpu_init();
        if(!devices.cpu) throw std::runtime_error("CPU backend unavailable");
        ggml_backend_cpu_set_n_threads(devices.cpu,4);
        if(backend!=0) {
            if(ggml_backend_vk_get_device_count()==0) throw std::runtime_error("Vulkan device unavailable; choose CPU");
            devices.gpu=ggml_backend_vk_init(0);
            if(!devices.gpu) throw std::runtime_error("Vulkan initialization failed");
            char description[256]{};
            ggml_backend_vk_get_device_description(0,description,sizeof(description));
            event(std::string("Vulkan device: ")+description);
        }
        auto modelBackend=backend==0?devices.cpu:devices.gpu;
        auto codecBackend=backend==2?devices.gpu:devices.cpu;
        stats.backendName=ggml_backend_name(modelBackend);
        stats.codecBackendName=ggml_backend_name(codecBackend);
        unref_ptr<moshi_context_t> mainContext=moshi_alloc(modelBackend,devices.cpu);
        unref_ptr<moshi_context_t> codecContext=moshi_alloc(codecBackend,devices.cpu);
        moshi_config_t config{};
        unref_ptr<moshi_lm_t> lm;
        unref_ptr<moshi_lm_gen_t> generator;
        unref_ptr<tokenizer_t> tokenizer;
        if(mode!=0) {
            event("Reading Moshi configuration");
            if(moshi_get_config(&config,(root+"/config.json").c_str())!=0) throw std::runtime_error("Invalid Moshi configuration");
            config.context=context;
            event("Allocating Moshi Q4_K weights");
            lm=moshi_lm_from_files(mainContext,&config,(root+"/model-q4_k.gguf").c_str());
            if(!lm.ptr) throw std::runtime_error("Model allocation failed");
            generator=moshi_lm_generator(lm);
            tokenizer=tokenizer_alloc((root+"/tokenizer_spm_32k_3.model").c_str(),false);
            event("Loading Moshi Q4_K weights");
            if(moshi_lm_load(lm)!=0) throw std::runtime_error("Model load failed");
        }
        event("Loading Mimi codec");
        unref_ptr<mimi_codec_t> codec=mimi_alloc(codecContext,(root+"/mimi-e351c8d8-125.gguf").c_str(),8);
        if(!codec.ptr || mimi_frame_size(codec)!=1920 || std::abs(mimi_frame_rate(codec)-12.5f)>.001f)
            throw std::runtime_error("Unexpected codec frame format");
        unref_ptr<mimi_encode_context_t> encoder=mimi_encode_alloc_context(codec);
        unref_ptr<mimi_decode_context_t> decoder=mimi_decode_alloc_context(codec);
        if(mode!=0) { event("Initializing streaming model state"); srand(0); moshi_lm_start(mainContext,generator,0.8f,0.7f); }
        stats.load=elapsed(started);
        if(mode==1) {
            event("Load test complete; releasing model state");
            return javaString(env,stats.json(mode,context,0,"LOAD_OK"));
        }
        event(mode==0?"Replaying recording through Mimi":"Replaying recording through Moshi");
        std::ofstream audio(outDir+"/generated.pcm",std::ios::binary);
        std::ofstream timings(outDir+"/frames.csv");
        if(!audio || !timings) throw std::runtime_error("Cannot create replay output");
        timings << "frame,encode_ms,model_ms,decode_ms,total_ms,output\n";
        std::vector<float> frame(1920), decoded(1920);
        std::vector<int16_t> tokens(8);
        const size_t recordedFrames=(input.size()+1919)/1920;
        const size_t frameCount=recordedFrames+(mode==2?125:0); // Ten seconds of silence for a response.
        auto replayStart=Clock::now();
        for(size_t i=0;i<frameCount;++i) {
            std::fill(frame.begin(),frame.end(),0.f);
            for(size_t j=0;j<frame.size() && i*1920+j<input.size();++j) frame[j]=input[i*1920+j];
            auto step=Clock::now();
            mimi_encode_send(encoder,frame.data()); mimi_encode_receive(encoder,tokens.data());
            double enc=elapsed(step); auto modelStart=Clock::now();
            int token=0; bool output=true;
            if(mode==2) { moshi_lm_send2(generator,tokens); output=moshi_lm_receive(generator,token,tokens)!=0; }
            double modelMs=elapsed(modelStart); auto decodeStart=Clock::now();
            if(output) {
                if(tokens.size()!=8) throw std::runtime_error("Unexpected generated codebook count");
                for(auto t:tokens) if(t<0 || t>=2048) throw std::runtime_error("Invalid audio token");
                mimi_decode_send(decoder,tokens.data()); mimi_decode_receive(decoder,decoded.data());
            }
            double dec=elapsed(decodeStart), total=elapsed(step);
            stats.encode+=enc;stats.model+=modelMs;stats.decode+=dec;stats.frames.push_back(total);
            if(total>80) ++stats.slowFrames;
            timings << i << ',' << enc << ',' << modelMs << ',' << dec << ',' << total << ',' << output << '\n';
            if(output) {
                if(stats.firstPcm<0) stats.firstPcm=elapsed(replayStart);
                for(float f:decoded) {
                    if(!std::isfinite(f)) throw std::runtime_error("Non-finite output audio; backend correctness failure");
                    stats.sumSquares+=double(f)*f; ++stats.samples;
                    int16_t sample=static_cast<int16_t>(std::clamp(f,-1.f,1.f)*32767.f);
                    unsigned char bytes[2]={static_cast<unsigned char>(sample&255),static_cast<unsigned char>((sample>>8)&255)};
                    audio.write(reinterpret_cast<char*>(bytes),2);
                }
                ++stats.outputFrames;
                if(mode==2 && token!=0 && token!=3) {
                    auto piece=tokenizer_id_to_piece(tokenizer,token);
                    const std::string space="\xe2\x96\x81";
                    for(size_t at=0;(at=piece.find(space,at))!=std::string::npos;) piece.replace(at,space.size()," ");
                    stats.text+=piece;
                }
            }
            if(i%12==0 || i+1==frameCount) {
                audio.flush();timings.flush();
                if(!audio || !timings) throw std::runtime_error("Replay output write failed");
                save(outDir+"/partial.json",stats.json(mode,context,elapsed(replayStart),"RUNNING"));
                event("Processed "+std::to_string(i+1)+" / "+std::to_string(frameCount)+" audio frames");
            }
        }
        replayWall=elapsed(replayStart);
        event("Replay complete; releasing model state");
    } catch(const std::exception& error) { stats.error=error.what(); }
    catch(...) { stats.error="Unknown native exception"; }
    return javaString(env,stats.json(mode,context,replayWall,stats.error.empty()?"REPLAY_COMPLETE":"ERROR"));
}
