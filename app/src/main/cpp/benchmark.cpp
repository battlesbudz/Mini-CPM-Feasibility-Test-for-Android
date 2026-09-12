#include <jni.h>
#include <chrono>
#include <thread>
#include <mutex>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <unistd.h>
#include "common.h"
#include "omni.h"
#include "nlohmann/json.hpp"
using Clock=std::chrono::steady_clock;
using json=nlohmann::ordered_json;
static double ms(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
static std::string str(JNIEnv* env,jstring value){const char* p=env->GetStringUTFChars(value,nullptr);std::string out(p);env->ReleaseStringUTFChars(value,p);return out;}
extern "C" JNIEXPORT jstring JNICALL Java_com_battlesbudz_minicpmtest_NativeBench_run(
    JNIEnv* env,jclass,jstring jroot,jstring jchunks,jstring jout,jint frames,jobject listener){
    auto root=str(env,jroot),chunks=str(env,jchunks),output=str(env,jout);
    auto cls=env->GetObjectClass(listener);
    auto method=env->GetMethodID(cls,"onEvent","(Ljava/lang/String;)V");
    auto emit=[&](const std::string& s){jstring js=env->NewStringUTF(s.c_str());env->CallVoidMethod(listener,method,js);env->DeleteLocalRef(js);};
    json report={{"schemaVersion",1},{"backend","cpu"},{"runtimeCommit","64d092c60db4b4ee45768476bd752f03fdcc98ea"},
        {"framesExpected",frames},{"threads",4},{"contextTokens",4096},{"visionEnabled",false},
        {"scope","recorded_audio_compute_screen_not_acoustic_duplex"}};
    omni_context* ctx=nullptr;
    common_params params;
    std::mutex mutex;
    json audio=json::array(),decisions=json::array();
    long long samples=0,invalid=0;
    double energy=0;
    int sampleRate=24000;
    auto start=Clock::now();
    std::ofstream pcm(output+"/generated.pcm",std::ios::binary);
    try{
        setenv("OMNI_VOC_DEVICE","cpu",1);
        params.model.path=root+"/MiniCPM-o-4_5-Q4_K_M.gguf";
        params.apm_model=root+"/audio/MiniCPM-o-4_5-audio-F16.gguf";
        params.tts_model=root+"/tts/MiniCPM-o-4_5-tts-F16.gguf";
        params.n_ctx=4096;params.n_gpu_layers=0;
        params.cpuparams.n_threads=4;params.cpuparams_batch.n_threads=4;
        params.sampling.seed=42;
        common_init();
        emit("Loading complete audio-only model on CPU");
        auto loadStart=Clock::now();
        ctx=omni_init(&params,1,true,root+"/tts",0,"cpu",true,nullptr,nullptr,output);
        if(!ctx)throw std::runtime_error("omni_init failed");
        report["loadMs"]=ms(loadStart,Clock::now());
        if(!ctx->token2wav_initialized||ctx->use_python_token2wav)throw std::runtime_error("Native Token2Wav unavailable; refusing a partial benchmark");
        ctx->async=true;ctx->ref_audio_path=root+"/reference.wav";
        if(!pcm)throw std::runtime_error("Cannot create PCM output");
        emit("Preparing duplex voice prompt");
        if(!omni_duplex_session_begin(ctx,ctx->ref_audio_path,output))throw std::runtime_error("Duplex prompt initialization failed");
        start=Clock::now();
        ctx->audio_output_cb=[&](const float* data,int count,int rate,bool final){
            std::lock_guard<std::mutex> guard(mutex);
            audio.push_back({{"atMs",ms(start,Clock::now())},{"samples",count},{"sampleRate",rate},{"final",final}});
            sampleRate=rate;
            for(int i=0;i<count;i++){
                float x=data[i];if(!std::isfinite(x)){invalid++;x=0;}
                energy+=double(x)*x;
                int16_t value=static_cast<int16_t>(std::clamp(x,-1.0f,1.0f)*32767);
                pcm.write(reinterpret_cast<const char*>(&value),sizeof(value));
            }
            samples+=count;
        };
        emit("Running scheduled one-second audio frames");
        // Schedule independently of inference: measured latency includes backlog.
        std::thread producer([&]{for(int i=0;i<frames;i++){
            std::this_thread::sleep_until(start+std::chrono::seconds(i));
            char name[32];snprintf(name,sizeof(name),"/%04d.wav",i%30);
            OmniDuplexFrame frame;frame.aud_fname=chunks+name;frame.user_seq=i+1;
            if(omni_duplex_push_frame(ctx,frame)<0)break;
        }});
        int completed=0,speak=0;
        for(int i=0;i<frames;i++){
            OmniDuplexFrameResult result;
            if(!omni_duplex_wait_next_frame(ctx,&result,90000)||!result.ok){
                emit("Frame failed or timed out; stopping worker");
                report["error"]="Native frame timeout or failure";
                report["framesCompleted"]=completed;
                std::ofstream(output+"/native_failure.json")<<report.dump(2);
                // Avoid a producer blocked in upstream code. The UI survives in
                // a different process and can display the last checkpoint.
                _exit(2);
            }
            completed++;if(result.is_speak)speak++;
            double now=ms(start,Clock::now());
            decisions.push_back({{"sequence",result.user_seq},{"dueMs",(result.user_seq-1)*1000},
                {"doneMs",now},{"latencyMs",now-(result.user_seq-1)*1000},{"decodeMs",result.ms_decode},
                {"speak",result.is_speak},{"kvTokens",result.n_past_after}});
            emit("Frame "+std::to_string(completed)+" / "+std::to_string(frames)+(result.is_speak?" - speaking":" - listening"));
        }
        producer.join();omni_duplex_session_end(ctx);
        emit("Finishing generated audio");
        bool drained=omni_duplex_drain_tts_audio(ctx,60000,1000);
        omni_free(ctx);ctx=nullptr;
        pcm.flush();if(!pcm)throw std::runtime_error("PCM output write failed");pcm.close();
        report["framesCompleted"]=completed;report["speakFrames"]=speak;
        report["audioDrained"]=drained;report["audioSamples"]=samples;report["sampleRate"]=sampleRate;
        report["invalidSamples"]=invalid;report["audioRms"]=samples?std::sqrt(energy/samples):0;
        report["elapsedMs"]=ms(start,Clock::now());report["decisions"]=decisions;report["audioChunks"]=audio;
        std::ofstream(output+"/native_report.json")<<report.dump(2);
    }catch(const std::exception& e){report["error"]=e.what();if(ctx){omni_free(ctx);ctx=nullptr;}}
    return env->NewStringUTF(report.dump().c_str());
}
