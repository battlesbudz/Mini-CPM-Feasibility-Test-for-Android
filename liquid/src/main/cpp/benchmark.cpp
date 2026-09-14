#include <jni.h>
#include <chrono>
#include <fstream>
#include <cmath>
#include <cstdio>
#include "runner.h"
#include "mtmd.h"
#include "common.h"
#include "nlohmann/json.hpp"
using Clock=std::chrono::steady_clock;
using json=nlohmann::ordered_json;
static double ms(Clock::time_point a){return std::chrono::duration<double,std::milli>(Clock::now()-a).count();}
static std::string str(JNIEnv* env,jstring value){const char* p=env->GetStringUTFChars(value,nullptr);std::string out(p);env->ReleaseStringUTFChars(value,p);return out;}
extern "C" JNIEXPORT jstring JNICALL Java_com_battlesbudz_liquidtest_NativeBench_run(
 JNIEnv* env,jclass,jstring jroot,jstring jchunks,jstring jout,jint,jobject listener){
 auto root=str(env,jroot),chunks=str(env,jchunks),output=str(env,jout);
 auto cls=env->GetObjectClass(listener);auto method=env->GetMethodID(cls,"onEvent","(Ljava/lang/String;)V");
 auto emit=[&](const std::string& s){jstring js=env->NewStringUTF(s.c_str());env->CallVoidMethod(listener,method,js);env->DeleteLocalRef(js);};
 json report={{"model","LFM2.5-Audio-1.5B"},{"profile","liquid-q4-cpu-smoke-v1"},{"backend","cpu"},
 {"runtimeCommit","ec9d1fdd9cc18643c5e161b65b8053f6f6f34a4b"},{"contextTokens",4096},{"batchTokens",256},{"microBatchTokens",64},
 {"threads",4},{"maxGenerationSteps",2048},{"scope","recorded_question_not_live_duplex"},{"generationTermination","not_exposed_by_runner"}};
 try{
  // This process is dedicated to one test. Retain native errors even after a crash.
  freopen((output+"/native.log").c_str(),"a",stderr);
  common_params params;params.model.path=root+"/LFM2.5-Audio-1.5B-Q4_0.gguf";
  params.mmproj.path=root+"/mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf";
  params.vocoder.model.path=root+"/vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf";
  params.vocoder.speaker_file=root+"/tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf";
  params.n_ctx=4096;params.n_batch=256;params.n_ubatch=64;params.n_gpu_layers=0;params.mmproj_use_gpu=false;
  params.cpuparams.n_threads=4;params.cpuparams_batch.n_threads=4;params.sampling.seed=42;
  common_init();liquid::audio::Runner runner;
  emit("Loading complete Liquid Q4 voice stack on CPU");auto began=Clock::now();
  if(runner.init(params)!=0)throw std::runtime_error("Liquid initialization failed; see native log");
  report["loadMs"]=ms(began);int rate=runner.get_output_sample_rate();
  if(rate<=0)throw std::runtime_error("Invalid output sample rate");
  std::ifstream in(chunks+"/question.wav",std::ios::binary|std::ios::ate);if(!in)throw std::runtime_error("Question missing");
  auto size=in.tellg();if(size<=44||size>4*1024*1024)throw std::runtime_error("Invalid question size");
  std::vector<std::byte> wav(static_cast<size_t>(size));in.seekg(0);in.read(reinterpret_cast<char*>(wav.data()),size);
  if(!in)throw std::runtime_error("Question read failed");
  std::vector<liquid::audio::Runner::Message> messages={{"system",liquid::audio::Runner::interleaved_system_prompt,{}},{"user",mtmd_default_marker(),std::move(wav)}};
  std::ofstream pcm(output+"/generated.pcm",std::ios::binary);if(!pcm)throw std::runtime_error("Cannot create PCM output");
  std::string text;long long samples=0;double energy=0,firstPcm=-1,firstText=-1;json audio=json::array();
  emit("Processing recorded question and generating speech");began=Clock::now();
  auto textCb=[&](const std::string& piece){if(!piece.empty()&&firstText<0)firstText=ms(began);text+=piece;};
  auto audioCb=[&](const std::vector<int16_t>& data){
   if(data.empty())return;double at=ms(began);if(firstPcm<0){firstPcm=at;emit("First speech produced; continuing answer");}
   audio.push_back({{"atMs",at},{"samples",data.size()}});
   pcm.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(int16_t));
   for(auto x:data){double value=double(x)/32768.0;energy+=value*value;}samples+=data.size();
  };
  int result=runner.generate(messages,2048,textCb,audioCb,{MTMD_OUTPUT_MODALITY_AUDIO,MTMD_OUTPUT_MODALITY_TEXT});
  report["generationMs"]=ms(began);pcm.flush();
  report["text"]=text;report["audioSamples"]=samples;report["audioRms"]=samples?std::sqrt(energy/samples):0;
  report["sampleRate"]=rate;report["audioMs"]=1000.0*samples/rate;report["audioChunks"]=audio;
  report["firstPcmMs"]=firstPcm<0?json(nullptr):json(firstPcm);report["firstTextMs"]=firstText<0?json(nullptr):json(firstText);
  if(result!=0)report["error"]=std::string(runner.get_last_error()?runner.get_last_error():"Generation failed");
  if(!pcm)report["error"]="PCM write failed";
  emit("Saving voice output and diagnostics");
 }catch(const std::exception& error){report["error"]=error.what();}
 std::ofstream(output+"/native_report.json")<<report.dump(2);
 return env->NewStringUTF(report.dump().c_str());
}
