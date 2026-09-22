#include "codec_precision.h"
#include "gpu_precision.h"
#include <moshi/moshi.h>
#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <unistd.h>
namespace {
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
std::string quote(const std::string& text){std::ostringstream s;s<<'"';for(unsigned char c:text){if(c=='"'||c=='\\')s<<'\\'<<c;else if(c<32)s<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;else s<<c;}s<<'"';return s.str();}
void checkpoint(const std::string& path,const std::string& data){
    auto temp=path+".tmp";FILE* f=fopen(temp.c_str(),"wb");if(!f)throw std::runtime_error("Cannot create precision checkpoint");
    bool ok=fwrite(data.data(),1,data.size(),f)==data.size();ok=fflush(f)==0&&ok;ok=fsync(fileno(f))==0&&ok;ok=fclose(f)==0&&ok;
    if(!ok||rename(temp.c_str(),path.c_str()))throw std::runtime_error("Cannot save precision checkpoint");
}
}
CodecPrecisionResult compareCodecPrecision(ggml_backend_t cpu,ggml_backend_t gpu,
    const std::string& model,const std::vector<float>& input,const std::string& output,
    const std::function<void(const std::string&)>& event){
    CodecPrecisionResult r;
    const char* names[]={"cpu_encode_cpu_decode","gpu_encode_cpu_decode","cpu_encode_gpu_decode","gpu_encode_gpu_decode","fp32_encode_cpu_decode","cpu_encode_fp32_decode","fp32_encode_fp32_decode"};
    const char* files[]={"generated.pcm","gpu_encode_cpu_decode.pcm","cpu_encode_gpu_decode.pcm","gpu_encode_gpu_decode.pcm","fp32_encode_cpu_decode.pcm","cpu_encode_fp32_decode.pcm","fp32_encode_fp32_decode.pcm"};
    double encode[3]{},decode[7]{},firstEncode[3]{},firstDecode[7]{},load=0;
    int codebookMismatches[2][8]{};
    auto began=Clock::now();std::string stage="initializing";
    auto report=[&](bool done){
        std::ostringstream s;s<<std::setprecision(10);
        s<<"{\"mode\":5,\"status\":"<<quote(!r.error.empty()?"ERROR":done?"PRECISION_COMPARISON_COMPLETE":"RUNNING")
         <<",\"summary\":\"CPU, default Vulkan and FP32 accumulation compared with independent streaming histories\""
         <<",\"sampleRate\":24000,\"frameSamples\":1920,\"inputFrames\":"<<r.frames<<",\"audioSamples\":"<<r.audio[0].count
         <<",\"audioRms\":"<<r.audio[0].rms()<<",\"stage\":"<<quote(stage)<<",\"gpuBackend\":"<<quote(ggml_backend_name(gpu))
         <<",\"loadMs\":"<<load<<",\"diagnosticWallMs\":"<<ms(began)<<",\"fp32Graphs\":"<<r.fp32Graphs<<",\"fp32Matmuls\":"<<r.fp32Matmuls
         <<",\"comparedTokens\":"<<r.comparedTokens<<",\"encoders\":[";
        for(int i=0;i<3;++i){if(i)s<<',';s<<"{\"name\":"<<quote(i==0?"cpu":i==1?"default_gpu":"fp32_gpu")<<",\"encodeMs\":"<<encode[i]<<",\"firstFrameMs\":"<<firstEncode[i]
            <<",\"tokenAgreement\":"<<(i==0?1:r.comparedTokens?1.-double(r.mismatches[i-1])/r.comparedTokens:0)<<",\"mismatchesByCodebook\":[";
            for(int c=0;c<8;++c){if(c)s<<',';s<<(i?codebookMismatches[i-1][c]:0);}s<<"]}";}
        s<<"],\"routes\":[";
        for(int i=0;i<7;++i){if(i)s<<',';auto& a=r.audio[i];s<<"{\"name\":"<<quote(names[i])<<",\"pcmFile\":"<<quote(files[i])<<",\"samples\":"<<a.count<<",\"rms\":"<<a.rms()<<",\"peak\":"<<a.peak<<",\"clippedSamples\":"<<a.clipped
            <<",\"nrmseVsCpuReference\":"<<a.nrmse()<<",\"correlationVsCpuReference\":"<<a.correlation()<<",\"levelRatioVsCpuReference\":"<<a.levelRatio()<<",\"levelCollapsed\":"<<(a.collapsed()?"true":"false")<<",\"decodeMs\":"<<decode[i]<<",\"firstFrameMs\":"<<firstDecode[i]<<'}';}
        s<<"],\"limitations\":\"Requested GGML_PREC_F32 for MUL_MAT; existing F16 weights and input conversions retained. Some kernels already accumulate in F32. Seven-route timing is not standalone throughput. CPU is a local reference, not official Kyutai parity. No full Moshi or live duplex validation.\"";
        if(!r.error.empty())s<<",\"error\":"<<quote(r.error);s<<'}';return s.str();
    };
    auto persist=[&]{checkpoint(output+"/partial.json",report(false));};
    auto precise=[&](auto work){GpuPrecisionScope scope(gpu);work();r.fp32Graphs+=scope.graphs;r.fp32Matmuls+=scope.matmuls;};
    try{
        if(!cpu||!gpu||cpu==gpu||input.empty())throw std::runtime_error("Invalid precision comparison inputs");
        stage="loading two Mimi instances";persist();event(stage);
        unref_ptr<moshi_context_t> cc=moshi_alloc(cpu,cpu),gc=moshi_alloc(gpu,cpu);
        unref_ptr<mimi_codec_t> cpuCodec=mimi_alloc(cc,model.c_str(),8),gpuCodec=mimi_alloc(gc,model.c_str(),8);
        if(mimi_frame_size(cpuCodec)!=1920||mimi_frame_size(gpuCodec)!=1920)throw std::runtime_error("Unexpected Mimi frame size");
        unref_ptr<mimi_encode_context_t> enc[3];
        enc[0]=mimi_encode_alloc_context(cpuCodec);enc[1]=mimi_encode_alloc_context(gpuCodec);
        precise([&]{enc[2]=mimi_encode_alloc_context(gpuCodec);});
        unref_ptr<mimi_decode_context_t> dec[7];
        for(int i=0;i<7;++i){auto init=[&]{dec[i]=mimi_decode_alloc_context(i==0||i==1||i==4?cpuCodec:gpuCodec);};if(i>=5)precise(init);else init();}
        std::ofstream pcm[7];for(int i=0;i<7;++i){pcm[i].open(output+"/"+files[i],std::ios::binary);if(!pcm[i])throw std::runtime_error("Cannot create precision PCM");}
        std::ofstream tokens(output+"/precision-tokens.csv"),timings(output+"/precision-frames.csv");
        if(!tokens||!timings)throw std::runtime_error("Cannot create precision diagnostics");
        tokens<<"frame,codebook,cpu,default_gpu,fp32_gpu\n";
        timings<<"frame,first_gpu_encoder,cpu_encode_ms,gpu_encode_ms,fp32_encode_ms";for(auto name:names)timings<<','<<name<<"_ms";timings<<'\n';
        std::vector<float> frame(1920),decoded[7];for(auto& d:decoded)d.resize(1920);
        std::array<int16_t,8> code[3];load=ms(began);
        size_t frames=(input.size()+1919)/1920;
        for(size_t f=0;f<frames;++f){
            std::fill(frame.begin(),frame.end(),0.f);std::copy_n(input.begin()+f*1920,std::min<size_t>(1920,input.size()-f*1920),frame.begin());
            double et[3]{},dt[7]{};
            for(int index:std::array<int,3>{0,f%2?2:1,f%2?1:2}){
                stage="frame "+std::to_string(f)+" encoder "+std::to_string(index);if(f==0){persist();event(stage);}
                auto start=Clock::now();auto work=[&]{mimi_encode_send(enc[index],frame.data());mimi_encode_receive(enc[index],code[index].data());};if(index==2)precise(work);else work();
                et[index]=ms(start);encode[index]+=et[index];if(f==0)firstEncode[index]=et[index];
                for(auto t:code[index])if(t<0||t>=2048)throw std::runtime_error("Invalid Mimi token");
            }
            for(int c=0;c<8;++c){++r.comparedTokens;for(int v=0;v<2;++v)if(code[0][c]!=code[v+1][c]){++r.mismatches[v];++codebookMismatches[v][c];}tokens<<f<<','<<c<<','<<code[0][c]<<','<<code[1][c]<<','<<code[2][c]<<'\n';}
            const int source[]={0,1,0,1,2,0,2};
            // Alternate the GPU variants to reduce fixed-order thermal bias; reference always first.
            auto order=f%2?std::array<int,7>{0,4,5,6,1,2,3}:std::array<int,7>{0,1,2,3,4,5,6};
            for(int i:order){
                stage="frame "+std::to_string(f)+" "+names[i];if(f==0){persist();event(stage);}
                auto start=Clock::now();auto work=[&]{mimi_decode_send(dec[i],code[source[i]].data());mimi_decode_receive(dec[i],decoded[i].data());};if(i>=5)precise(work);else work();
                dt[i]=ms(start);decode[i]+=dt[i];if(f==0)firstDecode[i]=dt[i];
                for(int j=0;j<1920;++j){r.audio[i].add(decoded[i][j],decoded[0][j]);int16_t sample=static_cast<int16_t>(std::clamp(decoded[i][j],-1.f,1.f)*32767.f);unsigned char b[]={static_cast<unsigned char>(sample&255),static_cast<unsigned char>((sample>>8)&255)};pcm[i].write(reinterpret_cast<char*>(b),2);}
            }
            timings<<f<<','<<(f%2?"fp32":"default");for(auto v:et)timings<<','<<v;for(auto v:dt)timings<<','<<v;timings<<'\n';++r.frames;
            if(f%6==0||f+1==frames){for(auto& p:pcm){p.flush();if(!p)throw std::runtime_error("Precision PCM write failed");}tokens.flush();timings.flush();if(!tokens||!timings)throw std::runtime_error("Precision log write failed");persist();event("Precision comparison: "+std::to_string(f+1)+" / "+std::to_string(frames)+" frames");}
        }
        if(!r.fp32Matmuls)throw std::runtime_error("FP32 policy did not reach a matrix multiplication");
        stage="complete";event("Precision comparison complete; releasing codec states");
    }catch(const std::exception& e){r.error=e.what();}
    r.json=report(true);checkpoint(output+"/precision.json",r.json);checkpoint(output+"/partial.json",r.json);return r;
}
