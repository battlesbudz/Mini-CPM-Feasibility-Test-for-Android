// Included inside benchmark.cpp's anonymous namespace; uses its JSON/I/O helpers.
// Independent decoder histories ensure each test consumes only its own token stream.
std::string compareCodec(const std::string& root, const std::vector<float>& input,
    const std::string& outDir, const std::function<void(const std::string&)>& event) {
    const char* names[]={"cpu_encode_cpu_decode","gpu_encode_cpu_decode","cpu_encode_gpu_decode","gpu_encode_gpu_decode"};
    const char* files[]={"generated.pcm","gpu_encode_cpu_decode.pcm","cpu_encode_gpu_decode.pcm","gpu_encode_gpu_decode.pcm"};
    AudioMetrics audio[4];
    double encMs[2]{},decMs[4]{};
    uint64_t comparedTokens=0,mismatchedTokens=0;
    int firstMismatch=-1,completeFrames=0;
    int mismatchesByCodebook[8]{};
    std::string error,device;
    double inputEnergy=0;for(float x:input) inputEnergy+=double(x)*x;
    const double inputRms=std::sqrt(inputEnergy/input.size());
    const auto began=Clock::now();double loadMs=0;
    auto result=[&](bool finished) {
        bool usableReference=audio[0].rms()>=0.01;
        // Differences are diagnostic signals, not proof of which mathematical operator is wrong.
        std::string status="RUNNING",summary="Comparing CPU and Vulkan with identical input";
        if(!error.empty()) {status="ERROR";summary=error;}
        else if(finished) {
            if(!usableReference) {status="REFERENCE_TOO_QUIET";summary="CPU reference is too quiet for a reliable level comparison";}
            else if(audio[2].nrmse()>0.1 || audio[1].nrmse()>0.1) {
                status="CPU_GPU_DIFFERENCES";
                summary=audio[2].nrmse()>0.1?"GPU decoding differs using identical CPU tokens":"GPU encoder tokens change the CPU-decoded audio";
                if(audio[2].nrmse()>0.1 && audio[1].nrmse()>0.1) summary+="; encoder path also differs";
            } else {status="NO_LARGE_AUDIO_DIFFERENCE";summary="No large waveform difference in this recording; this is not official-reference parity";}
        }
        std::ostringstream out;out<<std::setprecision(10);
        out<<"{\"status\":"<<quoted(status)<<",\"summary\":"<<quoted(summary)
            <<",\"mode\":3,\"sampleRate\":24000,\"frameSamples\":1920,\"device\":"<<quoted(device)
            <<",\"inputFrames\":"<<completeFrames<<",\"audioSamples\":"<<audio[0].count
            <<",\"audioRms\":"<<audio[0].rms()<<",\"inputRms\":"<<inputRms
            <<",\"loadMs\":"<<loadMs<<",\"diagnosticWallMs\":"<<elapsed(began)
            <<",\"comparedTokens\":"<<comparedTokens<<",\"mismatchedTokens\":"<<mismatchedTokens
            <<",\"tokenAgreement\":"<<(comparedTokens?1-double(mismatchedTokens)/comparedTokens:0)
            <<",\"firstTokenMismatchFrame\":"<<firstMismatch<<",\"mismatchesByCodebook\":[";
        for(int i=0;i<8;++i) {if(i) out<<',';out<<mismatchesByCodebook[i];}
        out<<"],\"cpuEncodeMs\":"<<encMs[0]<<",\"gpuEncodeMs\":"<<encMs[1]<<",\"routes\":[";
        for(int i=0;i<4;++i) {
            if(i) out<<',';
            out<<"{\"name\":"<<quoted(names[i])<<",\"pcmFile\":"<<quoted(files[i])
                <<",\"samples\":"<<audio[i].count<<",\"rms\":"<<audio[i].rms()<<",\"peak\":"<<audio[i].peak
                <<",\"clippedSamples\":"<<audio[i].clipped<<",\"decodeMs\":"<<decMs[i]
                <<",\"nrmseVsCpuReference\":"<<audio[i].nrmse()<<",\"levelRatioVsCpuReference\":"<<audio[i].levelRatio()
                <<",\"correlationVsCpuReference\":"<<audio[i].correlation()
                <<",\"levelCollapsed\":"<<(audio[i].collapsed()?"true":"false")<<"}";
        }
        out<<"],\"comparisonThresholds\":{\"minimumReferenceRms\":0.01,\"waveformNrmse\":0.1,\"collapseLevelRatio\":0.01}"
            <<",\"limitations\":\"Diagnostic runs both encoders and four independent decoders. Timing is not single-pipeline throughput. Token differences alone can reflect floating-point rounding. CPU is a local reference, not verified against Kyutai's official runtime.\"";
        if(!error.empty()) out<<",\"error\":"<<quoted(error);
        out<<"}";return out.str();
    };
    try {
        Backends backends;
        event("Comparison: initializing CPU reference");
        backends.cpu=ggml_backend_cpu_init();if(!backends.cpu) throw std::runtime_error("CPU unavailable");
        ggml_backend_cpu_set_n_threads(backends.cpu,4);
        event("Comparison: initializing Vulkan");
        if(ggml_backend_vk_get_device_count()==0) throw std::runtime_error("Vulkan unavailable");
        backends.gpu=ggml_backend_vk_init(0);if(!backends.gpu) throw std::runtime_error("Vulkan initialization failed");
        checkGpuIm2col(backends.gpu,outDir,event);
        char description[256]{};ggml_backend_vk_get_device_description(0,description,sizeof(description));device=description;
        event("Comparison: loading two Mimi codec instances");
        unref_ptr<moshi_context_t> cpuContext=moshi_alloc(backends.cpu,backends.cpu);
        unref_ptr<moshi_context_t> gpuContext=moshi_alloc(backends.gpu,backends.cpu);
        const std::string path=root+"/mimi-e351c8d8-125.gguf";
        unref_ptr<mimi_codec_t> cpuCodec=mimi_alloc(cpuContext,path.c_str(),8);
        unref_ptr<mimi_codec_t> gpuCodec=mimi_alloc(gpuContext,path.c_str(),8);
        if(mimi_frame_size(cpuCodec)!=1920 || mimi_frame_size(gpuCodec)!=1920) throw std::runtime_error("Unexpected codec frame format");
        unref_ptr<mimi_encode_context_t> cpuEncoder=mimi_encode_alloc_context(cpuCodec);
        unref_ptr<mimi_encode_context_t> gpuEncoder=mimi_encode_alloc_context(gpuCodec);
        unref_ptr<mimi_decode_context_t> decoders[4];
        for(int i=0;i<4;++i) decoders[i]=mimi_decode_alloc_context(i<2?cpuCodec:gpuCodec);
        std::ofstream pcm[4];
        for(int i=0;i<4;++i) {pcm[i].open(outDir+"/"+files[i],std::ios::binary);if(!pcm[i]) throw std::runtime_error("Cannot create comparison audio");}
        std::ofstream tokensFile(outDir+"/tokens.csv"),timing(outDir+"/comparison-frames.csv");
        if(!tokensFile||!timing) throw std::runtime_error("Cannot create comparison diagnostics");
        tokensFile<<"frame,codebook,cpu_token,gpu_token\n";
        timing<<"frame,cpu_encode_ms,gpu_encode_ms,cpu_cpu_decode_ms,gpu_cpu_decode_ms,cpu_gpu_decode_ms,gpu_gpu_decode_ms\n";
        loadMs=elapsed(began);
        std::vector<float> frame(1920),decoded[4];for(auto& d:decoded)d.resize(1920);
        std::vector<int16_t> cpuTokens(8),gpuTokens(8);
        const size_t frames=(input.size()+1919)/1920;
        for(size_t i=0;i<frames;++i) {
            std::fill(frame.begin(),frame.end(),0.f);
            for(size_t j=0;j<1920&&i*1920+j<input.size();++j) frame[j]=input[i*1920+j];
            // Persist the exact stage before first dispatch, which can abort inside a driver.
            auto stage=[&](const char* name) {if(i==0){save(outDir+"/partial.json",result(false));event(std::string("Comparison first frame: ")+name);}};
            stage("CPU encoding");auto t=Clock::now();mimi_encode_send(cpuEncoder,frame.data());mimi_encode_receive(cpuEncoder,cpuTokens.data());double cpuMs=elapsed(t);encMs[0]+=cpuMs;
            stage("GPU encoding");t=Clock::now();mimi_encode_send(gpuEncoder,frame.data());mimi_encode_receive(gpuEncoder,gpuTokens.data());double gpuMs=elapsed(t);encMs[1]+=gpuMs;
            timing<<i<<','<<cpuMs<<','<<gpuMs;
            for(int c=0;c<8;++c) {
                if(cpuTokens[c]<0||cpuTokens[c]>=2048||gpuTokens[c]<0||gpuTokens[c]>=2048) throw std::runtime_error("Invalid encoded token");
                ++comparedTokens;
                if(cpuTokens[c]!=gpuTokens[c]) {++mismatchedTokens;++mismatchesByCodebook[c];if(firstMismatch<0)firstMismatch=i;}
                tokensFile<<i<<','<<c<<','<<cpuTokens[c]<<','<<gpuTokens[c]<<'\n';
            }
            for(int route=0;route<4;++route) {
                stage(names[route]);t=Clock::now();
                auto& tokens=(route==0||route==2)?cpuTokens:gpuTokens;
                mimi_decode_send(decoders[route],tokens.data());mimi_decode_receive(decoders[route],decoded[route].data());
                double ms=elapsed(t);decMs[route]+=ms;timing<<','<<ms;
                for(size_t j=0;j<1920;++j) {
                    audio[route].add(decoded[route][j],decoded[0][j]);
                    int16_t sample=static_cast<int16_t>(std::clamp(decoded[route][j],-1.f,1.f)*32767.f);
                    unsigned char bytes[]={static_cast<unsigned char>(sample&255),static_cast<unsigned char>((sample>>8)&255)};
                    pcm[route].write(reinterpret_cast<char*>(bytes),2);
                }
            }
            timing<<'\n';++completeFrames;
            if(i%6==0||i+1==frames) {
                for(auto& file:pcm){file.flush();if(!file)throw std::runtime_error("Comparison audio write failed");}
                tokensFile.flush();timing.flush();if(!tokensFile||!timing)throw std::runtime_error("Comparison log write failed");
                save(outDir+"/partial.json",result(false));
                event("Compared "+std::to_string(i+1)+" / "+std::to_string(frames)+" frames across four routes");
            }
        }
        event("Comparison finished; releasing codec instances");
    } catch(const std::exception& e) {error=e.what();}
    auto report=result(true);save(outDir+"/comparison.json",report);save(outDir+"/partial.json",report);return report;
}
