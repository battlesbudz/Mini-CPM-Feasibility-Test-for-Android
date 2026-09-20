// Included inside benchmark.cpp's anonymous namespace.
std::string traceCodec(const std::string& root,const std::vector<float>& input,const std::string& outDir,
    const std::function<void(const std::string&)>& event){
    std::string error,traceSummary="null";AudioMetrics normalGpuAudio,tracedGpuAudio,traceVsNormal;
    std::vector<int16_t> cpuTokens(8),normalTokens(8),traceTokens(8);
    std::vector<float> cpuPcm(1920),normalPcm(1920),tracePcm(1920),frame(1920,0.f);
    std::copy_n(input.begin(),std::min(input.size(),frame.size()),frame.begin());
    int normalMatches=0,tracedMatches=0;auto start=Clock::now();
    try {
        Backends backends;backends.cpu=ggml_backend_cpu_init();if(!backends.cpu)throw std::runtime_error("CPU unavailable");
        ggml_backend_cpu_set_n_threads(backends.cpu,4);
        if(ggml_backend_vk_get_device_count()==0)throw std::runtime_error("Vulkan unavailable");
        backends.gpu=ggml_backend_vk_init(0);if(!backends.gpu)throw std::runtime_error("Vulkan unavailable");
        auto pass=[&](ggml_backend_t backend,std::vector<int16_t>& tokens,std::vector<float>& pcm,TensorTrace* trace){
            unref_ptr<moshi_context_t> context=moshi_alloc(backend,backends.cpu);
            if(trace)trace->stage("weights");
            unref_ptr<mimi_codec_t> codec=mimi_alloc(context,(root+"/mimi-e351c8d8-125.gguf").c_str(),8);
            if(trace)trace->stage("encoder_initialization");
            unref_ptr<mimi_encode_context_t> encoder=mimi_encode_alloc_context(codec);
            if(trace)trace->stage("decoder_initialization");
            unref_ptr<mimi_decode_context_t> decoder=mimi_decode_alloc_context(codec);
            if(trace)trace->stage("encoder_frame_0");
            mimi_encode_send(encoder,frame.data());mimi_encode_receive(encoder,tokens.data());
            for(auto token:tokens)if(token<0||token>=2048)throw std::runtime_error("Invalid traced codec token");
            if(trace)trace->stage("decoder_frame_0_cpu_tokens");
            // Same known CPU tokens for both GPU decoders, independent of GPU encoding.
            mimi_decode_send(decoder,cpuTokens.data());mimi_decode_receive(decoder,pcm.data());
        };
        event("Tensor test: normal CPU reference");pass(backends.cpu,cpuTokens,cpuPcm,nullptr);
        event("Tensor test: normal Vulkan baseline");pass(backends.gpu,normalTokens,normalPcm,nullptr);
        event("Tensor test: verified uploads and per-operation Vulkan/CPU comparison");
        TensorTrace trace(outDir,backends.gpu,backends.cpu,event);
        try{pass(backends.gpu,traceTokens,tracePcm,&trace);}catch(...){traceSummary=trace.summary();throw;}
        traceSummary=trace.summary();
        for(size_t i=0;i<1920;++i){normalGpuAudio.add(normalPcm[i],cpuPcm[i]);tracedGpuAudio.add(tracePcm[i],cpuPcm[i]);traceVsNormal.add(tracePcm[i],normalPcm[i]);}
        for(int i=0;i<8;++i){normalMatches+=normalTokens[i]==cpuTokens[i];tracedMatches+=traceTokens[i]==cpuTokens[i];}
        std::ofstream tokens(outDir+"/trace-tokens.csv");tokens<<"codebook,cpu,normal_gpu,traced_gpu\n";
        for(int i=0;i<8;++i)tokens<<i<<','<<cpuTokens[i]<<','<<normalTokens[i]<<','<<traceTokens[i]<<'\n';
        if(!tokens)throw std::runtime_error("Cannot write trace tokens");
    }catch(const std::exception& e){error=e.what();}
    std::ostringstream s;s<<std::setprecision(10)<<"{\"mode\":4,\"status\":"<<quoted(error.empty()?"TENSOR_TRACE_COMPLETE":"ERROR")
        <<",\"summary\":"<<quoted(error.empty()?"Tensor trace saved; inspect first divergence and normal-versus-traced output":error)
        <<",\"diagnosticWallMs\":"<<elapsed(start)<<",\"inputFrames\":1,\"normalGpuMatchingTokens\":"<<normalMatches
        <<",\"tracedGpuMatchingTokens\":"<<tracedMatches<<",\"normalGpuDecoderRms\":"<<normalGpuAudio.rms()
        <<",\"tracedGpuDecoderRms\":"<<tracedGpuAudio.rms()<<",\"cpuDecoderRms\":"<<normalGpuAudio.referenceRms()
        <<",\"tracedVsNormalDecoderNrmse\":"<<traceVsNormal.nrmse()<<",\"trace\":"<<traceSummary
        <<",\"limitations\":\"First frame only. Tensor samples locate candidate operations, not proven root causes. Per-node tracing changes fusion and synchronization. CPU graph clones begin with tested-backend state. Inspect upload verification and initialization traces. This test is not a throughput benchmark.\"";
    if(!error.empty())s<<",\"error\":"<<quoted(error);
    s<<'}';
    auto result=s.str();save(outDir+"/tensor-summary.json",result);save(outDir+"/partial.json",result);return result;
}
