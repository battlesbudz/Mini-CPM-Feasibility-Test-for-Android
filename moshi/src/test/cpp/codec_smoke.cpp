// Exercise actual codec state, encoding and decoding with the pinned Mimi weights.
#include <moshi/moshi.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
int main(int argc,char** argv) {
    if(argc!=2) return 2;
    auto backend=ggml_backend_cpu_init();
    if(!backend) return 3;
    ggml_backend_cpu_set_n_threads(backend,4);
    try {
        unref_ptr<moshi_context_t> context=moshi_alloc(backend,backend);
        unref_ptr<mimi_codec_t> codec=mimi_alloc(context,argv[1],8);
        if(mimi_frame_size(codec)!=1920) throw std::runtime_error("frame size");
        unref_ptr<mimi_encode_context_t> encoder=mimi_encode_alloc_context(codec);
        unref_ptr<mimi_decode_context_t> decoder=mimi_decode_alloc_context(codec);
        unref_ptr<mimi_encode_context_t> secondEncoder=mimi_encode_alloc_context(codec);
        unref_ptr<mimi_decode_context_t> secondDecoder=mimi_decode_alloc_context(codec);
        std::vector<float> input(1920),output(1920);
        std::vector<float> secondOutput(1920);
        std::vector<int16_t> tokens(8);
        std::vector<int16_t> secondTokens(8);
        double energy=0;
        for(int frame=0;frame<20;++frame) {
            for(int i=0;i<1920;++i) input[i]=frame<10?0.1f*std::sin(2*3.141592653589793*440*(frame*1920+i)/24000):0.f;
            mimi_encode_send(encoder,input.data());mimi_encode_receive(encoder,tokens.data());
            mimi_encode_send(secondEncoder,input.data());mimi_encode_receive(secondEncoder,secondTokens.data());
            if(tokens!=secondTokens) throw std::runtime_error("independent encoder histories differ");
            for(auto t:tokens) if(t<0||t>=2048) throw std::runtime_error("invalid token");
            mimi_decode_send(decoder,tokens.data());mimi_decode_receive(decoder,output.data());
            mimi_decode_send(secondDecoder,secondTokens.data());mimi_decode_receive(secondDecoder,secondOutput.data());
            if(output!=secondOutput) throw std::runtime_error("independent decoder histories differ");
            for(float x:output) {if(!std::isfinite(x)) throw std::runtime_error("nonfinite audio");energy+=x*x;}
        }
        if(energy<=0) throw std::runtime_error("silent output for tone input");
        mimi_encode_reset(encoder);mimi_decode_reset(decoder);
        mimi_encode_send(encoder,input.data());mimi_encode_receive(encoder,tokens.data());
        mimi_decode_send(decoder,tokens.data());mimi_decode_receive(decoder,output.data());
        for(float x:output) if(!std::isfinite(x)) throw std::runtime_error("nonfinite reset output");
        printf("PASS: 20 codec frames, exact independent-state agreement, valid tokens, finite nonzero PCM, reset/reuse; energy=%f\n",energy);
    } catch(const std::exception& error) {fprintf(stderr,"FAIL: %s\n",error.what());ggml_backend_free(backend);return 1;}
    ggml_backend_free(backend);return 0;
}
