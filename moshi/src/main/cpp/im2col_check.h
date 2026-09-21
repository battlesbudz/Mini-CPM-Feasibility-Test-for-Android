#pragma once
#include <ggml.h>
#include <ggml-backend.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Complete output checks against an independent scalar convolution-window oracle.
// Exactly representable dyadic inputs permit exact F16/F32 comparisons.
namespace moshi_im2col {
struct Case {
    const char* name;
    int w,h,c,n,kw,kh,sx,sy,px,py,dx,dy;
    bool twoD;
};
struct Graph {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ~Graph() { if(buffer) ggml_backend_buffer_free(buffer); if(ctx) ggml_free(ctx); }
    Graph() = default;
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
};
inline uint64_t check(ggml_backend_t backend, const std::function<void(const std::string&)>& log) {
    const Case cases[] = {
        {"encoder_1920",1926,1,1,1,7,1,1,0,0,0,1,0,false},
        {"decoder_512_channels",8,1,512,1,7,1,1,0,0,0,1,0,false},
        {"single_timestep",7,1,16,1,7,1,1,0,0,0,1,0,false},
        {"stride_padding_batch",513,1,3,2,3,1,2,0,1,0,1,0,false},
        {"dilated_1d",13,1,5,2,3,1,1,0,2,0,2,0,false},
        {"padded_2d",9,7,3,2,3,2,2,1,1,1,1,2,true},
        {"dilated_2d",11,9,2,1,3,3,1,2,2,2,2,2,true},
        {"partial_workgroup",67,1,2,1,1,1,1,0,0,0,1,0,false},
    };
    uint64_t checked=0;
    for(const auto& q:cases) for(auto type:{GGML_TYPE_F16,GGML_TYPE_F32}) {
        const std::string label=std::string(q.name)+"/"+ggml_type_name(type);
        log("START "+label);
        Graph g; g.ctx=ggml_init({2*1024*1024,nullptr,true});
        if(!g.ctx) throw std::runtime_error("IM2COL test context allocation failed");
        auto kernel=q.twoD?ggml_new_tensor_4d(g.ctx,GGML_TYPE_F16,q.kw,q.kh,q.c,1)
                          :ggml_new_tensor_3d(g.ctx,GGML_TYPE_F16,q.kw,q.c,1);
        auto input=q.twoD?ggml_new_tensor_4d(g.ctx,GGML_TYPE_F32,q.w,q.h,q.c,q.n)
                         :ggml_new_tensor_3d(g.ctx,GGML_TYPE_F32,q.w,q.c,q.n);
        auto output=ggml_im2col(g.ctx,kernel,input,q.sx,q.sy,q.px,q.py,q.dx,q.dy,q.twoD,type);
        auto graph=ggml_new_graph(g.ctx);ggml_build_forward_expand(graph,output);
        if(!ggml_backend_supports_op(backend,output)) throw std::runtime_error("IM2COL unsupported: "+label);
        g.buffer=ggml_backend_alloc_ctx_tensors(g.ctx,backend);
        if(!g.buffer) throw std::runtime_error("IM2COL test buffer allocation failed");
        ggml_backend_buffer_clear(g.buffer,0);
        const int ow=(q.w+2*q.px-q.dx*(q.kw-1)-1)/q.sx+1;
        const int oh=q.twoD?(q.h+2*q.py-q.dy*(q.kh-1)-1)/q.sy+1:1;
        const int chw=q.c*q.kh*q.kw;
        const size_t count=size_t(q.n)*oh*ow*chw;
        if(size_t(ggml_nelements(output))!=count) throw std::runtime_error("IM2COL unexpected shape");
        std::vector<float> data(size_t(q.w)*q.h*q.c*q.n), actual(count);
        std::vector<ggml_fp16_t> half(type==GGML_TYPE_F16?count:0);
        // Reuse the graph with changed input and poison every output first, checking
        // both missing writes and stale graph/buffer state on the second dispatch.
        for(int pass=0;pass<2;++pass) {
            for(size_t i=0;i<data.size();++i) data[i]=float(int((i*37+pass*113)%509)-254)/256.f;
            ggml_backend_tensor_set(input,data.data(),0,data.size()*sizeof(float));
            if(type==GGML_TYPE_F16) {
                std::fill(half.begin(),half.end(),ggml_fp32_to_fp16(17.f));
                ggml_backend_tensor_set(output,half.data(),0,half.size()*sizeof(ggml_fp16_t));
            } else {
                std::fill(actual.begin(),actual.end(),17.f);
                ggml_backend_tensor_set(output,actual.data(),0,actual.size()*sizeof(float));
            }
            if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)
                throw std::runtime_error("IM2COL dispatch failed: "+label);
            ggml_backend_synchronize(backend);
            if(type==GGML_TYPE_F16) {
                ggml_backend_tensor_get(output,half.data(),0,half.size()*sizeof(ggml_fp16_t));
                for(size_t i=0;i<count;++i)actual[i]=ggml_fp16_to_fp32(half[i]);
            } else ggml_backend_tensor_get(output,actual.data(),0,actual.size()*sizeof(float));
            for(int n=0;n<q.n;++n) for(int y=0;y<oh;++y) for(int x=0;x<ow;++x)
            for(int c=0;c<q.c;++c) for(int ky=0;ky<q.kh;++ky) for(int kx=0;kx<q.kw;++kx) {
                int ix=x*q.sx+kx*q.dx-q.px, iy=y*q.sy+ky*q.dy-q.py;
                float expected=0;
                if(ix>=0&&ix<q.w&&iy>=0&&iy<q.h) expected=data[((n*q.c+c)*q.h+iy)*q.w+ix];
                size_t index=(((n*oh+y)*ow+x)*q.c+c)*q.kh*q.kw+ky*q.kw+kx;
                if(actual[index]!=expected) {
                    std::ostringstream error;
                    error<<"IM2COL FAILED "<<label<<" pass="<<pass<<" index="<<index
                         <<" expected="<<expected<<" actual="<<actual[index];
                    throw std::runtime_error(error.str());
                }
                ++checked;
            }
        }
        log("PASS "+label+" elements="+std::to_string(count*2));
    }
    log("PASS all 16 cases / 32 dispatches, elements="+std::to_string(checked));
    return checked;
}
}
